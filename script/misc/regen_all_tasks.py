"""Regenerate dataset-info/all_tasks__ds_v<version>/ for one planner version.

Which configs a planner version can serve is decided by the dataset loader's own
_ANNOTATION_INDEX (what is published per dataset/plan-kind) and _PAUSED_ANNOTATIONS
(what is withheld), so both are read straight from MedVision.py rather than from a
hand-maintained CSV -- the shipped dataset-configs CSVs predate the MAMA-MIA/PI-CAI
withdrawal and still list 36 configs no 1.2.0 pin can load.

Counting a config means streaming it, so the run is keyed by (config, the annotation
version it RESOLVES to) and cached: detection plans are identical in every release,
which is what lets the 1.1.1 lists seed the cache instead of re-streaming
AbdomenAtlas1.0Mini and TotalSegmentator once per version.
"""

import argparse
import ast
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIND = {
    "MaskSize": "segmentation",
    "BoxSize": "detection",
    "TumorLesionSize": "biometry",
    "BiometricsFromLandmarks": "biometry",
}
# (family, planes, split) -> output file stem
OUTPUTS = [
    ("BoxSize", ["Axial"], "detect-CoT__Axial"),
    ("BoxSize", ["Coronal"], "detect-CoT__Coronal"),
    ("BoxSize", ["Sagittal"], "detect-CoT__Sagittal"),
    ("TumorLesionSize", ["Axial"], "TL-CoT__Axial"),
    ("TumorLesionSize", ["Coronal"], "TL-CoT__Coronal"),
    ("TumorLesionSize", ["Sagittal"], "TL-CoT__Sagittal"),
    ("BiometricsFromLandmarks", ["Axial", "Coronal", "Sagittal"], "AD-CoT__AllSlices"),
]


def _version_tuple(v):
    return tuple(int(x) for x in v.split("."))


def load_loader_tables(medvision_py):
    src = open(medvision_py, encoding="utf-8").read()

    def literal(name):
        i = src.index(f"{name} = {{")
        j = src.index("\n}", i) + 2
        return ast.literal_eval(src[i + len(f"{name} = "):j])

    names = re.findall(r"MedVisionConfig\(\s*name\s*=\s*[\"']([^\"']+)[\"']", src)
    return literal("_ANNOTATION_INDEX"), literal("_PAUSED_ANNOTATIONS"), names


def resolve(config, ceiling, ann, paused):
    """The annotation version `config` loads at `ceiling`, or None if unloadable."""
    dataset, family = config.split("_")[0], config.split("_")[1]
    kind = KIND[family]
    withheld = set(paused.get(dataset, {}).get(kind, ()))
    usable = [
        v for v in ann.get(dataset, {}).get(kind, ())
        if v not in withheld and _version_tuple(v) <= _version_tuple(ceiling)
    ]
    return max(usable, key=_version_tuple) if usable else None


def seed_cache_from_v111(cache, ann, paused):
    """Counts already published for 1.1.1, re-keyed by the annotation they resolve to."""
    src_dir = os.path.join(REPO, "dataset-info", "all_tasks__ds_v1.1.1")
    if not os.path.isdir(src_dir):
        return 0
    seeded = 0
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".json"):
            continue
        split = "Test" if fn.endswith("__Test.json") else "Train"
        for task, count in json.load(open(os.path.join(src_dir, fn))).items():
            base = task[:-4] if task.endswith("-CoT") else task
            config = base.replace("BoxCoordinate", "BoxSize") + f"_{split}"
            version = resolve(config, "1.1.1", ann, paused)
            if version and f"{config}|{version}" not in cache:
                cache[f"{config}|{version}"] = count
                seeded += 1
    return seeded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True, help="Planner version, e.g. 1.4.0")
    ap.add_argument("--data_dir", default=os.path.join(REPO, "Data"))
    ap.add_argument("--medvision_py", default="/mnt/vincent-pvc-rwm/MedVision/MedVision.py")
    ap.add_argument("--dataset_path", default="YongchengYAO/MedVision",
                    help="Hub id, or a local checkout of the dataset repo when counting "
                         "configs that are not pushed yet.")
    ap.add_argument("--cache", default=os.path.join(REPO, "dataset-info", ".all_tasks_counts_cache.json"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--no-count", action="store_true", help="Dry run: report the work, count nothing.")
    args = ap.parse_args()

    os.environ["MedVision_PLANNER_VERSION"] = args.version
    os.environ.setdefault("MedVision_FORCE_INSTALL_CODE", "false")

    ann, paused, names = load_loader_tables(args.medvision_py)

    # Pinning below the newest annotation published for a dataset is gated on an
    # explicit acknowledgement. Regenerating an older version's lists is exactly that
    # case, so derive the value from the index rather than hard-coding a release that
    # goes stale the next time one ships.
    newest = max(
        (v for kinds in ann.values() for versions in kinds.values() for v in versions),
        key=_version_tuple,
    )
    os.environ["MedVision_ACK_RELEASE"] = newest
    print(f"[env] MedVision_PLANNER_VERSION={args.version}  MedVision_ACK_RELEASE={newest}")

    out_dir = args.out_dir or os.path.join(REPO, "dataset-info", f"all_tasks__ds_v{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    seeded = seed_cache_from_v111(cache, ann, paused)
    print(f"[cache] {len(cache)} entries ({seeded} seeded from the 1.1.1 lists)")

    # Imported late: they pull in `datasets`, which a --no-count dry run does not need.
    if not args.no_count:
        from medvision_bm.utils import setup_env_hf_medvision_ds
        from medvision_bm.utils.configs_to_tasks import count_samples

        # Exports MedVision_DATA_DIR and the HF cache variables the loader reads at
        # import time; without it the builder raises before any config is streamed.
        setup_env_hf_medvision_ds(args.data_dir, force_install_code=False)

    todo = 0
    for family, planes, stem in OUTPUTS:
        for split in ("Test", "Train"):
            selected = []
            for config in names:
                parts = config.split("_")
                if parts[1] != family or parts[-2] not in planes or parts[-1] != split:
                    continue
                version = resolve(config, args.version, ann, paused)
                if version:
                    selected.append((config, version))

            tasks, out = {}, os.path.join(out_dir, f"tasks_MedVision-{stem}__{split}.json")
            for i, (config, version) in enumerate(selected, 1):
                base = config[: -len(f"_{split}")].replace("BoxSize", "BoxCoordinate")
                task = base + ("-CoT" if split == "Test" else "")
                key = f"{config}|{version}"
                if key in cache:
                    tasks[task] = cache[key]
                elif args.no_count:
                    todo += 1
                    tasks[task] = None
                else:
                    tasks[task] = cache[key] = count_samples(
                        config, split.lower(), True, dataset_path=args.dataset_path
                    )
                    json.dump(cache, open(args.cache, "w"), indent=1)
                    print(f"  [{i}/{len(selected)}] {task}: {tasks[task]}  (counted)")
                    continue
                print(f"  [{i}/{len(selected)}] {task}: {tasks[task]}")
            if not args.no_count:
                # A config can resolve to an annotation that yields nothing on this
                # plane/split -- e.g. the pre-1.4.0 pixel-count floor rejected every
                # off-axial cluster on some tasks. Such a subtask is not runnable, and
                # every published list holds positive counts only, so drop it here. The
                # count stays cached, so this costs no re-streaming.
                tasks = {k: n for k, n in tasks.items() if n}
                json.dump(tasks, open(out, "w"), indent=4)
            print(f"-> {len(tasks)} tasks {'(dry run)' if args.no_count else out}")

    if args.no_count:
        print(f"\nDRY RUN: {todo} configs would need streaming for v{args.version}")


if __name__ == "__main__":
    main()
