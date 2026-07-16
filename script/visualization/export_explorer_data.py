#!/usr/bin/env python3
"""Export the MedVision Dataset Explorer data blob for the project webpage.

Reads ``ConfigurationsList_Test.csv``, resolves every **non-MaskSize** test config to
``{dataset, task_type, task_id, plane, modality, subtype, anatomy_groups}``, and writes
``<page_dir>/static/js/explorer-data.js`` defining the ``window.MEDVISION_EXPLORER`` global that
``static/js/explorer.js`` consumes. Mirrors ``export_webpage_cases.py`` (same ``--page_dir`` convention).

Two different resolutions run per config, and they deliberately read different modules:

ANATOMY (drives the filter) — from the SEGMENTATION labels_map:
  - ``BoxSize`` (detection)        -> live ``preprocess_segmentation.benchmark_plan`` ``tasks[id-1]``;
                                      anatomy = regroup of EVERY label in ``labels_map``.
  - ``TumorLesionSize``            -> ``benchmark_plan_biometry_v*.json.gz`` ``tasks[id-1]``;
                                      anatomy = regroup of the SINGLE ``labels_map[str(target_label)]``.
  - ``BiometricsFromLandmarks``    -> dataset-level anatomy override
                                      (``Ceph-Biometrics-400`` -> Head-Neck, ``FeTA24`` -> Brain).
  - ``MaskSize``                   -> SKIPPED (segmentation mask area; excluded from the explorer).

TASK DETAIL (the panel shown on click) — from the module the LOADER reads (see TASK_MODULE).

Why they differ for ``BoxSize``: the detection and segmentation ``labels_map`` are NOT identical for
every dataset. KiPA22 label 4 is "tumor" in detection but "kidney tumor" in segmentation, and
CrossMoDA's detection name ("vestibular schwannoma (acoustic neuroma)") is absent from
``label_map_regroup`` entirely — that map was written against the segmentation names. So anatomy is
resolved from segmentation (better curated, and the grouping is the same structure either way) while
the panel shows the detection labels the config genuinely loads.

The script fails loudly (exit 1) on any unresolved config, unmapped label, unknown modality, or missing
benchmark plan, so the shipped data can never contain a silent hole.

Example
-------
    python -m medvision_bm.utils... ; or directly:
    python script/visualization/export_explorer_data.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io
"""
import argparse
import glob
import gzip
import importlib
import json
import os
import re
import sys

from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE, label_map_regroup
from medvision_bm.utils.parse_utils import (
    get_labelsMap_imgModality_from_seg_benchmark_plan,
)

# ── static config ───────────────────────────────────────────────────────────────────────────────
REPO = "/mnt/vincent-pvc-rwm/Github/MedVision"
DEFAULT_CONFIGS_CSV = os.path.join(REPO, "dataset-info/dataset-configs/ConfigurationsList_Test.csv")
DEFAULT_DATA_DIR = os.path.join(REPO, "Data")
DEFAULT_DATASET_INFO = os.path.join(REPO, "dataset-info/datasets_info.json")

# medvision_ds is pip-installed as a NON-editable copy in site-packages that silently shadows the
# working tree; task details must come from the sources under version control.
DEFAULT_DS_SRC = "/mnt/vincent-pvc-rwm/MedVision/src"

# Which preprocess module the LOADER actually reads for each config task type.
# Authority: MedVision.py::_split_generators — Box-Size -> MedVision_BenchmarkPlannerDetection,
# Tumor-Lesion-Size -> ...Biometry_fromSeg, Biometrics-From-Landmarks -> ...Biometry (all three
# biometry variants read benchmark_plan_biometry_v*). Displaying any other module's task would
# show folders/labels the config does not actually load — for KiPA22 and CrossMoDA the detection
# and segmentation labels_map genuinely disagree.
TASK_MODULE = {
    "BoxSize": "detection",
    "TumorLesionSize": "biometry",
    "BiometricsFromLandmarks": "biometry",
}

VERSIONS = ["1.1.1", "1.1.0", "1.0.0"]
LATEST_VERSION = "1.1.1"

# Body part -> anatomy groups, codified from the comment banners in
# medvision_bm/utils/configs.py::label_map_regroup. Order within each list is curated
# (organs before their tumor/lesion groups). Verified to cover all 36 groups with no orphans.
BODY_PART_GROUPS = {
    "Brain": ["Brain", "Brain Tumor/Lesion"],
    "Head & Neck": ["Head-Neck", "Head-Neck Tumor/Lesion"],
    "Heart": ["Heart"],
    "Thorax (Lungs)": ["Lung", "Lung Tumor/Lesion"],
    "Abdomen": [
        "Liver", "Liver Tumor/Lesion", "Kidney", "Kidney Tumor/Lesion",
        "Pancreas", "Pancreas Tumor/Lesion", "Gallbladder", "Spleen",
        "Adrenal Gland", "Colon", "Colon Tumor/Lesion", "Intestine",
        "Esophagus", "Stomach",
    ],
    "Pelvis (Uro-Gynae)": ["Urinary System", "Uterus", "Prostate"],
    "Vasculature": ["Artery", "Vein"],
    "Musculoskeletal": ["Hip", "Rib", "Spine", "Knee Bone", "Knee Soft Tissue"],
    "Dentistry": ["Jawbone", "Tooth"],
    "Lymphatics": ["Metastatic Lymph Node"],
    "Other Pathology": ["Miscellaneous Tumor/Lesion"],
    "Other": ["Others"],
}
GROUP2BODY = {g: bp for bp, gs in BODY_PART_GROUPS.items() for g in gs}

# Landmark-biometry datasets have no per-label mask; anatomy is assigned at the dataset level.
BIOMETRY_ANATOMY = {"Ceph-Biometrics-400": "Head-Neck", "FeTA24": "Brain"}

# Raw benchmark-plan modality string -> the five explorer modality buckets.
MODALITY_NORM = {
    "CT": "CT", "MRI": "MRI", "PET": "PET",
    "ultrasound": "Ultrasound", "X Ray": "X-Ray", "X-ray": "X-Ray", "X-Ray": "X-Ray",
}

TASK_TYPES = ("BoxSize", "TumorLesionSize", "BiometricsFromLandmarks")  # MaskSize deliberately excluded


# ── resolution helpers ──────────────────────────────────────────────────────────────────────────
_bio_cache: dict[str, dict | None] = {}


def _load_biometry_plan(data_dir: str, dataset: str):
    """Return the highest-version biometry benchmark plan dict for ``dataset`` (or None)."""
    if dataset in _bio_cache:
        return _bio_cache[dataset]
    pattern = os.path.join(data_dir, "Datasets", dataset, "benchmark_plan_biometry_v*.json.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        _bio_cache[dataset] = None
        return None
    with gzip.open(files[-1], "rt") as fh:
        _bio_cache[dataset] = json.load(fh)
    return _bio_cache[dataset]


def _seg_modality(dataset: str):
    """Segmentation-plan modality for ``dataset`` (used to fill an empty biometry modality)."""
    try:
        _, modality = get_labelsMap_imgModality_from_seg_benchmark_plan(dataset, 1)
        return modality
    except Exception:
        return None


def _biometry_subtype(dataset: str, task_id):
    """Derive Distance/Angle from the biometry plan's ``biometrics_map`` metric types when the config
    name carries no subtype token (e.g. FeTA24, whose metrics are all distances). Returns None if mixed
    or unknown."""
    plan = _load_biometry_plan(DEFAULT_DATA_DIR, dataset)
    if not plan:
        return None
    tasks = plan.get("tasks", [])
    if not (task_id and 1 <= task_id <= len(tasks)):
        return None
    types = {m.get("metric_type") for m in (tasks[task_id - 1].get("biometrics_map") or [])}
    if types == {"distance"}:
        return "Distance"
    if types == {"angle"}:
        return "Angle"
    return None


_mod_plan_cache: dict[tuple, dict | None] = {}


def _module_plan(dataset: str, kind: str):
    """``benchmark_plan`` dict from the live ``preprocess_<kind>`` module (metadata only, no case
    lists — never touch the 676 MB detection plan JSON)."""
    key = (dataset, kind)
    if key not in _mod_plan_cache:
        pkg = DATASETS_NAME2PACKAGE[dataset]
        module = importlib.import_module(f"medvision_ds.datasets.{pkg}.preprocess_{kind}")
        _mod_plan_cache[key] = getattr(module, "benchmark_plan", None)
    return _mod_plan_cache[key]


def _seg_description(dataset: str):
    """Segmentation-plan image_description (fills an empty biometry description, e.g. FeTA24)."""
    try:
        plan = _module_plan(dataset, "segmentation")
        return (plan or {}).get("tasks", [{}])[0].get("image_description") or ""
    except Exception:
        return ""


def _task_detail(dataset: str, task_type: str, task_id: int):
    """Task panel payload for one config, read from the module the loader actually uses.

    Folders are prefixed with the on-disk layout ``<dataset-folder>/<dataset-name>/`` (verified:
    Data/Datasets/<dataset-name>/...), leaving ``<dataset-folder>`` as a placeholder for the
    caller's MedVision_DATA_DIR/Datasets.
    """
    kind = TASK_MODULE[task_type]
    plan = _module_plan(dataset, kind)
    tasks = (plan or {}).get("tasks", [])
    if not (task_id and 1 <= task_id <= len(tasks)):
        raise ValueError(f"{dataset} {task_type} Task{task_id}: index out of range "
                         f"({len(tasks)} task(s) in preprocess_{kind})")
    t = tasks[task_id - 1]
    prefix = f"<dataset-folder>/{dataset}/"

    detail = {"kind": kind}
    desc = t.get("image_description") or ""
    if not desc:  # FeTA24's biometry task leaves it empty — same images as its segmentation task
        desc = _seg_description(dataset)
    if desc:
        detail["image_description"] = desc
    if t.get("image_folder"):
        detail["image_folder"] = prefix + t["image_folder"]
    if t.get("mask_folder"):
        detail["mask_folder"] = prefix + t["mask_folder"]
    if t.get("landmark_folder"):
        detail["landmark_folder"] = prefix + t["landmark_folder"]
        # The T/L planner stamps "-v<version>" onto landmark_folder at plan time
        # (benchmark_planner.py:2197, MedVision_BenchmarkPlannerBiometry_fromSeg) — but only from
        # v1.1.0 on; v1.0.0 predates that code and uses the bare folder. The A/D planner never
        # stamps. Verified against the on-disk plans of 4 T/L datasets x 3 versions.
        detail["landmark_folder_versioned"] = task_type == "TumorLesionSize"
    if t.get("labels_map"):
        detail["labels_map"] = t["labels_map"]
    if t.get("target_label") is not None:
        detail["target_label"] = str(t["target_label"])  # T/L measures only this label
    if task_type == "BiometricsFromLandmarks" and t.get("landmarks_map"):
        detail["landmarks_map"] = t["landmarks_map"]
    return detail


def _parse_config(name: str):
    """(dataset, task_type, subtype, task_id, plane) from a ``*_Test`` config name."""
    base = name[:-5] if name.endswith("_Test") else name
    parts = base.split("_")
    dataset, task_type, plane = parts[0], parts[1], parts[-1]
    task_id = next(
        (int(re.fullmatch(r"Task(\d+)", p).group(1)) for p in parts if re.fullmatch(r"Task(\d+)", p)),
        None,
    )
    subtype = parts[2] if (task_type == "BiometricsFromLandmarks" and parts[2] in ("Distance", "Angle")) else None
    return dataset, task_type, subtype, task_id, plane


def _resolve(name: str):
    """Resolve one config -> (modality_raw, anatomy_groups set). Raises ValueError on any hole."""
    dataset, task_type, _subtype, task_id, _plane = _parse_config(name)
    groups: set[str] = set()

    if task_type == "BoxSize":
        labels_map, modality_raw = get_labelsMap_imgModality_from_seg_benchmark_plan(dataset, task_id)
        for label in (labels_map or {}).values():
            g = label_map_regroup.get(label)
            if g is None:
                raise ValueError(f"{name}: label {label!r} not in label_map_regroup")
            groups.add(g)

    elif task_type == "TumorLesionSize":
        plan = _load_biometry_plan(DEFAULT_DATA_DIR, dataset)
        if plan is None:
            raise ValueError(f"{name}: no biometry benchmark plan on disk")
        tasks = plan.get("tasks", [])
        if not (task_id and 1 <= task_id <= len(tasks)):
            raise ValueError(f"{name}: task index {task_id} out of range ({len(tasks)} tasks)")
        task = tasks[task_id - 1]
        modality_raw = task.get("image_modality") or ""
        label = (task.get("labels_map") or {}).get(str(task.get("target_label")))
        if label is None:
            raise ValueError(f"{name}: target_label {task.get('target_label')} not in labels_map")
        g = label_map_regroup.get(label)
        if g is None:
            raise ValueError(f"{name}: label {label!r} not in label_map_regroup")
        groups.add(g)

    elif task_type == "BiometricsFromLandmarks":
        g = BIOMETRY_ANATOMY.get(dataset)
        if g is None:
            raise ValueError(f"{name}: no biometry anatomy override for {dataset}")
        groups.add(g)
        plan = _load_biometry_plan(DEFAULT_DATA_DIR, dataset)
        modality_raw = ""
        if plan and task_id and 1 <= task_id <= len(plan.get("tasks", [])):
            modality_raw = plan["tasks"][task_id - 1].get("image_modality") or ""
        if not modality_raw:  # e.g. FeTA24 biometry leaves image_modality empty
            modality_raw = _seg_modality(dataset) or ""
    else:
        raise ValueError(f"{name}: unexpected task_type {task_type!r}")

    return modality_raw, groups


# ── main ────────────────────────────────────────────────────────────────────────────────────────
def build_rows(configs_csv: str):
    with open(configs_csv) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]

    rows, problems, tasks = [], [], {}
    for name in names:
        dataset, task_type, subtype, task_id, plane = _parse_config(name)
        if task_type == "MaskSize":
            continue
        if task_type not in TASK_TYPES:
            problems.append(f"{name}: unknown task_type {task_type!r}")
            continue
        try:
            modality_raw, groups = _resolve(name)
        except Exception as exc:  # noqa: BLE001 - surface every failure
            problems.append(str(exc))
            continue
        modality = MODALITY_NORM.get(modality_raw)
        if modality is None:
            problems.append(f"{name}: unknown modality {modality_raw!r}")
            continue
        bad = [g for g in groups if g not in GROUP2BODY]
        if bad:
            problems.append(f"{name}: anatomy groups not in body-part map: {bad}")
            continue
        # Fail-loud on a silent hole: label-derived task types must yield >=1 anatomy group.
        if task_type in ("BoxSize", "TumorLesionSize") and not groups:
            problems.append(f"{name}: resolved to zero anatomy groups (empty/missing labels_map)")
            continue
        if task_type == "BiometricsFromLandmarks" and subtype is None:
            subtype = _biometry_subtype(dataset, task_id)  # e.g. FeTA24 -> "Distance"
        # Task detail is per (dataset, module, task_id) — the 3 planes of a config share it.
        task_key = f"{dataset}|{TASK_MODULE[task_type]}|{task_id}"
        if task_key not in tasks:
            try:
                tasks[task_key] = _task_detail(dataset, task_type, task_id)
            except Exception as exc:  # noqa: BLE001 - surface every failure
                problems.append(str(exc))
                continue
        rows.append({
            "config": name,
            "dataset": dataset,
            "task_type": task_type,
            "subtype": subtype,
            "task_id": task_id,
            "plane": plane,
            "modality": modality,
            "anatomy_groups": sorted(groups),
            "task_key": task_key,
        })
    return rows, tasks, problems


def build_body_parts(rows):
    """Body part -> anatomy groups, restricted to groups that appear in the shipped configs,
    preserving the curated order in BODY_PART_GROUPS."""
    present = set()
    for r in rows:
        present.update(r["anatomy_groups"])
    out = {}
    for bp, gs in BODY_PART_GROUPS.items():
        kept = [g for g in gs if g in present]
        if kept:
            out[bp] = kept
    return out


def load_dataset_info(path):
    """Per-dataset provenance compiled by script/misc/compile_dataset_info.py."""
    with open(path) as fh:
        blob = json.load(fh)
    out = {}
    for ds, info in blob.get("datasets", {}).items():
        out[ds] = {k: v for k, v in info.items() if not k.startswith("_") and v}
    return out


def emit_js(path, rows, body_parts, tasks, dataset_info):
    blob = {
        "latest_version": LATEST_VERSION,
        "versions": VERSIONS,
        "body_parts": body_parts,
        "dataset_info": dataset_info,
        "tasks": tasks,
        "configs": rows,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_explorer_data.py — DO NOT EDIT.\n"
        "// Schema: window.MEDVISION_EXPLORER = { latest_version, versions,\n"
        "//   body_parts:{ <body part>: [<anatomy group>,...] },\n"
        "//   dataset_info:{ <dataset>: {dataset_website,dataset_data[],license[],paper[]} },\n"
        "//   tasks:{ '<dataset>|<module>|<task_id>': {kind,image_description,image_folder,\n"
        "//     mask_folder?,landmark_folder?,landmark_folder_versioned?,labels_map?,target_label?,landmarks_map?} },\n"
        "//   configs:[ {config,dataset,task_type,subtype,task_id,plane,modality,anatomy_groups[],task_key} ] }\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_EXPLORER = ")
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")


def _spot_check(rows, group, modality):
    return sorted({r["dataset"] for r in rows if group in r["anatomy_groups"] and r["modality"] == modality})


def main():
    ap = argparse.ArgumentParser(description="Export MedVision Dataset Explorer data (explorer-data.js).")
    ap.add_argument("--page_dir", required=True, help="Project page repo (medvision-vlm.github.io).")
    ap.add_argument("--configs_csv", default=DEFAULT_CONFIGS_CSV, help="ConfigurationsList_Test.csv path.")
    ap.add_argument("--dataset_info", default=DEFAULT_DATASET_INFO,
                    help="datasets_info.json from script/misc/compile_dataset_info.py.")
    ap.add_argument("--out", default=None, help="Output JS path (default <page_dir>/static/js/explorer-data.js).")
    ap.add_argument("--medvision_ds_src", default=DEFAULT_DS_SRC,
                    help="medvision_ds source checkout to read (shadows the installed copy).")
    args = ap.parse_args()

    if args.medvision_ds_src:
        sys.path.insert(0, args.medvision_ds_src)
    import medvision_ds
    print(f"[explorer] medvision_ds v{medvision_ds.__version__} from {os.path.dirname(medvision_ds.__file__)}")
    if args.medvision_ds_src and not medvision_ds.__file__.startswith(args.medvision_ds_src):
        sys.exit(f"[explorer] medvision_ds resolved to {medvision_ds.__file__}, not {args.medvision_ds_src} "
                 "— refusing to export from a possibly stale installed copy.")

    rows, tasks, problems = build_rows(args.configs_csv)

    print(f"[explorer] resolved {len(rows)} non-MaskSize test configs | problems: {len(problems)}")
    if problems:
        for p in problems[:50]:
            print("   PROBLEM", p)
        sys.exit(f"[explorer] {len(problems)} unresolved config(s); refusing to write incomplete data.")

    body_parts = build_body_parts(rows)
    dataset_info = load_dataset_info(args.dataset_info)
    missing_info = sorted({r["dataset"] for r in rows} - set(dataset_info))
    if missing_info:
        sys.exit(f"[explorer] no dataset_info for: {missing_info} — re-run script/misc/compile_dataset_info.py")

    from collections import Counter
    by_type = Counter(r["task_type"] for r in rows)
    by_mod = Counter(r["modality"] for r in rows)
    print(f"[explorer] by task_type: {dict(by_type)}")
    print(f"[explorer] by modality : {dict(by_mod)}")
    print(f"[explorer] body parts  : {list(body_parts)}")
    print(f"[explorer] task details: {len(tasks)} unique (dataset|module|task_id)")
    print(f"[explorer] dataset_info: {len(dataset_info)} datasets")
    # spot checks (must match the validated expectations)
    print(f"[explorer] spot Liver@MRI            -> {_spot_check(rows, 'Liver', 'MRI')}")
    print(f"[explorer] spot Kidney Tumor/Lesion@CT -> {_spot_check(rows, 'Kidney Tumor/Lesion', 'CT')}")

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "explorer-data.js")
    emit_js(out_path, rows, body_parts, tasks, dataset_info)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[explorer] wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
