import argparse
import collections
import csv
import json
import os

from datasets import load_dataset

from medvision_bm.utils import setup_env_hf_medvision_ds
from medvision_bm.utils.configs_to_tasks import config_to_task


def _ps_key(ps):
    # `pixel_size` is the slice's in-plane [height, width] spacing in mm (float16).
    # Round to 3 decimals to absorb float16 noise (matches the planner's voxel_size
    # rounding), so one physical spacing maps to one bucket.
    h, w = float(ps[0]), float(ps[1])
    return f"{h:.3f}x{w:.3f}"


def _iso_summary(dist):
    # Isotropic = square pixels (height == width). Keys are "{h:.3f}x{w:.3f}",
    # so a string compare of the two sides is exact at the bucket precision.
    iso = aniso = 0
    for key, n in dist.items():
        h, w = key.split("x")
        if h == w:
            iso += n
        else:
            aniso += n
    return {"isotropic": iso, "anisotropic": aniso}


def collect_pixel_sizes(config, split, streaming):
    ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        split=split,
        trust_remote_code=True,
        streaming=streaming,
    )
    counter = collections.Counter()
    if streaming:
        # Keep only the pixel_size column. This avoids decoding anything else and
        # sidesteps the strict arrow-cast failure on annotation structs (e.g.
        # `bounding_boxes` carries an extra `mask_image_ratio` field the declared
        # Features schema omits, which streaming's cast rejects).
        ds = ds.select_columns(["pixel_size"])
        for row in ds:
            counter[_ps_key(row["pixel_size"])] += 1
    else:
        for ps in ds["pixel_size"]:
            counter[_ps_key(ps)] += 1
    # Sort by descending count, then key, for readable output.
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize per-task raw pixel-size distributions from a MedVision config list (CSV)."
    )
    parser.add_argument("--data_dir", required=True, help="MedVision data/code directory.")
    parser.add_argument("--configs_csv", required=True, help="Path to ConfigurationsList_*.csv.")
    parser.add_argument("--out", required=True, help="Output pixel-size JSON path.")
    parser.add_argument(
        "--families",
        default="BoxSize,MaskSize,TumorLesionSize,BiometricsFromLandmarks",
        help="Comma-separated task families to include.",
    )
    parser.add_argument(
        "--planes",
        default="Axial,Coronal,Sagittal",
        help="Comma-separated planes to include.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "all"],
        help="Split to include and to count (default: test).",
    )
    parser.add_argument("--cot", action="store_true", help="Append '-CoT' to task names.")
    parser.add_argument("--limit", type=int, help="Cap the number of configs (for testing).")
    parser.add_argument(
        "--no-count",
        action="store_true",
        help="Skip dataset loading; write empty distributions (fast naming-only run).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Count via load_dataset(...) instead of streaming (materializes Arrow cache).",
    )
    args = parser.parse_args()

    families = {f.lower() for f in args.families.split(",")}
    planes = {p.lower() for p in args.planes.split(",")}

    # Honor the MedVision_FORCE_INSTALL_CODE env var (default: install).
    # Passing force_install_code=True would clobber an exported "false";
    # passing False leaves the exported value intact.
    force_install_code = (
        os.environ.get("MedVision_FORCE_INSTALL_CODE", "true").lower() != "false"
    )
    setup_env_hf_medvision_ds(args.data_dir, force_install_code=force_install_code)

    with open(args.configs_csv, "r") as f:
        configs = [row[0] for row in csv.reader(f) if row]

    # Filter rows by family / plane / split (positions are split-robust).
    selected = []
    for config in configs:
        parts = config.split("_")
        family, plane, row_split = parts[1], parts[-2], parts[-1]
        if family.lower() not in families:
            continue
        if plane.lower() not in planes:
            continue
        if args.split != "all" and row_split.lower() != args.split:
            continue
        selected.append((config, row_split.lower()))

    if args.limit:
        selected = selected[: args.limit]

    # Sibling file with the isotropic/anisotropic sample counts per task.
    base, ext = os.path.splitext(args.out)
    iso_out = f"{base}__summary{ext}"

    print(f"\nSelected {len(selected)} configs -> {args.out}\n")

    tasks = {}
    iso_tasks = {}
    overall = collections.Counter()
    iso_overall = collections.Counter()
    for i, (config, row_split) in enumerate(selected, 1):
        task = config_to_task(config, args.cot)
        dist = {} if args.no_count else collect_pixel_sizes(config, row_split, not args.no_streaming)
        tasks[task] = dist
        overall.update(dist)
        iso = _iso_summary(dist)
        iso_tasks[task] = iso
        iso_overall.update(iso)
        n = sum(dist.values())
        print(
            f"[{i}/{len(selected)}] {task}: {n} samples, {len(dist)} distinct pixel sizes "
            f"(isotropic={iso['isotropic']}, anisotropic={iso['anisotropic']})"
        )
        # Write incrementally so a long run is resumable/inspectable. The rollups
        # are recomputed each time so partial files are always self-consistent.
        out = dict(tasks)
        out["__all_tasks__"] = dict(sorted(overall.items(), key=lambda kv: (-kv[1], kv[0])))
        with open(args.out, "w") as f:
            json.dump(out, f, indent=4)
        iso_out_data = dict(iso_tasks)
        iso_out_data["__all_tasks__"] = dict(iso_overall)
        with open(iso_out, "w") as f:
            json.dump(iso_out_data, f, indent=4)

    print(f"\nWrote {len(tasks)} tasks (+ __all_tasks__ rollup) to {args.out}")
    print(f"Wrote isotropy summary to {iso_out}\n")


if __name__ == "__main__":
    main()
