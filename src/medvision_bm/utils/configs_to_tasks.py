import argparse
import csv
import json
import os

from datasets import load_dataset
from datasets.features.features import Value

from medvision_bm.utils import setup_env_hf_medvision_ds


def config_to_task(config, cot):
    # Strip the trailing split suffix (the CSV config name is the HF config name)
    for suffix in ("_Train", "_Test"):
        if config.endswith(suffix):
            config = config[: -len(suffix)]
            break
    # [Legacy naming] dataset stores "BoxSize"; eval tasks are named "BoxCoordinate"
    # (see utils/data_utils.py:tasks_to_configs for the reverse mapping).
    task = config.replace("BoxSize", "BoxCoordinate")
    if cot:
        task += "-CoT"
    return task


def count_samples(config, split, streaming):
    ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        split=split,
        trust_remote_code=True,
        streaming=streaming,
    )
    if streaming:
        # Keep only scalar columns before iterating. This both avoids decoding
        # image bytes and sidesteps a strict arrow-cast failure on the annotation
        # structs: e.g. `bounding_boxes` in the data carries an extra
        # `mask_image_ratio` field that the declared Features schema omits, which
        # streaming's cast_table_to_features rejects (non-streaming silently drops
        # it). We only need the row count, so dropping these columns is safe.
        scalar_cols = [c for c, f in ds.features.items() if isinstance(f, Value)]
        ds = ds.select_columns(scalar_cols)
        return sum(1 for _ in ds)
    return len(ds)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a MedVision config list (CSV) into a task list (JSON)."
    )
    parser.add_argument("--data_dir", required=True, help="MedVision data/code directory.")
    parser.add_argument("--configs_csv", required=True, help="Path to ConfigurationsList_*.csv.")
    parser.add_argument("--out", required=True, help="Output task-list JSON path.")
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
        help="Skip dataset loading; write 0 as the count (fast naming-only run).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Count via len(load_dataset(...)) instead of streaming (materializes Arrow cache).",
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

    print(f"\nSelected {len(selected)} configs -> {args.out}\n")

    tasks = {}
    for i, (config, row_split) in enumerate(selected, 1):
        task = config_to_task(config, args.cot)
        count = 0 if args.no_count else count_samples(config, row_split, not args.no_streaming)
        tasks[task] = count
        print(f"[{i}/{len(selected)}] {task}: {count}")
        # Write incrementally so a long run is resumable/inspectable.
        with open(args.out, "w") as f:
            json.dump(tasks, f, indent=4)

    print(f"\nWrote {len(tasks)} tasks to {args.out}\n")


if __name__ == "__main__":
    main()
