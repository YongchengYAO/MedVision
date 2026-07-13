"""Shared driver for the per-task size/spacing distribution probes.

``configs_to_pixel_sizes.py`` and ``configs_to_image_sizes.py`` are the same tool up to one column
and one key formatter: they stream each MedVision config from the HF dataset and bucket a cheap
per-sample metadata column (``pixel_size`` mm, or ``image_size_2d`` px). Streaming the HF loader
means per-family and version-conditional filtering (only Tumor-Lesion differs by version) and
version selection (env var ``MedVision_PLANNER_VERSION``) are inherited from the dataset itself —
nothing is reimplemented here. This module holds everything the two entry points share so they
cannot drift; each supplies only ``column``, ``key_fn`` and ``summary_fn``.
"""

import argparse
import collections
import csv
import json
import os

from datasets import load_dataset

from medvision_bm.utils import setup_env_hf_medvision_ds
from medvision_bm.utils.configs_to_tasks import config_to_task


def build_parser(description):
    """The argparse parser shared by both probes (version comes from the env, not a flag)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", required=True, help="MedVision data/code directory.")
    parser.add_argument("--configs_csv", required=True, help="Path to ConfigurationsList_*.csv.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
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
    return parser


def select_configs(configs, families, planes, split):
    """Filter CSV config rows by family / plane / split. Positions are split-robust.

    Dataset names never contain ``_``, so ``parts[1]`` is the family and ``parts[-2:]`` are the
    plane and split regardless of how many ``_`` the dataset name itself would add.
    """
    selected = []
    for config in configs:
        parts = config.split("_")
        family, plane, row_split = parts[1], parts[-2], parts[-1]
        if family.lower() not in families:
            continue
        if plane.lower() not in planes:
            continue
        if split != "all" and row_split.lower() != split:
            continue
        selected.append((config, row_split.lower()))
    return selected


def collect_distribution(config, split, column, key_fn, streaming):
    """Bucket ``key_fn(row[column])`` over all HF samples of one config, from the HF loader."""
    ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        split=split,
        trust_remote_code=True,
        streaming=streaming,
    )
    counter = collections.Counter()
    if streaming:
        # Keep only the one column we bucket. This avoids decoding image bytes and sidesteps the
        # strict arrow-cast failure on annotation structs (e.g. `bounding_boxes` carries an extra
        # `mask_image_ratio` field the declared Features omits, which streaming's cast rejects).
        ds = ds.select_columns([column])
        for row in ds:
            counter[key_fn(row[column])] += 1
    else:
        for value in ds[column]:
            counter[key_fn(value)] += 1
    # Sort by descending count, then key, for readable output.
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def run(args, column, key_fn, summary_fn, item_label):
    """Full main loop: env setup, config selection, per-config collect, incremental write.

    Writes ``args.out`` (``{task: dist, __all_tasks__: merged}``) and a ``__summary`` sidecar
    (``{task: summary_fn(dist), __all_tasks__: summary_fn(merged)}``). The rollup summary is
    recomputed from the merged distribution every step, so a partial file is always self-consistent.
    """
    families = {f.lower() for f in args.families.split(",")}
    planes = {p.lower() for p in args.planes.split(",")}

    # Honor the MedVision_FORCE_INSTALL_CODE env var (default: install). Passing force_install_code=
    # True would clobber an exported "false"; passing False leaves the exported value intact.
    force_install_code = (
        os.environ.get("MedVision_FORCE_INSTALL_CODE", "true").lower() != "false"
    )
    setup_env_hf_medvision_ds(args.data_dir, force_install_code=force_install_code)

    with open(args.configs_csv, "r") as f:
        configs = [row[0] for row in csv.reader(f) if row]

    selected = select_configs(configs, families, planes, args.split)
    if args.limit:
        selected = selected[: args.limit]

    base, ext = os.path.splitext(args.out)
    sum_out = f"{base}__summary{ext}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"\nSelected {len(selected)} configs -> {args.out}\n")

    tasks, sum_tasks = {}, {}
    overall = collections.Counter()
    for i, (config, row_split) in enumerate(selected, 1):
        task = config_to_task(config, args.cot)
        dist = (
            {}
            if args.no_count
            else collect_distribution(config, row_split, column, key_fn, not args.no_streaming)
        )
        tasks[task] = dist
        overall.update(dist)
        sum_tasks[task] = summary_fn(dist)
        n = sum(dist.values())
        print(f"[{i}/{len(selected)}] {task}: {n} samples, {len(dist)} distinct {item_label}")
        # Write incrementally so a long run is resumable/inspectable.
        out = dict(tasks)
        out["__all_tasks__"] = dict(sorted(overall.items(), key=lambda kv: (-kv[1], kv[0])))
        with open(args.out, "w") as f:
            json.dump(out, f, indent=4)
        sum_data = dict(sum_tasks)
        sum_data["__all_tasks__"] = summary_fn(out["__all_tasks__"])
        with open(sum_out, "w") as f:
            json.dump(sum_data, f, indent=4)

    print(f"\nWrote {len(tasks)} tasks (+ __all_tasks__ rollup) to {args.out}")
    print(f"Wrote summary to {sum_out}\n")
