#!/usr/bin/env python
"""Read-only inspection of a prepared MedVision SFT dataset directory.

Purpose
-------
Phase A of every SFT launcher (``--process_dataset_only true``) writes a
``datasets.DatasetDict`` with ``save_to_disk`` into ``prepared_ds_dir``
(default ``<data_dir>/SFT-CoT_datasets/<model_family_name>/ds__AD..._D..._TL..._all...__resized-wh-WxH``).
This script loads that directory with ``datasets.load_from_disk`` and prints
the splits, columns, row counts per task (from the ``__task_name`` column the
trainers add for the temperature sampler), and one example's chat messages,
truncated. Nothing is written.

Prerequisites
-------------
``datasets`` (the ``medvision_bm`` pin is 3.6.0). No GPU, no network.

Examples
--------
    python inspect_prepared_dataset.py --prepared-ds-dir <data_dir>/SFT-CoT_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-wh-512x512
    python inspect_prepared_dataset.py --prepared-ds-dir <dir> --split validation --example-index 3 --max-chars 600
    python inspect_prepared_dataset.py --prepared-ds-dir <dir> --check-images 50
"""

from __future__ import annotations

import argparse
import collections
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prepared-ds-dir", required=True,
                   help="Directory written by dataset.save_to_disk (contains dataset_dict.json).")
    p.add_argument("--task-column", default="__task_name",
                   help="Column holding the task label per row (trainer default: __task_name).")
    p.add_argument("--split", default="train", help="Split to take the example from (default: train).")
    p.add_argument("--example-index", type=int, default=0, help="Row index of the example to print.")
    p.add_argument("--max-chars", type=int, default=400, help="Truncate each printed text block to this many characters.")
    p.add_argument("--check-images", type=int, default=0, metavar="N",
                   help="Also verify that the first N rows' image files (image_file_png, else image_file) exist on disk.")
    return p


def _truncate(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + f" ... [{len(text) - n} more chars]"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[error] the 'datasets' package is required: {exc}\n        pip install 'datasets==3.6.0'",
              file=sys.stderr)
        return 3

    ds_dir = args.prepared_ds_dir
    if not os.path.isdir(ds_dir):
        print(f"[error] not a directory: {ds_dir}", file=sys.stderr)
        return 2
    if not (os.path.exists(os.path.join(ds_dir, "dataset_dict.json"))
            or os.path.exists(os.path.join(ds_dir, "dataset_info.json"))):
        print(f"[error] {ds_dir} has neither dataset_dict.json nor dataset_info.json; "
              "is this the prepared_ds_dir printed by the trainer ('Prepared dataset saved at ...')?",
              file=sys.stderr)
        return 2

    ds = load_from_disk(ds_dir)
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"(single split)": ds})

    print(f"Prepared dataset: {os.path.realpath(ds_dir)}")
    if os.path.islink(ds_dir):
        print(f"  (symlink -> {os.readlink(ds_dir)})")
    print(f"Splits: {list(ds.keys())}")
    for name, split in ds.items():
        print(f"\n[{name}] rows={len(split)}")
        print(f"  columns: {split.column_names}")
        if args.task_column in split.column_names:
            counts = collections.Counter(split[args.task_column])
            for task, n in sorted(counts.items()):
                print(f"  {task:<10} {n:>10} rows  ({100.0 * n / max(1, len(split)):.1f}%)")
        else:
            print(f"  (no '{args.task_column}' column: prepared before task tagging or a custom column; "
                  "the temperature sampler needs it)")
        for col in ("image_file_png", "processed_images"):
            if col in split.column_names:
                print(f"  image source column present: {col}")

    if args.split not in ds:
        print(f"\n[error] split '{args.split}' not found", file=sys.stderr)
        return 2
    split = ds[args.split]
    if not (0 <= args.example_index < len(split)):
        print(f"\n[error] --example-index {args.example_index} out of range for {len(split)} rows", file=sys.stderr)
        return 2

    row = split[args.example_index]
    print(f"\nExample {args.split}[{args.example_index}]:")
    for key in ("image_file", "slice_dim", "slice_idx", "image_file_png", "labels", args.task_column):
        if key in row:
            print(f"  {key}: {_truncate(row[key], args.max_chars)}")
    messages = row.get("messages")
    if isinstance(messages, list):
        print(f"  messages: {len(messages)} turn(s)")
        for i, turn in enumerate(messages):
            role = turn.get("role") if isinstance(turn, dict) else "?"
            content = turn.get("content") if isinstance(turn, dict) else turn
            texts, n_images = [], 0
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        n_images += 1
                    elif isinstance(part, dict) and part.get("text") is not None:
                        texts.append(part["text"])
            else:
                texts.append(str(content))
            print(f"   [{i}] role={role} images={n_images}")
            for t in texts:
                print("       " + _truncate(t, args.max_chars).replace("\n", "\n       "))
    else:
        print("  (no 'messages' column: this is not a formatted SFT dataset)")

    if args.check_images > 0:
        n = min(args.check_images, len(split))
        col = "image_file_png" if "image_file_png" in split.column_names else "image_file"
        missing = 0
        for r in split.select(range(n)):
            paths = r[col] if isinstance(r[col], list) else [r[col]]
            missing += sum(1 for pth in paths if not os.path.exists(pth))
        print(f"\nImage check on first {n} rows of '{args.split}' via '{col}': {missing} missing file(s)")
        if missing:
            print("  Missing files mean the data_dir moved or the PNG cache (tmp_prepared_png/) was deleted; "
                  "re-run phase A with --skip_process_dataset false.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
