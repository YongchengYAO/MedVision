#!/usr/bin/env python
"""Inspect a verl-ready MedVision RFT parquet dataset (file or dataset directory).

Purpose
    Report, without loading images into RAM, what a parquet produced by
    ``python -m medvision_bm.rft.verl.build_parquet_ds`` (or its ``__checkpointed`` /
    ``_with_testset`` variants) contains: the Arrow schema, row counts, per-task counts
    (``data_source`` / ``ability`` columns), the first row's prompt / ground truth /
    reward_model / extra_info (truncated), and a description of the embedded ``images``
    column. When given a dataset DIRECTORY it also lists ``train_verl.parquet``,
    ``validation_verl.parquet``, ``test_verl.parquet``, ``shards/train_shard_*.parquet``
    and the ``checkpoint.json`` written by the checkpointed builder.

Prerequisites
    Only ``pyarrow`` (installed together with the ``datasets`` dependency of medvision_bm).
    ``Pillow`` is optional and only used to decode the first image's size.
    No GPU, no network, no medvision_bm import.

Examples
    python inspect_parquet_ds.py --path <data_dir>/verl_datasets/qwen25vl/ds__AD5500_D0_TL0_all5500__resized-hw-512x512
    python inspect_parquet_ds.py --path <dir>/validation_verl.parquet --truncate 200
    python inspect_parquet_ds.py --path <dir> --json            # machine-readable summary

Exit codes
    0 success; 1 path/argument problem; 2 the parquet lacks one of the columns verl requires
    (prompt, ground_truth, data_source, ability, reward_model, extra_info, images).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

REQUIRED_COLUMNS = [
    "prompt",
    "ground_truth",
    "data_source",
    "ability",
    "reward_model",
    "extra_info",
    "images",
]
SPLIT_FILES = ["train_verl.parquet", "validation_verl.parquet", "test_verl.parquet"]


def _import_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet as pq

        return pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        sys.stderr.write(
            "ERROR: pyarrow is required (it ships with the `datasets` dependency of medvision_bm).\n"
            f"       {exc}\n       Fix: `pip install pyarrow` inside the environment you use for the builders.\n"
        )
        sys.exit(1)


def _truncate(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def discover(path: str):
    """Return (kind, files) where kind is 'file' or 'dir'."""
    if os.path.isfile(path):
        return "file", {"single": [path]}
    if not os.path.isdir(path):
        sys.stderr.write(f"ERROR: --path does not exist: {path}\n")
        sys.exit(1)
    files = {}
    for name in SPLIT_FILES:
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            files[name.replace("_verl.parquet", "")] = [fp]
    shards = sorted(glob.glob(os.path.join(path, "shards", "train_shard_*.parquet")))
    if shards:
        files["shards"] = shards
    if not files:
        sys.stderr.write(
            f"ERROR: no {', '.join(SPLIT_FILES)} or shards/train_shard_*.parquet found under {path}\n"
        )
        sys.exit(1)
    return "dir", files


def summarize_files(pq, paths, truncate: int, show_first_row: bool):
    """Summarize one split (possibly several shard files)."""
    schema = pq.read_schema(paths[0])
    columns = schema.names
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]

    n_rows = 0
    per_source = collections.Counter()
    per_ability = collections.Counter()
    count_cols = [c for c in ("data_source", "ability") if c in columns]
    for fp in paths:
        meta = pq.read_metadata(fp)
        n_rows += meta.num_rows
        if count_cols:
            tbl = pq.read_table(fp, columns=count_cols)
            if "data_source" in count_cols:
                per_source.update(tbl.column("data_source").to_pylist())
            if "ability" in count_cols:
                per_ability.update(tbl.column("ability").to_pylist())

    summary = {
        "files": paths,
        "n_files": len(paths),
        "n_rows": n_rows,
        "columns": columns,
        "missing_required_columns": missing,
        "schema": schema.to_string(show_schema_metadata=False),
        "per_data_source": dict(per_source),
        "per_ability": dict(per_ability),
        "first_row": None,
        "images": None,
    }

    if "images" in columns:
        img_type = schema.field("images").type
        summary["images"] = {"arrow_type": str(img_type)}

    if show_first_row and n_rows > 0:
        # Read only the first row group of the first file to keep memory small.
        pf = pq.ParquetFile(paths[0])
        first = pf.read_row_group(0).slice(0, 1).to_pylist()[0]
        row = {}
        for key in ("prompt", "ground_truth", "data_source", "ability", "reward_model", "extra_info"):
            if key in first:
                row[key] = first[key]
        if "images" in first and first["images"] is not None:
            imgs = first["images"]
            info = {"n_images": len(imgs)}
            if imgs and isinstance(imgs[0], dict):
                info["path"] = imgs[0].get("path")
                data = imgs[0].get("bytes")
                info["bytes"] = len(data) if data is not None else None
                if data is not None:
                    try:
                        import io

                        from PIL import Image

                        with Image.open(io.BytesIO(data)) as im:
                            info["decoded"] = {"format": im.format, "mode": im.mode, "size_wh": list(im.size)}
                    except Exception as exc:  # Pillow missing or undecodable
                        info["decoded"] = f"not decoded ({type(exc).__name__})"
            summary["images"].update(info) if summary["images"] else None
        summary["first_row"] = row
    return summary


def print_summary(name: str, s: dict, truncate: int):
    print(f"\n=== {name} ===")
    print(f"files      : {s['n_files']}" + (f" (first: {s['files'][0]})" if s["n_files"] > 1 else f" ({s['files'][0]})"))
    print(f"rows       : {s['n_rows']}")
    print(f"columns    : {', '.join(s['columns'])}")
    if s["missing_required_columns"]:
        print(f"MISSING verl columns: {', '.join(s['missing_required_columns'])}")
    print("schema     :")
    for line in s["schema"].splitlines():
        print("    " + line)
    if s["per_data_source"]:
        print("per data_source:")
        for k, v in sorted(s["per_data_source"].items()):
            print(f"    {k:22s} {v}")
    if s["per_ability"]:
        print("per ability:")
        for k, v in sorted(s["per_ability"].items()):
            print(f"    {k:22s} {v}")
    if s["images"]:
        print(f"images     : {json.dumps(s['images'])}")
    row = s["first_row"]
    if row:
        print("first row  :")
        prompt = row.get("prompt")
        if isinstance(prompt, list):
            for msg in prompt:
                role = msg.get("role")
                parts = msg.get("content") or []
                for part in parts:
                    if part.get("type") == "image":
                        print(f"    [{role}] <image>")
                    elif part.get("type") == "text":
                        print(f"    [{role}] {_truncate(part.get('text', ''), truncate)}")
        for key in ("ground_truth", "data_source", "ability", "reward_model", "extra_info"):
            if key in row:
                print(f"    {key:12s}: {_truncate(json.dumps(row[key]), truncate)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 ok, 1 path/argument problem, 2 required verl column(s) missing.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="A *.parquet file, or a verl dataset directory holding train_verl.parquet / "
        "validation_verl.parquet / test_verl.parquet / shards/ / checkpoint.json",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "validation", "test", "shards"],
        help="Only inspect this split when --path is a directory (default: every split found).",
    )
    parser.add_argument("--truncate", type=int, default=400, help="Max characters printed per text field (default 400).")
    parser.add_argument("--no-first-row", action="store_true", help="Skip reading and printing the first row.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary instead of the human-readable report.")
    args = parser.parse_args(argv)

    pq = _import_pyarrow()
    kind, files = discover(args.path)
    if kind == "dir" and args.split:
        if args.split not in files:
            sys.stderr.write(f"ERROR: split '{args.split}' not found under {args.path}; available: {sorted(files)}\n")
            return 1
        files = {args.split: files[args.split]}

    report = {"path": args.path, "kind": kind, "splits": {}, "checkpoint": None}
    if kind == "dir":
        ckpt = os.path.join(args.path, "checkpoint.json")
        if os.path.isfile(ckpt):
            with open(ckpt) as fh:
                report["checkpoint"] = json.load(fh)

    any_missing = False
    for name, paths in files.items():
        s = summarize_files(pq, paths, args.truncate, show_first_row=not args.no_first_row)
        any_missing = any_missing or bool(s["missing_required_columns"])
        report["splits"][name] = s

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"path: {args.path}  ({kind})")
        if report["checkpoint"] is not None:
            c = report["checkpoint"]
            done = len(c.get("completed_train_shards", []))
            print(
                "checkpoint.json: "
                f"{done}/{c.get('n_shards')} train shards done, val_done={c.get('val_done')}, "
                f"test_done={c.get('test_done', 'n/a')}, merged={c.get('merged')}, total_train={c.get('total_train')}"
            )
        for name, s in report["splits"].items():
            print_summary(name, s, args.truncate)

    if any_missing:
        sys.stderr.write(
            "\nVALIDATION FAILED: at least one split lacks a column verl requires "
            f"({', '.join(REQUIRED_COLUMNS)}). This file was not produced by the MedVision verl builders "
            "or was post-processed.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
