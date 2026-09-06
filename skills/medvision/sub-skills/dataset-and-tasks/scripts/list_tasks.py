#!/usr/bin/env python3
"""List MedVision task names, sample counts and derived dataset configs from task-list JSONs.

Purpose
    A MedVision task list is a flat JSON object {"<task name>": <count>, ...}; the
    pipeline reads only the keys. This script prints, for every key, the informational
    count and the Hugging Face dataset config it resolves to for a split, so a list can be
    checked (and a download or evaluation planned) without any network access.

Config derivation
    Mirrors ``medvision_bm.utils.data_utils.tasks_to_configs`` (append ``_Train``/``_Test``,
    rewrite the legacy ``BoxCoordinate`` token to ``BoxSize``) with one extra safety step:
    everything after the plane token (``Axial``/``Coronal``/``Sagittal``) is a task-variant
    suffix (``-CoT``, ``-CoT-scaledPS``, ``-VP-woMedImg``) that never occurs in a config
    name, so it is removed first. Names without a plane token are reported and passed through.

Prerequisites
    Python >= 3.9 and the standard library. ``medvision_bm`` is used for the final
    conversion when importable; otherwise an equivalent local implementation is used and a
    note is printed to stderr.

Examples
    python list_tasks.py --tasks-json tasks_MedVision-TL-CoT.json
    python list_tasks.py --tasks-json tasks_MedVision-TL__train_SFT.json --split train
    # eval-style names and Test configs for a coronal plane-OOD study derived from an SFT list
    python list_tasks.py --tasks-json tasks_MedVision-TL__train_SFT.json --plane Coronal --cot add
    python list_tasks.py --tasks-json a.json --tasks-json b.json --json

Exit status
    0 on success, 1 if a file cannot be read or is not a JSON object, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

PLANES = ("Axial", "Coronal", "Sagittal")
_PLANE_RE = re.compile(r"^(?P<base>.*?_(?P<plane>Axial|Coronal|Sagittal))(?P<suffix>.*)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print task names, counts and derived MedVision dataset configs from task-list JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[1] if "Examples" in __doc__ else None,
    )
    p.add_argument("--tasks-json", action="append", required=True, metavar="PATH",
                   help="task-list JSON ({name: count}); repeatable")
    p.add_argument("--split", choices=("train", "test"), default="test",
                   help="split suffix for the derived configs (default: test)")
    p.add_argument("--plane", choices=PLANES, default=None,
                   help="rewrite the plane token of every task name before deriving configs")
    p.add_argument("--cot", choices=("keep", "add", "strip"), default="keep",
                   help="control the '-CoT' suffix of the printed task names (configs are unaffected)")
    p.add_argument("--json", action="store_true", help="emit a JSON array instead of a table")
    return p.parse_args()


def _local_tasks_to_configs(tasks: list[str], split: str) -> list[str]:
    split_token = "Train" if split.lower() == "train" else "Test"
    return [f"{t}_{split_token}".replace("BoxCoordinate", "BoxSize") for t in tasks]


def get_converter():
    try:
        from medvision_bm.utils.data_utils import tasks_to_configs  # type: ignore
        return tasks_to_configs, "medvision_bm.utils.data_utils.tasks_to_configs"
    except Exception as exc:  # ImportError or a missing transitive dependency such as `datasets`
        print(f"[list_tasks] medvision_bm not importable ({type(exc).__name__}: {exc}); "
              "using the local re-implementation of tasks_to_configs.", file=sys.stderr)
        return _local_tasks_to_configs, "local"


def split_name(name: str):
    """Return (base up to the plane token, plane, variant suffix); plane is None if absent."""
    m = _PLANE_RE.match(name)
    if not m:
        return name, None, ""
    return m.group("base"), m.group("plane"), m.group("suffix")


def edit_task_name(base: str, plane: str | None, suffix: str, new_plane: str | None, cot: str) -> str:
    if new_plane and plane:
        base = base[: -len(plane)] + new_plane
    if cot == "strip":
        suffix = suffix.replace("-CoT", "")
    elif cot == "add" and "-CoT" not in suffix:
        suffix = "-CoT" + suffix
    return base + suffix


def load_list(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a JSON object mapping task name -> count")
    return data


def main() -> int:
    args = parse_args()
    convert, backend = get_converter()
    rows = []
    failures = 0
    for path in args.tasks_json:
        try:
            tasks = load_list(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[list_tasks] cannot read {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        for name, count in tasks.items():
            base, plane, suffix = split_name(name)
            if plane is None:
                print(f"[list_tasks] warning: no plane token in '{name}'; config derived verbatim", file=sys.stderr)
            if args.plane and plane:
                base = base[: -len(plane)] + args.plane
            config = convert([base], args.split)[0]
            rows.append({
                "file": os.path.basename(path),
                "task": edit_task_name(base, args.plane or plane, suffix, None, args.cot),
                "count": count,
                "dataset": base.split("_")[0],
                "family": base.split("_")[1] if "_" in base else None,
                "plane": args.plane or plane,
                "split": args.split,
                "config": config,
                "variant_suffix": suffix,
            })

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
    else:
        widths = {k: max([len(k)] + [len(str(r[k])) for r in rows]) for k in ("task", "count", "config")}
        header = f"{'task':<{widths['task']}}  {'count':>{widths['count']}}  {'config':<{widths['config']}}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['task']:<{widths['task']}}  {str(r['count']):>{widths['count']}}  {r['config']:<{widths['config']}}")
        datasets = sorted({r["dataset"] for r in rows})
        families = sorted({r["family"] for r in rows if r["family"]})
        total = sum(c for c in (r["count"] for r in rows) if isinstance(c, int))
        print()
        print(f"{len(rows)} tasks | {len(datasets)} datasets: {', '.join(datasets)}")
        print(f"families: {', '.join(families)} | split: {args.split} | summed counts: {total} | converter: {backend}")
        print("note: counts are informational (taken when the list was written); only task names are used downstream.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
