#!/usr/bin/env python3
"""Inspect the benchmark plans of one downloaded MedVision dataset, offline.

Purpose
    Lists the ``benchmark_plan_<kind>_v<X.Y.Z>.json.gz`` files in a dataset directory,
    resolves the plan a given annotation pin would load (the loader's ceiling rule: the
    newest version at or below the pin), and summarises the resolved plan: dataset info,
    tasks, train/test case counts, per-plane slice-entry counts on a few cases, and the
    keys of the first case and first slice entry. No Hugging Face access, no ``nibabel``.

Prerequisites
    A downloaded dataset directory ``<data_dir>/Datasets/<dataset>``. Uses
    ``medvision_bm.utils.plan_utils`` when importable; otherwise a minimal local
    re-implementation of the same rules is used and a note is printed to stderr.

Examples
    python inspect_benchmark_plan.py --dataset-dir <data_dir>/Datasets/KiPA22 --plan-type biometry
    python inspect_benchmark_plan.py --dataset-dir <data_dir>/Datasets/KiPA22 --plan-type biometry --version 1.1.1
    python inspect_benchmark_plan.py --dataset-dir <data_dir>/Datasets/AMOS22 --plan-type detection --no-load
    python inspect_benchmark_plan.py --dataset-dir ... --plan-type segmentation --json

Exit status
    0 success; 1 no plan of that kind resolves at the pin; 2 bad arguments or missing directory.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

PLAN_TYPES = ("segmentation", "detection", "biometry")
AXES = ("x", "y", "z")
_LOCAL_AXIS_TO_PLANE = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect MedVision benchmark plans in a dataset directory (offline).")
    p.add_argument("--dataset-dir", required=True, help="<data_dir>/Datasets/<dataset>")
    p.add_argument("--plan-type", required=True, choices=PLAN_TYPES,
                   help="plan kind: MaskSize->segmentation, BoxSize->detection, TumorLesionSize/BiometricsFromLandmarks->biometry")
    p.add_argument("--version", default=None, metavar="X.Y.Z",
                   help="annotation pin (ceiling); default: newest available")
    p.add_argument("--max-cases", type=int, default=5,
                   help="cases per split used for per-plane slice-entry counts (default 5; -1 = all)")
    p.add_argument("--max-load-mb", type=float, default=200.0,
                   help="skip loading a plan file larger than this many MB (detection plans can be huge)")
    p.add_argument("--no-load", action="store_true", help="only list and resolve plan files")
    p.add_argument("--json", action="store_true", help="emit JSON")
    return p.parse_args()


# --------------------------------------------------------------------------- plan_utils import
def get_plan_utils():
    try:
        from medvision_bm.utils import plan_utils as pu  # type: ignore
        return pu, "medvision_bm.utils.plan_utils"
    except Exception as exc:
        print(f"[inspect_benchmark_plan] medvision_bm not importable ({type(exc).__name__}: {exc}); "
              "using a local re-implementation of the plan_utils rules.", file=sys.stderr)

        class _Local:
            AXIS_TO_PLANE = _LOCAL_AXIS_TO_PLANE

            @staticmethod
            def find_plan_files(dataset_dir, plan_type):
                return sorted(glob.glob(os.path.join(dataset_dir, f"benchmark_plan_{plan_type}_v*.json.gz")))

            @staticmethod
            def plan_version_of(path):
                return tuple(int(x) for x in os.path.basename(path).rsplit("_v", 1)[1].split(".json")[0].split("."))

            @classmethod
            def resolve_plan_path(cls, dataset_dir, plan_type, version=None):
                files = cls.find_plan_files(dataset_dir, plan_type)
                if version is not None:
                    cap = tuple(int(x) for x in version.split("."))
                    files = [f for f in files if cls.plan_version_of(f) <= cap]
                return max(files, key=cls.plan_version_of) if files else None

            @classmethod
            def dataset_exists_at(cls, dataset_dir, version=None):
                return any(cls.resolve_plan_path(dataset_dir, t, version) for t in PLAN_TYPES)

            @classmethod
            def load_benchmark_plan(cls, dataset_dir, plan_type, version=None):
                path = cls.resolve_plan_path(dataset_dir, plan_type, version)
                if path is None:
                    return None
                with gzip.open(path, "rt") as fh:
                    return json.load(fh)

            @staticmethod
            def split_cases(task, split):
                if split == "train":
                    return task.get("train_cases", []) or []
                if split == "test":
                    return task.get("test_cases", []) or []
                return (task.get("train_cases", []) or []) + (task.get("test_cases", []) or [])

            @staticmethod
            def slice_entries(case, axis):
                return case.get(f"slice_profiles_{axis}", []) or []

            @staticmethod
            def slice_2d_size(array_size, axis):
                idx = {"x": 0, "y": 1, "z": 2}[axis]
                dims = [d for i, d in enumerate(array_size) if i != idx]
                return int(dims[0]), int(dims[1])

        return _Local, "local"


def vstr(t) -> str:
    return ".".join(str(x) for x in t)


def summarise_task(pu, task, max_cases):
    info = {
        "task_ID": task.get("task_ID"),
        "task_type": task.get("task_type"),
        "image_modality": task.get("image_modality"),
        "image_folder": task.get("image_folder"),
        "labels_map": task.get("labels_map"),
        "train_cases": len(pu.split_cases(task, "train")),
        "test_cases": len(pu.split_cases(task, "test")),
    }
    for key in ("target_label", "landmark_folder", "cluster_size_threshold", "min_major_axis_mm", "biometrics_map"):
        if key in task:
            info[key] = task[key]
    per_plane = {}
    for split in ("train", "test"):
        cases = pu.split_cases(task, split)
        if max_cases >= 0:
            cases = cases[:max_cases]
        counts = {pu.AXIS_TO_PLANE[a]: sum(len(pu.slice_entries(c, a)) for c in cases) for a in AXES}
        per_plane[split] = {"cases_counted": len(cases), "slice_entries": counts}
    info["slice_entries_sample"] = per_plane
    first = (pu.split_cases(task, "test") or pu.split_cases(task, "train") or [None])[0]
    if first is not None:
        fi = first.get("image_file_info", {}) or {}
        arr = fi.get("array_size")
        info["first_case"] = {
            "keys": sorted(first.keys()),
            "case_ID": first.get("case_ID"),
            "image_file": first.get("image_file"),
            "array_size": arr,
            "voxel_size": fi.get("voxel_size"),
            "slice_2d_hw": {pu.AXIS_TO_PLANE[a]: pu.slice_2d_size(arr, a) for a in AXES} if arr else None,
        }
        for a in AXES:
            entries = pu.slice_entries(first, a)
            if entries:
                info["first_case"]["first_slice_entry"] = {
                    "plane": pu.AXIS_TO_PLANE[a],
                    "keys": sorted(entries[0].keys()),
                    "preview": json.dumps(entries[0])[:300],
                }
                break
    return info


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.dataset_dir):
        print(f"[inspect_benchmark_plan] not a directory: {args.dataset_dir}", file=sys.stderr)
        return 2
    pu, backend = get_plan_utils()

    report = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "dataset": os.path.basename(os.path.normpath(args.dataset_dir)),
        "plan_type": args.plan_type,
        "requested_version": args.version or "newest",
        "backend": backend,
        "versions_on_disk": {
            t: [vstr(pu.plan_version_of(f)) for f in pu.find_plan_files(args.dataset_dir, t)] for t in PLAN_TYPES
        },
        "dataset_exists_at_pin": bool(pu.dataset_exists_at(args.dataset_dir, args.version)),
    }
    resolved = pu.resolve_plan_path(args.dataset_dir, args.plan_type, args.version)
    if resolved is None:
        report["resolved"] = None
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"{report['dataset']}: no {args.plan_type} plan at or below {report['requested_version']}; "
                  f"on disk: {report['versions_on_disk']}")
        return 1
    size_mb = os.path.getsize(resolved) / 1e6
    report["resolved"] = {
        "path": resolved,
        "version": vstr(pu.plan_version_of(resolved)),
        "exact_match": args.version is None or resolved.endswith(f"_v{args.version}.json.gz"),
        "size_mb": round(size_mb, 2),
    }

    if args.no_load:
        report["loaded"] = False
    elif size_mb > args.max_load_mb:
        report["loaded"] = False
        report["skip_reason"] = f"file is {size_mb:.1f} MB > --max-load-mb {args.max_load_mb}"
    else:
        plan = pu.load_benchmark_plan(args.dataset_dir, args.plan_type, args.version)
        report["loaded"] = True
        report["dataset_info"] = plan.get("dataset_info")
        report["tasks_number"] = plan.get("tasks_number")
        report["tasks"] = [summarise_task(pu, t, args.max_cases) for t in plan.get("tasks", [])]

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"dataset      : {report['dataset']}  ({report['dataset_dir']})")
    print(f"plan type    : {args.plan_type}   requested: {report['requested_version']}   backend: {backend}")
    for t in PLAN_TYPES:
        print(f"  on disk {t:<13}: {', '.join(report['versions_on_disk'][t]) or '(none)'}")
    r = report["resolved"]
    tag = "exact" if r["exact_match"] else "CEILING FALLBACK"
    print(f"resolved     : {os.path.basename(r['path'])}  v{r['version']}  [{tag}]  {r['size_mb']} MB")
    print(f"exists at pin: {report['dataset_exists_at_pin']}")
    if not report.get("loaded"):
        print(f"not loaded   : {report.get('skip_reason', '--no-load')}")
        return 0
    di = report.get("dataset_info") or {}
    print(f"dataset_info : {di.get('dataset')}  license={di.get('license')}  tasks_number={report['tasks_number']}")
    for t in report["tasks"]:
        print(f"- Task{t['task_ID']}  type={t['task_type']}  modality={t['image_modality']}  "
              f"train={t['train_cases']}  test={t['test_cases']}")
        if t.get("labels_map"):
            print(f"    labels_map   : {t['labels_map']}")
        if "target_label" in t:
            print(f"    target_label : {t['target_label']}  landmark_folder={t.get('landmark_folder')}  "
                  f"min_major_axis_mm={t.get('min_major_axis_mm')}  cluster_size_threshold={t.get('cluster_size_threshold')}")
        for split, s in t["slice_entries_sample"].items():
            print(f"    {split:<5} slice entries over first {s['cases_counted']} cases: {s['slice_entries']}")
        fc = t.get("first_case")
        if fc:
            print(f"    first case   : {fc['case_ID']}  {fc['image_file']}  array_size={fc['array_size']}  "
                  f"voxel_size={fc['voxel_size']}")
            print(f"    2D (H,W)     : {fc['slice_2d_hw']}")
            print(f"    case keys    : {fc['keys']}")
            if fc.get("first_slice_entry"):
                e = fc["first_slice_entry"]
                print(f"    slice entry  : [{e['plane']}] keys={e['keys']}")
                print(f"                   {e['preview']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
