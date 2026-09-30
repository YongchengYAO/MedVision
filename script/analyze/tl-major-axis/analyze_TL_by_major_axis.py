"""
Group T/L (Tumor/Lesion size) task performance by ground-truth major axis length.

Dataset v1.4.0 selects T/L measurements by a physical floor (major axis >=
max(2 mm, 2x the coarser in-plane spacing)) instead of a 20 px count, which
admits many more small lesions into the test set. This script checks whether a
metric drop on the v1.4.0 test set is explained by those small lesions: it bins
every sample by its GT major axis (``target[0]``, in mm) and reports the usual
TL metrics per bin.

Two blocks are printed per model:
  - Per bin      : samples whose GT major axis lies in (lo, hi] mm.
  - Above t (>t) : every sample whose GT major axis exceeds t mm, i.e. the metric
                   a user would see after applying a size floor of t.
The "All" row pools every sample and equals the model-level weighted average in
summary_TL_task*.txt (when SR = 1).

Samples are read and scored with the same functions as
medvision_bm.benchmark.summarize_TL_task, so the sample set and metrics match
the official summary. medvision_ds must be importable (label lookup):
    python -m medvision_bm.benchmark.install_medvision_ds --data_dir <repo>/Data

Usage:
    python analyze_TL_by_major_axis.py \
        --model_dir <task_dir>/<model_A> <task_dir>/<model_B> \
        [--parsed_dirname parsed] [--resps_key filtered_resps] \
        [--bin_edges 2 5 10 20 50 100] [--datasets KiPA22 BraTS24 ...] \
        [--removed_samples_dir /path/to/MedVision/Data/Datasets] \
        [--out_json summary.json]
"""

import argparse
import ast
import glob
import json
import os
import re

from medvision_bm.benchmark.summarize_TL_task import (
    _build_removed_set,
    process_jsonl_file_TL_task,
    process_label_group_TL,
)
from medvision_bm.utils.parse_utils import convert_numpy_to_python

METRIC_COLUMNS = [
    ("MAE", "avgMAE"),
    ("MRE", "avgMRE"),
    ("SR", "SuccessRate"),
    ("nMAE", "avgNMAE"),
    ("MRE<0.1", "MRE<0.1"),
    ("MRE<0.2", "MRE<0.2"),
    ("MRE<0.3", "MRE<0.3"),
]


def _dataset_name(jsonl_path):
    match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    return match.group(1) if match else None


def collect_samples(
    model_dir,
    parsed_dirname,
    resps_key,
    limit=None,
    datasets=None,
    removed_samples_dir=None,
    removed_samples_filename=None,
):
    """Return [(major_axis_mm, target, response, doc_meta)] for one model."""
    parsed_dir = os.path.join(model_dir, parsed_dirname)
    if not os.path.isdir(parsed_dir):
        raise FileNotFoundError(
            f"No parsed directory: {parsed_dir} (run parse_outputs first)"
        )

    jsonl_files = sorted(
        f
        for f in glob.glob(os.path.join(parsed_dir, "*.jsonl"))
        if not ("_proc_acc" in os.path.basename(f) or "_eq_acc" in os.path.basename(f))
    )
    if datasets:
        jsonl_files = [f for f in jsonl_files if _dataset_name(f) in datasets]

    samples = []
    removed_cache = {}
    for jsonl_file in jsonl_files:
        removed_set = None
        if removed_samples_dir:
            ds_name = _dataset_name(jsonl_file)
            if ds_name not in removed_cache:
                json_path = os.path.join(
                    removed_samples_dir, ds_name, removed_samples_filename
                )
                removed_cache[ds_name] = (
                    _build_removed_set(json_path) if os.path.exists(json_path) else None
                )
            removed_set = removed_cache[ds_name]

        for _, _, target, resp, _, _, doc_meta in process_jsonl_file_TL_task(
            jsonl_file, limit, removed_set=removed_set, resps_key=resps_key
        ):
            major = float(ast.literal_eval(target)[0])
            samples.append((major, target, resp, doc_meta))
    return samples, jsonl_files


def _metrics(name, subset):
    if not subset:
        return None
    _, metrics = process_label_group_TL(
        name,
        {
            "targets": [s[1] for s in subset],
            # filtered_resps is a one-element list; flatten like group_by_label_modality_slice
            "responses": [r for s in subset for r in s[2]],
            "doc_metas": [s[3] for s in subset],
        },
    )
    return metrics


def _fmt_edge(x):
    return f"{x:g}"


def summarize_by_major_axis(samples, bin_edges):
    """Compute per-bin and above-threshold metrics."""
    edges = sorted(bin_edges)
    bounds = [(None, edges[0])] + list(zip(edges[:-1], edges[1:])) + [(edges[-1], None)]

    per_bin = []
    for lo, hi in bounds:
        if lo is None:
            name = f"<={_fmt_edge(hi)} mm"
        elif hi is None:
            name = f">{_fmt_edge(lo)} mm"
        else:
            name = f"{_fmt_edge(lo)}-{_fmt_edge(hi)} mm"
        subset = [
            s
            for s in samples
            if (lo is None or s[0] > lo) and (hi is None or s[0] <= hi)
        ]
        per_bin.append({"bin": name, "metrics": _metrics(name, subset)})

    above = []
    for t in edges:
        name = f">{_fmt_edge(t)} mm"
        subset = [s for s in samples if s[0] > t]
        above.append({"bin": name, "metrics": _metrics(name, subset)})

    return {
        "per_bin": per_bin,
        "above_threshold": above,
        "all": _metrics("All", samples),
    }


def _format_table(title, rows, total):
    header = f"{'Major axis':<14} | {'Samples':>8} | {'Share':>6} | " + " | ".join(
        f"{c:>8}" for c, _ in METRIC_COLUMNS
    )
    lines = [title, header, "-" * len(header)]
    for row in rows:
        m = row["metrics"]
        n = m["num_samples"] if m else 0
        share = n / total if total else 0.0
        if m:
            vals = " | ".join(f"{m[k]:>8.4f}" for _, k in METRIC_COLUMNS)
        else:
            vals = " | ".join(f"{'-':>8}" for _ in METRIC_COLUMNS)
        lines.append(f"{row['bin']:<14} | {n:>8} | {share:>6.3f} | {vals}")
    return lines


def format_report(model_dir, summary, jsonl_files):
    total = summary["all"]["num_samples"] if summary["all"] else 0
    lines = [
        f"Model: {os.path.basename(os.path.normpath(model_dir))}",
        f"Path : {model_dir}",
        f"Files: {len(jsonl_files)} ("
        + ", ".join(sorted({_dataset_name(f) for f in jsonl_files}))
        + ")",
        "",
    ]
    lines += _format_table(
        "Per bin (GT major axis in (lo, hi] mm)",
        summary["per_bin"] + [{"bin": "All", "metrics": summary["all"]}],
        total,
    )
    lines.append("")
    lines += _format_table(
        "Above threshold (GT major axis > t mm)", summary["above_threshold"], total
    )
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Group TL task metrics by ground-truth major axis length (mm)."
    )
    parser.add_argument(
        "--model_dir",
        nargs="+",
        required=True,
        help="One or more model result directories (each containing <parsed_dirname>/).",
    )
    parser.add_argument("--parsed_dirname", default="parsed")
    parser.add_argument(
        "--resps_key",
        default="filtered_resps",
        help="Use 'LLM_filtered_resps' for llm-parsed_* directories.",
    )
    parser.add_argument(
        "--bin_edges",
        nargs="+",
        type=float,
        default=[2, 5, 10, 20, 50, 100],
        help="Major axis bin edges in mm (default matches the v1.4.0 release post).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to these dataset names (e.g. to compare runs on a common task set).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--removed_samples_dir", default=None)
    parser.add_argument(
        "--removed_samples_filename",
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
    )
    parser.add_argument(
        "--out_json", default=None, help="Optional path to save all metrics as JSON."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    for model_dir in args.model_dir:
        samples, jsonl_files = collect_samples(
            model_dir,
            args.parsed_dirname,
            args.resps_key,
            limit=args.limit,
            datasets=args.datasets,
            removed_samples_dir=args.removed_samples_dir,
            removed_samples_filename=args.removed_samples_filename,
        )
        summary = summarize_by_major_axis(samples, args.bin_edges)
        print("\n".join(format_report(model_dir, summary, jsonl_files)))
        print("\n" + "=" * 120 + "\n")
        results[model_dir] = summary

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(convert_numpy_to_python(results), f, indent=2)
        print(f"Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()
