#!/usr/bin/env python3
"""Pretty-print MedVision summary files or inspect a parsed-results directory.

Purpose
    Read-only viewer for the files written by ``parse_outputs`` and the three
    ``summarize_*_task`` modules. It never writes anything. Given ``--path`` it
    detects what it is looking at:

    * ``summary_metrics_{TL,AD,detect}_Task*.json``  -> one row per group (label
      / anatomy / A-D metric), plus a sample-weighted overall row;
    * ``summary_metrics_anatomy_vs_lesion_detect_Task*.json`` -> the two group
      means (anatomy vs T/L) and their region lists;
    * ``summary_metrics_all_models_detect_Task*.json`` -> one row per model x group;
    * ``summary_values_*.json`` -> per-group counts of targets/responses and how
      many responses have the expected number of values;
    * ``<timestamp>_results.json`` (written by ``parse_outputs``) -> the metric
      block of that task;
    * a directory (``parsed/``, ``llm-parsed_<judge>/`` or a model dir) -> JSONL
      inventory (records, duplicate doc_ids, success count when scored) and the
      summary files found there.

Prerequisites
    Python 3.9+ standard library only. No ``medvision_bm`` import is required.

Examples
    python inspect_summary.py --path Results/MedVision-detect/<model>/parsed/summary_metrics_detect_Task.json
    python inspect_summary.py --path Results/MedVision-TL/<model>/parsed --sort-by num_samples
    python inspect_summary.py --path Results/MedVision-detect/summary_metrics_all_models_detect_Task.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

PREFERRED_COLUMNS = [
    "num_samples",
    "SuccessRate",
    "avgMAE",
    "avgMRE",
    "avgNMAE",
    "IoU",
    "F1",
    "Precision",
    "Recall",
    "IoU>0.5",
    "F1>0.5",
    "Acc" + chr(64) + "IoU>=0.50",
    "Acc" + chr(64) + "IoU[0.50:0.95]",
    "MRE<0.1",
    "MRE<0.2",
    "MRE<0.3",
]

EXPECTED_VALUES = {"TL": 2, "AD": 1, "detect": 4}


def _num(x):
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(x, width=9):
    if isinstance(x, str):
        return f"{x:<{width}}"
    v = _num(x)
    if math.isnan(v):
        return f"{'nan':<{width}}"
    if float(v).is_integer() and abs(v) >= 1:
        return f"{int(v):<{width}}"
    return f"{v:<{width}.4f}"


def _print_table(rows, columns, label_header="group", label_width=46):
    header = f"{label_header:<{label_width}} " + " ".join(f"{c:<9}" for c in columns)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        cells = " ".join(_fmt(metrics.get(c)) for c in columns)
        print(f"{str(name)[:label_width]:<{label_width}} {cells}")


def _weighted_row(rows, columns, weight_key="num_samples"):
    out = {}
    for c in columns:
        if c == weight_key:
            out[c] = sum(_num(m.get(weight_key)) for _, m in rows if not math.isnan(_num(m.get(weight_key))))
            continue
        num = den = 0.0
        for _, m in rows:
            v, w = _num(m.get(c)), _num(m.get(weight_key))
            if math.isnan(v) or math.isnan(w):
                continue
            num += v * w
            den += w
        out[c] = num / den if den > 0 else float("nan")
    return out


def _task_of(path):
    base = os.path.basename(path)
    for tag in ("TL", "AD", "detect"):
        if f"_{tag}_Task" in base:
            return tag
    return None


def show_metrics(data, path, sort_by, top):
    tag = _task_of(path)
    rows = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    if not rows:
        print("(empty metrics file)")
        return
    present = set().union(*(m.keys() for _, m in rows))
    columns = [c for c in PREFERRED_COLUMNS if c in present]
    if sort_by and sort_by in present:
        rows.sort(key=lambda kv: -_num(kv[1].get(sort_by)))
    if top:
        rows = rows[:top]
    print(f"metrics file: {path}\ntask: {tag or '?'}   groups: {len(data)}   shown: {len(rows)}")
    _print_table(rows, columns)
    wrow = _weighted_row(rows, columns)
    print("-" * (47 + 10 * len(columns)))
    _print_table([("[sample-weighted over shown rows]", wrow)], columns)
    print(
        "\nreading guide: avgMAE/avgMRE/avgNMAE average successful samples only; "
        "IoU/F1/Precision/Recall include failures as 0; every '>k' / '<k' key divides by num_samples."
    )


def show_anatomy_vs_lesion(data, path):
    print(f"grouped metrics file: {path}")
    rows = []
    for grp in ("anatomy", "T/L"):
        if grp in data:
            mm = dict(data[grp].get("mean_metrics", {}))
            mm["num_samples"] = mm.get("total_samples")
            rows.append((f"{grp} ({len(data[grp].get('regions', []))} regions)", mm))
    present = set().union(*(m.keys() for _, m in rows)) if rows else set()
    columns = [c for c in PREFERRED_COLUMNS if c in present]
    _print_table(rows, columns)
    for grp in ("anatomy", "T/L"):
        if grp in data:
            print(f"\n{grp} regions: {', '.join(data[grp].get('regions', []))}")
    print("\nregions with < MINIMUM_GROUP_SIZE samples or containing an EXCLUDED_KEYS word were dropped before these means.")


def show_all_models(data, path, sort_by, top):
    print(f"all-models file: {path}   models: {len(data)}")
    rows = []
    for model, groups in data.items():
        for grp, m in groups.items():
            mm = dict(m)
            mm["num_samples"] = mm.get("total_samples")
            rows.append((f"{model} | {grp}", mm))
    present = set().union(*(m.keys() for _, m in rows)) if rows else set()
    columns = [c for c in ("num_samples", "num_regions", "SuccessRate", "IoU", "F1", "Precision", "Recall", "IoU>0.5", "F1>0.5", "AccIoU_50", "AccIoU_75", "AccIoU_mean") if c in present]
    if sort_by and sort_by in present:
        rows.sort(key=lambda kv: -_num(kv[1].get(sort_by)))
    if top:
        rows = rows[:top]
    _print_table(rows, columns, label_header="model | group", label_width=60)


def _count_ok(responses, k):
    ok = 0
    for r in responses:
        parts = [p.strip() for p in str(r).split(",")] if r is not None else []
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) == k:
            ok += 1
    return ok


def show_values(data, path, top):
    tag = _task_of(path)
    k = EXPECTED_VALUES.get(tag or "", None)
    print(f"values file: {path}   task: {tag or '?'}")
    if isinstance(data, list):  # A/D values: flat list of {label, targets, responses, doc_meta}
        by_label = {}
        for item in data:
            d = by_label.setdefault(item.get("label"), {"targets": [], "responses": []})
            d["targets"].append(item.get("targets"))
            resp = item.get("responses") or [None]
            d["responses"].append(resp[0])
        data = by_label
    rows = []
    for grp, d in data.items():
        resps = d.get("responses", [])
        m = {"num_samples": len(d.get("targets", [])), "n_responses": len(resps)}
        if k:
            m["parseable"] = _count_ok(resps, k)
            m["SuccessRate"] = m["parseable"] / m["num_samples"] if m["num_samples"] else float("nan")
        rows.append((grp, m))
    rows.sort(key=lambda kv: -kv[1]["num_samples"])
    if top:
        rows = rows[:top]
    columns = ["num_samples", "n_responses"] + (["parseable", "SuccessRate"] if k else [])
    _print_table(rows, columns)


def show_results_json(data, path):
    results = data.get("results", {})
    print(f"parse_outputs results file: {path}")
    for task, block in results.items():
        print(f"\ntask: {task}")
        for key, val in block.items():
            if key == "alias":
                continue
            print(f"  {key:<22} {val}")
    print("\n(values are strings when the eval harness wrote them; 'N/A' marks metrics that do not apply to this task type)")


def show_directory(path, top):
    print(f"directory: {path}")
    entries = sorted(os.listdir(path))
    jsonl = [e for e in entries if e.endswith(".jsonl")]
    summaries = [e for e in entries if e.startswith("summary_")]
    results = [e for e in entries if e.endswith("_results.json")]
    subdirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    print(f"  jsonl files: {len(jsonl)}   results json: {len(results)}   summary files: {len(summaries)}   subdirs: {subdirs}")
    if jsonl:
        print(f"\n  {'jsonl file':<70} {'records':>8} {'dup_ids':>8} {'success':>8} {'scored':>7}")
        for name in jsonl[: top or None]:
            n = dups = succ = 0
            scored = False
            ids = Counter()
            with open(os.path.join(path, name)) as f:
                for line in f:
                    if not line.strip():
                        continue
                    n += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ids[rec.get("doc_id")] += 1
                    sr = rec.get("SuccessRate")
                    if isinstance(sr, dict):
                        scored = True
                        succ += int(bool(sr.get("success")))
            dups = sum(c - 1 for c in ids.values() if c > 1)
            print(f"  {name[:70]:<70} {n:>8} {dups:>8} {(succ if scored else '-'):>8} {str(scored):>7}")
        print("  dup_ids > 0 means the same doc_id appears more than once (see remove_duplicate_samples); 'scored' = records carry per-sample metrics.")
    for name in summaries:
        full = os.path.join(path, name)
        if name.endswith(".json") and "metrics" in name and "anatomy_vs_lesion" not in name and "all_models" not in name and "judge" not in name and "CDA" not in name:
            print()
            with open(full) as f:
                show_metrics(json.load(f), full, "num_samples", top)
        else:
            print(f"\n  summary file present: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only viewer for MedVision parsed/summary outputs.")
    parser.add_argument("--path", required=True, help="Summary JSON, parse_outputs *_results.json, or a directory (parsed/, llm-parsed_<judge>/, model dir).")
    parser.add_argument("--sort-by", default=None, help="Metric key to sort groups by, descending (e.g. num_samples, IoU, avgMRE).")
    parser.add_argument("--top", type=int, default=None, help="Show at most N rows/files.")
    parser.add_argument("--json", action="store_true", help="Also dump the loaded JSON (files only).")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"ERROR: path does not exist: {path}")
        return 2
    if os.path.isdir(path):
        show_directory(path, args.top)
        return 0
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"ERROR: not a JSON file ({exc}). For .jsonl records point --path at the directory instead.")
            return 2
    base = os.path.basename(path)
    if base.endswith("_results.json") and "results" in data:
        show_results_json(data, path)
    elif "anatomy_vs_lesion" in base:
        show_anatomy_vs_lesion(data, path)
    elif "all_models" in base:
        show_all_models(data, path, args.sort_by, args.top)
    elif base.startswith("summary_values"):
        show_values(data, path, args.top)
    elif base.startswith("summary_metrics"):
        show_metrics(data, path, args.sort_by, args.top)
    else:
        print(f"unrecognised summary file name; keys: {list(data)[:10]}")
    if args.json:
        print(json.dumps(data, indent=2)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
