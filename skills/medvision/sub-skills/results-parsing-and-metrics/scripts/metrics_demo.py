#!/usr/bin/env python3
"""Demonstrate MedVision per-sample metric semantics on a tiny synthetic fixture.

Purpose
    Run the REAL scoring code of ``medvision_bm`` (``cal_metrics``,
    ``cal_metrics_detection_task``, the summarizer counters) on a handful of
    hand-written responses and print what each metric becomes, so the failure
    semantics can be seen instead of guessed:

    * a response without a parseable ``<answer>...</answer>`` block is a
      FAILURE: ``SuccessRate.success = False`` and MAE/MRE = NaN for every task;
    * for Detection, the overlap metrics of a failure are 0 (NOT NaN), so the
      summarizer's mean IoU/F1/Precision/Recall are averaged over ALL samples
      while ``avgMAE`` is averaged over successful samples only;
    * every threshold metric (``IoU>0.5``, ``Acc@IoU>=0.50``, ``MRE<0.3``, ...)
      divides by the TOTAL sample count, so failures count as misses;
    * the A/D summarizer drops samples whose ground truth is below
      ``AD_NEAR_ZERO_GT_THRESHOLD`` before counting anything.

    The script exits non-zero if the installed package no longer behaves like
    the reference text of this sub-skill (assertions at the end of each block).

Prerequisites
    ``medvision_bm`` importable (``pip install medvision-bm`` or an editable
    install of the repository). No data, GPU, or network needed.

Example
    python metrics_demo.py
    python metrics_demo.py --json > demo.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def _import_apis():
    try:
        from medvision_bm.benchmark.summarize_AD_task import process_label_group
        from medvision_bm.benchmark.summarize_detection_task import (
            ACC_IOU_MEAN_KEY,
            _calculate_final_metrics_detection_task,
            _initialize_metric_counters_detection_task,
            _update_metric_counters_detection_task,
            acc_iou_key,
        )
        from medvision_bm.benchmark.summarize_TL_task import (
            cal_metrics_TL_task,
            process_label_group_TL,
        )
        from medvision_bm.utils.configs import AD_NEAR_ZERO_GT_THRESHOLD
        from medvision_bm.utils.parse_utils import (
            cal_metrics,
            extract_last_k_nums_within_answer_tag,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        print("ERROR: could not import medvision_bm scoring code.")
        print(f"Reason: {exc.__class__.__name__}: {exc}")
        print(
            "Recovery: install the benchmark package into the active Python "
            "(`pip install medvision-bm`, or `pip install -e <repo>` from a checkout) "
            "and re-run. This demo needs no data or GPU."
        )
        return None
    return {
        "process_label_group_AD": process_label_group,
        "ACC_IOU_MEAN_KEY": ACC_IOU_MEAN_KEY,
        "acc_iou_key": acc_iou_key,
        "det_init": _initialize_metric_counters_detection_task,
        "det_update": _update_metric_counters_detection_task,
        "det_final": _calculate_final_metrics_detection_task,
        "cal_metrics_TL_task": cal_metrics_TL_task,
        "process_label_group_TL": process_label_group_TL,
        "AD_NEAR_ZERO_GT_THRESHOLD": AD_NEAR_ZERO_GT_THRESHOLD,
        "cal_metrics": cal_metrics,
        "extract": extract_last_k_nums_within_answer_tag,
    }


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _fmt(x) -> str:
    if isinstance(x, bool):
        return str(x)
    if _isnan(x):
        return "nan"
    if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer() and abs(x) >= 2):
        return str(int(x))
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


# ---------------------------------------------------------------------------
# Fixtures: raw model responses exactly as an eval JSONL would carry them.
# ---------------------------------------------------------------------------
DETECTION_TARGET = "[0.2, 0.2, 0.6, 0.6]"  # relative [x_min, y_min, x_max, y_max]
DETECTION_RESPONSES = [
    ("exact box", "<think>...</think><answer>0.2, 0.2, 0.6, 0.6</answer>"),
    ("shifted box", "<answer>0.3,0.3,0.7,0.7</answer>"),
    ("disjoint box", "<answer>0.7,0.7,0.9,0.9</answer>"),
    ("FAILURE: 3 numbers", "<answer>0.2, 0.2, 0.6</answer>"),
    ("FAILURE: no answer tag", "The box is 0.2, 0.2, 0.6, 0.6"),
]

TL_TARGET = "[12.4, 8.3]"  # mm: [major axis, minor axis]
TL_RESPONSES = [
    ("two sizes with units", "<answer>12.0 mm x 8.0 mm</answer>"),
    ("FAILURE: prose, no tag", "The tumor measures about 12 by 8 mm."),
]

AD_TARGETS_RESPONSES = [
    ("distance 27.5 mm", "27.5", "<answer>30</answer>"),
    ("distance 27.5 mm, FAILURE", "27.5", "I cannot determine it."),
    ("near-zero GT 0.05 (dropped)", "0.05", "<answer>0.04</answer>"),
]


def run(api) -> dict:
    out = {"detection": {}, "tumor_lesion": {}, "angle_distance": {}}
    cal_metrics = api["cal_metrics"]
    extract = api["extract"]

    # ------------------------------------------------------------------ Detection
    print("=" * 78)
    print("DETECTION  (k = 4 numbers inside <answer>; target = relative box)")
    print("=" * 78)
    counters = api["det_init"]()
    per_sample = []
    sum_iou_all, sum_mae_ok, n_ok = 0.0, 0.0, 0
    for name, raw in DETECTION_RESPONSES:
        filtered = extract(raw, 4)
        m = cal_metrics({"filtered_resps": [filtered], "target": DETECTION_TARGET}, "Detection")
        api["det_update"](m, counters)
        row = {
            "case": name,
            "filtered_resps": filtered,
            "success": bool(m["SuccessRate"]["success"]),
            "MAE": m["avgMAE"]["MAE"],
            "IoU": m["avgIoU"]["IoU"],
            "F1": m["F1"]["F1"],
            "Precision": m["Precision"]["Precision"],
            "Recall": m["Recall"]["Recall"],
        }
        per_sample.append(row)
        sum_iou_all += float(m["avgIoU"]["IoU"])
        if row["success"]:
            sum_mae_ok += float(m["avgMAE"]["MAE"])
            n_ok += 1
        print(
            f"  {name:<26} filtered={filtered!r:<24} success={row['success']!s:<5} "
            f"MAE={_fmt(row['MAE'])} IoU={_fmt(row['IoU'])} F1={_fmt(row['F1'])} "
            f"P={_fmt(row['Precision'])} R={_fmt(row['Recall'])}"
        )
        if not row["success"]:
            assert _isnan(row["MAE"]), "failure must give MAE = NaN"
            for k in ("IoU", "F1", "Precision", "Recall"):
                assert row[k] == 0, f"failure must give {k} = 0, got {row[k]!r}"

    n_total = len(DETECTION_RESPONSES)
    agg = api["det_final"](counters, n_total)
    acc50 = api["acc_iou_key"](0.50)
    print("\n  summarizer aggregate over the group (real counters):")
    for k in ("num_samples", "SuccessRate", "avgMAE", "IoU", "F1", "Precision", "Recall", "IoU>0.5", acc50, api["ACC_IOU_MEAN_KEY"]):
        print(f"    {k:<20} = {_fmt(agg[k])}")
    naive = sum_iou_all / n_ok if n_ok else float("nan")
    print(f"\n  mean IoU over ALL {n_total} samples (failures = 0) : {sum_iou_all / n_total:.4f}  <- what the summary reports")
    print(f"  mean IoU over the {n_ok} successes only            : {naive:.4f}  <- NOT what is reported (= reported / SuccessRate)")
    print(f"  IoU>0.5 = samples with IoU >= 0.5 / {n_total} total  : {agg['IoU>0.5']:.4f}")

    assert agg["num_samples"] == n_total
    assert _close(agg["SuccessRate"], n_ok / n_total), "SuccessRate = successes / total"
    assert _close(agg["IoU"], sum_iou_all / n_total), "mean IoU must divide by ALL samples"
    assert _close(agg["avgMAE"], sum_mae_ok / n_ok), "avgMAE must divide by successes only"
    assert _close(agg["IoU>0.5"], 1 / n_total), "IoU>0.5 = count(IoU>=0.5)/total"
    assert _close(agg[acc50], agg["IoU>0.5"]), "Acc@IoU>=0.50 must equal IoU>0.5"
    assert _close(naive, agg["IoU"] / agg["SuccessRate"]), "success-only mean = reported / SR"
    out["detection"] = {"per_sample": per_sample, "aggregate": agg}

    # ----------------------------------------------------------------- T/L
    print("\n" + "=" * 78)
    print("TUMOR/LESION SIZE  (k = 2 numbers; target = [major, minor] in mm)")
    print("=" * 78)
    tl_rows, targets, responses = [], [], []
    for name, raw in TL_RESPONSES:
        filtered = extract(raw, 2)
        m = api["cal_metrics_TL_task"]({"filtered_resps": [filtered], "target": TL_TARGET, "doc_meta": None})
        row = {
            "case": name,
            "filtered_resps": filtered,
            "success": bool(m["SuccessRate"]["success"]),
            "MAE": m["avgMAE"]["MAE"],
            "MRE": m["avgMRE"]["MRE"],
            "nMAE": m["nMAE"]["NMAE"],
            "nMAE_success": bool(m["nMAE"]["success"]),
        }
        tl_rows.append(row)
        targets.append(TL_TARGET)
        responses.append(filtered)
        print(
            f"  {name:<26} filtered={filtered!r:<14} success={row['success']!s:<5} "
            f"MAE={_fmt(row['MAE'])} MRE={_fmt(row['MRE'])} nMAE={_fmt(row['nMAE'])} (nMAE needs NIfTI header -> {row['nMAE_success']})"
        )
        if not row["success"]:
            assert _isnan(row["MAE"]) and _isnan(row["MRE"]), "T/L failure must give NaN MAE/MRE"
    ok = tl_rows[0]
    assert ok["success"] and _close(ok["MAE"], (0.4 + 0.3) / 2, 1e-4), "MAE = mean |pred - gt| over the 2 axes"
    assert _close(ok["MRE"], (0.4 / 12.4 + 0.3 / 8.3) / 2, 1e-4), "MRE = mean(|pred-gt| / gt)"
    assert not ok["nMAE_success"], "without doc_meta there is no diagonal -> nMAE NaN/False"
    _, tl_agg = api["process_label_group_TL"]("demo group", {"targets": targets, "responses": responses})
    print("\n  summarizer aggregate over the group:")
    for k in ("num_samples", "SuccessRate", "avgMAE", "avgMRE", "avgNMAE", "MRE<0.1", "MRE<0.3", "MRE<1.0"):
        print(f"    {k:<12} = {_fmt(tl_agg[k])}")
    print("  note: MRE<1.0 equals SuccessRate because the last bucket is [0.9, inf) -> every parsed sample")
    assert tl_agg["num_samples"] == 2 and _close(tl_agg["SuccessRate"], 0.5)
    assert _close(tl_agg["avgMAE"], ok["MAE"], 1e-6), "avgMAE averages successes only"
    assert _close(tl_agg["MRE<1.0"], tl_agg["SuccessRate"]), "MRE<1.0 == SuccessRate"
    assert _close(tl_agg["MRE<0.1"], 1 / 2), "MRE<0.1: 1 of 2 TOTAL samples"
    out["tumor_lesion"] = {"per_sample": tl_rows, "aggregate": tl_agg}

    # ----------------------------------------------------------------- A/D
    print("\n" + "=" * 78)
    print(f"ANGLE/DISTANCE  (k = 1 number; GT < {api['AD_NEAR_ZERO_GT_THRESHOLD']} dropped by the summarizer)")
    print("=" * 78)
    ad_rows, targets, responses = [], [], []
    for name, target, raw in AD_TARGETS_RESPONSES:
        filtered = extract(raw, 1)
        m = cal_metrics({"filtered_resps": [filtered], "target": target}, "AD")
        row = {
            "case": name,
            "target": target,
            "filtered_resps": filtered,
            "success": bool(m["SuccessRate"]["success"]),
            "MAE": m["avgMAE"]["MAE"],
            "MRE": m["avgMRE"]["MRE"],
        }
        ad_rows.append(row)
        targets.append(target)
        responses.append(filtered)
        print(
            f"  {name:<28} filtered={filtered!r:<8} success={row['success']!s:<5} "
            f"MAE={_fmt(row['MAE'])} MRE={_fmt(row['MRE'])}"
        )
    assert _close(ad_rows[0]["MAE"], 2.5, 1e-4) and _close(ad_rows[0]["MRE"], 2.5 / 27.5, 1e-4)
    assert not ad_rows[1]["success"] and _isnan(ad_rows[1]["MAE"])
    assert ad_rows[2]["success"], "cal_metrics itself does NOT apply the near-zero rule"
    _, ad_agg = api["process_label_group_AD"]("demo label", {"targets": targets, "responses": responses})
    print("\n  summarizer aggregate over the label group:")
    for k in ("num_samples", "SuccessRate", "avgMAE", "avgMRE", "avgNMAE", "MRE<0.1", "MRE<1.0"):
        print(f"    {k:<12} = {_fmt(ad_agg[k])}")
    print("  note: 3 inputs, num_samples = 2 -> the near-zero-GT sample was dropped before counting")
    assert ad_agg["num_samples"] == 2, "near-zero GT sample must be excluded from the group"
    assert _close(ad_agg["SuccessRate"], 0.5) and _close(ad_agg["avgMAE"], 2.5, 1e-4)
    out["angle_distance"] = {"per_sample": ad_rows, "aggregate": ad_agg}
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "item"):
        try:
            v = obj.item()
            return None if isinstance(v, float) and math.isnan(v) else v
        except Exception:
            return str(obj)
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show MedVision metric semantics (NaN-vs-0 failures, total-count denominators) on a synthetic fixture."
    )
    parser.add_argument("--json", action="store_true", help="Print the per-sample and aggregate dicts as JSON after the report (NaN -> null).")
    args = parser.parse_args()

    api = _import_apis()
    if api is None:
        return 2
    try:
        result = run(api)
    except AssertionError as exc:
        print(f"\nFAIL: metric semantics differ from the reference: {exc}")
        return 1
    print("\nALL CHECKS PASSED: failure semantics and denominators match the reference text.")
    if args.json:
        print(json.dumps(_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
