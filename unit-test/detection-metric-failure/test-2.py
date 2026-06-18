print("=== Detection IoU aggregation: parse_outputs vs summarize agree ===")
print("Objective : Replicate the EXACT IoU-aggregation loops of parse_outputs.py")
print("            (per-file results.json) and summarize_detection_task.py on one")
print("            sample set that mixes valid and failed predictions, and prove the")
print("            two now report the SAME average IoU. Also reproduce the pre-fix")
print("            behavior (NaN-on-failure) to show it inflated parse_outputs by")
print("            exactly 1 / SuccessRate.")

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

import numpy as np

from medvision_bm.utils.parse_utils import cal_metrics, cal_metrics_detection_task

TARGET = "[10, 20, 30, 40]"
# 5 valid predictions (varying overlap) + 3 failures (malformed) = 8 samples.
RESPONSES = [
    "10,20,30,40",  # exact            -> IoU 1.0
    "11,19,31,41",  # slight offset    -> IoU < 1
    "12,22,32,42",  # larger offset
    "9,18,29,39",   # offset other way
    "10,20,30,41",  # one-coord off
    "10,20,30",     # FAILURE: 3 numbers
    "abc",          # FAILURE: non-numeric
    "",             # FAILURE: empty
]
samples = [{"filtered_resps": [r], "target": TARGET} for r in RESPONSES]
bar = "-" * 78


def parse_outputs_iou(samples, cal_fn):
    """Replicates parse_outputs.py:289-292 + 144-149 (skip when isnan)."""
    sum_iou, count_valid = 0.0, 0
    for s in samples:
        iou = cal_fn(s, "Detection")["avgIoU"]["IoU"]
        if not np.isnan(iou):  # parse_outputs guard: failures (NaN) excluded
            sum_iou += iou
            count_valid += 1
    return sum_iou / count_valid if count_valid > 0 else float("nan")


def summarize_iou(samples):
    """Replicates summarize_detection_task.py:102-106 + 150-153 (keep when finite)."""
    sum_iou, count_valid = 0.0, 0
    for s in samples:
        m = cal_metrics_detection_task(s)
        iou = m["avgIoU"]["IoU"]
        if np.isfinite(iou):  # 0 is finite -> failures counted as 0
            sum_iou += iou
            count_valid += 1
    return sum_iou / count_valid if count_valid > 0 else float("nan")


def old_cal_metrics(results, task_type):
    """Pre-fix cal_metrics: NaN for overlap metrics on detection failure."""
    m = cal_metrics_detection_task(results)
    if not m["SuccessRate"]["success"]:
        for k in ("avgIoU", "F1", "Precision", "Recall"):
            inner = next(iter(m[k]))
            m[k][inner] = np.nan
    return m


n_total = len(samples)
n_fail = sum(1 for s in samples if not cal_metrics_detection_task(s)["SuccessRate"]["success"])
sr = (n_total - n_fail) / n_total

authoritative = summarize_iou(samples)
fixed = parse_outputs_iou(samples, cal_metrics)        # post-fix (real code)
buggy = parse_outputs_iou(samples, old_cal_metrics)    # pre-fix (simulated)

print(f"\n{bar}")
print(f"samples={n_total}  failures={n_fail}  SuccessRate={sr:.3f}")
print(bar)
print(f"  summarize_detection_task IoU (authoritative) : {authoritative:.6f}")
print(f"  parse_outputs IoU  POST-FIX (real cal_metrics): {fixed:.6f}")
print(f"  parse_outputs IoU  PRE-FIX  (NaN on failure)  : {buggy:.6f}")
print(f"  pre-fix inflation factor vs authoritative     : {buggy / authoritative:.4f}x")
print(f"  expected inflation (1 / SuccessRate)          : {1 / sr:.4f}x")

print(f"\n{bar}\nASSERTIONS\n{bar}")

assert math.isclose(fixed, authoritative, rel_tol=1e-9, abs_tol=1e-12), (
    f"FAIL: post-fix parse_outputs IoU {fixed} != authoritative {authoritative}"
)
print("  post-fix parse_outputs IoU == summarize IoU  PASS")

assert buggy > authoritative + 1e-6, (
    "FAIL: pre-fix behavior should have inflated IoU above authoritative"
)
print("  pre-fix behavior was inflated (bug reproduced)  PASS")

assert math.isclose(buggy / authoritative, 1 / sr, rel_tol=1e-9), (
    "FAIL: pre-fix inflation should equal exactly 1 / SuccessRate"
)
print("  pre-fix inflation == 1 / SuccessRate (root cause)  PASS")

print("\nOK")
