print("=== Detection overlap-metric failure semantics ===")
print("Objective : On a FAILED detection prediction, the generic cal_metrics() must")
print("            count overlap metrics (IoU/F1/Precision/Recall) as 0 -- NOT NaN --")
print("            so parse_outputs averages them over the total sample count, exactly")
print("            like the authoritative cal_metrics_detection_task() used by")
print("            summarize_detection_task. Root-cause fix: cal_metrics delegates the")
print("            Detection branch to cal_metrics_detection_task (single source of")
print("            truth), so the two can never disagree again.")
print("Guard     : TL/AD behavior (MAE/MRE NaN on failure) must be unchanged.")

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

import numpy as np

from medvision_bm.utils.parse_utils import cal_metrics, cal_metrics_detection_task


def _num_eq(a, b):
    """Float equality that treats NaN == NaN (and tolerates fp noise)."""
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)


def _metrics_equal(d1, d2):
    """Deep, NaN-aware comparison of two metric dicts."""
    if d1.keys() != d2.keys():
        return False
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict):
            if not _metrics_equal(v1, v2):
                return False
        elif isinstance(v1, bool) or isinstance(v2, bool):
            if bool(v1) != bool(v2):
                return False
        else:
            if not _num_eq(v1, v2):
                return False
    return True


TARGET_DET = "[10, 20, 30, 40]"
bar = "-" * 78

# --------------------------------------------------------------------------
# 1. Detection failure via WRONG LENGTH (3 numbers parsed, expected 4)
# --------------------------------------------------------------------------
print(f"\n{bar}\n1. Detection failure -- wrong length ('10,20,30')\n{bar}")
r = {"filtered_resps": ["10,20,30"], "target": TARGET_DET}
m = cal_metrics(r, "Detection")
print(f"   cal_metrics -> IoU={m['avgIoU']['IoU']}, F1={m['F1']['F1']}, "
      f"P={m['Precision']['Precision']}, R={m['Recall']['Recall']}, "
      f"MAE={m['avgMAE']['MAE']}, success={m['SuccessRate']['success']}")
assert m["avgIoU"]["IoU"] == 0, "FAIL: IoU must be 0 on detection failure, not NaN"
assert m["F1"]["F1"] == 0, "FAIL: F1 must be 0 on detection failure, not NaN"
assert m["Precision"]["Precision"] == 0, "FAIL: Precision must be 0 on failure"
assert m["Recall"]["Recall"] == 0, "FAIL: Recall must be 0 on failure"
assert math.isnan(m["avgMAE"]["MAE"]), "FAIL: MAE must stay NaN (excluded from MAE avg)"
assert m["SuccessRate"]["success"] is False, "FAIL: failure must report success=False"
assert _metrics_equal(m, cal_metrics_detection_task(r)), (
    "FAIL: cal_metrics(Detection) must equal cal_metrics_detection_task (single source)"
)
print("   overlap=0, MAE=NaN, equals cal_metrics_detection_task  PASS")

# --------------------------------------------------------------------------
# 2. Detection failure via EXCEPTION (non-numeric token)
# --------------------------------------------------------------------------
print(f"\n{bar}\n2. Detection failure -- parse exception ('abc')\n{bar}")
r = {"filtered_resps": ["abc"], "target": TARGET_DET}
m = cal_metrics(r, "Detection")
print(f"   cal_metrics -> IoU={m['avgIoU']['IoU']}, F1={m['F1']['F1']}, "
      f"P={m['Precision']['Precision']}, R={m['Recall']['Recall']}")
assert m["avgIoU"]["IoU"] == 0, "FAIL: IoU must be 0 on parse exception, not NaN"
assert m["F1"]["F1"] == 0, "FAIL: F1 must be 0 on parse exception"
assert m["Precision"]["Precision"] == 0 and m["Recall"]["Recall"] == 0
assert _metrics_equal(m, cal_metrics_detection_task(r)), (
    "FAIL: exception path must equal cal_metrics_detection_task"
)
print("   overlap=0, equals cal_metrics_detection_task  PASS")

# --------------------------------------------------------------------------
# 3. Detection VALID (regression): both implementations already agree
# --------------------------------------------------------------------------
print(f"\n{bar}\n3. Detection valid ('11,19,31,41') -- single-source regression\n{bar}")
r = {"filtered_resps": ["11,19,31,41"], "target": TARGET_DET}
m = cal_metrics(r, "Detection")
print(f"   cal_metrics -> IoU={m['avgIoU']['IoU']:.4f}, success={m['SuccessRate']['success']}")
assert m["SuccessRate"]["success"] is True, "FAIL: valid prediction must report success"
assert 0 < m["avgIoU"]["IoU"] <= 1, "FAIL: valid IoU must be in (0, 1]"
assert _metrics_equal(m, cal_metrics_detection_task(r)), (
    "FAIL: valid Detection metrics must match cal_metrics_detection_task"
)
print("   real IoU, equals cal_metrics_detection_task  PASS")

# --------------------------------------------------------------------------
# 4. TL / AD regression: failure -> MAE/MRE NaN, no IoU key; valid -> finite MRE
# --------------------------------------------------------------------------
print(f"\n{bar}\n4. TL/AD unchanged (failure NaN, valid finite MRE)\n{bar}")
# TL failure (1 number parsed, expected 2)
tl_fail = cal_metrics({"filtered_resps": ["63.5"], "target": "[63.0, 36.0]"}, "TL")
assert "avgIoU" not in tl_fail, "FAIL: TL must not emit an avgIoU key"
assert math.isnan(tl_fail["avgMRE"]["MRE"]), "FAIL: TL MRE must be NaN on failure"
assert math.isnan(tl_fail["avgMAE"]["MAE"]), "FAIL: TL MAE must be NaN on failure"
assert tl_fail["SuccessRate"]["success"] is False
# TL valid
tl_ok = cal_metrics({"filtered_resps": ["63.0,36.0"], "target": "[63.0, 36.0]"}, "TL")
assert tl_ok["SuccessRate"]["success"] is True
assert _num_eq(tl_ok["avgMRE"]["MRE"], 0.0), "FAIL: exact TL match must give MRE 0"
# AD valid
ad_ok = cal_metrics({"filtered_resps": ["45.0"], "target": "[45.0]"}, "AD")
assert ad_ok["SuccessRate"]["success"] is True
assert _num_eq(ad_ok["avgMRE"]["MRE"], 0.0), "FAIL: exact AD match must give MRE 0"
print("   TL/AD failure NaN preserved, valid MRE finite, no IoU key  PASS")

print("\nOK")
