print("=== Acc@IoU localization-accuracy aggregation ===")
print("Objective : summarize_detection_task must report Acc@IoU = fraction of samples")
print("            with IoU >= tau over the COCO grid {0.50..0.95}, plus the swept mean")
print("            Acc@IoU[0.50:0.95]. Denominator is the TOTAL sample count, so a failed")
print("            parse (IoU=0) is a miss at every threshold. Acc@IoU>=0.50 must equal the")
print("            pre-existing IoU>0.5 aggregate (same quantity, finer-grid rename).")

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

from medvision_bm.benchmark.summarize_detection_task import (
    ACC_IOU_MEAN_KEY,
    COCO_IOU_THRESHOLDS,
    _calculate_final_metrics_detection_task,
    _initialize_metric_counters_detection_task,
    _update_metric_counters_detection_task,
    acc_iou_key,
)
from medvision_bm.utils.parse_utils import cal_metrics_detection_task


def _mk(iou, success):
    """Build a per-sample metrics dict shaped like cal_metrics_detection_task output.

    Only avgIoU.IoU and SuccessRate drive Acc@IoU; the other finite fields are filler.
    """
    return {
        "avgMAE": {"MAE": 0.05 if success else float("nan"), "success": success},
        "avgIoU": {"IoU": iou},
        "F1": {"F1": iou},
        "Precision": {"Precision": iou},
        "Recall": {"Recall": iou},
        "SuccessRate": {"success": success},
    }


# Worked example from the plan: 5 samples, one failed parse (IoU=0).
samples = [(0.0, False), (0.42, True), (0.55, True), (0.78, True), (0.93, True)]
counters = _initialize_metric_counters_detection_task()
for iou, success in samples:
    _update_metric_counters_detection_task(_mk(iou, success), counters)
task_metrics = _calculate_final_metrics_detection_task(counters, len(samples))


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


# 1. Headline values match the hand-computed worked example.
assert _close(task_metrics[acc_iou_key(0.50)], 0.60), task_metrics[acc_iou_key(0.50)]
assert _close(task_metrics[acc_iou_key(0.75)], 0.40), task_metrics[acc_iou_key(0.75)]
assert _close(task_metrics[ACC_IOU_MEAN_KEY], 0.34), task_metrics[ACC_IOU_MEAN_KEY]
print("[pass] Acc@IoU>=0.50=0.60, Acc@IoU>=0.75=0.40, swept-mean=0.34")

# 2. Swept mean equals the mean of the 10 grid points (internal consistency).
grid_vals = [task_metrics[acc_iou_key(t)] for t in COCO_IOU_THRESHOLDS]
assert _close(task_metrics[ACC_IOU_MEAN_KEY], sum(grid_vals) / len(grid_vals))
print("[pass] swept mean == mean of the 10 Acc@tau grid points")

# 3. Acc@IoU>=0.50 equals the pre-existing coarse IoU>0.5 key (same quantity).
assert _close(task_metrics[acc_iou_key(0.50)], task_metrics["IoU>0.5"]), (
    task_metrics[acc_iou_key(0.50)],
    task_metrics["IoU>0.5"],
)
print("[pass] Acc@IoU>=0.50 == IoU>0.5 (consistency with existing key)")

# 4. Monotone non-increasing in tau.
assert all(grid_vals[i] >= grid_vals[i + 1] for i in range(len(grid_vals) - 1)), grid_vals
print("[pass] Acc@tau non-increasing across the grid")

# 5. Failure-as-miss: denominator is 5 (not 4). 3 hits / 5 = 0.60; excluding the failed
#    sample would wrongly give 3/4 = 0.75.
assert _close(task_metrics[acc_iou_key(0.50)] * len(samples), 3), task_metrics[acc_iou_key(0.50)]
assert not _close(task_metrics[acc_iou_key(0.50)], 0.75)
print("[pass] failed parse counts as a miss (denominator = total sample count)")

# 6. The real metric path yields IoU=0 on an unparseable answer, so failure-as-miss holds
#    end-to-end (not just via the synthetic dict above).
failed = cal_metrics_detection_task(
    {"filtered_resps": ["no box here"], "target": "[0.1, 0.1, 0.2, 0.2]"}
)
assert failed["avgIoU"]["IoU"] == 0, failed["avgIoU"]["IoU"]
assert failed["SuccessRate"]["success"] is False
print("[pass] cal_metrics_detection_task -> IoU=0 on failed parse")

print("\n=== ALL PASSED ===")
