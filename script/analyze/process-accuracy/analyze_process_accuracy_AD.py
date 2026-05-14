"""
Analyze intermediate-step accuracy for A/D (Angle & Distance) task model responses.

GT landmark coordinates are loaded directly from each sample's landmark JSON file
(referenced via doc["landmark_file"]), with benchmark_plan used to resolve which
landmark point names correspond to each metric_key.  This approach is independent
of the parquet dataset and works for any model regardless of image size or prompt format.

For each sample in the input JSONL file(s):
  - Distance tasks (3 steps):
      Step 1: landmark 1 relative coords  → MAE vs GT
      Step 2: landmark 2 relative coords  → MAE vs GT
      Step 3: computed distance           → MRE vs GT
  - Angle tasks (3 steps):
      Step 1: line 1 endpoints (2 coords) → MAE vs GT
      Step 2: line 2 endpoints (2 coords) → MAE vs GT
      Step 3: computed angle              → MRE vs GT

Usage:
    python analyze_process_accuracy_AD.py \
        --task_dir /path/to/MedVision-AD-v2-CoT \
        [--jsonl /path/to/explicit.jsonl ...] \
        [--output_suffix _proc_acc]
"""

import argparse
import gzip
import importlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from medvision_bm.utils.configs import AD_NEAR_ZERO_GT_THRESHOLD

def _cal_point_dist(pred_xy, gt_xy):
    return float(np.sqrt((pred_xy[0] - gt_xy[0])**2 + (pred_xy[1] - gt_xy[1])**2))


def _cal_nMAE(pred, gt, scale):
    return float(np.mean(np.abs(np.array(pred, float) - np.array(gt, float))/(scale + 1e-15)))


def _cal_MRE(pred, gt):
    gt = np.array(gt, float)
    return float(np.mean(np.abs(np.array(pred, float) - gt) / (gt + 1e-15)))


# ---------------------------------------------------------------------------
# Landmark file helpers (mirrors sft_utils.py logic)
# ---------------------------------------------------------------------------

def _load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_3d_to_2d(coor_3d, slice_dim):
    """Extract 2D [row, col] array-space coordinates from a 3D coordinate list."""
    if slice_dim == 0:
        return coor_3d[1:3]
    elif slice_dim == 1:
        return [coor_3d[0], coor_3d[2]]
    elif slice_dim == 2:
        return coor_3d[0:2]
    raise ValueError(f"slice_dim must be 0/1/2, got {slice_dim}")


def _load_slice_landmarks(doc, landmark_keys):
    """
    Load array-space [row, col] coordinates for the requested landmark_keys
    from the landmark JSON file described in `doc`.

    Returns dict {key: [row, col]} or None on failure.
    """
    lm_data   = _load_json(doc["landmark_file"])
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    dim_to_key = {0: "slice_landmarks_x", 1: "slice_landmarks_y", 2: "slice_landmarks_z"}
    lm_slice_ls = lm_data[dim_to_key[slice_dim]]

    matched = [e for e in lm_slice_ls if e.get("slice_idx") == slice_idx]
    if not matched:
        return None

    # Merge all matched entries (handles multi-lesion cases by taking the first element)
    landmarks = {}
    for entry in matched:
        entry_lms = entry.get("landmarks", {})
        if isinstance(entry_lms, list):
            entry_lms = entry_lms[0]
        landmarks.update(entry_lms)

    result = {}
    for key in landmark_keys:
        coor_3d = landmarks.get(key)
        if coor_3d is None:
            return None
        result[key] = _extract_3d_to_2d(coor_3d, slice_dim)
    return result


def _to_wh(array_coor_2d, img_h, img_w):
    """
    Convert array coords to normalized (w, h) image coords.
    
    # ------------------------------------------------------------------
    # NOTE: CAVEAT!
    # !!! We need to convert the coordinates from the benchmark planner format to the output format. !!!
    #
    #              #---------------+   --
    #              |   * (P1)      |    |
    #              |               |    | -> image_size_height
    #              |               |    |
    #              &---------------+   --
    #
    # #: array space origin (upper-left corner)
    # &: image space origin (lower-left corner)
    # The point * can be written in array space as P1 and in image space as P1':
    #   - P1: (idx_dim0, idx_dim1)
    #   - P1': (x_1, y_1) = (idx_dim1, image_size_height - idx_dim0)
    # --------------------------------------
    """
    coor_dim0_h, coor_dim1_w = array_coor_2d
    return [coor_dim1_w / img_w, 1.0 - coor_dim0_h / img_h]


# ---------------------------------------------------------------------------
# Benchmark plan cache (avoid re-loading per sample)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _get_benchmark_plan(dataset_name):
    from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASETS_NAME2PACKAGE.")
    bm_module = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_biometry"
    )
    return bm_module.benchmark_plan


# ---------------------------------------------------------------------------
# GT extraction from doc
# ---------------------------------------------------------------------------

def _extract_gt_from_doc(doc):
    """
    Return GT landmark wh-coords and measurement value for a JSONL doc.

    Distance → {metric_type, landmark_1_wh, landmark_2_wh, gt_value}
    Angle    → {metric_type, line_1_point_1_wh, line_1_point_2_wh,
                             line_2_point_1_wh, line_2_point_2_wh, gt_value}
    Returns None on failure.
    """
    bp          = doc["biometric_profile"]
    metric_type = bp["metric_type"]
    metric_key  = bp["metric_key"]
    gt_value    = float(bp["metric_value"])

    img_h, img_w = doc["image_size_2d"]

    try:
        bm_plan   = _get_benchmark_plan(doc["dataset_name"])
        task_info = bm_plan["tasks"][int(doc["taskID"]) - 1]
    except Exception as e:
        return None, f"benchmark_plan error: {e}"

    try:
        if metric_type == "distance":
            line_dict  = task_info[bp["metric_map_name"]][metric_key]
            lm_keys    = line_dict["element_keys"]  # e.g. ["P1", "P2"]
            lm_coords  = _load_slice_landmarks(doc, lm_keys)
            if lm_coords is None:
                return None, "landmark_not_found"
            return {
                "metric_type":    "distance",
                "landmark_1_wh":  _to_wh(lm_coords[lm_keys[0]], img_h, img_w),
                "landmark_2_wh":  _to_wh(lm_coords[lm_keys[1]], img_h, img_w),
                "gt_value":       gt_value,
            }, None

        elif metric_type == "angle":
            angles_map  = task_info[bp["metric_map_name"]]
            angle_dict  = angles_map[metric_key]
            lines_map   = task_info[angle_dict["element_map_name"]]
            line1_dict  = lines_map[angle_dict["element_keys"][0]]
            line2_dict  = lines_map[angle_dict["element_keys"][1]]
            l1_keys     = line1_dict["element_keys"]
            l2_keys     = line2_dict["element_keys"]
            all_keys    = list(dict.fromkeys(l1_keys + l2_keys))  # deduplicated
            lm_coords   = _load_slice_landmarks(doc, all_keys)
            if lm_coords is None:
                return None, "landmark_not_found"
            return {
                "metric_type":         "angle",
                "line_1_point_1_wh":   _to_wh(lm_coords[l1_keys[0]], img_h, img_w),
                "line_1_point_2_wh":   _to_wh(lm_coords[l1_keys[1]], img_h, img_w),
                "line_2_point_1_wh":   _to_wh(lm_coords[l2_keys[0]], img_h, img_w),
                "line_2_point_2_wh":   _to_wh(lm_coords[l2_keys[1]], img_h, img_w),
                "gt_value":            gt_value,
            }, None
        else:
            return None, f"unknown metric_type: {metric_type}"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Regex patterns (mirror medvision_ad.py)
# ---------------------------------------------------------------------------

_RP  = r"(?:0(?:\.\d+)?|1(?:\.0+)?)"
_RPG = rf"({_RP})"
_NNR  = r"\d+(?:\.\d+)?"
_NNRG = rf"({_NNR})"
_CG   = rf"\(\s*{_RPG}\s*,\s*{_RPG}\s*\)"


def _tag(n): return rf"<{n}>"
def _end(n): return rf"</{n}>"
def _rea(k): return rf"{_tag(f'step-{k}-reasoning')}.*?{_end(f'step-{k}-reasoning')}"


PATTERNS_DIST_GROUP = {
    1: rf"{_rea(1)}\s*.*?{_tag('step-1-answer')}.*?{_CG}.*?{_end('step-1-answer')}.*?",
    2: rf"{_rea(2)}\s*.*?{_tag('step-2-answer')}.*?{_CG}.*?{_end('step-2-answer')}.*?",
    3: rf"{_rea(3)}\s*.*?{_tag('step-3-answer')}.*?{_NNRG}.*?{_end('step-3-answer')}.*?",
}
PATTERNS_DIST_ANSWER_ONLY = {
    1: rf".*?{_tag('step-1-answer')}.*?{_CG}.*?{_end('step-1-answer')}.*?",
    2: rf".*?{_tag('step-2-answer')}.*?{_CG}.*?{_end('step-2-answer')}.*?",
    3: rf".*?{_tag('step-3-answer')}.*?{_NNRG}.*?{_end('step-3-answer')}.*?",
}

PATTERNS_ANGLE_GROUP = {
    1: rf"{_rea(1)}\s*.*?{_tag('step-1-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-1-answer')}.*?",
    2: rf"{_rea(2)}\s*.*?{_tag('step-2-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-2-answer')}.*?",
    3: rf"{_rea(3)}\s*.*?{_tag('step-3-answer')}.*?{_NNRG}.*?{_end('step-3-answer')}.*?",
}
PATTERNS_ANGLE_ANSWER_ONLY = {
    1: rf".*?{_tag('step-1-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-1-answer')}.*?",
    2: rf".*?{_tag('step-2-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-2-answer')}.*?",
    3: rf".*?{_tag('step-3-answer')}.*?{_NNRG}.*?{_end('step-3-answer')}.*?",
}

FLAGS = re.DOTALL

# Tool-use model outputs the step-3 result in <answer>...</answer> instead of <step-3-answer>
PATTERN_TOOLUSE_ANSWER = rf".*?<answer>\s*({_NNR})\s*</answer>.*?"


def _search(pat, txt):
    return re.search(pat, txt, FLAGS)


# ---------------------------------------------------------------------------
# Per-task analyzers
# ---------------------------------------------------------------------------

def analyze_distance_sample(solution, gt, scale):
    result = {
        "metric_type":    "distance",
        "gt_landmark_1_wh": gt["landmark_1_wh"],
        "gt_landmark_2_wh": gt["landmark_2_wh"],
        "gt_distance":    gt["gt_value"],
    }
    gp1 = gt["landmark_1_wh"]
    gp2 = gt["landmark_2_wh"]
    gv  = gt["gt_value"]

    m1 = _search(PATTERNS_DIST_GROUP[1], solution) or _search(PATTERNS_DIST_ANSWER_ONLY[1], solution)
    if m1:
        p1 = [float(m1.group(1)), float(m1.group(2))]
        result["step1_pred"] = p1
        result["step1_MAE"]  = _cal_point_dist(p1, gp1)
    else:
        result["step1_pred"] = result["step1_MAE"] = None

    m2 = _search(PATTERNS_DIST_GROUP[2], solution) or _search(PATTERNS_DIST_ANSWER_ONLY[2], solution)
    if m2:
        p2 = [float(m2.group(1)), float(m2.group(2))]
        result["step2_pred"] = p2
        result["step2_MAE"]  = _cal_point_dist(p2, gp2)
    else:
        result["step2_pred"] = result["step2_MAE"] = None

    m3 = (_search(PATTERNS_DIST_GROUP[3], solution)
          or _search(PATTERNS_DIST_ANSWER_ONLY[3], solution)
          or _search(PATTERN_TOOLUSE_ANSWER, solution))
    if m3:
        pd = float(m3.group(1))
        result["step3_pred"] = pd
        result["step3_MRE"]  = _cal_MRE([pd], [gv])
        result["step3_nMAE"]  = _cal_nMAE([pd], [gv], scale)
    else:
        result["step3_pred"] = result["step3_MRE"] = None
        result["step3_nMAE"] = None

    return result


def analyze_angle_sample(solution, gt):
    gl1p1 = gt["line_1_point_1_wh"]
    gl1p2 = gt["line_1_point_2_wh"]
    gl2p1 = gt["line_2_point_1_wh"]
    gl2p2 = gt["line_2_point_2_wh"]
    gv    = gt["gt_value"]

    result = {
        "metric_type":         "angle",
        "gt_line1_point1_wh":  gl1p1,
        "gt_line1_point2_wh":  gl1p2,
        "gt_line2_point1_wh":  gl2p1,
        "gt_line2_point2_wh":  gl2p2,
        "gt_angle":            gv,
    }

    m1 = _search(PATTERNS_ANGLE_GROUP[1], solution) or _search(PATTERNS_ANGLE_ANSWER_ONLY[1], solution)
    if m1:
        p = [float(m1.group(i)) for i in range(1, 5)]
        result["step1_pred"] = p
        pred_pts = [[p[0], p[1]], [p[2], p[3]]]
        d1 = (_cal_point_dist(pred_pts[0], gl1p1) + _cal_point_dist(pred_pts[1], gl1p2)) / 2
        d2 = (_cal_point_dist(pred_pts[0], gl1p2) + _cal_point_dist(pred_pts[1], gl1p1)) / 2
        result["step1_MAE"] = min(d1, d2)
    else:
        result["step1_pred"] = result["step1_MAE"] = None

    m2 = _search(PATTERNS_ANGLE_GROUP[2], solution) or _search(PATTERNS_ANGLE_ANSWER_ONLY[2], solution)
    if m2:
        p = [float(m2.group(i)) for i in range(1, 5)]
        result["step2_pred"] = p
        pred_pts = [[p[0], p[1]], [p[2], p[3]]]
        d1 = (_cal_point_dist(pred_pts[0], gl2p1) + _cal_point_dist(pred_pts[1], gl2p2)) / 2
        d2 = (_cal_point_dist(pred_pts[0], gl2p2) + _cal_point_dist(pred_pts[1], gl2p1)) / 2
        result["step2_MAE"] = min(d1, d2)
    else:
        result["step2_pred"] = result["step2_MAE"] = None

    m3 = (_search(PATTERNS_ANGLE_GROUP[3], solution)
          or _search(PATTERNS_ANGLE_ANSWER_ONLY[3], solution)
          or _search(PATTERN_TOOLUSE_ANSWER, solution))
    if m3:
        pa = float(m3.group(1))
        result["step3_pred"] = pa
        result["step3_MRE"]  = _cal_MRE([pa], [gv])
    else:
        result["step3_pred"] = result["step3_MRE"] = None

    return result


# ---------------------------------------------------------------------------
# Process a single JSONL file
# ---------------------------------------------------------------------------

def process_jsonl(jsonl_path, output_suffix):
    jsonl_path = Path(jsonl_path)
    out_path   = jsonl_path.with_name(jsonl_path.stem + output_suffix + ".jsonl")

    n_total = n_distance = n_angle = n_gt_fail = n_parse_fail = 0
    n_dist_success = n_angle_success = 0
    results = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            n_total += 1

            doc       = sample.get("doc", {})
            doc_id    = sample.get("doc_id")
            solution  = sample.get("resps", [[""]])[0]
            if isinstance(solution, list):
                solution = solution[0] if solution else ""

            bp = doc.get("biometric_profile", {})
            record = {
                "doc_id":      doc_id,
                "dataset":     doc.get("dataset_name"),
                "task_type":   doc.get("taskType"),
                "metric_key":  bp.get("metric_key"),
                "image_file":  doc.get("image_file"),
                "slice_dim":   doc.get("slice_dim"),
                "slice_idx":   doc.get("slice_idx"),
                "image_size_2d": doc.get("image_size_2d"),
                "pixel_size": doc.get("pixel_size"),
            }

            gt, err = _extract_gt_from_doc(doc)
            if gt is None:
                n_gt_fail += 1
                record["error"] = f"gt_extraction_failed: {err}"
                results.append(record)
                continue

            record["metric_type"] = gt["metric_type"]
            try:
                if gt["metric_type"] == "distance":
                    n_distance += 1
                    # scale distance error by the image diagonal to get a more interpretable relative error metric (e.g. 0.05 means 5% of the image diagonal)
                    image_size_2d = doc.get("image_size_2d")
                    pixel_size = doc.get("pixel_size")
                    image_diagonal = np.sqrt((image_size_2d[0]*pixel_size[0])**2 + (image_size_2d[1]*pixel_size[1])**2)
                    record.update(analyze_distance_sample(solution, gt, image_diagonal))
                    if all(record.get(k) is not None for k in ("step1_MAE", "step2_MAE", "step3_MRE")):
                        n_dist_success += 1
                elif gt["metric_type"] == "angle":
                    n_angle += 1
                    record.update(analyze_angle_sample(solution, gt))
                    if all(record.get(k) is not None for k in ("step1_MAE", "step2_MAE", "step3_MRE")):
                        n_angle_success += 1
            except Exception as e:
                n_parse_fail += 1
                record["error"] = str(e)

            results.append(record)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def _stats(key, mtype):
        vals = [r[key] for r in results if r.get("metric_type") == mtype and r.get(key) is not None]
        if not vals:
            return None, None, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    success_parts = []
    if n_distance > 0:
        success_parts.append(f"success_reasoning_rate={100*n_dist_success/n_distance:.1f}% {n_dist_success}/{n_distance}(dist)")
    if n_angle > 0:
        success_parts.append(f"success_reasoning_rate={100*n_angle_success/n_angle:.1f}% {n_angle_success}/{n_angle}(angle)")
    success_str = (", " + ", ".join(success_parts)) if success_parts else ""

    print(f"\n[{jsonl_path.name}]")
    print(f"  Total: {n_total}  (distance={n_distance}, angle={n_angle}, gt_fail={n_gt_fail}, parse_fail={n_parse_fail}{success_str})")
    mtype_total = {"distance": n_distance, "angle": n_angle}
    for key, label, mtype in [
        ("step1_MAE", "Step1 MAE (landmark 1)",    "distance"),
        ("step2_MAE", "Step2 MAE (landmark 2)",    "distance"),
        ("step3_MRE", "Step3 MRE (distance)",      "distance"),
        ("step3_nMAE", "nMAE (step 3) (distance)", "distance"),
        ("step1_MAE", "Step1 MAE (line-1 endpts)", "angle"),
        ("step2_MAE", "Step2 MAE (line-2 endpts)", "angle"),
        ("step3_MRE", "Step3 MRE (angle)",         "angle"),
    ]:
        if mtype_total[mtype] == 0:
            continue
        mean, sd, n = _stats(key, mtype)
        mean_str = f"{mean:.4f}" if mean is not None else "nan"
        sd_str   = f"{sd:.4f}"   if sd   is not None else "nan"
        fail = mtype_total[mtype] - n
        print(f"  [{mtype}] {label}: mean={mean_str} ± sd={sd_str} (n={n}, fail={fail})")
    print(f"  Output: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Path discovery helpers
# ---------------------------------------------------------------------------

def _collect_from_model_dir(model_dir):
    paths = []
    parsed_dir = Path(model_dir) / "parsed"
    if parsed_dir.is_dir():
        for p in sorted(parsed_dir.glob("*.jsonl")):
            if "_proc_acc" not in p.stem:
                paths.append(p)
    return paths


def _collect_from_task_dir(task_dir):
    paths = []
    for model_dir in sorted(Path(task_dir).iterdir()):
        paths.extend(_collect_from_model_dir(model_dir))
    return paths


def _collect_from_jsonl_args(jsonl_args):
    paths = []
    for pattern in jsonl_args:
        if "*" in pattern:
            paths.extend(sorted(Path(".").glob(pattern)))
        else:
            paths.append(Path(pattern))
    return [p for p in paths if p.exists()]


# ---------------------------------------------------------------------------
# Per-model aggregation and cross-model summary
# ---------------------------------------------------------------------------

SUMMARY_PROC_ACC_AD_METRICS_FILENAME = "summary_proc_acc_AD_metrics.json"
SUMMARY_PROC_ACC_AD_MODEL_FILENAME = "summary_proc_acc_AD_model.txt"


def _get_ad_label(record):
    dataset = record.get("dataset", "")
    metric_type = record.get("metric_type", "")
    metric_key = record.get("metric_key", "")
    if not (dataset and metric_type and metric_key):
        return None
    return f"{dataset}_{metric_type}_{metric_key}"


def _aggregate_by_label_AD(all_results):
    """Aggregate per-sample results by label; return {label: averaged step metrics}."""
    grouped = {}
    for r in all_results:
        label = _get_ad_label(r)
        if label is None:
            continue
        if label not in grouped:
            grouped[label] = {
                "metric_type": r.get("metric_type"),
                "s1": [], "s2": [], "s3_mre": [], "s3_msc": [],
                "n_success": 0, "n_samples": 0, "n_valid": 0, "n_ignored": 0,
            }
        g = grouped[label]
        g["n_samples"] += 1

        if r.get("metric_type") == "distance":
            gt_scalar = r.get("gt_distance")
        else:
            gt_scalar = r.get("gt_angle")
        skip_s3 = gt_scalar is not None and gt_scalar < AD_NEAR_ZERO_GT_THRESHOLD
        if skip_s3:
            g["n_ignored"] += 1
        else:
            g["n_valid"] += 1

        s1, s2, s3_mre, s3_msc = r.get("step1_MAE"), r.get("step2_MAE"), r.get("step3_MRE"), r.get("step3_nMAE")
        if s1 is not None: g["s1"].append(s1)
        if s2 is not None: g["s2"].append(s2)
        if s3_mre is not None and not skip_s3: g["s3_mre"].append(s3_mre)
        if s3_msc is not None and not skip_s3: g["s3_msc"].append(s3_msc)
        if s1 is not None and s2 is not None and s3_mre is not None:
            g["n_success"] += 1

    def _avg(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        label: {
            "metric_type": g["metric_type"],
            "step1_avg_MAE": _avg(g["s1"]),
            "step2_avg_MAE": _avg(g["s2"]),
            "step3_avg_MRE": _avg(g["s3_mre"]),
            "step3_avg_nMAE": _avg(g["s3_msc"]),
            "n_samples": g["n_samples"],
            "n_valid":   g["n_valid"],
            "n_ignored": g["n_ignored"],
            "success_rate": g["n_success"] / g["n_samples"] if g["n_samples"] > 0 else 0.0,
        }
        for label, g in grouped.items()
    }


def _process_model_dir(model_dir, output_suffix):
    """Process all JSONL files in model's parsed/ dir; save per-label summary JSON."""
    model_dir = Path(model_dir)
    parsed_dir = model_dir / "parsed"
    if not parsed_dir.is_dir():
        print(f"[skip] no parsed/ dir: {model_dir}")
        return None
    jsonl_paths = [p for p in sorted(parsed_dir.glob("*.jsonl")) if output_suffix not in p.stem]
    if not jsonl_paths:
        print(f"[skip] no JSONL files in: {parsed_dir}")
        return None

    print(f"\nProcessing model: {model_dir.name}")
    all_results = []
    for jp in jsonl_paths:
        all_results.extend(process_jsonl(jp, output_suffix))

    summary = _aggregate_by_label_AD(all_results)
    out_path = parsed_dir / SUMMARY_PROC_ACC_AD_METRICS_FILENAME
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] per-label summary → {out_path}")
    _print_model_summary_AD(model_dir, summary)
    return summary


def _group_classify_AD(label):
    if "FeTA24_distance" in label:
        return "FeTA-Distance"
    if "Ceph-Biometrics-400_angle" in label:
        return "Ceph-Angle"
    if "Ceph-Biometrics-400_distance" in label:
        return "Ceph-Distance"
    return "Other"


def _calc_group_avg_AD(label_metrics_list):
    def _wavg(key, weight_key="n_samples"):
        s, n = 0.0, 0
        for m in label_metrics_list:
            v = m.get(key, float("nan"))
            w = m.get(weight_key, 0)
            if v is not None and not np.isnan(v):
                s += v * w
                n += w
        return s / n if n > 0 else float("nan")

    total_samples = sum(m.get("n_samples", 0) for m in label_metrics_list)
    total_valid   = sum(m.get("n_valid", m.get("n_samples", 0)) for m in label_metrics_list)
    return {
        "step1_avg_MAE":        _wavg("step1_avg_MAE"),
        "step2_avg_MAE":        _wavg("step2_avg_MAE"),
        "step3_avg_MRE":        _wavg("step3_avg_MRE",        "n_valid"),
        "step3_avg_nMAE": _wavg("step3_avg_nMAE", "n_valid"),
        "n_samples": total_samples,
        "n_valid":   total_valid,
    }


def _print_model_summary_AD(model_dir, summary):
    """Write per-model process-accuracy summary TXT to model_dir."""
    model_dir = Path(model_dir)
    out_path  = model_dir / SUMMARY_PROC_ACC_AD_MODEL_FILENAME
    lines     = []

    def _p(text):
        lines.append(text)

    total_n = total_v = 0
    wsum    = {k: 0.0 for k in ("step1_avg_MAE", "step2_avg_MAE", "step3_avg_MRE", "step3_avg_nMAE")}
    wcount  = {k: 0   for k in wsum}
    groups  = {"FeTA-Distance": [], "Ceph-Angle": [], "Ceph-Distance": [], "Other": []}

    for label, lm in summary.items():
        n       = lm.get("n_samples", 0)
        n_valid = lm.get("n_valid", n)
        if n <= 0:
            continue
        total_n += n
        total_v += n_valid
        groups[_group_classify_AD(label)].append(lm)
        for k in wsum:
            v = lm.get(k, float("nan"))
            w = n_valid if k in ("step3_avg_MRE", "step3_avg_nMAE") else n
            if v is not None and not np.isnan(v):
                wsum[k]   += v * w
                wcount[k] += w

    def _wf(k):
        return wsum[k] / wcount[k] if wcount[k] > 0 else float("nan")

    _p(f"\nModel: {model_dir.name}")
    _p(
        f"Weighted Average → Step1_MAE: {_wf('step1_avg_MAE'):.4f}, "
        f"Step2_MAE: {_wf('step2_avg_MAE'):.4f}, "
        f"Step3_MRE: {_wf('step3_avg_MRE'):.4f}, "
        f"nMAE (step 3): {_wf('step3_avg_nMAE'):.4f} "
        f"(Valid: {total_v}, Total: {total_n})"
    )

    _p("\nGroup averages:")
    _p(f"{'Group':<15} | {'Step1_MAE':<10} | {'Step2_MAE':<10} | {'Step3_MRE':<10} | {'nMAE (step 3)':<14} | {'Valid':<7} | {'Samples':<8}")
    _p("-" * 87)
    for gname in ("FeTA-Distance", "Ceph-Angle", "Ceph-Distance"):
        ga = _calc_group_avg_AD(groups[gname])
        _p(
            f"{gname:<15} | "
            f"{ga['step1_avg_MAE']:<10.4f} | "
            f"{ga['step2_avg_MAE']:<10.4f} | "
            f"{ga['step3_avg_MRE']:<10.4f} | "
            f"{ga['step3_avg_nMAE']:<14.4f} | "
            f"{ga['n_valid']:<7} | "
            f"{ga['n_samples']:<8}"
        )

    _p("\nLabel-specific metrics:")
    _p(
        f"{'Label':<50} | {'Type':<8} | {'Step1_MAE':<10} | {'Step2_MAE':<10} | "
        f"{'Step3_MRE':<10} | {'nMAE (step 3)':<14} | {'SR':<6} | {'Ignored':<8} | {'Samples':<8}"
    )
    _p("-" * 144)
    for label, lm in sorted(summary.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True):
        _p(
            f"{label:<50} | "
            f"{lm.get('metric_type', ''):<8} | "
            f"{lm.get('step1_avg_MAE', float('nan')):<10.4f} | "
            f"{lm.get('step2_avg_MAE', float('nan')):<10.4f} | "
            f"{lm.get('step3_avg_MRE', float('nan')):<10.4f} | "
            f"{lm.get('step3_avg_nMAE', float('nan')):<10.4f} | "
            f"{lm.get('success_rate', float('nan')):<6.4f} | "
            f"{lm.get('n_ignored', 0):<8} | "
            f"{lm.get('n_samples', 0):<8}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [saved] per-model summary → {out_path}")


def _print_cross_model_summaries_AD(task_dir):
    """Read per-model summary JSONs, print group/label tables, save summary TXT."""
    task_dir = Path(task_dir)
    out_path = task_dir / "summary_proc_acc_AD_task.txt"
    lines = []

    def _p(text):
        print(text)
        lines.append(text)

    _p("\n\n========== MODEL SUMMARIES (Process Accuracy - AD Task) ==========\n")

    for model_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        summary_json = model_dir / "parsed" / SUMMARY_PROC_ACC_AD_METRICS_FILENAME
        if not summary_json.exists():
            continue
        with open(summary_json) as f:
            metrics = json.load(f)

        _p(f"\nModel: {model_dir.name}")

        total_n = total_v = 0
        wsum = {k: 0.0 for k in ("step1_avg_MAE", "step2_avg_MAE", "step3_avg_MRE", "step3_avg_nMAE")}
        wcount = {k: 0 for k in wsum}
        groups = {"FeTA-Distance": [], "Ceph-Angle": [], "Ceph-Distance": [], "Other": []}

        for label, lm in metrics.items():
            n       = lm.get("n_samples", 0)
            n_valid = lm.get("n_valid", n)
            if n <= 0:
                continue
            total_n += n
            total_v += n_valid
            groups[_group_classify_AD(label)].append(lm)
            for k in wsum:
                v = lm.get(k, float("nan"))
                w = n_valid if k in ("step3_avg_MRE", "step3_avg_nMAE") else n
                if v is not None and not np.isnan(v):
                    wsum[k] += v * w
                    wcount[k] += w

        def _wf(k):
            return wsum[k] / wcount[k] if wcount[k] > 0 else float("nan")

        _p(
            f"Weighted Average → Step1_MAE: {_wf('step1_avg_MAE'):.4f}, "
            f"Step2_MAE: {_wf('step2_avg_MAE'):.4f}, "
            f"Step3_MRE: {_wf('step3_avg_MRE'):.4f}, "
            f"nMAE (step 3): {_wf('step3_avg_nMAE'):.4f} "
            f"(Valid: {total_v}, Total: {total_n})"
        )

        _p("\nGroup averages:")
        _p(f"{'Group':<15} | {'Step1_MAE':<10} | {'Step2_MAE':<10} | {'Step3_MRE':<10} | {'nMAE':<10} | {'Valid':<7} | {'Samples':<8}")
        _p("-" * 83)
        for gname in ("FeTA-Distance", "Ceph-Angle", "Ceph-Distance"):
            ga = _calc_group_avg_AD(groups[gname])
            _p(
                f"{gname:<15} | "
                f"{ga['step1_avg_MAE']:<10.4f} | "
                f"{ga['step2_avg_MAE']:<10.4f} | "
                f"{ga['step3_avg_MRE']:<10.4f} | "
                f"{ga['step3_avg_nMAE']:<14.4f} | "
                f"{ga['n_valid']:<7} | "
                f"{ga['n_samples']:<8}"
            )

        _p("\nLabel-specific metrics:")
        _p(
            f"{'Label':<50} | {'Type':<8} | {'Step1_MAE':<10} | {'Step2_MAE':<10} | "
            f"{'Step3_MRE':<10} | {'nMAE':<10} | {'SR':<6} | {'Ignored':<8} | {'Samples':<8}"
        )
        _p("-" * 140)
        for label, lm in sorted(metrics.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True):
            _p(
                f"{label:<50} | "
                f"{lm.get('metric_type', ''):<8} | "
                f"{lm.get('step1_avg_MAE', float('nan')):<10.4f} | "
                f"{lm.get('step2_avg_MAE', float('nan')):<10.4f} | "
                f"{lm.get('step3_avg_MRE', float('nan')):<10.4f} | "
                f"{lm.get('step3_avg_nMAE', float('nan')):<14.4f} | "
                f"{lm.get('success_rate', float('nan')):<6.4f} | "
                f"{lm.get('n_ignored', 0):<8} | "
                f"{lm.get('n_samples', 0):<8}"
            )
        _p("\n" + "=" * 100 + "\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze intermediate-step accuracy for A/D task JSONL files."
    )
    parser.add_argument(
        "--task_dir", default=None,
        help=(
            "Task results directory whose immediate subdirectories are model folders. "
            "Each model folder must contain a 'parsed/' subfolder with JSONL files."
        ),
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Single model directory containing a 'parsed/' subfolder with JSONL files.",
    )
    parser.add_argument(
        "--jsonl", nargs="+", default=None,
        help="One or more explicit JSONL file paths (or glob patterns) to analyze"
    )
    parser.add_argument(
        "--output_suffix", default="_proc_acc",
        help="Suffix appended before .jsonl in the output filename (default: _proc_acc)"
    )
    args = parser.parse_args()

    if args.task_dir is None and args.model_dir is None and args.jsonl is None:
        parser.error("Provide at least one of --task_dir, --model_dir, or --jsonl.")

    if args.jsonl:
        paths = _collect_from_jsonl_args(args.jsonl)
        if not paths:
            print("Error: no valid JSONL files found.", file=sys.stderr)
            sys.exit(1)
        for jp in paths:
            process_jsonl(jp, args.output_suffix)

    if args.model_dir:
        print(f"[Info] Processing model dir: {args.model_dir}")
        _process_model_dir(args.model_dir, args.output_suffix)

    if args.task_dir:
        model_dirs = sorted(d for d in Path(args.task_dir).iterdir() if d.is_dir())
        print(f"[Info] Discovered {len(model_dirs)} model dir(s) under: {args.task_dir}")
        for model_dir in model_dirs:
            _process_model_dir(model_dir, args.output_suffix)
        _print_cross_model_summaries_AD(args.task_dir)


if __name__ == "__main__":
    main()
