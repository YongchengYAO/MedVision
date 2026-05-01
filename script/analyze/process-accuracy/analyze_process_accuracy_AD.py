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

def _cal_MAE(pred, gt):
    return float(np.mean(np.abs(np.array(pred, float) - np.array(gt, float))))


def _cal_MAE_scaled(pred, gt, scale):
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
        result["step1_MAE"]  = _cal_MAE(p1, gp1)
    else:
        result["step1_pred"] = result["step1_MAE"] = None

    m2 = _search(PATTERNS_DIST_GROUP[2], solution) or _search(PATTERNS_DIST_ANSWER_ONLY[2], solution)
    if m2:
        p2 = [float(m2.group(1)), float(m2.group(2))]
        result["step2_pred"] = p2
        result["step2_MAE"]  = _cal_MAE(p2, gp2)
    else:
        result["step2_pred"] = result["step2_MAE"] = None

    m3 = _search(PATTERNS_DIST_GROUP[3], solution) or _search(PATTERNS_DIST_ANSWER_ONLY[3], solution)
    if m3:
        pd = float(m3.group(1))
        result["step3_pred"] = pd
        result["step3_MRE"]  = _cal_MRE([pd], [gv])
        result["step3_MAE_scaled"]  = _cal_MAE_scaled([pd], [gv], scale)
    else:
        result["step3_pred"] = result["step3_MRE"] = None
        result["step3_MAE_scaled"] = None

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
        g1 = [gl1p1[0], gl1p1[1], gl1p2[0], gl1p2[1]]
        g2 = [gl1p2[0], gl1p2[1], gl1p1[0], gl1p1[1]]
        result["step1_MAE"] = min(_cal_MAE(p, g1), _cal_MAE(p, g2))
    else:
        result["step1_pred"] = result["step1_MAE"] = None

    m2 = _search(PATTERNS_ANGLE_GROUP[2], solution) or _search(PATTERNS_ANGLE_ANSWER_ONLY[2], solution)
    if m2:
        p = [float(m2.group(i)) for i in range(1, 5)]
        result["step2_pred"] = p
        g1 = [gl2p1[0], gl2p1[1], gl2p2[0], gl2p2[1]]
        g2 = [gl2p2[0], gl2p2[1], gl2p1[0], gl2p1[1]]
        result["step2_MAE"] = min(_cal_MAE(p, g1), _cal_MAE(p, g2))
    else:
        result["step2_pred"] = result["step2_MAE"] = None

    m3 = _search(PATTERNS_ANGLE_GROUP[3], solution) or _search(PATTERNS_ANGLE_ANSWER_ONLY[3], solution)
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
        ("step3_MAE_scaled", "Step3 MAE Scaled (distance)", "distance"),
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

    jsonl_paths = []
    if args.model_dir:
        discovered = _collect_from_model_dir(args.model_dir)
        print(f"[Info] Discovered {len(discovered)} JSONL file(s) under: {args.model_dir}")
        jsonl_paths.extend(discovered)
    if args.task_dir:
        discovered = _collect_from_task_dir(args.task_dir)
        print(f"[Info] Discovered {len(discovered)} JSONL file(s) under: {args.task_dir}")
        jsonl_paths.extend(discovered)
    if args.jsonl:
        jsonl_paths.extend(_collect_from_jsonl_args(args.jsonl))

    seen = set()
    jsonl_paths = [p for p in jsonl_paths if not (p in seen or seen.add(p))]

    if not jsonl_paths:
        print("Error: no valid JSONL files found.", file=sys.stderr)
        sys.exit(1)

    for jp in jsonl_paths:
        process_jsonl(jp, args.output_suffix)


if __name__ == "__main__":
    main()
