"""
Analyze intermediate-step accuracy for T/L (Tumor/Lesion size) task model responses.

GT landmark coordinates are loaded directly from each sample's landmark JSON file
(referenced via doc["landmark_file"]).  This approach is independent of the parquet
dataset and works for any model regardless of image size or prompt format.

For each sample in the input JSONL file(s):
  - T/L tasks (4 steps):
      Step 1: major axis endpoints (P1 & P2) → norm-L2 vs GT
      Step 2: minor axis endpoints (P3 & P4) → norm-L2 vs GT
      Step 3: major axis length (scalar)      → MRE vs GT
      Step 4: minor axis length (scalar)      → MRE vs GT

Usage:
    python analyze_process_accuracy_TL.py \
        --task_dir /path/to/MedVision-TL-v2-CoT \
        [--jsonl /path/to/explicit.jsonl ...] \
        [--output_suffix _proc_acc]
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

from medvision_bm.utils.configs import label_map_rename
from medvision_bm.utils.parse_utils import (
    get_labelsMap_imgModality_from_biometry_benchmark_plan,
    get_targetLabel_imgModality_from_biometry_benchmark_plan,
)


def _cal_norm_L2_dist(pred_xy, gt_xy):
    return float(
        np.sqrt((pred_xy[0] - gt_xy[0]) ** 2 + (pred_xy[1] - gt_xy[1]) ** 2)
        / np.sqrt(2)
    )


def _cal_nMAE(pred, gt, scale):
    return float(
        np.mean(np.abs(np.array(pred, float) - np.array(gt, float)) / (scale + 1e-15))
    )


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
    """Extract 2D  coordinate from a 3D coordinate list by dropping the slice dimension."""
    if slice_dim == 0:
        return coor_3d[1:3]
    elif slice_dim == 1:
        return [coor_3d[0], coor_3d[2]]
    elif slice_dim == 2:
        return coor_3d[0:2]
    raise ValueError(f"slice_dim must be 0/1/2, got {slice_dim}")


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


def _extract_gt_from_doc(doc):
    """
    Load GT P1-P4 normalized wh-coordinates and axis lengths from the JSONL doc.

    P1, P2 are the major axis endpoints; P3, P4 are the minor axis endpoints.
    Landmark positions are computed using the raw image dimensions stored in
    doc["image_size_2d"], which are the same dimensions used during dataset building.

    Returns:
        (gt_dict, None)  on success
        (None, error_str) on failure
    """
    try:
        bp = doc["biometric_profile"]
        # GT axis lengths (in original physical units, scale-invariant in mm)
        gt_major = float(bp["metric_value_major_axis"][0])
        gt_minor = float(bp["metric_value_minor_axis"][0])
    except (KeyError, IndexError, TypeError) as e:
        return None, f"biometric_profile parse error: {e}"

    try:
        lm_data = _load_json(doc["landmark_file"])
        slice_dim = doc["slice_dim"]
        slice_idx = doc["slice_idx"]
        img_h, img_w = doc["image_size_2d"]  # raw image [height, width]

        dim_to_key = {
            0: "slice_landmarks_x",
            1: "slice_landmarks_y",
            2: "slice_landmarks_z",
        }
        lm_slice_ls = lm_data[dim_to_key[slice_dim]]

        matched = [e for e in lm_slice_ls if e.get("slice_idx") == slice_idx]
        if not matched:
            return None, f"no landmarks for slice_dim={slice_dim} slice_idx={slice_idx}"

        # Merge all matched entries (handles multi-lesion cases; take first element)
        landmarks = {}
        for entry in matched:
            entry_lms = entry.get("landmarks", {})
            if isinstance(entry_lms, list):
                entry_lms = entry_lms[0]
            landmarks.update(entry_lms)

        result = {}
        for p_name in ("P1", "P2", "P3", "P4"):
            coor_3d = landmarks.get(p_name)
            if coor_3d is None:
                return None, f"landmark {p_name} not found in slice"
            coor_2d = _extract_3d_to_2d(coor_3d, slice_dim)
            result[p_name] = _to_wh(coor_2d, img_h, img_w)

        return {
            "P1_wh": result["P1"],
            "P2_wh": result["P2"],
            "P3_wh": result["P3"],
            "P4_wh": result["P4"],
            "gt_major": gt_major,
            "gt_minor": gt_minor,
        }, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Regex patterns (mirror medvision_tl.py)
# ---------------------------------------------------------------------------

_RP = r"(?:0(?:\.\d+)?|1(?:\.0+)?)"
_RPG = rf"({_RP})"
_NNR = r"\d+(?:\.\d+)?"
_NNRG = rf"({_NNR})"
_CG = rf"\(\s*{_RPG}\s*,\s*{_RPG}\s*\)"


def _tag(n):
    return rf"<{n}>"


def _end(n):
    return rf"</{n}>"


def _rea(k):
    return rf"{_tag(f'step-{k}-reasoning')}.*?{_end(f'step-{k}-reasoning')}"


PATTERNS_TL_GROUP = {
    1: rf"{_rea(1)}\s*.*?{_tag('step-1-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-1-answer')}.*?",
    2: rf"{_rea(2)}\s*.*?{_tag('step-2-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-2-answer')}.*?",
    3: rf"{_rea(3)}\s*.*?{_tag('step-3-answer')}.*?{_NNRG}.*?{_end('step-3-answer')}.*?",
    4: rf"{_rea(4)}\s*.*?{_tag('step-4-answer')}.*?{_NNRG}.*?{_end('step-4-answer')}.*?",
}
PATTERNS_TL_ANSWER_ONLY = {
    1: rf".*?{_tag('step-1-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-1-answer')}.*?",
    2: rf".*?{_tag('step-2-answer')}.*?{_CG}\s*,\s*{_CG}.*?{_end('step-2-answer')}.*?",
    3: rf".*?{_tag('step-3-answer')}.*?{_NNRG}.*?{_end('step-3-answer')}.*?",
    4: rf".*?{_tag('step-4-answer')}.*?{_NNRG}.*?{_end('step-4-answer')}.*?",
}

FLAGS = re.DOTALL

# Tool-use model outputs both axis lengths together in <answer> major, minor </answer>
PATTERN_TOOLUSE_TL_ANSWER = rf".*?<answer>\s*({_NNR})\s*,\s*({_NNR})\s*</answer>.*?"


def _search(pat, txt):
    return re.search(pat, txt, FLAGS)


# ---------------------------------------------------------------------------
# Per-sample analyzer
# ---------------------------------------------------------------------------


def analyze_tl_sample(solution, gt, scale):
    """
    Compute per-step metrics for a T/L sample.

    GT:
      - P1, P2 (major axis endpoints) → step 1 norm-L2
      - P3, P4 (minor axis endpoints) → step 2 norm-L2
      - gt_major (major axis length)  → step 3 MRE
      - gt_minor (minor axis length)  → step 4 MRE
    """
    p1, p2 = gt["P1_wh"], gt["P2_wh"]
    p3, p4 = gt["P3_wh"], gt["P4_wh"]
    gm, gmi = gt["gt_major"], gt["gt_minor"]

    result = {
        "metric_type": "tl",
        "gt_P1_wh": p1,
        "gt_P2_wh": p2,
        "gt_P3_wh": p3,
        "gt_P4_wh": p4,
        "gt_major_length": gm,
        "gt_minor_length": gmi,
    }

    # --- Step 1: major axis (P1, P2) ---
    m1 = _search(PATTERNS_TL_GROUP[1], solution) or _search(
        PATTERNS_TL_ANSWER_ONLY[1], solution
    )
    if m1:
        pred = [float(m1.group(i)) for i in range(1, 5)]
        result["step1_pred"] = pred
        pred_pts = [[pred[0], pred[1]], [pred[2], pred[3]]]
        d1 = (
            _cal_norm_L2_dist(pred_pts[0], p1) + _cal_norm_L2_dist(pred_pts[1], p2)
        ) / 2
        d2 = (
            _cal_norm_L2_dist(pred_pts[0], p2) + _cal_norm_L2_dist(pred_pts[1], p1)
        ) / 2
        result["step1_normL2"] = min(d1, d2)
    else:
        result["step1_pred"] = result["step1_normL2"] = None

    # --- Step 2: minor axis (P3, P4) ---
    m2 = _search(PATTERNS_TL_GROUP[2], solution) or _search(
        PATTERNS_TL_ANSWER_ONLY[2], solution
    )
    if m2:
        pred = [float(m2.group(i)) for i in range(1, 5)]
        result["step2_pred"] = pred
        pred_pts = [[pred[0], pred[1]], [pred[2], pred[3]]]
        d1 = (
            _cal_norm_L2_dist(pred_pts[0], p3) + _cal_norm_L2_dist(pred_pts[1], p4)
        ) / 2
        d2 = (
            _cal_norm_L2_dist(pred_pts[0], p4) + _cal_norm_L2_dist(pred_pts[1], p3)
        ) / 2
        result["step2_normL2"] = min(d1, d2)
    else:
        result["step2_pred"] = result["step2_normL2"] = None

    # --- Step 3: major axis length ---
    m3 = _search(PATTERNS_TL_GROUP[3], solution) or _search(
        PATTERNS_TL_ANSWER_ONLY[3], solution
    )
    if m3:
        pred = float(m3.group(1))
        result["step3_pred"] = pred
        result["step3_MRE"] = _cal_MRE([pred], [gm])
        result["step3_nMAE"] = _cal_nMAE([pred], [gm], scale)
    else:
        result["step3_pred"] = result["step3_MRE"] = result["step3_nMAE"] = None

    # --- Step 4: minor axis length ---
    m4 = _search(PATTERNS_TL_GROUP[4], solution) or _search(
        PATTERNS_TL_ANSWER_ONLY[4], solution
    )
    if m4:
        pred = float(m4.group(1))
        result["step4_pred"] = pred
        result["step4_MRE"] = _cal_MRE([pred], [gmi])
        result["step4_nMAE"] = _cal_nMAE([pred], [gmi], scale)
    else:
        result["step4_pred"] = result["step4_MRE"] = result["step4_nMAE"] = None

    # Tool-use model outputs both axis lengths together in <answer> major, minor </answer>
    if result["step3_MRE"] is None and result["step4_MRE"] is None:
        m_tool = re.search(PATTERN_TOOLUSE_TL_ANSWER, solution, FLAGS)
        if m_tool:
            major_pred = float(m_tool.group(1))
            minor_pred = float(m_tool.group(2))
            result["step3_pred"] = major_pred
            result["step3_MRE"] = _cal_MRE([major_pred], [gm])
            result["step3_nMAE"] = _cal_nMAE([major_pred], [gm], scale)
            result["step4_pred"] = minor_pred
            result["step4_MRE"] = _cal_MRE([minor_pred], [gmi])
            result["step4_nMAE"] = _cal_nMAE([minor_pred], [gmi], scale)

    return result


# ---------------------------------------------------------------------------
# Removed-samples filtering helpers (mirrors summarize_TL_task.py)
# ---------------------------------------------------------------------------

_DIM_MAP = {"x": 0, "y": 1, "z": 2}


def _build_removed_set(json_path):
    with open(json_path) as f:
        entries = json.load(f)
    return frozenset(
        (
            e["image_file"],
            _DIM_MAP[e["slice_dim"]],
            int(e["slice_idx"]),
            int(e["task_ID"]),
        )
        for e in entries
    )


def _relative_image_file(full_path, dataset_name):
    marker = f"/{dataset_name}/"
    idx = full_path.find(marker)
    return full_path[idx + len(marker) :] if idx >= 0 else Path(full_path).name


# ---------------------------------------------------------------------------
# Process a single JSONL file
# ---------------------------------------------------------------------------


def process_jsonl(jsonl_path, output_suffix, removed_set=None, dataset_name=None):
    jsonl_path = Path(jsonl_path)
    out_path = jsonl_path.with_name(jsonl_path.stem + output_suffix + ".jsonl")

    n_total = n_gt_fail = n_parse_fail = n_success = 0
    results = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            n_total += 1

            doc = sample.get("doc", {})
            doc_id = sample.get("doc_id")
            solution = sample.get("resps", [[""]])[0]
            if isinstance(solution, list):
                solution = solution[0] if solution else ""

            if removed_set is not None and dataset_name is not None:
                _img = doc.get("image_file", "")
                _key = (
                    _relative_image_file(_img, dataset_name),
                    doc.get("slice_dim"),
                    doc.get("slice_idx"),
                    int(doc.get("taskID", 0)),
                )
                if _key in removed_set:
                    continue

            record = {
                "doc_id": doc_id,
                "dataset": doc.get("dataset_name"),
                "taskID": doc.get("taskID"),
                "taskType": doc.get("taskType"),
                "image_file": doc.get("image_file"),
                "slice_dim": doc.get("slice_dim"),
                "slice_idx": doc.get("slice_idx"),
                "image_size_2d": doc.get("image_size_2d"),
                "pixel_size": doc.get("pixel_size"),
            }

            gt, err = _extract_gt_from_doc(doc)
            if gt is None:
                n_gt_fail += 1
                record["error"] = f"gt_extraction_failed: {err}"
                results.append(record)
                continue

            try:
                # scale distance error by the image diagonal to get a more interpretable relative error metric (e.g. 0.05 means 5% of the image diagonal)
                image_size_2d = doc.get("image_size_2d")
                pixel_size = doc.get("pixel_size")
                image_diagonal = np.sqrt(
                    (image_size_2d[0] * pixel_size[0]) ** 2
                    + (image_size_2d[1] * pixel_size[1]) ** 2
                )
                record.update(analyze_tl_sample(solution, gt, image_diagonal))
                if all(
                    record.get(k) is not None
                    for k in ("step1_normL2", "step2_normL2", "step3_MRE", "step4_MRE")
                ):
                    n_success += 1
            except Exception as e:
                n_parse_fail += 1
                record["error"] = str(e)

            results.append(record)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def _stats(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return None, None, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    n_tl = n_total - n_gt_fail
    success_str = (
        f", success_reasoning_rate={100*n_success/n_tl:.1f}% {n_success}/{n_tl}(tl)"
        if n_tl > 0
        else ""
    )

    print(f"\n[{jsonl_path.name}]")
    print(
        f"  Total: {n_total}  (gt_fail={n_gt_fail}, parse_fail={n_parse_fail}{success_str})"
    )
    for key, label in [
        ("step1_normL2", "Step1 normalized L2 distance (major endpts)"),
        ("step2_normL2", "Step2 normalized L2 distance (minor endpts)"),
        ("step3_MRE", "Step3 MRE (major length)"),
        ("step3_nMAE", "nMAE (step 3) (major length)"),
        ("step4_MRE", "Step4 MRE (minor length)"),
        ("step4_nMAE", "nMAE (step 4) (minor length)"),
    ]:
        if n_tl == 0:
            continue
        mean, sd, n = _stats(key)
        mean_str = f"{mean:.4f}" if mean is not None else "nan"
        sd_str = f"{sd:.4f}" if sd is not None else "nan"
        print(f"  {label}: mean={mean_str} ± sd={sd_str} (n={n})")
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
            if "_proc_acc" not in p.stem and "_eq_acc" not in p.stem:
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

SUMMARY_PROC_ACC_TL_METRICS_FILENAME = "summary_proc_acc_TL_metrics.json"
SUMMARY_PROC_ACC_TL_MODEL_FILENAME = "summary_proc_acc_TL_model.txt"

_IMGMOD_MAP = {"MRI": "MR", "CT": "CT", "ultrasound": "US", "X-ray": "XR", "PET": "PET"}
_SLICE_MAP = {0: "S", 1: "C", 2: "A"}


def _get_tl_label(record):
    """Derive anatomy label key (e.g. 'Hepatocellular Carcinoma @ CT (S)') from a TL record."""
    dataset = record.get("dataset")
    task_id = record.get("taskID")
    slice_dim = record.get("slice_dim")
    if dataset is None or task_id is None or slice_dim is None:
        return None
    try:
        label, _ = get_targetLabel_imgModality_from_biometry_benchmark_plan(
            dataset, int(task_id)
        )
        labels_map, img_modality = (
            get_labelsMap_imgModality_from_biometry_benchmark_plan(
                dataset, int(task_id)
            )
        )
        label_name = labels_map.get(str(label))
        if label_name is None:
            return None
        new_label = label_map_rename.get(label_name)
        if new_label is None:
            return None
        img_mod = _IMGMOD_MAP.get(img_modality, img_modality)
        slicetype = _SLICE_MAP.get(int(slice_dim))
        if slicetype is None:
            return None
        return f"{new_label} @ {img_mod} ({slicetype})"
    except Exception:
        return None


def _aggregate_by_label_TL(all_results):
    """Aggregate per-sample results by anatomy label; return {label: averaged step metrics}."""
    grouped = {}
    for r in all_results:
        if r.get("error"):
            continue
        label = _get_tl_label(r)
        if label is None:
            continue
        if label not in grouped:
            grouped[label] = {
                "s1": [],
                "s2": [],
                "s3_mre": [],
                "s4_mre": [],
                "s3_msc": [],
                "s4_msc": [],
                "n_success": 0,
                "n_samples": 0,
            }
        g = grouped[label]
        g["n_samples"] += 1
        s1, s2 = r.get("step1_normL2"), r.get("step2_normL2")
        s3_mre, s4_mre = r.get("step3_MRE"), r.get("step4_MRE")
        s3_msc, s4_msc = r.get("step3_nMAE"), r.get("step4_nMAE")
        if s1 is not None:
            g["s1"].append(s1)
        if s2 is not None:
            g["s2"].append(s2)
        if s3_mre is not None:
            g["s3_mre"].append(s3_mre)
        if s4_mre is not None:
            g["s4_mre"].append(s4_mre)
        if s3_msc is not None:
            g["s3_msc"].append(s3_msc)
        if s4_msc is not None:
            g["s4_msc"].append(s4_msc)
        if (
            s1 is not None
            and s2 is not None
            and s3_mre is not None
            and s4_mre is not None
        ):
            g["n_success"] += 1

    def _avg(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        label: {
            "step1_avg_normL2": _avg(g["s1"]),
            "step2_avg_normL2": _avg(g["s2"]),
            "step3_avg_MRE": _avg(g["s3_mre"]),
            "step4_avg_MRE": _avg(g["s4_mre"]),
            "step3_avg_nMAE": _avg(g["s3_msc"]),
            "step4_avg_nMAE": _avg(g["s4_msc"]),
            "n_samples": g["n_samples"],
            "success_rate": (
                g["n_success"] / g["n_samples"] if g["n_samples"] > 0 else 0.0
            ),
        }
        for label, g in grouped.items()
    }


def _print_model_summary_TL(model_dir, summary, filtered=False):
    """Write per-model process-accuracy summary TXT to model_dir."""
    model_dir = Path(model_dir)
    out_path = model_dir / (
        "summary_proc_acc_TL_model_filtered.txt"
        if filtered
        else SUMMARY_PROC_ACC_TL_MODEL_FILENAME
    )
    lines = []

    def _p(text):
        lines.append(text)

    total_n = 0
    wsum = {
        k: 0.0
        for k in (
            "step1_avg_normL2",
            "step2_avg_normL2",
            "step3_avg_MRE",
            "step4_avg_MRE",
            "step3_avg_nMAE",
            "step4_avg_nMAE",
        )
    }
    wcount = {k: 0 for k in wsum}

    for label, lm in summary.items():
        n = lm.get("n_samples", 0)
        if n <= 0:
            continue
        total_n += n
        for k in wsum:
            v = lm.get(k, float("nan"))
            if v is not None and not np.isnan(v):
                wsum[k] += v * n
                wcount[k] += n

    def _wf(k):
        return wsum[k] / wcount[k] if wcount[k] > 0 else float("nan")

    _p(f"\nModel: {model_dir.name}")
    _p(
        f"Weighted Average → Step1_normL2: {_wf('step1_avg_normL2'):.4f}, "
        f"Step2_normL2: {_wf('step2_avg_normL2'):.4f}, "
        f"Step3_MRE: {_wf('step3_avg_MRE'):.4f}, Step4_MRE: {_wf('step4_avg_MRE'):.4f}, "
        f"nMAE (step 3): {_wf('step3_avg_nMAE'):.4f}, nMAE (step 4): {_wf('step4_avg_nMAE'):.4f} "
        f"(Total Samples: {total_n})"
    )

    _p("\nLabel-specific metrics:")
    _p(
        f"{'Label':<52} | {'S1_MAE':<8} | {'S2_MAE':<8} | {'S3_MRE':<8} | {'S4_MRE':<8} | "
        f"{'nMAE (step 3)':<14} | {'nMAE (step 4)':<14} | {'SR':<6} | {'Samples':<8}"
    )
    _p("-" * 152)
    for label, lm in sorted(
        summary.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True
    ):
        _p(
            f"{label:<52} | "
            f"{lm.get('step1_avg_normL2', float('nan')):<8.4f} | "
            f"{lm.get('step2_avg_normL2', float('nan')):<8.4f} | "
            f"{lm.get('step3_avg_MRE', float('nan')):<8.4f} | "
            f"{lm.get('step4_avg_MRE', float('nan')):<8.4f} | "
            f"{lm.get('step3_avg_nMAE', float('nan')):<8.4f} | "
            f"{lm.get('step4_avg_nMAE', float('nan')):<8.4f} | "
            f"{lm.get('success_rate', float('nan')):<6.4f} | "
            f"{lm.get('n_samples', 0):<8}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [saved] per-model summary → {out_path}")


def _process_model_dir(
    model_dir, output_suffix, removed_samples_dir=None, removed_samples_filename=None
):
    """Process all JSONL files in model's parsed/ dir; save per-label summary JSON."""
    model_dir = Path(model_dir)
    parsed_dir = model_dir / "parsed"
    if not parsed_dir.is_dir():
        print(f"[skip] no parsed/ dir: {model_dir}")
        return None
    jsonl_paths = [
        p
        for p in sorted(parsed_dir.glob("*.jsonl"))
        if "_eq_acc" not in p.stem and "_proc_acc" not in p.stem
    ]
    if not jsonl_paths:
        print(f"[skip] no JSONL files in: {parsed_dir}")
        return None

    filtered = removed_samples_dir is not None
    effective_suffix = output_suffix + ("_filtered" if filtered else "")

    print(f"\nProcessing model: {model_dir.name}")
    all_results = []
    _removed_cache = {}
    for jp in jsonl_paths:
        m = re.search(r"samples_([^_]+)_", jp.name)
        ds_name = m.group(1) if m else None
        removed_set = None
        if filtered and ds_name is not None:
            if ds_name not in _removed_cache:
                json_path = (
                    Path(removed_samples_dir) / ds_name / removed_samples_filename
                )
                _removed_cache[ds_name] = (
                    _build_removed_set(json_path) if json_path.exists() else None
                )
            removed_set = _removed_cache.get(ds_name)
        all_results.extend(
            process_jsonl(
                jp, effective_suffix, removed_set=removed_set, dataset_name=ds_name
            )
        )

    summary = _aggregate_by_label_TL(all_results)
    metrics_fname = (
        "summary_proc_acc_TL_metrics_filtered.json"
        if filtered
        else SUMMARY_PROC_ACC_TL_METRICS_FILENAME
    )
    out_path = parsed_dir / metrics_fname
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] per-label summary → {out_path}")
    _print_model_summary_TL(model_dir, summary, filtered=filtered)
    return summary


def _print_cross_model_summaries_TL(task_dir, filtered=False):
    """Read per-model summary JSONs, print label table, save summary TXT."""
    task_dir = Path(task_dir)
    metrics_fname = (
        "summary_proc_acc_TL_metrics_filtered.json"
        if filtered
        else SUMMARY_PROC_ACC_TL_METRICS_FILENAME
    )
    out_path = task_dir / (
        "summary_proc_acc_TL_task_filtered.txt"
        if filtered
        else "summary_proc_acc_TL_task.txt"
    )
    lines = []

    def _p(text):
        print(text)
        lines.append(text)

    _p("\n\n========== MODEL SUMMARIES (Process Accuracy - TL Task) ==========\n")

    for model_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        summary_json = model_dir / "parsed" / metrics_fname
        if not summary_json.exists():
            continue
        with open(summary_json) as f:
            metrics = json.load(f)

        _p(f"\nModel: {model_dir.name}")

        total_n = 0
        wsum = {
            k: 0.0
            for k in (
                "step1_avg_normL2",
                "step2_avg_normL2",
                "step3_avg_MRE",
                "step4_avg_MRE",
                "step3_avg_nMAE",
                "step4_avg_nMAE",
            )
        }
        wcount = {k: 0 for k in wsum}

        for label, lm in metrics.items():
            n = lm.get("n_samples", 0)
            if n <= 0:
                continue
            total_n += n
            for k in wsum:
                v = lm.get(k, float("nan"))
                if v is not None and not np.isnan(v):
                    wsum[k] += v * n
                    wcount[k] += n

        def _wf(k):
            return wsum[k] / wcount[k] if wcount[k] > 0 else float("nan")

        _p(
            f"Weighted Average → Step1_normL2: {_wf('step1_avg_normL2'):.4f}, "
            f"Step2_normL2: {_wf('step2_avg_normL2'):.4f}, "
            f"Step3_MRE: {_wf('step3_avg_MRE'):.4f}, Step4_MRE: {_wf('step4_avg_MRE'):.4f}, "
            f"nMAE (step 3): {_wf('step3_avg_nMAE'):.4f}, nMAE (step 4): {_wf('step4_avg_nMAE'):.4f} "
            f"(Total Samples: {total_n})"
        )

        _p("\nLabel-specific metrics:")
        _p(
            f"{'Label':<52} | {'S1_MAE':<8} | {'S2_MAE':<8} | {'S3_MRE':<8} | {'S4_MRE':<8} | "
            f"{'S3_nMAE':<8} | {'S4_nMAE':<8} | {'SR':<6} | {'Samples':<8}"
        )
        _p("-" * 140)
        for label, lm in sorted(
            metrics.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True
        ):
            _p(
                f"{label:<52} | "
                f"{lm.get('step1_avg_normL2', float('nan')):<8.4f} | "
                f"{lm.get('step2_avg_normL2', float('nan')):<8.4f} | "
                f"{lm.get('step3_avg_MRE', float('nan')):<8.4f} | "
                f"{lm.get('step4_avg_MRE', float('nan')):<8.4f} | "
                f"{lm.get('step3_avg_nMAE', float('nan')):<14.4f} | "
                f"{lm.get('step4_avg_nMAE', float('nan')):<14.4f} | "
                f"{lm.get('success_rate', float('nan')):<6.4f} | "
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
        description="Analyze intermediate-step accuracy for T/L task JSONL files."
    )
    parser.add_argument(
        "--task_dir",
        default=None,
        help=(
            "Task results directory whose immediate subdirectories are model folders. "
            "Each model folder must contain a 'parsed/' subfolder with JSONL files."
        ),
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Single model directory containing a 'parsed/' subfolder with JSONL files.",
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=None,
        help="One or more explicit JSONL file paths (or glob patterns) to analyze",
    )
    parser.add_argument(
        "--output_suffix",
        default="_proc_acc",
        help="Suffix appended before .jsonl in the output filename (default: _proc_acc)",
    )
    parser.add_argument(
        "--removed_samples_dir",
        type=str,
        default=None,
        help=(
            "Root directory containing per-dataset removed_samples JSON files "
            "(e.g. .../Data/Datasets). When provided, samples listed in those files "
            "are excluded and output filenames get a '_filtered' suffix."
        ),
    )
    parser.add_argument(
        "--removed_samples_filename",
        type=str,
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Filename of the removed-samples JSON within each dataset subdirectory.",
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
        _process_model_dir(
            args.model_dir,
            args.output_suffix,
            removed_samples_dir=args.removed_samples_dir,
            removed_samples_filename=args.removed_samples_filename,
        )

    if args.task_dir:
        filtered = args.removed_samples_dir is not None
        model_dirs = sorted(d for d in Path(args.task_dir).iterdir() if d.is_dir())
        print(
            f"[Info] Discovered {len(model_dirs)} model dir(s) under: {args.task_dir}"
        )
        for model_dir in model_dirs:
            _process_model_dir(
                model_dir,
                args.output_suffix,
                removed_samples_dir=args.removed_samples_dir,
                removed_samples_filename=args.removed_samples_filename,
            )
        _print_cross_model_summaries_TL(args.task_dir, filtered=filtered)


if __name__ == "__main__":
    main()
