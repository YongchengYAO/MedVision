"""
Visualize MedVision A/D task predictions: image + GT landmarks + model-predicted landmarks.

For each sample in a JSONL file, extracts landmark coordinates from the CoT response
(step-1-answer / step-2-answer tags), loads the GT coordinates from the landmark file,
and saves a matplotlib figure showing both on the normalized image.

Supported task types (detected automatically from doc["biometric_profile"]["metric_type"]):
  - "distance": 2 landmarks + 1 connecting line each for GT and model prediction
  - "angle":    4 landmarks + 2 lines each for GT and model prediction (4 lines total)

Color scheme:
  GT line 1:    #A21CAF (purple dashed)
  GT line 2:    #4F46E5 (indigo dashed, angle task only)
  Pred line 1:  #F37020 (orange solid)
  Pred line 2:  #FBBC05 (yellow solid, angle task only)
  Dots (4):     #4285F4, #EA4335, #FDB813, #34A853

Coordinate convention (model image space → array space → plot):
    Model predicts (x_rel, y_rel) with origin lower-left:
        idx_dim0 = H * (1 - y_rel)   (row)
        idx_dim1 = x_rel * W          (col)
    imshow(image_2d.T, origin="lower") → plot_x = idx_dim0, plot_y = idx_dim1
"""

import argparse
import glob
import gzip
import importlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    _load_nifti_2d,
)
from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE
from medvision_bm.utils.plot_utils import plot_ad_on_image

_SLICE_DIM_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

# ---------------------------------------------------------------------------
# GT landmark loading helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_3d_to_2d(coor_3d, slice_dim):
    """Return [idx_dim0, idx_dim1] array-space coords from a 3-element landmark list."""
    if slice_dim == 0:
        return coor_3d[1:3]
    elif slice_dim == 1:
        return [coor_3d[0], coor_3d[2]]
    elif slice_dim == 2:
        return coor_3d[0:2]
    raise ValueError(f"slice_dim must be 0/1/2, got {slice_dim}")


def _load_slice_landmarks(doc, landmark_keys):
    """
    Load (idx_dim0, idx_dim1) array-space coords for the given landmark_keys from
    the landmark JSON file referenced by doc. Returns dict {key: [row, col]} or None.
    """
    lm_data = _load_json(doc["landmark_file"])
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    dim_to_key = {0: "slice_landmarks_x", 1: "slice_landmarks_y", 2: "slice_landmarks_z"}
    lm_slice_ls = lm_data[dim_to_key[slice_dim]]

    matched = [e for e in lm_slice_ls if e.get("slice_idx") == slice_idx]
    if not matched:
        return None

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


@lru_cache(maxsize=128)
def _get_benchmark_plan(dataset_name):
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset '{dataset_name}' not in DATASETS_NAME2PACKAGE.")
    bm_module = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_biometry"
    )
    return bm_module.benchmark_plan


def _get_gt_coords(doc):
    """
    Return GT landmark array-space coords for a distance or angle task doc.

    Distance → {"metric_type": "distance", "p1": [r,c], "p2": [r,c]}
    Angle    → {"metric_type": "angle",
                "l1p1": [r,c], "l1p2": [r,c], "l2p1": [r,c], "l2p2": [r,c]}
    Returns (result_dict, None) on success, (None, error_str) on failure.
    """
    bp = doc["biometric_profile"]
    metric_type = bp["metric_type"]
    metric_key = bp["metric_key"]

    try:
        bm_plan = _get_benchmark_plan(doc["dataset_name"])
        task_info = bm_plan["tasks"][int(doc["taskID"]) - 1]
    except Exception as e:
        return None, f"benchmark_plan error: {e}"

    try:
        if metric_type == "distance":
            line_dict = task_info[bp["metric_map_name"]][metric_key]
            lm_keys = line_dict["element_keys"]
            lm_coords = _load_slice_landmarks(doc, lm_keys)
            if lm_coords is None:
                return None, "landmark_not_found"
            return {
                "metric_type": "distance",
                "p1": lm_coords[lm_keys[0]],
                "p2": lm_coords[lm_keys[1]],
            }, None

        elif metric_type == "angle":
            angles_map = task_info[bp["metric_map_name"]]
            angle_dict = angles_map[metric_key]
            lines_map = task_info[angle_dict["element_map_name"]]
            l1_keys = lines_map[angle_dict["element_keys"][0]]["element_keys"]
            l2_keys = lines_map[angle_dict["element_keys"][1]]["element_keys"]
            all_keys = list(dict.fromkeys(l1_keys + l2_keys))
            lm_coords = _load_slice_landmarks(doc, all_keys)
            if lm_coords is None:
                return None, "landmark_not_found"
            return {
                "metric_type": "angle",
                "l1p1": lm_coords[l1_keys[0]],
                "l1p2": lm_coords[l1_keys[1]],
                "l2p1": lm_coords[l2_keys[0]],
                "l2p2": lm_coords[l2_keys[1]],
            }, None

        else:
            return None, f"unknown metric_type: {metric_type}"

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Model prediction parsing
# ---------------------------------------------------------------------------

def _extract_coords_from_tag(text, n):
    """Extract the last n floats in [0, 1] from step-k-answer tag content.
    Handles both strict format '(x, y)' and named-variable format
    '(x_name, y_name) = (0.3, 0.6)'."""
    vals = []
    for tok in re.findall(r"-?\d+\.?\d*", text):
        try:
            v = float(tok)
            if 0.0 <= v <= 1.0:
                vals.append(v)
        except ValueError:
            pass
    return vals[-n:] if len(vals) >= n else None


def _extract_resp_text(resps):
    """Unwrap nested list until a string is found."""
    val = resps
    while isinstance(val, list):
        if not val:
            return ""
        val = val[0]
    return val if isinstance(val, str) else str(val)


def _to_array(x_rel, y_rel, H, W):
    """Relative image coords (origin lower-left) → (idx_dim0, idx_dim1) array space."""
    return (H * (1 - y_rel), x_rel * W)


def _parse_dist_preds(resp_text, H, W):
    """
    Parse distance task response for model-predicted landmark coords.
    Returns (p1, p2) as (idx_dim0, idx_dim1) tuples, or None on failure.
    """
    m1 = re.search(r"<step-1-answer>(.*?)</step-1-answer>", resp_text, re.DOTALL)
    m2 = re.search(r"<step-2-answer>(.*?)</step-2-answer>", resp_text, re.DOTALL)
    if not m1 or not m2:
        return None
    c1 = _extract_coords_from_tag(m1.group(1), 2)
    c2 = _extract_coords_from_tag(m2.group(1), 2)
    if c1 is None or c2 is None:
        return None
    p1 = _to_array(c1[0], c1[1], H, W)
    p2 = _to_array(c2[0], c2[1], H, W)
    return p1, p2


def _parse_angle_preds(resp_text, H, W):
    """
    Parse angle task response for model-predicted line endpoint coords.
    Returns (l1p1, l1p2, l2p1, l2p2) as (idx_dim0, idx_dim1) tuples, or None.
    """
    m1 = re.search(r"<step-1-answer>(.*?)</step-1-answer>", resp_text, re.DOTALL)
    m2 = re.search(r"<step-2-answer>(.*?)</step-2-answer>", resp_text, re.DOTALL)
    if not m1 or not m2:
        return None
    c1 = _extract_coords_from_tag(m1.group(1), 4)
    c2 = _extract_coords_from_tag(m2.group(1), 4)
    if c1 is None or c2 is None:
        return None
    l1p1 = _to_array(c1[0], c1[1], H, W)
    l1p2 = _to_array(c1[2], c1[3], H, W)
    l2p1 = _to_array(c2[0], c2[1], H, W)
    l2p2 = _to_array(c2[2], c2[3], H, W)
    return l1p1, l1p2, l2p1, l2p2


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_model_dir(model_dir, task_folder, base_fig_dir, limit_per_jsonl, show_coords=False):
    model_name = os.path.basename(model_dir.rstrip("/"))
    out_dir = os.path.join(base_fig_dir, task_folder, model_name)
    os.makedirs(out_dir, exist_ok=True)

    jsonl_files = sorted(glob.glob(os.path.join(model_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"  No JSONL files found in {model_dir}")
        return

    for jsonl_path in jsonl_files:
        jsonl_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        print(f"  Processing {jsonl_name} ...")

        with open(jsonl_path) as f:
            lines = f.readlines()

        if limit_per_jsonl is not None:
            lines = lines[:limit_per_jsonl]

        for line in tqdm(lines, desc=f"    {jsonl_name}", leave=False):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc = record["doc"]
            doc_id = record["doc_id"]
            resp_text = _extract_resp_text(record["resps"])
            image_hw = doc["image_size_2d"]
            H, W = image_hw
            slice_dim = doc["slice_dim"]
            slice_idx = doc["slice_idx"]
            dataset_name = doc["dataset_name"]
            task_id = doc["taskID"]
            metric_key = doc["biometric_profile"]["metric_key"]
            metric_type = doc["biometric_profile"]["metric_type"]

            # Parse model predictions
            if metric_type == "distance":
                pred_pts = _parse_dist_preds(resp_text, H, W)
            elif metric_type == "angle":
                pred_pts = _parse_angle_preds(resp_text, H, W)
            else:
                continue
            if pred_pts is None:
                continue

            # Load GT coords
            gt_pts, err = _get_gt_coords(doc)
            if gt_pts is None:
                print(f"    WARNING: GT load failed for doc {doc_id}: {err}")
                continue

            # Load image
            img_path = doc["image_file"]
            if not os.path.exists(img_path):
                print(f"    WARNING: image not found: {img_path}")
                continue
            try:
                pixel_sizes_from_nii, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)
                img_2d = normalize_img(doc, img_2d)
            except Exception as e:
                print(f"    WARNING: failed to load image {img_path}: {e}")
                continue

            # Output path
            fig_name = (
                f"{dataset_name}__Task{task_id}__doc{doc_id}"
                f"__dim{slice_dim}__idx{slice_idx}__{metric_key}.png"
            )
            fig_path = os.path.join(out_dir, dataset_name, fig_name)

            try:
                plot_ad_on_image(
                    image_2d=img_2d,
                    pixel_sizes=pixel_sizes_from_nii,
                    metric_type=metric_type,
                    gt_pts=gt_pts,
                    pred_pts=pred_pts,
                    slice_dim=slice_dim,
                    slice_idx=slice_idx,
                    fig_path=fig_path,
                    show_coords=show_coords,
                )
            except Exception as e:
                print(f"    WARNING: plotting failed for doc {doc_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MedVision A/D task predictions (GT + model-predicted landmarks)"
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=None,
        help="Task directory containing model subdirectories (e.g. Results/MedVision-AD-v2-CoT/)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Single model directory (alternative to --task_dir)",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="/mnt/vincent-pvc-rwm/Github/MedVision/Figures",
        help="Base output directory for figures",
    )
    parser.add_argument(
        "--limit-per-jsonl",
        type=int,
        default=None,
        dest="limit_per_jsonl",
        help="Max samples to process per JSONL file",
    )
    parser.add_argument(
        "--show-coords",
        action="store_true",
        default=False,
        dest="show_coords",
        help="Annotate each landmark dot with its relative (x, y) coordinates",
    )
    args = parser.parse_args()

    if args.task_dir is None and args.model_dir is None:
        parser.error("Provide --task_dir or --model_dir")

    if args.task_dir is not None:
        task_folder = os.path.basename(args.task_dir.rstrip("/"))
        model_dirs = sorted(
            d for d in glob.glob(os.path.join(args.task_dir, "*/")) if os.path.isdir(d)
        )
        if not model_dirs:
            print(f"No model directories found in {args.task_dir}")
            return
        for model_dir in model_dirs:
            print(f"Model: {os.path.basename(model_dir.rstrip('/'))}")
            process_model_dir(
                model_dir, task_folder, args.fig_dir, args.limit_per_jsonl, args.show_coords
            )
    else:
        model_dir = args.model_dir.rstrip("/")
        task_folder = os.path.basename(os.path.dirname(model_dir))
        print(f"Model: {os.path.basename(model_dir)}")
        process_model_dir(
            model_dir, task_folder, args.fig_dir, args.limit_per_jsonl, args.show_coords
        )


if __name__ == "__main__":
    main()
