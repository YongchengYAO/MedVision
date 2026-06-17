"""
Visualize MedVision Detection task predictions: image + GT bounding box + model-predicted box.

For each sample in a JSONL file, parses the predicted bounding box from the CoT response
(step-1-answer / answer tags), loads the original NIfTI slice, and saves a matplotlib figure.

Coordinate conversion (model image space → array space):
    Model predicts [x_min, y_min, x_max, y_max] in image space (origin lower-left):
        dim1_min = x_min * W           # col
        dim1_max = x_max * W
        dim0_min = H * (1 - y_max)     # row; y_max → smaller row index
        dim0_max = H * (1 - y_min)

Display convention:
    imshow(img.T, origin="lower") → plot_x = idx_dim0 (row), plot_y = idx_dim1 (col)
"""

import argparse
import glob
import json
import os
import re

from tqdm import tqdm

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    _load_nifti_2d,
)
from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.plot_utils import plot_detection_on_image

# Matches exactly two parenthesized (x, y) pairs where each value is in [0, 1].
# Mirrors the regex in plot_tl_axes.py for consistent CoT parsing.
_RP = r"(?:0(?:\.\d+)?|1(?:\.0+)?)"
_CG = rf"\(\s*({_RP})\s*,\s*({_RP})\s*\)"
_COORD_PAT = re.compile(rf"{_CG}\s*,\s*{_CG}", re.DOTALL)


def _extract_resp_text(resps):
    """Unwrap nested list until a string is found. Handles [[text]] and [[[text]]]."""
    val = resps
    while isinstance(val, list):
        if not val:
            return ""
        val = val[0]
    return val if isinstance(val, str) else str(val)


def _extract_dataset_name(jsonl_path):
    """Extract dataset name from JSONL filename (e.g. '..._samples_CrossMoDA_BoxCoordinate_...')."""
    m = re.search(r"_samples_(.+?)_BoxCoordinate_", os.path.basename(jsonl_path))
    if not m:
        raise ValueError(f"Cannot extract dataset name from: {jsonl_path}")
    return m.group(1)


def parse_box_coords(resp_text):
    """
    Parse predicted bounding box from model response text.

    Tries three strategies in order:
    1. CoT <step-1-answer>(x1,y1),(x2,y2)</step-1-answer> — parenthesized pair format
    2. <answer>x1,y1,x2,y2</answer> — 4 floats within answer tag
    3. Last 4 numbers in text, validated to [0, 1] range — non-CoT fallback

    Returns [x_min, y_min, x_max, y_max] normalized in [0,1] image space, or None on failure.
    """
    # Strategy 1: CoT step-1-answer with two parenthesized pairs
    m1 = re.search(r"<step-1-answer>(.*?)</step-1-answer>", resp_text, re.DOTALL)
    if m1:
        pm = _COORD_PAT.search(m1.group(1))
        if pm:
            x1, y1, x2, y2 = [float(pm.group(i)) for i in range(1, 5)]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    # Strategy 2: <answer> tag — extract 4 floats
    m2 = re.search(r"<answer>(.*?)</answer>", resp_text, re.DOTALL)
    if m2:
        nums = [
            s.replace(",", "")
            for s in re.findall(
                r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?",
                m2.group(1),
            )
        ]
        if len(nums) >= 4:
            x1, y1, x2, y2 = [float(n) for n in nums[-4:]]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    # Strategy 3: last 4 numbers in full text, validated to [0, 1]
    nums = [
        s.replace(",", "")
        for s in re.findall(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", resp_text
        )
    ]
    if len(nums) >= 4:
        x1, y1, x2, y2 = [float(n) for n in nums[-4:]]
        if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    return None


def _box_to_array_space(box_norm, H, W):
    """Convert [x_min, y_min, x_max, y_max] normalized image space to [dim0_min, dim1_min, dim0_max, dim1_max] array space."""
    x_min, y_min, x_max, y_max = box_norm
    return [H * (1 - y_max), x_min * W, H * (1 - y_min), x_max * W]


def process_model_dir(model_dir, task_folder, base_fig_dir, limit_per_jsonl):
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

        try:
            dataset_name = _extract_dataset_name(jsonl_path)
        except ValueError as e:
            print(f"  WARNING: {e}, skipping")
            continue

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

            image_hw = doc["image_size_2d"]  # [H, W] in array space
            H, W = image_hw
            slice_dim = doc["slice_dim"]
            slice_idx = doc["slice_idx"]
            task_id = doc["taskID"]
            label = doc["label"]

            # Parse GT from target field
            try:
                gt_norm = json.loads(record["target"])
            except (KeyError, json.JSONDecodeError, TypeError):
                continue
            gt_box = _box_to_array_space(gt_norm, H, W)

            # Parse prediction from response
            pred_norm = parse_box_coords(resp_text)
            if pred_norm is None:
                continue
            pred_box = _box_to_array_space(pred_norm, H, W)

            # Load image at original resolution
            img_path = doc["image_file"]
            if not os.path.exists(img_path):
                print(f"    WARNING: image not found: {img_path}")
                continue
            try:
                pixel_sizes_from_nii, img_2d = _load_nifti_2d(
                    img_path, slice_dim, slice_idx
                )
                img_2d = normalize_img(doc, img_2d)
            except Exception as e:
                print(f"    WARNING: failed to load image {img_path}: {e}")
                continue

            fig_name = (
                f"{dataset_name}__Task{task_id}__doc{doc_id}"
                f"__dim{slice_dim}__idx{slice_idx}__label{label}.png"
            )
            fig_path = os.path.join(out_dir, dataset_name, fig_name)

            try:
                plot_detection_on_image(
                    image_2d=img_2d,
                    pixel_sizes=pixel_sizes_from_nii,
                    gt_box=gt_box,
                    pred_box=pred_box,
                    slice_dim=slice_dim,
                    slice_idx=slice_idx,
                    fig_path=fig_path,
                )
            except Exception as e:
                print(f"    WARNING: plotting failed for doc {doc_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MedVision Detection task predictions (GT + model-predicted bounding boxes)"
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=None,
        help="Task directory containing model subdirectories (e.g. Results/MedVision-detect-v2/)",
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
                model_dir, task_folder, args.fig_dir, args.limit_per_jsonl
            )
    else:
        model_dir = args.model_dir.rstrip("/")
        task_folder = os.path.basename(os.path.dirname(model_dir))
        print(f"Model: {os.path.basename(model_dir)}")
        process_model_dir(model_dir, task_folder, args.fig_dir, args.limit_per_jsonl)


if __name__ == "__main__":
    main()
