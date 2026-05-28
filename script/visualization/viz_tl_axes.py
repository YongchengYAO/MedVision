"""
Visualize MedVision TL task predictions: image + GT mask contour + GT axes (dashed) +
model-predicted axes (solid).

For each sample in a JSONL file, extracts the major/minor axis endpoints from the
CoT response (step-1-answer / step-2-answer tags), loads the original NIfTI slice
and segmentation mask, and saves a matplotlib figure.

Coordinate conversion (model image space → array space):
    Model predicts (x_rel, y_rel) in image space (origin lower-left):
        idx_dim1 = x_rel * W           # col unchanged
        idx_dim0 = H * (1 - y_rel)     # row flipped: y=0 at bottom → large idx_dim0

Display convention:
    imshow(img.T, origin="lower") → plot_x = idx_dim0 (row), plot_y = idx_dim1 (col)
"""

import argparse
import gzip
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    _load_nifti_2d,
)
from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.plot_utils import plot_tl_axes_on_image

_SLICE_DIM_NAMES = {0: "Sagittal", 1: "Coronal", 2: "Axial"}


def _build_removed_set(json_path):
    """Load a removed-samples JSON and return a frozenset of (relative_image_file, slice_dim_int, slice_idx, task_id) keys."""
    _dim_map = {"x": 0, "y": 1, "z": 2}
    with open(json_path) as f:
        entries = json.load(f)
    return frozenset(
        (e["image_file"], _dim_map[e["slice_dim"]], int(e["slice_idx"]), int(e["task_ID"]))
        for e in entries
    )


def _relative_image_file(full_path, dataset_name):
    """Extract the relative image file path (after the dataset-name component) from an absolute path."""
    marker = f"/{dataset_name}/"
    idx = full_path.find(marker)
    return full_path[idx + len(marker):] if idx >= 0 else Path(full_path).name

# Matches exactly two parenthesized (x, y) pairs where each value is in [0, 1].
# Mirrors the regex in analyze_process_accuracy_TL.py to ensure consistent parsing.
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


def parse_axis_coords(resp_text, image_hw):
    """
    Extract major/minor axis endpoints from CoT response and convert to array space.

    Parses <step-1-answer> (major) and <step-2-answer> (minor) tags.
    Uses a parenthesized-pair regex matching (x, y), (x, y) with values in [0, 1],
    identical to analyze_process_accuracy_TL.py, to avoid spurious numbers in
    the model's reasoning text corrupting the coordinate extraction.

    Model coordinates are in image space (origin lower-left):
        idx_dim1 = x_rel * W
        idx_dim0 = H * (1 - y_rel)

    Returns:
        (major_axis_pts, minor_axis_pts) each as [(dim0, dim1), (dim0, dim1)],
        or (None, None) if parsing fails.
    """
    H, W = image_hw

    m1 = re.search(r"<step-1-answer>(.*?)</step-1-answer>", resp_text, re.DOTALL)
    m2 = re.search(r"<step-2-answer>(.*?)</step-2-answer>", resp_text, re.DOTALL)
    if not m1 or not m2:
        return None, None

    pm1 = _COORD_PAT.search(m1.group(1))
    pm2 = _COORD_PAT.search(m2.group(1))
    if not pm1 or not pm2:
        return None, None

    x1_maj, y1_maj, x2_maj, y2_maj = [float(pm1.group(i)) for i in range(1, 5)]
    x1_min, y1_min, x2_min, y2_min = [float(pm2.group(i)) for i in range(1, 5)]

    def _to_array(x_rel, y_rel):
        return (H * (1 - y_rel), x_rel * W)

    major_axis_pts = [_to_array(x1_maj, y1_maj), _to_array(x2_maj, y2_maj)]
    minor_axis_pts = [_to_array(x1_min, y1_min), _to_array(x2_min, y2_min)]
    return major_axis_pts, minor_axis_pts


def load_nifti_slice(nii_path, slice_dim, slice_idx, doc):
    """Load NIfTI 2D slice at original resolution, normalized via normalize_img."""
    pixel_size_hw, img_2d = _load_nifti_2d(nii_path, slice_dim, slice_idx)
    img_2d = normalize_img(doc, img_2d)
    return pixel_size_hw, img_2d


def _load_json_gz(path):
    """Load JSON or gzip-compressed JSON file."""
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _extract_gt_axis_pts(doc):
    """
    Return (major_pts, minor_pts) as [(dim0, dim1), (dim0, dim1)] in array space,
    or (None, None) on failure.

    Reads the landmark JSON referenced by doc['landmark_file'] and projects P1–P4
    3D voxel coordinates onto the 2D slice plane.
    """
    lm_file = doc.get('landmark_file')
    if not lm_file or not os.path.exists(lm_file):
        return None, None
    try:
        lm_data = _load_json_gz(lm_file)
        sd = doc['slice_dim']
        si = doc['slice_idx']
        key = {0: 'slice_landmarks_x', 1: 'slice_landmarks_y', 2: 'slice_landmarks_z'}[sd]
        matched = [e for e in lm_data[key] if e.get('slice_idx') == si]
        if not matched:
            return None, None

        landmarks = {}
        for entry in matched:
            lms = entry.get('landmarks', {})
            if isinstance(lms, list):
                lms = lms[0]
            landmarks.update(lms)

        def _proj(c3d):
            # Extract (dim0, dim1) in array space by dropping the slice axis
            if sd == 0:
                return (c3d[1], c3d[2])
            if sd == 1:
                return (c3d[0], c3d[2])
            return (c3d[0], c3d[1])

        pts = {}
        for name in ('P1', 'P2', 'P3', 'P4'):
            c3d = landmarks.get(name)
            if c3d is None:
                return None, None
            pts[name] = _proj(c3d)

        return [pts['P1'], pts['P2']], [pts['P3'], pts['P4']]
    except Exception:
        return None, None


def process_model_dir(model_dir, task_folder, base_fig_dir, limit_per_jsonl, show_coords=False,
                      removed_samples_dir=None, removed_samples_filename=None):
    model_name = os.path.basename(model_dir.rstrip("/"))
    out_dir = os.path.join(base_fig_dir, task_folder, model_name)
    os.makedirs(out_dir, exist_ok=True)

    jsonl_files = sorted(glob.glob(os.path.join(model_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"  No JSONL files found in {model_dir}")
        return

    _removed_cache = {}  # dataset_name → frozenset | None

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

            image_hw = doc["image_size_2d"]  # [H, W] in array space
            slice_dim = doc["slice_dim"]
            slice_idx = doc["slice_idx"]
            pixel_sizes = doc["pixel_size"]  # [px_dim0, px_dim1]
            dataset_name = doc["dataset_name"]
            task_id = doc["taskID"]
            label = doc["label"]

            major_pts, minor_pts = parse_axis_coords(resp_text, image_hw)
            if major_pts is None:
                continue

            # Load image at original resolution
            img_path = doc["image_file"]

            # Skip samples removed in the updated dataset
            if removed_samples_dir:
                if dataset_name not in _removed_cache:
                    fn = removed_samples_filename
                    p = os.path.join(removed_samples_dir, dataset_name, fn)
                    _removed_cache[dataset_name] = _build_removed_set(p) if os.path.exists(p) else None
                removed_set = _removed_cache.get(dataset_name)
                if removed_set is not None:
                    _key = (_relative_image_file(img_path, dataset_name), slice_dim, slice_idx, int(task_id))
                    if _key in removed_set:
                        continue

            if not os.path.exists(img_path):
                print(f"    WARNING: image not found: {img_path}")
                continue
            try:
                pixel_sizes_from_nii, img_2d = load_nifti_slice(
                    img_path, slice_dim, slice_idx, doc
                )
            except Exception as e:
                print(f"    WARNING: failed to load image {img_path}: {e}")
                continue

            # Load mask (optional)
            mask_2d = None
            mask_path = doc.get("mask_file")
            if mask_path and os.path.exists(mask_path):
                try:
                    _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx)
                    mask_2d = (mask_2d == label).astype(np.float32)
                except Exception as e:
                    print(f"    WARNING: failed to load mask {mask_path}: {e}")

            # Extract GT axis landmarks from the doc
            gt_major_pts, gt_minor_pts = _extract_gt_axis_pts(doc)

            fig_name = (
                f"{dataset_name}__Task{task_id}__doc{doc_id}"
                f"__dim{slice_dim}__idx{slice_idx}__label{label}.png"
            )
            fig_path = os.path.join(out_dir, dataset_name, fig_name)

            try:
                plot_tl_axes_on_image(
                    image_2d=img_2d,
                    pixel_sizes=pixel_sizes_from_nii,
                    major_axis_pts=major_pts,
                    minor_axis_pts=minor_pts,
                    slice_dim=slice_dim,
                    slice_idx=slice_idx,
                    fig_path=fig_path,
                    mask_2d=mask_2d,
                    show_coords=show_coords,
                    gt_major_pts=gt_major_pts,
                    gt_minor_pts=gt_minor_pts,
                )
            except Exception as e:
                print(f"    WARNING: plotting failed for doc {doc_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MedVision TL task predictions (model-predicted axes + GT axes + mask contour)"
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=None,
        help="Task directory containing model subdirectories (e.g. Results/MedVision-TL-v2-CoT/)",
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
    parser.add_argument(
        "--removed_samples_dir",
        type=str,
        default=None,
        help="Root directory containing per-dataset removed_samples JSON files "
             "(e.g. .../Data/Datasets). Matching samples are skipped.",
    )
    parser.add_argument(
        "--removed_samples_filename",
        type=str,
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Filename of the removed-samples JSON within each dataset subdirectory.",
    )
    args = parser.parse_args()

    if args.task_dir is None and args.model_dir is None:
        parser.error("Provide --task_dir or --model_dir")

    if args.task_dir is not None:
        task_folder = os.path.basename(args.task_dir.rstrip("/"))
        model_dirs = sorted(
            d
            for d in glob.glob(os.path.join(args.task_dir, "*/"))
            if os.path.isdir(d)
        )
        if not model_dirs:
            print(f"No model directories found in {args.task_dir}")
            return
        for model_dir in model_dirs:
            print(f"Model: {os.path.basename(model_dir.rstrip('/'))}")
            process_model_dir(
                model_dir, task_folder, args.fig_dir, args.limit_per_jsonl, args.show_coords,
                removed_samples_dir=args.removed_samples_dir,
                removed_samples_filename=args.removed_samples_filename,
            )
    else:
        # Single model dir: infer task_folder from parent directory name
        model_dir = args.model_dir.rstrip("/")
        task_folder = os.path.basename(os.path.dirname(model_dir))
        print(f"Model: {os.path.basename(model_dir)}")
        process_model_dir(
            model_dir, task_folder, args.fig_dir, args.limit_per_jsonl, args.show_coords,
            removed_samples_dir=args.removed_samples_dir,
            removed_samples_filename=args.removed_samples_filename,
        )


if __name__ == "__main__":
    main()
