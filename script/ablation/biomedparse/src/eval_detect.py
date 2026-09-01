import argparse
import glob
import json
import os
import re
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import find_objects, label
from tqdm import tqdm

from _paths import REPO_ROOT, add_medvision_to_path

add_medvision_to_path()

from medvision_bm.sft.sft_utils import _load_single_dataset
from medvision_bm.utils.configs import EXCLUDED_KEYS, TUMOR_LESION_GROUP_KEYS, label_map_regroup
from medvision_bm.utils.parse_utils import (
    cal_F1,
    cal_IoU,
    cal_Precision,
    cal_Recall,
    get_labelsMap_imgModality_from_seg_benchmark_plan,
)


def analyze_predictions(prediction_dir, output_dir):
    print(f"Analyzing predictions in: {prediction_dir}")

    # Matching files with pattern *pred_mask.nii.gz
    files = glob.glob(os.path.join(prediction_dir, "*pred_mask.nii.gz"))
    total_files = len(files)

    failure_files = []
    success_files = []

    if total_files == 0:
        print("WARNING: No *pred_mask.nii.gz files found in prediction directory.")

    print(f"Found {total_files} files. Processing...")

    for i, file_path in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_files} files...")

        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()

            # Check if it contains only 0
            if np.all(data == 0):
                failure_files.append(file_path)
            else:
                success_files.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\nAnalysis Results:")
    print(f"Total files: {total_files}")

    with open(
        os.path.join(
            output_dir, "eval_biomedparse_medvision_detect_failure_predictions.txt"
        ),
        "w",
    ) as f:
        f.write("\n".join(failure_files))
        print(
            f"Saved failure predictions to {os.path.join(output_dir, 'eval_biomedparse_medvision_detect_failure_predictions.txt')}"
        )

    with open(
        os.path.join(
            output_dir, "eval_biomedparse_medvision_detect_success_predictions.txt"
        ),
        "w",
    ) as f:
        f.write("\n".join(success_files))
        print(
            f"Saved success predictions to {os.path.join(output_dir, 'eval_biomedparse_medvision_detect_success_predictions.txt')}"
        )


def _find_bounding_boxes_2D(binary_mask, pixel_spacing):
    """
    Finds 2D bounding boxes for connected components in a binary mask.
    Args:
        binary_mask (np.ndarray): 2D binary mask array
        pixel_spacing (tuple): Physical spacing between pixels (dim1_spacing, dim2_spacing)
    Returns:
        list[dict]: List of bounding boxes, each containing:
            - min_coords: (dim1_min, dim2_min)
            - max_coords: (dim1_max, dim2_max)
            - center_coords: (dim1_center, dim2_center)
            - dimensions: (dim1_length, dim2_length) in pixels
            - sizes: (dim1_size, dim2_size) in physical units
    Raises:
        ValueError: If mask is empty or not 2D
    """
    # Input validation
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D array, got {binary_mask.ndim}D array")
    if binary_mask.sum() == 0:
        raise ValueError("Empty mask - no objects found")
    # Label connected components
    labeled_array, num_clusters = label(binary_mask)
    all_slices = find_objects(labeled_array)
    bboxes = []
    # Process each cluster
    for cluster_id in range(1, num_clusters + 1):
        # Create mask for this object
        cluster_mask = labeled_array == cluster_id
        # Get bounding box using find_objects
        slices = all_slices[cluster_id - 1]
        # Extract coordinates
        dim1_min, dim1_max = slices[0].start, slices[0].stop - 1
        dim2_min, dim2_max = slices[1].start, slices[1].stop - 1
        # Calculate center coordinates
        dim1_center = int((dim1_min + dim1_max) / 2)
        dim2_center = int((dim2_min + dim2_max) / 2)
        # Calculate dimensions
        dim1_length = dim1_max - dim1_min + 1
        dim2_length = dim2_max - dim2_min + 1
        bbox_info = {
            "min_coords": (int(dim1_min), int(dim2_min)),
            "max_coords": (int(dim1_max), int(dim2_max)),
            "center_coords": (dim1_center, dim2_center),
            "dimensions": (dim1_length, dim2_length),
            "sizes": (
                dim1_length * pixel_spacing[0],
                dim2_length * pixel_spacing[1],
            ),
            "mask_image_ratio": np.sum(cluster_mask) / np.prod(cluster_mask.shape),
        }
        bboxes.append(bbox_info)
    return bboxes


def load_hf_lookup(tasks_json):
    print(f"Loading HF test data from tasks: {tasks_json}")
    with open(tasks_json) as f:
        tasks_dict = json.load(f)

    # Build lookup per task so modality and label_name are available per row.
    # Key format: "{dataset_name}__{image_file_basename}__dim{slice_dim}__idx{slice_idx}__lbl{label}"
    lookup = {}
    total_rows = 0
    for task_key in tasks_dict:
        dataset_name = task_key.split("_BoxSize_")[0]
        task_id = int(re.search(r"Task(\d+)", task_key).group(1))
        labels_map, imgModality = get_labelsMap_imgModality_from_seg_benchmark_plan(dataset_name, task_id)

        config = task_key + "_Test"
        ds = _load_single_dataset(
            "YongchengYAO/MedVision",
            dataset_name=dataset_name,
            config=config,
            split="test",
            limit=None,
        )
        df = ds.to_pandas()
        print(f"  {config}: {len(df)} samples, modality={imgModality}")
        total_rows += len(df)

        for _, row in df.iterrows():
            img_basename = os.path.basename(row["image_file"]).replace(".nii.gz", "")
            key = f"{row['dataset_name']}__{img_basename}__dim{row['slice_dim']}__idx{row['slice_idx']}__lbl{row['label']}"
            label_name = labels_map.get(str(row["label"]), f"label_{row['label']}")
            lookup[key] = {
                "dataset_name": row["dataset_name"],
                "pixel_size": row["pixel_size"],
                "voxel_size": row["voxel_size"],
                "slice_dim": row["slice_dim"],
                "slice_idx": row["slice_idx"],
                "min_coords": row["bounding_boxes"]["min_coords"][0],
                "max_coords": row["bounding_boxes"]["max_coords"][0],
                "imgModality": imgModality,
                "label_name": label_name,
            }

    print(f"Total rows: {total_rows}. Built lookup with {len(lookup)} unique entries.")
    return lookup


def calculate_distance(p1, p2, voxel_sizes):
    p1_phys = np.array(p1) * np.array(voxel_sizes)
    p2_phys = np.array(p2) * np.array(voxel_sizes)
    return np.linalg.norm(p2_phys - p1_phys)


def plot_bbox_on_image(
    viz_dir, basename, img_2d, coords_gt, coords_model, label_name, pred_results
):
    # Draw the predicted and GT boxes on the image for visualization, save the figure in png under bbox_figures_detect
    os.makedirs(viz_dir, exist_ok=True)
    import matplotlib.patches as patches

    # Rotate image 90 degrees counter-clockwise
    # We need the original width to transform coordinates
    h_orig, w_orig = img_2d.shape[:2]
    img_2d = np.rot90(img_2d)

    def rotate_coords(coords, w_dim):
        # NOTE: coords are array indices: (height, width) for one point, the y and x below are image coordinates: y is height axis, x is width axis
        ymin, xmin, ymax, xmax = coords
        # Rotate coordinates 90 degrees CCW
        # (y, x) -> (w-1-x, y)
        # Therefore:
        # new_ymin comes from old xmax (because inverted axis)
        # new_ymax comes from old xmin
        # new_xmin comes from old ymin
        # new_xmax comes from old ymax
        new_ymin = w_dim - 1 - xmax
        new_ymax = w_dim - 1 - xmin
        new_xmin = ymin
        new_xmax = ymax
        return [new_ymin, new_xmin, new_ymax, new_xmax]

    coords_gt = rotate_coords(coords_gt, w_orig)
    coords_model = rotate_coords(coords_model, w_orig)

    # aspact ratio of the rotated image
    pixel_size = pred_results["pixel_size"]
    aspect_ratio = pixel_size[0] / pixel_size[1]

    fig, ax = plt.subplots(1)
    # Display image
    if img_2d.ndim == 2:
        ax.imshow(img_2d, cmap="gray", aspect=aspect_ratio)
    else:
        ax.imshow(img_2d, aspect=aspect_ratio)

    # GT Box (Green) - coords are [ymin, xmin, ymax, xmax]
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = coords_gt
    rect_gt = patches.Rectangle(
        (gt_x_min, gt_y_min),
        gt_x_max - gt_x_min,
        gt_y_max - gt_y_min,
        linewidth=2,
        edgecolor="g",
        facecolor="none",
        label="GT",
    )
    ax.add_patch(rect_gt)

    # Model Box (Red)
    pred_y_min, pred_x_min, pred_y_max, pred_x_max = coords_model
    rect_pred = patches.Rectangle(
        (pred_x_min, pred_y_min),
        pred_x_max - pred_x_min,
        pred_y_max - pred_y_min,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
        label="Pred",
    )
    ax.add_patch(rect_pred)

    ax.legend()
    ax.axis("off")
    plt.title(
        f"'{label_name}'\n(P: {pred_results['Precision']:.2f}, R: {pred_results['Recall']:.2f}, F1: {pred_results['F1']:.2f}, IoU: {pred_results['IoU']:.2f})"
    )
    plt.savefig(os.path.join(viz_dir, f"{basename}.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)


def _write_detect_summary_txt(grouped_metrics, output_dir):
    """Write a detection summary txt matching the format of print_summary_metrics."""
    model_name = os.path.basename(os.path.normpath(output_dir))
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY METRICS: Recall, Precision, and F1")
    lines.append("=" * 80)
    lines.append(f"\nModel: {model_name}")
    lines.append("-" * len(f"Model: {model_name}"))
    for group_key in ["anatomy", "T/L"]:
        if group_key not in grouped_metrics:
            continue
        mm = grouped_metrics[group_key]["mean_metrics"]
        regions = mm.get("num_regions", 0)
        samples = mm.get("total_samples", 0)
        if regions == 0:
            continue
        recall = mm.get("Recall", float("nan"))
        precision = mm.get("Precision", float("nan"))
        f1 = mm.get("F1", float("nan"))
        iou = mm.get("IoU", float("nan"))
        sr = mm.get("SuccessRate", float("nan"))
        iou_05 = mm.get("IoU>0.5", float("nan"))
        f1_05 = mm.get("F1>0.5", float("nan"))
        lines.append(
            f"  {group_key.upper():8} ({regions:2d} regions, {samples:4d} samples): "
            f"Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, IoU={iou:.3f}, "
            f"SuccessRate={sr:.3f}, IoU>0.5={iou_05:.3f}, F1>0.5={f1_05:.3f}"
        )
        region_keys = grouped_metrics[group_key].get("regions", [])
        for rk in region_keys:
            lines.append(f"    - {rk}")
    lines.append("\n" + "=" * 80)

    txt_path = os.path.join(output_dir, "summary_detection_task.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved detection summary to {txt_path}")


def _build_region_key(label_name, imgModality, slice_dim):
    """Map raw label → parent class and build the region key used in the VLM benchmark.

    Mirrors the logic in group_by_anatomy_modality_slice (parse_utils.py):
      key = label_map_regroup[label_name] + " @ " + modality + " (SliceType)"
    Returns (region_key, parent_class, label_group), or (None, None, None) if the
    label is unknown.
    """
    parent_class = label_map_regroup.get(label_name.lower())
    if parent_class is None:
        return None, None, None

    # Normalize modality strings to match the VLM benchmark convention
    modality_map = {"MRI": "MR", "ultrasound": "US", "X-ray": "XR", "PET": "PET", "CT": "CT", "MR": "MR"}
    modality = modality_map.get(imgModality, imgModality)

    slice_map = {0: "S", 1: "C", 2: "A"}
    slice_type = slice_map.get(slice_dim, "A")

    region_key = f"{parent_class} @ {modality} ({slice_type})"

    parent_lower = parent_class.lower()
    if any(kw in parent_lower for kw in EXCLUDED_KEYS):
        label_group = "miscellaneous"
    elif any(kw in parent_lower for kw in TUMOR_LESION_GROUP_KEYS):
        label_group = "tumor_lesion"
    else:
        label_group = "anatomy"

    return region_key, parent_class, label_group


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiomedParse Detection task predictions")
    parser.add_argument("--pred_dir", required=True,
                        help="Directory containing *_pred_mask.nii.gz files from inference")
    parser.add_argument("--npz_dir", required=True,
                        help="Directory containing prepared .npz test files")
    parser.add_argument("--tasks_json",
                        default=os.path.join(REPO_ROOT, "tasks_list", "tasks_MedVision-detect__train_SFT.json"),
                        help="Path to tasks list JSON (e.g. tasks_MedVision-detect__train_SFT.json)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for metric files (CSVs, JSONs, TXT lists, distribution plot)")
    parser.add_argument("--fig_dir", required=True,
                        help="Directory for per-sample bounding-box figures")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    npz_dir = args.npz_dir
    tasks_json = args.tasks_json
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    success_list_path = os.path.join(output_dir, "eval_biomedparse_medvision_detect_success_predictions.txt")
    failure_list_path = os.path.join(output_dir, "eval_biomedparse_medvision_detect_failure_predictions.txt")
    bbox_figure_dir = args.fig_dir
    os.makedirs(bbox_figure_dir, exist_ok=True)

    analyze_predictions(pred_dir, output_dir)

    # helper for pixel size lookup lookup
    img_info_lookup = load_hf_lookup(tasks_json)

    with open(success_list_path, "r") as f:
        success_files = [line.strip() for line in f.readlines() if line.strip()]

    with open(failure_list_path, "r") as f:
        failure_files = [line.strip() for line in f.readlines() if line.strip()]

    results = []

    for pred_path in tqdm(
        success_files, desc="Evaluating successful predictions for detection task"
    ):
        try:
            basename = os.path.basename(pred_path).replace("_pred_mask.nii.gz", "")
            npz_filename = f"{basename}.npz"
            npz_path = os.path.join(npz_dir, npz_filename)

            # read image from the imgs of npz
            npz_data = np.load(npz_path, allow_pickle=True)
            img_rgb = npz_data["imgs"]
            if 3 in img_rgb.shape:
                dim_idx = img_rgb.shape.index(3)
                img_2d = np.squeeze(np.take(img_rgb, 1, axis=dim_idx))
            else:
                img_2d = np.squeeze(img_rgb)
            H, W = img_2d.shape[:2]

            # read case info from npz
            text_prompts = npz_data["text_prompts"].item()
            label_name = [v for k, v in text_prompts.items() if k != "instance_label"][
                0
            ]
            slice_dim = img_info_lookup[basename]["slice_dim"]
            slice_idx = img_info_lookup[basename]["slice_idx"]
            pixel_sizes = img_info_lookup[basename]["pixel_size"]
            imgModality = img_info_lookup[basename]["imgModality"]

            # Build region key and classify using label_map_regroup (aligns with VLM benchmark)
            region_key, _, label_group = _build_region_key(label_name, imgModality, slice_dim)
            if region_key is None:
                print(f"Warning: label '{label_name}' not in label_map_regroup, skipping {basename}")
                continue

            # GT
            min_coords_gt = img_info_lookup[basename]["min_coords"]
            max_coords_gt = img_info_lookup[basename]["max_coords"]
            coords_gt = np.array(
                [min_coords_gt[0], min_coords_gt[1], max_coords_gt[0], max_coords_gt[1]]
            )

            # 3. Load mask 2D
            # Note: We need to load the nii.gz mask from pred_path
            # Since these are 2D slices saved as NIfTI, we can load them directly.
            mask_nii = nib.load(pred_path)
            mask_data = mask_nii.get_fdata()
            mask_rgb = np.squeeze(mask_data)

            # find the channel dimension where the size is 3
            if 3 in mask_rgb.shape:
                dim_idx = mask_rgb.shape.index(3)
                mask_2d = np.squeeze(np.take(mask_rgb, 1, axis=dim_idx))
            else:
                mask_2d = np.squeeze(mask_rgb)
            if mask_2d.ndim != 2:
                raise ValueError(
                    f"Expected 2D mask, got {mask_2d.ndim}D for file {basename}"
                )

            # NOTE: in the testing set, cases with multiple objects has been filtered out
            bboxes = _find_bounding_boxes_2D(mask_2d, pixel_sizes)
            min_coords_model = bboxes[0]["min_coords"]
            max_coords_model = bboxes[0]["max_coords"]
            coords_model = np.array(
                [
                    min_coords_model[0],
                    min_coords_model[1],
                    max_coords_model[0],
                    max_coords_model[1],
                ]
            )

            # Calculate metrics
            norm_factor = np.array([H, W, H, W], dtype=float)
            avgMAE = float(np.mean(np.abs(coords_model / norm_factor - coords_gt / norm_factor)))
            F1 = cal_F1(pred=coords_model, target=coords_gt)
            IoU = cal_IoU(pred=coords_model, target=coords_gt)
            Precision = cal_Precision(pred=coords_model, target=coords_gt)
            Recall = cal_Recall(pred=coords_model, target=coords_gt)

            pred_results = {
                "file": basename,
                "label_name": label_name,
                "region_key": region_key,
                "label_group": label_group,
                "coords_model": coords_model,
                "coords_gt": coords_gt,
                "slice_dim": int(slice_dim),
                "slice_idx": int(slice_idx),
                "pixel_size": pixel_sizes,
                "avgMAE": avgMAE,
                "F1": F1,
                "IoU": IoU,
                "Precision": Precision,
                "Recall": Recall,
            }
            results.append(pred_results)

            # Plot bounding boxes on image
            plot_bbox_on_image(
                bbox_figure_dir,
                basename,
                img_2d,
                coords_gt,
                coords_model,
                label_name,
                pred_results,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error processing {pred_path}: {e}")
            continue

    print(f"\nProcessed {len(results)} files.")

    failure_results = []
    print("Processing failure files to extract label groups...")
    for pred_path in tqdm(
        failure_files, desc="Evaluating failed predictions for label grouping"
    ):
        try:
            basename = os.path.basename(pred_path).replace("_pred_mask.nii.gz", "")
            if basename not in img_info_lookup:
                print(f"Warning: {basename} not found in lookup, skipping")
                continue

            info = img_info_lookup[basename]
            label_name = info["label_name"]
            imgModality = info["imgModality"]
            slice_dim = info["slice_dim"]

            # Build region key and classify using label_map_regroup (aligns with VLM benchmark)
            region_key, _, label_group = _build_region_key(label_name, imgModality, slice_dim)
            if region_key is None:
                print(f"Warning: label '{label_name}' not in label_map_regroup, skipping {basename}")
                continue

            failure_results.append(
                {
                    "file": basename,
                    "label_name": label_name,
                    "region_key": region_key,
                    "label_group": label_group,
                }
            )
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            print(f"Error processing failure file {pred_path}: {e}")
            continue

    print(f"\nProcessed {len(failure_results)} failure files.")

    # print summary statistics
    if results:
        F1_scores = [res["F1"] for res in results]
        IoUs = [res["IoU"] for res in results]
        Precisions = [res["Precision"] for res in results]
        Recalls = [res["Recall"] for res in results]
        print(
            f"F1 Score: Mean={np.mean(F1_scores):.4f}, Std={np.std(F1_scores):.4f}, Min={np.min(F1_scores):.4f}, Max={np.max(F1_scores):.4f}"
        )
        print(
            f"IoU: Mean={np.mean(IoUs):.4f}, Std={np.std(IoUs):.4f}, Min={np.min(IoUs):.4f}, Max={np.max(IoUs):.4f}"
        )
        print(
            f"Precision: Mean={np.mean(Precisions):.4f}, Std={np.std(Precisions):.4f}, Min={np.min(Precisions):.4f}, Max={np.max(Precisions):.4f}"
        )
        print(
            f"Recall: Mean={np.mean(Recalls):.4f}, Std={np.std(Recalls):.4f}, Min={np.min(Recalls):.4f}, Max={np.max(Recalls):.4f}"
        )

        # Group-level summary
        groups_to_eval = ["anatomy", "tumor_lesion", "miscellaneous"]
        group_summary_data = []

        for group in groups_to_eval:
            success_group = [res for res in results if res["label_group"] == group]
            failure_group = [
                res for res in failure_results if res["label_group"] == group
            ]

            n_success = len(success_group)
            n_failure = len(failure_group)
            n_total = n_success + n_failure
            success_rate = n_success / n_total if n_total > 0 else 0.0

            if n_total > 0:
                print(f"\nGroup: {group}")
                print(
                    f"  Total: {n_total}, Success: {n_success}, Failure: {n_failure}, Success Rate: {success_rate:.4f}"
                )

                group_stats = {
                    "label_group": group,
                    "total_count": n_total,
                    "success_count": n_success,
                    "failure_count": n_failure,
                    "success_rate": success_rate,
                    "F1_mean": 0,
                    "F1_std": 0,
                    "IoU_mean": 0,
                    "IoU_std": 0,
                    "IoU_gt_0.5_rate": 0,
                    "Precision_mean": 0,
                    "Precision_std": 0,
                    "Recall_mean": 0,
                    "Recall_std": 0,
                }

                if success_group:
                    F1_scores_g = [res["F1"] for res in success_group]
                    IoUs_g = [res["IoU"] for res in success_group]
                    IoU_gt_0_5_count = sum(1 for iou in IoUs_g if iou > 0.5)
                    IoU_gt_0_5_rate = IoU_gt_0_5_count / len(IoUs_g)

                    Precisions_g = [res["Precision"] for res in success_group]
                    Recalls_g = [res["Recall"] for res in success_group]

                    print(
                        f"  F1 Score: Mean={np.mean(F1_scores_g):.4f}, Std={np.std(F1_scores_g):.4f}, Min={np.min(F1_scores_g):.4f}, Max={np.max(F1_scores_g):.4f}"
                    )
                    print(
                        f"  IoU: Mean={np.mean(IoUs_g):.4f}, Std={np.std(IoUs_g):.4f}, Min={np.min(IoUs_g):.4f}, Max={np.max(IoUs_g):.4f}"
                    )
                    print(f"  IoU > 0.5 Rate: {IoU_gt_0_5_rate:.4f}")
                    print(
                        f"  Precision: Mean={np.mean(Precisions_g):.4f}, Std={np.std(Precisions_g):.4f}, Min={np.min(Precisions_g):.4f}, Max={np.max(Precisions_g):.4f}"
                    )
                    print(
                        f"  Recall: Mean={np.mean(Recalls_g):.4f}, Std={np.std(Recalls_g):.4f}, Min={np.min(Recalls_g):.4f}, Max={np.max(Recalls_g):.4f}"
                    )

                    group_stats.update(
                        {
                            "F1_mean": np.mean(F1_scores_g),
                            "F1_std": np.std(F1_scores_g),
                            "IoU_mean": np.mean(IoUs_g),
                            "IoU_std": np.std(IoUs_g),
                            "IoU_gt_0.5_rate": IoU_gt_0_5_rate,
                            "Precision_mean": np.mean(Precisions_g),
                            "Precision_std": np.std(Precisions_g),
                            "Recall_mean": np.mean(Recalls_g),
                            "Recall_std": np.std(Recalls_g),
                        }
                    )

                group_summary_data.append(group_stats)

        # Save group-level summary
        group_summary_csv = os.path.join(output_dir, "eval_biomedparse_medvision_detect_group_summary.csv")
        pd.DataFrame(group_summary_data).to_csv(group_summary_csv, index=False)
        print(f"Saved group summary to {group_summary_csv}")

        # Plot distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics_data = [
            (F1_scores, "F1 Score"),
            (IoUs, "IoU"),
            (Precisions, "Precision"),
            (Recalls, "Recall"),
        ]

        for ax, (data, title) in zip(axes.flatten(), metrics_data):
            ax.hist(data, bins=20, edgecolor="black", alpha=0.7)
            ax.set_title(f"{title} Distribution")
            ax.set_xlabel(title)
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        dist_plot_path = os.path.join(output_dir, "eval_biomedparse_medvision_detect_metrics_dist.png")
        plt.savefig(dist_plot_path)
        print(f"Saved metric distribution plot to {dist_plot_path}")
        plt.close()

    # Save the updated success and failure lists
    with open(success_list_path, "w") as f:
        f.write("\n".join(success_files))
    print(f"Updated success predictions saved to {success_list_path}")
    with open(failure_list_path, "w") as f:
        f.write("\n".join(failure_files))
    print(f"Updated failure predictions saved to {failure_list_path}")

    # Print the counts and percentages of success and failure files
    total_files = len(success_files) + len(failure_files)
    success_count = len(success_files)
    failure_count = len(failure_files)
    if total_files > 0:
        success_pct = (success_count / total_files) * 100
        failure_pct = (failure_count / total_files) * 100
        print("\nSummary of Prediction Results (updated):")
        print(f"Total files: {total_files}")
        print(f"Success files: {success_count} ({success_pct:.2f}%)")
        print(f"Failure files: {failure_count} ({failure_pct:.2f}%)")

    # Save results
    output_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, "eval_biomedparse_medvision_detect_results.csv")
    output_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    # Build MedVision-format summary JSONs

    success_by_label = defaultdict(list)
    failure_by_label = defaultdict(list)
    label_to_group = {}
    for r in results:
        success_by_label[r["region_key"]].append(r)
        label_to_group[r["region_key"]] = r["label_group"]
    for r in failure_results:
        failure_by_label[r["region_key"]].append(r)
        label_to_group[r["region_key"]] = r["label_group"]

    region_metrics = {}
    for lname in sorted(set(list(success_by_label) + list(failure_by_label))):
        s = success_by_label[lname]
        f = failure_by_label[lname]
        n_s, n_f = len(s), len(f)
        n_total = n_s + n_f

        # Overlap metrics: failures contribute 0
        all_iou = [r["IoU"] for r in s] + [0.0] * n_f
        all_f1 = [r["F1"] for r in s] + [0.0] * n_f
        all_p = [r["Precision"] for r in s] + [0.0] * n_f
        all_r = [r["Recall"] for r in s] + [0.0] * n_f
        # MAE: success cases only
        mae_vals = [r["avgMAE"] for r in s if np.isfinite(r["avgMAE"])]

        m = {
            "avgMAE": float(np.mean(mae_vals)) if mae_vals else None,
            "IoU": float(np.mean(all_iou)),
            "F1": float(np.mean(all_f1)),
            "Precision": float(np.mean(all_p)),
            "Recall": float(np.mean(all_r)),
            "SuccessRate": n_s / n_total,
            "num_samples": n_total,
        }
        # MAE cumulative thresholds
        mae_bins = [0] * 10
        for v in mae_vals:
            mae_bins[min(int(v * 10), 9)] += 1
        for k in range(1, 11):
            m[f"MAE<{k / 10:.1f}"] = sum(mae_bins[:k]) / n_total
        # Overlap thresholds
        for metric_name, vals in [("IoU", all_iou), ("F1", all_f1), ("Precision", all_p), ("Recall", all_r)]:
            for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
                m[f"{metric_name}>{thresh:.1f}"] = sum(1 for v in vals if v >= thresh) / n_total

        region_metrics[lname] = m

    detect_json_path = os.path.join(output_dir, "summary_metrics_detect_Task.json")
    with open(detect_json_path, "w") as f:
        json.dump(region_metrics, f, indent=2)
    print(f"Saved region metrics to {detect_json_path}")

    # Anatomy vs T/L grouped metrics
    MINIMUM_GROUP_SIZE = 50
    anatomy_data, tl_data = {}, {}
    for lname, m in region_metrics.items():
        grp = label_to_group.get(lname, "miscellaneous")
        if grp == "miscellaneous" or m["num_samples"] < MINIMUM_GROUP_SIZE:
            continue
        if grp == "tumor_lesion":
            tl_data[lname] = m
        else:
            anatomy_data[lname] = m

    def _group_mean(group_dict):
        if not group_dict:
            return {"total_samples": 0, "num_regions": 0}
        totals, weights = defaultdict(float), defaultdict(float)
        total_n = 0
        for m in group_dict.values():
            n = m["num_samples"]
            total_n += n
            for k, v in m.items():
                if k == "num_samples" or v is None:
                    continue
                totals[k] += v * n
                weights[k] += n
        result = {k: totals[k] / weights[k] for k in totals if weights[k] > 0}
        result["total_samples"] = total_n
        result["num_regions"] = len(group_dict)
        return result

    grouped_metrics = {
        "anatomy": {
            "mean_metrics": _group_mean(anatomy_data),
            "regions": sorted(anatomy_data.keys()),
            "detailed_data": anatomy_data,
        },
        "T/L": {
            "mean_metrics": _group_mean(tl_data),
            "regions": sorted(tl_data.keys()),
            "detailed_data": tl_data,
        },
    }

    grouped_json_path = os.path.join(output_dir, "summary_metrics_anatomy_vs_lesion_detect_Task.json")
    with open(grouped_json_path, "w") as f:
        json.dump(grouped_metrics, f, indent=2)
    print(f"Saved anatomy vs lesion metrics to {grouped_json_path}")
    _write_detect_summary_txt(grouped_metrics, output_dir)


if __name__ == "__main__":
    main()
