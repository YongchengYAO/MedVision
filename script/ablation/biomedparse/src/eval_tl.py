import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import find_objects, label
from tqdm import tqdm

from _paths import REPO_ROOT, add_medvision_to_path

add_medvision_to_path()

from medvision_bm.sft.sft_utils import _load_single_dataset
from medvision_bm.utils.configs import label_map_rename


def analyze_predictions(prediction_dir, output_dir, filter_dataset=None):
    print(f"Analyzing predictions in: {prediction_dir}")

    files = glob.glob(os.path.join(prediction_dir, "*pred_mask.nii.gz"))
    if filter_dataset:
        prefix = f"{filter_dataset}__"
        files = [f for f in files if os.path.basename(f).startswith(prefix)]
        print(f"Filtered to dataset '{filter_dataset}': {len(files)} files")
    total_files = len(files)

    if total_files == 0:
        print("No *pred_mask.nii.gz files found.")
        return

    failure_files = []
    success_files = []

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

    success_txt = os.path.join(output_dir, "eval_biomedparse_medvision_tl_success_predictions.txt")
    failure_txt = os.path.join(output_dir, "eval_biomedparse_medvision_tl_failure_predictions.txt")

    if filter_dataset:
        prefix = f"{filter_dataset}__"
        for txt_path, new_list in [(success_txt, success_files), (failure_txt, failure_files)]:
            existing = []
            if os.path.exists(txt_path):
                with open(txt_path) as fh:
                    existing = [ln.strip() for ln in fh if ln.strip()]
            kept = [e for e in existing if not os.path.basename(e).startswith(prefix)]
            with open(txt_path, "w") as fh:
                fh.write("\n".join(kept + new_list))
    else:
        with open(failure_txt, "w") as f:
            f.write("\n".join(failure_files))
        with open(success_txt, "w") as f:
            f.write("\n".join(success_files))

    print(f"Saved failure predictions to {failure_txt}")
    print(f"Saved success predictions to {success_txt}")


def _get_appropriate_scale(pixel_size, img_size, init_scale=10):
    """
    Calculate appropriate scale bar size in mm and pixels.
    Args:
        pixel_size (float): Size of one pixel in mm
        img_size (int): Smallest image dimension in pixels
        init_scale (int): Initial scale in mm (default 10mm)
    Returns:
        tuple: (scale_mm, scale_pixels) - Selected scale in mm and pixels
    """
    scales = [
        1,
        2,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]  # Standard scales in mm
    # Convert initial scale to pixels
    scale_pixels_num = int(init_scale / pixel_size)
    # Scale should be between 5% and 25% of smallest image dimension
    min_pixels = img_size * 0.05
    max_pixels = img_size * 0.25
    if scale_pixels_num < min_pixels:
        # Find next larger scale
        for scale in scales:
            if scale > init_scale:
                return _get_appropriate_scale(pixel_size, img_size, scale)
    elif scale_pixels_num > max_pixels:
        # Find next smaller scale
        for scale in reversed(scales):
            if scale < init_scale:
                return _get_appropriate_scale(pixel_size, img_size, scale)
    return init_scale, scale_pixels_num


def __plot_img_ellipse_landmarks(
    image_2d,
    pixel_sizes,
    valid_ellipses_info,
    slice_dim,
    slice_idx,
    case_id,
    landmarks_fig_dir,
):
    # Extract ellipse information
    valid_ellipses = valid_ellipses_info["ellipses"]
    valid_centers = valid_ellipses_info["centers"]
    valid_axes = valid_ellipses_info["axes"]
    valid_angles = valid_ellipses_info["angles"]
    valid_ROIs = valid_ellipses_info["ROIs"]
    valid_landmarks_coords = valid_ellipses_info["landmarks_coords"]
    colors = [
        "#4285F4",
        "#EA4335",
        "#FDB813",
        "#34A853",
    ]

    # Create visualization
    img_height, img_width = image_2d.shape
    aspect_ratio = img_width / img_height
    base_size = 10
    figsize = (
        (base_size * aspect_ratio, base_size)
        if aspect_ratio > 1
        else (base_size, base_size / aspect_ratio)
    )
    # Calculate aspect ratio based on pixel sizes
    aspect_ratio = pixel_sizes[1] / pixel_sizes[0]
    # Plot image and landmarks with correct aspect ratio
    plt.figure(figsize=figsize)
    plt.imshow(
        image_2d.T,
        cmap="gray",
        origin="lower",
        aspect=aspect_ratio,
        zorder=-1,
    )

    # Plot all valid ellipses and landmarks
    for i in range(len(valid_ellipses)):
        # Add ellipse
        ellipse_patch = plt.matplotlib.patches.Ellipse(
            xy=(valid_centers[i][1], valid_centers[i][0]),
            width=valid_axes[i][1],
            height=valid_axes[i][0],
            angle=-valid_angles[i],
            fill=False,
            color="red",
            linewidth=2,
            zorder=1,
        )
        plt.gca().add_patch(ellipse_patch)
        # Plot mask contour
        plt.contour(
            valid_ROIs[i].T,
            levels=[0.5],
            colors="#97D540",
            linewidths=2,
            origin="lower",
            zorder=0,
        )
        # Plot landmarks
        for j, (x, y) in enumerate(valid_landmarks_coords[i]):
            plt.scatter(
                x,
                y,
                color=colors[j],
                edgecolors="black",
                marker="o",
                s=60,
                linewidth=1,
                label=f"P{j+1}",
                zorder=2,
            )
        # Plot axes
        plt.plot(
            [
                valid_landmarks_coords[i][0][0],
                valid_landmarks_coords[i][1][0],
            ],
            [
                valid_landmarks_coords[i][0][1],
                valid_landmarks_coords[i][1][1],
            ],
            color="#F37020",
            linestyle="-",
            linewidth=2,
            label="major axis",
            zorder=3,
        )
        plt.plot(
            [
                valid_landmarks_coords[i][2][0],
                valid_landmarks_coords[i][3][0],
            ],
            [
                valid_landmarks_coords[i][2][1],
                valid_landmarks_coords[i][3][1],
            ],
            color="#FBBC05",
            linestyle="-",
            linewidth=2,
            label="minor axis",
            zorder=3,
        )
    # Add scale bar
    min_idx = np.argmin(image_2d.shape[:2])
    scale_mm, num_pixels_dim_min = _get_appropriate_scale(
        pixel_sizes[min_idx],
        image_2d.shape[min_idx],
        init_scale=10,
    )
    num_pixels_dim_max = int(scale_mm / pixel_sizes[1 - min_idx])
    if min_idx == 0:
        scale_pixels_dim0, scale_pixels_dim1 = (
            num_pixels_dim_min,
            num_pixels_dim_max,
        )
    else:
        scale_pixels_dim0, scale_pixels_dim1 = (
            num_pixels_dim_max,
            num_pixels_dim_min,
        )
    start_x, start_y = int(img_height * 0.05), int(img_width * 0.05)
    end_x, end_y = (
        start_x + scale_pixels_dim0,
        start_y + scale_pixels_dim1,
    )
    plt.plot([start_x, end_x], [start_y, start_y], "w-", linewidth=2)
    plt.plot([start_x, start_x], [start_y, end_y], "w-", linewidth=2)
    plt.text(
        end_x + img_height * 0.01,
        start_y,
        f"{scale_mm} mm",
        color="white",
        horizontalalignment="left",
    )
    # Set title and labels
    if slice_dim == 0:
        slice_filename = f"Sagittal_{slice_idx}.png"
        plt.xlabel("Anterior →", fontsize=14)
        plt.ylabel("Superior →", fontsize=14)
    elif slice_dim == 1:
        slice_filename = f"Coronal_{slice_idx}.png"
        plt.xlabel("Right →", fontsize=14)
        plt.ylabel("Superior →", fontsize=14)
    else:
        slice_filename = f"Axial_{slice_idx}.png"
        plt.xlabel("Right →", fontsize=14)
        plt.ylabel("Anterior →", fontsize=14)
    plt.tight_layout(pad=1.5, rect=[0.05, 0.05, 0.95, 0.95])
    # Save visualization
    case_fig_dir = os.path.join(landmarks_fig_dir, case_id)
    os.makedirs(case_fig_dir, exist_ok=True)
    plt.savefig(
        os.path.join(case_fig_dir, slice_filename),
        bbox_inches="tight",
    )
    plt.close()


def _find_scaled_bounding_boxes_2D(binary_mask, scale):
    # Input validation
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D array, got {binary_mask.ndim}D array")
    if binary_mask.sum() == 0:
        raise ValueError("Empty mask - no objects found")
    if scale <= 0:
        raise ValueError(f"Invalid scale value: {scale}. It must be positive.")
    # Label connected components
    labeled_array, num_objects = label(binary_mask)
    bboxes = []
    # Process each object
    objects_slices = find_objects(labeled_array)
    for slices in objects_slices:
        if slices is None:
            continue
        # Get original bounding box coordinates
        dim0_min, dim0_max = slices[0].start, slices[0].stop - 1
        dim1_min, dim1_max = slices[1].start, slices[1].stop - 1
        # Calculate center coordinates
        dim0_center = (dim0_min + dim0_max) / 2
        dim1_center = (dim1_min + dim1_max) / 2
        # Calculate original dimensions
        dim0_length = dim0_max - dim0_min + 1
        dim1_length = dim1_max - dim1_min + 1
        # Calculate enlarged dimensions
        dim0_length_scaled = int(dim0_length * scale)
        dim1_length_scaled = int(dim1_length * scale)
        # Calculate new min/max coordinates while keeping center fixed
        dim0_min_scaled = int(dim0_center - dim0_length_scaled / 2)
        dim0_max_scaled = int(dim0_center + dim0_length_scaled / 2)
        dim1_min_scaled = int(dim1_center - dim1_length_scaled / 2)
        dim1_max_scaled = int(dim1_center + dim1_length_scaled / 2)
        # Clip to image boundaries
        dim0_min_scaled = max(0, dim0_min_scaled)
        dim0_max_scaled = min(binary_mask.shape[0] - 1, dim0_max_scaled)
        dim1_min_scaled = max(0, dim1_min_scaled)
        dim1_max_scaled = min(binary_mask.shape[1] - 1, dim1_max_scaled)
        bbox_info = {
            "min_coords": (int(dim0_min_scaled), int(dim1_min_scaled)),
            "max_coords": (int(dim0_max_scaled), int(dim1_max_scaled)),
        }
        bboxes.append(bbox_info)
    return bboxes


# NOTE:
# The following function is adapted from medvision_ds.utils.benchmark_planner
# It has been modified for inference-time use:
# - Do not check if all landmarks are within the buffer zone between shrunk and enlarged boxes


def __fit_ellipses(mask_2d, cluster_size_threshold, pixel_sizes, slice_dim, slice_idx):
    # Find connected components and store them with sizes
    labeled_array, _ = label(mask_2d)
    sizes = np.bincount(labeled_array.ravel())[1:]
    # Store visualization info
    valid_ellipses = []
    valid_centers = []
    valid_axes = []
    valid_angles = []
    valid_landmarks_coords = []
    valid_ROIs = []
    # Sort clusters by size (largest to smallest)
    sorted_cluster_indices = np.argsort(-sizes)  # Negative for descending order
    # Loop through all clusters
    landmarks = []
    for cluster_idx in sorted_cluster_indices:
        cluster_label = cluster_idx + 1
        cluster_size = sizes[cluster_label - 1]
        if cluster_size < cluster_size_threshold:
            continue
        # Get mask for current cluster
        mask_1ROI = (labeled_array == cluster_label).astype(np.uint8)
        # Fit ellipse to current cluster
        contours, _ = cv2.findContours(
            mask_1ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Convert contour points to real-world coordinates
        contour_real = contours[0].squeeze() * pixel_sizes
        # Fit ellipse in real-world coordinates
        ellipse_real = cv2.fitEllipse(contour_real.astype(np.float32))
        center_real, axes_real, angle = ellipse_real
        # Convert center back to pixel coordinates
        center = (
            center_real[0] / pixel_sizes[0],
            center_real[1] / pixel_sizes[1],
        )
        # Convert axes back to pixel coordinates while preserving aspect ratio
        axes = (
            axes_real[0] / pixel_sizes[0],
            axes_real[1] / pixel_sizes[1],
        )
        # Calculate ellipse points in pixel coordinates
        angle_rad = np.deg2rad(angle)
        a, b = axes[0] / 2, axes[1] / 2
        major_x = a * np.cos(angle_rad)
        major_y = a * np.sin(angle_rad)
        minor_x = -b * np.sin(angle_rad)
        minor_y = b * np.cos(angle_rad)
        # Calculate landmark coordinates in pixel space
        idx1_dim1 = center[0] + major_x
        idx1_dim0 = center[1] + major_y
        idx2_dim1 = center[0] - major_x
        idx2_dim0 = center[1] - major_y
        idx3_dim1 = center[0] + minor_x
        idx3_dim0 = center[1] + minor_y
        idx4_dim1 = center[0] - minor_x
        idx4_dim0 = center[1] - minor_y
        # Calculate axis lengths
        p1p2_length = np.sqrt(
            (idx1_dim0 - idx2_dim0) ** 2 + (idx1_dim1 - idx2_dim1) ** 2
        )
        p3p4_length = np.sqrt(
            (idx3_dim0 - idx4_dim0) ** 2 + (idx3_dim1 - idx4_dim1) ** 2
        )
        # Skip if either axis has zero length
        if p1p2_length < 1e-6 or p3p4_length < 1e-6:
            continue
        # Swap points if needed for consistency
        if p1p2_length < p3p4_length:
            idx1_dim0, idx3_dim0 = idx3_dim0, idx1_dim0
            idx1_dim1, idx3_dim1 = idx3_dim1, idx1_dim1
            idx2_dim0, idx4_dim0 = idx4_dim0, idx2_dim0
            idx2_dim1, idx4_dim1 = idx4_dim1, idx2_dim1
        # Reorder points based on index values
        if (idx1_dim0 < idx2_dim0) or (
            idx1_dim0 == idx2_dim0 and idx1_dim1 < idx2_dim1
        ):
            idx1_dim0, idx2_dim0 = idx2_dim0, idx1_dim0
            idx1_dim1, idx2_dim1 = idx2_dim1, idx1_dim1

        if (idx3_dim0 < idx4_dim0) or (
            idx3_dim0 == idx4_dim0 and idx3_dim1 < idx4_dim1
        ):
            idx3_dim0, idx4_dim0 = idx4_dim0, idx3_dim0
            idx3_dim1, idx4_dim1 = idx4_dim1, idx3_dim1

        # Check if all landmarks are within the buffer zone between shrunk and enlarged boxes
        points = [
            (idx1_dim0, idx1_dim1),
            (idx2_dim0, idx2_dim1),
            (idx3_dim0, idx3_dim1),
            (idx4_dim0, idx4_dim1),
        ]

        # Create landmark dictionary
        landmark_dict = {}
        if slice_dim == 0:
            landmark_dict = {
                "P1": [
                    int(slice_idx),
                    int(round(idx1_dim0)),
                    int(round(idx1_dim1)),
                ],
                "P2": [
                    int(slice_idx),
                    int(round(idx2_dim0)),
                    int(round(idx2_dim1)),
                ],
                "P3": [
                    int(slice_idx),
                    int(round(idx3_dim0)),
                    int(round(idx3_dim1)),
                ],
                "P4": [
                    int(slice_idx),
                    int(round(idx4_dim0)),
                    int(round(idx4_dim1)),
                ],
                "ROI_pixels_count": int(cluster_size),
            }
        elif slice_dim == 1:
            landmark_dict = {
                "P1": [
                    int(round(idx1_dim0)),
                    int(slice_idx),
                    int(round(idx1_dim1)),
                ],
                "P2": [
                    int(round(idx2_dim0)),
                    int(slice_idx),
                    int(round(idx2_dim1)),
                ],
                "P3": [
                    int(round(idx3_dim0)),
                    int(slice_idx),
                    int(round(idx3_dim1)),
                ],
                "P4": [
                    int(round(idx4_dim0)),
                    int(slice_idx),
                    int(round(idx4_dim1)),
                ],
                "ROI_pixels_count": int(cluster_size),
            }
        else:
            landmark_dict = {
                "P1": [
                    int(round(idx1_dim0)),
                    int(round(idx1_dim1)),
                    int(slice_idx),
                ],
                "P2": [
                    int(round(idx2_dim0)),
                    int(round(idx2_dim1)),
                    int(slice_idx),
                ],
                "P3": [
                    int(round(idx3_dim0)),
                    int(round(idx3_dim1)),
                    int(slice_idx),
                ],
                "P4": [
                    int(round(idx4_dim0)),
                    int(round(idx4_dim1)),
                    int(slice_idx),
                ],
                "ROI_pixels_count": int(cluster_size),
            }
        landmarks.append(landmark_dict)
        # Store visualization info
        valid_ellipses.append(ellipse_real)
        valid_centers.append(center)
        valid_axes.append(axes)
        valid_angles.append(angle)
        valid_landmarks_coords.append(points)
        valid_ROIs.append(mask_1ROI)
    valid_ellipses_info = {
        "ellipses": valid_ellipses,
        "centers": valid_centers,
        "axes": valid_axes,
        "angles": valid_angles,
        "landmarks_coords": valid_landmarks_coords,
        "ROIs": valid_ROIs,
    }
    return landmarks, valid_ellipses_info


def load_hf_lookup(tasks_json):
    print(f"Loading HF test data from tasks: {tasks_json}")
    with open(tasks_json) as f:
        tasks_dict = json.load(f)

    all_dfs = []
    for task_key in tasks_dict:
        config = task_key + "_Test"
        ds = _load_single_dataset(
            "YongchengYAO/MedVision",
            dataset_name=task_key.split("_TumorLesionSize_")[0],
            config=config,
            split="test",
            limit=None,
        )
        all_dfs.append(ds.to_pandas())
        print(f"  {config}: {len(ds)} samples")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total: {len(df)} rows")

    # Filename format: "{dataset_name}__{image_file_basename}__dim{slice_dim}__idx{slice_idx}__lbl{label}.npz"
    lookup = {}
    for _, row in df.iterrows():
        img_basename = os.path.basename(row["image_file"]).replace(".nii.gz", "")
        key = f"{row['dataset_name']}__{img_basename}__dim{row['slice_dim']}__idx{row['slice_idx']}__lbl{row['label']}"
        lookup[key] = {
            "dataset_name": row["dataset_name"],
            "pixel_size": row["pixel_size"],
            "voxel_size": row["voxel_size"],
            "slice_dim": row["slice_dim"],
            "slice_idx": row["slice_idx"],
            "metric_value_major_axis": row["biometric_profile"]["metric_value_major_axis"],
            "metric_value_minor_axis": row["biometric_profile"]["metric_value_minor_axis"],
            "mask_file": row["mask_file"],
            "label": row["label"],
        }
    print(f"Built lookup with {len(lookup)} entries.")
    return lookup


def calculate_distance(p1, p2, voxel_sizes):
    p1_phys = np.array(p1) * np.array(voxel_sizes)
    p2_phys = np.array(p2) * np.array(voxel_sizes)
    return np.linalg.norm(p2_phys - p1_phys)


def plot_ellipse_on_image(
    fig_dir, basename, img_2d, mask_2d, valid_ellipses_info,
    pixel_sizes, major_gt, minor_gt, d1, d2, label_name, mae, mre,
    gt_mask_2d=None,
):
    os.makedirs(fig_dir, exist_ok=True)
    H_orig, W_orig = img_2d.shape[:2]  # original (dim0, dim1) sizes

    # Rotate 90° CCW for display.
    # np.rot90: img_2d[dim0, dim1] → rotated[W-1-dim1, dim0] → display (x=dim0, y=W-1-dim1)
    # After rotation: x-axis = dim0 (pixel_sizes[0]), y-axis = dim1 flipped (pixel_sizes[1])
    img_rot = np.rot90(img_2d)
    mask_rot = np.rot90(mask_2d)
    aspect_ratio = pixel_sizes[1] / pixel_sizes[0]  # y-physical / x-physical

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_rot, cmap="gray", aspect=aspect_ratio)
    ax.contour(mask_rot > 0, levels=[0.5], colors=["#97D540"], linewidths=2)

    gt_mask_rot = np.rot90(gt_mask_2d) if gt_mask_2d is not None else np.zeros_like(mask_rot, dtype=np.uint8)
    if gt_mask_rot.any():
        ax.contour(gt_mask_rot > 0, levels=[0.5], colors=["cyan"], linewidths=2)

    # Landmark transform: array (dim0, dim1) → display (x=dim0, y=W_orig-1-dim1)
    def _to_disp(pt):
        return pt[0], W_orig - 1 - pt[1]

    if valid_ellipses_info["landmarks_coords"]:
        lm = valid_ellipses_info["landmarks_coords"][0]
        P1, P2, P3, P4 = lm[0], lm[1], lm[2], lm[3]
        x1, y1 = _to_disp(P1)
        x2, y2 = _to_disp(P2)
        x3, y3 = _to_disp(P3)
        x4, y4 = _to_disp(P4)
        ax.plot([x1, x2], [y1, y2], color="#F37020", linewidth=2, label=f"major pred={d1:.1f}mm")
        ax.plot([x3, x4], [y3, y4], color="#FBBC05", linewidth=2, label=f"minor pred={d2:.1f}mm")
        for xp, yp in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
            ax.scatter(xp, yp, s=30, color="white", edgecolors="black", linewidths=0.5, zorder=3)

    # Adaptive L-shaped scale bar in lower-left corner.
    # Same physical length (scale_mm) shown along both x (dim0) and y (dim1) axes,
    # mirroring __plot_img_ellipse_landmarks: pick scale from the shorter dimension,
    # then express that same mm count in pixels for the other dimension.
    min_idx = np.argmin(img_2d.shape[:2])
    scale_mm, scale_px_min = _get_appropriate_scale(
        pixel_sizes[min_idx], img_2d.shape[min_idx], 10
    )
    scale_px_other = int(scale_mm / pixel_sizes[1 - min_idx])
    if min_idx == 0:
        scale_px_dim0, scale_px_dim1 = scale_px_min, scale_px_other
    else:
        scale_px_dim0, scale_px_dim1 = scale_px_other, scale_px_min
    # In rotated display: x = dim0 (0…H_orig-1), y = dim1 flipped (0=top, W_orig-1=bottom)
    sb_x = int(H_orig * 0.05)
    sb_y = int(W_orig * 0.88)
    ax.plot([sb_x, sb_x + scale_px_dim0], [sb_y, sb_y],
            color="white", linewidth=3, solid_capstyle="butt", zorder=4)
    ax.plot([sb_x, sb_x], [sb_y, sb_y - scale_px_dim1],
            color="white", linewidth=3, solid_capstyle="butt", zorder=4)
    ax.text(sb_x + scale_px_dim0 + int(H_orig * 0.01), sb_y,
            f"{scale_mm} mm", ha="left", va="center",
            color="white", fontsize=14, fontweight="bold", zorder=4)

    handles, leg_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#97D540", linewidth=2))
    leg_labels.append("Pred mask")
    if gt_mask_rot.any():
        handles.append(Line2D([0], [0], color="cyan", linewidth=2))
        leg_labels.append("GT mask")
    ax.legend(handles=handles, labels=leg_labels, fontsize=14, loc="upper right")
    ax.axis("off")
    ax.set_title(
        f"'{label_name}'\n"
        f"GT: major={major_gt:.1f}mm, minor={minor_gt:.1f}mm\n"
        f"Pred: major={d1:.1f}mm, minor={d2:.1f}mm  |  MAE={mae:.2f}mm, MRE={mre:.3f}",
        fontsize=16,
    )
    plt.tight_layout()
    mre_bucket = min(int(mre / 0.1) + 1, 9)
    save_dir = os.path.join(fig_dir, f"MRE0{mre_bucket}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{basename}.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)


def _write_tl_summary_txt(region_metrics, output_dir):
    """Apply label_map_rename to merge synonymous labels, then write a summary txt."""
    renamed = {}
    for lname, m in region_metrics.items():
        new_name = label_map_rename.get(lname, lname)
        n = m["num_samples"]
        if new_name not in renamed:
            renamed[new_name] = {**m}
        else:
            prev = renamed[new_name]
            prev_n = prev["num_samples"]
            total = prev_n + n

            def wavg(a, b):
                if a is None and b is None:
                    return None
                if a is None:
                    return b
                if b is None:
                    return a
                return (a * prev_n + b * n) / total

            renamed[new_name]["avgMAE"] = wavg(prev["avgMAE"], m["avgMAE"])
            renamed[new_name]["avgMRE"] = wavg(prev["avgMRE"], m["avgMRE"])
            renamed[new_name]["SuccessRate"] = wavg(prev["SuccessRate"], m["SuccessRate"])
            for k in ["MRE<0.1", "MRE<0.2", "MRE<0.3"]:
                renamed[new_name][k] = wavg(prev.get(k), m.get(k))
            renamed[new_name]["num_samples"] = total

    total_samples = sum(m["num_samples"] for m in renamed.values())
    w_mae = sum(m["avgMAE"] * m["num_samples"] for m in renamed.values() if m["avgMAE"] is not None) / total_samples
    w_mre = sum(m["avgMRE"] * m["num_samples"] for m in renamed.values() if m["avgMRE"] is not None) / total_samples
    w_sr = sum(m["SuccessRate"] * m["num_samples"] for m in renamed.values()) / total_samples

    def _wmre_k(key):
        wsum = sum(m[key] * m["num_samples"] for m in renamed.values() if m.get(key) is not None)
        wn = sum(m["num_samples"] for m in renamed.values() if m.get(key) is not None)
        return wsum / wn if wn > 0 else None

    w_re01 = _wmre_k("MRE<0.1")
    w_re02 = _wmre_k("MRE<0.2")
    w_re03 = _wmre_k("MRE<0.3")

    sorted_labels = sorted(renamed.items(), key=lambda x: x[1]["num_samples"], reverse=True)

    model_name = os.path.basename(os.path.normpath(output_dir))
    lines = []
    lines.append(f"\nModel: {model_name}")
    lines.append(
        f"Weighted Average MAE: {w_mae:.4f}, MRE: {w_mre:.4f}, SR: {w_sr:.4f}, "
        f"nMAE: N/A (Total Samples: {total_samples})"
    )
    acc_parts = []
    if w_re01 is not None:
        acc_parts.append(f"Weighted MRE<0.1: {w_re01:.4f}")
    if w_re02 is not None:
        acc_parts.append(f"Weighted MRE<0.2: {w_re02:.4f}")
    if w_re03 is not None:
        acc_parts.append(f"Weighted MRE<0.3: {w_re03:.4f}")
    lines.append(" | ".join(acc_parts) if acc_parts else "No MRE<k metrics")
    lines.append("")
    lines.append("Label-specific metrics:")
    lines.append(
        f"{'Label':<50}  | {'MAE':<8} | {'MRE':<8} | {'SR':<8} | {'nMAE':<8} | "
        f"{'MRE<0.1':<8} | {'MRE<0.2':<8} | {'MRE<0.3':<8} | {'Samples':<8}"
    )
    lines.append("-" * 146)
    for lbl, m in sorted_labels:
        mae = m.get("avgMAE")
        mre = m.get("avgMRE")
        re01 = m.get("MRE<0.1")
        re02 = m.get("MRE<0.2")
        re03 = m.get("MRE<0.3")
        lines.append(
            f"{lbl:<50}  | "
            f"{(mae if mae is not None else float('nan')):<8.4f} | "
            f"{(mre if mre is not None else float('nan')):<8.4f} | {m['SuccessRate']:<8.4f} | "
            f"{'N/A':<8} | "
            f"{(re01 if re01 is not None else float('nan')):<8.4f} | "
            f"{(re02 if re02 is not None else float('nan')):<8.4f} | "
            f"{(re03 if re03 is not None else float('nan')):<8.4f} | "
            f"{m['num_samples']:<8}"
        )
    lines.append("\n" + "=" * 100 + "\n")

    txt_path = os.path.join(output_dir, "summary_tl_task.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved TL summary to {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiomedParse Tumor/Lesion task predictions")
    parser.add_argument("--pred_dir", required=True,
                        help="Directory containing *_pred_mask.nii.gz files from inference")
    parser.add_argument("--npz_dir", required=True,
                        help="Directory containing prepared .npz test files")
    parser.add_argument("--tasks_json",
                        default=os.path.join(REPO_ROOT, "tasks_list", "tasks_MedVision-TL__train_SFT.json"),
                        help="Path to tasks list JSON (e.g. tasks_MedVision-TL__train_SFT.json)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for metric files (CSVs, JSONs, TXT lists, distribution plot)")
    parser.add_argument("--fig_dir", required=True,
                        help="Directory for per-sample ellipse figures (bucketed by MRE)")
    parser.add_argument("--filter_dataset", type=str, default=None,
                        help="If set, only evaluate samples from this dataset (e.g. KiPA22). "
                             "Results are merged back into existing output files.")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    npz_dir = args.npz_dir
    tasks_json = args.tasks_json
    output_dir = args.output_dir
    filter_dataset = args.filter_dataset
    filter_prefix = f"{filter_dataset}__" if filter_dataset else None
    os.makedirs(output_dir, exist_ok=True)

    success_list_path = os.path.join(output_dir, "eval_biomedparse_medvision_tl_success_predictions.txt")
    failure_list_path = os.path.join(output_dir, "eval_biomedparse_medvision_tl_failure_predictions.txt")
    ellipse_fig_dir = args.fig_dir
    os.makedirs(ellipse_fig_dir, exist_ok=True)

    analyze_predictions(pred_dir, output_dir, filter_dataset=filter_dataset)

    # helper for pixel size lookup lookup
    img_info_lookup = load_hf_lookup(tasks_json)

    with open(success_list_path, "r") as f:
        success_files = [line.strip() for line in f.readlines() if line.strip()]

    with open(failure_list_path, "r") as f:
        failure_files = [line.strip() for line in f.readlines() if line.strip()]

    if filter_dataset:
        success_files = [p for p in success_files if os.path.basename(p).startswith(filter_prefix)]
        failure_files = [p for p in failure_files if os.path.basename(p).startswith(filter_prefix)]
        # Remove stale figures for this dataset before regenerating them.
        # MRE is embedded in the filename, so a changed MRE produces a new name and the
        # old file is never overwritten — clean all matching figures from all MRE subfolders.
        stale_figs = glob.glob(os.path.join(ellipse_fig_dir, "**", f"{filter_prefix}*.png"), recursive=True)
        for fig_path in stale_figs:
            os.remove(fig_path)
        if stale_figs:
            print(f"Removed {len(stale_figs)} stale {filter_dataset} figures from {ellipse_fig_dir}")

    results = []

    for pred_path in tqdm(
        success_files, desc="Evaluating successful predictions for TL task"
    ):
        try:
            basename = os.path.basename(pred_path).replace("_pred_mask.nii.gz", "")
            npz_filename = f"{basename}.npz"
            npz_path = os.path.join(npz_dir, npz_filename)

            # read image and label name from npz
            npz_data = np.load(npz_path, allow_pickle=True)
            text_prompts = npz_data["text_prompts"].item()
            label_name = [v for k, v in text_prompts.items() if k != "instance_label"][0]
            img_rgb = npz_data["imgs"]
            if 3 in img_rgb.shape:
                dim_idx = img_rgb.shape.index(3)
                img_2d = np.squeeze(np.take(img_rgb, 1, axis=dim_idx))
            else:
                img_2d = np.squeeze(img_rgb)
            if img_2d.ndim != 2:
                raise ValueError(
                    f"Expected 2D image, got {img_2d.ndim}D for file {basename}"
                )

            slice_dim = img_info_lookup[basename]["slice_dim"]
            slice_idx = img_info_lookup[basename]["slice_idx"]
            pixel_sizes = img_info_lookup[basename]["pixel_size"]
            voxel_sizes = img_info_lookup[basename]["voxel_size"]

            mask_file = img_info_lookup[basename].get("mask_file")
            label_id = img_info_lookup[basename].get("label")
            gt_mask_2d = np.zeros(img_2d.shape, dtype=np.uint8)
            if mask_file and os.path.exists(mask_file):
                try:
                    gt_mask_data = nib.load(mask_file).get_fdata()
                    if slice_dim == 0:
                        gt_mask_slice = gt_mask_data[slice_idx, :, :]
                    elif slice_dim == 1:
                        gt_mask_slice = gt_mask_data[:, slice_idx, :]
                    else:
                        gt_mask_slice = gt_mask_data[:, :, slice_idx]
                    if label_id is not None:
                        gt_mask_2d = (gt_mask_slice == label_id).astype(np.uint8)
                        if not gt_mask_2d.any():
                            gt_mask_2d = (gt_mask_slice > 0).astype(np.uint8)
                    else:
                        gt_mask_2d = (gt_mask_slice > 0).astype(np.uint8)
                except Exception as e:
                    print(f"Warning: could not load GT mask for {basename}: {e}")

            # GT
            metric_value_major_axis = img_info_lookup[basename][
                "metric_value_major_axis"
            ]
            metric_value_minor_axis = img_info_lookup[basename][
                "metric_value_minor_axis"
            ]

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

            # 4. __fit_ellipses
            # cluster_size_threshold as per user
            # NOTE:
            # - In MedVision dataset construction stage, cluster_size_threshold is 200 pixels to filter small ROIs.
            # - In inference stage, we should not filter small ROIs. Set it to small value.
            cluster_size_threshold = 10  # Default/hardcoded

            landmarks, valid_ellipses_info = __fit_ellipses(
                mask_2d, cluster_size_threshold, pixel_sizes, slice_dim, slice_idx
            )

            if not landmarks:
                print(f"No valid ellipses for {basename}")
                # remove pred_path from success list and add to failure list
                success_files.remove(pred_path)
                failure_files.append(pred_path)
                continue

            # 5. Calculate distances for P1-P2 and P3-P4
            # Use the first landmark dict (largest cluster)
            # - Cases with multiple clusters are filtered out in data loading stage
            lm = landmarks[0]
            d1 = calculate_distance(lm["P1"], lm["P2"], voxel_sizes)
            d2 = calculate_distance(lm["P3"], lm["P4"], voxel_sizes)

            # Metrics: MAE, MRE
            major_gt = float(metric_value_major_axis[0])
            minor_gt = float(metric_value_minor_axis[0])

            mae = (abs(d1 - major_gt) + abs(d2 - minor_gt)) / 2.0
            mre = (
                abs(d1 - major_gt) / major_gt
                + abs(d2 - minor_gt) / minor_gt
            ) / 2.0

            mae = float(mae)
            mre = float(mre)

            pred_results = {
                "file": basename,
                "label_name": label_name,
                "major_axis_model": float(d1),
                "minor_axis_model": float(d2),
                "major_axis_gt": major_gt,
                "minor_axis_gt": minor_gt,
                "slice_dim": int(slice_dim),
                "slice_idx": int(slice_idx),
                "pixel_size": pixel_sizes,
                "mae": mae,
                "mre": mre,
            }
            results.append(pred_results)

            plot_ellipse_on_image(
                ellipse_fig_dir, basename, img_2d, mask_2d, valid_ellipses_info,
                pixel_sizes, major_gt, minor_gt, d1, d2, label_name, mae, mre,
                gt_mask_2d=gt_mask_2d,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error processing {pred_path}: {e}")
            continue

    print(f"\nProcessed {len(results)} files.")

    # print summary statistics
    summary_metrics = {}
    if results:
        maes = [res["mae"] for res in results]
        mres = [res["mre"] for res in results]
        mre_lt_0_1 = [mre for mre in mres if mre < 0.1]
        mre_lt_0_1_pct = len(mre_lt_0_1) / len(mres)
        mre_lt_0_2 = [mre for mre in mres if mre < 0.2]
        mre_lt_0_2_pct = len(mre_lt_0_2) / len(mres)
        mre_lt_0_3 = [mre for mre in mres if mre < 0.3]
        mre_lt_0_3_pct = len(mre_lt_0_3) / len(mres)

        mae_mean = np.mean(maes)
        mae_std = np.std(maes)
        mae_min = np.min(maes)
        mae_max = np.max(maes)

        mre_mean = np.mean(mres)
        mre_std = np.std(mres)
        mre_min = np.min(mres)
        mre_max = np.max(mres)

        print(
            f"MAE: Mean={mae_mean:.4f}, Std={mae_std:.4f}, Min={mae_min:.4f}, Max={mae_max:.4f}"
        )
        print(
            f"MRE: Mean={mre_mean:.4f}, Std={mre_std:.4f}, Min={mre_min:.4f}, Max={mre_max:.4f}"
        )
        print(f"MRE < 0.1: {mre_lt_0_1_pct:.2f}")
        print(f"MRE < 0.2: {mre_lt_0_2_pct:.2f}")
        print(f"MRE < 0.3: {mre_lt_0_3_pct:.2f}")

        summary_metrics = {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "mae_min": mae_min,
            "mae_max": mae_max,
            "mre_mean": mre_mean,
            "mre_std": mre_std,
            "mre_min": mre_min,
            "mre_max": mre_max,
            "mre_lt_0_1_pct": mre_lt_0_1_pct,
            "mre_lt_0_2_pct": mre_lt_0_2_pct,
            "mre_lt_0_3_pct": mre_lt_0_3_pct,
        }

        # Plot distribution of MAE and MRE in one figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(maes, bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[0].set_title("Distribution of MAE")
        axes[0].set_xlabel("MAE")
        axes[0].set_ylabel("Count")

        axes[1].hist(mres, bins=50, alpha=0.7, color="green", edgecolor="black")
        axes[1].set_title("Distribution of MRE")
        axes[1].set_xlabel("MRE")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        dist_path = os.path.join(output_dir, "eval_biomedparse_medvision_tl_metrics_dist.png")
        plt.savefig(dist_path)
        plt.close()
        print(f"Saved metrics distribution to {dist_path}")

    # Save the updated success and failure lists (merge when filter_dataset is set)
    if filter_dataset:
        for txt_path, updated_list in [(success_list_path, success_files), (failure_list_path, failure_files)]:
            existing = []
            if os.path.exists(txt_path):
                with open(txt_path) as fh:
                    existing = [ln.strip() for ln in fh if ln.strip()]
            kept = [e for e in existing if not os.path.basename(e).startswith(filter_prefix)]
            with open(txt_path, "w") as fh:
                fh.write("\n".join(kept + updated_list))
    else:
        with open(success_list_path, "w") as f:
            f.write("\n".join(success_files))
        with open(failure_list_path, "w") as f:
            f.write("\n".join(failure_files))
    print(f"Updated success predictions saved to {success_list_path}")
    print(f"Updated failure predictions saved to {failure_list_path}")

    # Print the counts and percentages of success and failure files
    total_files = len(success_files) + len(failure_files)
    success_count = len(success_files)
    failure_count = len(failure_files)
    success_pct = 0.0
    failure_pct = 0.0

    if total_files > 0:
        success_pct = (success_count / total_files) * 100
        failure_pct = (failure_count / total_files) * 100
        print("\nSummary of Prediction Results (updated):")
        print(f"Total files: {total_files}")
        print(f"Success files: {success_count} ({success_pct:.2f}%)")
        print(f"Failure files: {failure_count} ({failure_pct:.2f}%)")

    # Save summary metrics to CSV
    summary_metrics.update(
        {
            "total_files": total_files,
            "success_files": success_count,
            "success_pct": success_pct,
            "failure_files": failure_count,
            "failure_pct": failure_pct,
        }
    )
    summary_df = pd.DataFrame([summary_metrics])
    summary_csv = os.path.join(output_dir, "eval_biomedparse_medvision_tl_group_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics to {summary_csv}")

    # Save results (merge when filter_dataset is set)
    output_csv = os.path.join(output_dir, "eval_biomedparse_medvision_tl_results.csv")
    new_df = pd.DataFrame(results)
    if filter_dataset and os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        existing_df = existing_df[~existing_df["file"].str.startswith(filter_prefix)]
        output_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        output_df = new_df
    output_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    if filter_dataset:
        print(
            f"\nNote: group_summary.csv and per-label JSON not updated for "
            f"--filter_dataset={filter_dataset}. Run the full eval script to refresh aggregate stats."
        )
        print(f"Saved ellipse figures to {ellipse_fig_dir}/")
        return

    # Collect label names from failure files for per-label summary
    failure_results = []
    print("Processing failure files to extract label names...")
    for pred_path in tqdm(failure_files, desc="Processing failure files"):
        try:
            basename = os.path.basename(pred_path).replace("_pred_mask.nii.gz", "")
            npz_data = np.load(os.path.join(npz_dir, f"{basename}.npz"), allow_pickle=True)
            text_prompts = npz_data["text_prompts"].item()
            lname = [v for k, v in text_prompts.items() if k != "instance_label"][0]
            failure_results.append({"file": basename, "label_name": lname})
        except Exception as e:
            print(f"Error processing failure file {pred_path}: {e}")

    # Build per-label summary JSON (analogous to summary_metrics_detect_Task.json)
    success_by_label = defaultdict(list)
    failure_by_label = defaultdict(list)
    for r in results:
        success_by_label[r["label_name"]].append(r)
    for r in failure_results:
        failure_by_label[r["label_name"]].append(r)

    region_metrics = {}
    for lname in sorted(set(list(success_by_label) + list(failure_by_label))):
        s = success_by_label[lname]
        f = failure_by_label[lname]
        n_s, n_f = len(s), len(f)
        n_total = n_s + n_f
        mae_vals = [r["mae"] for r in s if np.isfinite(r["mae"])]
        mre_vals = [r["mre"] for r in s if np.isfinite(r["mre"])]
        m = {
            "avgMAE": float(np.mean(mae_vals)) if mae_vals else None,
            "avgMRE": float(np.mean(mre_vals)) if mre_vals else None,
            "SuccessRate": n_s / n_total,
            "num_samples": n_total,
        }
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            m[f"MRE<{thresh:.1f}"] = sum(1 for v in mre_vals if v < thresh) / n_total
        region_metrics[lname] = m

    tl_json_path = os.path.join(output_dir, "summary_metrics_tl_Task.json")
    with open(tl_json_path, "w") as f:
        json.dump(region_metrics, f, indent=2)
    print(f"Saved per-label metrics to {tl_json_path}")
    _write_tl_summary_txt(region_metrics, output_dir)
    print(f"Saved ellipse figures to {ellipse_fig_dir}/")


if __name__ == "__main__":
    main()
