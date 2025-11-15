import re

import nibabel as nib
import numpy as np


def _load_nifti_2d(img_path, slice_dim, slice_idx):
    """Map function to load 2D slice from a 3D NIFTI images."""
    img_nib = nib.load(img_path)
    voxel_size = img_nib.header.get_zooms()
    image_3d = img_nib.get_fdata().astype("float32")
    if slice_dim == 0:
        image_2d = image_3d[slice_idx, :, :]
        pixel_size = voxel_size[1:3]
    elif slice_dim == 1:
        image_2d = image_3d[:, slice_idx, :]
        pixel_size = voxel_size[0:1] + voxel_size[2:3]
    elif slice_dim == 2:
        image_2d = image_3d[:, :, slice_idx]
        pixel_size = voxel_size[0:2]
    else:
        raise ValueError("slice_dim must be 0, 1 or 2")
    return (pixel_size, image_2d)


def extract_last_k_nums(text, k):
    # Find all numbers in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    # Return the last k numbers
    if len(numbers) < k:
        return ""
    return ",".join(numbers[-k:])


# Convert NumPy values to native Python types for JSON serialization
def convert_numpy_to_python(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


def cal_IoU(pred, target):
    # Ensure inputs are 1D numpy arrays with 4 numbers
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()

    if len(pred) != 4 or len(target) != 4:
        raise ValueError(
            "Both pred and target must be 1D arrays with exactly 4 numbers"
        )

    # Calculate intersection coordinates
    x1 = max(pred[0], target[0])  # max of lower_x values
    y1 = max(pred[1], target[1])  # max of lower_y values
    x2 = min(pred[2], target[2])  # min of upper_x values
    y2 = min(pred[3], target[3])  # min of upper_y values

    # Check if there is an intersection
    if x1 >= x2 or y1 >= y2:
        return 0.0  # No intersection

    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate areas of both bounding boxes
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    target_area = (target[2] - target[0]) * (target[3] - target[1])

    # Calculate union area
    union_area = pred_area + target_area - intersection_area

    # Return IoU
    return intersection_area / union_area if union_area > 0 else 0.0


# NOTE: This function is used specifically for bounding box corner coordinate prediction accuracy evaluation.
# NOTE: Do no use relative error for bounding box corner coordinate prediction evaluation.
#       Use mean absolute error and IoU instead.
def cal_metrics_detection_task(results):
    pred = results["filtered_resps"][0]
    target_metrics = np.array(eval(results["target"]))
    try:
        # Split the results string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip())
                                for part in prd_parts])
        if len(pred_metrics) != 4:
            mean_absolute_error = np.nan
            IoU = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            IoU = cal_IoU(pred_metrics, target_metrics)
            success = True
    except:
        mean_absolute_error = np.nan
        IoU = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgIoU": {"IoU": IoU},
        "SuccessRate": {"success": success},
    }


def cal_metrics_TL_task(results):
    pred = results["filtered_resps"][0]
    target_metrics = np.array(eval(results["target"]))
    try:
        # Split the results string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip())
                                for part in prd_parts])
        if len(pred_metrics) != 2:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(
                absolute_error / (target_metrics + 1e-15))
            success = True
    except:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgMRE": {"MRE": mean_relative_error, "success": success},
        "SuccessRate": {"success": success},
    }


def cal_metrics_AD_task(results):
    pred = results["filtered_resps"][0]
    target_metrics = np.array(eval(results["target"]))
    try:
        # Split the results string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip())
                                for part in prd_parts])
        if len(pred_metrics) != 1:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(
                absolute_error / (target_metrics + 1e-15))
            success = True
    except:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgMRE": {"MRE": mean_relative_error, "success": success},
        "SuccessRate": {"success": success},
    }
