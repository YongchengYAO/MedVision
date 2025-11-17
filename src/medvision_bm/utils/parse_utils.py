import importlib
import os
import re
from collections import defaultdict

import nibabel as nib
import numpy as np

from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE


def get_subfolders(task_dir):
    model_dirs = []
    for entry in os.scandir(task_dir):
        if entry.is_dir():
            model_dirs.append(entry.path)
    return model_dirs


def load_nifti_2d(img_path, slice_dim, slice_idx):
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


def cal_DSC(pred, target):
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

    # Calculate DSC (Dice Similarity Coefficient)
    # DSC = 2 * intersection / (area1 + area2)
    dsc = (
        (2.0 * intersection_area) / (pred_area + target_area)
        if (pred_area + target_area) > 0
        else np.nan
    )

    return dsc


def cal_Precision(pred, target):
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

    # Calculate Precision
    Precision = intersection_area / pred_area if pred_area > 0 else np.nan

    return Precision


def cal_Recall(pred, target):
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
    target_area = (target[2] - target[0]) * (target[3] - target[1])

    # Calculate Recall
    Recall = intersection_area / target_area if target_area > 0 else np.nan

    return Recall


def cal_metrics_detection_task(results):
    """
    Calculate detection task metrics from model predictions.

    Args:
        results: Dictionary containing 'filtered_resps' (predictions) and 'target' (ground truth)

    Returns:
        Dictionary with metrics: avgMAE, avgIoU, DSC, Precision, Recall, SuccessRate
    """
    pred = results["filtered_resps"][0]
    target_metrics = eval(results["target"])
    try:
        # Parse prediction string: split by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if len(pred_metrics) != 4:
            # Invalid prediction format: return 0 for overlap metrics instead of NaN
            # This ensures failed predictions are counted in averages (0% performance)
            # rather than excluded from calculations (which NaN would do)
            mean_absolute_error = np.nan
            IoU = 0
            dsc = 0
            precision = 0
            recall = 0
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            IoU = cal_IoU(pred_metrics, target_metrics)
            dsc = cal_DSC(pred_metrics, target_metrics)
            precision = cal_Precision(pred_metrics, target_metrics)
            recall = cal_Recall(pred_metrics, target_metrics)
            success = True
    except:
        # Exception during parsing: treat as failed prediction
        # Return 0 for overlap metrics to penalize failures in averages
        mean_absolute_error = np.nan
        IoU = 0
        dsc = 0
        precision = 0
        recall = 0
        success = False

    # Return dictionary keys match the "metric" field in task YAML configuration
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgIoU": {"IoU": IoU},
        "DSC": {"DSC": dsc},
        "Precision": {"Precision": precision},
        "Recall": {"Recall": recall},
        "SuccessRate": {"success": success},
    }


# NOTE: This function is used for metric calculation across different task types.
# NOTE: For Detection task (bounding box corner coordinate prediction), do not use relative error.
#       Use mean absolute error and IoU instead.
def cal_metrics(results, task_type):
    """
    Calculate metrics for different task types.

    Args:
        results: Dictionary containing 'filtered_resps' and 'target'
        task_type: Type of task - 'Detection', 'TL', or 'AD'

    Returns:
        Dictionary with calculated metrics
    """
    pred = results["filtered_resps"][0]
    target_metrics = np.array(eval(results["target"]))

    # Determine expected length based on task type
    if task_type == "Detection":
        expected_length = 4
    elif task_type == "TL":
        expected_length = 2
    elif task_type == "AD":
        expected_length = 1
    else:
        raise ValueError(
            f"Invalid task_type: {task_type}. Must be 'Detection', 'TL', or 'AD'"
        )

    try:
        # Split the results string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])

        if len(pred_metrics) != expected_length:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            IoU = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)

            if task_type == "Detection":
                IoU = cal_IoU(pred_metrics, target_metrics)
            else:
                mean_relative_error = np.mean(absolute_error / (target_metrics + 1e-15))

            success = True
    except:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        IoU = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    if task_type == "Detection":
        return {
            "avgMAE": {"MAE": mean_absolute_error, "success": success},
            "avgIoU": {"IoU": IoU},
            "SuccessRate": {"success": success},
        }
    else:  # TL or AD
        return {
            "avgMAE": {"MAE": mean_absolute_error, "success": success},
            "avgMRE": {"MRE": mean_relative_error, "success": success},
            "SuccessRate": {"success": success},
        }


def get_labelsMap_imgModality_from_seg_benchmark_plan(dataset_name, task_id):
    """
    Import benchmark_plan and get labels_map for the given dataset and task_id.

    Args:
        dataset_name: Name of the dataset
        task_id: Task ID (1-based)

    Returns:
        Labels map from the benchmark plan
    """
    try:
        package_name = DATASETS_NAME2PACKAGE[dataset_name]
        # Import the module dynamically
        module = importlib.import_module(
            f"medvision_ds.datasets.{package_name}.preprocess_segmentation"
        )

        # Get benchmark_plan and labels_map
        benchmark_plan = getattr(module, "benchmark_plan")
        assert benchmark_plan is not None, "benchmark_plan not found in the module"
        if (
            benchmark_plan
            and "tasks" in benchmark_plan
            and task_id > 0
            and task_id <= len(benchmark_plan["tasks"])
        ):
            imgModality = benchmark_plan["tasks"][task_id - 1].get("image_modality")
            labels_map = benchmark_plan["tasks"][task_id - 1].get("labels_map")
            return (labels_map, imgModality)
    except (ImportError, AttributeError, IndexError) as e:
        raise ValueError(
            f"Error loading benchmark plan for {dataset_name}, task {task_id}: {e}"
        )


def get_labelsMap_imgModality_from_biometry_benchmark_plan(dataset_name, task_id):
    """
    Import benchmark_plan and get labels_map for the given dataset and task_id.

    Args:
        dataset_name: Name of the dataset
        task_id: Task ID (1-based)

    Returns:
        Labels map from the benchmark plan
    """
    if dataset_name not in DATASETS_NAME2PACKAGE:
        return {}

    package_name = DATASETS_NAME2PACKAGE[dataset_name]
    try:
        # Import the module dynamically
        module = importlib.import_module(
            f"medvision_ds.datasets.{package_name}.preprocess_biometry"
        )

        # Get benchmark_plan and labels_map
        benchmark_plan = getattr(module, "benchmark_plan", None)
        if (
            benchmark_plan
            and "tasks" in benchmark_plan
            and task_id > 0
            and task_id <= len(benchmark_plan["tasks"])
        ):
            imgModality = benchmark_plan["tasks"][task_id - 1].get("image_modality")
            labels_map = benchmark_plan["tasks"][task_id - 1].get("labels_map")
            return (labels_map, imgModality)
    except (ImportError, AttributeError, IndexError) as e:
        raise ValueError(
            f"Error loading benchmark plan for {dataset_name}, task {task_id}: {e}"
        )


def get_targetLabel_imgModality_from_biometry_benchmark_plan(dataset_name, task_id):
    try:
        package_name = DATASETS_NAME2PACKAGE[dataset_name]
        # Import the module dynamically
        module = importlib.import_module(
            f"medvision_ds.datasets.{package_name}.preprocess_biometry"
        )

        # Get benchmark_plan and labels_map
        benchmark_plan = getattr(module, "benchmark_plan", None)
        if (
            benchmark_plan
            and "tasks" in benchmark_plan
            and task_id > 0
            and task_id <= len(benchmark_plan["tasks"])
        ):
            imgModality = benchmark_plan["tasks"][task_id - 1].get("image_modality")
            target_label = benchmark_plan["tasks"][task_id - 1].get("target_label")
            return (target_label, imgModality)
    except (ImportError, AttributeError, IndexError) as e:
        raise ValueError(
            f"Error loading benchmark plan for {dataset_name}, task {task_id}: {e}"
        )


def group_by_anatomy_modality_slice(data):
    from medvision_bm.utils.configs import label_map_regroup

    result = defaultdict(lambda: {"targets": [], "responses": []})

    for (
        imgModality,
        label_name,
        target,
        filtered_resps,
        _,
        slice_dim,
    ) in data:
        if label_name not in list(label_map_regroup.keys()):
            raise ValueError("" f"Label '{label_name}' not found in label_map_regroup")
        parent_class = label_map_regroup.get(label_name)
        # -------------
        if imgModality == "MRI":
            imgModality = "MR"
        elif imgModality == "CT":
            imgModality = "CT"
        elif imgModality == "ultrasound":
            imgModality = "US"
        elif imgModality == "X-ray":
            imgModality = "XR"
        elif imgModality == "PET":
            imgModality = "PET"
        # -------------
        if slice_dim == 0:
            slicetype = "S"
        elif slice_dim == 1:
            slicetype = "C"
        elif slice_dim == 2:
            slicetype = "A"
        else:
            raise ValueError(f"Unknown slice dimension: {slice_dim}")
        new_parent_class = parent_class + " @ " + imgModality + " " + f"({slicetype})"
        result[new_parent_class]["targets"].append(target)
        result[new_parent_class]["responses"].extend(filtered_resps)

    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in result.items()}


def group_by_label_modality_slice(data):
    from medvision_bm.utils.configs import label_map_rename

    result = defaultdict(lambda: {"targets": [], "responses": []})

    for (
        imgModality,
        label_name,
        target,
        filtered_resps,
        _,
        slice_dim,
    ) in data:
        if label_name not in list(label_map_rename.keys()):
            raise ValueError("" f"Label '{label_name}' not found in label_map_rename")
        new_label = label_map_rename.get(label_name)
        # -------------
        if imgModality == "MRI":
            imgModality = "MR"
        elif imgModality == "CT":
            imgModality = "CT"
        elif imgModality == "ultrasound":
            imgModality = "US"
        elif imgModality == "X-ray":
            imgModality = "XR"
        elif imgModality == "PET":
            imgModality = "PET"
        # -------------
        if slice_dim == 0:
            slicetype = "S"
        elif slice_dim == 1:
            slicetype = "C"
        elif slice_dim == 2:
            slicetype = "A"
        else:
            raise ValueError(f"Unknown slice dimension: {slice_dim}")
        new_parent_class = new_label + " @ " + imgModality + " " + f"({slicetype})"

        # TODO: debug
        result[new_parent_class]["targets"].append(target)
        result[new_parent_class]["responses"].extend(filtered_resps)

    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in result.items()}


def group_by_boxImgRatio(data):
    result = defaultdict(
        lambda: defaultdict(
            lambda: {"targets": [], "responses": [], "image_size_2d": []}
        )
    )

    # Define thresholds and their corresponding labels
    thresholds = [
        (0.05, "Box/Image < 5%"),
        (0.1, "5% <= Box/Image < 10%"),
        (0.15, "10% <= Box/Image < 15%"),
        (0.2, "15% <= Box/Image < 20%"),
        (0.25, "20% <= Box/Image < 25%"),
        (0.3, "25% <= Box/Image < 30%"),
        (0.35, "30% <= Box/Image < 35%"),
        (0.4, "35% <= Box/Image < 40%"),
        (0.45, "40% <= Box/Image < 45%"),
        (0.5, "45% <= Box/Image < 50%"),
        (0.55, "50% <= Box/Image < 55%"),
        (0.6, "55% <= Box/Image < 60%"),
        (0.65, "60% <= Box/Image < 65%"),
        (0.7, "65% <= Box/Image < 70%"),
        (0.75, "70% <= Box/Image < 75%"),
        (0.8, "75% <= Box/Image < 80%"),
        (0.85, "80% <= Box/Image < 85%"),
        (0.9, "85% <= Box/Image < 90%"),
    ]

    for task_type, _, target, filtered_resps, _, box_img_ratio, image_size_2d in data:
        # Find the appropriate bin for this box_img_ratio
        bin_label = "90% <= Box/Image"  # Default for values >= 0.9
        for threshold, label in thresholds:
            if box_img_ratio < threshold:
                bin_label = label
                break

        result[bin_label][task_type]["targets"].append(target)
        result[bin_label][task_type]["responses"].extend(filtered_resps)
        result[bin_label][task_type]["image_size_2d"].append(image_size_2d)

    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in result.items()}
