"""Parsing and metric utilities for the MedVision benchmark.

This module collects helpers used when parsing model outputs and scoring them
against ground truth, including:

- Bounding-box overlap metrics (IoU, F1/Dice, Precision, Recall).
- Extraction of the last ``k`` numbers from free-form text, optionally scoped to
  an ``<answer>`` block.
- Loading a 2D slice (and its in-plane pixel spacing) from a 3D NIfTI volume.
- Converting NumPy values to native Python types for JSON serialization.
- Grouping parsed results by anatomy/label, imaging modality, slice orientation,
  or box-to-image area ratio for stratified reporting.
"""

import ast
import importlib
import json
import os
import re
from collections import defaultdict

import nibabel as nib
import numpy as np

from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE

# Matches optional sign, optional thousands separators, decimal part, and exponent.
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def get_subfolders(task_dir, models=None):
    """Return the paths of all immediate subdirectories of a directory.

    Args:
        task_dir: Directory to scan for subfolders.
        models: Optional iterable of directory BASENAMES to keep. ``None`` (the
            default) returns every subdirectory, which is the historical behaviour
            and what every existing caller gets.

            A results tree holds far more model directories than any one study
            reports on -- superseded ``_bugfix-*`` variants, training checkpoints,
            baselines like ``random_detection``. Measured 2026-08-15: 57 dirs under
            MedVision-TL-v2-CoT against an 18-model roster. Filtering here rather
            than at each call site is deliberate: a summarizer enumerates model dirs
            twice, once to process and once to aggregate, and two filters that
            disagree would produce a report covering a different set than was
            actually processed -- with nothing anywhere saying so.

    Returns:
        list[str]: Path of each immediate subdirectory (typically one per model),
        restricted to ``models`` when given.
    """
    model_dirs = []
    for entry in os.scandir(task_dir):
        if entry.is_dir():
            model_dirs.append(entry.path)
    if models is not None:
        want = set(models)
        model_dirs = [p for p in model_dirs if os.path.basename(p) in want]
    return model_dirs


def assert_resps_key(parsed_files_dir, jsonl_files, resps_key):
    """Abort if the requested prediction key is absent from the records.

    Every summarizer guards each record with ``... and filtered_resps is not None
    and ...``, so pointing one at a directory whose records use a different key
    drops EVERY record with no exception raised. The observed failure modes are
    both silent and both wrong:

    - ``summarize_detection_task`` finds no metrics file for the model, skips it,
      and writes a report listing zero models -- exiting 0.
    - ``summarize_TL_task`` / ``summarize_AD_task`` either raise a confusing
      ``FileNotFoundError`` or, on a re-run, cheerfully re-report the metrics file
      a previous run left behind.

    ``llm-parsed*/`` records carry ``LLM_filtered_resps`` and have the strict key
    removed, so this is reachable by simply forgetting ``--resps_key``. One record
    is probed before any file is read, making that mistake cost a millisecond
    instead of an hour and a wrong table.

    Presence is tested with ``in`` rather than ``.get(...) is not None``: "key
    absent" (the forgotten flag) and "key present but null" (a legitimate, if
    unused, data state) are different situations and only the first is an error.

    Args:
        parsed_files_dir: Directory being summarized, for the message.
        jsonl_files: Sample JSONL paths about to be read.
        resps_key: The record key the caller intends to read.

    Raises:
        SystemExit: If the first readable record lacks ``resps_key``.
    """
    for path in jsonl_files:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if resps_key in record:
                    return
                present = sorted(
                    k for k in record if "resps" in k or "filtered" in k
                )
                raise SystemExit(
                    f"\n[FATAL] {os.path.basename(path)} has no {resps_key!r} key.\n"
                    f"  directory   : {parsed_files_dir}\n"
                    f"  --resps_key : {resps_key!r}\n"
                    f"  keys present: {present}\n\n"
                    f"  llm-parsed*/ records carry 'LLM_filtered_resps' and have the\n"
                    f"  strict 'filtered_resps' key REMOVED. Add:\n"
                    f"      --resps_key LLM_filtered_resps\n"
                    f"  (or drop --parsed_dirname to summarize the published parsed/).\n"
                )
        # The first file with any readable record settles it; a directory that
        # mixes schemas is caught per-record by the summarizers themselves.
        return


def load_nifti_2d(img_path, slice_dim, slice_idx):
    """Load a single 2D slice and its in-plane pixel spacing from a 3D NIfTI image.

    Args:
        img_path: Path to the NIfTI (``.nii`` / ``.nii.gz``) file.
        slice_dim: Axis to slice along; must be ``0``, ``1`` or ``2``.
        slice_idx: Index of the slice to extract along ``slice_dim``.

    Returns:
        tuple: ``(pixel_size, image_2d)`` where ``pixel_size`` is the in-plane
        voxel spacing (the two voxel dimensions not sliced along) and ``image_2d``
        is the extracted 2D slice as a ``float32`` array.

    Raises:
        ValueError: If ``slice_dim`` is not ``0``, ``1`` or ``2``.
    """
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
    """Extract the last ``k`` numbers found in a text string.

    Numbers are matched with an internal regex that accepts an optional sign,
    optional thousands separators, a decimal part and an exponent. Thousands
    separators are stripped so the comma-joined result splits back into the same
    numbers downstream.

    Args:
        text: Text to search for numbers.
        k: Number of trailing numbers to return.

    Returns:
        str: A comma-separated string of the last ``k`` numbers, or an empty
        string if fewer than ``k`` numbers are present.
    """
    # Find all numbers in the text (strip thousands separators so the
    # comma-joined result splits back into the same numbers downstream)
    numbers = [m.replace(",", "") for m in _NUM_RE.findall(text)]

    # Return the last k numbers
    if len(numbers) < k:
        return ""
    return ",".join(numbers[-k:])


def extract_last_k_nums_within_answer_tag(text, k):
    """Extract the last ``k`` numbers found inside an ``<answer>...</answer>`` block.

    The content between the first ``<answer>`` and ``</answer>`` tags is searched
    for numbers (matching an optional sign, thousands separators, a decimal part and
    an exponent). Thousands separators are stripped so the comma-joined result splits
    back into the same numbers downstream.

    Args:
        text: Text expected to contain an ``<answer>`` block.
        k: Number of trailing numbers to return.

    Returns:
        str: A comma-separated string of the last ``k`` numbers within the answer
        block, or an empty string if no answer tag is found or it contains fewer
        than ``k`` numbers.
    """
    # Extract content within <answer> </answer> tags
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not match:
        return ""

    # Find all numbers within the answer tag (strip thousands separators so the
    # comma-joined result splits back into the same numbers downstream)
    numbers = [m.replace(",", "") for m in _NUM_RE.findall(match.group(1))]

    # Return the last k numbers
    if len(numbers) < k:
        return ""
    return ",".join(numbers[-k:])


# Convert NumPy values to native Python types for JSON serialization
def convert_numpy_to_python(obj):
    """Recursively convert NumPy values to native Python types for JSON serialization.

    Args:
        obj: Value to convert. May be a scalar, array, or a nested ``dict``,
            ``list`` or ``tuple`` containing such values.

    Returns:
        The input with ``np.float32`` scalars converted to ``float``, ``np.ndarray``
        converted to lists, and containers converted recursively. Values of other
        types are returned unchanged.
    """
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
    """Compute the Intersection over Union (IoU) of two axis-aligned boxes.

    Each box is normalized so its corners are ordered, tolerating inputs given as
    ``[xmax, xmin, ymax, ymin]`` (they are sorted into ``[xmin, xmax, ymin, ymax]``).

    Args:
        pred: Predicted box as 4 numbers ``[x1, y1, x2, y2]``.
        target: Ground-truth box as 4 numbers ``[x1, y1, x2, y2]``.

    Returns:
        float: IoU in ``[0.0, 1.0]``; ``0.0`` when the boxes do not overlap.

    Raises:
        ValueError: If either input does not contain exactly 4 numbers.
    """
    # Ensure inputs are 1D numpy arrays with 4 numbers
    pred = np.asarray(pred, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()

    if len(pred) != 4 or len(target) != 4:
        raise ValueError(
            "Both pred and target must be 1D arrays with exactly 4 numbers"
        )

    # Extract coordinates
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target

    # Normalize both boxes: to accommodate incorrect input order [xmax, xmin, ymax, ymin]
    # which will be sorted as if they were [xmin, xmax, ymin, ymax]
    px1, px2 = sorted([px1, px2])
    py1, py2 = sorted([py1, py2])
    tx1, tx2 = sorted([tx1, tx2])
    ty1, ty2 = sorted([ty1, ty2])

    # Calculate intersection coordinates
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    # Check if there is an intersection
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0  # No intersection

    # Calculate intersection area
    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    # Calculate areas of both bounding boxes
    pred_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)

    # Calculate union area
    union_area = pred_area + target_area - intersection_area

    # Return IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return min(iou, 1.0)


def cal_F1(pred, target):
    """Compute the F1 score (Dice similarity coefficient) of two axis-aligned boxes.

    F1 is ``2 * intersection / (pred_area + target_area)``. Each box is normalized so
    its corners are ordered before computing areas.

    Args:
        pred: Predicted box as 4 numbers ``[x1, y1, x2, y2]``.
        target: Ground-truth box as 4 numbers ``[x1, y1, x2, y2]``.

    Returns:
        float: F1 score clamped to ``[0.0, 1.0]``; ``0.0`` when the boxes do not
        overlap, or ``nan`` if both boxes have zero area.

    Raises:
        ValueError: If either input does not contain exactly 4 numbers.
    """
    # Ensure inputs are 1D numpy arrays with 4 numbers
    pred = np.asarray(pred, dtype=np.float64).flatten()
    target = np.asarray(target, dtype=np.float64).flatten()

    if len(pred) != 4 or len(target) != 4:
        raise ValueError(
            "Both pred and target must be 1D arrays with exactly 4 numbers"
        )

    # Extract coordinates
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target

    # Normalize both boxes
    px1, px2 = sorted([px1, px2])
    py1, py2 = sorted([py1, py2])
    tx1, tx2 = sorted([tx1, tx2])
    ty1, ty2 = sorted([ty1, ty2])

    # Calculate intersection coordinates
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    # Check if there is an intersection
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0  # No intersection

    # Calculate intersection area
    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    # Calculate areas of both bounding boxes
    pred_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)

    # Calculate F1 (Dice Similarity Coefficient)
    # F1 = 2 * intersection / (area1 + area2)
    denominator = pred_area + target_area
    f1 = (2.0 * intersection_area) / denominator if denominator > 0 else np.nan

    if not np.isnan(f1):
        f1 = min(f1, 1.0)

    return f1


def cal_Precision(pred, target):
    """Compute precision (intersection over predicted area) of two axis-aligned boxes.

    Each box is normalized so its corners are ordered before computing areas.

    Args:
        pred: Predicted box as 4 numbers ``[x1, y1, x2, y2]``.
        target: Ground-truth box as 4 numbers ``[x1, y1, x2, y2]``.

    Returns:
        float: Precision clamped to ``[0.0, 1.0]``; ``0.0`` when the boxes do not
        overlap, or ``nan`` if the predicted box has zero area.

    Raises:
        ValueError: If either input does not contain exactly 4 numbers.
    """
    # Ensure inputs are 1D numpy arrays with 4 numbers
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()

    if len(pred) != 4 or len(target) != 4:
        raise ValueError(
            "Both pred and target must be 1D arrays with exactly 4 numbers"
        )

    # Extract coordinates
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target

    # Normalize both boxes
    px1, px2 = sorted([px1, px2])
    py1, py2 = sorted([py1, py2])
    tx1, tx2 = sorted([tx1, tx2])
    ty1, ty2 = sorted([ty1, ty2])

    # Calculate intersection coordinates
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    # Check if there is an intersection
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0  # No intersection

    # Calculate intersection area
    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    # Calculate areas of both bounding boxes
    pred_area = (px2 - px1) * (py2 - py1)

    # Calculate Precision
    Precision = intersection_area / pred_area if pred_area > 0 else np.nan

    # Robustness clamp
    if not np.isnan(Precision):
        Precision = min(Precision, 1.0)

    return Precision


def cal_Recall(pred, target):
    """
    Calculates Recall with robustness fixes for floating point errors
    and invalid box checks.

    Args:
        pred: (list or np.array) [xmin, ymin, xmax, ymax]
        target: (list or np.array) [xmin, ymin, xmax, ymax]

    Returns:
        float: Recall value (0.0 to 1.0)
    """
    # Flatten and ensure numpy arrays
    pred = np.asarray(pred).flatten()
    target = np.asarray(target).flatten()

    if len(pred) != 4 or len(target) != 4:
        raise ValueError("Inputs must be 1D arrays with 4 elements.")

    # Extract coordinates
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target

    # Normalize both boxes: to accommodate incorrect input order [xmax, xmin, ymax, ymin]
    # which will be sorted as if they were [xmin, xmax, ymin, ymax]
    px1, px2 = sorted([px1, px2])
    py1, py2 = sorted([py1, py2])
    tx1, tx2 = sorted([tx1, tx2])
    ty1, ty2 = sorted([ty1, ty2])

    # Calculate Intersection
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    # Check for no overlap
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)

    # Calculate Target Area
    target_area = (tx2 - tx1) * (ty2 - ty1)

    # Calculate Recall
    if target_area <= 0:
        raise ValueError("Target box has non-positive area.")

    # Calculate Recall
    recall = intersection_area / target_area

    # CRITICAL FIX: Floating point clamping
    # Simple clip to handle precision errors (e.g. 1.000000000004 -> 1.0)
    return min(recall, 1.0)


def cal_metrics_detection_task(results):
    """
    Calculate detection task metrics from model predictions.

    Args:
        results: Dictionary containing 'filtered_resps' (predictions) and 'target' (ground truth)

    Returns:
        Dictionary with metrics: avgMAE, avgIoU, F1, Precision, Recall, SuccessRate
    """
    pred = results["filtered_resps"][0]
    target_metrics = ast.literal_eval(results["target"])
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
            f1 = 0
            precision = 0
            recall = 0
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            IoU = cal_IoU(pred_metrics, target_metrics)
            f1 = cal_F1(pred_metrics, target_metrics)
            precision = cal_Precision(pred_metrics, target_metrics)
            recall = cal_Recall(pred_metrics, target_metrics)
            success = True
    except Exception:
        # Exception during parsing: treat as failed prediction
        # Return 0 for overlap metrics to penalize failures in averages
        mean_absolute_error = np.nan
        IoU = 0
        f1 = 0
        precision = 0
        recall = 0
        success = False

    # Return dictionary keys match the "metric" field in task YAML configuration
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgIoU": {"IoU": IoU},
        "F1": {"F1": f1},
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
    # Detection shares ONE implementation with the authoritative summarize path
    # (cal_metrics_detection_task): overlap metrics (IoU/F1/P/R) count failures as 0,
    # not NaN. Delegating keeps parse_outputs and summarize_detection_task from ever
    # disagreeing on detection metrics.
    if task_type == "Detection":
        return cal_metrics_detection_task(results)

    pred = results["filtered_resps"][0]
    target_metrics = np.array(ast.literal_eval(results["target"]))

    # Determine expected length based on task type
    if task_type == "TL":
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
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(absolute_error / (target_metrics + 1e-15))
            success = True
    except Exception:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
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
    """Group parsed results by parent anatomy class, modality and slice orientation.

    Each label is mapped to its parent anatomy class via ``label_map_regroup``, the
    imaging modality is normalized to a short code (e.g. ``MRI`` to ``MR``,
    ``ultrasound`` to ``US``, ``X-ray`` to ``XR``), and the slice dimension is mapped
    to an orientation code (``0`` to ``S``, ``1`` to ``C``, ``2`` to ``A``). Results
    are keyed as ``"<parent> @ <modality> (<orientation>)"``.

    Args:
        data: Iterable of tuples
            ``(imgModality, label_name, target, filtered_resps, _, slice_dim)``.

    Returns:
        dict: Mapping of each group key to a dict with ``"targets"`` (list of targets)
        and ``"responses"`` (flattened list of responses).

    Raises:
        ValueError: If a label is missing from ``label_map_regroup`` or ``slice_dim``
            is not ``0``, ``1`` or ``2``.
    """
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
    """Group parsed results by renamed label, modality and slice orientation.

    Each label is renamed via ``label_map_rename``, the imaging modality is normalized
    to a short code (e.g. ``MRI`` to ``MR``, ``ultrasound`` to ``US``, ``X-ray`` to
    ``XR``), and the slice dimension is mapped to an orientation code (``0`` to ``S``,
    ``1`` to ``C``, ``2`` to ``A``). Results are keyed as
    ``"<label> @ <modality> (<orientation>)"``.

    Args:
        data: Iterable of tuples
            ``(imgModality, label_name, target, filtered_resps, _, slice_dim)``.

    Returns:
        dict: Mapping of each group key to a dict with ``"targets"`` (list of targets)
        and ``"responses"`` (flattened list of responses).

    Raises:
        ValueError: If a label is missing from ``label_map_rename`` or ``slice_dim``
            is not ``0``, ``1`` or ``2``.
    """
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
    """Group parsed results into bins by box-to-image area ratio.

    Each item is placed into a 5%-wide bin based on its box-to-image ratio, ranging
    from ``"Box/Image < 5%"`` up to ``"90% <= Box/Image"``.

    Args:
        data: Iterable of tuples
            ``(_, target, filtered_resps, _, box_img_ratio, image_size_2d)``.

    Returns:
        dict: Mapping of each bin label to a dict with ``"targets"`` (list of targets),
        ``"responses"`` (flattened list of responses) and ``"image_size_2d"`` (list of
        image sizes).
    """
    result = defaultdict(lambda: {"targets": [], "responses": [], "image_size_2d": []})

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

    for _, target, filtered_resps, _, box_img_ratio, image_size_2d in data:
        # Find the appropriate bin for this box_img_ratio
        bin_label = "90% <= Box/Image"  # Default for values >= 0.9
        for threshold, label in thresholds:
            if box_img_ratio < threshold:
                bin_label = label
                break

        result[bin_label]["targets"].append(target)
        result[bin_label]["responses"].extend(filtered_resps)
        result[bin_label]["image_size_2d"].append(image_size_2d)

    # Convert defaultdict to regular dict
    return {k: dict(v) for k, v in result.items()}
