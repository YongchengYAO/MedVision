import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from medvision_bm.utils.configs import (
    RANDOM_BOX_SIMULATIONS,
    SEED,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES,
)
from medvision_bm.utils.parse_utils import (
    cal_F1,
    cal_IoU,
    cal_Precision,
    cal_Recall,
    convert_numpy_to_python,
    get_labelsMap_imgModality_from_seg_benchmark_plan,
    get_subfolders,
)


def _initialize_metric_counters_detection_task():
    """
    Initialize metric counters for detection task evaluation.

    Returns:
        Dictionary with counters for sums, counts, and threshold-based metrics
    """
    return {
        "sum_MAE": 0,
        "sum_IoU": 0,
        "sum_F1": 0,
        "sum_Precision": 0,
        "sum_Recall": 0,
        "num_success": 0,
        "count_valid_AE": 0,
        "count_valid_IoU": 0,
        "count_valid_F1": 0,
        "count_valid_Precision": 0,
        "count_valid_Recall": 0,
        "count_AE_thresholds": [0] * 10,
        "count_IoU_thresholds": [0] * 5,
        "count_F1_thresholds": [0] * 5,
        "count_Precision_thresholds": [0] * 5,
        "count_Recall_thresholds": [0] * 5,
    }


def _update_mae_counters(mae_value, counters):
    """
    Update MAE-related counters with a new MAE value.

    Args:
        mae_value: Mean Absolute Error value
        counters: Dictionary of metric counters to update
    """
    if not np.isnan(mae_value):
        counters["sum_MAE"] += mae_value
        counters["count_valid_AE"] += 1

        # Determine threshold bin (0.0-0.1, 0.1-0.2, etc.)
        threshold_index = min(int(mae_value * 10), 9)
        counters["count_AE_thresholds"][threshold_index] += 1


def _update_threshold_counters(metric_value, threshold_counts):
    """
    Update threshold counters for overlap metrics (IoU, F1, Precision, Recall).

    Args:
        metric_value: Metric value to evaluate against thresholds
        threshold_counts: List of counts for each threshold level
    """
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i, threshold in enumerate(thresholds):
        if metric_value >= threshold:
            threshold_counts[i] += 1


def _update_metric_counters_detection_task(metrics_dict, counters):
    """
    Update all metric counters with a single sample's calculated metrics.

    Args:
        metrics_dict: Dictionary of calculated metrics for one sample
        counters: Dictionary of metric counters to update
    """
    # Update MAE
    _update_mae_counters(metrics_dict["avgMAE"]["MAE"], counters)

    # Update IoU
    if not np.isnan(metrics_dict["avgIoU"]["IoU"]):
        iou = metrics_dict["avgIoU"]["IoU"]
        counters["sum_IoU"] += iou
        counters["count_valid_IoU"] += 1
        _update_threshold_counters(iou, counters["count_IoU_thresholds"])

    # Update F1
    if not np.isnan(metrics_dict["F1"]["F1"]):
        f1 = metrics_dict["F1"]["F1"]
        counters["sum_F1"] += f1
        counters["count_valid_F1"] += 1
        _update_threshold_counters(f1, counters["count_F1_thresholds"])

    # Update Precision
    if not np.isnan(metrics_dict["Precision"]["Precision"]):
        precision = metrics_dict["Precision"]["Precision"]
        counters["sum_Precision"] += precision
        counters["count_valid_Precision"] += 1
        _update_threshold_counters(precision, counters["count_Precision_thresholds"])

    # Update Recall
    if not np.isnan(metrics_dict["Recall"]["Recall"]):
        recall = metrics_dict["Recall"]["Recall"]
        counters["sum_Recall"] += recall
        counters["count_valid_Recall"] += 1
        _update_threshold_counters(recall, counters["count_Recall_thresholds"])

    # Update success count
    counters["num_success"] += metrics_dict["SuccessRate"]["success"]


def _calculate_final_metrics_detection_task(counters, count_total):
    """
    Calculate final aggregate metrics from accumulated counters.

    Args:
        counters: Dictionary of accumulated metric counters
        count_total: Total number of samples processed

    Returns:
        Dictionary with final averaged metrics and threshold statistics
    """
    task_metrics = {
        "avgMAE": (
            counters["sum_MAE"] / counters["count_valid_AE"]
            if counters["count_valid_AE"] > 0
            else np.nan
        ),
        "IoU": (
            counters["sum_IoU"] / counters["count_valid_IoU"]
            if counters["count_valid_IoU"] > 0
            else np.nan
        ),
        "F1": (
            counters["sum_F1"] / counters["count_valid_F1"]
            if counters["count_valid_F1"] > 0
            else np.nan
        ),
        "Precision": (
            counters["sum_Precision"] / counters["count_valid_Precision"]
            if counters["count_valid_Precision"] > 0
            else np.nan
        ),
        "Recall": (
            counters["sum_Recall"] / counters["count_valid_Recall"]
            if counters["count_valid_Recall"] > 0
            else np.nan
        ),
        "SuccessRate": (
            counters["num_success"] / count_total if count_total > 0 else 0.0
        ),
        "num_samples": count_total,
    }

    for k in range(1, 11):
        cumulative_count = sum(counters["count_AE_thresholds"][0:k])
        task_metrics[f"MAE<{k/10:.1f}"] = (
            cumulative_count / count_total if count_total > 0 else 0.0
        )

    # Add threshold-based metrics for overlap measures
    # e.g., "IoU>0.5" means proportion of samples with IoU >= 0.5
    metric_names = ["IoU", "F1", "Precision", "Recall"]
    for metric_name in metric_names:
        threshold_key = f"count_{metric_name}_thresholds"
        for k in range(5, 10):
            threshold_value = k / 10
            count_at_threshold = counters[threshold_key][k - 5]
            task_metrics[f"{metric_name}>{threshold_value:.1f}"] = (
                count_at_threshold / count_total if count_total > 0 else 0.0
            )

    return task_metrics


def calculate_summary_metrics_per_anatomy_detection_task(grouped_data):
    """
    Calculate summary metrics for each anatomy group.

    Args:
        grouped_data: Dictionary with parent_class as keys and task_data as values

    Returns:
        Dictionary with summary metrics per parent class and task type
    """
    summary_metrics = {}

    for parent_class, data in grouped_data.items():
        if parent_class is None:
            continue

        summary_metrics[parent_class] = {}

        targets = data["targets"]
        responses = data["responses"]

        # Skip if targets or responses are empty
        if not targets or not responses:
            continue

        # Initialize counters
        counters = _initialize_metric_counters_detection_task()
        count_total = len(targets)

        # Process each target-response pair
        for target, response in zip(targets, responses):
            mock_results = {"filtered_resps": [response], "target": target}
            metrics_dict = cal_metrics_detection_task(mock_results)
            _update_metric_counters_detection_task(metrics_dict, counters)

        # Calculate and store final metrics
        task_metrics = _calculate_final_metrics_detection_task(counters, count_total)
        summary_metrics[parent_class] = task_metrics

    return summary_metrics


def _random_bboxes(
    image_size,
    num_boxes,
    min_box_size,
    max_box_size,
):
    """
    Simulate `num_boxes` random detections in an image.

    Args:
        image_size: (width, height) of the image.
        num_boxes: how many boxes to generate.
        min_box_size: (min_width, min_height) of each box.
        max_box_size: (max_width, max_height) of each box;
                      defaults to full image size.

    Returns:
        List of (x1, y1, x2, y2) with 0 ≤ x1 < x2 ≤ W, 0 ≤ y1 < y2 ≤ H.
    """
    W, H = image_size
    max_w, max_h = max_box_size if max_box_size else (W, H)
    boxes = []

    for _ in range(num_boxes):
        w = random.randint(min_box_size[0], min(max_w, W))
        h = random.randint(min_box_size[1], min(max_h, H))
        # NOTE: use relative coordinates (x1, y1, x2, y2) for the box
        x1 = random.randint(0, W - w)
        y1 = random.randint(0, H - h)
        x2 = x1 + w
        y2 = y1 + h
        boxes.append((x1 / W, y1 / H, x2 / W, y2 / H))

    return boxes


def simulate_random_detection(target, image_size, num=100):
    target_metrics = np.array(json.loads(target), dtype=np.float32)
    W = image_size[1]
    H = image_size[0]

    mean_absolute_error = 0
    IoU = 0
    f1 = 0
    precision = 0
    recall = 0
    for _ in tqdm(range(num), desc="Simulating random detections", leave=False):
        pred_coords = _random_bboxes(
            image_size=(W, H),
            num_boxes=1,
            min_box_size=(3, 3),
            max_box_size=(W, H),
        )[0]
        pred_metrics = np.array(pred_coords, dtype=np.float32)
        # Calculate metrics
        mean_absolute_error += np.mean(np.abs(pred_metrics - target_metrics))
        IoU += cal_IoU(pred_metrics, target_metrics)
        f1 += cal_F1(pred_metrics, target_metrics)
        precision += cal_Precision(pred_metrics, target_metrics)
        recall += cal_Recall(pred_metrics, target_metrics)
    # Average the metrics over the number of simulations
    mean_absolute_error /= num
    IoU /= num
    f1 /= num
    precision /= num
    recall /= num

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": True},
        "avgIoU": {"IoU": IoU},
        "F1": {"F1": f1},
        "Precision": {"Precision": precision},
        "Recall": {"Recall": recall},
        "SuccessRate": {"success": True},
    }


def calculate_summary_metrics_per_anatomy_detection_task_for_randomModel(
    grouped_data, num_simulations
):
    """
    Calculate summary metrics for each anatomy group.

    Args:
        grouped_data: Dictionary with parent_class as keys and task_data as values
        num_simulations: Number of random simulations per sample
    Returns:
        Dictionary with summary metrics per parent class and task type
    """
    summary_metrics = {}

    for parent_class, data in tqdm(
        grouped_data.items(),
        desc="Simulating random detection for a box/image ratio group",
    ):
        if parent_class is None:
            continue

        summary_metrics[parent_class] = {}

        targets = data["targets"]
        image_sizes = data["image_size_2d"]

        # Initialize counters
        counters = _initialize_metric_counters_detection_task()
        count_total = len(targets)

        # Process each target-response pair
        for target, image_size in zip(targets, image_sizes):
            # Simulate random detection per sample, return the average metrics
            metrics_dict = simulate_random_detection(
                target, image_size, num_simulations
            )
            _update_metric_counters_detection_task(metrics_dict, counters)

        # Calculate and store final metrics
        task_metrics = _calculate_final_metrics_detection_task(counters, count_total)
        summary_metrics[parent_class] = task_metrics

    return summary_metrics


def _find_boxcoordinate_jsonl_files(model_dir, parsed_dirname="parsed"):
    parsed_dir = os.path.join(model_dir, parsed_dirname)
    all_jsonl = glob.glob(os.path.join(parsed_dir, "*.jsonl"))
    return [f for f in all_jsonl if "_BoxCoordinate_" in os.path.basename(f)]


_GT_BINS = [
    (0.05, "Box/Image < 5%"),
    (0.10, "5% <= Box/Image < 10%"),
    (0.15, "10% <= Box/Image < 15%"),
    (0.20, "15% <= Box/Image < 20%"),
    (0.25, "20% <= Box/Image < 25%"),
    (0.30, "25% <= Box/Image < 30%"),
    (0.35, "30% <= Box/Image < 35%"),
    (0.40, "35% <= Box/Image < 40%"),
    (0.45, "40% <= Box/Image < 45%"),
    (0.50, "45% <= Box/Image < 50%"),
    (0.55, "50% <= Box/Image < 55%"),
    (0.60, "55% <= Box/Image < 60%"),
    (0.65, "60% <= Box/Image < 65%"),
    (0.70, "65% <= Box/Image < 70%"),
    (0.75, "70% <= Box/Image < 75%"),
    (0.80, "75% <= Box/Image < 80%"),
    (0.85, "80% <= Box/Image < 85%"),
    (0.90, "85% <= Box/Image < 90%"),
]


def _group_by_boxImgRatio(data):
    result = defaultdict(
        lambda: {
            "mae": [],
            "iou": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "success": [],
        }
    )
    for box_img_ratio, mae, iou, f1, precision, recall, success in data:
        label = "Box/Image >= 90%"
        for threshold, name in _GT_BINS:
            if box_img_ratio < threshold:
                label = name
                break
        result[label]["mae"].append(mae)
        result[label]["iou"].append(iou)
        result[label]["f1"].append(f1)
        result[label]["precision"].append(precision)
        result[label]["recall"].append(recall)
        result[label]["success"].append(success)
    return dict(result)


def calculate_summary_metrics_per_boxImgRatio(grouped_data):
    summary_metrics = {}
    for bin_label, data in grouped_data.items():
        if bin_label is None:
            continue
        f1s = data["f1"]
        if not f1s:
            continue
        counters = _initialize_metric_counters_detection_task()
        count_total = len(f1s)
        for mae, iou, f1, prec, rec, success in zip(
            data["mae"],
            data["iou"],
            f1s,
            data["precision"],
            data["recall"],
            data["success"],
        ):
            metrics_dict = {
                "avgMAE": {"MAE": mae, "success": success},
                "avgIoU": {"IoU": iou},
                "F1": {"F1": f1},
                "Precision": {"Precision": prec},
                "Recall": {"Recall": rec},
                "SuccessRate": {"success": success},
            }
            _update_metric_counters_detection_task(metrics_dict, counters)
        summary_metrics[bin_label] = _calculate_final_metrics_detection_task(
            counters, count_total
        )
    return summary_metrics


def _read_gt_for_random_baseline(jsonl_path, limit=None):
    """Read ground-truth targets and image sizes for random detection baseline.

    Only reads GT fields (target, image_size_2d, box_img_ratio); filtered_resps
    is not needed for the random simulation.
    """
    results = []
    match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    dataset_name = match.group(1)

    count = 0
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            if not data:
                continue
            doc = data.get("doc", {})
            task_id = int(doc.get("taskID"))
            target = data.get("target")
            image_size_2d = doc.get("image_size_2d")
            label = doc.get("label")
            if None in (label, task_id, target, image_size_2d):
                continue

            if "box_img_ratio" in data:
                box_img_ratio = data["box_img_ratio"]
            elif "bounding_boxes" in doc:
                dims = doc["bounding_boxes"]["dimensions"][0]
                box_img_ratio = (dims[0] * dims[1]) / (
                    image_size_2d[0] * image_size_2d[1]
                )
            else:
                coords = json.loads(target) if isinstance(target, str) else target
                box_img_ratio = abs(coords[2] - coords[0]) * abs(coords[3] - coords[1])

            labels_map, _ = get_labelsMap_imgModality_from_seg_benchmark_plan(
                dataset_name, task_id
            )
            if labels_map.get(str(label)):
                results.append((target, box_img_ratio, image_size_2d))

            count += 1
            if limit is not None and count >= limit:
                break

    return results


def _group_gt_by_boxImgRatio(data):
    """Group (target, box_img_ratio, image_size_2d) tuples by box/image ratio bin."""
    result = defaultdict(lambda: {"targets": [], "image_size_2d": []})
    for target, box_img_ratio, image_size_2d in data:
        label = "Box/Image >= 90%"
        for threshold, name in _GT_BINS:
            if box_img_ratio < threshold:
                label = name
                break
        result[label]["targets"].append(target)
        result[label]["image_size_2d"].append(image_size_2d)
    return dict(result)


def generate_random_detection_baseline(ref_model_parsed_dir, out_dir, limit=None):
    """Generate random detection baseline metrics grouped by box/image ratio.

    Args:
        ref_model_parsed_dir: Path to a model's parsed/ folder (used as GT source).
        out_dir: Parent directory; a random_detection/ subfolder is written here.
        limit: Maximum samples per JSONL file (None = all).
    """
    # Sort by dataset+task suffix (strip timestamp prefix) so RNG consumption
    # order is stable regardless of which reference model's directory is scanned.
    jsonl_files = sorted(
        (
            f
            for f in glob.glob(os.path.join(ref_model_parsed_dir, "*.jsonl"))
            if "_BoxCoordinate_" in os.path.basename(f)
        ),
        key=lambda f: re.sub(r"^\d{8}_\d{6}_", "", os.path.basename(f)),
    )
    if not jsonl_files:
        raise FileNotFoundError(
            f"No BoxCoordinate JSONL files in {ref_model_parsed_dir}"
        )

    all_data = []
    for jsonl_file in tqdm(jsonl_files, desc="Reading reference JSONL files"):
        all_data.extend(_read_gt_for_random_baseline(jsonl_file, limit))

    grouped_data = _group_gt_by_boxImgRatio(all_data)

    summary_metrics = (
        calculate_summary_metrics_per_anatomy_detection_task_for_randomModel(
            grouped_data, num_simulations=RANDOM_BOX_SIMULATIONS
        )
    )

    random_model_path = os.path.join(out_dir, "random_detection")
    os.makedirs(random_model_path, exist_ok=True)

    values_path = os.path.join(
        random_model_path, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES
    )
    with open(values_path, "w") as f:
        json.dump(convert_numpy_to_python(grouped_data), f, indent=2)
    print(f"Saved GT values to {values_path}")

    metrics_path = os.path.join(
        random_model_path, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS
    )
    with open(metrics_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved random detection metrics to {metrics_path}")


def process_jsonl_file_detection_task(jsonl_path, limit=None):
    """
    Parse a JSONL results file, reading pre-computed detection metrics.

    Requires JSONL records to contain F1, Precision, Recall, avgIoU, avgMAE,
    SuccessRate fields (written by parse_outputs.py).

    Returns:
        List of tuples: (box_img_ratio, mae, iou, f1, precision, recall, success)
    """
    results = []
    match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    dataset_name = match.group(1)

    count = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            if not data:
                continue

            doc = data.get("doc", {})
            task_id = int(doc.get("taskID"))
            target = data.get("target")
            image_size_2d = doc.get("image_size_2d")
            label = doc.get("label")
            if None in (label, task_id, target, image_size_2d):
                continue

            mae = data.get("avgMAE", {}).get("MAE")
            iou = data.get("avgIoU", {}).get("IoU")
            f1 = data.get("F1", {}).get("F1")
            precision = data.get("Precision", {}).get("Precision")
            recall = data.get("Recall", {}).get("Recall")
            success = data.get("SuccessRate", {}).get("success", False)
            if None in (mae, iou, f1, precision, recall):
                continue

            if "box_img_ratio" in data:
                box_img_ratio = data["box_img_ratio"]
            elif "bounding_boxes" in doc:
                dims = doc["bounding_boxes"]["dimensions"][0]
                box_img_ratio = (dims[0] * dims[1]) / (
                    image_size_2d[0] * image_size_2d[1]
                )
            else:
                coords = target if isinstance(target, list) else json.loads(target)
                box_img_ratio = abs(coords[2] - coords[0]) * abs(coords[3] - coords[1])

            labels_map, _ = get_labelsMap_imgModality_from_seg_benchmark_plan(
                dataset_name, task_id
            )
            if labels_map.get(str(label)):
                results.append(
                    (box_img_ratio, mae, iou, f1, precision, recall, success)
                )

            count += 1
            if limit is not None and count >= limit:
                break

    return results


def _process_wrapper(args):
    """Wrapper function for multiprocessing process_jsonl_file_detection_task."""
    return process_jsonl_file_detection_task(*args)


def process_parsed_file_in_model_folder(
    model_dir,
    limit=None,
    processes=None,
    parsed_dirname="parsed",
):
    """
    Process all JSONL files in a model's parsed folder and generate summary metrics.

    This function performs the complete pipeline:
    1. Finds all JSONL files in model_dir/<parsed_dirname>/
    2. Parses each file to extract detection data
    3. Groups data by anatomy-modality-slice combinations
    4. Calculates summary metrics per group
    5. Saves intermediate and final results as JSON files
    6. Generates anatomy vs tumor/lesion grouped metrics

    Args:
        model_dir: Path to the model folder
        limit: Maximum number of samples to process per file (None = all)
        processes (int, optional): Number of processes to use for parallel calculation.
        parsed_dirname: Name of the parsed-results subfolder inside model_dir.
    """
    # Find parsed JSONL files
    parsed_files_dir = os.path.join(model_dir, parsed_dirname)
    assert os.path.exists(
        parsed_files_dir
    ), f"Parsed files directory does not exist: {parsed_files_dir}"
    jsonl_files = _find_boxcoordinate_jsonl_files(model_dir, parsed_dirname)

    # Collect all data from the parsed JSONL files
    all_data = []

    if processes is not None and processes > 1:
        print(f"Processing JSONL files with {processes} processes...")
        items = [(jsonl_file, limit) for jsonl_file in jsonl_files]
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.imap_unordered(_process_wrapper, items)
            for file_data in tqdm(results, total=len(items), desc="Processing files"):
                all_data.extend(file_data)
    else:
        for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
            file_data = process_jsonl_file_detection_task(jsonl_file, limit)
            all_data.extend(file_data)

    # Early exit if no valid data found
    if not all_data:
        print(f"No valid data found in {parsed_files_dir}, skipping...")
        return

    grouped_data = _group_by_boxImgRatio(all_data)

    # Early exit if grouping failed
    if not grouped_data:
        print(f"No grouped data found for {parsed_files_dir}, skipping...")
        return

    summary_metrics = calculate_summary_metrics_per_boxImgRatio(grouped_data)

    # Save values JSON file
    output_path = os.path.join(
        parsed_files_dir, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES
    )
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(grouped_data), f, indent=2)
    print(f"Saved target and model-predicted values to {output_path}")

    # Save summary metrics JSON file
    output_path = os.path.join(
        parsed_files_dir, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS
    )
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved summary metrics to {output_path}")


def _process_task_directory(
    task_dir,
    limit,
    processes=None,
    skip_model_wo_parsed_files=False,
    parsed_dirname="parsed",
):
    """
    Process all model directories within a task directory.

    This is the main processing function for task-level analysis.
    It loops through all model folders, processes their results,
    and generates a final summary comparing all models.

    Args:
        task_dir: Path to task directory containing model folders
        limit: Maximum samples to process per file (None = all)
        processes (int, optional): Number of processes to use for parallel calculation
        skip_model_wo_parsed_files: Skip models without parsed folders
        parsed_dirname: Name of the parsed-results subfolder inside each model folder
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Exclude "random_detection" folder if it exists
    model_dirs = [d for d in model_dirs if os.path.basename(d) != "random_detection"]

    # Process each model directory
    for model_dir in model_dirs:
        # Skip models without parsed results if requested
        parsed_files_dir = os.path.join(model_dir, parsed_dirname)
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(
            model_dir, limit, processes=processes, parsed_dirname=parsed_dirname
        )

    print("\nGenerating random detection baseline...")
    # The baseline reads ground truth only, so any model folder carrying the
    # requested parsed-results subfolder works as the reference. Not every model
    # is re-parsed by every parser, so pick the first one that actually has it.
    ref_model_parsed_dir = next(
        (
            os.path.join(d, parsed_dirname)
            for d in model_dirs
            if os.path.isdir(os.path.join(d, parsed_dirname))
        ),
        None,
    )
    if ref_model_parsed_dir is None:
        raise FileNotFoundError(
            f"No model folder in {task_dir} contains a '{parsed_dirname}' subfolder; "
            "cannot generate the random detection baseline."
        )
    generate_random_detection_baseline(ref_model_parsed_dir, task_dir, limit)


def _process_single_model_directory(
    model_dir, limit, processes=None, parsed_dirname="parsed"
):
    """
    Process a single model directory.

    Args:
        model_dir: Path to the model directory
        limit: Maximum number of samples to process per file
        processes (int, optional): Number of processes to use for parallel calculation
        parsed_dirname: Name of the parsed-results subfolder inside model_dir
    """
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(
        model_dir, limit, processes=processes, parsed_dirname=parsed_dirname
    )


def main(**kwargs):
    """
    Main function to process model folders based on provided arguments.

    Args:
        task_dir: Path to task directory (mutually exclusive with model_dir/ref_model_dir)
        model_dir: Path to model directory (mutually exclusive with task_dir/ref_model_dir)
        ref_model_dir: Path to a parsed model folder for random baseline only
        out_dir: Output directory for random baseline (required with ref_model_dir)
        limit: Maximum number of samples to process per file
        skip_model_wo_parsed_files: Whether to skip model directories without parsed folders
        processes: Number of processes to use for parallel calculation
        parsed_dirname: Name of the parsed-results subfolder inside each model folder
    """
    random.seed(SEED)

    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    ref_model_dir = kwargs.get("ref_model_dir")
    out_dir = kwargs.get("out_dir")
    limit = kwargs.get("limit")
    skip_model_wo_parsed_files = kwargs.get("skip_model_wo_parsed_files", False)
    processes = kwargs.get("processes")
    parsed_dirname = kwargs.get("parsed_dirname") or "parsed"

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directories within this folder will be looped over."
        )
        _process_task_directory(
            task_dir,
            limit,
            processes=processes,
            skip_model_wo_parsed_files=skip_model_wo_parsed_files,
            parsed_dirname=parsed_dirname,
        )

    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all JSONL files within this directory."
        )
        _process_single_model_directory(
            model_dir, limit, processes=processes, parsed_dirname=parsed_dirname
        )

    elif ref_model_dir is not None:
        print(f"Generating random detection baseline from {ref_model_dir} → {out_dir}")
        generate_random_detection_baseline(ref_model_dir, out_dir, limit)

    else:
        raise ValueError(
            "One of --task_dir, --model_dir, or --ref_model_dir must be provided."
        )


def parse_args():
    """
    Parse command line arguments.

    Supports three modes:
    - Task mode (--task_dir): per-model box-size analysis + random baseline
    - Model mode (--model_dir): per-model box-size analysis for a single directory
    - Random-only mode (--ref_model_dir + --out_dir): random baseline only

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze detection task performance by bounding box size and generate "
            "a random detection baseline. Use --task_dir to process all models and "
            "generate the random baseline, or --ref_model_dir/--out_dir for the "
            "random baseline alone (same interface as simulate_random_detection.py)."
        )
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        help="Path to the task directory containing model result folders.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to a specific model directory containing JSONL files.",
    )
    parser.add_argument(
        "--ref_model_dir",
        type=str,
        help=(
            "Path to a model's parsed/ folder used as GT source for the random baseline. "
            "Requires --out_dir. Generates random_detection/ output only."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory for random baseline (used with --ref_model_dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process per JSONL file. If not set, processes all samples.",
    )
    parser.add_argument(
        "--parsed_dirname",
        type=str,
        default="parsed",
        help=(
            "Name of the parsed-results subfolder to read inside each model directory "
            "(e.g. 'parsed' for the regex parser, 'llm-parsed_gemma-4-31b' for the "
            "LLM-judge re-parse). Ignored in --ref_model_dir mode, which takes the "
            "folder path directly."
        ),
    )
    parser.add_argument(
        "--skip_model_wo_parsed_files",
        action="store_true",
        help="Skip model directories that don't have a parsed-results folder. Only valid with --task_dir.",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=None,
        help="Number of worker processes for metric calculation.",
    )

    args = parser.parse_args()

    modes = sum(
        [
            args.task_dir is not None,
            args.model_dir is not None,
            args.ref_model_dir is not None,
        ]
    )
    if modes == 0:
        parser.error(
            "One of --task_dir, --model_dir, or --ref_model_dir must be provided."
        )
    if modes > 1:
        parser.error(
            "--task_dir, --model_dir, and --ref_model_dir are mutually exclusive."
        )
    if args.ref_model_dir is not None and args.out_dir is None:
        parser.error("--out_dir is required when using --ref_model_dir.")
    if args.skip_model_wo_parsed_files and args.task_dir is None:
        parser.error("--skip_model_wo_parsed_files can only be used with --task_dir.")

    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
