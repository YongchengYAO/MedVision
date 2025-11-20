import argparse
import glob
import json
import os
import random
import re

import numpy as np
from tqdm import tqdm

from medvision_bm.utils.configs import (
    RANDOM_BOX_SIMULATIONS,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES,
)
from medvision_bm.utils.parse_utils import (
    cal_F1,
    cal_IoU,
    cal_metrics_detection_task,
    cal_Precision,
    cal_Recall,
    convert_numpy_to_python,
    get_labelsMap_imgModality_from_seg_benchmark_plan,
    get_subfolders,
    group_by_boxImgRatio,
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

    # Add cumulative MRE (Mean Relative Error) metrics
    # MRE<k means proportion of samples with MAE less than k/10
    for k in range(1, 11):
        cumulative_count = sum(counters["count_AE_thresholds"][0:k])
        task_metrics[f"MRE<{k/10:.1f}"] = (
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
        task_metrics = _calculate_final_metrics_detection_task(
            counters, count_total
        )
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
    target_metrics = eval(target)
    W = image_size[1]
    H = image_size[0]

    mean_absolute_error = 0
    IoU = 0
    f1 = 0
    precision = 0
    recall = 0
    for i in tqdm(range(num), desc="Simulating random detections"):
        # Generate random bounding boxes
        pred_coords = _random_bboxes(
            image_size=(W, H),
            num_boxes=1,
            min_box_size=(1, 1),
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
        task_metrics = _calculate_final_metrics_detection_task(
            counters, count_total
        )
        summary_metrics[parent_class] = task_metrics

    return summary_metrics


def process_jsonl_file_detection_task(
    jsonl_path,
    limit=None,
):
    """
    Parse a JSONL results file and extract detection task data.

    This function:
    1. Extracts dataset name from filename
    2. Parses each line for label, target, response, task_id, etc.
    3. Resolves label names using benchmark plan configuration
    4. Returns structured data for downstream processing

    Args:
        jsonl_path: Path to the JSONL file
        limit: Maximum number of samples to process (None = process all)

    Returns:
        List of tuples: (imgModality, label_name, target,
                        filtered_resps, task_id, slice_dim)
    """
    results = []
    # Extract dataset name from filename pattern: "samples_{dataset}_..."
    match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    dataset_name = match.group(1)

    count = 0
    with open(jsonl_path, "r") as f:
        for _, line in enumerate(f):
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())

                if not data:
                    continue

                # Extract fields from JSONL entry
                doc = data.get("doc", {})
                task_id = int(doc.get("taskID"))
                filtered_resps = data.get("filtered_resps")
                target = data.get("target")
                image_size_2d = doc["image_size_2d"]
                box_img_ratio = data["box_img_ratio"]

                # Get label
                label = doc.get("label")

                if (
                    label is not None
                    and task_id is not None
                    and filtered_resps is not None
                    and target is not None
                ):

                    # Resolve label name and image modality from benchmark plan
                    labels_map, _ = get_labelsMap_imgModality_from_seg_benchmark_plan(
                        dataset_name, task_id
                    )
                    label_name = labels_map.get(str(label))
                    if label_name:
                        results.append(
                            (
                                label_name,
                                target,
                                filtered_resps,
                                task_id,
                                box_img_ratio,
                                image_size_2d,
                            )
                        )
                count += 1
                if limit is not None and count >= limit:
                    break
            except json.JSONDecodeError:
                raise ValueError(f"Error in parsing the JSON file {jsonl_path}")

    return results


def process_parsed_file_in_model_folder(
    model_dir,
    limit=None,
):
    """
    Process all JSONL files in a model's parsed folder and generate summary metrics.

    This function performs the complete pipeline:
    1. Finds all JSONL files in model_dir/parsed/
    2. Parses each file to extract detection data
    3. Groups data by anatomy-modality-slice combinations
    4. Calculates summary metrics per group
    5. Saves intermediate and final results as JSON files
    6. Generates anatomy vs tumor/lesion grouped metrics

    Args:
        model_dir: Path to the model folder
        limit: Maximum number of samples to process per file (None = all)
    """
    # Find parsed JSONL files
    parsed_files_dir = os.path.join(model_dir, "parsed")
    assert os.path.exists(
        parsed_files_dir
    ), f"Parsed files directory does not exist: {parsed_files_dir}"
    jsonl_files = glob.glob(os.path.join(parsed_files_dir, "*.jsonl"))

    # Collect all data from the parsed JSONL files
    all_data = []
    for jsonl_file in jsonl_files:
        file_data = process_jsonl_file_detection_task(jsonl_file, limit)
        all_data.extend(file_data)

    # Early exit if no valid data found
    if not all_data:
        print(f"No valid data found in {parsed_files_dir}, skipping...")
        return

    # Group by anatomy-modality-slice combinations
    grouped_data = group_by_boxImgRatio(all_data)

    # Early exit if grouping failed
    if not grouped_data:
        print(f"No grouped data found for {parsed_files_dir}, skipping...")
        return

    # Calculate summary metrics per anatomy
    summary_metrics = calculate_summary_metrics_per_anatomy_detection_task(grouped_data)

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


def _process_task_directory(task_dir, limit, skip_model_wo_parsed_files=False):
    """
    Process all model directories within a task directory.

    This is the main processing function for task-level analysis.
    It loops through all model folders, processes their results,
    and generates a final summary comparing all models.

    Args:
        task_dir: Path to task directory containing model folders
        limit: Maximum samples to process per file (None = all)
        skip_model_wo_parsed_files: Skip models without parsed folders
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Exclude "random_detection" folder if it exists
    model_dirs = [d for d in model_dirs if os.path.basename(d) != "random_detection"]

    # Process each model directory
    for model_dir in model_dirs:
        # Skip models without parsed results if requested
        parsed_files_dir = os.path.join(model_dir, "parsed")
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(model_dir, limit)

    # -----------
    # Add a folder for random detectoin model
    random_model_path = os.path.join(task_dir, "random_detection")
    if not os.path.exists(random_model_path):
        print("Processing random detection model...")
        if not os.path.exists(random_model_path):
            os.makedirs(random_model_path)

        # Read jsonl files from the first model folder
        jsonl_files_in_first_model = glob.glob(
            os.path.join(model_dirs[0], "parsed", "*.jsonl")
        )
        random_detection_data = []
        for jsonl_file in jsonl_files_in_first_model:
            file_data = process_jsonl_file_detection_task(jsonl_file)
            random_detection_data.extend(file_data)

        # Group by parent class: box-image-ratio
        grouped_random_data = group_by_boxImgRatio(random_detection_data)

        # Calculate summary metrics for random detection model
        summary_random_metrics = (
            calculate_summary_metrics_per_anatomy_detection_task_for_randomModel(
                grouped_random_data, num_simulations=RANDOM_BOX_SIMULATIONS
            )
        )

        # Save values JSON file for random detection model
        output_path = os.path.join(
            random_model_path, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES
        )
        with open(output_path, "w") as f:
            json.dump(convert_numpy_to_python(grouped_random_data), f, indent=2)
        print(
            f"Saved target and model-predicted values for random detection model to {output_path}"
        )

        # Save summary metrics JSON file for random detection model
        output_path = os.path.join(
            random_model_path, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS
        )
        with open(output_path, "w") as f:
            json.dump(convert_numpy_to_python(summary_random_metrics), f, indent=2)
        print(f"Saved summary metrics for random detection model to {output_path}")
    # -----------


def _process_single_model_directory(model_dir, limit):
    """
    Process a single model directory.

    Args:
        model_dir: Path to the model directory
        limit: Maximum number of samples to process per file
    """
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(model_dir, limit)


def main(**kwargs):
    """
    Main function to process model folders based on provided arguments.

    Args:
        task_dir: Path to task directory (mutually exclusive with model_dir)
        model_dir: Path to model directory (mutually exclusive with task_dir)
        limit: Maximum number of samples to process per file
        skip_model_wo_parsed_files: Whether to skip model directories without parsed folders
    """
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    limit = kwargs.get("limit")
    skip_model_wo_parsed_files = kwargs.get("skip_model_wo_parsed_files", False)

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directories within this folder will be looped over."
        )
        _process_task_directory(task_dir, limit, skip_model_wo_parsed_files)

    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all JSONL files within this directory."
        )
        _process_single_model_directory(model_dir, limit)

    else:
        raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def parse_args():
    """
    Parse command line arguments.

    Supports two modes:
    - Task mode (--task_dir): Process all models in a task directory
    - Model mode (--model_dir): Process a single model directory

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process model folders and generate anatomy-grouped summary metrics."
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
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process per JSONL file. If not set, processes all samples.",
    )
    parser.add_argument(
        "--skip_model_wo_parsed_files",
        action="store_true",
        help="Skip model directories that don't have a 'parsed' folder. Only valid with --task_dir.",
    )

    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.task_dir is None and args.model_dir is None:
        parser.error("Either --task_dir or --model_dir must be provided.")

    # Validate skip flag only with task_dir
    if args.skip_model_wo_parsed_files and args.task_dir is None:
        parser.error("--skip_model_wo_parsed_files can only be used with --task_dir")

    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
