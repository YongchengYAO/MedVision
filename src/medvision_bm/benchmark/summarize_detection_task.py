import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from medvision_bm.utils.configs import (
    EXCLUDED_KEYS,
    MINIMUM_GROUP_SIZE,
    SUMMARY_FILENAME_ALL_MODELS_DETECT_METRICS,
    SUMMARY_FILENAME_DETECT_METRICS,
    SUMMARY_FILENAME_DETECT_VALUES,
    SUMMARY_FILENAME_GROUPED_ANATOMY_VS_TUMOR_LESION_DETECT_METRICS,
    TUMOR_LESION_GROUP_KEYS,
)
from medvision_bm.utils.parse_utils import (
    cal_metrics_detection_task,
    convert_numpy_to_python,
    get_labelsMap_imgModality_from_seg_benchmark_plan,
    get_subfolders,
    group_by_anatomy_modality_slice,
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

    # Add cumulative MAE (Mean Absolute Error) metrics
    # MAE<k means proportion of samples with MAE less than or equal to k/10
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


def group_anatomy_vs_tumor_lesion(model_path):
    """
    Group anatomical regions into 'anatomy' vs 'tumor/lesion' (T/L) categories
    and calculate weighted mean metrics for each group.

    This function:
    1. Reads per-region metrics from SUMMARY_FILENAME_DETECT_METRICS
    2. Classifies regions as anatomy or tumor/lesion based on keywords
    3. Filters out regions marked as miscellaneous/others or with insufficient samples
    4. Calculates sample-weighted mean metrics for each group
    5. Saves results to SUMMARY_FILENAME_GROUPED_ANATOMY_VS_TUMOR_LESION_DETECT_METRICS

    Args:
        model_path: Path to the model folder containing summary metrics file
    """
    input_file = os.path.join(model_path, SUMMARY_FILENAME_DETECT_METRICS)

    if not os.path.exists(input_file):
        print(f"Summary metrics file not found: {input_file}")
        return

    # Read the summary metrics
    with open(input_file, "r") as f:
        data = json.load(f)

    # Initialize groups
    anatomy_group = {}
    tumor_lesion_group = {}

    # Classify regions into anatomical or tumor/lesion groups
    for region_name, task_data in data.items():
        region_lower = region_name.lower()
        # Exclude regions classified as miscellaneous or others
        if any(keyword in region_lower for keyword in EXCLUDED_KEYS):
            print(
                f"[Exclude] Skipping region '{region_name}' classified as miscellaneous/others."
            )
            continue
        # Exclude regions with too few samples for reliable statistics
        if task_data["num_samples"] < MINIMUM_GROUP_SIZE:
            print(
                f"[Exclude] Skipping region '{region_name}' due to insufficient samples: {task_data['num_samples']} < minimum sample size {MINIMUM_GROUP_SIZE}."
            )
            continue

        # Classify based on tumor/lesion keywords (dataset-specific)
        if any(
            keyword in region_lower for keyword in TUMOR_LESION_GROUP_KEYS
        ):  # NOTE: dataset-specific keywords can be added here
            tumor_lesion_group[region_name] = task_data
        else:
            anatomy_group[region_name] = task_data

    # Calculate sample-weighted mean metrics for each group
    def calculate_group_mean_metrics(group_data):
        """
        Calculate sample-weighted mean metrics across all regions in a group.

        Args:
            group_data: Dictionary of region_name -> task_data mappings

        Returns:
            Dictionary with weighted mean metrics and metadata
        """
        if not group_data:
            return {}

        # Collect all metrics from all regions and tasks with their sample weights
        all_metrics = defaultdict(list)
        all_sample_sizes = defaultdict(list)
        total_samples = 0

        # Aggregate metrics with sample size weights
        for _, metrics in group_data.items():
            # Weight each metric by its sample size
            sample_size = metrics.get("num_samples", 0)

            for metric_name, value in metrics.items():
                if metric_name == "num_samples":
                    total_samples += value
                elif not np.isnan(value) and np.isfinite(value):
                    all_metrics[metric_name].append(value)
                    all_sample_sizes[metric_name].append(sample_size)

        # Calculate weighted means: sum(value * weight) / sum(weight)
        mean_metrics = {}
        for metric_name, values in all_metrics.items():
            num_sample = all_sample_sizes[metric_name]
            if sum(num_sample) > 0:
                mean_metrics[metric_name] = np.sum(
                    np.array(values) * np.array(num_sample)
                ) / np.sum(num_sample)
            else:
                raise ValueError(f"No valid samples for metric {metric_name}")

        mean_metrics["total_samples"] = total_samples
        mean_metrics["num_regions"] = len(group_data)

        return mean_metrics

    # Calculate mean metrics for both groups
    anatomy_mean = calculate_group_mean_metrics(anatomy_group)
    tumor_lesion_mean = calculate_group_mean_metrics(tumor_lesion_group)

    # Create output structure
    grouped_results = {
        "anatomy": {
            "mean_metrics": anatomy_mean,
            "regions": list(anatomy_group.keys()),
            "detailed_data": anatomy_group,
        },
        "T/L": {
            "mean_metrics": tumor_lesion_mean,
            "regions": list(tumor_lesion_group.keys()),
            "detailed_data": tumor_lesion_group,
        },
    }

    # Save grouped results
    output_path = os.path.join(
        model_path, SUMMARY_FILENAME_GROUPED_ANATOMY_VS_TUMOR_LESION_DETECT_METRICS
    )
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(grouped_results), f, indent=2)

    print(f"Saved grouped anatomy vs tumor/lesion metrics to {output_path}")
    print(
        f"Anatomy group: {len(anatomy_group)} regions, {anatomy_mean.get('total_samples', 0)} total samples"
    )
    print(
        f"T/L group: {len(tumor_lesion_group)} regions, {tumor_lesion_mean.get('total_samples', 0)} total samples"
    )


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
                slice_dim = doc.get("slice_dim")
                task_id = int(doc.get("taskID"))
                filtered_resps = data.get("filtered_resps")
                target = data.get("target")

                # Get label
                label = doc.get("label")

                if (
                    label is not None
                    and task_id is not None
                    and filtered_resps is not None
                    and target is not None
                ):
                    # Resolve label name and image modality from benchmark plan
                    labels_map, imgModality = (
                        get_labelsMap_imgModality_from_seg_benchmark_plan(
                            dataset_name, task_id
                        )
                    )
                    label_name = labels_map.get(str(label))
                    if label_name:
                        results.append(
                            (
                                imgModality,
                                label_name,
                                target,
                                filtered_resps,
                                task_id,
                                slice_dim,
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
    grouped_data = group_by_anatomy_modality_slice(all_data)

    # Early exit if grouping failed
    if not grouped_data:
        print(f"No grouped data found for {parsed_files_dir}, skipping...")
        return

    # Calculate summary metrics per anatomy
    summary_metrics = calculate_summary_metrics_per_anatomy_detection_task(grouped_data)

    # Save values JSON file
    output_path = os.path.join(parsed_files_dir, SUMMARY_FILENAME_DETECT_VALUES)
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(grouped_data), f, indent=2)
    print(f"Saved target and model-predicted values to {output_path}")

    # Save summary metrics JSON file
    output_path = os.path.join(parsed_files_dir, SUMMARY_FILENAME_DETECT_METRICS)
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved summary metrics to {output_path}")

    # Group anatomy vs tumor/lesion and calculate mean metrics
    group_anatomy_vs_tumor_lesion(parsed_files_dir)


def print_summary_metrics(task_dir, skip_model_wo_parsed_files=False):
    """
    Print and save summary metrics for all models in a task directory.

    This function:
    1. Collects metrics from all model directories
    2. Prints formatted summary table to console
    3. Saves metrics to JSON file
    4. Saves console output to text file

    Args:
        task_dir: Path to task directory containing model folders
        skip_model_wo_parsed_files: If True, skip models without parsed folders
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Prepare output file path
    output_file_path = os.path.join(task_dir, "summary_detection_task.txt")

    # Collect all output lines
    output_lines = []

    def print_and_capture(text):
        """Helper function to print and capture text"""
        print(text)
        output_lines.append(text)

    print_and_capture("\n" + "=" * 80)
    print_and_capture("SUMMARY METRICS: Recall, Precision, and F1")
    print_and_capture("=" * 80)

    # Collect metrics for all models
    all_model_metrics = {}

    for model_dir in model_dirs:
        parsed_dir = os.path.join(model_dir, "parsed")

        # Skip models without parsed results if requested
        if skip_model_wo_parsed_files and not os.path.exists(parsed_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        metrics_file = os.path.join(
            model_dir,
            "parsed",
            SUMMARY_FILENAME_GROUPED_ANATOMY_VS_TUMOR_LESION_DETECT_METRICS,
        )

        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                data = json.load(f)

            model_metrics = {}
            for group_name in ["anatomy", "T/L"]:
                if group_name in data and "mean_metrics" in data[group_name]:
                    mean_metrics = data[group_name]["mean_metrics"]
                    model_metrics[group_name] = {
                        "Recall": mean_metrics.get("Recall", np.nan),
                        "Precision": mean_metrics.get("Precision", np.nan),
                        "F1": mean_metrics.get("F1", np.nan),
                        "IoU": mean_metrics.get("IoU", np.nan),
                        "SuccessRate": mean_metrics.get("SuccessRate", np.nan),
                        "IoU>0.5": mean_metrics.get("IoU>0.5", np.nan),
                        "F1>0.5": mean_metrics.get("F1>0.5", np.nan),
                        "total_samples": mean_metrics.get("total_samples", 0),
                        "num_regions": mean_metrics.get("num_regions", 0),
                    }

            all_model_metrics[os.path.basename(model_dir)] = model_metrics

    # Print metrics for each model
    for model, metrics in all_model_metrics.items():
        print_and_capture(f"\nModel: {model}")
        print_and_capture("-" * len(f"Model: {model}"))

        for group_name in ["anatomy", "T/L"]:
            if group_name in metrics:
                group_metrics = metrics[group_name]
                recall = group_metrics["Recall"]
                precision = group_metrics["Precision"]
                f1 = group_metrics["F1"]
                iou = group_metrics["IoU"]
                success_rate = group_metrics["SuccessRate"]
                iou_05 = group_metrics["IoU>0.5"]
                f1_05 = group_metrics["F1>0.5"]
                samples = group_metrics["total_samples"]
                regions = group_metrics["num_regions"]

                print_and_capture(
                    f"  {group_name.upper():8} ({regions:2d} regions, {samples:4d} samples): "
                    f"Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}, IoU={iou:.3f}, "
                    f"SuccessRate={success_rate:.3f}, IoU>0.5={iou_05:.3f}, F1>0.5={f1_05:.3f}"
                )

    # Save summary metrics to JSON
    summary_output_path = os.path.join(
        task_dir, SUMMARY_FILENAME_ALL_MODELS_DETECT_METRICS
    )
    with open(summary_output_path, "w") as f:
        json.dump(convert_numpy_to_python(all_model_metrics), f, indent=2)

    print_and_capture("\n" + "=" * 80)
    print_and_capture(f"Summary metrics saved to: {summary_output_path}")
    print_and_capture("=" * 80)

    # Save printed output to text file
    with open(output_file_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Table output saved to: {output_file_path}")


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

    # NOTE: Exclude random_detection folder
    model_dirs = [d for d in model_dirs if os.path.basename(d) != "random_detection"]

    # Print configuration info once at the beginning
    print(f"\nConfigurations in medvision_bm/utils/configs.py:")
    print(f"  TUMOR_LESION_GROUP_KEYS: {TUMOR_LESION_GROUP_KEYS}")
    print(f"  EXCLUDED_KEYS: {EXCLUDED_KEYS}")
    print(f"  MINIMUM_GROUP_SIZE: {MINIMUM_GROUP_SIZE}\n")

    # Process each model directory
    for model_dir in model_dirs:
        # Skip models without parsed results if requested
        parsed_files_dir = os.path.join(model_dir, "parsed")
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(model_dir, limit)

    # Print summary metrics at the end
    print_summary_metrics(task_dir, skip_model_wo_parsed_files)


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
