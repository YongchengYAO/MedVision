import argparse
import glob
import json
import os
import re

import numpy as np

from medvision_bm.utils.configs import (
    SUMMARY_FILENAME_AD_METRICS,
    SUMMARY_FILENAME_AD_VALUES,
)
from medvision_bm.utils.parse_utils import convert_numpy_to_python, get_subfolders


def cal_metrics_AD_task(results):
    """
    Calculate metrics for Angle/Distance (AD) estimation task.

    This function evaluates a model's prediction against the ground truth by computing:
    - Mean Absolute Error (MAE)
    - Mean Relative Error (MRE)
    - Success flag (whether parsing was successful)

    Args:
        results (dict): Dictionary containing:
            - 'filtered_resps' (list): List with a single prediction string
            - 'target' (str): Ground truth value as a string (to be evaluated)

    Returns:
        dict: Dictionary with three metric entries:
            - 'avgMAE': {'MAE': float, 'success': bool}
            - 'avgMRE': {'MRE': float, 'success': bool}
            - 'SuccessRate': {'success': bool}

    Note:
        - The metric key names must match the "metric" field in the task's YAML configuration
        - Returns np.nan for MAE/MRE if parsing fails or validation errors occur
    """
    pred = results["filtered_resps"][0]
    target_metrics = np.array(eval(results["target"]))
    try:
        # Parse prediction: expect comma-separated values, convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])

        # Validate: prediction must be a single value (not multiple)
        if len(pred_metrics) != 1:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            # Calculate errors
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            # Add small epsilon to avoid division by zero
            mean_relative_error = np.mean(absolute_error / (target_metrics + 1e-15))
            success = True
    except:
        # Handle any parsing or computation errors
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgMRE": {"MRE": mean_relative_error, "success": success},
        "SuccessRate": {"success": success},
    }


def _initialize_metric_counters_AD_task():
    """
    Initialize all metric counters for AD task aggregation.

    Returns:
        dict: Dictionary containing initialized counters:
            - 'sum_MAE': Cumulative sum of MAE values
            - 'sum_MRE': Cumulative sum of MRE values
            - 'num_success': Count of successful predictions
            - 'count_valid_AE': Count of samples with valid absolute error
            - 'count_valid_RE': Count of samples with valid relative error
            - 'count_AE_thresholds': List of 10 bins for AE distribution
            - 'count_RE_thresholds': List of 10 bins for RE distribution

    Note:
        Threshold bins: [0.0-0.1), [0.1-0.2), ..., [0.8-0.9), [0.9-∞)
    """
    return {
        "sum_MAE": 0,
        "sum_MRE": 0,
        "num_success": 0,
        "count_valid_AE": 0,
        "count_valid_RE": 0,
        "count_AE_thresholds": [0] * 10,
        "count_RE_thresholds": [0] * 10,
    }


def _update_mae_counters(value, counters):
    """
    Update Mean Absolute Error (MAE) related counters.

    Updates cumulative sum, valid count, and threshold distribution bins.

    Args:
        value (float): MAE value to process
        counters (dict): Counter dictionary to update in-place

    Note:
        - Only processes non-NaN values
        - Threshold buckets: i covers [i*0.1, (i+1)*0.1), bucket 9 covers [0.9, ∞)
    """
    if not np.isnan(value):
        counters["sum_MAE"] += value
        counters["count_valid_AE"] += 1

        # Determine which threshold bucket this value belongs to
        threshold_index = min(int(value * 10), 9)
        counters["count_AE_thresholds"][threshold_index] += 1


def _update_mre_counters(value, counters):
    """
    Update Mean Relative Error (MRE) related counters.

    Updates cumulative sum, valid count, and threshold distribution bins.

    Args:
        value (float): MRE value to process
        counters (dict): Counter dictionary to update in-place

    Note:
        - Only processes non-NaN values
        - Threshold buckets: i covers [i*0.1, (i+1)*0.1), bucket 9 covers [0.9, ∞)
    """
    if not np.isnan(value):
        counters["sum_MRE"] += value
        counters["count_valid_RE"] += 1

        # Determine which threshold bucket this value belongs to
        threshold_index = min(int(value * 10), 9)
        counters["count_RE_thresholds"][threshold_index] += 1


def _update_metric_counters_AD_task(metrics_dict, counters):
    """
    Update all metric counters based on calculated metrics for a single sample.

    This is a convenience function that updates MAE, MRE, and success counters
    in a single call.

    Args:
        metrics_dict (dict): Dictionary containing avgMAE, avgMRE, and SuccessRate metrics
        counters (dict): Counter dictionary to update in-place
    """
    # Update MAE counters
    if not np.isnan(metrics_dict["avgMAE"]["MAE"]):
        _update_mae_counters(metrics_dict["avgMAE"]["MAE"], counters)

    # Update MRE counters
    if not np.isnan(metrics_dict["avgMRE"]["MRE"]):
        _update_mre_counters(metrics_dict["avgMRE"]["MRE"], counters)

    # Update success count
    counters["num_success"] += metrics_dict["SuccessRate"]["success"]


def _calculate_final_metrics_AD_task(counters, count_total):
    """
    Calculate final aggregated metrics from accumulated counters.

    Computes average errors, success rate, and cumulative threshold-based accuracies.

    Args:
        counters (dict): Dictionary of accumulated counters
        count_total (int): Total number of samples processed

    Returns:
        dict: Dictionary containing computed metrics:
            - 'avgMAE': Average mean absolute error across all valid samples
            - 'avgMRE': Average mean relative error across all valid samples
            - 'SuccessRate': Percentage of successful predictions (0.0 to 1.0)
            - 'MAE<k': Cumulative accuracy for k in [0.1, 0.2, ..., 1.0]
            - 'MRE<k': Cumulative accuracy for k in [0.1, 0.2, ..., 1.0]
            - 'num_samples': Total sample count

    Note:
        - Returns np.nan for averages if no valid samples exist
        - Cumulative accuracies represent proportion of samples below threshold k
    """
    task_metrics = {
        "avgMAE": (
            counters["sum_MAE"] / counters["count_valid_AE"]
            if counters["count_valid_AE"] > 0
            else np.nan
        ),
        "avgMRE": (
            counters["sum_MRE"] / counters["count_valid_RE"]
            if counters["count_valid_RE"] > 0
            else np.nan
        ),
        "SuccessRate": (
            counters["num_success"] / count_total if count_total > 0 else 0.0
        ),
        "num_samples": count_total,
    }

    # Calculate cumulative threshold-based accuracies
    keys = ["RE"]
    for key in keys:
        for k in range(1, 11):
            # Sum counts from bucket 0 to bucket k-1 (inclusive)
            cumulative_count = sum(counters[f"count_{key}_thresholds"][0:k])
            task_metrics[f"M{key}<{k/10:.1f}"] = (
                cumulative_count / count_total if count_total > 0 else 0.0
            )

    return task_metrics


def calculate_summary_metrics_per_anatomy_AD_task(all_data):
    """
    Calculate summary metrics grouped by label (anatomy/metric type).

    This function aggregates predictions by label and computes comprehensive metrics
    for each unique label (e.g., dataset_metricType_metricKey combinations).

    Args:
        all_data (list): List of dictionaries, each containing:
            - 'label' (str): Label identifier (e.g., "FeTA24_distance_BPD")
            - 'targets' (str): Ground truth value
            - 'responses' (list): Model predictions

    Returns:
        dict: Dictionary mapping each label to its computed metrics:
            - avgMAE, avgMRE, SuccessRate
            - MAE<k and MRE<k for k in [0.1, 0.2, ..., 1.0]
            - num_samples

    Note:
        Groups data by label before computing metrics to enable per-anatomy analysis
    """
    # Group data by label
    grouped_data = {}
    for item in all_data:
        label = item["label"]
        target = item["targets"]
        response = item["responses"][0]

        if label not in grouped_data:
            grouped_data[label] = {"targets": [], "responses": []}

        grouped_data[label]["targets"].append(target)
        grouped_data[label]["responses"].append(response)
    summary_metrics = {}

    # Calculate metrics for each label group
    for label, data in grouped_data.items():
        targets = data["targets"]
        responses = data["responses"]

        # Skip empty groups
        if not targets or not responses:
            continue

        # Initialize counters for this group
        counters = _initialize_metric_counters_AD_task()
        count_total = len(targets)

        # Process each target-response pair
        for target, response in zip(targets, responses):
            mock_results = {"filtered_resps": [response], "target": target}
            metrics_dict = cal_metrics_AD_task(mock_results)
            _update_metric_counters_AD_task(metrics_dict, counters)

        # Calculate and store final metrics for this label
        task_metrics = _calculate_final_metrics_AD_task(counters, count_total)

        # Initialize the label entry if it doesn't exist
        if label not in summary_metrics:
            summary_metrics[label] = {}
        summary_metrics[label] = task_metrics

    return summary_metrics


def find_and_group_jsonl_files(model_path):
    """
    Find and group JSONL files in a model directory by dataset and task.

    Different datasets have different grouping strategies:
    - Ceph-Biometrics-400: Each file is kept separate (one file per group)
    - FeTA24: All files for the same task are grouped together

    Args:
        model_path (str): Path to the directory containing JSONL files

    Returns:
        dict: Dictionary mapping group keys to lists of file paths
            Example: {
                'samples_Ceph-Biometrics-400_Task01.jsonl': ['path/to/file.jsonl'],
                'FeTA24_BiometricsFromLandmarks_Task01_combined': ['path/to/file1.jsonl', 'path/to/file2.jsonl']
            }
    """
    # Find all JSONL files in the model folder
    jsonl_files = glob.glob(os.path.join(model_path, "*.jsonl"))

    # Group files by dataset and task
    grouped_files = {}

    for jsonl_file in jsonl_files:
        filename = os.path.basename(jsonl_file)

        # Handle Ceph-Biometrics-400: no grouping, one file per key
        if "Ceph-Biometrics-400" in filename:
            grouped_files[filename] = [jsonl_file]

        # Handle FeTA24: group all files for the same task together
        elif "FeTA24_BiometricsFromLandmarks_Task01" in filename:
            key = "FeTA24_BiometricsFromLandmarks_Task01_combined"
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(jsonl_file)

    return grouped_files


def process_jsonl_file(
    jsonl_path,
    limit,
):
    """
    Process a single JSONL file and extract label, target, and response data.

    Each line in the JSONL file represents one prediction sample. This function
    extracts the biometric profile information and constructs a structured label.

    Args:
        jsonl_path (str): Path to the JSONL file
        limit (int, optional): Maximum number of samples to process. None for all samples.

    Returns:
        list: List of dictionaries, each containing:
            - 'label' (str): Constructed label (dataset_metricType_metricKey)
            - 'targets' (str): Ground truth value
            - 'responses' (list): Model predictions

    Raises:
        ValueError: If JSON parsing fails
        AssertionError: If metric_key is None
    """
    results = []
    # Extract dataset name from filename pattern 'samples_{dataset_name}_'
    match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    dataset_name = match.group(1)

    count = 0
    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            # Skip empty lines
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())
                # Skip empty data
                if not data:
                    continue

                doc = data.get("doc", {})

                filtered_resps = data.get("filtered_resps")
                target = data.get("target")

                # Extract biometric profile information
                biometric_profile = doc.get("biometric_profile", {})
                metric_type = biometric_profile.get("metric_type", "")
                metric_key = biometric_profile.get("metric_key")
                assert (
                    metric_key is not None
                ), f"metric_key is None in line {line_idx+1} of {jsonl_path}"

                # Construct label: dataset_metricType_metricKey
                # Example: "FeTA24_distance_BPD" or "Ceph-Biometrics-400_angle_SNA"
                label_name = f"{dataset_name}_{metric_type}_{metric_key}"
                results.append(
                    {
                        "label": label_name,
                        "targets": target,
                        "responses": filtered_resps,
                    }
                )

                count += 1
                if limit is not None and count >= limit:
                    break

            except json.JSONDecodeError:
                raise ValueError(f"Error in parsing the JSON file {jsonl_path}")

    return results


def process_combined_jsonl_files(jsonl_paths, limit):
    """
    Process multiple JSONL files and combine their data into a single list.

    Useful for combining data from multiple files that belong to the same
    dataset or task group (e.g., multiple FeTA24 files).

    Args:
        jsonl_paths (list): List of paths to JSONL files to combine
        limit (int, optional): Maximum number of samples to process per file.
                              None for all samples.

    Returns:
        list: Combined list of dictionaries with 'label', 'targets', and 'responses' keys
    """
    combined_data = []

    for jsonl_path in jsonl_paths:
        file_data = process_jsonl_file(jsonl_path, limit)
        combined_data.extend(file_data)

    return combined_data


def process_parsed_file_in_model_folder(
    model_dir,
    limit=None,
):
    """
    Process all JSONL files in a model folder and generate summary metrics.

    This is the main processing function that:
    1. Finds and groups JSONL files in the 'parsed' subdirectory
    2. Extracts targets and model predictions from each file
    3. Calculates comprehensive metrics per anatomy/metric type
    4. Saves results to two JSON files in the parsed directory:
       - summary_AD_values.json (raw data: targets and predictions)
       - summary_AD_metrics.json (aggregated metrics per label)

    Args:
        model_dir (str): Path to the model folder (must contain a 'parsed' subdirectory)
        limit (int, optional): Maximum number of samples to process per file.
                              None for all samples.

    Raises:
        AssertionError: If parsed directory doesn't exist
    """
    # Locate parsed files directory
    parsed_files_dir = os.path.join(model_dir, "parsed")
    assert os.path.exists(
        parsed_files_dir
    ), f"Parsed files directory does not exist: {parsed_files_dir}"
    grouped_files = find_and_group_jsonl_files(parsed_files_dir)

    # Collect all data from the parsed JSONL files
    all_data = []
    for group_name, file_paths in grouped_files.items():
        if len(file_paths) == 1:
            # Single file processing (e.g., Ceph-Biometrics-400)
            file_data = process_jsonl_file(file_paths[0], limit)
        else:
            # Combined file processing (e.g., multiple FeTA24 files)
            file_data = process_combined_jsonl_files(file_paths, limit)

        all_data.extend(file_data)

    # Skip if no valid data was collected
    if not all_data:
        print(f"No valid data found in {parsed_files_dir}, skipping...")
        return

    # Calculate summary metrics per anatomy
    summary_metrics = calculate_summary_metrics_per_anatomy_AD_task(all_data)

    # Save raw values JSON file (targets and predictions for each sample)
    # Output: parsed/summary_AD_values.json
    output_path = os.path.join(
        parsed_files_dir, SUMMARY_FILENAME_AD_VALUES
    )
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(all_data), f, indent=2)
    print(f"Saved target and model-predicted values to {output_path}")

    # Save aggregated metrics JSON file (metrics per label)
    # Output: parsed/summary_AD_metrics.json
    output_path = os.path.join(
        parsed_files_dir, SUMMARY_FILENAME_AD_METRICS
    )
    with open(output_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved summary metrics to {output_path}")


def print_model_summaries(task_dir, skip_model_wo_parsed_files=False):
    """
    Print and save summary metrics for all models in a task directory.

    This function generates a comprehensive summary report that includes:
    1. Overall weighted averages across all anatomies/metrics
    2. Group-level averages (FeTA-Distance, Ceph-Angle, Ceph-Distance)
    3. Label-specific detailed metrics

    The output is both printed to console and saved to a text file.

    Args:
        task_dir (str): Path to the task directory containing model folders
        skip_model_wo_parsed_files (bool): Whether to skip models without parsed folders

    Input:
        - Reads from {model_dir}/parsed/summary_AD_metrics.json for each model
    
    Output:
        - Prints formatted tables to console
        - Saves summary to {task_dir}/summary_AD_task.txt
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Prepare output file path
    output_file_path = os.path.join(task_dir, "summary_AD_task.txt")

    # Collect all output lines
    output_lines = []

    def print_and_capture(text):
        """Helper function to print and capture text for file output"""
        print(text)
        output_lines.append(text)

    print_and_capture("\n\n========== MODEL SUMMARIES ==========\n")

    model_summaries = {}

    for model_dir in model_dirs:
        parsed_dir = os.path.join(model_dir, "parsed")

        # Skip if parsed folder doesn't exist and flag is set
        if skip_model_wo_parsed_files and not os.path.exists(parsed_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        metrics_file = os.path.join(
            parsed_dir, SUMMARY_FILENAME_AD_METRICS
        )

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        model_summary = {
            "labels": {},
            "total_samples": 0,
            "weighted_sum_mae": 0.0,
            "weighted_avg_mae": None,
            "weighted_sum_mre": 0.0,
            "weighted_avg_mre": None,
            "weighted_sum_sr": 0.0,
            "weighted_avg_sr": None,
            "weighted_mre<0.1": None,
            "weighted_mre<0.2": None,
            "weighted_mre<0.3": None,
        }

        # Process metrics: each label maps to {avgMAE, avgMRE, SuccessRate, MRE<k, num_samples}
        for label, label_metrics in metrics.items():
            mae = label_metrics.get("avgMAE")
            mre = label_metrics.get("avgMRE")
            sr = label_metrics.get("SuccessRate")
            samples = label_metrics.get("num_samples", 0)
            mre_lt_01 = label_metrics.get("MRE<0.1")
            mre_lt_02 = label_metrics.get("MRE<0.2")
            mre_lt_03 = label_metrics.get("MRE<0.3")

            # Skip if samples <= 0
            if samples is None or samples <= 0:
                continue

            model_summary["labels"][label] = {
                "MAE": mae,
                "MRE": mre,
                "SR": sr,
                "samples": samples,
                "task_type": "A/D",  # Angle/Distance task type
                "MRE<0.1": mre_lt_01,
                "MRE<0.2": mre_lt_02,
                "MRE<0.3": mre_lt_03,
            }

            # Accumulate weighted sums (only for non-NaN values)
            if mae is not None and not np.isnan(mae):
                model_summary["weighted_sum_mae"] += mae * samples
            if mre is not None and not np.isnan(mre):
                model_summary["weighted_sum_mre"] += mre * samples
            if sr is not None and not np.isnan(sr):
                model_summary["weighted_sum_sr"] += sr * samples
            model_summary["total_samples"] += samples

        # Compute weighted averages for MAE / MRE / SR
        if model_summary["total_samples"] > 0:
            denom = model_summary["total_samples"]
            model_summary["weighted_avg_mae"] = (
                model_summary["weighted_sum_mae"] / denom
            )
            model_summary["weighted_avg_mre"] = (
                model_summary["weighted_sum_mre"] / denom
            )
            model_summary["weighted_avg_sr"] = model_summary["weighted_sum_sr"] / denom

        # Compute weighted (micro-averaged) MRE<0.1/0.2/0.3 across all labels
        wsum_re01 = wsum_re02 = wsum_re03 = 0.0
        wcount_re01 = wcount_re02 = wcount_re03 = 0
        for _m in model_summary["labels"].values():
            samples_lbl = _m["samples"]
            v1 = _m.get("MRE<0.1")
            v2 = _m.get("MRE<0.2")
            v3 = _m.get("MRE<0.3")
            if v1 is not None and not np.isnan(v1):
                wsum_re01 += v1 * samples_lbl
                wcount_re01 += samples_lbl
            if v2 is not None and not np.isnan(v2):
                wsum_re02 += v2 * samples_lbl
                wcount_re02 += samples_lbl
            if v3 is not None and not np.isnan(v3):
                wsum_re03 += v3 * samples_lbl
                wcount_re03 += samples_lbl

        if wcount_re01 > 0:
            model_summary["weighted_mre<0.1"] = wsum_re01 / wcount_re01
        if wcount_re02 > 0:
            model_summary["weighted_mre<0.2"] = wsum_re02 / wcount_re02
        if wcount_re03 > 0:
            model_summary["weighted_mre<0.3"] = wsum_re03 / wcount_re03
        model_summaries[os.path.basename(model_dir)] = model_summary

    # Output formatted summaries
    for model, summary in model_summaries.items():
        model_header = f"\nModel: {model}"
        weighted_avg = (
            f"Weighted Average MAE: "
            f"{(summary['weighted_avg_mae'] if summary['weighted_avg_mae'] is not None else float('nan')):.4f}, "
            f"MRE: "
            f"{(summary['weighted_avg_mre'] if summary['weighted_avg_mre'] is not None else float('nan')):.4f}, "
            f"SR: "
            f"{(summary['weighted_avg_sr'] if summary['weighted_avg_sr'] is not None else float('nan')):.4f} "
            f"(Total Samples: {summary['total_samples']})"
        )
        acc_parts = []
        if summary["weighted_mre<0.1"] is not None:
            acc_parts.append(f"Weighted MRE<0.1: {summary['weighted_mre<0.1']:.4f}")
        if summary["weighted_mre<0.2"] is not None:
            acc_parts.append(f"Weighted MRE<0.2: {summary['weighted_mre<0.2']:.4f}")
        if summary["weighted_mre<0.3"] is not None:
            acc_parts.append(f"Weighted MRE<0.3: {summary['weighted_mre<0.3']:.4f}")
        acc_summary = " | ".join(acc_parts) if acc_parts else "No MRE metrics"

        # Calculate group averages for this model
        feta_distance_labels = []
        ceph_angle_labels = []
        ceph_distance_labels = []
        # Group labels by dataset and metric type
        for label, label_metrics in summary["labels"].items():
            if "FeTA24_distance" in label:
                feta_distance_labels.append(label_metrics)
            elif "Ceph-Biometrics-400_angle" in label:
                ceph_angle_labels.append(label_metrics)
            elif "Ceph-Biometrics-400_distance" in label:
                ceph_distance_labels.append(label_metrics)

        def calculate_group_avg(group_labels):
            """
            Calculate micro-averaged (weighted by sample size) metrics for a group of labels.

            Micro-averaging weights each sample equally, as opposed to macro-averaging
            which would weight each label equally.

            Args:
                group_labels (list): List of label metric dictionaries, each containing
                                    MAE, MRE, SR, threshold metrics, and sample count

            Returns:
                dict: Dictionary with micro-averaged metrics:
                    - 'MAE': Weighted average mean absolute error
                    - 'MRE': Weighted average mean relative error
                    - 'SR': Weighted average success rate
                    - 'MRE<0.1', 'MRE<0.2', 'MRE<0.3': Weighted threshold-based accuracies
                    - 'samples': Total number of samples in the group

            Note:
                Returns NaN for metrics if no valid samples exist for that metric
            """
            if not group_labels:
                return {
                    "MAE": float("nan"),
                    "MRE": float("nan"),
                    "SR": float("nan"),
                    "MRE<0.1": float("nan"),
                    "MRE<0.2": float("nan"),
                    "MRE<0.3": float("nan"),
                    "samples": 0,
                }

            # Calculate micro averages (weighted by sample size)
            total_samples = sum(l["samples"] for l in group_labels)

            if total_samples == 0:
                return {
                    "MAE": float("nan"),
                    "MRE": float("nan"),
                    "SR": float("nan"),
                    "MRE<0.1": float("nan"),
                    "MRE<0.2": float("nan"),
                    "MRE<0.3": float("nan"),
                    "samples": 0,
                }

            weighted_sum_mae = 0
            weighted_sum_mre = 0
            weighted_sum_sr = 0
            weighted_sum_re01 = 0
            weighted_sum_re02 = 0
            weighted_sum_re03 = 0

            valid_samples_mae = 0
            valid_samples_mre = 0
            valid_samples_sr = 0
            valid_samples_re01 = 0
            valid_samples_re02 = 0
            valid_samples_re03 = 0

            for l in group_labels:
                samples = l["samples"]
                if l["MAE"] is not None and not np.isnan(l["MAE"]):
                    weighted_sum_mae += l["MAE"] * samples
                    valid_samples_mae += samples
                if l["MRE"] is not None and not np.isnan(l["MRE"]):
                    weighted_sum_mre += l["MRE"] * samples
                    valid_samples_mre += samples
                if l["SR"] is not None and not np.isnan(l["SR"]):
                    weighted_sum_sr += l["SR"] * samples
                    valid_samples_sr += samples
                if l["MRE<0.1"] is not None and not np.isnan(l["MRE<0.1"]):
                    weighted_sum_re01 += l["MRE<0.1"] * samples
                    valid_samples_re01 += samples
                if l["MRE<0.2"] is not None and not np.isnan(l["MRE<0.2"]):
                    weighted_sum_re02 += l["MRE<0.2"] * samples
                    valid_samples_re02 += samples
                if l["MRE<0.3"] is not None and not np.isnan(l["MRE<0.3"]):
                    weighted_sum_re03 += l["MRE<0.3"] * samples
                    valid_samples_re03 += samples

            return {
                "MAE": (
                    weighted_sum_mae / valid_samples_mae
                    if valid_samples_mae > 0
                    else float("nan")
                ),
                "MRE": (
                    weighted_sum_mre / valid_samples_mre
                    if valid_samples_mre > 0
                    else float("nan")
                ),
                "SR": (
                    weighted_sum_sr / valid_samples_sr
                    if valid_samples_sr > 0
                    else float("nan")
                ),
                "MRE<0.1": (
                    weighted_sum_re01 / valid_samples_re01
                    if valid_samples_re01 > 0
                    else float("nan")
                ),
                "MRE<0.2": (
                    weighted_sum_re02 / valid_samples_re02
                    if valid_samples_re02 > 0
                    else float("nan")
                ),
                "MRE<0.3": (
                    weighted_sum_re03 / valid_samples_re03
                    if valid_samples_re03 > 0
                    else float("nan")
                ),
                "samples": total_samples,
            }

        group_averages = {
            "FeTA-Distance": calculate_group_avg(feta_distance_labels),
            "Ceph-Angle": calculate_group_avg(ceph_angle_labels),
            "Ceph-Distance": calculate_group_avg(ceph_distance_labels),
        }

        # Group averages output
        group_header = "\nGroup averages:"
        group_table_header = (
            f"{'Group':<15} | {'MAE':<8} | {'MRE':<8} | {'SR':<8} | "
            f"{'MRE<0.1':<8} | {'MRE<0.2':<8} | {'MRE<0.3':<8} | {'Samples':<8}"
        )
        group_separator = "-" * 88

        print_and_capture(model_header)
        print_and_capture(weighted_avg)
        print_and_capture(acc_summary)
        print_and_capture(group_header)
        print_and_capture(group_table_header)
        print_and_capture(group_separator)

        for group_name, group_avg in group_averages.items():
            group_line = (
                f"{group_name:<15} | "
                f"{group_avg['MAE']:<8.4f} | "
                f"{group_avg['MRE']:<8.4f} | "
                f"{group_avg['SR']:<8.4f} | "
                f"{group_avg['MRE<0.1']:<8.4f} | "
                f"{group_avg['MRE<0.2']:<8.4f} | "
                f"{group_avg['MRE<0.3']:<8.4f} | "
                f"{group_avg['samples']:<8}"
            )
            print_and_capture(group_line)

        label_header = "\nLabel-specific metrics:"
        table_header = (
            f"{'Label':<50} | {'MAE':<8} | {'MRE':<8} | {'SR':<8} | "
            f"{'MRE<0.1':<8} | {'MRE<0.2':<8} | {'MRE<0.3':<8} | {'Samples':<8}"
        )
        separator = "-" * 128

        print_and_capture(label_header)
        print_and_capture(table_header)
        print_and_capture(separator)

        # Sort labels by sample size
        sorted_labels = sorted(
            summary["labels"].items(), key=lambda x: x[1]["samples"], reverse=True
        )
        for label, m in sorted_labels:
            re01 = m.get("MRE<0.1")
            re02 = m.get("MRE<0.2")
            re03 = m.get("MRE<0.3")
            line = (
                f"{label:<50} | "
                f"{(m['MAE'] if m['MAE'] is not None else float('nan')):<8.4f} | "
                f"{(m['MRE'] if m['MRE'] is not None else float('nan')):<8.4f} | "
                f"{(m['SR'] if m['SR'] is not None else float('nan')):<8.4f} | "
                f"{(re01 if re01 is not None else float('nan')):<8.4f} | "
                f"{(re02 if re02 is not None else float('nan')):<8.4f} | "
                f"{(re03 if re03 is not None else float('nan')):<8.4f} | "
                f"{m['samples']:<8}"
            )
            print_and_capture(line)

        section_end = "\n" + "=" * 100 + "\n"
        print_and_capture(section_end)

    # Write all captured lines to file
    with open(output_file_path, "w") as output_file:
        output_file.write("\n".join(output_lines))

    print(f"\nSummary saved to {output_file_path}")


def _process_task_directory(task_dir, limit, skip_model_wo_parsed_files=False):
    """
    Process all model directories within a task directory.

    This function orchestrates the entire processing pipeline:
    1. Finds all model subdirectories in the task directory
    2. Processes each model's JSONL files
    3. Generates summary metrics across all models

    Args:
        task_dir (str): Path to the task directory containing model folders
        limit (int, optional): Maximum number of samples to process per file
        skip_model_wo_parsed_files (bool): Whether to skip model directories
                                          without parsed folders
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Loop over each model directory and process JSONL files
    for model_dir in model_dirs:
        # Skip if parsed folder doesn't exist and flag is set
        parsed_files_dir = os.path.join(model_dir, "parsed")
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(model_dir, limit)

    # Print summary metrics at the end
    print_model_summaries(task_dir, skip_model_wo_parsed_files)


def _process_single_model_directory(model_dir, limit):
    """
    Process a single model directory.

    Convenience function for processing just one model without generating
    cross-model summaries.

    Args:
        model_dir (str): Path to the model directory
        limit (int, optional): Maximum number of samples to process per file
    """
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(model_dir, limit)


def main(**kwargs):
    """
    Main entry point for processing model folders based on provided arguments.

    Supports two modes:
    1. **task_dir mode**: Process all model directories within a task directory
       and generate cross-model summary
    2. **model_dir mode**: Process a single model directory in isolation

    Args:
        task_dir (str, optional): Path to task directory (mutually exclusive with model_dir)
        model_dir (str, optional): Path to model directory (mutually exclusive with task_dir)
        limit (int, optional): Maximum number of samples to process per file.
                              None for all samples.
        skip_model_wo_parsed_files (bool): Whether to skip model directories without
                                          parsed folders (task_dir mode only)

    Raises:
        ValueError: If neither task_dir nor model_dir is provided
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
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - task_dir: Path to task directory (or None)
            - model_dir: Path to model directory (or None)
            - limit: Sample limit (or None for all samples)
            - skip_model_wo_parsed_files: Boolean flag

    Raises:
        SystemExit: If arguments are invalid (via parser.error)
    """
    parser = argparse.ArgumentParser(
        description="Process model folders and generate summary metrics."
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

    # Validate that at least one of task_dir or model_dir is provided
    if args.task_dir is None and args.model_dir is None:
        parser.error("Either --task_dir or --model_dir must be provided.")

    # Validate that skip_model_wo_parsed_files is only used with task_dir
    if args.skip_model_wo_parsed_files and args.task_dir is None:
        parser.error("--skip_model_wo_parsed_files can only be used with --task_dir")

    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
