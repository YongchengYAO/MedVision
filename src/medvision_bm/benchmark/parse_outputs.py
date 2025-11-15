import argparse
import glob
import json
import os
import re

import numpy as np

from medvision_bm.utils.parse_utils import (_load_nifti_2d,
                                            cal_metrics_AD_task,
                                            cal_metrics_detection_task,
                                            cal_metrics_TL_task,
                                            convert_numpy_to_python,
                                            extract_last_k_nums)


def _extract_task_id(filename):
    """Extract task ID from filename."""
    match = re.search(r"([^/\\]+)_samples_", filename)
    if not match:
        raise ValueError(
            f"Unable to determine task ID from filename: {filename}")
    return match.group(1)


def _load_results_file(jsonl_file):
    """Load the results JSON file for a given task."""
    filename = os.path.basename(jsonl_file)
    print(f"\n[Info] Processing: {filename}")

    task_id = _extract_task_id(filename)
    print(f"[Info] Task ID: {task_id}")

    results_json_file = task_id + "_results.json"
    results_json_path = os.path.join(
        os.path.dirname(jsonl_file), results_json_file)

    if not os.path.exists(results_json_path):
        raise ValueError(
            f"Results file not found for task {task_id}. Expected at: {results_json_path}")

    try:
        with open(results_json_path, "r") as rf:
            print(f"[Info] Successfully loaded results file for {task_id}")
            return json.load(rf), results_json_file
    except Exception as e:
        raise ValueError(
            f"Failed to parse results file for {task_id}: {str(e)}")


def _get_parsed_file_path(model_dir, jsonl_file):
    parsed_file_dir = os.path.join(model_dir, "parsed")
    return jsonl_file.replace(model_dir, parsed_file_dir)


def _extract_response(data):
    """Extract response from nested data structure."""
    try:
        if isinstance(data["resps"][0][0], list):
            return data["resps"][0][0][0]
        else:
            return data["resps"][0][0]
    except (IndexError, TypeError):
        return data["resps"][0][0]


def _patch_doc_detection_task(data, doc):
    """Process Box-Size specific logic."""
    if "image_size_2d" in doc:
        image_size_2d = doc["image_size_2d"]
    else:
        img_path = doc["mask_file"]
        slice_dim = doc["slice_dim"]
        slice_idx = doc["slice_idx"]
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)
        image_size_2d = img_2d.shape
        data["doc"]["image_size_2d"] = image_size_2d

    if "bounding_boxes" in doc:
        box_dimensions = doc["bounding_boxes"]["dimensions"][0]
        box_size = box_dimensions[0] * box_dimensions[1]
        box_img_ratio = box_size / (image_size_2d[0] * image_size_2d[1])
    else:
        box_relative_coords = eval(data["target"])
        box_img_ratio = abs(box_relative_coords[2] - box_relative_coords[0]) * abs(
            box_relative_coords[3] - box_relative_coords[1]
        )

    data["box_img_ratio"] = box_img_ratio


def _update_re_counts(re_val, count_RE_ls):
    """Update RE count list based on relative error value."""
    if re_val < 0.1:
        count_RE_ls[0] += 1
    elif re_val < 0.2:
        count_RE_ls[1] += 1
    elif re_val < 0.3:
        count_RE_ls[2] += 1
    elif re_val < 0.4:
        count_RE_ls[3] += 1
    elif re_val < 0.5:
        count_RE_ls[4] += 1
    elif re_val < 0.6:
        count_RE_ls[5] += 1
    elif re_val < 0.7:
        count_RE_ls[6] += 1
    elif re_val < 0.8:
        count_RE_ls[7] += 1
    elif re_val < 0.9:
        count_RE_ls[8] += 1
    elif re_val < 1.0:
        count_RE_ls[9] += 1
    return count_RE_ls


def _update_results_summary(results_summary_data, metrics, count_total):
    """Update results summary with calculated metrics."""
    task_name = list(results_summary_data["results"].keys())[0]

    avg_mae = metrics["sum_MAE"] / \
        metrics["count_valid_AE"] if isinstance(
            metrics["sum_MAE"], (int, float)) and metrics["count_valid_AE"] > 0 else np.nan
    avg_mre = metrics["sum_MRE"] / \
        metrics["count_valid_RE"] if isinstance(
            metrics["sum_MRE"], (int, float)) and metrics["count_valid_RE"] > 0 else np.nan
    avg_iou = metrics["sum_IoU"] / \
        metrics["count_valid_IoU"] if isinstance(
            metrics["sum_IoU"], (int, float)) and metrics["count_valid_IoU"] > 0 else np.nan
    success_rate = metrics["num_success"] / count_total

    results_summary_data["results"][task_name]["avgMAE,none"] = str(avg_mae)
    results_summary_data["results"][task_name]["avgMRE,none"] = str(avg_mre)
    results_summary_data["results"][task_name]["avgIoU,none"] = str(avg_iou)
    results_summary_data["results"][task_name]["SuccessRate,none"] = success_rate
    
    # Use labels like MRE<0.1, MRE<0.2, ..., MRE<1.0
    count_RE_ls = metrics["count_RE_ls"]
    if isinstance(count_RE_ls, list):
        for i in range(1, 11):
            key = f"MRE<{i/10:.1f}"
            results_summary_data["results"][task_name][key] = np.sum(
                count_RE_ls[0:i]) / count_total
    else:
        # For tasks where RE is not applicable (e.g., AD) set entries to "N/A"
        for i in range(1, 11):
            key = f"MRE<{i/10:.1f}"
            results_summary_data["results"][task_name][key] = "N/A"

    return results_summary_data


def _process_jsonl_file(jsonl_file, temp_file, task_type, limit):
    """Process a single JSONL file and return metrics."""
    metrics = {
        "sum_MAE": 0,
        "sum_MRE": 0,
        "sum_IoU": 0,
        "num_success": 0,
        "count_valid_AE": 0,
        "count_valid_RE": 0,
        "count_valid_IoU": 0,
        "count_RE_ls": [0] * 10,
    }
    count_total = 0

    if task_type == 'AD':
        target_nums = 1
        # IoU not applicable for AD task
        metrics["sum_IoU"] = "N/A"
        metrics["count_valid_IoU"] = "N/A"
    elif task_type == 'TL':
        target_nums = 2
        # IoU not applicable for TL task
        metrics["sum_IoU"] = "N/A"
        metrics["count_valid_IoU"] = "N/A"
    elif task_type == 'Detection':
        target_nums = 4
        # Relative Error not applicable for Detection task
        metrics["sum_MRE"] = "N/A"
        metrics["count_valid_RE"] = "N/A"
        metrics["count_RE_ls"] = "N/A"

    with open(jsonl_file, "r") as f, open(temp_file, "w") as temp:
        for line in f:
            data = json.loads(line)
            doc = data["doc"]
            resps = _extract_response(data)

            data["filtered_resps"] = [extract_last_k_nums(resps, target_nums)]

            if task_type == 'Detection':
                metrics_dict = cal_metrics_detection_task(data)
                data["avgMAE"] = metrics_dict["avgMAE"]
                data["avgIoU"] = metrics_dict["avgIoU"]
                data["SuccessRate"] = metrics_dict["SuccessRate"]
            elif task_type == "AD":
                metrics_dict = cal_metrics_AD_task(data)
                data["avgMAE"] = metrics_dict["avgMAE"]
                data["avgMRE"] = metrics_dict["avgMRE"]
                data["SuccessRate"] = metrics_dict["SuccessRate"]
            elif task_type == "TL":
                metrics_dict = cal_metrics_TL_task(data)
                data["avgMAE"] = metrics_dict["avgMAE"]
                data["avgMRE"] = metrics_dict["avgMRE"]
                data["SuccessRate"] = metrics_dict["SuccessRate"]

            if "avgMAE" in metrics_dict:
                if not np.isnan(metrics_dict["avgMAE"]["MAE"]):
                    metrics["sum_MAE"] += metrics_dict["avgMAE"]["MAE"]
                    metrics["count_valid_AE"] += 1

            if "avgMRE" in metrics_dict:
                if not np.isnan(metrics_dict["avgMRE"]["MRE"]):
                    metrics["sum_MRE"] += metrics_dict["avgMRE"]["MRE"]
                    metrics["count_valid_RE"] += 1
                    metrics["count_RE_ls"] = _update_re_counts(metrics_dict["avgMRE"]["MRE"], metrics["count_RE_ls"])

            if "avgIoU" in metrics_dict:
                if not np.isnan(metrics_dict["avgIoU"]["IoU"]):
                    metrics["sum_IoU"] += metrics_dict["avgIoU"]["IoU"]
                    metrics["count_valid_IoU"] += 1

            metrics["num_success"] += metrics_dict["SuccessRate"]["success"]
            count_total += 1

            if task_type == 'Detection':
                _patch_doc_detection_task(data, doc)

            temp.write(json.dumps(data, default=convert_numpy_to_python) + "\n")

            # Limit the number of processed samples if limit is set
            if limit is not None and count_total == limit:
                print(
                    f"[Warning] Reached limit of {limit} samples for file {jsonl_file}. Stopping processing.")
                break

    return metrics, count_total


def _process_model_directory(model_dir, task_type, limit, skip_existing):
    # For loop to open all *.jsonl files in model_dir
    jsonl_files = glob.glob(os.path.join(model_dir, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {model_dir}")

    for jsonl_file in jsonl_files:

        # Get parsed file path: model_dir/parsed/*.jsonl
        parsed_file_path = _get_parsed_file_path(model_dir, jsonl_file)

        if skip_existing and os.path.exists(parsed_file_path):
            print(
                f"[Info] Parsed file already exists at {parsed_file_path}. Skipping as per 'skip_existing' flag.")
            continue

        # Load existing results summary file for the jsonl_file
        results_summary_data, results_json_file = _load_results_file(
            jsonl_file)

        temp_file = jsonl_file + ".temp"
        metrics, count_total = _process_jsonl_file(
            jsonl_file, temp_file, task_type, limit)

        results_summary_data = _update_results_summary(
            results_summary_data, metrics, count_total)

        os.makedirs(os.path.dirname(parsed_file_path), exist_ok=True)
        os.replace(temp_file, parsed_file_path)
        print(f"[Info] Saved parsed data to {parsed_file_path}")

        parsed_results_json_path = os.path.join(
            os.path.dirname(parsed_file_path), results_json_file)
        with open(parsed_results_json_path, "w") as f:
            json.dump(results_summary_data, f, indent=2)
        print(
            f"[Info] Saved updated results summary to {parsed_results_json_path}")


def main(**kwargs):
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    task_type = kwargs.get("task_type")
    limit = kwargs.get("limit")
    skip_existing = kwargs.get("skip_existing", False)

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directory within this folder will be looped over, and each JSONL file will be processed.")

        # Get list of model folder within task_dir
        model_dirs = []
        for entry in os.scandir(task_dir):
            if entry.is_dir():
                model_dirs.append(entry.path)

        # Loop over each model directory and process JSONL files
        for model_dir in model_dirs:
            print(f"\nProcessing model directory: {model_dir}")
            _process_model_directory(
                model_dir, task_type, limit, skip_existing)

    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all JSONL files within this directory.")
        _process_model_directory(model_dir, task_type, limit, skip_existing)

    else:
        raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse benchmark output JSONL files and update summaries."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Type of the task to process: ['AD', 'TL', 'Detection'].",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        help="Path to the benchmark result directory for a specific task where model results directory is located.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the model results directory containing JSONL files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of samples to process per JSONL file.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing files that already have parsed outputs.",
    )

    args = parser.parse_args()
    assert args.task_type in [
        'AD', 'TL', 'Detection'], "task_type must be one of ['AD', 'TL', 'Detection']"
    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
