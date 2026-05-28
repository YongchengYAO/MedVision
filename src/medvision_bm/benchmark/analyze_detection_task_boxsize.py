import argparse
import glob
import json
import multiprocessing
import os
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from medvision_bm.utils.configs import (
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES,
)
from medvision_bm.utils.parse_utils import (
    convert_numpy_to_python,
    get_labelsMap_imgModality_from_seg_benchmark_plan,
    get_subfolders,
)

_BINS = [
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


def _find_boxcoordinate_jsonl_files(model_dir):
    parsed_dir = os.path.join(model_dir, "parsed")
    all_jsonl = glob.glob(os.path.join(parsed_dir, "*.jsonl"))
    return [f for f in all_jsonl if "_BoxCoordinate_" in os.path.basename(f)]


def _group_by_boxImgRatio(data):
    result = defaultdict(
        lambda: {"mae": [], "iou": [], "f1": [], "precision": [], "recall": [], "success": []}
    )
    for box_img_ratio, mae, iou, f1, precision, recall, success in data:
        label = "Box/Image >= 90%"
        for threshold, name in _BINS:
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


def _initialize_metric_counters():
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


def _update_threshold_counters(metric_value, threshold_counts):
    for i, t in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
        if metric_value >= t:
            threshold_counts[i] += 1


def _update_metric_counters(metrics_dict, counters):
    mae = metrics_dict["avgMAE"]["MAE"]
    if not np.isnan(mae):
        counters["sum_MAE"] += mae
        counters["count_valid_AE"] += 1
        counters["count_AE_thresholds"][min(int(mae * 10), 9)] += 1

    iou = metrics_dict["avgIoU"]["IoU"]
    if not np.isnan(iou):
        counters["sum_IoU"] += iou
        counters["count_valid_IoU"] += 1
        _update_threshold_counters(iou, counters["count_IoU_thresholds"])

    f1 = metrics_dict["F1"]["F1"]
    if not np.isnan(f1):
        counters["sum_F1"] += f1
        counters["count_valid_F1"] += 1
        _update_threshold_counters(f1, counters["count_F1_thresholds"])

    precision = metrics_dict["Precision"]["Precision"]
    if not np.isnan(precision):
        counters["sum_Precision"] += precision
        counters["count_valid_Precision"] += 1
        _update_threshold_counters(precision, counters["count_Precision_thresholds"])

    recall = metrics_dict["Recall"]["Recall"]
    if not np.isnan(recall):
        counters["sum_Recall"] += recall
        counters["count_valid_Recall"] += 1
        _update_threshold_counters(recall, counters["count_Recall_thresholds"])

    counters["num_success"] += metrics_dict["SuccessRate"]["success"]


def _calculate_final_metrics(counters, count_total):
    m = {
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
        m[f"MAE<{k/10:.1f}"] = (
            sum(counters["count_AE_thresholds"][0:k]) / count_total
            if count_total > 0
            else 0.0
        )

    for metric_name in ["IoU", "F1", "Precision", "Recall"]:
        for k in range(5, 10):
            count = counters[f"count_{metric_name}_thresholds"][k - 5]
            m[f"{metric_name}>{k/10:.1f}"] = (
                count / count_total if count_total > 0 else 0.0
            )

    return m


def calculate_summary_metrics_per_boxImgRatio(grouped_data):
    summary_metrics = {}

    for bin_label, data in grouped_data.items():
        if bin_label is None:
            continue

        f1s = data["f1"]
        if not f1s:
            continue

        counters = _initialize_metric_counters()
        count_total = len(f1s)

        for mae, iou, f1, prec, rec, success in zip(
            data["mae"], data["iou"], f1s, data["precision"], data["recall"], data["success"]
        ):
            metrics_dict = {
                "avgMAE": {"MAE": mae, "success": success},
                "avgIoU": {"IoU": iou},
                "F1": {"F1": f1},
                "Precision": {"Precision": prec},
                "Recall": {"Recall": rec},
                "SuccessRate": {"success": success},
            }
            _update_metric_counters(metrics_dict, counters)

        summary_metrics[bin_label] = _calculate_final_metrics(counters, count_total)

    return summary_metrics


def process_jsonl_file(jsonl_path, limit=None):
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

            # Read pre-computed per-sample metrics
            mae = data.get("avgMAE", {}).get("MAE")
            iou = data.get("avgIoU", {}).get("IoU")
            f1 = data.get("F1", {}).get("F1")
            precision = data.get("Precision", {}).get("Precision")
            recall = data.get("Recall", {}).get("Recall")
            success = data.get("SuccessRate", {}).get("success", False)

            if None in (mae, iou, f1, precision, recall):
                continue

            # Resolve box-to-image ratio: prefer explicit field, fall back to
            # bounding_boxes dimensions, then compute from relative target coords.
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
            label_name = labels_map.get(str(label))
            if label_name:
                results.append((box_img_ratio, mae, iou, f1, precision, recall, success))

            count += 1
            if limit is not None and count >= limit:
                break

    return results


def _process_wrapper(args):
    return process_jsonl_file(*args)


def process_parsed_file_in_model_folder(model_dir, limit=None, processes=None):
    """
    Process all BoxCoordinate JSONL files in a model's parsed folder and generate
    box-to-image-ratio grouped detection metrics.

    Args:
        model_dir: Path to the model folder containing a 'parsed' subdirectory
        limit: Maximum number of samples to process per file (None = process all)
        processes: Number of processes to use for parallel calculation
    """
    jsonl_files = _find_boxcoordinate_jsonl_files(model_dir)
    parsed_files_dir = os.path.join(model_dir, "parsed")

    if not jsonl_files:
        print(f"  No BoxCoordinate JSONL files found in {parsed_files_dir}, skipping.")
        return

    all_data = []
    if processes is not None and processes > 1:
        print(f"Processing JSONL files with {processes} processes...")
        items = [(f, limit) for f in jsonl_files]
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.imap_unordered(_process_wrapper, items)
            for file_data in tqdm(results, total=len(items), desc="Processing files"):
                all_data.extend(file_data)
    else:
        for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
            all_data.extend(process_jsonl_file(jsonl_file, limit))

    if not all_data:
        print(f"  No valid samples parsed in {parsed_files_dir}, skipping.")
        return

    grouped_data = _group_by_boxImgRatio(all_data)
    if not grouped_data:
        print(f"  Grouping produced no bins for {parsed_files_dir}, skipping.")
        return

    summary_metrics = calculate_summary_metrics_per_boxImgRatio(grouped_data)

    values_path = os.path.join(
        parsed_files_dir, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES
    )
    with open(values_path, "w") as f:
        json.dump(convert_numpy_to_python(grouped_data), f, indent=2)
    print(f"Saved values to {values_path}")

    metrics_path = os.path.join(
        parsed_files_dir, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS
    )
    with open(metrics_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def _process_task_directory(
    task_dir, limit, processes=None, skip_model_wo_parsed_files=False
):
    model_dirs = get_subfolders(task_dir)
    model_dirs = [d for d in model_dirs if os.path.basename(d) != "random_detection"]

    for model_dir in model_dirs:
        parsed_files_dir = os.path.join(model_dir, "parsed")
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(model_dir, limit, processes=processes)


def _process_single_model_directory(model_dir, limit, processes=None):
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(model_dir, limit, processes=processes)


def main(**kwargs):
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    limit = kwargs.get("limit")
    skip_model_wo_parsed_files = kwargs.get("skip_model_wo_parsed_files", False)
    processes = kwargs.get("processes")

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directories within this folder will be looped over."
        )
        _process_task_directory(
            task_dir, limit, processes=processes,
            skip_model_wo_parsed_files=skip_model_wo_parsed_files,
        )
    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all BoxCoordinate JSONL files within this directory."
        )
        _process_single_model_directory(model_dir, limit, processes=processes)
    else:
        raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze detection task performance grouped by bounding box size relative to image size. "
            "Reads BoxCoordinate JSONL files from model_dir/parsed/. Outputs "
            f"{SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS} and "
            f"{SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_VALUES} "
            "into each model's parsed/ folder."
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
        help="Path to a specific model directory containing a parsed/ subfolder.",
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
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=None,
        help="Number of worker processes for metric calculation.",
    )

    args = parser.parse_args()

    if args.task_dir is None and args.model_dir is None:
        parser.error("Either --task_dir or --model_dir must be provided.")

    if args.skip_model_wo_parsed_files and args.task_dir is None:
        parser.error("--skip_model_wo_parsed_files can only be used with --task_dir")

    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
