import argparse
import glob
import json
import os
import re

import pandas as pd

from medvision_bm.utils.parse_utils import (
    cal_metrics_detection_task,
    get_labelsMap_imgModality_from_seg_benchmark_plan, get_subfolders,
    get_targetLabel_imgModality_from_biometry_benchmark_plan)


def process_jsonl_file_detection_task_w_boxSize(
    jsonl_path,
    limit=None,
):
    """
    Parse a JSONL results file and extract detection task data with box size information.

    This function:
    1. Extracts dataset name from filename pattern
    2. Parses each line to extract label, target, response, task_id, and box-to-image ratio
    3. Resolves label names and image modality using benchmark plan configuration
    4. Returns structured data for downstream box size analysis

    Args:
        jsonl_path: Path to the JSONL file
        limit: Maximum number of samples to process (None = process all)

    Returns:
        List of tuples: (imgModality, planner_task_type, label_name, target, 
                        filtered_resps, box_img_ratio, task_id)
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
                box_img_ratio = data["box_img_ratio"]

                # Determine task type from filename
                if "MaskSize" in os.path.basename(jsonl_path):
                    planner_task_type = "MaskSize"
                elif "BoxSize" in os.path.basename(jsonl_path):
                    planner_task_type = "BoxSize"
                elif "BoxCoordinate" in os.path.basename(jsonl_path):
                    planner_task_type = "BoxCoordinate"
                elif "TumorLesionSize" in os.path.basename(jsonl_path):
                    planner_task_type = "TumorLesionSize"
                else:
                    raise ValueError(f"Unknown task type in file {jsonl_path}")

                # Get label using appropriate lookup method
                if planner_task_type == "TumorLesionSize":
                    label, _ = get_targetLabel_imgModality_from_biometry_benchmark_plan(
                        dataset_name, task_id
                    )
                else:
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
                                planner_task_type,
                                label_name,
                                target,
                                filtered_resps,
                                box_img_ratio,
                                task_id,
                            )
                        )
                count += 1
                if limit is not None and count >= limit:
                    break
            except json.JSONDecodeError:
                raise ValueError(
                    f"Error in parsing the JSON file {jsonl_path}")

    return results


def _normalize_modality(img_modality):
    """
    Normalize image modality names to standardized abbreviations.
    
    Converts full modality names (e.g., "MRI", "ultrasound") to their 
    standard abbreviated forms (e.g., "MR", "US").
    
    Args:
        img_modality: Original modality name
        
    Returns:
        Standardized modality abbreviation (e.g., "MR", "CT", "US", "XR", "PET")
    """
    modality_map = {
        "MRI": "MR",
        "CT": "CT",
        "ultrasound": "US",
        "X-ray": "XR",
        "PET": "PET"
    }
    return modality_map.get(img_modality, img_modality)


def _get_box_img_group(box_img_ratio):
    """
    Categorize box-to-image ratio into predefined size groups.
    
    Groups are defined in 0.05 increments from <0.05 to >=0.90, allowing
    for fine-grained analysis of detection performance by relative object size.
    
    Args:
        box_img_ratio: Ratio of bounding box size to image size (float between 0 and 1)
        
    Returns:
        String representing the ratio group (e.g., "<0.05", "0.05~0.10", ">=0.90")
    """
    thresholds = [
        (0.05, "<0.05"),
        (0.10, "0.05~0.10"),
        (0.15, "0.10~0.15"),
        (0.20, "0.15~0.20"),
        (0.25, "0.20~0.25"),
        (0.30, "0.25~0.30"),
        (0.35, "0.30~0.35"),
        (0.40, "0.35~0.40"),
        (0.45, "0.40~0.45"),
        (0.50, "0.45~0.50"),
        (0.55, "0.50~0.55"),
        (0.60, "0.55~0.60"),
        (0.65, "0.60~0.65"),
        (0.70, "0.65~0.70"),
        (0.75, "0.70~0.75"),
        (0.80, "0.75~0.80"),
        (0.85, "0.80~0.85"),
        (0.90, "0.85~0.90"),
    ]
    
    for threshold, group_label in thresholds:
        if box_img_ratio < threshold:
            return group_label
    return ">=0.90"


def collect_data(data):
    """
    Convert collected detection data to a pandas DataFrame with computed metrics.

    This function processes raw detection results and:
    1. Normalizes image modalities
    2. Creates composite labels combining anatomy and modality
    3. Categorizes box sizes into groups
    4. Calculates IoU, DSC, Precision, and Recall metrics for each sample
    5. Returns a structured DataFrame for analysis

    Args:
        data: List of tuples containing (imgModality, task_type, label_name, target, 
              filtered_resps, box_img_ratio, taskID)

    Returns:
        pandas DataFrame with columns: task_type, box_img_group, label, box_img_ratio,
        IoU, DSC, Precision, Recall
    """
    from medvision_bm.utils.configs import label_map_regroup

    df_data = []
    
    for (
        imgModality,
        task_type,
        label_name,
        target,
        filtered_resps,
        box_img_ratio,
        task_id,
    ) in data:
        # Normalize modality and create composite label
        parent_class = label_map_regroup.get(label_name)
        normalized_modality = _normalize_modality(imgModality)
        new_parent_class = f"{parent_class} @ {normalized_modality}"
        
        # Categorize box size
        box_img_group = _get_box_img_group(box_img_ratio)

        # Process each response
        for response in filtered_resps:
            # Calculate metrics
            mock_results = {"filtered_resps": [response], "target": target}
            metrics_dict = cal_metrics_detection_task(mock_results)

            # Append row data
            df_data.append({
                "task_type": task_type,
                "box_img_group": box_img_group,
                "label": new_parent_class,
                "box_img_ratio": box_img_ratio,
                "IoU": metrics_dict["avgIoU"]["IoU"],
                "DSC": metrics_dict["DSC"]["DSC"],
                "Precision": metrics_dict["Precision"]["Precision"],
                "Recall": metrics_dict["Recall"]["Recall"],
            })

    return pd.DataFrame(df_data)


def group_data_by_boxSize_label(df):
    """
    Aggregate metrics by box size group and anatomical label.
    
    Computes mean values for IoU, DSC, Precision, and Recall, along with 
    sample counts for each combination of box size group and label.
    
    Args:
        df: DataFrame with individual sample metrics
        
    Returns:
        DataFrame with aggregated metrics grouped by box_img_group and label,
        including columns: box_img_group, label, IoU (mean), sample_size, 
        DSC (mean), Precision (mean), Recall (mean)
    """
    # Group by box_img_group and label, then calculate averages and sample size
    result = (
        df.groupby(["box_img_group", "label"])
        .agg(
            {
                "IoU": ["mean", "count"],
                "DSC": "mean",
                "Precision": "mean",
                "Recall": "mean",
            }
        )
        .round(6)
    )

    # Flatten the column names
    result.columns = ["IoU", "sample_size", "DSC", "Precision", "Recall"]

    # Reset index to make box_img_group and label regular columns
    result = result.reset_index()

    return result


def process_parsed_file_in_model_folder(
    model_dir,
    limit=None,
):
    """
    Process all JSONL files in a model's parsed folder and generate box size analysis metrics.

    This function performs the complete analysis pipeline:
    1. Finds all JSONL files in model_dir/parsed/
    2. Parses each file to extract detection data with box size information
    3. Computes detection metrics (IoU, DSC, Precision, Recall) for each sample
    4. Groups data by box size ranges and anatomical labels
    5. Saves both fine-grained and summary metrics as CSV files:
       - metrics-boxSize-label-fine.csv: Individual sample metrics
       - summary_metrics_per_boxSize-label-fine.csv: Aggregated statistics

    Args:
        model_dir: Path to the model folder containing a 'parsed' subdirectory
        limit: Maximum number of samples to process per file (None = process all)
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
        file_data = process_jsonl_file_detection_task_w_boxSize(
            jsonl_file, limit)
        all_data.extend(file_data)

    # Early exit if no valid data found
    if not all_data:
        print(f"No valid data found in {parsed_files_dir}, skipping...")
        return

    # Collect data and return a DataFrame
    path_metrics_csv = os.path.join(
        parsed_files_dir, "metrics_per_boxSize_label.csv")
    df = collect_data(all_data)
    df.to_csv(
        path_metrics_csv,
        index=False,
        na_rep="NaN",
    )
    print(
        f"Saved metrics grouped by boxSize and label to CSV: {path_metrics_csv}")

    # Group data by boxSize and label, then calculate summary metrics
    path_metrics_summary_csv = os.path.join(
        parsed_files_dir, "summary_metrics_per_boxSize_label.csv")
    result_df = group_data_by_boxSize_label(df)
    result_df.to_csv(
        path_metrics_summary_csv,
        index=False,
        na_rep="NaN",
    )
    print(
        f"Saved summary (average) metrics grouped by boxSize and label to CSV: {path_metrics_summary_csv}")


def _process_task_directory(task_dir, limit, skip_model_wo_parsed_files=False):
    """
    Process all model directories within a task directory for box size analysis.

    This function loops through all model subdirectories in the task folder,
    processing each model's detection results and generating box size metrics.

    Args:
        task_dir: Path to task directory containing model result folders
        limit: Maximum samples to process per JSONL file (None = all)
        skip_model_wo_parsed_files: If True, skip models without parsed folders
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Process each model directory
    for model_dir in model_dirs:
        # Skip models without parsed results if requested
        parsed_files_dir = os.path.join(model_dir, "parsed")
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(
                f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(model_dir, limit)


def _process_single_model_directory(model_dir, limit):
    """
    Process a single model directory for box size analysis.

    Args:
        model_dir: Path to the model directory containing parsed results
        limit: Maximum number of samples to process per JSONL file (None = all)
    """
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(model_dir, limit)


def main(**kwargs):
    """
    Main function to analyze detection task performance by box size.

    Processes model results to generate metrics grouped by bounding box size 
    relative to image size. Supports two modes:
    - Task mode: Process all models in a task directory
    - Model mode: Process a single model directory

    Args:
        task_dir: Path to task directory containing model folders (mutually exclusive with model_dir)
        model_dir: Path to single model directory (mutually exclusive with task_dir)
        limit: Maximum number of samples to process per JSONL file (None = all)
        skip_model_wo_parsed_files: Whether to skip model directories without parsed folders
    """
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    limit = kwargs.get("limit")
    skip_model_wo_parsed_files = kwargs.get(
        "skip_model_wo_parsed_files", False)

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directories within this folder will be looped over."
        )
        _process_task_directory(
            task_dir, limit, skip_model_wo_parsed_files)

    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all JSONL files within this directory."
        )
        _process_single_model_directory(
            model_dir, limit)

    else:
        raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def parse_args():
    """
    Parse command line arguments for box size analysis.

    Supports two mutually exclusive modes:
    - Task mode (--task_dir): Process all model subdirectories within a task folder
    - Model mode (--model_dir): Process a single model directory

    Returns:
        Parsed command line arguments as argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze detection task performance grouped by bounding box size and anatomical label."
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
        parser.error(
            "--skip_model_wo_parsed_files can only be used with --task_dir")

    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
