"""Summarize Tumor/Lesion (TL) size-estimation benchmark results across models.

Reads the parsed ``*.jsonl`` prediction files under each model's ``parsed``
subdirectory, groups samples by anatomy/modality/slice, and aggregates MAE,
MRE, nMAE, success rate, and threshold-based accuracies. Per-model summaries
are written as JSON (raw values and metrics) and a formatted cross-model report
is saved to a text file. Samples listed in per-dataset removed-samples files can
optionally be excluded.

Run as a CLI; see :func:`parse_args` for the accepted arguments.
"""

import argparse
import ast
import glob
import json
import multiprocessing
import os
import re
from pathlib import Path

import numpy as np

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    _compute_physical_diagonal,
)
from medvision_bm.utils.configs import (
    EXCLUDED_KEYS,
    MINIMUM_GROUP_SIZE,
    SUMMARY_FILENAME_TL_METRICS,
    SUMMARY_FILENAME_TL_VALUES,
    TUMOR_LESION_GROUP_KEYS,
    label_map_rename,
)
from medvision_bm.utils.parse_utils import (
    convert_numpy_to_python,
    get_labelsMap_imgModality_from_biometry_benchmark_plan,
    get_subfolders,
    get_targetLabel_imgModality_from_biometry_benchmark_plan,
    group_by_label_modality_slice,
)


def cal_metrics_TL_task(results):
    """Calculate metrics for a single Tumor/Lesion (TL) size-estimation sample.

    Parses the two predicted size values from ``filtered_resps`` and compares
    them with the ground-truth ``target`` to compute the mean absolute error
    (MAE), mean relative error (MRE), and a success flag. When metadata is
    available, a normalized MAE (nMAE) is read from a precomputed value or
    reconstructed by dividing the MAE by the physical image diagonal.

    Args:
        results (dict): A single sample with keys:

            - ``filtered_resps`` (list): One-element list holding the
              prediction string (expected to contain two comma-separated
              values).
            - ``target`` (str): Ground-truth values, parsed with
              ``ast.literal_eval``.
            - ``doc_meta`` (dict, optional): Metadata used to derive nMAE.

    Returns:
        dict: Metric entries whose keys match the task YAML ``metric`` fields:

            - ``avgMAE``: ``{"MAE": float, "success": bool}``
            - ``avgMRE``: ``{"MRE": float, "success": bool}``
            - ``SuccessRate``: ``{"success": bool}``
            - ``nMAE``: ``{"NMAE": float, "success": bool}``

    Note:
        MAE/MRE are ``np.nan`` and ``success`` is ``False`` when the prediction
        cannot be parsed or does not contain exactly two values.
    """
    pred = results["filtered_resps"][0]
    target_metrics = np.array(ast.literal_eval(results["target"]))
    try:
        # Split the prediction string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if len(pred_metrics) != 2:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metrics)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(absolute_error / (target_metrics + 1e-15))
            success = True
    except:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    doc_meta = results.get("doc_meta")
    nmae_precomputed = doc_meta.get("nmae_precomputed") if doc_meta else None

    if nmae_precomputed is not None:
        nmae_raw = nmae_precomputed.get("NMAE")
        nmae = float(nmae_raw) if nmae_raw is not None else np.nan
        nmae_success = bool(nmae_precomputed.get("success", False)) and np.isfinite(
            nmae
        )
    elif success and doc_meta is not None:
        # Fallback: recompute diagonal from stored or hash-derived scale.
        # Tier 2 (pixel_size_scale present): uses the scale factor stored at eval time — guaranteed correct.
        # Tier 3 (pixel_size_scale absent, old pre-fix JSONL): hash-based derivation; requires
        #   MEDVISION_SCALED_PS_LOW/HIGH env vars to match eval-time values for scaledPS tasks.
        pixel_size_scale = doc_meta.get("pixel_size_scale")
        try:
            diagonal = _compute_physical_diagonal(
                doc_meta,
                scale_mode=doc_meta.get("scale_mode"),
                explicit_scale=pixel_size_scale,
            )
            nmae = float(mean_absolute_error) / diagonal
            nmae_success = True
        except Exception:
            nmae = np.nan
            nmae_success = False
    else:
        nmae = np.nan
        nmae_success = False

    # NOTE: These key names must match the "metric" field in the task's YAML configuration file
    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgMRE": {"MRE": mean_relative_error, "success": success},
        "SuccessRate": {"success": success},
        "nMAE": {"NMAE": nmae, "success": nmae_success},
    }


def _initialize_metric_counters_TL_task():
    """Initialize all metric counters for TL task aggregation."""
    return {
        "sum_MAE": 0,
        "sum_MRE": 0,
        "num_success": 0,
        "count_valid_AE": 0,  # Count of valid absolute error samples
        "count_valid_RE": 0,  # Count of valid relative error samples
        # Counts for AE thresholds [0.0-0.1), [0.1-0.2), ..., [0.9-1.0+]
        "count_AE_thresholds": [0] * 10,
        # Counts for RE thresholds [0.0-0.1), [0.1-0.2), ..., [0.9-1.0+]
        "count_RE_thresholds": [0] * 10,
        "sum_NMAE": 0,
        "count_valid_NMAE": 0,
    }


def _update_mae_counters(value, counters):
    """
    Update Mean Absolute Error (MAE) related counters.

    Args:
        value: MAE value to add
        counters: Dictionary of counters to update
    """
    # Guard against non-finite values (inf, -inf, nan) that cannot be
    # converted to int or meaningfully summed.
    if np.isfinite(value):
        counters["sum_MAE"] += value
        counters["count_valid_AE"] += 1

        # Assign to threshold bucket (bucket 0 = [0.0, 0.1), ..., bucket 9 = [0.9, inf))
        threshold_index = min(int(value * 10), 9)
        counters["count_AE_thresholds"][threshold_index] += 1


def _update_mre_counters(value, counters):
    """
    Update Mean Relative Error (MRE) related counters.

    Args:
        value: MRE value to add
        counters: Dictionary of counters to update
    """
    # Guard against non-finite values (inf, -inf, nan) that cannot be
    # converted to int or meaningfully summed.
    if np.isfinite(value):
        counters["sum_MRE"] += value
        counters["count_valid_RE"] += 1

        # Assign to threshold bucket (bucket 0 = [0.0, 0.1), ..., bucket 9 = [0.9, inf))
        threshold_index = min(int(value * 10), 9)
        counters["count_RE_thresholds"][threshold_index] += 1


def _update_metric_counters_TL_task(metrics_dict, counters):
    """Update all metric counters based on calculated metrics."""
    # Update MAE counters (skip nan and inf — non-finite values are unparseable
    # responses and should not contribute to the running average)
    if np.isfinite(metrics_dict["avgMAE"]["MAE"]):
        _update_mae_counters(metrics_dict["avgMAE"]["MAE"], counters)

    # Update MRE counters
    if np.isfinite(metrics_dict["avgMRE"]["MRE"]):
        _update_mre_counters(metrics_dict["avgMRE"]["MRE"], counters)

    # Update success count
    counters["num_success"] += metrics_dict["SuccessRate"]["success"]

    # Update nMAE counters
    if metrics_dict["nMAE"]["success"] and np.isfinite(metrics_dict["nMAE"]["NMAE"]):
        counters["sum_NMAE"] += metrics_dict["nMAE"]["NMAE"]
        counters["count_valid_NMAE"] += 1


def _calculate_final_metrics_TL_task(counters, count_total):
    """
    Calculate final aggregated metrics from counters.

    Args:
        counters: Dictionary of accumulated counters
        count_total: Total number of samples processed

    Returns:
        Dictionary with final computed metrics including MAE<k and MRE<k cumulative accuracies
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
        "avgNMAE": (
            counters["sum_NMAE"] / counters["count_valid_NMAE"]
            if counters["count_valid_NMAE"] > 0
            else np.nan
        ),
        "num_samples": count_total,
    }

    # Add cumulative accuracy metrics: MAE<k and MRE<k for k in [0.1, 0.2, ..., 1.0]
    keys = ["RE"]
    for key in keys:
        for k in range(1, 11):
            cumulative_count = sum(counters[f"count_{key}_thresholds"][0:k])
            task_metrics[f"M{key}<{k/10:.1f}"] = (
                cumulative_count / count_total if count_total > 0 else 0.0
            )

    return task_metrics


def process_label_group_TL(parent_class, data):
    """
    Helper function to process metrics for a single anatomy group (parent_class).
    Used for both sequential and parallel processing.
    """
    if parent_class is None:
        return parent_class, None

    targets = data["targets"]
    responses = data["responses"]
    doc_metas = data.get("doc_metas", [None] * len(targets))

    # Skip if targets or responses are empty
    if not targets or not responses:
        return parent_class, None

    # Initialize counters
    counters = _initialize_metric_counters_TL_task()
    count_total = len(targets)

    # Process each target-response pair
    for target, response, doc_meta in zip(targets, responses, doc_metas):
        mock_results = {
            "filtered_resps": [response],
            "target": target,
            "doc_meta": doc_meta,
        }
        metrics_dict = cal_metrics_TL_task(mock_results)
        _update_metric_counters_TL_task(metrics_dict, counters)

    # Calculate and store final metrics
    task_metrics = _calculate_final_metrics_TL_task(counters, count_total)
    return parent_class, task_metrics


def calculate_summary_metrics_per_anatomy_TL_task(grouped_data, processes=None):
    """
    Calculate summary metrics for each anatomy group.

    Args:
        grouped_data: Dictionary with parent_class as keys and task_data as values
        processes (int, optional): Number of processes to use for parallel calculation.

    Returns:
        Dictionary with summary metrics per parent class and task type
    """
    summary_metrics = {}

    # Prepare items for processing
    items = list(grouped_data.items())

    if processes is not None and processes > 1:
        print(f"Calculating metrics with {processes} processes...")
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(process_label_group_TL, items)
    else:
        results = [
            process_label_group_TL(parent_class, data) for parent_class, data in items
        ]

    # Collect results
    for parent_class, task_metrics in results:
        if task_metrics is not None:
            summary_metrics[parent_class] = task_metrics

    return summary_metrics


def _build_removed_set(json_path):
    """Load a removed-samples JSON file and return a frozenset of (relative_image_file, slice_dim_int, slice_idx, task_id) keys."""
    _dim_map = {"x": 0, "y": 1, "z": 2}
    with open(json_path) as f:
        entries = json.load(f)
    return frozenset(
        (
            e["image_file"],
            _dim_map[e["slice_dim"]],
            int(e["slice_idx"]),
            int(e["task_ID"]),
        )
        for e in entries
    )


def _relative_image_file(full_path, dataset_name):
    """Extract the relative image file path (after the dataset-name component) from an absolute path."""
    marker = f"/{dataset_name}/"
    idx = full_path.find(marker)
    return full_path[idx + len(marker) :] if idx >= 0 else Path(full_path).name


def process_jsonl_file_TL_task(
    jsonl_path,
    limit=None,
    removed_set=None,
):
    """
    Process a JSONL file and extract relevant fields for TL task evaluation.

    Args:
        jsonl_path: Path to the JSONL file
        limit: Maximum number of samples to process (None for no limit)

    Returns:
        List of tuples: (imgModality, label_name, target, filtered_resps, task_id, slice_dim)
    """
    results = []
    # Extract dataset name from filename pattern 'samples_{dataset_name}_'
    match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    dataset_name = match.group(1)
    scale_mode = "uniform" if "scaledPS" in os.path.basename(jsonl_path) else None

    count = 0
    with open(jsonl_path, "r") as f:
        for _, line in enumerate(f):
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())

                if not data:
                    continue

                # Extract required fields from the JSONL entry
                doc = data.get("doc", {})
                slice_dim = doc.get("slice_dim")
                task_id = int(doc.get("taskID"))
                filtered_resps = data.get("filtered_resps")
                target = data.get("target")

                # Skip samples removed in the updated dataset
                if removed_set is not None:
                    _img = doc.get("image_file", "")
                    _key = (
                        _relative_image_file(_img, dataset_name),
                        slice_dim,
                        doc.get("slice_idx"),
                        task_id,
                    )
                    if _key in removed_set:
                        count += 1
                        continue

                # Get label from benchmark plan
                label, _ = get_targetLabel_imgModality_from_biometry_benchmark_plan(
                    dataset_name, task_id
                )

                if (
                    label is not None
                    and task_id is not None
                    and filtered_resps is not None
                    and target is not None
                ):
                    # Get label mapping and image modality from benchmark plan
                    labels_map, imgModality = (
                        get_labelsMap_imgModality_from_biometry_benchmark_plan(
                            dataset_name, task_id
                        )
                    )
                    label_name = labels_map.get(str(label))
                    if label_name:
                        doc_meta = {
                            "image_file": doc.get("image_file"),
                            "slice_dim": doc.get("slice_dim"),
                            "slice_idx": doc.get("slice_idx"),
                            "image_size_2d": doc.get("image_size_2d"),
                            "scale_mode": scale_mode,
                            "nmae_precomputed": data.get("nMAE"),
                            "taskID": doc.get("taskID"),
                            "label": doc.get("label"),
                            "pixel_size_scale": data.get("pixel_size_scale"),
                        }
                        results.append(
                            (
                                imgModality,
                                label_name,
                                target,
                                filtered_resps,
                                task_id,
                                slice_dim,
                                doc_meta,
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
    processes=None,
    removed_samples_dir=None,
    removed_samples_filename=None,
):
    """
    Process all JSONL files in a model folder and generate summary metrics.

    Args:
        model_dir: Path to the model folder
        limit: Maximum number of samples to process per file (None for no limit)
        processes (int, optional): Number of processes to use for parallel calculation.
        removed_samples_dir (str, optional): Root directory containing per-dataset
            removed_samples JSON files. When provided, matching samples are excluded.
        removed_samples_filename (str, optional): Filename within each dataset subdirectory.
    """
    # Find parsed JSONL files
    parsed_files_dir = os.path.join(model_dir, "parsed")

    # # Option 1: Early exit if parsed directory does not exist
    # assert os.path.exists(
    #     parsed_files_dir
    # ), f"Parsed files directory does not exist: {parsed_files_dir}"

    # Option 2: Warning and skip if parsed directory does not exist
    if not os.path.exists(parsed_files_dir):
        print(f"Parsed files directory does not exist: {parsed_files_dir}, skipping...")
        return

    jsonl_files = [
        f
        for f in glob.glob(os.path.join(parsed_files_dir, "*.jsonl"))
        if not ("_proc_acc" in os.path.basename(f) or "_eq_acc" in os.path.basename(f))
    ]

    # Collect all data from the parsed JSONL files
    _removed_cache = {}  # dataset_name → frozenset | None
    all_data = []
    for jsonl_file in jsonl_files:
        removed_set = None
        if removed_samples_dir:
            match = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_file))
            ds_name = match.group(1) if match else None
            if ds_name and ds_name not in _removed_cache:
                fname = removed_samples_filename
                json_path = os.path.join(removed_samples_dir, ds_name, fname)
                _removed_cache[ds_name] = (
                    _build_removed_set(json_path) if os.path.exists(json_path) else None
                )
            removed_set = _removed_cache.get(ds_name) if ds_name else None
        file_data = process_jsonl_file_TL_task(
            jsonl_file, limit, removed_set=removed_set
        )
        all_data.extend(file_data)

    # Skip processing if no data was collected
    if not all_data:
        print(f"No valid data found in {parsed_files_dir}, skipping...")
        return

    # group_by_label_modality_slice expects 6-tuples; strip the 7th doc_meta element
    all_data_6 = [t[:6] for t in all_data]

    # Group by parent class
    grouped_data = group_by_label_modality_slice(all_data_6)

    # Build parallel doc_metas using the same key construction as group_by_label_modality_slice
    _imgmod_map = {
        "MRI": "MR",
        "CT": "CT",
        "ultrasound": "US",
        "X-ray": "XR",
        "PET": "PET",
    }
    _slice_map = {0: "S", 1: "C", 2: "A"}
    for t in all_data:
        imgModality, label_name, _tgt, _resp, _tid, slice_dim, doc_meta = t
        new_label = label_map_rename.get(label_name)
        img_mod = _imgmod_map.get(imgModality, imgModality)
        slicetype = _slice_map.get(slice_dim)
        if new_label is None or slicetype is None:
            continue
        key = f"{new_label} @ {img_mod} ({slicetype})"
        if key in grouped_data:
            grouped_data[key].setdefault("doc_metas", []).append(doc_meta)

    # Skip if no grouped data
    if not grouped_data:
        print(f"No grouped data found for {parsed_files_dir}, skipping...")
        return

    # Calculate summary metrics per anatomy
    summary_metrics = calculate_summary_metrics_per_anatomy_TL_task(
        grouped_data, processes=processes
    )

    # Build filename suffix: _filtered and/or _limit{N}
    _suffix = ("_filtered" if removed_samples_dir else "") + (
        f"_limit{limit}" if limit is not None else ""
    )

    # Save values JSON file
    values_filename = (
        f"{SUMMARY_FILENAME_TL_VALUES.removesuffix('.json')}{_suffix}.json"
    )
    values_path = os.path.join(parsed_files_dir, values_filename)
    with open(values_path, "w") as f:
        json.dump(convert_numpy_to_python(grouped_data), f, indent=2)
    print(f"Saved target and model-predicted values to {values_path}")

    # Save summary metrics JSON file
    metrics_filename = (
        f"{SUMMARY_FILENAME_TL_METRICS.removesuffix('.json')}{_suffix}.json"
    )
    metrics_path = os.path.join(parsed_files_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved summary metrics to {metrics_path}")


def print_model_summaries(
    task_dir, limit=None, skip_model_wo_parsed_files=False, removed_samples_dir=None
):
    """
    Print and save summary metrics for all models in a task directory.

    Args:
        task_dir: Path to the task directory containing model folders
        limit: Maximum number of samples to process per file (None for no limit)
        skip_model_wo_parsed_files: Whether to skip models without parsed folders
        removed_samples_dir (str, optional): When provided, reads filtered metrics files.
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Build filename suffix consistent with process_parsed_file_in_model_folder
    _suffix = ("_filtered" if removed_samples_dir else "") + (
        f"_limit{limit}" if limit is not None else ""
    )

    # Prepare output file path
    output_filename = f"summary_TL_task{_suffix}.txt"
    output_file_path = os.path.join(task_dir, output_filename)

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

        metrics_filename = (
            f"{SUMMARY_FILENAME_TL_METRICS.removesuffix('.json')}{_suffix}.json"
        )
        metrics_file = os.path.join(parsed_dir, metrics_filename)

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
            "weighted_sum_nmae": 0.0,
            "weighted_avg_nmae": None,
            "weighted_nmae_count": 0,
            "weighted_mre<0.1": None,
            "weighted_mre<0.2": None,
            "weighted_mre<0.3": None,
        }

        # Aggregate metrics across all labels and task types
        for label, task_metrics in metrics.items():
            mae = task_metrics.get("avgMAE")
            mre = task_metrics.get("avgMRE")
            sr = task_metrics.get("SuccessRate")
            nmae = task_metrics.get("avgNMAE")
            samples = task_metrics.get("num_samples", 0)
            mre_lt_01 = task_metrics.get("MRE<0.1")
            mre_lt_02 = task_metrics.get("MRE<0.2")
            mre_lt_03 = task_metrics.get("MRE<0.3")

            if mre is not None and not np.isnan(mre) and samples > 0:
                if label not in model_summary["labels"]:
                    model_summary["labels"][label] = {
                        "mae": mae,
                        "mre": mre,
                        "sr": sr,
                        "nmae": nmae,
                        "samples": samples,
                        "MRE<0.1": mre_lt_01,
                        "MRE<0.2": mre_lt_02,
                        "MRE<0.3": mre_lt_03,
                    }
                else:
                    # Merge metrics from multiple task types for the same label using weighted average
                    prev = model_summary["labels"][label]
                    prev_samples = prev["samples"]
                    new_total = prev_samples + samples

                    def wavg(old_val, new_val):
                        """Calculate weighted average, handling None and NaN values"""
                        if (old_val is None or np.isnan(old_val)) and (
                            new_val is None or np.isnan(new_val)
                        ):
                            return None
                        if old_val is None or np.isnan(old_val):
                            return new_val
                        if new_val is None or np.isnan(new_val):
                            return old_val
                        return (old_val * prev_samples + new_val * samples) / new_total

                    prev["mae"] = wavg(prev["mae"], mae)
                    prev["mre"] = wavg(prev["mre"], mre)
                    prev["sr"] = wavg(prev["sr"], sr)
                    prev["nmae"] = wavg(prev["nmae"], nmae)
                    prev["MRE<0.1"] = wavg(prev["MRE<0.1"], mre_lt_01)
                    prev["MRE<0.2"] = wavg(prev["MRE<0.2"], mre_lt_02)
                    prev["MRE<0.3"] = wavg(prev["MRE<0.3"], mre_lt_03)
                    prev["samples"] = new_total

                if mae is not None and not np.isnan(mae):
                    model_summary["weighted_sum_mae"] += mae * samples
                model_summary["weighted_sum_mre"] += mre * samples
                model_summary["weighted_sum_sr"] += sr * samples
                if nmae is not None and not np.isnan(nmae):
                    model_summary["weighted_sum_nmae"] += nmae * samples
                    model_summary["weighted_nmae_count"] += samples
                model_summary["total_samples"] += samples

        # Calculate overall weighted averages across all labels
        if model_summary["total_samples"] > 0:
            model_summary["weighted_avg_mae"] = (
                model_summary["weighted_sum_mae"] / model_summary["total_samples"]
            )
            model_summary["weighted_avg_mre"] = (
                model_summary["weighted_sum_mre"] / model_summary["total_samples"]
            )
            model_summary["weighted_avg_sr"] = (
                model_summary["weighted_sum_sr"] / model_summary["total_samples"]
            )
        if model_summary["weighted_nmae_count"] > 0:
            model_summary["weighted_avg_nmae"] = (
                model_summary["weighted_sum_nmae"]
                / model_summary["weighted_nmae_count"]
            )

        # Compute micro-averaged (sample-weighted) MRE<k accuracy metrics
        wsum_re01 = wsum_re02 = wsum_re03 = 0.0
        wcount_re01 = wcount_re02 = wcount_re03 = 0
        for _m in model_summary["labels"].values():
            samples_lbl = _m.get("samples", 0)
            if samples_lbl <= 0:
                continue
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

    # Print summary table for each model
    for model, summary in model_summaries.items():
        model_header = f"\nModel: {model}"
        if summary["weighted_avg_mae"] is None:
            model_header += " (No valid MAE samples)"
            print_and_capture(model_header)
            continue
        nmae_avg_str = (
            f"{summary['weighted_avg_nmae']:.4f}"
            if summary["weighted_avg_nmae"] is not None
            else "N/A"
        )
        weighted_avg = (
            f"Weighted Average MAE: {summary['weighted_avg_mae']:.4f}, "
            f"MRE: {summary['weighted_avg_mre']:.4f}, "
            f"SR: {summary['weighted_avg_sr']:.4f}, "
            f"nMAE: {nmae_avg_str} (Total Samples: {summary['total_samples']})"
        )
        acc_line_parts = []
        if summary["weighted_mre<0.1"] is not None:
            acc_line_parts.append(
                f"Weighted MRE<0.1: {summary['weighted_mre<0.1']:.4f}"
            )
        if summary["weighted_mre<0.2"] is not None:
            acc_line_parts.append(
                f"Weighted MRE<0.2: {summary['weighted_mre<0.2']:.4f}"
            )
        if summary["weighted_mre<0.3"] is not None:
            acc_line_parts.append(
                f"Weighted MRE<0.3: {summary['weighted_mre<0.3']:.4f}"
            )
        acc_summary = (
            " | ".join(acc_line_parts) if acc_line_parts else "No MRE<k metrics"
        )
        label_header = "\nLabel-specific metrics:"
        table_header = (
            f"{'Label':<50}  | {'MAE':<8} | {'MRE':<8} | {'SR':<8} | {'nMAE':<8} | "
            f"{'MRE<0.1':<8} | {'MRE<0.2':<8} | {'MRE<0.3':<8} | {'Samples':<8}"
        )
        separator = "-" * 146

        print_and_capture(model_header)
        print_and_capture(weighted_avg)
        print_and_capture(acc_summary)
        print_and_capture(label_header)
        print_and_capture(table_header)
        print_and_capture(separator)

        # Sort labels by sample count (descending) for better readability
        sorted_labels = sorted(
            summary["labels"].items(), key=lambda x: x[1]["samples"], reverse=True
        )

        for label, metrics in sorted_labels:
            mae = metrics.get("mae")
            nmae_lbl = metrics.get("nmae")
            re01 = metrics.get("MRE<0.1")
            re02 = metrics.get("MRE<0.2")
            re03 = metrics.get("MRE<0.3")
            line = (
                f"{label:<50}  | "
                f"{(mae if mae is not None else float('nan')):<8.4f} | "
                f"{metrics['mre']:<8.4f} | {metrics['sr']:<8.4f} | "
                f"{(nmae_lbl if nmae_lbl is not None and not np.isnan(nmae_lbl) else float('nan')):<8.4f} | "
                f"{(re01 if re01 is not None else float('nan')):<8.4f} | "
                f"{(re02 if re02 is not None else float('nan')):<8.4f} | "
                f"{(re03 if re03 is not None else float('nan')):<8.4f} | "
                f"{metrics['samples']:<8}"
            )
            print_and_capture(line)

        section_end = "\n" + "=" * 100 + "\n"
        print_and_capture(section_end)

    # Save all printed output to text file
    with open(output_file_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Summary saved to {output_file_path}")


def _process_task_directory(
    task_dir,
    limit,
    processes=None,
    skip_model_wo_parsed_files=False,
    removed_samples_dir=None,
    removed_samples_filename=None,
):
    """
    Process all model directories within a task directory.

    Args:
        task_dir: Path to the task directory containing model folders
        limit: Maximum number of samples to process per file
        processes (int, optional): Number of processes to use for parallel calculation
        skip_model_wo_parsed_files: Whether to skip model directories without parsed folders
        removed_samples_dir (str, optional): Root directory with per-dataset removed_samples JSON files.
        removed_samples_filename (str, optional): Filename within each dataset subdirectory.
    """
    # Get list of model folders within task_dir
    model_dirs = get_subfolders(task_dir)

    # Print configuration info once at the beginning
    print("\nConfigurations in medvision_bm/utils/configs.py:")
    print(f"  TUMOR_LESION_GROUP_KEYS: {TUMOR_LESION_GROUP_KEYS}")
    print(f"  EXCLUDED_KEYS: {EXCLUDED_KEYS}")
    print(f"  MINIMUM_GROUP_SIZE: {MINIMUM_GROUP_SIZE}\n")

    # Loop over each model directory and process JSONL files
    for model_dir in model_dirs:
        # Skip if parsed folder doesn't exist and flag is set
        parsed_files_dir = os.path.join(model_dir, "parsed")
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no parsed folder): {model_dir}")
            continue

        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(
            model_dir,
            limit,
            processes=processes,
            removed_samples_dir=removed_samples_dir,
            removed_samples_filename=removed_samples_filename,
        )

    # Print summary metrics at the end
    print_model_summaries(
        task_dir,
        limit,
        skip_model_wo_parsed_files,
        removed_samples_dir=removed_samples_dir,
    )


def _process_single_model_directory(
    model_dir,
    limit,
    processes=None,
    removed_samples_dir=None,
    removed_samples_filename=None,
):
    """
    Process a single model directory.

    Args:
        model_dir: Path to the model directory
        limit: Maximum number of samples to process per file
        processes (int, optional): Number of processes to use for parallel calculation
        removed_samples_dir (str, optional): Root directory with per-dataset removed_samples JSON files.
        removed_samples_filename (str, optional): Filename within each dataset subdirectory.
    """
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(
        model_dir,
        limit,
        processes=processes,
        removed_samples_dir=removed_samples_dir,
        removed_samples_filename=removed_samples_filename,
    )


def main(**kwargs):
    """Process TL model folders based on the provided arguments.

    Dispatches to task_dir mode (loop over all model directories and generate a
    cross-model summary) or model_dir mode (process a single model directory).

    Args:
        task_dir (str, optional): Path to task directory (mutually exclusive
            with model_dir).
        model_dir (str, optional): Path to model directory (mutually exclusive
            with task_dir).
        limit (int, optional): Maximum number of samples to process per file.
        skip_model_wo_parsed_files (bool): Whether to skip model directories
            without parsed folders.
        processes (int, optional): Number of processes to use for parallel
            calculation.
        removed_samples_dir (str, optional): Root directory with per-dataset
            removed_samples JSON files.
        removed_samples_filename (str, optional): Filename within each dataset
            subdirectory.

    Raises:
        ValueError: If neither task_dir nor model_dir is provided.
    """
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    limit = kwargs.get("limit")
    skip_model_wo_parsed_files = kwargs.get("skip_model_wo_parsed_files", False)
    processes = kwargs.get("processes")
    removed_samples_dir = kwargs.get("removed_samples_dir")
    removed_samples_filename = kwargs.get("removed_samples_filename")

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directories within this folder will be looped over."
        )
        _process_task_directory(
            task_dir,
            limit,
            processes=processes,
            skip_model_wo_parsed_files=skip_model_wo_parsed_files,
            removed_samples_dir=removed_samples_dir,
            removed_samples_filename=removed_samples_filename,
        )

    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all JSONL files within this directory."
        )
        _process_single_model_directory(
            model_dir,
            limit,
            processes=processes,
            removed_samples_dir=removed_samples_dir,
            removed_samples_filename=removed_samples_filename,
        )

    else:
        raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def parse_args():
    """Parse command line arguments."""
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
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=None,
        help="Number of worker processes for metric calculation.",
    )
    parser.add_argument(
        "--removed_samples_dir",
        type=str,
        default=None,
        help=(
            "Root directory containing per-dataset removed_samples JSON files "
            "(e.g. .../Data/Datasets). When provided, samples listed in those files "
            "are excluded from metric computation and output filenames get a '_filtered' suffix."
        ),
    )
    parser.add_argument(
        "--removed_samples_filename",
        type=str,
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Filename of the removed-samples JSON within each dataset subdirectory.",
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
