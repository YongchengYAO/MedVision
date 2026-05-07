"""
Checkpointed / resumable variant of build_parquet_ds_with_testset.py.

Key differences vs build_parquet_ds_with_testset.py:
  1. Training data is processed in SHARDS (--shard_size, default 50 000).
     Each shard is formatted, cleaned, and written to a parquet file
     immediately after processing, then freed from RAM.
  2. A checkpoint.json tracks completed shards so that an OOM-killed job
     resumes from the last saved shard instead of starting over.
  3. After all shards are done the script stream-merges them into the final
     train_verl.parquet via PyArrow ParquetWriter (~one-shard-at-a-time RAM).
  4. num_workers_format_dataset defaults to 64 (instead of 32) but should be
     set to 64 in the calling shell script; worker count is the primary knob
     for controlling peak RAM during the map() phase.
  5. Test split support: pass --test_sample_limit_per_subset (or
     --test_sample_limit_task_* / --test_sample_limit) to also build
     test_verl.parquet from the MedVision _Test configs.
     The test set is small, so no sharding is applied.

Memory budget at shard_size=50 000, 64 workers, writer_batch_size=50:
  map() buffers : 64 * 50 * ~0.75 MB ≈  2.4 GB
  shard in Arrow: 50 000 * ~100 KB   ≈  5   GB
  peak per shard:                    ≈ 10–15 GB  (vs 200 GB pod RAM ✓)
  merge pass    :     1 shard        ≈  5   GB   (reading one shard at a time)
"""

import argparse
import gc
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psutil
import pyarrow.parquet as pq
from datasets import concatenate_datasets

from medvision_bm.rft.verl.verl_utils import (
    _format_data_AngleDistanceTask_CoT_verl,
    _format_data_AngleDistanceTask_verl,
    _format_data_DetectionTask_CoT_verl,
    _format_data_DetectionTask_verl,
    _format_data_TumorLesionTask_CoT_verl,
    _format_data_TumorLesionTask_verl,
)
from medvision_bm.sft.sft_utils import (
    _load_single_dataset,
    clean_dataset,
    format_dataset,
    get_cgroup_limited_cpus,
    load_split_limit_dataset,
    parse_sample_limits,
)
from medvision_bm.utils.configs import SEED

# Fields required by the Verl training framework.
_VERL_KEYS = [
    "prompt",
    "ground_truth",
    "data_source",
    "ability",
    "reward_model",
    "extra_info",
    "images",
]

# Dispatch table: task tag → {cot/no_cot} → format function.
# Used when multiple tasks are combined into a single shard.
_TASK_FORMAT_MAP = {
    "BoxSize": {
        "cot": _format_data_DetectionTask_CoT_verl,
        "no_cot": _format_data_DetectionTask_verl,
    },
    "BiometricsFromLandmarks": {
        "cot": _format_data_AngleDistanceTask_CoT_verl,
        "no_cot": _format_data_AngleDistanceTask_verl,
    },
    "TumorLesionSize": {
        "cot": _format_data_TumorLesionTask_CoT_verl,
        "no_cot": _format_data_TumorLesionTask_verl,
    },
}


# ---------------------------------------------------------------------------
# Dispatch format function (picklable — no function objects in fn_kwargs)
# ---------------------------------------------------------------------------

def _dispatch_format(example, use_cot, model_name, model_hf, new_shape_hw):
    """Apply the correct per-task format function based on the '_task_tag' column."""
    task_tag = example["_task_tag"]
    func_key = "cot" if use_cot else "no_cot"
    func = _TASK_FORMAT_MAP[task_tag][func_key]
    return func(example, model_name=model_name, model_hf=model_hf, new_shape_hw=new_shape_hw)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "completed_train_shards": [],
        "val_done": False,
        "test_done": False,
        "merged": False,
        "n_shards": None,
        "total_train": None,
    }


def _save_checkpoint(path, ckpt):
    with open(path, "w") as f:
        json.dump(ckpt, f, indent=2)


def _load_raw_test_dataset(tasks_json, tag_ds, test_limit, num_workers, download_mode, test_limit_per_subset=None):
    """Load raw test samples from MedVision *_Test HF configs (no images).

    Mirrors the loading block in ds_utils.load_split_limit_dataset_tr_val_ts.
    Returns a single HF Dataset (the combined raw test split).
    """
    with open(tasks_json) as f:
        tasks = list(json.load(f).keys())

    available_cpus = get_cgroup_limited_cpus()
    workers = min(num_workers, available_cpus, len(tasks))

    data_dir = os.environ.get("MedVision_DATA_DIR")
    assert data_dir is not None, "\n [Error] MedVision_DATA_DIR environment variable must be set."
    with open(os.path.join(data_dir, ".downloaded_datasets.json")) as f:
        downloaded = list(json.load(f).keys())
    for task in tasks:
        dataset_name = task.split(f"_{tag_ds}")[0]
        if f"dataset_{dataset_name}" not in downloaded:
            workers = 1
            break

    datasets_list = []
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                _load_single_dataset,
                "YongchengYAO/MedVision",
                task.split(f"_{tag_ds}")[0],
                task + "_Test",
                "test",
                download_mode=download_mode,
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                ds = future.result(timeout=120)
                if test_limit_per_subset is not None and test_limit_per_subset > 0 and test_limit_per_subset < len(ds):
                    ds = ds.shuffle(seed=SEED).select(range(test_limit_per_subset))
                datasets_list.append(ds)
                print(f"  ✓ Test {task} ({len(datasets_list)}/{len(tasks)})")
                if psutil.virtual_memory().percent > 80:
                    print(f"  ⚠️  High memory usage: {psutil.virtual_memory().percent}%")
            except Exception as exc:
                failed.append((task, str(exc)))
                print(f"  ❌ Test {task} failed: {exc}")

    if failed:
        raise RuntimeError(f"Failed to load test data for {len(failed)} task(s): {failed}")

    combined = concatenate_datasets(datasets_list)
    del datasets_list
    gc.collect()

    if test_limit is not None and test_limit > 0 and test_limit < len(combined):
        combined = combined.shuffle(seed=SEED).select(range(test_limit))

    return combined


# ---------------------------------------------------------------------------
# Argument parser (mirrors build_parquet_ds_with_testset.py plus --shard_size)
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Build parquet dataset for RL finetuning in Verl framework "
            "(checkpointed / resumable variant, with test split)."
        )
    )
    # -- Model identifier
    parser.add_argument(
        "--model_family_name",
        type=str,
        required=True,
        help="Model family name, used to identify the model groups that share the same image processor.",
    )
    parser.add_argument(
        "--model_hf",
        type=str,
        required=True,
        help="Model Hugging Face identifier, used to load the model's image processor for dataset preparation.",
    )
    # -- Data folder
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset folder",
    )
    parser.add_argument(
        "--prepared_ds_dir",
        type=str,
        help="Path to the prepared dataset directory to load from disk",
    )
    # -- Dataset download mode
    parser.add_argument(
        "--ds_download_mode",
        type=str,
        default="reuse_dataset_if_exists",
        help="Dataset download mode: 'reuse_dataset_if_exists' (default), 'reuse_cache_if_exists', 'force_redownload'",
    )
    # -- Data processing
    parser.add_argument(
        "--new_shape_hw",
        default=None,
        type=int,
        nargs=2,
        help="Target resize shape as (height, width). Ignore to use the original size. Example: --new_shape_hw 1080 1920. Result: args.new_shape_hw → [1080, 1920]",
    )
    parser.add_argument(
        "--without_cot_instruction",
        action="store_true",
        help="If specified, do not include CoT instruction in the prompts.",
    )
    # -- Tasks list
    parser.add_argument(
        "--tasks_list_json_path_AD",
        type=str,
        help="Path to the tasks list JSON file for angle distance task",
    )
    parser.add_argument(
        "--tasks_list_json_path_detect",
        type=str,
        help="Path to the tasks list JSON file for detection task",
    )
    parser.add_argument(
        "--tasks_list_json_path_TL",
        type=str,
        help="Path to the tasks list JSON file for tumor lesion size task",
    )
    # -- Multi-processing settings
    parser.add_argument(
        "--num_workers_concat_datasets",
        type=int,
        default=4,
        help="Number of workers for concatenating datasets, should be <= number of tasks",
    )
    parser.add_argument(
        "--num_workers_format_dataset",
        type=int,
        default=64,
        help="Number of workers for formatting datasets",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    # -- Sample limits (per-task)
    parser.add_argument(
        "--train_sample_limit_per_task",
        type=int,
        default=-1,
        help="Limit the number of training samples per task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_per_task",
        type=int,
        default=100,
        help="Limit the number of validation samples per task",
    )
    # Task-specific sample limits
    parser.add_argument(
        "--train_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of training samples for angle distance task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of validation samples for angle distance task, -1 means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of testing samples for angle distance task, -1 means no limit",
    )
    parser.add_argument(
        "--train_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of training samples for detection task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of validation samples for detection task, -1 means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of testing samples for detection task, -1 means no limit",
    )
    parser.add_argument(
        "--train_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of training samples for tumor lesion task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of validation samples for tumor lesion task, -1 means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of testing samples for tumor lesion task, -1 means no limit",
    )
    # -- Sample limits (global)
    parser.add_argument(
        "--train_sample_limit",
        type=int,
        default=-1,
        help="Limit the number of training samples, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit",
        type=int,
        default=-1,
        help="Limit the number of total validation samples, -1 (default) means no limit",
    )
    parser.add_argument(
        "--test_sample_limit",
        type=int,
        default=-1,
        help="Global test sample limit applied after combining all tasks, -1 means no limit",
    )
    # -- Dataset limit per subset (i.e., HF dataset config)
    parser.add_argument(
        "--train_sample_limit_per_subset",
        type=int,
        default=-1,
        help="Limit training samples per HF dataset config (subset) before merging, -1 means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_per_subset",
        type=int,
        default=-1,
        help="Limit test samples per HF dataset config (subset) before merging, -1 means no limit",
    )
    # -- Checkpointing / sharding
    parser.add_argument(
        "--shard_size",
        type=int,
        default=50000,
        help=(
            "Number of training samples per shard. Each shard is formatted, saved "
            "to disk, and freed from RAM before the next shard starts. "
            "Smaller shards use less peak RAM; larger shards are faster. "
            "Default: 50 000."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_parquet_dataset_checkpointed(**kwargs):
    model_family_name = kwargs["model_family_name"]
    shard_size = kwargs.get("shard_size", 50000)
    num_workers_format = kwargs.get("num_workers_format_dataset", 64)
    num_workers_concat = kwargs.get("num_workers_concat_datasets", 4)
    use_cot = not kwargs.get("without_cot_instruction", False)

    print(
        "\n[WARNING] The prepared dataset directory name must uniquely include the model identifier and sample limits.\n"
        "Dataset preparation depends on model-specific image processing (e.g., resize scale and pixel dimensions).\n"
        "Reusing a dataset prepared with different settings or a different model may lead to incorrect results."
    )

    # Parse limits (reuse the same helper as build_parquet_ds_with_testset.py).
    (
        train_limit_AD,
        val_limit_AD,
        train_limit_detect,
        val_limit_detect,
        train_limit_TL,
        val_limit_TL,
        train_limit_total,
    ) = parse_sample_limits(**kwargs)

    # Compute output directory (same naming convention as build_parquet_ds_with_testset.py).
    cot_tag = "_wo-CoT-Instruct" if kwargs.get("without_cot_instruction") else ""
    if kwargs.get("new_shape_hw") is not None:
        h, w = kwargs["new_shape_hw"]
        ds_dir = (
            f"ds__AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}"
            f"_all{train_limit_total}{cot_tag}__resized-hw-{h}x{w}"
        )
    else:
        ds_dir = (
            f"ds__AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}"
            f"_all{train_limit_total}{cot_tag}__original"
        )
    parquet_ds_dir = os.path.join(
        kwargs["data_dir"], "verl_datasets", model_family_name, ds_dir
    )
    shards_dir = os.path.join(parquet_ds_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    checkpoint_path = os.path.join(parquet_ds_dir, "checkpoint.json")
    ckpt = _load_checkpoint(checkpoint_path)

    print(f"\nPrepared Verl parquet dataset directory: {parquet_ds_dir}")
    print(f"Shard size: {shard_size}")
    print(f"Checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # Determine which tasks are active and their configuration.
    # Each entry: (tasks_json, per_task_train_limit, per_task_val_limit, task_tag)
    # ------------------------------------------------------------------
    active_tasks = []
    if kwargs.get("tasks_list_json_path_detect"):
        active_tasks.append((
            kwargs["tasks_list_json_path_detect"],
            train_limit_detect,
            val_limit_detect,
            "BoxSize",
        ))
    if kwargs.get("tasks_list_json_path_AD"):
        active_tasks.append((
            kwargs["tasks_list_json_path_AD"],
            train_limit_AD,
            val_limit_AD,
            "BiometricsFromLandmarks",
        ))
    if kwargs.get("tasks_list_json_path_TL"):
        active_tasks.append((
            kwargs["tasks_list_json_path_TL"],
            train_limit_TL,
            val_limit_TL,
            "TumorLesionSize",
        ))

    if not active_tasks:
        raise ValueError("No task JSON paths provided. At least one of "
                         "--tasks_list_json_path_detect / _AD / _TL must be set.")

    mapping_func_args = {
        "use_cot": use_cot,
        "model_name": model_family_name,
        "model_hf": kwargs["model_hf"],
        "new_shape_hw": kwargs.get("new_shape_hw"),
    }

    # ------------------------------------------------------------------
    # STEP 1 — Load ALL raw task datasets (no images, lightweight).
    # ------------------------------------------------------------------
    # We load raw train with limit=-1 (all samples) and apply the per-task
    # limit manually on the raw (pre-image) data. This keeps the combined
    # raw train in memory cheaply while deferring heavy image processing to
    # the per-shard format step.

    print("\n" + "=" * 60)
    print("STEP 1: Loading raw datasets (no images)")
    print("=" * 60)

    raw_train_parts = []   # list of raw_Dataset slices per task
    raw_val_parts = []     # list of raw_Dataset per task

    for tasks_json, per_task_train_lim, per_task_val_lim, task_tag in active_tasks:
        print(f"\n  Loading task '{task_tag}' from {tasks_json} ...")
        raw_ds = load_split_limit_dataset(
            tasks_list_json_path=tasks_json,
            limit_train_sample=-1,          # load all; we apply limit below
            limit_val_sample=per_task_val_lim,
            num_workers_concat_datasets=num_workers_concat,
            tag_ds=task_tag,
            download_mode=kwargs.get("ds_download_mode", "reuse_dataset_if_exists"),
        )

        # Apply per-task train limit on raw (cheap) data.
        raw_train = raw_ds["train"]
        if per_task_train_lim > 0 and per_task_train_lim < len(raw_train):
            print(
                f"  Applying per-task train limit: {per_task_train_lim} "
                f"(from {len(raw_train)})"
            )
            raw_train = raw_train.shuffle(seed=SEED).select(range(per_task_train_lim))

        # Tag each sample with its task so the dispatch format function knows
        # which format function to call after sharding the combined dataset.
        raw_train = raw_train.add_column("_task_tag", [task_tag] * len(raw_train))
        raw_val = raw_ds["validation"].add_column(
            "_task_tag", [task_tag] * len(raw_ds["validation"])
        )

        raw_train_parts.append(raw_train)
        raw_val_parts.append(raw_val)
        del raw_ds
        gc.collect()
        print(f"  '{task_tag}' raw train: {len(raw_train)}, raw val: {len(raw_val)}")

    # ------------------------------------------------------------------
    # STEP 2 — Combine raw trains, shuffle, apply global limit.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Combining and shuffling raw train data")
    print("=" * 60)

    combined_raw_train = concatenate_datasets(raw_train_parts)
    del raw_train_parts
    gc.collect()
    print(f"  Combined raw train size: {len(combined_raw_train)}")

    # Mirror the shuffle + global-limit logic from build_parquet_ds_with_testset.py.
    if train_limit_total > 0:
        total_raw = len(combined_raw_train)
        if train_limit_total > total_raw:
            print(
                f"  Global train limit ({train_limit_total}) > dataset size ({total_raw}). "
                f"Sampling with replacement."
            )
            np.random.seed(SEED)
            indices = np.random.choice(total_raw, size=train_limit_total, replace=True)
            combined_raw_train = combined_raw_train.select(indices)
        else:
            combined_raw_train = (
                combined_raw_train.shuffle(seed=SEED).select(range(train_limit_total))
            )
    else:
        combined_raw_train = combined_raw_train.shuffle(seed=SEED)

    total_train = len(combined_raw_train)
    n_shards = math.ceil(total_train / shard_size)
    print(f"  Final combined train size: {total_train}, shards: {n_shards}")

    # Reconcile checkpoint: detect orphaned shard files (exist on disk but not
    # in checkpoint, e.g. from a partial write before the crash).
    for i in range(n_shards):
        shard_path = os.path.join(shards_dir, f"train_shard_{i:04d}.parquet")
        if os.path.exists(shard_path) and i not in ckpt["completed_train_shards"]:
            print(f"  [Resume] Detected existing shard {i:04d}, adding to checkpoint.")
            ckpt["completed_train_shards"].append(i)
    ckpt["n_shards"] = n_shards
    ckpt["total_train"] = total_train
    _save_checkpoint(checkpoint_path, ckpt)

    # ------------------------------------------------------------------
    # STEP 3 — Per-shard format → clean → save → free loop.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Per-shard format + save")
    print("=" * 60)

    completed = set(ckpt["completed_train_shards"])
    for i in range(n_shards):
        shard_path = os.path.join(shards_dir, f"train_shard_{i:04d}.parquet")

        if i in completed:
            print(f"  [Skip] Shard {i:04d}/{n_shards - 1} already done.")
            continue

        start = i * shard_size
        end = min(start + shard_size, total_train)
        print(
            f"\n  [Shard {i:04d}/{n_shards - 1}] "
            f"samples {start}–{end - 1} ({end - start} samples) ..."
        )

        raw_shard = combined_raw_train.select(range(start, end))

        # Format: apply per-sample dispatch (images are loaded here).
        formatted = format_dataset(
            dataset=raw_shard,
            mapping_func=_dispatch_format,
            mapping_func_args=mapping_func_args,
            num_workers_format_dataset=num_workers_format,
            writer_batch_size=50,
        )
        del raw_shard
        gc.collect()

        # Clean: keep only Verl-required fields (also removes _task_tag).
        cleaned = clean_dataset(formatted, _VERL_KEYS)
        del formatted
        gc.collect()

        # Write shard to parquet immediately, then free RAM.
        cleaned.to_parquet(shard_path)
        del cleaned
        gc.collect()

        ckpt["completed_train_shards"].append(i)
        _save_checkpoint(checkpoint_path, ckpt)
        print(f"  [Shard {i:04d}] Saved → {shard_path}")

    del combined_raw_train
    gc.collect()

    # ------------------------------------------------------------------
    # STEP 4 — Validation split (tiny, no sharding needed).
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Validation split")
    print("=" * 60)

    val_parquet_path = os.path.join(parquet_ds_dir, "validation_verl.parquet")
    if ckpt["val_done"]:
        print("  [Skip] Validation split already done.")
    else:
        combined_raw_val = concatenate_datasets(raw_val_parts)
        del raw_val_parts
        gc.collect()

        val_formatted = format_dataset(
            dataset=combined_raw_val,
            mapping_func=_dispatch_format,
            mapping_func_args=mapping_func_args,
            num_workers_format_dataset=num_workers_format,
            writer_batch_size=50,
        )
        del combined_raw_val
        gc.collect()

        val_cleaned = clean_dataset(val_formatted, _VERL_KEYS)
        del val_formatted
        gc.collect()

        # Apply global val limit (mirror build_parquet_ds_with_testset.py logic).
        val_limit = kwargs.get("val_sample_limit", -1)
        if val_limit > 0:
            val_size = len(val_cleaned)
            if val_limit > val_size:
                np.random.seed(SEED)
                indices = np.random.choice(val_size, size=val_limit, replace=True)
                val_cleaned = val_cleaned.select(indices)
            else:
                val_cleaned = val_cleaned.shuffle(seed=SEED).select(range(val_limit))
        else:
            val_cleaned = val_cleaned.shuffle(seed=SEED)

        val_cleaned.to_parquet(val_parquet_path)
        del val_cleaned
        gc.collect()

        ckpt["val_done"] = True
        _save_checkpoint(checkpoint_path, ckpt)
        print(f"  Saved → {val_parquet_path}")

    # ------------------------------------------------------------------
    # STEP 4b — Test split (optional, only when test limits are provided).
    # Test data comes from separate MedVision _Test HF configs (not _Train).
    # Test sets are small (~1000 samples); no sharding needed.
    # ------------------------------------------------------------------
    _has_test = any(
        (kwargs.get(k) or -1) > 0
        for k in [
            "test_sample_limit_task_AD",
            "test_sample_limit_task_Detection",
            "test_sample_limit_task_TL",
            "test_sample_limit",
            "test_sample_limit_per_subset",
        ]
    )
    test_parquet_path = os.path.join(parquet_ds_dir, "test_verl.parquet")
    if _has_test:
        print("\n" + "=" * 60)
        print("STEP 4b: Test split")
        print("=" * 60)

        if ckpt.get("test_done"):
            print("  [Skip] Test split already done.")
        else:
            # Resolve per-task-type test limits.
            # A positive task-specific limit overrides; None means no per-task-type cap
            # (the per-subset limit in _load_raw_test_dataset still applies).
            def _test_limit_for(specific_key):
                specific = kwargs.get(specific_key) or -1
                return specific if specific > 0 else None

            test_task_configs = []
            if kwargs.get("tasks_list_json_path_detect"):
                test_task_configs.append((
                    kwargs["tasks_list_json_path_detect"],
                    _test_limit_for("test_sample_limit_task_Detection"),
                    "BoxSize",
                ))
            if kwargs.get("tasks_list_json_path_AD"):
                test_task_configs.append((
                    kwargs["tasks_list_json_path_AD"],
                    _test_limit_for("test_sample_limit_task_AD"),
                    "BiometricsFromLandmarks",
                ))
            if kwargs.get("tasks_list_json_path_TL"):
                test_task_configs.append((
                    kwargs["tasks_list_json_path_TL"],
                    _test_limit_for("test_sample_limit_task_TL"),
                    "TumorLesionSize",
                ))

            _per_subset_test = kwargs.get("test_sample_limit_per_subset")
            if _per_subset_test is not None and _per_subset_test < 0:
                _per_subset_test = None

            raw_test_parts = []
            for tasks_json, per_task_test_lim, task_tag in test_task_configs:
                print(f"\n  Loading test data for '{task_tag}' ...")
                raw_test = _load_raw_test_dataset(
                    tasks_json=tasks_json,
                    tag_ds=task_tag,
                    test_limit=per_task_test_lim,
                    num_workers=num_workers_concat,
                    download_mode=kwargs.get("ds_download_mode", "reuse_dataset_if_exists"),
                    test_limit_per_subset=_per_subset_test,
                )
                raw_test = raw_test.add_column("_task_tag", [task_tag] * len(raw_test))
                raw_test_parts.append(raw_test)
                print(f"  '{task_tag}' raw test: {len(raw_test)}")

            combined_raw_test = concatenate_datasets(raw_test_parts)
            del raw_test_parts
            gc.collect()

            test_formatted = format_dataset(
                dataset=combined_raw_test,
                mapping_func=_dispatch_format,
                mapping_func_args=mapping_func_args,
                num_workers_format_dataset=num_workers_format,
                writer_batch_size=50,
            )
            del combined_raw_test
            gc.collect()

            test_cleaned = clean_dataset(test_formatted, _VERL_KEYS)
            del test_formatted
            gc.collect()

            # Apply global test limit.
            global_test_limit = kwargs.get("test_sample_limit") or -1
            if global_test_limit > 0:
                test_size = len(test_cleaned)
                if global_test_limit > test_size:
                    np.random.seed(SEED)
                    indices = np.random.choice(test_size, size=global_test_limit, replace=True)
                    test_cleaned = test_cleaned.select(indices)
                else:
                    test_cleaned = test_cleaned.shuffle(seed=SEED).select(range(global_test_limit))
            else:
                test_cleaned = test_cleaned.shuffle(seed=SEED)

            test_cleaned.to_parquet(test_parquet_path)
            del test_cleaned
            gc.collect()

            ckpt["test_done"] = True
            _save_checkpoint(checkpoint_path, ckpt)
            print(f"  Saved → {test_parquet_path}")

    # ------------------------------------------------------------------
    # STEP 5 — Stream-merge all shard parquets into train_verl.parquet.
    # Uses PyArrow ParquetWriter: reads one shard at a time (~5 GB RAM).
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Merging shards → train_verl.parquet")
    print("=" * 60)

    train_parquet_path = os.path.join(parquet_ds_dir, "train_verl.parquet")
    if ckpt["merged"]:
        print("  [Skip] Already merged.")
    else:
        shard_paths = sorted(
            os.path.join(shards_dir, f"train_shard_{i:04d}.parquet")
            for i in range(n_shards)
        )
        # Verify all shards exist before merging.
        missing = [p for p in shard_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Cannot merge: {len(missing)} shard(s) missing:\n"
                + "\n".join(missing)
            )

        writer = None
        for idx, shard_path in enumerate(shard_paths):
            print(f"  Merging shard {idx:04d}/{n_shards - 1} ...", end=" ", flush=True)
            table = pq.read_table(shard_path)
            if writer is None:
                writer = pq.ParquetWriter(train_parquet_path, table.schema)
            writer.write_table(table)
            del table
            gc.collect()
            print("done")

        if writer is not None:
            writer.close()

        ckpt["merged"] = True
        _save_checkpoint(checkpoint_path, ckpt)
        print(f"\n  Saved → {train_parquet_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print(f"  train_verl.parquet      : {train_parquet_path}")
    print(f"  validation_verl.parquet : {val_parquet_path}")
    if _has_test:
        print(f"  test_verl.parquet       : {test_parquet_path}")
    print(f"  checkpoint.json         : {checkpoint_path}")
    print("=" * 60)


def main():
    args = parse_arguments()
    build_parquet_dataset_checkpointed(**vars(args))


if __name__ == "__main__":
    main()
