import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil
from datasets import concatenate_datasets

from medvision_bm.sft.sft_utils import (
    _load_single_dataset,
    get_cgroup_limited_cpus,
    group_train_test_split,
)
from medvision_bm.utils.configs import SEED


def parse_sample_limits_tr_val_ts(**kwargs):
    """
    Determine sample limits for each task with fallbacks.

    Logic:
        - If task-specific limit > 0: use it
        - Else: use per-task limit
        - If task JSON path is None: set limit to 0 (task not used)

    Returns:
        A tuple of sample limits:
        (train_limit_AD, val_limit_AD, test_limit_AD,
         train_limit_detect, val_limit_detect, test_limit_detect,
         train_limit_TL, val_limit_TL, test_limit_TL)
    """

    # Determine sample limits for each task
    # Angle/distance task
    if kwargs.get("train_sample_limit_task_AD") > 0:
        train_limit_AD = kwargs.get("train_sample_limit_task_AD")
    else:
        train_limit_AD = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_AD") > 0:
        val_limit_AD = kwargs.get("val_sample_limit_task_AD")
    else:
        val_limit_AD = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("test_sample_limit_task_AD") > 0:
        test_limit_AD = kwargs.get("test_sample_limit_task_AD")
    else:
        test_limit_AD = kwargs.get("test_sample_limit_per_task")
    if kwargs.get("tasks_list_json_path_AD") is None:
        train_limit_AD = 0
        val_limit_AD = 0

    # Detection task
    if kwargs.get("train_sample_limit_task_Detection") > 0:
        train_limit_detect = kwargs.get("train_sample_limit_task_Detection")
    else:
        train_limit_detect = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_Detection") > 0:
        val_limit_detect = kwargs.get("val_sample_limit_task_Detection")
    else:
        val_limit_detect = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("test_sample_limit_task_Detection") > 0:
        test_limit_detect = kwargs.get("test_sample_limit_task_Detection")
    else:
        test_limit_detect = kwargs.get("test_sample_limit_per_task")
    if kwargs.get("tasks_list_json_path_detect") is None:
        train_limit_detect = 0
        val_limit_detect = 0

    # Tumor lesion size task
    if kwargs.get("train_sample_limit_task_TL") > 0:
        train_limit_TL = kwargs.get("train_sample_limit_task_TL")
    else:
        train_limit_TL = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_TL") > 0:
        val_limit_TL = kwargs.get("val_sample_limit_task_TL")
    else:
        val_limit_TL = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("test_sample_limit_task_TL") > 0:
        test_limit_TL = kwargs.get("test_sample_limit_task_TL")
    else:
        test_limit_TL = kwargs.get("test_sample_limit_per_task")
    if kwargs.get("tasks_list_json_path_TL") is None:
        train_limit_TL = 0
        val_limit_TL = 0

    return (
        train_limit_AD,
        val_limit_AD,
        test_limit_AD,
        train_limit_detect,
        val_limit_detect,
        test_limit_detect,
        train_limit_TL,
        val_limit_TL,
        test_limit_TL,
    )


def load_split_limit_dataset_tr_val_ts(
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    limit_test_sample,
    limit_train_sample_per_subset=None,
    limit_test_sample_per_subset=None,
    num_workers_concat_datasets=4,
    tag_ds=None,
    download_mode="reuse_dataset_if_exists",
):
    """
    Load, split, and limit a MedVision task dataset into train / validation / test splits.

    The function loads the HuggingFace MedVision dataset for all tasks listed in
    ``tasks_list_json_path``, constructs train/val/test splits, and applies sample
    limits at two granularities:

    **Per-subset limit** (applied during loading, before any merging):
        - ``limit_train_sample_per_subset``: max samples loaded from each individual
          dataset config in the training pool.  Pass ``None`` or ``-1`` for no limit.
        - ``limit_test_sample_per_subset``: same for the test pool.

    **Per-task (total) limit** (applied after all subsets are merged):
        - ``limit_train_sample``: max training samples **after** the val split is
          carved out.  Use ``> 0`` to cap, ``< 0`` for no limit.  Must not be ``0``.
        - ``limit_val_sample``: exact number of validation samples carved out of the
          training pool via a group-aware split (3D volumes are kept together to
          prevent data leakage).  Must be ``> 0``.
        - ``limit_test_sample``: max test samples after loading.  Use ``> 0`` to cap,
          ``< 0`` or ``None`` for no limit.

    Split construction order:
        1. Load all ``{task}_Train`` configs; apply ``limit_train_sample_per_subset``
           per config.
        2. Concatenate → group-split on ``image_file`` to produce a val set of
           exactly ``limit_val_sample`` samples.
        3. Apply ``limit_train_sample`` to the remaining training pool.
        4. Load all ``{task}_Test`` configs; apply ``limit_test_sample_per_subset``
           per config.
        5. Concatenate → apply ``limit_test_sample`` to the merged test pool.

    Args:
        tasks_list_json_path (str): Path to a JSON file mapping task config names to
            their metadata.
        limit_train_sample (int): Max training samples after val split (``< 0`` = no
            limit, ``> 0`` = cap).  Must not be ``0``.
        limit_val_sample (int): Number of validation samples to carve out (must be
            ``> 0``).
        limit_test_sample (int): Max test samples (``< 0`` or ``None`` = no limit,
            ``> 0`` = cap).
        limit_train_sample_per_subset (int | None): Per-config cap during training
            loading.  ``None`` / ``-1`` = no limit.
        limit_test_sample_per_subset (int | None): Per-config cap during test loading.
            ``None`` / ``-1`` = no limit.
        num_workers_concat_datasets (int): Worker processes for parallel dataset
            loading.  Automatically reduced when new (not-yet-cached) datasets are
            detected.
        tag_ds (str): Suffix in task config names used to extract the dataset name
            (e.g., ``"BiometricsFromLandmarks"`` → task ``"Brain_BiometricsFromLandmarks"``
            yields dataset name ``"Brain"``).  Required.
        download_mode (str): HuggingFace ``download_mode`` passed to
            ``load_dataset``.  Default: ``"reuse_dataset_if_exists"``.

    Returns:
        DatasetDict with keys ``"train"``, ``"validation"``, and ``"test"``.
    """
    # NOTE:
    # - limit_val_sample must be greater than 0 to ensure validation set is not empty.
    # - limit_train_sample can be <0 (no limit) or >0 (limited training set).
    assert limit_val_sample > 0, "\n [Error] limit_val_sample must be greater than 0."
    assert (
        limit_train_sample != 0
    ), "\n [Error] limit_train_sample cannot be 0. Use <0 for no limit or >0 for limited training set."

    # Early assertions
    assert (
        tag_ds is not None
    ), "\n [Error] tag_ds (i.e., the string in tasks names: <dataset_name>_<tag_ds>) must be provided."

    print(f"\n[Info] Starting dataset preparation from {tasks_list_json_path}")

    # Load tasks list from JSON file
    with open(tasks_list_json_path, "r") as f:
        tasks_dict = json.load(f)
    tasks = list(tasks_dict.keys())

    print(f"[Info] Found {len(tasks)} tasks to process")

    # Reduce parallelism to avoid memory issues - use fewer workers
    available_cpus = get_cgroup_limited_cpus()
    concat_workers = min(num_workers_concat_datasets, available_cpus, len(tasks))

    # NOTE: Force single-threaded loading for new datasets to avoid conflicts, otherwise errors may occur.
    data_dir = os.environ.get("MedVision_DATA_DIR")
    assert (
        data_dir is not None
    ), "\n [Error] MedVision_DATA_DIR environment variable must be set."
    # Read the .downloaded_datasets.json file in data_dir
    with open(os.path.join(data_dir, ".downloaded_datasets.json"), "r") as f:
        downloaded_datasets = list(json.load(f).keys())
    for task in tasks:
        # NOTE: This is specific to the MedVision dataset and configs: extract dataset name (part before "_<tag_ds>")
        dataset_name = task.split(f"_{tag_ds}")[0]
        if f"dataset_{dataset_name}" not in downloaded_datasets:
            concat_workers = 1
            print(
                f"[Info] Dataset {dataset_name} is newly downloaded. Using single-threaded loading to avoid conflicts."
            )
            break

    print(
        f"[Info] Using {concat_workers} workers for dataset loading (available CPUs: {available_cpus})"
    )

    # ============================
    # [1] Load training set from MedVision dataset, split into training and validation sets
    # ============================
    datasets_list = []
    failed_tasks = []
    with ProcessPoolExecutor(max_workers=concat_workers) as executor:
        # Load training splits for all tasks in parallel
        # ------
        # NOTE: This is specific to the MedVision dataset and configs
        # For MedVision dataset:
        # - Config name for training set is in the format of "{task}_Train", while the test set is "{task}_Test"
        # - Dataset name can be extracted from task name (e.g., part before f"_{tag_ds}"): task.split(f"_{tag_ds}")[0]
        # ------
        future_to_task = {
            executor.submit(
                _load_single_dataset,
                "YongchengYAO/MedVision",
                task.split(f"_{tag_ds}")[0],
                task + "_Train",
                "train",
                limit_train_sample_per_subset,
                download_mode=download_mode,
            ): task
            for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                ds = future.result(timeout=120)  # 2 minute timeout per task
                datasets_list.append(ds)
                print(f"✓ Completed {task} ({len(datasets_list)}/{len(tasks)})")

                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    print(f"⚠️  High memory usage: {memory_percent}%")

            except Exception as exc:
                error_msg = f"Task {task} generated an exception: {exc}"
                print(error_msg)
                failed_tasks.append((task, str(exc)))
                # Continue with other tasks instead of failing completely

    # Report results
    if failed_tasks:
        print(f"❌ Failed to load {len(failed_tasks)} tasks:")
        for task, error in failed_tasks:
            print(f"  - {task}: {error}")

        raise RuntimeError(
            "❌ ERROR: Some tasks failed to load. Check the logs above for details."
        )

    # Combine all datasets
    print("\n[Info] Combining datasets...")
    combined_dataset = concatenate_datasets(datasets_list)
    print(f"[Info] Combined dataset has {len(combined_dataset)} total samples")

    # Clear intermediate datasets to free memory
    del datasets_list
    gc.collect()

    # Split the dataset into training and validation sets
    print(
        f"\n[Info] Splitting dataset into training and validation (target val size: {limit_val_sample}) sets"
    )

    # Split with group consideration (image_file) to prevent leakage
    print(
        f"\n[Info] Splitting dataset into training and validation (target val size: {limit_val_sample}) keeping 3D volumes grouped."
    )

    # NOTE: "image_file" is a column in the MedVision dataset representing the path to the 3D NIfTI image.
    # TODO: Make group_column configurable if needed.
    dataset = group_train_test_split(
        combined_dataset,
        group_column="image_file",
        test_size=limit_val_sample,
        seed=SEED,
        stratify_column="dataset_name",
    )

    # Limit the number of training and validation samples if specified
    if limit_train_sample > 0 and limit_train_sample < len(dataset["train"]):
        print(
            f"\n[Info][Warning] Limiting training samples to {limit_train_sample} (original: {len(dataset['train'])})"
        )
        dataset["train"] = (
            dataset["train"].shuffle(seed=SEED).select(range(limit_train_sample))
        )
    # ============================

    # ============================
    # [2] Load testing set from MedVision dataset
    # ============================
    datasets_list_test = []
    failed_tasks_test = []
    with ProcessPoolExecutor(max_workers=concat_workers) as executor:
        # Load testing splits for all tasks in parallel
        # ------
        # NOTE: This is specific to the MedVision dataset and configs
        # For MedVision dataset:
        # - Config name for training set is in the format of "{task}_Train", while the test set is "{task}_Test"
        # - Dataset name can be extracted from task name (e.g., part before f"_{tag_ds}"): task.split(f"_{tag_ds}")[0]
        # ------
        future_to_task = {
            executor.submit(
                _load_single_dataset,
                "YongchengYAO/MedVision",
                task.split(f"_{tag_ds}")[0],
                task + "_Test",
                "test",
                limit_test_sample_per_subset,
                download_mode=download_mode,
            ): task
            for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                ds_test = future.result(timeout=120)  # 2 minute timeout per task
                datasets_list_test.append(ds_test)
                print(f"✓ Completed {task} ({len(datasets_list_test)}/{len(tasks)})")

                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    print(f"⚠️  High memory usage: {memory_percent}%")

            except Exception as exc:
                error_msg = f"Task {task} generated an exception: {exc}"
                print(error_msg)
                failed_tasks_test.append((task, str(exc)))
                # Continue with other tasks instead of failing completely

    # Report results
    if failed_tasks_test:
        print(f"❌ Failed to load {len(failed_tasks_test)} tasks:")
        for task, error in failed_tasks_test:
            print(f"  - {task}: {error}")

        raise RuntimeError(
            "❌ ERROR: Some tasks failed to load. Check the logs above for details."
        )

    # Combine all datasets
    print("\n[Info] Combining datasets...")
    combined_dataset_test = concatenate_datasets(datasets_list_test)
    print(f"[Info] Combined dataset has {len(combined_dataset_test)} total samples")

    # Clear intermediate datasets to free memory
    del datasets_list_test
    gc.collect()

    # Assign the combined testing set
    dataset["test"] = combined_dataset_test

    # Limit the number of testing samples if specified
    if limit_test_sample > 0 and limit_test_sample < len(dataset["test"]):
        print(
            f"\n[Info][Warning] Limiting testing samples to {limit_test_sample} (original: {len(dataset['test'])})"
        )
        dataset["test"] = (
            dataset["test"].shuffle(seed=SEED).select(range(limit_test_sample))
        )
    # ============================

    return dataset


def build_parquet_dataset(
    *,
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    limit_test_sample,
    limit_train_sample_per_subset,
    limit_test_sample_per_subset,
    num_workers_concat_datasets=4,
    tag_ds=None,
    download_mode="reuse_dataset_if_exists",
):
    # Load and split dataset
    dataset = load_split_limit_dataset_tr_val_ts(
        tasks_list_json_path=tasks_list_json_path,
        limit_train_sample=limit_train_sample,
        limit_val_sample=limit_val_sample,
        limit_test_sample=limit_test_sample,
        limit_train_sample_per_subset=limit_train_sample_per_subset,
        limit_test_sample_per_subset=limit_test_sample_per_subset,
        num_workers_concat_datasets=num_workers_concat_datasets,
        tag_ds=tag_ds,
        download_mode=download_mode,
    )

    return dataset
