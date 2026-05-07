import argparse
import os

from datasets import DatasetDict, concatenate_datasets

from medvision_bm.dataset.ds_utils import (
    build_parquet_dataset,
    parse_sample_limits_tr_val_ts,
)
from medvision_bm.utils.configs import SEED


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build parquet dataset for MedVision Dataset."
    )
    # -- Data folder
    parser.add_argument(
        "--parquet_ds_dir",
        type=str,
        help="Directory of the final parquet dataset.",
    )
    # -- Dataset download mode
    parser.add_argument(
        "--ds_download_mode",
        type=str,
        default="reuse_dataset_if_exists",
        help="Dataset download mode: 'reuse_dataset_if_exists' (default), 'reuse_cache_if_exists', 'force_redownload'",
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
    # -- Sample limits
    # Limit the number of samples per subset (e.g., per dataset) for training
    parser.add_argument(
        "--train_sample_limit_per_subset",
        type=int,
        default=-1,
        help="Limit the number of training samples per subset, -1 (default) means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_per_subset",
        type=int,
        default=-1,
        help="Limit the number of test samples per subset, -1 (default) means no limit",
    )
    # Limit the number of samples per task (e.g., A/D, T/L, and detection tasks, each of which contains multiple datasets)
    parser.add_argument(
        "--train_sample_limit_per_task",
        type=int,
        default=-1,
        help="Limit the number of training samples per task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_per_task",
        type=int,
        default=100, 
        help="Limit the number of validation samples per task",
    )
    parser.add_argument(
        "--test_sample_limit_per_task",
        type=int,
        default=-1,
        help="Limit the number of test samples per task, -1 (default) means no limit",
    )
    # Task-specific sample limit
    parser.add_argument(
        "--train_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of training samples for angle distance task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of validation samples for angle distance task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of test samples for angle distance task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--train_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of training samples for detection task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of validation samples for detection task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of test samples for detection task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--train_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of training samples for tumor lesion task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of validation samples for tumor lesion task, -1 (default) means no limit",
    )
    parser.add_argument(
        "--test_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of test samples for tumor lesion task, -1 (default) means no limit",
    )
    # This is to limit the number of samples in total
    parser.add_argument(
        "--train_sample_limit",
        type=int,
        default=-1,
        help="Limit the number of total training samples, -1 (default) means no limit",
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
        help="Limit the number of total test samples, -1 (default) means no limit",
    )

    args = parser.parse_args()

    return args


def main(**kwargs):
    # Output parquet dataset directory
    parquet_ds_dir = kwargs.get("parquet_ds_dir")

    # =========================================================================
    # Sample limit hierarchy (applied in order, from fine-grained to coarse):
    #
    # [Level 1] Per-subset limit  (--train/test_sample_limit_per_subset)
    #   └─ Caps samples loaded from each individual dataset/config (e.g., one
    #      CT study series) before any merging.  Use -1 for no limit.
    #
    # [Level 2] Per-task limit    (--train/val/test_sample_limit_per_task)
    #   └─ Caps the merged pool for each task group (AD / TL / Detection)
    #      after all subsets are concatenated.  Use -1 for no limit.
    #
    # [Level 3] Task-specific override
    #           (--train/val/test_sample_limit_task_{AD|Detection|TL})
    #   └─ Overrides the per-task limit for one particular task group when > 0.
    #      Takes priority over Level 2 if set to a positive value.
    #
    # [Level 4] Total limit       (--train/val/test_sample_limit)
    #   └─ Caps the grand total after all task groups are concatenated.
    #      Use -1 for no limit.
    #
    # Validation split is always carved out of the training pool (after
    # Level-1/2/3 limits are applied) before the Level-4 total cap.
    # =========================================================================

    # Resolve per-task limits (Level 2/3) for each task group
    (
        train_limit_AD,
        val_limit_AD,
        test_limit_AD,
        train_limit_detect,
        val_limit_detect,
        test_limit_detect,
        train_limit_TL,
        val_limit_TL,
        test_limit_TL,
    ) = parse_sample_limits_tr_val_ts(**kwargs)

    # Per-subset limits (Level 1) — shared across all task groups
    train_limit_per_subset = kwargs.get("train_sample_limit_per_subset")
    test_limit_per_subset = kwargs.get("test_sample_limit_per_subset")

    # Prepare datasets
    train_ds_list = []
    val_ds_list = []
    test_ds_list = []
    if kwargs.get("tasks_list_json_path_AD") is not None:
        dataset_AD = build_parquet_dataset(
            tasks_list_json_path=kwargs.get("tasks_list_json_path_AD"),
            limit_train_sample=train_limit_AD,
            limit_val_sample=val_limit_AD,
            limit_test_sample=test_limit_AD,
            limit_train_sample_per_subset=train_limit_per_subset,
            limit_test_sample_per_subset=test_limit_per_subset,
            num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
            # MedVision dataset specific, used to extract dataset name from AD task configs
            tag_ds="BiometricsFromLandmarks",
            download_mode=kwargs.get("ds_download_mode"),
        )
        train_ds_list.append(dataset_AD["train"])
        val_ds_list.append(dataset_AD["validation"])
        test_ds_list.append(dataset_AD["test"])
    if kwargs.get("tasks_list_json_path_TL") is not None:
        dataset_TL = build_parquet_dataset(
            tasks_list_json_path=kwargs.get("tasks_list_json_path_TL"),
            limit_train_sample=train_limit_TL,
            limit_val_sample=val_limit_TL,
            limit_test_sample=test_limit_TL,
            limit_train_sample_per_subset=train_limit_per_subset,
            limit_test_sample_per_subset=test_limit_per_subset,
            num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
            # MedVision dataset specific, used to extract dataset name from AD task configs
            tag_ds="TumorLesionSize",
            download_mode=kwargs.get("ds_download_mode"),
        )
        train_ds_list.append(dataset_TL["train"])
        val_ds_list.append(dataset_TL["validation"])
        test_ds_list.append(dataset_TL["test"])
    if kwargs.get("tasks_list_json_path_detect") is not None:
        dataset_detect = build_parquet_dataset(
            tasks_list_json_path=kwargs.get("tasks_list_json_path_detect"),
            limit_train_sample=train_limit_detect,
            limit_val_sample=val_limit_detect,
            limit_test_sample=test_limit_detect,
            limit_train_sample_per_subset=train_limit_per_subset,
            limit_test_sample_per_subset=test_limit_per_subset,
            num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
            # MedVision dataset specific, used to extract dataset name from AD task configs
            tag_ds="BoxSize",
            download_mode=kwargs.get("ds_download_mode"),
        )
        train_ds_list.append(dataset_detect["train"])
        val_ds_list.append(dataset_detect["validation"])
        test_ds_list.append(dataset_detect["test"])

    # Combine all tasks' datasets
    dataset = DatasetDict()
    dataset["train"] = concatenate_datasets(train_ds_list)
    dataset["validation"] = concatenate_datasets(val_ds_list)
    dataset["test"] = concatenate_datasets(test_ds_list)

    # Limit the total number of training samples if specified
    train_limit = kwargs.get("train_sample_limit")
    if train_limit > 0:
        dataset["train"] = (
            dataset["train"]
            .shuffle(seed=SEED)
            .select(range(min(len(dataset["train"]), train_limit)))
        )
    else:
        dataset["train"] = dataset["train"].shuffle(seed=SEED)

    # Limit the total number of validation samples if specified
    val_limit = kwargs.get("val_sample_limit")
    if val_limit > 0:
        dataset["validation"] = (
            dataset["validation"]
            .shuffle(seed=SEED)
            .select(range(min(len(dataset["validation"]), val_limit)))
        )
    else:
        dataset["validation"] = dataset["validation"].shuffle(seed=SEED)

    # Limit the total number of testing samples if specified
    test_limit = kwargs.get("test_sample_limit")
    if test_limit > 0:
        dataset["test"] = (
            dataset["test"]
            .shuffle(seed=SEED)
            .select(range(min(len(dataset["test"]), test_limit)))
        )
    else:
        dataset["test"] = dataset["test"].shuffle(seed=SEED)

    # Save the dataset to Parquet format
    os.makedirs(parquet_ds_dir, exist_ok=True)
    print(f"\nSaving the prepared parquet dataset to {parquet_ds_dir} ...")
    for split in dataset.keys():
        output_path = os.path.join(parquet_ds_dir, f"{split}.parquet")
        print(f"  - Saving {split} split to {output_path} ...")
        dataset[split].to_parquet(output_path)


if __name__ == "__main__":
    args = parse_arguments()
    args_dict = vars(args)
    main(**args_dict)
