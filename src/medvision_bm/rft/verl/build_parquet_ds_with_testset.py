import argparse
import os

from datasets import DatasetDict, concatenate_datasets

from medvision_bm.rft.verl.verl_utils import (
    _format_data_AngleDistanceTask_CoT_verl,
    _format_data_AngleDistanceTask_verl,
    _format_data_DetectionTask_CoT_verl,
    _format_data_DetectionTask_verl,
    _format_data_TumorLesionTask_CoT_verl,
    _format_data_TumorLesionTask_verl,
    prepare_dataset_for_verl_with_testset,
)
from medvision_bm.sft.sft_utils import parse_sample_limits
from medvision_bm.utils.configs import SEED


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Build parquet dataset for RL finetuning in Verl framework."
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
        help="Target resize shape as (height, width). Ignore to use the original size. Example: --new_shape_hw 1080 1920. Result: args.new_shape_hw → [1080, 1920]"
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
        default=32,
        help="Number of workers for formatting datasets",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    # -- Sample limits
    # This is only for multi-task training to limit the number of samples per task
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
    # Task-specific sample limit
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
    # This is to limit the number of samples in total
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
        help="Limit the number of testing samples, -1 means no limit",
    )
    # Dataset limit per subset (i.e., dataset config)
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

    args = parser.parse_args()

    return args


def build_parquet_dataset(**kwargs):
    model_family_name = kwargs.get("model_family_name")
    data_dir = kwargs.get("data_dir")

    # Parse sample limits
    (
        train_limit_AD,
        val_limit_AD,
        train_limit_detect,
        val_limit_detect,
        train_limit_TL,
        val_limit_TL,
        train_limit_total,
    ) = parse_sample_limits(**kwargs)

    # Prepare the dataset cache directory
    # NOTE:
    # IMPORTANT: The prepared dataset directory must uniquely encode the sample limits and the model identifier.
    # This is because dataset preparation performs model-specific processing (for example, the model's image_processor
    # determines image resize ratios and final pixel dimensions). Loading a dataset prepared with different limits
    # or a different model can produce incorrect preprocessing or mismatched prompts.
    print(
        "\n[WARNING] The prepared dataset directory name must uniquely include the model identifier and sample limits.\n"
        "Dataset preparation depends on model-specific image processing (e.g., resize scale and pixel dimensions).\n"
        "Reusing a dataset prepared with different settings or a different model may lead to incorrect results."
    )

    # NOTE: TODO
    # The CLI argument --without_cot_instruction is deprecated, as we want to keep the CoT instructions RFT training. 
    # The main reason is that we adpot the SFT-CoT + RFT training pipeline, where the SFT-CoT stage use the CoT instructions in the prompts and we want to keep the prompts consistent between SFT and RFT training. 
    # If we remove the CoT instructions in the RFT stage, there will be a distribution shift between the SFT and RFT training data.

    # Prepare the output parquet dataset directory
    if kwargs.get("without_cot_instruction"):
        cot_tag = "_wo-CoT-Instruct"
    else:
        cot_tag = ""
    if kwargs.get("new_shape_hw") is not None:
        ds_dir = f"ds__AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}_all{train_limit_total}{cot_tag}__resized-hw-{kwargs.get('new_shape_hw')[0]}x{kwargs.get('new_shape_hw')[1]}"
    else:
        ds_dir = f"ds__AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}_all{train_limit_total}{cot_tag}__original"
    parquet_ds_dir = os.path.join(
        data_dir,
        "verl_datasets",
        model_family_name,
        ds_dir,
    )
    print(f"\nPrepared Verl parquet dataset directory: {parquet_ds_dir}")

    # Convert -1 sentinel → None for per-subset limits (matching load_split_limit_dataset_tr_val_ts convention)
    _per_subset_train = kwargs.get("train_sample_limit_per_subset") or None
    if _per_subset_train is not None and _per_subset_train < 0:
        _per_subset_train = None
    _per_subset_test = kwargs.get("test_sample_limit_per_subset") or None
    if _per_subset_test is not None and _per_subset_test < 0:
        _per_subset_test = None

    # Prepare datasets for Verl
    train_ds_list = []
    val_ds_list = []
    test_ds_list = []
    if kwargs.get("tasks_list_json_path_AD") is not None:
        format_func = _format_data_AngleDistanceTask_CoT_verl if not kwargs.get("without_cot_instruction") else _format_data_AngleDistanceTask_verl
        dataset_AD = prepare_dataset_for_verl_with_testset(
            tasks_list_json_path=kwargs.get("tasks_list_json_path_AD"),
            limit_train_sample=train_limit_AD,
            limit_val_sample=val_limit_AD,
            limit_test_sample=kwargs.get("test_sample_limit_task_AD"),
            limit_train_sample_per_subset=_per_subset_train,
            limit_test_sample_per_subset=_per_subset_test,
            mapping_func=format_func,
            model_family_name=model_family_name,
            model_hf=kwargs.get("model_hf"),
            num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
            num_workers_format_dataset=kwargs.get("num_workers_format_dataset"),
            # MedVision dataset specific, used to extract dataset name from AD task configs
            tag_ds="BiometricsFromLandmarks",
            new_shape_hw=kwargs.get("new_shape_hw"),
            download_mode=kwargs.get("ds_download_mode"),
        )
        train_ds_list.append(dataset_AD["train"])
        val_ds_list.append(dataset_AD["validation"])
        test_ds_list.append(dataset_AD["test"])
    if kwargs.get("tasks_list_json_path_TL") is not None:
        format_func = _format_data_TumorLesionTask_CoT_verl if not kwargs.get("without_cot_instruction") else _format_data_TumorLesionTask_verl
        dataset_TL = prepare_dataset_for_verl_with_testset(
            tasks_list_json_path=kwargs.get("tasks_list_json_path_TL"),
            limit_train_sample=train_limit_TL,
            limit_val_sample=val_limit_TL,
            limit_test_sample=kwargs.get("test_sample_limit_task_TL"),
            limit_train_sample_per_subset=_per_subset_train,
            limit_test_sample_per_subset=_per_subset_test,
            mapping_func=format_func,
            model_family_name=model_family_name,
            model_hf=kwargs.get("model_hf"),
            num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
            num_workers_format_dataset=kwargs.get("num_workers_format_dataset"),
            # MedVision dataset specific, used to extract dataset name from AD task configs
            tag_ds="TumorLesionSize",
            new_shape_hw=kwargs.get("new_shape_hw"),
            download_mode=kwargs.get("ds_download_mode"),
        )
        train_ds_list.append(dataset_TL["train"])
        val_ds_list.append(dataset_TL["validation"])
        test_ds_list.append(dataset_TL["test"])
    if kwargs.get("tasks_list_json_path_detect") is not None:
        # TODO: implement _format_data_DetectionTask_CoT_verl() and _format_data_DetectionTask_verl()
        format_func = _format_data_DetectionTask_CoT_verl if not kwargs.get("without_cot_instruction") else _format_data_DetectionTask_verl
        dataset_detect = prepare_dataset_for_verl_with_testset(
            tasks_list_json_path=kwargs.get("tasks_list_json_path_detect"),
            limit_train_sample=train_limit_detect,
            limit_val_sample=val_limit_detect,
            limit_test_sample=kwargs.get("test_sample_limit_task_Detection"),
            limit_train_sample_per_subset=_per_subset_train,
            limit_test_sample_per_subset=_per_subset_test,
            mapping_func=format_func,
            model_family_name=model_family_name,
            model_hf=kwargs.get("model_hf"),
            num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
            num_workers_format_dataset=kwargs.get("num_workers_format_dataset"),
            # MedVision dataset specific, used to extract dataset name from AD task configs
            tag_ds="BoxSize",
            new_shape_hw=kwargs.get("new_shape_hw"),
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

    # Limit the training samples (allow sampling with replacement if limit exceeds dataset size)
    train_limit = kwargs.get("train_sample_limit")
    if train_limit > 0:
        train_size = len(dataset["train"])
        if train_limit > train_size:
            import numpy as np
            np.random.seed(SEED)
            indices = np.random.choice(train_size, size=train_limit, replace=True)
            dataset["train"] = dataset["train"].select(indices)
        else:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=SEED)
                .select(range(train_limit))
            )
    else:
        dataset["train"] = dataset["train"].shuffle(seed=SEED)

    # Limit the validation samples (allow sampling with replacement if limit exceeds dataset size)
    val_limit = kwargs.get("val_sample_limit", -1)
    if val_limit > 0:
        val_size = len(dataset["validation"])
        if val_limit > val_size:
            # Allow sampling with replacement if limit exceeds dataset size
            import numpy as np
            np.random.seed(SEED)
            indices = np.random.choice(val_size, size=val_limit, replace=True)
            dataset["validation"] = dataset["validation"].select(indices)
        else:
            dataset["validation"] = (
                dataset["validation"]
                .shuffle(seed=SEED)
                .select(range(val_limit))
            )
    else:
        dataset["validation"] = dataset["validation"].shuffle(seed=SEED)

    # Limit the test samples (allow sampling with replacement if limit exceeds dataset size)
    test_limit = kwargs.get("test_sample_limit")
    if test_limit > 0:
        test_size = len(dataset["test"])
        if test_limit > test_size:
            # Allow sampling with replacement if limit exceeds dataset size
            import numpy as np
            np.random.seed(SEED)
            indices = np.random.choice(test_size, size=test_limit, replace=True)
            dataset["test"] = dataset["test"].select(indices)
        else:
            dataset["test"] = (
                dataset["test"]
                .shuffle(seed=SEED)
                .select(range(test_limit))
            )
    else:
        dataset["test"] = dataset["test"].shuffle(seed=SEED)


    # Save the dataset to Parquet format
    os.makedirs(parquet_ds_dir, exist_ok=True)
    print(f"\nSaving the prepared Verl parquet dataset to {parquet_ds_dir} ...")
    for split in dataset.keys():
        output_path = os.path.join(parquet_ds_dir, f"{split}_verl.parquet")
        print(f"  - Saving {split} split to {output_path} ...")
        dataset[split].to_parquet(output_path)


def main():
    args = parse_arguments()
    args_dict = vars(args)
    build_parquet_dataset(**args_dict)


if __name__ == "__main__":
    main()
