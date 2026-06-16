"""
Full finetuning (no LoRA) variant of train__SFT-CoT__qwen2_5_vl.py.

Key differences from the LoRA script:
  - Model is loaded in BF16 without quantization
  - No PEFT/LoRA config; all parameters are trained
  - Lower learning rate (2e-5 instead of 2e-4)
  - No merge step after training
  - checkpoint_dir replaces lora_checkpoint_dir

Multi-GPU usage:
  > accelerate launch -m medvision_bm.sft.train__fullFT-CoT__qwen2_5_vl [args]
"""

import datetime
import gc
import os

import torch
from accelerate.utils import InitProcessGroupKwargs
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from transformers.trainer_utils import get_last_checkpoint

from medvision_bm.sft.qwen25vl_utils import make_collate_fn_Qwen25VL
from medvision_bm.sft.sft_utils import (
    _format_data_AngleDistanceTask_CoT,
    _format_data_DetectionTask_CoT,
    _format_data_TumorLesionTask_CoT,
    parse_sample_limits,
    parse_validate_args_multiTask,
    prepare_trainer_fullFT,
    train_resume_from_checkpoint,
)
from medvision_bm.utils import setup_env_hf_medvision_ds
from medvision_bm.utils.configs import SEED

pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=1))

try:
    from accelerate import PartialState

    _PS = PartialState()
    _IS_MAIN = _PS.is_main_process

    def is_main_process() -> bool:
        return _IS_MAIN

    def barrier() -> None:
        _PS.wait_for_everyone()

except Exception:

    def is_main_process() -> bool:
        r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        return r in (None, "", "0")

    def barrier() -> None:
        pass


def main(
    run_name,
    model_family_name,
    base_model_hf,
    data_dir,
    checkpoint_dir,
    **kwargs,
):
    if is_main_process():
        setup_env_hf_medvision_ds(data_dir=data_dir)

    (
        train_limit_AD,
        val_limit_AD,
        train_limit_detect,
        val_limit_detect,
        train_limit_TL,
        val_limit_TL,
        train_limit_total,
    ) = parse_sample_limits(**kwargs)

    if is_main_process():
        print(
            "\n[WARNING] The prepared dataset directory name must uniquely include the model identifier and sample limits.\n"
            "Dataset preparation depends on model-specific image processing (e.g., resize scale and pixel dimensions).\n"
            "Reusing a dataset prepared with different settings or a different model may lead to incorrect results."
        )

    new_shape_hw = kwargs.get("new_shape_hw")
    if kwargs.get("prepared_ds_dir") is not None:
        prepared_ds_dir = kwargs.get("prepared_ds_dir")
        if is_main_process():
            print(
                f"[Info] Using user-specified prepared dataset directory: {prepared_ds_dir}\n"
            )
    else:
        prepared_ds_dir = os.path.join(
            data_dir,
            "SFT-CoT_datasets",
            model_family_name,
            f"ds__AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}_all{train_limit_total}",
        )
        if new_shape_hw is not None:
            prepared_ds_dir += f"__resized-wh-{new_shape_hw[1]}x{new_shape_hw[0]}"
        else:
            prepared_ds_dir += "__original"

        if is_main_process():
            os.makedirs(prepared_ds_dir, exist_ok=True)
            print(
                f"[Info] Using default prepared dataset directory: {prepared_ds_dir}\n"
            )

    if is_main_process():
        if not kwargs.get("skip_process_dataset"):

            train_ds_list = []
            val_ds_list = []

            if kwargs.get("tasks_list_json_path_AD") is not None:
                dataset_AD = _prepare_dataset_task(
                    kwargs,
                    "tasks_list_json_path_AD",
                    train_limit_AD,
                    val_limit_AD,
                    _format_data_AngleDistanceTask_CoT,
                    model_family_name,
                    base_model_hf,
                    tag_ds="BiometricsFromLandmarks",
                    task_label="AD",
                    temperature_sampler_task_column=kwargs.get(
                        "temperature_sampler_task_column"
                    ),
                )
                train_ds_list.append(dataset_AD["train"])
                val_ds_list.append(dataset_AD["validation"])

            if kwargs.get("tasks_list_json_path_detect") is not None:
                dataset_detect = _prepare_dataset_task(
                    kwargs,
                    "tasks_list_json_path_detect",
                    train_limit_detect,
                    val_limit_detect,
                    _format_data_DetectionTask_CoT,
                    model_family_name,
                    base_model_hf,
                    tag_ds="BoxSize",
                    task_label="Detection",
                    temperature_sampler_task_column=kwargs.get(
                        "temperature_sampler_task_column"
                    ),
                )
                train_ds_list.append(dataset_detect["train"])
                val_ds_list.append(dataset_detect["validation"])

            if kwargs.get("tasks_list_json_path_TL") is not None:
                dataset_TL = _prepare_dataset_task(
                    kwargs,
                    "tasks_list_json_path_TL",
                    train_limit_TL,
                    val_limit_TL,
                    _format_data_TumorLesionTask_CoT,
                    model_family_name,
                    base_model_hf,
                    tag_ds="TumorLesionSize",
                    task_label="TL",
                    temperature_sampler_task_column=kwargs.get(
                        "temperature_sampler_task_column"
                    ),
                )
                train_ds_list.append(dataset_TL["train"])
                val_ds_list.append(dataset_TL["validation"])

            dataset = DatasetDict()
            dataset["train"] = concatenate_datasets(train_ds_list)
            dataset["validation"] = concatenate_datasets(val_ds_list)

            train_limit = kwargs.get("train_sample_limit")
            if train_limit > 0:
                train_size = len(dataset["train"])
                if train_limit > train_size:
                    import numpy as np

                    np.random.seed(SEED)
                    indices = np.random.choice(
                        train_size, size=train_limit, replace=True
                    )
                    dataset["train"] = dataset["train"].select(indices)
                else:
                    dataset["train"] = (
                        dataset["train"].shuffle(seed=SEED).select(range(train_limit))
                    )
            else:
                dataset["train"] = dataset["train"].shuffle(seed=SEED)

            val_limit = kwargs.get("val_sample_limit")
            if val_limit > 0:
                val_size = len(dataset["validation"])
                if val_limit > val_size:
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

            os.makedirs(prepared_ds_dir, exist_ok=True)
            dataset.save_to_disk(prepared_ds_dir)

    barrier()

    if kwargs.get("process_dataset_only"):
        if is_main_process():
            print(
                f"Data processing completed. Prepared dataset saved at '{prepared_ds_dir}'."
            )
        return

    dataset = load_from_disk(prepared_ds_dir)

    trainer = prepare_trainer_fullFT(
        run_name=run_name,
        base_model_hf=base_model_hf,
        checkpoint_dir=checkpoint_dir,
        data=dataset,
        make_collate_fn=make_collate_fn_Qwen25VL,
        per_device_train_batch_size=kwargs.get("per_device_train_batch_size"),
        per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size"),
        gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps"),
        use_flash_attention_2=kwargs.get("use_flash_attention_2"),
        num_train_epochs=kwargs.get("epoch"),
        save_steps=kwargs.get("save_steps"),
        eval_steps=kwargs.get("eval_steps"),
        logging_steps=kwargs.get("logging_steps"),
        save_total_limit=kwargs.get("save_total_limit"),
        dataloader_num_workers=kwargs.get("dataloader_num_workers"),
        gradient_checkpointing=kwargs.get("gradient_checkpointing"),
        dataloader_pin_memory=kwargs.get("dataloader_pin_memory"),
        push_model=kwargs.get("push_LoRA"),  # reuse existing CLI arg
        enable_temperature_sampler=kwargs.get("enable_temperature_sampler"),
        temperature_sampler_T=kwargs.get("temperature_sampler_T"),
        temperature_sampler_task_column=kwargs.get("temperature_sampler_task_column"),
        temperature_sampler_num_samples=kwargs.get("temperature_sampler_num_samples"),
    )

    if kwargs.get("resume_from_checkpoint"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        if last_checkpoint is not None:
            train_resume_from_checkpoint(
                trainer=trainer, last_checkpoint=last_checkpoint
            )
        else:
            if is_main_process():
                print(
                    f"No valid checkpoint found in '{checkpoint_dir}'. Starting training from scratch."
                )
            trainer.train()
    else:
        trainer.train()

    # Save the trained model
    trainer.save_model()

    try:
        del trainer
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


def _prepare_dataset_task(
    kwargs,
    path_key,
    train_limit,
    val_limit,
    mapping_func,
    model_family_name,
    base_model_hf,
    *,
    tag_ds,
    task_label,
    temperature_sampler_task_column,
):
    from medvision_bm.sft.sft_utils import prepare_dataset

    ds = prepare_dataset(
        tasks_list_json_path=kwargs.get(path_key),
        limit_train_sample=train_limit,
        limit_val_sample=val_limit,
        mapping_func=mapping_func,
        model_family_name=model_family_name,
        base_model_hf=base_model_hf,
        num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
        num_workers_format_dataset=kwargs.get("num_workers_format_dataset"),
        tag_ds=tag_ds,
        process_img=kwargs.get("process_img"),
        save_processed_img_to_disk=kwargs.get("save_processed_img_to_disk"),
        new_shape_hw=kwargs.get("new_shape_hw"),
        download_mode=kwargs.get("ds_download_mode"),
    )
    ds["train"] = ds["train"].add_column(
        temperature_sampler_task_column, [task_label] * len(ds["train"])
    )
    ds["validation"] = ds["validation"].add_column(
        temperature_sampler_task_column, [task_label] * len(ds["validation"])
    )
    return ds


if __name__ == "__main__":
    args_dict = parse_validate_args_multiTask()
    # parse_validate_args_multiTask expects lora_checkpoint_dir; remap to checkpoint_dir
    args_dict["checkpoint_dir"] = args_dict.pop("lora_checkpoint_dir")
    main(**args_dict)
