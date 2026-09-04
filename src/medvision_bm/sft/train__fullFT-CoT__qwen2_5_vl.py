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
    broadcast_object_from_main,
    format_clean_dataset,
    get_cgroup_limited_cpus,
    load_split_limit_dataset,
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


def _fmt_limit(v):
    """Format a sample limit for the prepared-dataset dir name (<0 = full dataset)."""
    return "full" if v < 0 else str(v)


def _true_size(limit, n_actual):
    """Dir-name token for one train split: the requested cap when set, else its true size."""
    if limit > 0 or n_actual is None:
        return _fmt_limit(limit)
    return str(n_actual)


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
    if new_shape_hw is not None:
        ds_dir_suffix = f"__resized-wh-{new_shape_hw[1]}x{new_shape_hw[0]}"
    else:
        ds_dir_suffix = "__original"

    # The default dir name encodes the TRUE train size of every split whose limit
    # is unset, so it is only known after the load+split stage below. The main
    # process resolves it (a user-specified dir is taken as-is) and broadcasts it
    # to the other ranks after the barrier, so every rank loads the same directory.
    prepared_ds_dir = kwargs.get("prepared_ds_dir")
    if prepared_ds_dir is not None and is_main_process():
        print(
            f"[Info] Using user-specified prepared dataset directory: {prepared_ds_dir}\n"
        )

    if is_main_process():
        # (task label, tasks-list kwarg, train limit, val limit, mapping func, tag_ds)
        task_specs = [
            (
                "AD",
                "tasks_list_json_path_AD",
                train_limit_AD,
                val_limit_AD,
                _format_data_AngleDistanceTask_CoT,
                "BiometricsFromLandmarks",
            ),
            (
                "Detection",
                "tasks_list_json_path_detect",
                train_limit_detect,
                val_limit_detect,
                _format_data_DetectionTask_CoT,
                "BoxSize",
            ),
            (
                "TL",
                "tasks_list_json_path_TL",
                train_limit_TL,
                val_limit_TL,
                _format_data_TumorLesionTask_CoT,
                "TumorLesionSize",
            ),
        ]

        # Stage 1: load and split every requested task FIRST. Row counts are final
        # here (formatting is a row-preserving map), so the true train sizes are
        # known before any directory is named or created. It is needed to NAME the
        # default dir and to FORMAT (stage 2); with an explicit --prepared_ds_dir
        # plus --skip_process_dataset neither applies (the launchers pass the path
        # printed by the prep run), so training loads that dir without touching
        # the raw data.
        raw_ds = {}
        if prepared_ds_dir is None or not kwargs.get("skip_process_dataset"):
            for task_label, path_key, task_train_limit, task_val_limit, _, tag_ds in task_specs:
                if kwargs.get(path_key) is not None:
                    raw_ds[task_label] = _load_split_dataset_task(
                        kwargs, path_key, task_train_limit, task_val_limit, tag_ds=tag_ds
                    )

        if prepared_ds_dir is None:
            n_train = {label: len(ds["train"]) for label, ds in raw_ds.items()}
            prepared_ds_dir = os.path.join(
                data_dir,
                "SFT-CoT_datasets",
                model_family_name,
                f"ds__AD{_true_size(train_limit_AD, n_train.get('AD'))}"
                f"_D{_true_size(train_limit_detect, n_train.get('Detection'))}"
                f"_TL{_true_size(train_limit_TL, n_train.get('TL'))}"
                f"_all{_true_size(train_limit_total, sum(n_train.values()))}"
                + ds_dir_suffix,
            )
            print(
                f"[Info] Using default prepared dataset directory: {prepared_ds_dir}\n"
            )

        if not kwargs.get("skip_process_dataset"):
            # Stage 2: format each task, tag it for temperature sampling, combine.
            train_ds_list = []
            val_ds_list = []
            for task_label, _, _, _, mapping_func, _ in task_specs:
                if task_label not in raw_ds:
                    continue
                dataset_task = _format_dataset_task(
                    kwargs,
                    raw_ds.pop(task_label),
                    mapping_func,
                    model_family_name,
                    base_model_hf,
                    task_label=task_label,
                    temperature_sampler_task_column=kwargs.get(
                        "temperature_sampler_task_column"
                    ),
                )
                train_ds_list.append(dataset_task["train"])
                val_ds_list.append(dataset_task["validation"])

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
            # num_proc must not exceed the smallest split's row count (datasets
            # raises IndexError otherwise, e.g. on tiny smoke-test splits).
            save_workers = max(
                1,
                min(get_cgroup_limited_cpus(), *(len(ds) for ds in dataset.values())),
            )
            dataset.save_to_disk(prepared_ds_dir, num_proc=save_workers)

    barrier()
    # Only the main process resolved the default (true-size) directory name above.
    prepared_ds_dir = broadcast_object_from_main(prepared_ds_dir)

    if kwargs.get("process_dataset_only"):
        if is_main_process():
            print(
                f"Data processing completed. Prepared dataset saved at '{prepared_ds_dir}'."
            )
        return

    dataset = load_from_disk(prepared_ds_dir)

    # Detect an existing checkpoint BEFORE building the trainer: its weights must load
    # through the FSDP-aware from_pretrained path (model_weights_from), not
    # Trainer._load_from_checkpoint, which OOMs on sharded checkpoints under FSDP
    # (see prepare_trainer_fullFT / train_resume_from_checkpoint in sft_utils).
    last_checkpoint = None
    if kwargs.get("resume_from_checkpoint"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_checkpoint = get_last_checkpoint(checkpoint_dir)

    trainer = prepare_trainer_fullFT(
        run_name=run_name,
        base_model_hf=base_model_hf,
        model_weights_from=last_checkpoint,
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
        if last_checkpoint is not None:
            train_resume_from_checkpoint(
                trainer=trainer,
                last_checkpoint=last_checkpoint,
                weights_preloaded=True,
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


def _load_split_dataset_task(kwargs, path_key, train_limit, val_limit, *, tag_ds):
    """Stage 1 of dataset preparation: load one task's MedVision configs and split them."""
    return load_split_limit_dataset(
        tasks_list_json_path=kwargs.get(path_key),
        limit_train_sample=train_limit,
        limit_val_sample=val_limit,
        num_workers_concat_datasets=kwargs.get("num_workers_concat_datasets"),
        tag_ds=tag_ds,
        download_mode=kwargs.get("ds_download_mode"),
    )


def _format_dataset_task(
    kwargs,
    ds,
    mapping_func,
    model_family_name,
    base_model_hf,
    *,
    task_label,
    temperature_sampler_task_column,
):
    """Stage 2: format one task's splits into chat messages and tag them for temperature sampling."""
    ds = format_clean_dataset(
        ds,
        mapping_func=mapping_func,
        model_family_name=model_family_name,
        base_model_hf=base_model_hf,
        num_workers_format_dataset=kwargs.get("num_workers_format_dataset"),
        process_img=kwargs.get("process_img"),
        save_processed_img_to_disk=kwargs.get("save_processed_img_to_disk"),
        new_shape_hw=kwargs.get("new_shape_hw"),
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
