"""
Tutorial:
    - medgemma finetuning: https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
    - other visual SFT: https://huggingface.co/docs/trl/main/en/training_vlm_sft
                        https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md

Trainer (and thus SFTTrainer) supports multi-GPU training.
If you run your script with `python script.py` it will default to using DP as the strategy, which may be slower than expected.
To use DDP (which is generally recommended, see here for more info) you must launch the script with
> python -m torch.distributed.launch script.py
or
> accelerate launch script.py
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
    _format_data_AngleDistanceTask,
    _format_data_DetectionTask,
    _format_data_TumorLesionTask,
    broadcast_object_from_main,
    format_clean_dataset,
    get_cgroup_limited_cpus,
    load_split_limit_dataset,
    merge_models,
    parse_sample_limits,
    parse_validate_args_multiTask,
    prepare_trainer,
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
        """Return True only on the main (rank 0) process."""
        return _IS_MAIN

    def barrier() -> None:
        """Synchronize all processes (no-op in single process)."""
        _PS.wait_for_everyone()

except Exception:
    # Fallback if Accelerate or torch.distributed is unavailable
    def is_main_process() -> bool:
        """Best-effort check for main process in non-distributed runs."""
        r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        return r in (None, "", "0")

    def barrier() -> None:
        """No-op fallback for single-process runs."""
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
    lora_checkpoint_dir,
    **kwargs,
):
    # Set up the environment variables for Hugging Face and medvision_ds
    if is_main_process():
        setup_env_hf_medvision_ds(data_dir=data_dir)

    if not kwargs.get("merge_only"):
        # NOTE: Keep it here (out of the main process block) as it is used in all processes for dataset loading later
        # ---
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
        # ---

        # Print a clear runtime warning on the main process so users notice this requirement
        if is_main_process():
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

        # ---
        # Prepared dataset directory. The default name encodes the TRUE train size of
        # every split whose limit is unset, so it is only known after the load+split
        # stage below. The main process resolves it (a user-specified dir is taken
        # as-is) and broadcasts it to the other ranks after the barrier, so every
        # rank loads the same directory.
        # ---
        new_shape_hw = kwargs.get("new_shape_hw")
        if new_shape_hw is not None:
            ds_dir_suffix = f"__resized-wh-{new_shape_hw[1]}x{new_shape_hw[0]}"
        else:
            ds_dir_suffix = "__original"
        prepared_ds_dir = kwargs.get("prepared_ds_dir")
        if prepared_ds_dir is not None and is_main_process():
            print(
                f"[Info] Using user-specified prepared dataset directory: {prepared_ds_dir}\n"
            )

        # Prepare the dataset on the main process ONLY
        if is_main_process():
            # (task label, tasks-list kwarg, train limit, val limit, mapping func, tag_ds)
            task_specs = [
                (
                    "AD",
                    "tasks_list_json_path_AD",
                    train_limit_AD,
                    val_limit_AD,
                    _format_data_AngleDistanceTask,
                    "BiometricsFromLandmarks",
                ),
                (
                    "Detection",
                    "tasks_list_json_path_detect",
                    train_limit_detect,
                    val_limit_detect,
                    _format_data_DetectionTask,
                    "BoxSize",
                ),
                (
                    "TL",
                    "tasks_list_json_path_TL",
                    train_limit_TL,
                    val_limit_TL,
                    _format_data_TumorLesionTask,
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
                # Default folder with naming convention encoding model identifier and sample sizes
                n_train = {label: len(ds["train"]) for label, ds in raw_ds.items()}
                prepared_ds_dir = os.path.join(
                    data_dir,
                    "SFT_datasets",
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

                # Combine all tasks' datasets
                dataset = DatasetDict()
                dataset["train"] = concatenate_datasets(train_ds_list)
                dataset["validation"] = concatenate_datasets(val_ds_list)

                # Limit the training samples (allow sampling with replacement if limit exceeds dataset size)
                train_limit = kwargs.get("train_sample_limit")
                if train_limit > 0:
                    train_size = len(dataset["train"])
                    if train_limit > train_size:
                        # Allow sampling with replacement if limit exceeds dataset size
                        import numpy as np

                        np.random.seed(SEED)
                        indices = np.random.choice(
                            train_size, size=train_limit, replace=True
                        )
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
                val_limit = kwargs.get("val_sample_limit")
                if val_limit > 0:
                    val_size = len(dataset["validation"])
                    if val_limit > val_size:
                        # Allow sampling with replacement if limit exceeds dataset size
                        import numpy as np

                        np.random.seed(SEED)
                        indices = np.random.choice(
                            val_size, size=val_limit, replace=True
                        )
                        dataset["validation"] = dataset["validation"].select(indices)
                    else:
                        dataset["validation"] = (
                            dataset["validation"]
                            .shuffle(seed=SEED)
                            .select(range(val_limit))
                        )
                else:
                    dataset["validation"] = dataset["validation"].shuffle(seed=SEED)

                # Save the prepared dataset to disk for other processes to load
                os.makedirs(prepared_ds_dir, exist_ok=True)
                # num_proc must not exceed the smallest split's row count (datasets
                # raises IndexError otherwise, e.g. on tiny smoke-test splits).
                save_workers = max(
                    1,
                    min(get_cgroup_limited_cpus(), *(len(ds) for ds in dataset.values())),
                )
                dataset.save_to_disk(prepared_ds_dir, num_proc=save_workers)

        # All processes synchronize here: wait for dataset preparation to complete
        barrier()
        # Only the main process resolved the default (true-size) directory name above.
        prepared_ds_dir = broadcast_object_from_main(prepared_ds_dir)

        # Stop here if only processing dataset
        if kwargs.get("process_dataset_only"):
            if is_main_process():
                print(
                    f"Data processing completed. Prepared dataset saved at '{prepared_ds_dir}'."
                )
            return

        # All processes load the prepared dataset
        dataset = load_from_disk(prepared_ds_dir)

        # Prepare trainer (DO NOT guard this with is_main_process())
        trainer = prepare_trainer(
            run_name=run_name,
            base_model_hf=base_model_hf,
            lora_checkpoint_dir=lora_checkpoint_dir,
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
            # Maximum number of checkpoints to save
            save_total_limit=kwargs.get("save_total_limit"),
            dataloader_num_workers=kwargs.get("dataloader_num_workers"),
            gradient_checkpointing=kwargs.get("gradient_checkpointing"),
            dataloader_pin_memory=kwargs.get("dataloader_pin_memory"),
            push_LoRA=kwargs.get("push_LoRA"),
            enable_temperature_sampler=kwargs.get("enable_temperature_sampler"),
            temperature_sampler_T=kwargs.get("temperature_sampler_T"),
            temperature_sampler_task_column=kwargs.get(
                "temperature_sampler_task_column"
            ),
            temperature_sampler_num_samples=kwargs.get(
                "temperature_sampler_num_samples"
            ),
        )

        # Train the model (DO NOT guard this with is_main_process())
        if kwargs.get("resume_from_checkpoint"):
            # Create LoRA checkpoint directory if it doesn't exist
            # This is needed even if this is the first run
            os.makedirs(lora_checkpoint_dir, exist_ok=True)

            last_checkpoint = get_last_checkpoint(lora_checkpoint_dir)
            if last_checkpoint is not None:
                train_resume_from_checkpoint(
                    trainer=trainer,
                    last_checkpoint=last_checkpoint,
                )
            else:
                if is_main_process():
                    print(
                        f"No valid checkpoint found in '{lora_checkpoint_dir}'. Starting training from scratch."
                    )
                trainer.train()
        else:
            trainer.train()

        # Save the trained model
        trainer.save_model()

    # Free VRAM
    # Safe delete trainer only if it exists (prevents NameError when trainer was never created)
    if "trainer" in locals():
        try:
            del trainer
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()

    # Optionally merge LoRA with base model and push to Hub
    if kwargs.get("merge_model") or kwargs.get("merge_only"):
        if is_main_process():
            merge_models(
                base_model_hf=base_model_hf,
                lora_checkpoint_dir=lora_checkpoint_dir,
                merged_model_hf=kwargs.get("merged_model_hf"),
                merged_model_dir=kwargs.get("merged_model_dir"),
                push_to_hub=kwargs.get("push_merged_model"),
            )


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
    main(**args_dict)
