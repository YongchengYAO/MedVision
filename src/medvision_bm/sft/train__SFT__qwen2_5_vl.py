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
import os

from accelerate.utils import InitProcessGroupKwargs
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from transformers import AutoProcessor
from transformers.trainer_utils import get_last_checkpoint

from medvision_bm.sft.qwen25vl_utils import make_collate_fn_Qwen25VL
from medvision_bm.sft.utils import (_format_data_AngleDistanceTask,
                                    _format_data_DetectionTask,
                                    _format_data_TumorLesionTask,
                                    cleanup_all_gpu, merge_models,
                                    parse_validate_args_multiTask,
                                    prepare_dataset, prepare_trainer,
                                    train_resume_from_checkpoint)
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


def main(
    run_name,
    base_model_hf,
    data_dir,
    tasks_list_json_path_AD,
    tasks_list_json_path_detect,
    tasks_list_json_path_TL,
    lora_checkpoint_dir,
    **kwargs,
):
    # Set up the environment variables for Hugging Face and medvision_ds
    setup_env_hf_medvision_ds(data_dir=data_dir)

    if not kwargs.get("merge_only"):
        # Determine sample limits for each task
        if kwargs.get("train_sample_limit_task_AD") > 0:
            train_limit_AD = kwargs.get("train_sample_limit_task_AD")
        else:
            train_limit_AD = kwargs.get("train_sample_limit_per_task")
        if kwargs.get("val_sample_limit_task_AD") > 0:
            val_limit_AD = kwargs.get("val_sample_limit_task_AD")
        else:
            val_limit_AD = kwargs.get("val_sample_limit_per_task")
        if kwargs.get("train_sample_limit_task_Detection") > 0:
            train_limit_detect = kwargs.get(
                "train_sample_limit_task_Detection")
        else:
            train_limit_detect = kwargs.get("train_sample_limit_per_task")
        if kwargs.get("val_sample_limit_task_Detection") > 0:
            val_limit_detect = kwargs.get(
                "val_sample_limit_task_Detection")
        else:
            val_limit_detect = kwargs.get("val_sample_limit_per_task")
        if kwargs.get("train_sample_limit_task_TL") > 0:
            train_limit_TL = kwargs.get("train_sample_limit_task_TL")
        else:
            train_limit_TL = kwargs.get("train_sample_limit_per_task")
        if kwargs.get("val_sample_limit_task_TL") > 0:
            val_limit_TL = kwargs.get("val_sample_limit_task_TL")
        else:
            val_limit_TL = kwargs.get("val_sample_limit_per_task")
        train_limit_total = kwargs.get("train_sample_limit")

        # Prepare the dataset cache directory
        if kwargs.get("prepared_ds_dir") is not None:
            prepared_ds_dir = kwargs.get("prepared_ds_dir")
        else:
            prepared_ds_dir = os.path.join(
                data_dir, f"tmp_prepared_ds_AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}_all{train_limit_total}")

        # Prepare the dataset on the main process ONLY
        if is_main_process():
            if not kwargs.get("skip_process_dataset"):
                img_processor = AutoProcessor.from_pretrained(
                    base_model_hf).image_processor

                train_ds_list = []
                val_ds_list = []

                if tasks_list_json_path_AD is not None:
                    # Prepare datasets for AD task
                    dataset_AD = prepare_dataset(
                        tasks_list_json_path=tasks_list_json_path_AD,
                        limit_train_sample=train_limit_AD,
                        limit_val_sample=val_limit_AD,
                        mapping_func=_format_data_AngleDistanceTask,
                        num_workers_concat_datasets=kwargs.get(
                            "num_workers_concat_datasets"),
                        num_workers_format_dataset=kwargs.get(
                            "num_workers_format_dataset"),
                        # MedVision dataset specific, used to extract dataset name from AD task configs
                        tag_ds="BiometricsFromLandmarks",
                        img_processor=img_processor,
                        process_img=kwargs.get("process_img"),
                        save_processed_img_to_disk=kwargs.get(
                            "save_processed_img_to_disk"),
                    )
                    train_ds_list.append(dataset_AD["train"])
                    val_ds_list.append(dataset_AD["validation"])

                if tasks_list_json_path_detect is not None:
                    # Prepare datasets for Detection task
                    dataset_detect = prepare_dataset(
                        tasks_list_json_path=tasks_list_json_path_detect,
                        limit_train_sample=train_limit_detect,
                        limit_val_sample=val_limit_detect,
                        mapping_func=_format_data_DetectionTask,
                        num_workers_concat_datasets=kwargs.get(
                            "num_workers_concat_datasets"),
                        num_workers_format_dataset=kwargs.get(
                            "num_workers_format_dataset"),
                        # MedVision dataset specific, used to extract dataset name from detection task configs
                        tag_ds="BoxSize",
                        img_processor=img_processor,
                        process_img=kwargs.get("process_img"),
                        save_processed_img_to_disk=kwargs.get(
                            "save_processed_img_to_disk"),
                    )
                    train_ds_list.append(dataset_detect["train"])
                    val_ds_list.append(dataset_detect["validation"])

                if tasks_list_json_path_TL is not None:
                    # Prepare datasets for Tumor Lesion Size task
                    dataset_TL = prepare_dataset(
                        tasks_list_json_path=tasks_list_json_path_TL,
                        limit_train_sample=train_limit_TL,
                        limit_val_sample=val_limit_TL,
                        mapping_func=_format_data_TumorLesionTask,
                        num_workers_concat_datasets=kwargs.get(
                            "num_workers_concat_datasets"),
                        num_workers_format_dataset=kwargs.get(
                            "num_workers_format_dataset"),
                        # MedVision dataset specific, used to extract dataset name from TL task configs
                        tag_ds="TumorLesionSize",
                        img_processor=img_processor,
                        process_img=kwargs.get("process_img"),
                        save_processed_img_to_disk=kwargs.get(
                            "save_processed_img_to_disk"),
                    )
                    train_ds_list.append(dataset_TL["train"])
                    val_ds_list.append(dataset_TL["validation"])

                # Combine all tasks' datasets
                dataset = DatasetDict()
                dataset["train"] = concatenate_datasets(train_ds_list)
                dataset["validation"] = concatenate_datasets(val_ds_list)

                # Limit the total number of samples if specified
                dataset["train"] = (
                    dataset["train"].shuffle(seed=SEED).select(
                        range(kwargs.get("train_sample_limit")))
                )
                dataset["validation"] = (
                    dataset["validation"].shuffle(seed=SEED).select(
                        range(kwargs.get("val_sample_limit")))
                )

                # Save the prepared dataset to disk for other processes to load
                os.makedirs(prepared_ds_dir, exist_ok=True)
                dataset.save_to_disk(prepared_ds_dir)
        barrier()

        # Stop here if only processing dataset
        if kwargs.get("process_dataset_only"):
            if is_main_process():
                print(
                    f"Data processing completed. Prepared dataset saved at '{prepared_ds_dir}'.")
            cleanup_all_gpu()
            return

        # All processes load the prepared dataset
        dataset = load_from_disk(prepared_ds_dir)

        # Prepare trainer (DO NOT guard this with is_main_process())
        trainer = prepare_trainer(
            run_name,
            base_model_hf,
            lora_checkpoint_dir=lora_checkpoint_dir,
            data=dataset,
            make_collate_fn=make_collate_fn_Qwen25VL,
            per_device_train_batch_size=kwargs.get(
                "per_device_train_batch_size"),
            per_device_eval_batch_size=kwargs.get(
                "per_device_eval_batch_size"),
            gradient_accumulation_steps=kwargs.get(
                "gradient_accumulation_steps"),
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
        )

        # Train the model (DO NOT guard this with is_main_process())
        if kwargs.get("resume_from_checkpoint"):
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
        if is_main_process():
            trainer.save_model()
        barrier()

        cleanup_all_gpu()

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
        barrier()

    cleanup_all_gpu()


if __name__ == "__main__":
    args_dict = parse_validate_args_multiTask()
    main(**args_dict)
