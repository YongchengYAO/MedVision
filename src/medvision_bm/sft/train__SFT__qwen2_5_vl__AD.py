"""
Tutorial:
    - medgemma finetuning: https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
    - other visual SFT: https://huggingface.co/docs/trl/training_vlm_sft

NOTE:
ref: https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md#multi-gpu-training

Trainer (and thus SFTTrainer) supports multi-GPU training.gg
If you run your script with `python script.py` it will default to using DP as the strategy, which may be slower than expected.
To use DDP (which is generally recommended, see here for more info) you must launch the script with
> python -m torch.distributed.launch script.py
or
> accelerate launch script.py
"""

import argparse
import os

from transformers import AutoProcessor

from medvision_bm.sft.utils import (cleanup_all_gpu, is_main_process,
                                    merge_models,
                                    prepare_dataset_angle_distance,
                                    prepare_trainer,
                                    train_resume_from_checkpoint)
from medvision_bm.utils import setup_env_hf_medvision_ds


def main(
    run_name,
    base_model_hf,
    data_dir,
    tasks_list_json_path,
    lora_checkpoint_dir,
    **kwargs,
):
    # Set up the environment variables for Hugging Face and medvision_ds
    setup_env_hf_medvision_ds(data_dir=data_dir)

    if not kwargs.get("merge_only"):
        # Prepare the dataset
        img_processor = AutoProcessor.from_pretrained(
            base_model_hf).image_processor
        dataset = prepare_dataset_angle_distance(
            tasks_list_json_path=tasks_list_json_path,
            limit_train_sample=kwargs.get("train_sample_limit"),
            limit_val_sample=kwargs.get("val_sample_limit"),
            num_workers_concat_datasets=kwargs.get(
                "num_workers_concat_datasets"),
            num_workers_format_dataset=kwargs.get(
                "num_workers_format_dataset"),
            model_hf=base_model_hf,
            tag_ds="BiometricsFromLandmarks",
            img_processor=img_processor,
        )

        # Prepare trainer
        trainer = prepare_trainer(
            run_name,
            base_model_hf,
            lora_checkpoint_dir=lora_checkpoint_dir,
            data=dataset,
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

        from transformers.trainer_utils import get_last_checkpoint

        # Train the model
        if kwargs.get("resume_from_checkpoint"):
            last_checkpoint = get_last_checkpoint(lora_checkpoint_dir)
            if last_checkpoint is not None:
                train_resume_from_checkpoint(
                    trainer=trainer,
                    last_checkpoint=last_checkpoint,
                    gradient_accumulation_steps=kwargs.get(
                        "gradient_accumulation_steps"),
                    num_train_epochs=kwargs.get("epoch"),
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

        cleanup_all_gpu()

    # Optionally merge LoRA with base model and push to Hub
    if kwargs.get("merge_model") or kwargs.get("merge_only"):
        if kwargs.get("push_merged_model"):
            merge_models(
                base_model_hf=base_model_hf,
                lora_checkpoint_dir=lora_checkpoint_dir,
                merged_model_hf=kwargs.get("merged_model_hf"),
                merged_model_dir=kwargs.get("merged_model_dir"),
                push_to_hub=True,
            )
        else:
            merge_models(
                base_model_hf=base_model_hf,
                lora_checkpoint_dir=lora_checkpoint_dir,
                merged_model_hf=kwargs.get("merged_model_hf"),
                merged_model_dir=kwargs.get("merged_model_dir"),
                push_to_hub=False,
            )

    cleanup_all_gpu()


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT on the MedVision dataset"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the run",
    )

    # -- Model arguments
    parser.add_argument(
        "--base_model_hf",
        type=str,
        help="Hugging Face model ID for the base model",
    )
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        help="Local directory path for LoRA checkpoint",
    )
    parser.add_argument(
        "--merged_model_hf",
        type=str,
        help="Hugging Face repository ID for merged model",
    )
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        help="Local directory path for merged model",
    )

    # -- wandb logging arguments
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default="allow",
        help="Wandb resume mode (e.g., 'allow', 'must', 'never')",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        help="Directory for wandb logs",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        help="Wandb run ID for resuming",
    )

    # -- Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Dataset folder",
    )
    parser.add_argument(
        "--tasks_list_json_path",
        type=str,
        help="Path to the tasks list JSON file",
    )

    # -- Training arguments
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between model saves",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=10,
        help="Maximum number of checkpoints to save",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=20,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=20,
        help="Batch size per device during evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps before performing a backward/update pass",
    )
    parser.add_argument(
        "--use_flash_attention_2",
        type=str,
        default=True,
        help="Use Flash Attention 2 for training",
    )
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
    parser.add_argument(
        "--train_sample_limit",
        type=int,
        default=1000,
        help="Limit the number of training samples",
    )
    parser.add_argument(
        "--val_sample_limit",
        type=int,
        default=200,
        help="Limit the number of validation samples",
    )
    parser.add_argument(
        "--push_LoRA",
        type=str,
        default=False,
        help="Push LoRA checkpoint to HF Hub after each save",
    )
    parser.add_argument(
        "--push_merged_model",
        type=str,
        default=False,
        help="Push merged model to HF Hub after merging",
    )
    parser.add_argument(
        "--merge_model",
        type=str,
        default=False,
        help="Merge LoRA with base model after training",
    )
    parser.add_argument(
        "--merge_only",
        type=str,
        default=False,
        help="ONLY Merge LoRA with base model and push to HF Hub, no training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=False,
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        default=False,
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=str,
        default=True,
        help="Pin memory for faster GPU transfer",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Arguments
    # ------------------------------------------------------------
    run_name = args.run_name
    # -- Model
    base_model_hf = args.base_model_hf
    lora_checkpoint_dir = args.lora_checkpoint_dir
    # -- wandb logging
    wandb_resume = args.wandb_resume
    wandb_dir = args.wandb_dir
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_run_id = args.wandb_run_id
    # -- Data
    data_dir = args.data_dir
    tasks_list_json_path = args.tasks_list_json_path
    # ------------------------------------------------------------

    assert base_model_hf is not None, "--base_model_hf is required"
    assert lora_checkpoint_dir is not None, "--lora_checkpoint_dir is required"
    assert data_dir is not None, "--data_dir is required"
    assert tasks_list_json_path is not None, "--tasks_list_json_path is required"
    os.makedirs(lora_checkpoint_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Set wandb environment variables
    os.environ["WANDB_RESUME"] = wandb_resume
    if wandb_dir is not None:
        os.environ["WANDB_DIR"] = wandb_dir
        os.makedirs(wandb_dir, exist_ok=True)
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name is not None:
        os.environ["WANDB_NAME"] = wandb_run_name
    if wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = wandb_run_id

    # Prepare kwargs: remove explicitly passed args so they aren't provided twice.
    args_dict = vars(args).copy()
    for key in ("run_name", "base_model_hf", "data_dir", "tasks_list_json_path", "lora_checkpoint_dir"):
        args_dict.pop(key, None)

    # Convert string "true"/"false" to boolean
    for key, value in args_dict.items():
        if isinstance(value, str) and value.lower() in ("true", "false"):
            args_dict[key] = value.lower() == "true"

    main(
        run_name,
        base_model_hf,
        data_dir,
        tasks_list_json_path,
        lora_checkpoint_dir,
        **args_dict,
    )
