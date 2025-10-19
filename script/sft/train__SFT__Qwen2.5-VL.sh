ENV_NAME="sft-qwen25vl"


# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"
conda install -c nvidia cuda-toolkit=12.4 -y


# Set paths
benchmark_dir="/root/Documents/MedVision"
train_sft_dir="${benchmark_dir}/SFT"
data_dir="${benchmark_dir}/Data"


# Data configs
# ----------------------------------------------------------------------------------
# NOTE: At least one of the following 3 task JSON paths must be provided
#       Set multiple task JSON paths for multi-task training
# ----------------------------------------------------------------------------------
tasks_list_json_path_AD="${benchmark_dir}/tasks_list/tasks_MedVision-AD__train_SFT.json" # Total samples: 5545
tasks_list_json_path_detect="${benchmark_dir}/tasks_list/tasks_MedVision-detect__train_SFT.json" # Total samples: 2695205
tasks_list_json_path_TL="${benchmark_dir}/tasks_list/tasks_MedVision-TL__train_SFT.json" # Total samples: 5551
# ----------------------------------------------------------------------------------


# Model configs
base_model_hf="Qwen/Qwen2.5-VL-32B-Instruct"
run_name="MedVision__SFT__qwen25vl-32b__detect-AD-TL"
lora_checkpoint_dir="${train_sft_dir}/${run_name}/checkpoints/${run_name}" # Put ${run_name} at the end for distinct HF repo names when pushing LoRA checkpoints
merged_model_hf="MedVision__SFT-m__qwen25vl-32b__detect-AD-TL"
merged_model_dir="${train_sft_dir}/${run_name}/merged_model"


# Training configs
epoch=10
save_steps=100
eval_steps=100
logging_steps=50
save_total_limit=10 # Maximum number of checkpoints to save
use_flash_attention_2=true
num_workers_concat_datasets=4
num_workers_format_dataset=64
dataloader_num_workers=8
# ----------------------------------------------------------------------------------
# NOTE: If the sample limit is larger than the dataset size, the full dataset will be used.
# ----------------------------------------------------------------------------------
# [Required] Sample limits in total
train_sample_limit=22000
val_sample_limit=200

# # [Option 1] For approximately balanced sampling across 3 tasks
# train_sample_limit_per_task=333333 
# val_sample_limit_per_task=166

# # [Option 2] For task-specific sampling across 3 tasks (these numbers are the maximum samples per task)
train_sample_limit_task_AD=5500
val_sample_limit_task_AD=45
train_sample_limit_task_Detection=11000
val_sample_limit_task_Detection=105
train_sample_limit_task_TL=5500
val_sample_limit_task_TL=50
# ----------------------------------------------------------------------------------
dataloader_pin_memory=true
use_flash_attention_2=true


# Resumed training configs
resume_from_checkpoint=true # Enable resuming from the last checkpoint


# Resource-constrained training configs
gradient_checkpointing=true # Enable gradient checkpointing to save memory
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=12 # Control effective batch size: effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus


# Merge and push configs
push_LoRA=false # Push LoRA checkpoint to HF Hub after each save
push_merged_model=true # Push merged model to HF Hub after training
merge_only=false # [No training] Merge the last checkpoint and push to HF Hub
merge_model=true # [With training] Merge after training and push to HF Hub


# Set wandb configs for logging
wandb_resume="allow" # Wandb resume mode (e.g., 'allow', 'must', 'never')
wandb_dir="${train_sft_dir}/${run_name}"
wandb_project="MedVision-SFT"
wandb_run_name=${run_name}
# NOTE: For continuing an existing run, set the wandb_run_id to the ID of the existing run.
wandb_run_id="multitask-D11k-AD5.5k-TL5.5k" # run ID must be unique in the wandb_project


# Install medvision_bm
rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
pip install "${benchmark_dir}"


# Setup training env
python -m medvision_bm.sft.env_setup --data_dir ${data_dir}
# # [Alternative] Setup training env: use a specific requirements file
# python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt"


# # [Debugging] Disable WANDB online logging
# export WANDB_MODE=offline      # or HF_DISABLE_WANDB=1
# export WANDB_CORE_DEBUG=true
# export WANDB_DEBUG=true


# # Debugging
# export NCCL_P2P_DISABLE=0 # allow GPU↔GPU direct communication (default, desired)
# export NCCL_SHM_DISABLE=0 # allow shared-memory fallback (default)


# ------------------------------------------------------------------------------
# NOTE: Adjust args below
# # [Option 1] For approximately balanced sampling across 3 tasks
# --train_sample_limit_per_task ${train_sample_limit_per_task} \
# --val_sample_limit_per_task ${val_sample_limit_per_task} \
#
# # [Option 2] For custom sampling ratios across 3 tasks <-- current setting
# --train_sample_limit_task_AD ${train_sample_limit_task_AD} \
# --val_sample_limit_task_AD ${val_sample_limit_task_AD} \
# --train_sample_limit_task_Detection ${train_sample_limit_task_Detection} \
# --val_sample_limit_task_Detection ${val_sample_limit_task_Detection} \
# --train_sample_limit_task_TL ${train_sample_limit_task_TL} \
# --val_sample_limit_task_TL ${val_sample_limit_task_TL} \
#
# # Add --prepared_ds_dir if needed
# --prepared_ds_dir ${prepared_ds_dir} \
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# NOTE:
# Dataset processing configs
# ------------------------------------------------------------------------------
# Config 1
# - Set skip_process_dataset=true if the prepared dataset already exists on disk and you want to skip dataset processing. 
# - Set skip_process_dataset=false to process the dataset again (this will overwrite the existing prepared dataset on disk).
skip_process_dataset=false 

# Config 2
# - Set save_processed_img_to_disk=true to save processed images to PNG files on disk during dataset processing for faster subsequent loading (recommended)
# - Set save_processed_img_to_disk=false to not save processed images to disk (default), images will be processed on-the-fly during training
save_processed_img_to_disk=true

# Config 3
# - prepared_ds_dir is the path to the prepared dataset directory to load from disk; comment out to use default path
# - default: os.path.join(data_dir, f"tmp_prepared_ds_AD{train_limit_AD}_D{train_limit_detect}_TL{train_limit_TL}_all{train_limit_total}")
# prepared_ds_dir="" 
# ------------------------------------------------------------------------------

# Offload dataset processing from training to a separate run to avoid timeout issues
python -m  medvision_bm.sft.train__SFT__qwen2_5_vl \
--skip_process_dataset ${skip_process_dataset} \
--process_dataset_only true \
--save_processed_img_to_disk ${save_processed_img_to_disk} \
--run_name ${run_name} \
--base_model_hf ${base_model_hf} \
--lora_checkpoint_dir ${lora_checkpoint_dir} \
--merged_model_hf ${merged_model_hf} \
--merged_model_dir ${merged_model_dir} \
--wandb_resume ${wandb_resume} \
--wandb_dir ${wandb_dir} \
--wandb_project ${wandb_project} \
--wandb_run_name ${wandb_run_name} \
--wandb_run_id ${wandb_run_id} \
--data_dir ${data_dir} \
--tasks_list_json_path_AD ${tasks_list_json_path_AD} \
--tasks_list_json_path_detect ${tasks_list_json_path_detect} \
--tasks_list_json_path_TL ${tasks_list_json_path_TL} \
--epoch ${epoch} \
--save_steps ${save_steps} \
--eval_steps ${eval_steps} \
--logging_steps ${logging_steps} \
--save_total_limit ${save_total_limit} \
--per_device_train_batch_size ${per_device_train_batch_size} \
--per_device_eval_batch_size ${per_device_eval_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--use_flash_attention_2 ${use_flash_attention_2} \
--num_workers_concat_datasets ${num_workers_concat_datasets} \
--num_workers_format_dataset ${num_workers_format_dataset} \
--dataloader_num_workers ${dataloader_num_workers} \
--train_sample_limit ${train_sample_limit} \
--val_sample_limit ${val_sample_limit} \
--train_sample_limit_task_AD ${train_sample_limit_task_AD} \
--val_sample_limit_task_AD ${val_sample_limit_task_AD} \
--train_sample_limit_task_Detection ${train_sample_limit_task_Detection} \
--val_sample_limit_task_Detection ${val_sample_limit_task_Detection} \
--train_sample_limit_task_TL ${train_sample_limit_task_TL} \
--val_sample_limit_task_TL ${val_sample_limit_task_TL} \
--push_LoRA ${push_LoRA} \
--push_merged_model ${push_merged_model} \
--merge_model ${merge_model} \
--merge_only ${merge_only} \
--resume_from_checkpoint ${resume_from_checkpoint} \
--gradient_checkpointing ${gradient_checkpointing} \
--dataloader_pin_memory ${dataloader_pin_memory} \

# Skip dataset processing and directly load from disk for training
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 --main_process_port=29502 --mixed_precision=bf16 \
-m  medvision_bm.sft.train__SFT__qwen2_5_vl \
--skip_process_dataset true \
--process_dataset_only false \
--run_name ${run_name} \
--base_model_hf ${base_model_hf} \
--lora_checkpoint_dir ${lora_checkpoint_dir} \
--merged_model_hf ${merged_model_hf} \
--merged_model_dir ${merged_model_dir} \
--wandb_resume ${wandb_resume} \
--wandb_dir ${wandb_dir} \
--wandb_project ${wandb_project} \
--wandb_run_name ${wandb_run_name} \
--wandb_run_id ${wandb_run_id} \
--data_dir ${data_dir} \
--tasks_list_json_path_AD ${tasks_list_json_path_AD} \
--tasks_list_json_path_detect ${tasks_list_json_path_detect} \
--tasks_list_json_path_TL ${tasks_list_json_path_TL} \
--epoch ${epoch} \
--save_steps ${save_steps} \
--eval_steps ${eval_steps} \
--logging_steps ${logging_steps} \
--save_total_limit ${save_total_limit} \
--per_device_train_batch_size ${per_device_train_batch_size} \
--per_device_eval_batch_size ${per_device_eval_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--use_flash_attention_2 ${use_flash_attention_2} \
--num_workers_concat_datasets ${num_workers_concat_datasets} \
--num_workers_format_dataset ${num_workers_format_dataset} \
--dataloader_num_workers ${dataloader_num_workers} \
--train_sample_limit ${train_sample_limit} \
--val_sample_limit ${val_sample_limit} \
--train_sample_limit_task_AD ${train_sample_limit_task_AD} \
--val_sample_limit_task_AD ${val_sample_limit_task_AD} \
--train_sample_limit_task_Detection ${train_sample_limit_task_Detection} \
--val_sample_limit_task_Detection ${val_sample_limit_task_Detection} \
--train_sample_limit_task_TL ${train_sample_limit_task_TL} \
--val_sample_limit_task_TL ${val_sample_limit_task_TL} \
--push_LoRA ${push_LoRA} \
--push_merged_model ${push_merged_model} \
--merge_model ${merge_model} \
--merge_only ${merge_only} \
--resume_from_checkpoint ${resume_from_checkpoint} \
--gradient_checkpointing ${gradient_checkpointing} \
--dataloader_pin_memory ${dataloader_pin_memory} \

conda deactivate
# conda remove -n $ENV_NAME --all -y
