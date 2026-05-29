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
model_family_name="qwen25vl" # NOTE: model_family_name must be in AVAILABLE_MODELS from lmms_eval.models
base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct"
run_name="MedVision__fullSFT__Qwen2.5VL-7B__D110k-AD5k-TL5k__CoT__512x512"
# NOTE: --lora_checkpoint_dir is remapped to checkpoint_dir internally for full finetuning
lora_checkpoint_dir="${train_sft_dir}/${run_name}/checkpoints/${run_name}"


# Training configs
epoch=3
save_steps=100
eval_steps=100
logging_steps=20
save_total_limit=10  # Full FT checkpoints are large; keep fewer
use_flash_attention_2=true
num_workers_concat_datasets=4
num_workers_format_dataset=64
dataloader_num_workers=4
# ----------------------------------------------------------------------------------
# NOTE: If the sample limit is larger than the dataset size, the full dataset will be used.
# ----------------------------------------------------------------------------------
# [Required] Sample limits in total
train_sample_limit=121000
val_sample_limit=200

# [Option 2] For task-specific sampling across 3 tasks
train_sample_limit_task_AD=5500
val_sample_limit_task_AD=45
train_sample_limit_task_Detection=110000
val_sample_limit_task_Detection=105
train_sample_limit_task_TL=5500
val_sample_limit_task_TL=50
# ----------------------------------------------------------------------------------
dataloader_pin_memory=true
use_flash_attention_2=true


# Resumed training configs
resume_from_checkpoint=true # Enable resuming from the last checkpoint


# Resource-constrained training configs
# NOTE: Full FT requires much more VRAM than LoRA — use small batch + large grad accumulation
gradient_checkpointing=true  # Required for full FT at 7B scale
per_device_train_batch_size=8
per_device_eval_batch_size=8
gradient_accumulation_steps=8  # effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus


# Set wandb configs for logging
wandb_resume="allow" # Wandb resume mode (e.g., 'allow', 'must', 'never')
wandb_dir="${train_sft_dir}/${run_name}"
wandb_project="MedVision-SFT-CoT-Qwen2.5VL-multiTasks"
wandb_run_name=${run_name}
# NOTE: For continuing an existing run, set the wandb_run_id to the ID of the existing run.
wandb_run_id="Qwen25VL7B-fullSFT-D110k-AD5k-TL5k-512x512" # run ID must be unique in the wandb_project


# Install medvision_bm (locked shared build)
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
flock "${lockfile}" bash -c '
    set -euo pipefail
    benchmark_dir="'"${benchmark_dir}"'"
    wheelhouse="'"${wheelhouse}"'"
    rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
    python -m pip wheel "${benchmark_dir}" -w "${wheelhouse}" --no-deps
    latest_wheel="$(ls -t "${wheelhouse}"/medvision_bm-*.whl | head -n1)"
    python -m pip install --force-reinstall "${latest_wheel}"
'


# Setup training env
python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --lmms_eval_opt_deps qwen2_5_vl


# # [Debugging] Disable WANDB online logging
# export WANDB_MODE=offline
# export WANDB_CORE_DEBUG=true
# export WANDB_DEBUG=true


# ------------------------------------------------------------------------------
# Dataset processing configs
# ------------------------------------------------------------------------------
# Config 1: skip dataset processing if prepared dataset already exists on disk
skip_process_dataset=false

# Config 2: save processed images to disk for faster subsequent loading
save_processed_img_to_disk=true

# Config 3: prepared_ds_dir — comment out to use default path
# prepared_ds_dir=""

# Config 4: temperature-based sampling for multi-task training
# NOTE: only effective for training, not dataset processing
enable_temperature_sampler=true
temperature_sampler_T=5
# ------------------------------------------------------------------------------

# Offload dataset processing from training to a separate run to avoid timeout issues
python -m medvision_bm.sft.train__fullFT-CoT__qwen2_5_vl \
--skip_process_dataset ${skip_process_dataset} \
--process_dataset_only true \
--save_processed_img_to_disk ${save_processed_img_to_disk} \
--run_name ${run_name} \
--model_family_name ${model_family_name} \
--base_model_hf ${base_model_hf} \
--lora_checkpoint_dir ${lora_checkpoint_dir} \
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
--resume_from_checkpoint ${resume_from_checkpoint} \
--gradient_checkpointing ${gradient_checkpointing} \
--dataloader_pin_memory ${dataloader_pin_memory} \
--new_shape_hw 512 512

# Ensure CUDA_HOME is set (required by DeepSpeed compatibility check at import time)
# even when DeepSpeed is not used as the training backend.
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Skip dataset processing and directly load from disk for training
# NOTE: FSDP (FULL_SHARD) is required for full FT of 7B on 80GB GPUs — standard DDP exceeds
#   capacity: weights(14GB) + gradients(14GB) + AdamW-FP32(56GB) = ~84GB before activations.
#   FSDP shards all three components across 4 GPUs, reducing per-GPU usage to ~31GB.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --num_processes=4 \
  --main_process_port=29502 \
  --mixed_precision=bf16 \
  --use_fsdp \
  --fsdp_sharding_strategy FULL_SHARD \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLDecoderLayer \
  --fsdp_state_dict_type FULL_STATE_DICT \
  --fsdp_offload_params false \
  --fsdp_cpu_ram_efficient_loading true \
  --fsdp_sync_module_states true \
  -m medvision_bm.sft.train__fullFT-CoT__qwen2_5_vl \
--skip_process_dataset true \
--process_dataset_only false \
--run_name ${run_name} \
--model_family_name ${model_family_name} \
--base_model_hf ${base_model_hf} \
--lora_checkpoint_dir ${lora_checkpoint_dir} \
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
--resume_from_checkpoint ${resume_from_checkpoint} \
--gradient_checkpointing ${gradient_checkpointing} \
--dataloader_pin_memory ${dataloader_pin_memory} \
--enable_temperature_sampler ${enable_temperature_sampler} \
--temperature_sampler_T ${temperature_sampler_T} \
--new_shape_hw 512 512

conda deactivate
# conda remove -n $ENV_NAME --all -y
