ENV_NAME="sft-qwen25vl"

# TODO: debug
# temp fix
conda config --set solver classic
# or try this
# conda install conda=26.1.1

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

# Sanitize HF_TOKEN: pod-injected secrets can carry a trailing newline that corrupts the
# HTTP Authorization header (-> 401 on gated models/datasets). No-op if unset or clean.
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '[:space:]')"

# Use MedVision dataset v1.0.0
export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.1.1'

# Set paths
benchmark_dir="/root/Documents/MedVision"
train_sft_dir="${benchmark_dir}/SFT"
data_dir="${benchmark_dir}/Data"

# Data configs
# ----------------------------------------------------------------------------------
# NOTE: At least one of the following 3 task JSON paths must be provided
#       Set multiple task JSON paths for multi-task training
# ----------------------------------------------------------------------------------
tasks_list_json_path_AD="${benchmark_dir}/tasks_list/tasks_MedVision-AD__train_SFT.json"         # Total samples: 5545
tasks_list_json_path_detect="${benchmark_dir}/tasks_list/tasks_MedVision-detect__train_SFT.json" # Total samples: 2695205
tasks_list_json_path_TL="${benchmark_dir}/tasks_list/tasks_MedVision-TL__train_SFT.json"         # Total samples: 5551
# ----------------------------------------------------------------------------------

# Model configs
model_family_name="qwen25vl" # NOTE: model_family_name must be in AVAILABLE_MODELS from lmms_eval.models (the "vllm_" prefix may be omitted)
base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct"
run_name="MedVision__SFT__Qwen2.5VL-7B__D110k-AD5k-TL5k__CoT"
lora_checkpoint_dir="${train_sft_dir}/${run_name}/checkpoints/${run_name}" # Put a ${run_name} subfolder at the end for distinct HF repo names when pushing LoRA checkpoints
merged_model_hf="MedVision__SFT__Qwen2.5VL-7B__D110k-AD5k-TL5k__CoT"
merged_model_dir="${train_sft_dir}/${run_name}/merged_model"

# Training configs
epoch=10
save_steps=100
eval_steps=100
logging_steps=50
save_total_limit=10 # Maximum number of checkpoints to save
use_flash_attention_2=true
num_workers_concat_datasets=4
num_workers_format_dataset=32
dataloader_num_workers=8
# ----------------------------------------------------------------------------------
# NOTE: Allow sampling with replacement if limit exceeds dataset size
# NOTE: Any limit below may be left unset (commented out); -1 is then passed:
#       - unset train limits => full dataset
#       - unset val_sample_limit (total) => keep all per-task validation samples
#       - unset per-task val limits => fallback of 100 validation samples per task
#       Do NOT set a limit to 0 (rejected as ambiguous); to skip a task, comment out
#       its tasks_list_json_path_* above instead.
# ----------------------------------------------------------------------------------
# [Optional] Sample limits in total
train_sample_limit=121000
val_sample_limit=200

# # [Option 1] For approximately balanced sampling across 3 tasks
# train_sample_limit_per_task=333333
# val_sample_limit_per_task=166

# # [Option 2] For task-specific sampling across 3 tasks
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
gradient_checkpointing=true # Enable gradient checkpointing to save memory
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=8 # Control effective batch size: effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

# Merge and push configs
push_LoRA=false        # Push LoRA checkpoint to HF Hub after each save
push_merged_model=true # Push merged model to HF Hub after training
merge_only=false       # [No training] Merge the last checkpoint and push to HF Hub
merge_model=true       # [With training] Merge after training and push to HF Hub

# Set wandb configs for logging
wandb_resume="allow" # Wandb resume mode (e.g., 'allow', 'must', 'never')
wandb_dir="${train_sft_dir}/${run_name}"
wandb_project="MedVision-SFT-CoT-Qwen2.5VL-multiTasks"
wandb_run_name=${run_name}
# NOTE: For continuing an existing run, set the wandb_run_id to the ID of the existing run.
wandb_run_id="Qwen25VL7B-D110k-AD5k-TL5k" # run ID must be unique in the wandb_project

# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
# tree). setuptools build_py caches created dirs in a process-global memo, and on
# CephFS a build subdir can transiently vanish (async delete/recreate lag or an
# unguarded concurrent writer), after which the cache refuses to recreate it and a
# later file copy dies with: could not create '...': No such file or directory.
# A private local build dir is immune; only the shared-env install needs the lock.
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"

# Setup training env
python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --lmms_eval_opt_deps qwen2_5_vl
# Fix protobuf: env_setup leaves a protobuf incompatible with wandb>=0.21's generated stubs
# (-> "cannot import name 'Imports' from wandb.proto..." which breaks the trl.SFTTrainer
# import at train time). 6.33.0 matches the validated requirements_sft_*.txt pin.
python -m pip install "protobuf==6.33.0"
# # [Alternative] Setup training env: use a specific requirements file
# python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt" --lmms_eval_opt_deps qwen2_5_vl

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

# Config 4
# NOTE: --enable_temperature_sampler and related CLI args are only effective for training. They do not affect dataset processing when --process_dataset_only is true
# - Add CLI arg "enable_temperature_sampler=true" to enable temperature-based sampling during multi-task training with imbalanced datasets
# - If you use it, you can set "temperature_sampler_T" below: default=3, T=1 means proportional to counts; larger T flattens task probabilities.
enable_temperature_sampler=true
temperature_sampler_T=5

# Config 5
# - Add CLI arg "--new_shape_hw <height> <width>" to resize all images to a new shape during dataset preparation
#   Example: --new_shape_hw 1080 1920.
# ------------------------------------------------------------------------------

# The dataset-prep run below reports where it saved (or found) the prepared dataset. Its
# default name encodes the TRUE train sizes, which are only known after the load+split
# stage, so capture that report and hand the directory to the training launch via
# --prepared_ds_dir: training then loads it directly instead of re-running load+split on
# rank 0 just to recompute the name. Set prepared_ds_dir above to override the prep run's
# output location; the training launch always uses whatever the prep run reports.
mkdir -p "${lora_checkpoint_dir}"
prep_log="${lora_checkpoint_dir}/prepare_dataset.log"

# Offload dataset processing from training to a separate run to avoid timeout issues
python -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --skip_process_dataset ${skip_process_dataset} \
    --process_dataset_only true \
    ${prepared_ds_dir:+--prepared_ds_dir ${prepared_ds_dir}} \
    --save_processed_img_to_disk ${save_processed_img_to_disk} \
    --run_name ${run_name} \
    --model_family_name ${model_family_name} \
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
    --train_sample_limit ${train_sample_limit:--1} \
    --val_sample_limit ${val_sample_limit:--1} \
    --train_sample_limit_task_AD ${train_sample_limit_task_AD:--1} \
    --val_sample_limit_task_AD ${val_sample_limit_task_AD:--1} \
    --train_sample_limit_task_Detection ${train_sample_limit_task_Detection:--1} \
    --val_sample_limit_task_Detection ${val_sample_limit_task_Detection:--1} \
    --train_sample_limit_task_TL ${train_sample_limit_task_TL:--1} \
    --val_sample_limit_task_TL ${val_sample_limit_task_TL:--1} \
    --push_LoRA ${push_LoRA} \
    --push_merged_model ${push_merged_model} \
    --merge_model ${merge_model} \
    --merge_only ${merge_only} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --dataloader_pin_memory ${dataloader_pin_memory} 2>&1 | tee "${prep_log}"

# Resolve the prepared-dataset directory from the prep run's report (see prep_log above).
prepared_ds_dir="$(sed -n "s/.*Prepared dataset saved at '\([^']*\)'.*/\1/p" "${prep_log}" | tail -n 1)"
if [ -z "${prepared_ds_dir}" ] || [ ! -d "${prepared_ds_dir}" ]; then
    echo "[Error] Could not resolve the prepared dataset directory from ${prep_log}; aborting before the training launch."
    exit 1
fi
echo "[Info] Training will load the prepared dataset from: ${prepared_ds_dir}"

# Skip dataset processing and directly load from disk for training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch --num_processes=4 --main_process_port=29502 --mixed_precision=bf16 \
    -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --skip_process_dataset true \
    --prepared_ds_dir ${prepared_ds_dir} \
    --process_dataset_only false \
    --run_name ${run_name} \
    --model_family_name ${model_family_name} \
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
    --train_sample_limit ${train_sample_limit:--1} \
    --val_sample_limit ${val_sample_limit:--1} \
    --train_sample_limit_task_AD ${train_sample_limit_task_AD:--1} \
    --val_sample_limit_task_AD ${val_sample_limit_task_AD:--1} \
    --train_sample_limit_task_Detection ${train_sample_limit_task_Detection:--1} \
    --val_sample_limit_task_Detection ${val_sample_limit_task_Detection:--1} \
    --train_sample_limit_task_TL ${train_sample_limit_task_TL:--1} \
    --val_sample_limit_task_TL ${val_sample_limit_task_TL:--1} \
    --push_LoRA ${push_LoRA} \
    --push_merged_model ${push_merged_model} \
    --merge_model ${merge_model} \
    --merge_only ${merge_only} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --dataloader_pin_memory ${dataloader_pin_memory} \
    --temperature_sampler_T ${temperature_sampler_T} \
    --enable_temperature_sampler ${enable_temperature_sampler}

conda deactivate
# conda remove -n $ENV_NAME --all -y
