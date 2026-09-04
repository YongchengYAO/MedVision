ENV_NAME="sft-qwen3vl"

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
model_family_name="qwen3vl" # NOTE: model_family_name must be in AVAILABLE_MODELS from lmms_eval.models (qwen3vl <- vllm_qwen3vl)
base_model_hf="Qwen/Qwen3.5-27B"
run_name="MedVision__SFT__Qwen3.5-27B__D110k-AD5k-TL5k__CoT__512x512"
lora_checkpoint_dir="${train_sft_dir}/${run_name}/checkpoints/${run_name}" # Put a ${run_name} subfolder at the end for distinct HF repo names when pushing LoRA checkpoints
merged_model_hf="MedVision__SFT__Qwen3.5-27B__D110k-AD5k-TL5k__CoT__512x512"
merged_model_dir="${train_sft_dir}/${run_name}/merged_model"

# Dependency versions
# ----------------------------------------------------------------------------------
# NOTE: env_setup.py force-installs transformers==4.54.0 at the end regardless of the
#   --lmms_eval_opt_deps group. Qwen3.5/3.6 report model_type=qwen3_5, which transformers 4.57.0
#   does NOT recognize (model fails to load: "architecture not recognized"). transformers 5.5.0
#   loads them (verified 2026-06-30 via meta-device). We re-pin transformers AFTER env_setup.
transformers_version="5.5.0"
# NOTE: qwen3_5 is a hybrid linear-attention arch; for its fast path install
#   flash-linear-attention + causal-conv1d (optional; otherwise it falls back to torch).
# ----------------------------------------------------------------------------------

# Training configs
epoch=10
save_steps=100
eval_steps=100
logging_steps=20
save_total_limit=10 # LoRA ckpts: adapters + fp32 modules_to_save (embed+lm_head) + optimizer state (~19GB each at 27B)
# FA2 disabled for qwen3_5: the flash-attn 2.7.3 wheel env_setup installs targets the
# transformers-4.5x era and is unvalidated against the qwen3_5 hybrid linear-attention
# arch on transformers 5.5.0. The GPU-smoke-validated attention is SDPA — set via the
# MEDVISION_SFT_ATTN export below (false alone would fall back to eager, which
# materializes O(seq^2) attention matrices and wastes memory).
use_flash_attention_2=false
num_workers_concat_datasets=4
num_workers_format_dataset=64
dataloader_num_workers=4
# ----------------------------------------------------------------------------------
# NOTE: If the sample limit is larger than the dataset size, the full dataset will be used.
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

# [Option 2] For task-specific sampling across 3 tasks
train_sample_limit_task_AD=5500
val_sample_limit_task_AD=45
train_sample_limit_task_Detection=110000
val_sample_limit_task_Detection=105
train_sample_limit_task_TL=5500
val_sample_limit_task_TL=50
# ----------------------------------------------------------------------------------
dataloader_pin_memory=true
use_flash_attention_2=false # duplicate of the setting above — keep both in sync

# Resumed training configs
resume_from_checkpoint=true # Enable resuming from the last checkpoint

# Merge and push configs
# NOTE: merge_models loads the base model in fp32 on CPU (~108GB RAM at 27B — check the
#   container CGROUP limit, not `free`) and saves fp32.
push_LoRA=false        # Push LoRA checkpoint to HF Hub after each save
push_merged_model=true # Push merged model to HF Hub after training
merge_only=false       # [No training] Merge the last checkpoint and push to HF Hub
merge_model=true       # [With training] Merge after training and push to HF Hub

# Resource-constrained training configs
# NOTE: QLoRA keeps the whole NF4 base (~13GB at 27B) resident on every GPU under DDP; with
#   LoRA adapters + fp32 modules_to_save (~152k Qwen vocab) this fits bs=2 per GPU (see the
#   memory NOTE above the launch). Effective batch 128 mirrors the 7B LoRA recipe.
gradient_checkpointing=true
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=16 # effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus (= 2 * 16 * 4 = 128)

# Set wandb configs for logging
wandb_resume="allow" # Wandb resume mode (e.g., 'allow', 'must', 'never')
wandb_dir="${train_sft_dir}/${run_name}"
wandb_project="MedVision-SFT-CoT-Qwen3VL-multiTasks"
wandb_run_name=${run_name}
# NOTE: For continuing an existing run, set the wandb_run_id to the ID of the existing run.
wandb_run_id="Qwen3.5-27B-SFT-D110k-AD5k-TL5k-512x512" # run ID must be unique in the wandb_project

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
python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --lmms_eval_opt_deps qwen3_vl
# Re-pin transformers to a Qwen3-VL-capable version (env_setup forces 4.54.0; see NOTE above)
python -m pip install "transformers==${transformers_version}"
# Fix protobuf: env_setup leaves a protobuf incompatible with wandb>=0.21's generated stubs
# (-> "cannot import name 'Imports' from wandb.proto..." which breaks the trl.SFTTrainer
# import at train time). 6.33.0 matches the validated requirements_sft_*.txt pin.
python -m pip install "protobuf==6.33.0"
# # [Alternative] Setup training env: use a specific requirements file
# python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_eval_qwen3vl.txt" --lmms_eval_opt_deps qwen3_vl

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

# The dataset-prep run below reports where it saved (or found) the prepared dataset. Its
# default name encodes the TRUE train sizes, which are only known after the load+split
# stage, so capture that report and hand the directory to the training launch via
# --prepared_ds_dir: training then loads it directly instead of re-running load+split on
# rank 0 just to recompute the name. Set prepared_ds_dir above to override the prep run's
# output location; the training launch always uses whatever the prep run reports.
mkdir -p "${lora_checkpoint_dir}"
prep_log="${lora_checkpoint_dir}/prepare_dataset.log"

# Offload dataset processing from training to a separate run to avoid timeout issues
python -m medvision_bm.sft.train__SFT-CoT__qwen3vl \
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
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --dataloader_pin_memory ${dataloader_pin_memory} \
    --push_LoRA ${push_LoRA} \
    --push_merged_model ${push_merged_model} \
    --merge_model ${merge_model} \
    --merge_only ${merge_only} \
    --new_shape_hw 512 512 2>&1 | tee "${prep_log}"

# Resolve the prepared-dataset directory from the prep run's report (see prep_log above).
prepared_ds_dir="$(sed -n "s/.*Prepared dataset saved at '\([^']*\)'.*/\1/p" "${prep_log}" | tail -n 1)"
if [ -z "${prepared_ds_dir}" ] || [ ! -d "${prepared_ds_dir}" ]; then
    echo "[Error] Could not resolve the prepared dataset directory from ${prep_log}; aborting before the training launch."
    exit 1
fi
echo "[Info] Training will load the prepared dataset from: ${prepared_ds_dir}"

# Self-heal deps: the dataset-prep step above reinstalls medvision_ds, whose exact pin
# huggingface_hub==0.36.0 drags hub below transformers 5.x's floor (>=1.5.0) on EVERY prep
# run -> ImportError at train start even though dataset prep succeeded. Probe the real
# train-time import chain (trl.SFTTrainer also catches protobuf/wandb drift) and repair
# SURGICALLY. Do NOT `pip install --force-reinstall transformers` here: re-resolving its
# whole dep tree can pull an fsspec newer than datasets' cap, which the NEXT prep run then
# downgrades on disk mid-process (observed crash: ModuleNotFoundError
# fsspec.implementations.chained). The joint install below keeps transformers at its pin,
# lifts hub only as far as that pin requires, re-asserts the protobuf pin, and leaves
# everything else (fsspec, datasets) untouched. Aborts before the expensive launch on failure.
if ! python -c "import transformers; from trl import SFTTrainer; from peft import LoraConfig, PeftModel" >/dev/null 2>&1; then
    echo "[WARN] train-time imports broken after dataset prep (dependency drift) — repairing"
    python -m pip install --upgrade "transformers==${transformers_version}" huggingface_hub "protobuf==6.33.0"
    python -c "import transformers; from trl import SFTTrainer; from peft import LoraConfig, PeftModel"
fi

# Ensure CUDA_HOME is set (required by DeepSpeed compatibility check at import time)
# even when DeepSpeed is not used as the training backend.
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Attention implementation for training: SDPA — the GPU-smoke-validated config for qwen3_5.
# (use_flash_attention_2=false by itself falls back to EAGER in prepare_trainer;
# this env knob, read by sft_utils.py, overrides it.)
export MEDVISION_SFT_ATTN=sdpa

# Skip dataset processing and directly load from disk for training
# NOTE: QLoRA + plain DDP (no FSDP — prepare_trainer places the full model per rank via
#   device_map). Per-GPU budget at 27B: NF4 base ~13GB + fp32 frozen originals ~6GB + fp32
#   modules_to_save weights/grads ~12GB + fused-AdamW state ~12GB + LoRA adapters ~2GB +
#   activations (bs=2, 512^2, grad ckpt) => ~49-57GB — inside the ~62GB effective ceiling on
#   4×80GB pods (~18GB/GPU is held by sibling CUDA contexts / NCCL / VMM). If OOM, drop the
#   per-device batch to 1 and double gradient_accumulation_steps.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
    --num_processes=4 \
    --main_process_port=29508 \
    --mixed_precision=bf16 \
    -m medvision_bm.sft.train__SFT-CoT__qwen3vl \
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
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --dataloader_pin_memory ${dataloader_pin_memory} \
    --push_LoRA ${push_LoRA} \
    --push_merged_model ${push_merged_model} \
    --merge_model ${merge_model} \
    --merge_only ${merge_only} \
    --enable_temperature_sampler ${enable_temperature_sampler} \
    --temperature_sampler_T ${temperature_sampler_T} \
    --new_shape_hw 512 512

conda deactivate
# conda remove -n $ENV_NAME --all -y
