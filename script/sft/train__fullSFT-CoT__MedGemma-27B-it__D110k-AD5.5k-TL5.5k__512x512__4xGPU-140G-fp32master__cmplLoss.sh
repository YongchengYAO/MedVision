ENV_NAME="sft-medgemma"

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
# HTTP Authorization header (-> 401 on gated models like google/medgemma-27b-it). No-op if
# unset or clean.
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '[:space:]')"

# Use MedVision dataset v1.0.0
export MedVision_PLANNER_VERSION='1.0.0'
export MEDVISION_SFT_COMPLETION_ONLY=1  # opt-in: Gemma completion-only assistant-turn loss masking
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
model_family_name="medgemma" # NOTE: model_family_name must be in AVAILABLE_MODELS from lmms_eval.models
base_model_hf="google/medgemma-27b-it"
run_name="MedVision__fullSFT__MedGemma-27B-it__D110k-AD5k-TL5k__CoT__512x512__4xGPU-140G-fp32master__cmplLoss"
# NOTE: --lora_checkpoint_dir is remapped to checkpoint_dir internally for full finetuning
lora_checkpoint_dir="${train_sft_dir}/${run_name}/checkpoints/${run_name}"

# Dependency versions
# ----------------------------------------------------------------------------------
# NOTE: MedGemma-27B is Gemma 3 architecture, which is supported by transformers 4.54.0 —
#   the version env_setup installs by default. The re-pin below is therefore a no-op kept
#   for parallelism with the other base-model scripts; bump it if you need a newer release.
#   Source of truth: requirements/requirements_sft_medgemma.txt (transformers==4.54.0).
transformers_version="4.54.0"
# FSDP transformer layer class to wrap (passed to accelerate below). This MUST match the
# decoder layer class name in the installed transformers for this checkpoint.
#   - MedGemma-27B (Gemma 3 text backbone): Gemma3DecoderLayer
# Verify once with:
#   python -c "from transformers import AutoModelForImageTextToText as M; m=M.from_pretrained('${base_model_hf}'); print(sorted({type(x).__name__ for x in m.modules() if 'DecoderLayer' in type(x).__name__}))"
fsdp_layer_cls="Gemma3DecoderLayer"
# ----------------------------------------------------------------------------------

# Training configs
epoch=3
save_steps=100
eval_steps=100
logging_steps=20
save_total_limit=3 # Resumable full-FT ckpts are huge at 27B (~54GB bf16 weights + ~108GB optimizer state ≈ 160GB each); keep few
use_flash_attention_2=true
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
use_flash_attention_2=true

# Resumed training configs
resume_from_checkpoint=true # Enable resuming from the last checkpoint

# Resource-constrained training configs
# NOTE: Full FT of a 27B model requires much more VRAM than the 7B reference — use the
#   smallest per-device batch + large grad accumulation, and FSDP FULL_SHARD across 4 GPUs.
gradient_checkpointing=true # Required for full FT at 27B scale
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=64 # effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus (= 1 * 64 * 4 = 256)

# Set wandb configs for logging
wandb_resume="allow" # Wandb resume mode (e.g., 'allow', 'must', 'never')
wandb_dir="${train_sft_dir}/${run_name}"
wandb_project="MedVision-SFT-CoT-MedGemma-multiTasks"
wandb_run_name=${run_name}
# NOTE: For continuing an existing run, set the wandb_run_id to the ID of the existing run.
wandb_run_id="MedGemma-27B-fullSFT-D110k-AD5k-TL5k-512x512-4xGPU-140G-fp32master-cmplLoss" # run ID must be unique in the wandb_project

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

# Setup training env (default stack: transformers 4.54.0 supports Gemma 3 / MedGemma)
python -m medvision_bm.sft.env_setup --data_dir ${data_dir}
# Re-pin transformers (no-op at 4.54.0; see NOTE above). Bump if a newer release is needed.
python -m pip install "transformers==${transformers_version}"
# Fix protobuf: env_setup leaves a protobuf incompatible with wandb>=0.21's generated stubs
# (-> "cannot import name 'Imports' from wandb.proto..." which breaks the trl.SFTTrainer
# import at train time). 6.33.0 matches the validated requirements_sft_*.txt pin.
python -m pip install "protobuf==6.33.0"
# # [Alternative] Setup training env: use a specific requirements file
# python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_medgemma.txt"

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
python -m medvision_bm.sft.train__fullFT-CoT__medgemma \
    --skip_process_dataset ${skip_process_dataset} \
    --process_dataset_only true \
    ${prepared_ds_dir:+--prepared_ds_dir ${prepared_ds_dir}} \
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
    --new_shape_hw 512 512 2>&1 | tee "${prep_log}"

# Resolve the prepared-dataset directory from the prep run's report (see prep_log above).
prepared_ds_dir="$(sed -n "s/.*Prepared dataset saved at '\([^']*\)'.*/\1/p" "${prep_log}" | tail -n 1)"
if [ -z "${prepared_ds_dir}" ] || [ ! -d "${prepared_ds_dir}" ]; then
    echo "[Error] Could not resolve the prepared dataset directory from ${prep_log}; aborting before the training launch."
    exit 1
fi
echo "[Info] Training will load the prepared dataset from: ${prepared_ds_dir}"

# Ensure CUDA_HOME is set (required by DeepSpeed compatibility check at import time)
# even when DeepSpeed is not used as the training backend.
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NON-anti-OOM (fp32-master) variant: default fused fp32 AdamW (adamw_torch_fused) with
# fully RESUMABLE checkpoints — deliberately NO MEDVISION_SFT_OPTIM / SAVE_ONLY_MODEL /
# PURE_BF16 / LR / USE_LIGER overrides (the 4-GPU anti-OOM recipe lives in the sibling
# script without the __4xGPU-140G-fp32master suffix; without liger the 262k-vocab logits spike
# at the loss returns, but the 140GB fp32-master budget absorbs it). CKPT CAVEAT: each
# resumable checkpoint is ~160GB at 27B (bf16 weights + optimizer state) and the
# FULL_STATE_DICT save all-gathers the fp32 optimizer state to rank0 host RAM — never
# smoke-validated at 27B+ scale (the smokes used weights-only saves); run a
# 1-save+1-resume probe (save_steps=1) before a multi-day run.
# Disable no_sync during gradient accumulation — KEPT even in this non-anti-OOM variant:
# under no_sync, FSDP accumulates FULL UNSHARDED grads (~48-54GB bf16 at 27B) that do NOT
# shrink with world size, OOMing the FIRST backward even on 8xH200 (proven root cause of
# the 2026-07-03 step-0 OOMs on the 4-GPU twin). Sync-each-micro-batch reduce-scatters
# every micro-batch so grads accumulate fp32 SHARDED — numerics-neutral, negligible cost
# on NVLink.
export MEDVISION_SFT_SYNC_EACH_BATCH=1
# Print per-rank memory after FSDP wrap and after step 1 (2 lines, no overhead) so the
# actual peak margin is visible in the log.
export MEDVISION_SFT_MEMPROBE=1

# Skip dataset processing and directly load from disk for training
# NOTE: 4-GPU 140GB-class fp32-master recipe (--mixed_precision=bf16 => fp32 master
#   weights + bf16 compute under FSDP). Per-GPU fixed memory at 27B/4 ranks: ~27 fp32
#   masters + ~13.5 bf16 _mp_shard + ~27 fp32 grad shards + ~54 fused-AdamW fp32 states
#   ≈ 121.5GB worst case, plus activations (incl. the un-liger'd 262k-vocab loss spike)
#   — does NOT fit 80GB cards. The 27B/4-rank budget is VALIDATED on 4x 140GB GPUs
#   (Qwen3.6-27B twin, 2026-07-09: post-wrap 38.2GiB/rank), but Gemma's 262k-vocab loss
#   spike is larger than Qwen's 152k — UNVALIDATED for this model; watch the after_step_1
#   MEMPROBE on the first run. Verify: post-wrap allocated ~38GB (a full-model figure
#   means sharding didn't engage).
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
    --num_processes=4 \
    --main_process_port=29514 \
    --mixed_precision=bf16 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap ${fsdp_layer_cls} \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params false \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_sync_module_states true \
    -m medvision_bm.sft.train__fullFT-CoT__medgemma \
    --skip_process_dataset true \
    --prepared_ds_dir ${prepared_ds_dir} \
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
    --enable_temperature_sampler ${enable_temperature_sampler} \
    --temperature_sampler_T ${temperature_sampler_T} \
    --new_shape_hw 512 512

conda deactivate
# conda remove -n $ENV_NAME --all -y
