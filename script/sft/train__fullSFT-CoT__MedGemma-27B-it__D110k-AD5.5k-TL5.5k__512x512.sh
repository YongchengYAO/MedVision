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
run_name="MedVision__fullSFT__MedGemma-27B-it__D110k-AD5k-TL5k__CoT__512x512"
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
wandb_run_id="MedGemma-27B-fullSFT-D110k-AD5k-TL5k-512x512" # run ID must be unique in the wandb_project

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
# Liger kernels for the fused-linear-cross-entropy memory fix (MEDVISION_SFT_USE_LIGER=1
# below; Gemma3 support needs >=0.5.4). --no-deps: liger only needs torch+triton (already
# pinned in this env) and must not drag the resolved stack.
python -m pip install --no-deps "liger-kernel>=0.5.4"
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

# Offload dataset processing from training to a separate run to avoid timeout issues
python -m medvision_bm.sft.train__fullFT-CoT__medgemma \
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
# Optimizer for 4-GPU GPU-resident full FT: NON-paged 8-bit AdamW. paged_adamw_8bit was
# measured (2026-07-03 15:42 run, MEMPROBE after_step_1) leaving its ~13.5GB/rank state as
# CUDA unified-memory pages RESIDENT ON DEVICE but OUTSIDE the torch allocator
# (device_used - allocated = 13-27GB) — torch's cudaMalloc cannot evict UVM pages, so the
# next micro-batch's loss buffers OOM'd. adamw_bnb_8bit keeps the same 13.5GB/rank state
# inside the torch pool where the allocator manages it. (fsdp_offload_params remains
# infeasible: needs ~600GB host RAM, pod cgroup allows 400GB.) CAVEAT: bnb's quantized
# optimizer state cannot be gathered by FSDP FULL_STATE_DICT, so SAVE_ONLY_MODEL is
# required — checkpoints store weights only, and resume_from_checkpoint continues from the
# saved weights with a FRESH optimizer/LR state.
export MEDVISION_SFT_OPTIM=adamw_bnb_8bit
export MEDVISION_SFT_SAVE_ONLY_MODEL=1
# Liger kernels (fused linear cross-entropy): removes Gemma3's 262k-vocab logits spike at
# the loss (seq x 262k in bf16 + fp32 + grads, linear in seq len). Verified applied on the
# 2026-07-03 run — but NOT sufficient alone: that run still OOM'd during the backward
# layer-gather (see MEDVISION_SFT_BF16_GRADS below). Kept: it caps the loss-spike peak so
# long-CoT batches can't reintroduce it (the collator has no seq-len cap).
export MEDVISION_SFT_USE_LIGER=1
# Disable no_sync during gradient accumulation — THE root cause of the 4x identical step-0
# OOMs at ~75.6GiB (2026-07-03; proven by the CUDA allocator snapshot from the instrumented
# run): under no_sync, FSDP accumulates FULL UNSHARDED grads — one 788.00MiB bf16 flat-grad
# per Gemma3DecoderLayer (413M params x 2B), ~48GB/rank — and OOM'd ~54 layers into the
# first backward. With sync-each-micro-batch, grads reduce-scatter immediately and
# accumulate SHARDED (~13.5GB fp32 / ~6.75GB bf16 per rank).
export MEDVISION_SFT_SYNC_EACH_BATCH=1
# PURE BF16 (user decision 2026-07-03): no accelerate mixed precision, no fp32 master
# weights. The fp32-master recipe has a fixed cost of 67.5GB/GPU at 27B/4 ranks (27 fp32
# masters + 13.5 FSDP bf16 _mp_shard + 13.5 fp32 grad shards + 13.5 8-bit optim) — it can
# NEVER fit 80GB, and its bf16-grads escape hatch (MEDVISION_SFT_BF16_GRADS) hard-fails
# (torch .grad setter: bf16 grad vs fp32 param dtype mismatch, run of 2026-07-03 15:58).
# Pure bf16 costs ~40.5GB fixed (13.5 params + 13.5 grads + 13.5 optim) => peak ~52-55GB,
# ~20GB margin. Requires BOTH this export (SFTConfig bf16=False) AND no --mixed_precision
# flag on the accelerate launch below. Precedent: torchtune trains full-bf16 by default;
# arXiv 2408.15793 reports near-identical loss to mixed precision for full-param SFT.
export MEDVISION_SFT_PURE_BF16=1
# LR 2e-5 -> 4e-5: with bf16 weights (no fp32 master), AdamW updates below bf16's ~0.4%
# relative resolution round away ("stale weights"); a moderately higher LR keeps updates
# above the rounding floor. NOTE: this mitigation is engineering judgment, not a cited
# recipe (the literature's validated fixes are fp32 masters / stochastic rounding) — watch
# the early wandb loss curve and revert to 2e-5 if it misbehaves.
export MEDVISION_SFT_LR=4e-5
# Print per-rank memory after FSDP wrap and after step 1 (2 lines, no overhead) so the
# actual peak margin is visible in the log. Also prints which forward is live (liger or
# stock), whether gradient checkpointing engaged, and the FSDP mixed-precision policy.
export MEDVISION_SFT_MEMPROBE=1
# On OOM, dump a per-rank CUDA allocator snapshot (every allocation + stacks) into the
# checkpoint dir for offline analysis. The 3 identical step-0 OOMs on 2026-07-03 were
# insensitive to config levers — this pinpoints the true consumers. Remove once training
# is past step 1 reliably (small steady-state recording overhead).
export MEDVISION_SFT_MEMSNAPSHOT=1

# Skip dataset processing and directly load from disk for training
# NOTE: FSDP (FULL_SHARD) is required for full FT of 27B, in PURE BF16 on this 4x80GB pod
#   (see MEDVISION_SFT_PURE_BF16 above for the memory math; --mixed_precision is
#   deliberately NOT passed — the env var it sets would re-enable the fp32-master upcast).
#   Expected per-GPU budget: 13.5 bf16 param shards + 13.5 bf16 grad shards (sync-each-
#   batch keeps them SHARDED during accumulation) + 13.5 8-bit optim state + ~10-12
#   activations/loss => ~52-55GB peak vs 79.19 usable. Verify in the log: the accelerate
#   "Upcasted low precision parameters" warning must be GONE, MEMPROBE post-wrap allocated
#   ~14-16GB (38.3 would mean the upcast is back), and the after_step_1 optimizer probe
#   must show uint8 state (accelerate#1902 rebuilt bnb 8-bit as fp32 in the past). Do NOT
#   enable fsdp_offload_params on this pod: needs ~600GB host RAM (cgroup allows 400GB).
#   To restore the fp32-master recipe + fully resumable checkpoints, move to >=8 GPUs:
#   re-add --mixed_precision=bf16 and unset MEDVISION_SFT_PURE_BF16 / MEDVISION_SFT_LR /
#   MEDVISION_SFT_OPTIM / MEDVISION_SFT_SAVE_ONLY_MODEL.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
    --num_processes=4 \
    --main_process_port=29504 \
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
