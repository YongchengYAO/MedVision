#!/usr/bin/env bash
#
# sft_launcher_template.sh — parameterised MedVision SFT launcher (two phases).
#
# PURPOSE
#   A ready-to-edit replacement for the repository's own `script/sft/train__*.sh`
#   recipes. It reproduces their structure exactly: one variable block, then
#   PHASE A (`--process_dataset_only true`, CPU, single process) and PHASE B
#   (`accelerate launch ... --skip_process_dataset true`, GPU, N processes).
#   Like the recipes, it tees phase A into a log, reads the prepared-dataset
#   directory phase A reports back from that log, and hands it to phase B as
#   --prepared_ds_dir so training never re-runs the load+split stage.
#   Every variable keeps the repository's own name so the values can be copied
#   between this template and a real recipe without translation.
#
# PREREQUISITES
#   * `medvision_bm` installed with the SFT stack for the chosen model family
#     (torch, accelerate, trl, peft, transformers, flash-attn as applicable).
#   * `medvision_ds` installed under "${data_dir}/src" (see the environment-setup
#     sub-skill, or set RUN_ENV_SETUP=1 below).
#   * The three SFT-namespace task lists (tasks_MedVision-{AD,detect,TL}__train_SFT.json).
#   * PHASE B requires GPUs. PHASE A does not, but downloads/reads the dataset.
#
# EXAMPLE
#   # 1. Preview everything, no GPU, no data, no installs:
#   DRY_RUN=1 bash sft_launcher_template.sh \
#       --benchmark-dir /work/MedVision --tasks-dir /work/MedVision/tasks_list \
#       --family qwen25vl --base-model Qwen/Qwen2.5-VL-7B-Instruct \
#       --run-name MedVision__SFT__Qwen2.5VL-7B__D110k-AD5k-TL5k__CoT__512x512
#
#   # 2. Only build the prepared dataset (CPU, hours):
#   bash sft_launcher_template.sh --phase A --benchmark-dir ... --family qwen25vl ...
#
#   # 3. Only train from an already-prepared dataset (GPU); pass the directory
#   #    phase A reported so rank 0 does not re-run load+split to derive the name:
#   bash sft_launcher_template.sh --phase B --gpus 0,1,2,3 --prepared-ds-dir <dir> --benchmark-dir ... ...
#
#   # 4. Full-parameter FSDP run on a Gemma-lineage model:
#   bash sft_launcher_template.sh --mode fullft --family gemma4 \
#       --base-model google/gemma-4-31B-it --fsdp-layer-cls Gemma4TextDecoderLayer ...
#
# SAFETY
#   * Runs NOTHING by default beyond the two documented commands; `DRY_RUN=1`
#     prints them instead of executing.
#   * Never creates a conda environment and never calls `sft.env_setup` unless
#     RUN_ENV_SETUP=1 is exported (installs mutate the active environment).
#   * `-h` / `--help` prints usage and exits 0 without touching anything.
#
set -euo pipefail

# =============================================================================
# 1. VARIABLE BLOCK — edit here, or override with the flags parsed in section 2.
# =============================================================================

# -- Paths ---------------------------------------------------------------------
# benchmark_dir : working root that holds the task lists and receives SFT output.
# data_dir      : dataset cache; medvision_ds is installed under "${data_dir}/src",
#                 the prepared dataset defaults to "${data_dir}/SFT-CoT_datasets/...".
# train_sft_dir : where checkpoints, merged models and wandb logs live.
# Values left EMPTY here are derived from benchmark_dir/run_name in section 2b,
# after the command-line options have been applied.
benchmark_dir="${benchmark_dir:-<benchmark_dir>}"
data_dir="${data_dir:-}"          # default: ${benchmark_dir}/Data
train_sft_dir="${train_sft_dir:-}" # default: ${benchmark_dir}/SFT
tasks_dir="${tasks_dir:-}"         # default: ${benchmark_dir}/tasks_list

# -- Task lists ----------------------------------------------------------------
# At least ONE must be set; the trainer raises AssertionError otherwise. Set a
# path to "" to DROP that task entirely (never use a sample limit of 0).
# These are the SFT-namespace lists (BoxSize / TumorLesionSize / BiometricsFromLandmarks);
# eval-side "_BoxCoordinate_" names in a detection list are auto-renamed by the loader.
# Unset (not empty) => derived from tasks_dir in section 2b. Set to "" to DROP a task.
tasks_list_json_path_AD="${tasks_list_json_path_AD-__DERIVE__}"
tasks_list_json_path_detect="${tasks_list_json_path_detect-__DERIVE__}"
tasks_list_json_path_TL="${tasks_list_json_path_TL-__DERIVE__}"

# -- Model ---------------------------------------------------------------------
# model_family_name : MUST be accepted by check_model_supported() — i.e. present in
#                     lmms_eval's AVAILABLE_MODELS, with the "vllm_" prefix optional
#                     (qwen25vl, qwen3vl, gemma4, medgemma, ...). It selects the image
#                     processor used to compute the pixel size printed in the prompt.
# base_model_hf     : Hub id or local folder of the base checkpoint.
# run_name          : identifier reused for checkpoints, wandb and the merged repo.
model_family_name="${model_family_name:-qwen25vl}"
base_model_hf="${base_model_hf:-Qwen/Qwen2.5-VL-7B-Instruct}"
run_name="${run_name:-MedVision__SFT__<model>__D110k-AD5k-TL5k__CoT__512x512}"

# lora_checkpoint_dir : output dir for adapter (LoRA) or full checkpoints. The
#   full-FT and tool-use entry points rename this argument to checkpoint_dir
#   internally, so the SAME flag is used in every mode. Keep a ${run_name}
#   subfolder at the end so pushed LoRA repos get distinct names.
lora_checkpoint_dir="${lora_checkpoint_dir:-}"   # default: ${train_sft_dir}/${run_name}/checkpoints/${run_name}
# merged_model_hf / merged_model_dir : LoRA merge targets (ignored in full-FT mode).
merged_model_hf="${merged_model_hf:-}"           # default: ${run_name}
merged_model_dir="${merged_model_dir:-}"         # default: ${train_sft_dir}/${run_name}/merged_model

# -- Dataset construction ------------------------------------------------------
# skip_process_dataset      : true = load the prepared dataset from disk unchanged.
#                             false = rebuild it (OVERWRITES prepared_ds_dir).
# save_processed_img_to_disk: true (recommended) writes one PNG per slice into a
#                             tmp_prepared_png/ folder next to each NIfTI and stores
#                             the path in image_file_png; re-runs reuse the PNGs.
# process_img               : embeds decoded images IN the Arrow dataset — huge; leave false.
# new_shape_hw              : "<H> <W>" resize applied during preparation AND reflected
#                             in the prompt's pixel size. Empty string = native resolution.
# prepared_ds_dir           : leave EMPTY to use the default
#   <data_dir>/SFT-CoT_datasets/<model_family_name>/ds__AD<a>_D<d>_TL<t>_all<n>__resized-wh-<W>x<H>
#   where <a>/<d>/<t>/<n> are the requested caps or, for unset limits, the TRUE
#   split sizes (known only after phase A's load+split). Phase B receives the
#   directory phase A reports, so it never has to re-derive that name.
# prep_log                  : phase A's console output is tee'd here; the reported
#                             directory is read back from it for phase B.
# ds_download_mode          : reuse_dataset_if_exists | reuse_cache_if_exists | force_redownload
skip_process_dataset="${skip_process_dataset:-false}"
save_processed_img_to_disk="${save_processed_img_to_disk:-true}"
process_img="${process_img:-false}"
new_shape_hw="${new_shape_hw:-512 512}"
prepared_ds_dir="${prepared_ds_dir:-}"
prep_log="${prep_log:-}"                  # default: ${lora_checkpoint_dir}/prepare_dataset.log
ds_download_mode="${ds_download_mode:-reuse_dataset_if_exists}"

# -- Sample limits -------------------------------------------------------------
# Unset or -1 => the full pool. NEVER 0 (rejected as ambiguous); drop a task by
# clearing its tasks_list_json_path_* above. Per-task train limits cap each task
# AFTER the validation carve-out; train_sample_limit is a GLOBAL post-concatenation
# cap that truncates silently when it is smaller than the sum of the per-task limits
# and bootstraps WITH replacement when it is larger than the concatenated pool.
train_sample_limit="${train_sample_limit:--1}"
val_sample_limit="${val_sample_limit:--1}"
train_sample_limit_task_AD="${train_sample_limit_task_AD:-5500}"
val_sample_limit_task_AD="${val_sample_limit_task_AD:-45}"
train_sample_limit_task_Detection="${train_sample_limit_task_Detection:-110000}"
val_sample_limit_task_Detection="${val_sample_limit_task_Detection:-105}"
train_sample_limit_task_TL="${train_sample_limit_task_TL:-5500}"
val_sample_limit_task_TL="${val_sample_limit_task_TL:-50}"

# -- Schedule ------------------------------------------------------------------
# epoch: 10 in the LoRA recipes, 3 in the full-FT recipes.
# save_total_limit: full-FT checkpoints are ~190GB each at 31B — keep few.
epoch="${epoch:-10}"
save_steps="${save_steps:-100}"
eval_steps="${eval_steps:-100}"
logging_steps="${logging_steps:-50}"
save_total_limit="${save_total_limit:-10}"

# -- Compute -------------------------------------------------------------------
# effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# use_flash_attention_2=false falls back to EAGER; export MEDVISION_SFT_ATTN=sdpa for SDPA.
# gradient_checkpointing is mandatory for full-FT at 7B+ scale.
# num_workers_concat_datasets is clamped to min(CPUs, number of tasks); LOWER it (2)
#   when Arrow generation of a large detection plan exhausts the cgroup memory limit.
# num_workers_format_dataset drives the per-slice formatting map.
per_device_train_batch_size="${per_device_train_batch_size:-4}"
per_device_eval_batch_size="${per_device_eval_batch_size:-4}"
gradient_accumulation_steps="${gradient_accumulation_steps:-8}"
use_flash_attention_2="${use_flash_attention_2:-true}"
gradient_checkpointing="${gradient_checkpointing:-true}"
dataloader_pin_memory="${dataloader_pin_memory:-true}"
dataloader_num_workers="${dataloader_num_workers:-8}"
num_workers_concat_datasets="${num_workers_concat_datasets:-4}"
num_workers_format_dataset="${num_workers_format_dataset:-32}"

# -- Multi-task sampling -------------------------------------------------------
# Detection dominates the mixture, so the recipes oversample the minority tasks:
# p(task) ~ count^(1/T). T=1 is proportional, larger T flattens. Needs the
# __task_name column, which dataset preparation adds. Training-only knob.
enable_temperature_sampler="${enable_temperature_sampler:-true}"
temperature_sampler_T="${temperature_sampler_T:-5}"
temperature_sampler_task_column="${temperature_sampler_task_column:-__task_name}"
temperature_sampler_num_samples="${temperature_sampler_num_samples:--1}"

# -- Resume / merge / push -----------------------------------------------------
# resume_from_checkpoint : picks up the newest checkpoint in lora_checkpoint_dir and
#                          recomputes max_steps from the CURRENT dataset size + epochs.
# push_LoRA              : push the adapter after every save (full-FT reuses this flag
#                          as push_model for the trained weights).
# merge_model            : merge the adapter into the base model after training (LoRA only).
# merge_only             : skip training entirely, merge the last checkpoint and push.
# push_merged_model      : push the merged model (requires merged_model_hf).
resume_from_checkpoint="${resume_from_checkpoint:-true}"
push_LoRA="${push_LoRA:-false}"
push_merged_model="${push_merged_model:-false}"
merge_model="${merge_model:-true}"
merge_only="${merge_only:-false}"

# -- Weights & Biases ----------------------------------------------------------
# wandb_run_id must be UNIQUE inside wandb_project; reuse it (with wandb_resume=allow)
# to continue an interrupted run's chart, change it to start a fresh one.
wandb_resume="${wandb_resume:-allow}"
wandb_dir="${wandb_dir:-}"                # default: ${train_sft_dir}/${run_name}
wandb_project="${wandb_project:-MedVision-SFT-CoT-multiTasks}"
wandb_run_name="${wandb_run_name:-}"      # default: ${run_name}
wandb_run_id="${wandb_run_id:-}"          # default: ${run_name}

# -- Dataset annotation version -------------------------------------------------
# The loader hard-fails without MedVision_PLANNER_VERSION; pinning below the latest
# release additionally requires MedVision_ACK_RELEASE. 1.0.0 is the published recipe.
MedVision_PLANNER_VERSION="${MedVision_PLANNER_VERSION:-1.0.0}"
MedVision_ACK_RELEASE="${MedVision_ACK_RELEASE:-}"

# -- Mode / topology -----------------------------------------------------------
# mode  : lora | fullft | tooluse | noncot   (selects the train__* entry point)
# gpus  : value for CUDA_VISIBLE_DEVICES; num_processes is derived from it.
# fsdp_layer_cls : decoder layer class FSDP wraps in full-FT mode. It MUST match the
#   class name in the INSTALLED transformers for this checkpoint, e.g.
#   Gemma4TextDecoderLayer, Gemma3DecoderLayer, Qwen3_5DecoderLayer.
# mixed_precision : "bf16" gives fp32 master weights (fully resumable, 140GB-class GPUs);
#   pass "" together with MEDVISION_SFT_PURE_BF16=1 for the 80GB pure-bf16 recipe.
mode="${mode:-lora}"
gpus="${gpus:-0,1,2,3}"
main_process_port="${main_process_port:-29502}"
fsdp_layer_cls="${fsdp_layer_cls:-}"
mixed_precision="${mixed_precision:-bf16}"

# -- Optional environment knobs (exported only when non-empty) -----------------
# MEDVISION_SFT_ATTN=sdpa            attention impl override (new architectures)
# MEDVISION_SFT_COMPLETION_ONLY=1    Gemma/MedGemma completion-only loss ("cmplLoss")
# MEDVISION_SFT_OPTIM=...            adamw_bnb_8bit / paged_adamw_8bit / adafactor
# MEDVISION_SFT_SAVE_ONLY_MODEL=1    weights-only checkpoints (required with 8-bit optims)
# MEDVISION_SFT_PURE_BF16=1          no fp32 master weights (80GB full-FT recipe)
# MEDVISION_SFT_LR=4e-5              LR override (raise it under pure bf16)
# MEDVISION_SFT_USE_LIGER=1          fused linear cross-entropy (large-vocab models)
# MEDVISION_SFT_SYNC_EACH_BATCH=1    disable no_sync -> FSDP grads stay sharded
# MEDVISION_SFT_BF16_GRADS=1         bf16 FSDP grad shards (incompatible with bf16 MP)
# MEDVISION_SFT_MEMPROBE=1           per-rank memory report after wrap and step 1
# MEDVISION_SFT_MEMSNAPSHOT=1        dump a CUDA allocator snapshot on OOM
SFT_ENV_KNOBS="${SFT_ENV_KNOBS:-}"   # e.g. "MEDVISION_SFT_ATTN=sdpa MEDVISION_SFT_MEMPROBE=1"

# -- Behaviour switches --------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"              # 1 = print the commands instead of running them
RUN_ENV_SETUP="${RUN_ENV_SETUP:-0}"  # 1 = also run sft.env_setup (MUTATES the env)
PHASE="${PHASE:-AB}"                 # A | B | AB
PYTHON_BIN="${PYTHON_BIN:-python}"
lmms_eval_opt_deps="${lmms_eval_opt_deps:-}"   # env_setup extra, e.g. qwen2_5_vl

# =============================================================================
# 2. ARGUMENT PARSING
# =============================================================================

usage() {
    cat <<'USAGE'
Usage: sft_launcher_template.sh [OPTIONS]

Two-phase MedVision SFT launcher. Phase A prepares the dataset on CPU
(--process_dataset_only true); phase B trains with `accelerate launch`
(--skip_process_dataset true). Edit the variable block at the top of the file
or override individual values with the options below.

Options:
  --benchmark-dir DIR     Working root (default: <benchmark_dir> placeholder)
  --data-dir DIR          Dataset cache            (default: <benchmark-dir>/Data)
  --sft-dir DIR           Checkpoint/wandb root    (default: <benchmark-dir>/SFT)
  --tasks-dir DIR         Folder holding the three SFT task-list JSON files
  --tasks-ad FILE         A/D task list      ("" drops the A/D task)
  --tasks-detect FILE     Detection task list ("" drops the detection task)
  --tasks-tl FILE         T/L task list      ("" drops the T/L task)
  --family NAME           --model_family_name (qwen25vl|qwen3vl|gemma4|medgemma|...)
  --base-model ID         --base_model_hf (Hub id or local folder)
  --run-name NAME         Run identifier used for checkpoints and wandb
  --checkpoint-dir DIR    --lora_checkpoint_dir (also the full-FT checkpoint dir)
  --prepared-ds-dir DIR   Explicit prepared-dataset dir (default: the directory phase A
                          reports; a phase-B-only run without it re-derives the name)
  --mode MODE             lora | fullft | tooluse | noncot        (default: lora)
  --gpus LIST             CUDA_VISIBLE_DEVICES; --num_processes is derived
  --port N                accelerate --main_process_port
  --shape "H W"           --new_shape_hw ("" = native resolution)
  --epochs N              --epoch
  --batch-size N          --per_device_train_batch_size
  --grad-accum N          --gradient_accumulation_steps
  --fsdp-layer-cls NAME   FSDP transformer layer class (full-FT only)
  --mixed-precision P     accelerate --mixed_precision ("" to omit, pure bf16)
  --phase A|B|AB          Which phase(s) to emit/run                (default: AB)
  --dry-run               Same as DRY_RUN=1
  -h, --help              Show this message and exit

Environment:
  DRY_RUN=1        print the commands instead of executing them
  RUN_ENV_SETUP=1  additionally run `python -m medvision_bm.sft.env_setup`
  SFT_ENV_KNOBS    space-separated MEDVISION_SFT_* assignments to export
  PYTHON_BIN       interpreter to use (default: python)
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --benchmark-dir)   benchmark_dir="$2"; shift 2 ;;
        --data-dir)        data_dir="$2"; shift 2 ;;
        --sft-dir)         train_sft_dir="$2"; shift 2 ;;
        --tasks-dir)       tasks_dir="$2"; shift 2 ;;
        --tasks-ad)        tasks_list_json_path_AD="$2"; shift 2 ;;
        --tasks-detect)    tasks_list_json_path_detect="$2"; shift 2 ;;
        --tasks-tl)        tasks_list_json_path_TL="$2"; shift 2 ;;
        --family)          model_family_name="$2"; shift 2 ;;
        --base-model)      base_model_hf="$2"; shift 2 ;;
        --run-name)        run_name="$2"; shift 2 ;;
        --checkpoint-dir)  lora_checkpoint_dir="$2"; shift 2 ;;
        --prepared-ds-dir) prepared_ds_dir="$2"; shift 2 ;;
        --mode)            mode="$2"; shift 2 ;;
        --gpus)            gpus="$2"; shift 2 ;;
        --port)            main_process_port="$2"; shift 2 ;;
        --shape)           new_shape_hw="$2"; shift 2 ;;
        --epochs)          epoch="$2"; shift 2 ;;
        --batch-size)      per_device_train_batch_size="$2"; shift 2 ;;
        --grad-accum)      gradient_accumulation_steps="$2"; shift 2 ;;
        --fsdp-layer-cls)  fsdp_layer_cls="$2"; shift 2 ;;
        --mixed-precision) mixed_precision="$2"; shift 2 ;;
        --phase)           PHASE="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "[error] unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# -----------------------------------------------------------------------------
# 2b. DERIVED DEFAULTS — filled in only after the options above have been applied.
# -----------------------------------------------------------------------------
: "${data_dir:=${benchmark_dir}/Data}"
: "${train_sft_dir:=${benchmark_dir}/SFT}"
: "${tasks_dir:=${benchmark_dir}/tasks_list}"
[ "${tasks_list_json_path_AD}" = "__DERIVE__" ] &&
    tasks_list_json_path_AD="${tasks_dir}/tasks_MedVision-AD__train_SFT.json"
[ "${tasks_list_json_path_detect}" = "__DERIVE__" ] &&
    tasks_list_json_path_detect="${tasks_dir}/tasks_MedVision-detect__train_SFT.json"
[ "${tasks_list_json_path_TL}" = "__DERIVE__" ] &&
    tasks_list_json_path_TL="${tasks_dir}/tasks_MedVision-TL__train_SFT.json"
: "${lora_checkpoint_dir:=${train_sft_dir}/${run_name}/checkpoints/${run_name}}"
: "${prep_log:=${lora_checkpoint_dir}/prepare_dataset.log}"
: "${merged_model_hf:=${run_name}}"
: "${merged_model_dir:=${train_sft_dir}/${run_name}/merged_model}"
: "${wandb_dir:=${train_sft_dir}/${run_name}}"
: "${wandb_run_name:=${run_name}}"
: "${wandb_run_id:=${run_name}}"

case "${mode}" in
    lora)    train_module_suffix="SFT-CoT" ;;
    fullft)  train_module_suffix="fullFT-CoT" ;;
    tooluse) train_module_suffix="tooluse" ;;
    noncot)  train_module_suffix="SFT" ;;
    *) echo "[error] --mode must be lora, fullft, tooluse or noncot (got '${mode}')" >&2; exit 2 ;;
esac

# Map (mode, family) onto the public entry point module name.
case "${mode}:${model_family_name}" in
    lora:qwen25vl)    train_module="medvision_bm.sft.train__SFT-CoT__qwen2_5_vl" ;;
    lora:qwen3vl)     train_module="medvision_bm.sft.train__SFT-CoT__qwen3vl" ;;
    lora:gemma4)      train_module="medvision_bm.sft.train__SFT-CoT__gemma4" ;;
    lora:medgemma)    train_module="medvision_bm.sft.train__SFT-CoT__medgemma" ;;
    fullft:qwen25vl)  train_module="medvision_bm.sft.train__fullFT-CoT__qwen2_5_vl" ;;
    fullft:qwen3vl)   train_module="medvision_bm.sft.train__fullFT-CoT__qwen3vl" ;;
    fullft:gemma4)    train_module="medvision_bm.sft.train__fullFT-CoT__gemma4" ;;
    fullft:medgemma)  train_module="medvision_bm.sft.train__fullFT-CoT__medgemma" ;;
    noncot:qwen25vl)  train_module="medvision_bm.sft.train__SFT__qwen2_5_vl" ;;
    tooluse:qwen25vl) train_module="medvision_bm.sft.train__qwen25vl_AD_TL_tooluse" ;;
    *)
        echo "[error] no entry point for mode='${mode}' family='${model_family_name}'." >&2
        echo "        LoRA/full-FT CoT: qwen25vl, qwen3vl, gemma4, medgemma." >&2
        echo "        Non-CoT and tool-use exist for qwen25vl only." >&2
        exit 2
        ;;
esac

if [ "${mode}" = "tooluse" ]; then
    # The tool-use entry point formats A/D and T/L only; a detection list is ignored.
    tasks_list_json_path_detect=""
fi

num_processes="$(printf '%s' "${gpus}" | tr ',' '\n' | grep -c '[0-9]')"

# =============================================================================
# 3. COMMAND ASSEMBLY
# =============================================================================

common_args=()
add() { common_args+=("$1" "$2"); }

add --run_name "${run_name}"
add --model_family_name "${model_family_name}"
add --base_model_hf "${base_model_hf}"
add --lora_checkpoint_dir "${lora_checkpoint_dir}"
add --data_dir "${data_dir}"
add --ds_download_mode "${ds_download_mode}"
[ -n "${tasks_list_json_path_AD}" ]     && add --tasks_list_json_path_AD "${tasks_list_json_path_AD}"
[ -n "${tasks_list_json_path_detect}" ] && add --tasks_list_json_path_detect "${tasks_list_json_path_detect}"
[ -n "${tasks_list_json_path_TL}" ]     && add --tasks_list_json_path_TL "${tasks_list_json_path_TL}"
[ -n "${prepared_ds_dir}" ]             && add --prepared_ds_dir "${prepared_ds_dir}"
add --process_img "${process_img}"
add --epoch "${epoch}"
add --save_steps "${save_steps}"
add --eval_steps "${eval_steps}"
add --logging_steps "${logging_steps}"
add --save_total_limit "${save_total_limit}"
add --per_device_train_batch_size "${per_device_train_batch_size}"
add --per_device_eval_batch_size "${per_device_eval_batch_size}"
add --gradient_accumulation_steps "${gradient_accumulation_steps}"
add --use_flash_attention_2 "${use_flash_attention_2}"
add --gradient_checkpointing "${gradient_checkpointing}"
add --dataloader_pin_memory "${dataloader_pin_memory}"
add --dataloader_num_workers "${dataloader_num_workers}"
add --num_workers_concat_datasets "${num_workers_concat_datasets}"
add --num_workers_format_dataset "${num_workers_format_dataset}"
add --train_sample_limit "${train_sample_limit}"
add --val_sample_limit "${val_sample_limit}"
add --train_sample_limit_task_AD "${train_sample_limit_task_AD}"
add --val_sample_limit_task_AD "${val_sample_limit_task_AD}"
add --train_sample_limit_task_Detection "${train_sample_limit_task_Detection}"
add --val_sample_limit_task_Detection "${val_sample_limit_task_Detection}"
add --train_sample_limit_task_TL "${train_sample_limit_task_TL}"
add --val_sample_limit_task_TL "${val_sample_limit_task_TL}"
add --resume_from_checkpoint "${resume_from_checkpoint}"
add --wandb_resume "${wandb_resume}"
add --wandb_dir "${wandb_dir}"
add --wandb_project "${wandb_project}"
add --wandb_run_name "${wandb_run_name}"
add --wandb_run_id "${wandb_run_id}"
# LoRA-only merge/push flags: the full-FT and tool-use entry points ignore
# merge_model / merge_only / merged_model_* and reuse push_LoRA as push_model.
add --push_LoRA "${push_LoRA}"
if [ "${mode}" = "lora" ] || [ "${mode}" = "noncot" ]; then
    add --merged_model_hf "${merged_model_hf}"
    add --merged_model_dir "${merged_model_dir}"
    add --merge_model "${merge_model}"
    add --merge_only "${merge_only}"
    add --push_merged_model "${push_merged_model}"
fi

shape_args=()
if [ -n "${new_shape_hw}" ]; then
    # shellcheck disable=SC2206  # deliberate word split: "H W" -> two argv entries
    shape_args=(--new_shape_hw ${new_shape_hw})
fi

phaseA_cmd=("${PYTHON_BIN}" -m "${train_module}"
    --process_dataset_only true
    --skip_process_dataset "${skip_process_dataset}"
    --save_processed_img_to_disk "${save_processed_img_to_disk}"
    "${common_args[@]}" "${shape_args[@]}")

accelerate_args=(launch
    "--num_processes=${num_processes}"
    "--main_process_port=${main_process_port}")
[ -n "${mixed_precision}" ] && accelerate_args+=("--mixed_precision=${mixed_precision}")
if [ "${mode}" = "fullft" ] || [ "${mode}" = "tooluse" ]; then
    # Full-parameter training shards with FSDP. cpu_ram_efficient_loading +
    # sync_module_states are what keep from_pretrained off-GPU until the wrap.
    if [ -z "${fsdp_layer_cls}" ]; then
        echo "[error] --fsdp-layer-cls is required in mode '${mode}'" >&2
        exit 2
    fi
    accelerate_args+=(--use_fsdp
        --fsdp_sharding_strategy FULL_SHARD
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
        --fsdp_transformer_layer_cls_to_wrap "${fsdp_layer_cls}"
        --fsdp_state_dict_type FULL_STATE_DICT
        --fsdp_offload_params false
        --fsdp_cpu_ram_efficient_loading true
        --fsdp_sync_module_states true)
fi

phaseB_cmd=(accelerate "${accelerate_args[@]}" -m "${train_module}"
    --skip_process_dataset true
    --process_dataset_only false
    "${common_args[@]}"
    --enable_temperature_sampler "${enable_temperature_sampler}"
    --temperature_sampler_T "${temperature_sampler_T}"
    --temperature_sampler_task_column "${temperature_sampler_task_column}"
    --temperature_sampler_num_samples "${temperature_sampler_num_samples}"
    "${shape_args[@]}")

# =============================================================================
# 4. EXECUTION
# =============================================================================

emit() { printf '%q ' "$@"; printf '\n'; }

run() {
    if [ "${DRY_RUN}" = "1" ]; then
        emit "$@"
    else
        "$@"
    fi
}

echo "# MedVision SFT launcher"
echo "#   mode               : ${mode} (${train_module_suffix})"
echo "#   entry point        : python -m ${train_module}"
echo "#   model_family_name  : ${model_family_name}"
echo "#   base_model_hf      : ${base_model_hf}"
echo "#   GPUs               : ${gpus} (--num_processes=${num_processes})"
echo "#   resize             : ${new_shape_hw:-native}"
echo "#   phases             : ${PHASE}"
echo "#   DRY_RUN            : ${DRY_RUN}"
echo

# Sanitise HF_TOKEN: pod-injected secrets often carry a trailing newline, which
# corrupts the Authorization header and yields 401 on gated models/datasets.
if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '[:space:]')"
    export HF_TOKEN
fi

# The dataset loader hard-fails without a planner version.
export MedVision_PLANNER_VERSION
[ -n "${MedVision_ACK_RELEASE}" ] && export MedVision_ACK_RELEASE

# Optional MEDVISION_SFT_* knobs.
if [ -n "${SFT_ENV_KNOBS}" ]; then
    for kv in ${SFT_ENV_KNOBS}; do
        echo "export ${kv}"
        [ "${DRY_RUN}" = "1" ] || export "${kv?}"
    done
    echo
fi

if [ "${RUN_ENV_SETUP}" = "1" ]; then
    echo "## env setup (MUTATES the active environment)"
    setup_cmd=("${PYTHON_BIN}" -m medvision_bm.sft.env_setup --data_dir "${data_dir}")
    [ -n "${lmms_eval_opt_deps}" ] && setup_cmd+=(--lmms_eval_opt_deps "${lmms_eval_opt_deps}")
    run "${setup_cmd[@]}"
    # env_setup leaves a protobuf that wandb>=0.21's generated stubs reject
    # ("cannot import name 'Imports' from wandb.proto"), which breaks the
    # trl.SFTTrainer import at train time. 6.33.0 matches the frozen SFT pins.
    run "${PYTHON_BIN}" -m pip install "protobuf==6.33.0"
    echo
fi

case "${PHASE}" in
    A|AB)
        echo "## PHASE A — dataset preparation (CPU; writes the prepared dataset)"
        if [ "${DRY_RUN}" = "1" ]; then
            emit "${phaseA_cmd[@]}"
            echo "#   ... 2>&1 | tee ${prep_log}   (phase B reads the reported directory from this log)"
        else
            mkdir -p "$(dirname "${prep_log}")"
            # pipefail (set above) makes a failed phase A abort the script here.
            "${phaseA_cmd[@]}" 2>&1 | tee "${prep_log}"
        fi
        # Hand the directory phase A reported to phase B, unless the user fixed one:
        # its default name encodes the TRUE split sizes, so it is only known now.
        if [ -z "${prepared_ds_dir}" ]; then
            if [ "${DRY_RUN}" = "1" ]; then
                prepared_ds_dir="<dir reported by phase A>"
            else
                prepared_ds_dir="$(sed -n "s/.*Prepared dataset saved at '\([^']*\)'.*/\1/p" "${prep_log}" | tail -n 1)"
                if [ -z "${prepared_ds_dir}" ] || [ ! -d "${prepared_ds_dir}" ]; then
                    echo "[error] could not read the prepared dataset directory from ${prep_log}; not launching phase B" >&2
                    exit 1
                fi
            fi
            echo "#   prepared_ds_dir (from phase A): ${prepared_ds_dir}"
            phaseB_cmd+=(--prepared_ds_dir "${prepared_ds_dir}")
        fi
        echo
        ;;
esac

case "${PHASE}" in
    B|AB)
        echo "## PHASE B — training (REQUIRES GPUs)"
        if [ -z "${prepared_ds_dir}" ]; then
            echo "#   NOTE: no --prepared-ds-dir given: rank 0 will re-run the load+split stage just to"
            echo "#   derive the default directory name. Pass the directory phase A reported to avoid it."
        fi
        if [ "${DRY_RUN}" = "1" ]; then
            printf 'CUDA_VISIBLE_DEVICES=%s ' "${gpus}"
            emit "${phaseB_cmd[@]}"
        else
            CUDA_VISIBLE_DEVICES="${gpus}" "${phaseB_cmd[@]}"
        fi
        echo
        ;;
esac
