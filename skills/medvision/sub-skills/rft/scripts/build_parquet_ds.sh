#!/usr/bin/env bash
# build_parquet_ds.sh -- build a verl-ready MedVision RFT parquet dataset with explicit paths and SMALL defaults.
#
# Purpose
#   Safe wrapper around the MedVision parquet builders
#     python -m medvision_bm.rft.verl.build_parquet_ds                 (default)
#     python -m medvision_bm.rft.verl.build_parquet_ds__checkpointed   (--checkpointed: sharded, resumable, OOM-safe)
#   Adapted from the repository's `script/rft/build_parquet_ds__verl__*.sh` launchers WITHOUT their
#   conda-env creation, wheel build and `medvision_bm.sft.env_setup` steps -- prepare the environment
#   first (see ../../environment-setup/SKILL.md and ../SKILL.md "Prerequisites").
#
# Prerequisites
#   * medvision_bm importable with the SFT extras of the target model family: the builders import
#     medvision_bm.sft.sft_utils and load the model's image processor (transformers) for --model-hf.
#   * medvision_ds installed into <data_dir>/src (mvbm install mvds -d <data_dir>).
#   * MedVision_PLANNER_VERSION exported (the loader requires it; '1.0.0' = the paper's data).
#   * The MedVision datasets named in the task JSONs are downloaded under <data_dir> or downloadable
#     (first build triggers Hugging Face downloads -- network + disk).
#   * CPU only. Peak RAM ~ workers-format x 50 buffered images (~0.75 MB each at 512x512) + the
#     formatted split; use --checkpointed for >~100K rows.
#
# Examples
#   # 1) preview only (no build):
#   MedVision_PLANNER_VERSION=1.0.0 bash build_parquet_ds.sh --data-dir <data_dir> \
#       --tasks-tl <repo>/tasks_list/tasks_MedVision-TL__train_SFT.json --dry-run
#   # 2) tiny 3-task smoke build (defaults: 100 train / 10 val per task, 512x512, qwen25vl):
#   MedVision_PLANNER_VERSION=1.0.0 bash build_parquet_ds.sh --data-dir <data_dir> \
#       --tasks-ad <repo>/tasks_list/tasks_MedVision-AD__train_SFT.json \
#       --tasks-tl <repo>/tasks_list/tasks_MedVision-TL__train_SFT.json \
#       --tasks-detect <repo>/tasks_list/tasks_MedVision-detect__train_SFT.json
#   # 3) the paper's 1M detection set, resumable:
#   MedVision_PLANNER_VERSION=1.0.0 bash build_parquet_ds.sh --data-dir <data_dir> \
#       --tasks-detect <repo>/tasks_list/tasks_MedVision-detect__train_SFT.json \
#       --train-limit-detect 1000000 --val-limit-detect 500 --checkpointed --shard-size 50000 \
#       --workers-concat 16 --workers-format 64
#
# Output
#   <data_dir>/verl_datasets/<model-family>/ds__AD<a>_D<d>_TL<t>_all<total>[_wo-CoT-Instruct]__resized-hw-<H>x<W>/
#     train_verl.parquet, validation_verl.parquet  (+ shards/, checkpoint.json with --checkpointed)
#   The builders have no output-directory flag; --out-dir MOVES the finished directory there afterwards.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: build_parquet_ds.sh --data-dir <dir> (--tasks-ad <json> | --tasks-detect <json> | --tasks-tl <json>)... [options]

Required
  --data-dir DIR            MedVision data directory (exported as MedVision_DATA_DIR; output goes to DIR/verl_datasets/)
  at least one of:
  --tasks-ad JSON           A/D task list (BiometricsFromLandmarks configs, e.g. tasks_MedVision-AD__train_SFT.json)
  --tasks-detect JSON       Detection task list (BoxSize configs, e.g. tasks_MedVision-detect__train_SFT.json)
  --tasks-tl JSON           T/L task list (TumorLesionSize configs, e.g. tasks_MedVision-TL__train_SFT.json)

Model / image processor (the dataset is ONLY valid for models sharing this processor)
  --model-family NAME       model_family_name key of get_resized_img_shape (default: qwen25vl)
  --model-hf ID_OR_PATH     Hugging Face id or local dir whose image processor is loaded (default: Qwen/Qwen2.5-VL-7B-Instruct)
  --new-shape-hw H W        resize before embedding (default: 512 512); pass "--new-shape-hw none" for original size

Sample limits (small by default; -1 = whole pool)
  --train-limit-ad N        (default 100)     --val-limit-ad N      (default 10)
  --train-limit-detect N    (default 100)     --val-limit-detect N  (default 10)
  --train-limit-tl N        (default 100)     --val-limit-tl N      (default 10)
  --train-limit N           global post-concat cap (default: SUM of the per-task train limits, or -1 if any is -1)
  --val-limit N             global post-concat val cap (default: SUM of the per-task val limits)

Builder / resources
  --checkpointed            use build_parquet_ds__checkpointed (shards + checkpoint.json + stream-merge)
  --shard-size N            rows per shard, --checkpointed only (default 50000)
  --workers-concat N        --num_workers_concat_datasets (default 2; <= number of task configs)
  --workers-format N        --num_workers_format_dataset (default 8; the main RAM knob)
  --without-cot-instruction build the "lite" prompts (SYSTEM_PROMPT_LITE, no per-step tags; NOT the paper recipe)
  --download-mode MODE      reuse_dataset_if_exists (default) | reuse_cache_if_exists | force_redownload
  --python EXE              interpreter to use (default: python)
  --out-dir DIR             move the finished dataset directory to DIR (must not exist yet)
  --dry-run                 print the resolved command and output directory, run nothing
  -h, --help
USAGE
}

# ---- defaults -------------------------------------------------------------------------------
data_dir=""; tasks_ad=""; tasks_detect=""; tasks_tl=""
model_family="qwen25vl"; model_hf="Qwen/Qwen2.5-VL-7B-Instruct"
shape_h=512; shape_w=512; use_shape=1
tr_ad=100; tr_det=100; tr_tl=100; va_ad=10; va_det=10; va_tl=10
train_limit=""; val_limit=""
checkpointed=0; shard_size=50000; workers_concat=2; workers_format=8
without_cot=0; download_mode="reuse_dataset_if_exists"; python_exe="python"; out_dir=""; dry_run=0

# ---- parse ----------------------------------------------------------------------------------
need() { [[ $# -ge 2 && -n "${2:-}" ]] || { echo "ERROR: $1 needs a value" >&2; exit 1; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) need "$@"; data_dir="$2"; shift 2;;
    --tasks-ad) need "$@"; tasks_ad="$2"; shift 2;;
    --tasks-detect) need "$@"; tasks_detect="$2"; shift 2;;
    --tasks-tl) need "$@"; tasks_tl="$2"; shift 2;;
    --model-family) need "$@"; model_family="$2"; shift 2;;
    --model-hf) need "$@"; model_hf="$2"; shift 2;;
    --new-shape-hw)
      need "$@"
      if [[ "$2" == "none" ]]; then use_shape=0; shift 2
      else [[ $# -ge 3 ]] || { echo "ERROR: --new-shape-hw needs H W (or 'none')" >&2; exit 1; }
           shape_h="$2"; shape_w="$3"; use_shape=1; shift 3; fi;;
    --train-limit-ad) need "$@"; tr_ad="$2"; shift 2;;
    --train-limit-detect) need "$@"; tr_det="$2"; shift 2;;
    --train-limit-tl) need "$@"; tr_tl="$2"; shift 2;;
    --val-limit-ad) need "$@"; va_ad="$2"; shift 2;;
    --val-limit-detect) need "$@"; va_det="$2"; shift 2;;
    --val-limit-tl) need "$@"; va_tl="$2"; shift 2;;
    --train-limit) need "$@"; train_limit="$2"; shift 2;;
    --val-limit) need "$@"; val_limit="$2"; shift 2;;
    --checkpointed) checkpointed=1; shift;;
    --shard-size) need "$@"; shard_size="$2"; shift 2;;
    --workers-concat) need "$@"; workers_concat="$2"; shift 2;;
    --workers-format) need "$@"; workers_format="$2"; shift 2;;
    --without-cot-instruction) without_cot=1; shift;;
    --download-mode) need "$@"; download_mode="$2"; shift 2;;
    --python) need "$@"; python_exe="$2"; shift 2;;
    --out-dir) need "$@"; out_dir="$2"; shift 2;;
    --dry-run) dry_run=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 1;;
  esac
done

# ---- validate -------------------------------------------------------------------------------
[[ -n "$data_dir" ]] || { echo "ERROR: --data-dir is required" >&2; exit 1; }
[[ -d "$data_dir" || $dry_run -eq 1 ]] || { echo "ERROR: --data-dir '$data_dir' is not a directory" >&2; exit 1; }
[[ -n "$tasks_ad$tasks_detect$tasks_tl" ]] || { echo "ERROR: give at least one of --tasks-ad / --tasks-detect / --tasks-tl" >&2; exit 1; }
for f in "$tasks_ad" "$tasks_detect" "$tasks_tl"; do
  [[ -z "$f" || -f "$f" ]] || { echo "ERROR: task list JSON not found: $f" >&2; exit 1; }
done
for v in "$tr_ad" "$tr_det" "$tr_tl" "$va_ad" "$va_det" "$va_tl" "$shard_size" "$workers_concat" "$workers_format"; do
  [[ "$v" =~ ^-?[0-9]+$ ]] || { echo "ERROR: numeric value expected, got '$v'" >&2; exit 1; }
done
for v in "$tr_ad" "$tr_det" "$tr_tl" "$va_ad" "$va_det" "$va_tl"; do
  [[ "$v" != "0" ]] || { echo "ERROR: a sample limit of 0 is rejected by parse_sample_limits (ambiguous); omit the task instead" >&2; exit 1; }
done
[[ $checkpointed -eq 1 || "$shard_size" == "50000" ]] || echo "WARNING: --shard-size is ignored without --checkpointed" >&2
if [[ -n "$out_dir" && -e "$out_dir" ]]; then echo "ERROR: --out-dir '$out_dir' already exists" >&2; exit 1; fi

# Known model_family_name keys of get_resized_img_shape (local VLM families; the function also has
# API-model branches that are irrelevant for RFT). A wrong key raises inside the T/L and A/D formatters.
known_families="qwen25vl qwen3vl gemma3 gemma4 medgemma lingshu llama_3_2_vision llava_onevision internvl3 minimax_m3 glm4v meddr llava_med huatuogpt_vision healthgpt"
if ! grep -qw -- "$model_family" <<<"$known_families"; then
  echo "WARNING: model family '$model_family' is not one of: $known_families" >&2
  echo "         T/L and A/D prompts embed the perceived image size of this family's processor; an unknown key fails at format time." >&2
fi

if [[ -z "${MedVision_PLANNER_VERSION:-}" ]]; then
  msg="MedVision_PLANNER_VERSION is not set; the medvision_ds loader requires it (e.g. export MedVision_PLANNER_VERSION=1.0.0 for the paper's data; pinning below the latest also needs MedVision_ACK_RELEASE)."
  if [[ $dry_run -eq 1 ]]; then echo "WARNING: $msg" >&2; else echo "ERROR: $msg" >&2; exit 1; fi
fi

# ---- resolve limits exactly like parse_sample_limits (absent task -> 0; val fallback 100) ----------
lim_ad=$tr_ad;   [[ -n "$tasks_ad" ]]     || lim_ad=0
lim_det=$tr_det; [[ -n "$tasks_detect" ]] || lim_det=0
lim_tl=$tr_tl;   [[ -n "$tasks_tl" ]]     || lim_tl=0
if [[ -z "$train_limit" ]]; then
  if [[ ( -n "$tasks_ad" && $tr_ad -lt 0 ) || ( -n "$tasks_detect" && $tr_det -lt 0 ) || ( -n "$tasks_tl" && $tr_tl -lt 0 ) ]]; then
    train_limit=-1
  else
    train_limit=$(( lim_ad + lim_det + lim_tl ))
  fi
fi
if [[ -z "$val_limit" ]]; then
  s=0; [[ -n "$tasks_ad" ]] && s=$(( s + va_ad )); [[ -n "$tasks_detect" ]] && s=$(( s + va_det )); [[ -n "$tasks_tl" ]] && s=$(( s + va_tl ))
  val_limit=$s
fi
sum_train=$(( lim_ad + lim_det + lim_tl ))
if [[ $train_limit -gt 0 && $sum_train -gt 0 && $train_limit -ne $sum_train ]]; then
  echo "WARNING: --train-limit ($train_limit) != sum of per-task train limits ($sum_train): the global cap silently truncates (or oversamples with replacement) the concatenated set." >&2
fi

cot_tag=""; [[ $without_cot -eq 1 ]] && cot_tag="_wo-CoT-Instruct"
if [[ $use_shape -eq 1 ]]; then shape_tag="resized-hw-${shape_h}x${shape_w}"; else shape_tag="original"; fi
ds_dir="ds__AD${lim_ad}_D${lim_det}_TL${lim_tl}_all${train_limit}${cot_tag}__${shape_tag}"
built_dir="${data_dir}/verl_datasets/${model_family}/${ds_dir}"

module="medvision_bm.rft.verl.build_parquet_ds"
[[ $checkpointed -eq 1 ]] && module="medvision_bm.rft.verl.build_parquet_ds__checkpointed"

# ---- assemble command -----------------------------------------------------------------------
cmd=( "$python_exe" -m "$module"
      --model_family_name "$model_family" --model_hf "$model_hf" --data_dir "$data_dir"
      --ds_download_mode "$download_mode"
      --num_workers_concat_datasets "$workers_concat" --num_workers_format_dataset "$workers_format"
      --train_sample_limit "$train_limit" --val_sample_limit "$val_limit" )
[[ -n "$tasks_ad" ]]     && cmd+=( --tasks_list_json_path_AD "$tasks_ad" --train_sample_limit_task_AD "$tr_ad" --val_sample_limit_task_AD "$va_ad" )
[[ -n "$tasks_detect" ]] && cmd+=( --tasks_list_json_path_detect "$tasks_detect" --train_sample_limit_task_Detection "$tr_det" --val_sample_limit_task_Detection "$va_det" )
[[ -n "$tasks_tl" ]]     && cmd+=( --tasks_list_json_path_TL "$tasks_tl" --train_sample_limit_task_TL "$tr_tl" --val_sample_limit_task_TL "$va_tl" )
[[ $use_shape -eq 1 ]]   && cmd+=( --new_shape_hw "$shape_h" "$shape_w" )
[[ $without_cot -eq 1 ]] && cmd+=( --without_cot_instruction )
[[ $checkpointed -eq 1 ]] && cmd+=( --shard_size "$shard_size" )

echo "Builder module      : $module"
echo "Model family / HF   : $model_family / $model_hf"
echo "Per-task train/val  : AD ${lim_ad}/${va_ad}  Detection ${lim_det}/${va_det}  TL ${lim_tl}/${va_tl}   (0 = task not used)"
echo "Global train/val cap: ${train_limit} / ${val_limit}"
echo "Expected output dir : $built_dir"
[[ -n "$out_dir" ]] && echo "Will move to        : $out_dir"
echo "Command:"; printf '  %q' "${cmd[@]}"; echo

if [[ $dry_run -eq 1 ]]; then echo "(dry run: nothing executed)"; exit 0; fi

# ---- run ------------------------------------------------------------------------------------
"$python_exe" -c "import medvision_bm.rft.verl.build_parquet_ds" 2>/dev/null \
  || { echo "ERROR: '$python_exe' cannot import medvision_bm.rft.verl.build_parquet_ds -- install medvision_bm (+ SFT extras) first; see ../../environment-setup/SKILL.md" >&2; exit 1; }
export MedVision_DATA_DIR="$data_dir"
export PYTHONFAULTHANDLER=1          # traceback on fatal signals (as in the repository's 1M launcher)
export HF_DATASETS_VERBOSITY=warning # surface datasets-level errors before they become silent hangs
"${cmd[@]}"

if [[ ! -d "$built_dir" ]]; then
  echo "WARNING: expected output directory not found: $built_dir (builder naming may have changed; check the builder's own 'Prepared Verl parquet dataset directory' line above)" >&2
  exit 0
fi
if [[ -n "$out_dir" ]]; then
  mkdir -p "$(dirname "$out_dir")"
  mv "$built_dir" "$out_dir"
  echo "Moved dataset to: $out_dir  (NOTE: a resumed --checkpointed build would restart; move only finished builds)"
  built_dir="$out_dir"
fi
echo "Done. Inspect with: python $(dirname "$0")/inspect_parquet_ds.py --path \"$built_dir\""
