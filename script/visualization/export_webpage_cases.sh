#!/bin/bash
# Used to generate samples for the case viewers in MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Export benchmark results as case-study data for the project webpage's interactive
# case viewer: writes one overlay PNG per case into <PAGE_DIR>/figure/cases/<model>/
# and regenerates <PAGE_DIR>/static/js/cases.js (nested by task -> model -> cases).
#
# Multiple models per task are supported. Per task, every model's case list is the SAME
# samples in the SAME order, so switching models in the viewer compares answers on one case.
# For models whose response can't be parsed into GT-vs-prediction coordinates, the figure is
# image-only and the case is flagged (the viewer's metrics panel shows a parsing-failure note).
#
# Run in the MedVision conda env (needs nibabel + matplotlib + medvision_bm).
#
# Usage:
#   # Full re-render (overwrites every overlay PNG; default):
#   bash export_webpage_cases.sh
#
#   # Append/incremental: reuse PNGs already on disk, only draw NEW cases:
#   SKIP_EXISTING=1 bash export_webpage_cases.sh
#
#   # Smaller/faster sample set:
#   PER_DATASET_TL=2 PER_TASK_MAX=40 bash export_webpage_cases.sh
#
# Environment-variable knobs (all map to export_webpage_cases.py CLI flags):
#   PAGE_DIR             Project page repo root           -> --page_dir
#   PER_DATASET_DET      Detection samples / dataset (20) -> --per_dataset_det
#   PER_DATASET_TL       TL samples / dataset (20)        -> --per_dataset_tl
#   PER_DATASET_AD       AD samples / dataset (40)        -> --per_dataset_ad
#   PER_TASK_MAX         Hard cap on samples / task       -> --per_task_max
#   SEED                 Sample-selection seed (1234)     -> --seed
#   REMOVED_SAMPLES_DIR  TL multi-cluster exclusion root  -> --removed_samples_dir
#   SKIP_EXISTING=1      Skip re-rendering existing PNGs  -> --skip_existing
#
# Hardcoded below (not env-controlled): --nonmedvision_topleft enables DUAL-ORIGIN mode
# for off-the-shelf (non-MedVision) TL & AD cases — each such case gets both a top-left and
# a lower-left overlay + per-origin localization metrics, and the viewer shows an origin
# toggle. MedVision-V0 and all Detection cases stay single-version. cases.js is always
# rebuilt in full regardless of SKIP_EXISTING.
#
# IMPORTANT: SKIP_EXISTING trusts on-disk PNGs. If any model's JSONLs were re-generated
# since the last export, run WITHOUT it so stale overlays are redrawn.
#
# Edit the *_MODELS arrays below to add/remove models ("DisplayName=/path/to/model_dir").
# Missing dirs are skipped gracefully.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$MEDVISION_DIR/Results"

# Per-task eval folders + the MedVision-V0 (fullRFT, AD-TL-D, s250) model dir name.
DET_FOLDER="$RESULTS/MedVision-detect-v2"
TL_FOLDER="$RESULTS/MedVision-TL-v2-CoT"
AD_FOLDER="$RESULTS/MedVision-AD-v2-CoT"
MV_V0="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250"

# All 13 leaderboard models (Tables 2/3/4). Display name = table text. MedVision-V0 first
# (viewer default). Detection uses HF-org-prefixed folders for Lingshu/MedGemma-4B/HuatuoGPT;
# TL/AD use the bare names.
DET_MODELS=(
    "MedVision-V0 (7B)=$DET_FOLDER/${MV_V0}_CoT"
    "Qwen2.5-VL (7B)=$DET_FOLDER/Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL (32B)=$DET_FOLDER/Qwen2.5-VL-32B-Instruct"
    "InternVL3 (38B)=$DET_FOLDER/InternVL3-38B"
    "Gemma3 (27B)=$DET_FOLDER/gemma-3-27b-it"
    "Llama3.2-Vision (11B)=$DET_FOLDER/Llama-3.2-11B-Vision-Instruct"
    "LLaVA-OneVision (72B)=$DET_FOLDER/llava-onevision-qwen2-72b-ov-hf"
    "Lingshu (32B)=$DET_FOLDER/lingshu-medical-mllm__Lingshu-32B"
    "MedGemma (4B)=$DET_FOLDER/google__medgemma-4b-it"
    "MedGemma (27B)=$DET_FOLDER/medgemma-27b-it"
    "MedDr (40B)=$DET_FOLDER/MedDr__BF16"
    "HuatuoGPT-Vision (34B)=$DET_FOLDER/FreedomIntelligence__HuatuoGPT-Vision-34B"
    "HealthGPT-L14 (14B)=$DET_FOLDER/HealthGPT-L14"
)
TL_MODELS=(
    "MedVision-V0 (7B)=$TL_FOLDER/${MV_V0}"
    "Qwen2.5-VL (7B)=$TL_FOLDER/Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL (32B)=$TL_FOLDER/Qwen2.5-VL-32B-Instruct"
    "InternVL3 (38B)=$TL_FOLDER/InternVL3-38B"
    "Gemma3 (27B)=$TL_FOLDER/gemma-3-27b-it"
    "Llama3.2-Vision (11B)=$TL_FOLDER/Llama-3.2-11B-Vision-Instruct"
    "LLaVA-OneVision (72B)=$TL_FOLDER/llava-onevision-qwen2-72b-ov-hf"
    "Lingshu (32B)=$TL_FOLDER/lingshu-32b"
    "MedGemma (4B)=$TL_FOLDER/medgemma-4b-it"
    "MedGemma (27B)=$TL_FOLDER/medgemma-27b-it"
    "MedDr (40B)=$TL_FOLDER/MedDr__BF16"
    "HuatuoGPT-Vision (34B)=$TL_FOLDER/HuatuoGPT-Vision-34B"
    "HealthGPT-L14 (14B)=$TL_FOLDER/HealthGPT-L14"
)
AD_MODELS=(
    "MedVision-V0 (7B)=$AD_FOLDER/${MV_V0}"
    "Qwen2.5-VL (7B)=$AD_FOLDER/Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL (32B)=$AD_FOLDER/Qwen2.5-VL-32B-Instruct"
    "InternVL3 (38B)=$AD_FOLDER/InternVL3-38B"
    "Gemma3 (27B)=$AD_FOLDER/gemma-3-27b-it"
    "Llama3.2-Vision (11B)=$AD_FOLDER/Llama-3.2-11B-Vision-Instruct"
    "LLaVA-OneVision (72B)=$AD_FOLDER/llava-onevision-qwen2-72b-ov-hf"
    "Lingshu (32B)=$AD_FOLDER/lingshu-32b"
    "MedGemma (4B)=$AD_FOLDER/medgemma-4b-it"
    "MedGemma (27B)=$AD_FOLDER/medgemma-27b-it"
    "MedDr (40B)=$AD_FOLDER/MedDr__BF16"
    "HuatuoGPT-Vision (34B)=$AD_FOLDER/HuatuoGPT-Vision-34B"
    "HealthGPT-L14 (14B)=$AD_FOLDER/HealthGPT-L14"
)

PAGE_DIR="${PAGE_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io}"
PER_DATASET_DET="${PER_DATASET_DET:-20}"
PER_DATASET_TL="${PER_DATASET_TL:-20}"
PER_DATASET_AD="${PER_DATASET_AD:-40}"
PER_TASK_MAX="${PER_TASK_MAX:-1000}"
SEED="${SEED:-1234}"

# Optional T/L sample filtering (matches summarize_TL_task.py). Set REMOVED_SAMPLES_DIR
# to your dataset root (e.g. .../Data/Datasets) to drop the v1.0.0 -> v1.1.0 multi-cluster
# T/L samples before selection. Unset -> no filtering.
REMOVED_SAMPLES_DIR="${REMOVED_SAMPLES_DIR:-/mnt/vincent-pvc-rwm/Github/MedVision/Data/Datasets}"
REMOVED_ARGS=()
if [ -n "$REMOVED_SAMPLES_DIR" ]; then
    REMOVED_ARGS=(--removed_samples_dir "$REMOVED_SAMPLES_DIR")
fi

# Append mode: set SKIP_EXISTING=1 to reuse overlay PNGs already on disk (skip
# re-rendering, only new cases are drawn). Unset/empty -> overwrite every PNG (default).
SKIP_EXISTING="${SKIP_EXISTING:-}"
SKIP_ARGS=()
if [ -n "$SKIP_EXISTING" ]; then
    SKIP_ARGS=(--skip_existing)
fi

python "$SCRIPT_DIR/export_webpage_cases.py" \
    --det_models "${DET_MODELS[@]}" \
    --tl_models "${TL_MODELS[@]}" \
    --ad_models "${AD_MODELS[@]}" \
    --page_dir "$PAGE_DIR" \
    --per_dataset_det "$PER_DATASET_DET" \
    --per_dataset_tl "$PER_DATASET_TL" \
    --per_dataset_ad "$PER_DATASET_AD" \
    --per_task_max "$PER_TASK_MAX" \
    --seed "$SEED" \
    --nonmedvision_topleft \
    "${REMOVED_ARGS[@]}" \
    "${SKIP_ARGS[@]}"
