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
#   # Smaller/faster sample set (cap total cases per task):
#   PER_TASK_MAX=40 bash export_webpage_cases.sh
#
# By default the selection SWEEPS every distinct displayed target (the viewer's TARGET
# buttons), taking one seeded case per target — so every label is covered.
#
# Environment-variable knobs (all map to export_webpage_cases.py CLI flags):
#   PAGE_DIR             Project page repo root           -> --page_dir
#   PER_TARGET_DET       Detection samples / target (1)   -> --per_target_det
#   PER_TARGET_TL        TL samples / target (1)          -> --per_target_tl
#   PER_TARGET_AD        AD samples / target (1)          -> --per_target_ad
#   PER_TASK_MAX         Optional cap on cases / task     -> --per_task_max (unset = no cap)
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

# All 18 leaderboard models (Tables 2/3/4), in leaderboard order. Display name = table text.
# MedVision-V0 first (viewer default). Result dirs are kept in sync with the radar configs
# config-{detect,TL,AD}-CoT.yaml, which are the source of truth for folder -> display name;
# folder names differ per task (HF-org prefixes on detection, _bugfix-* variants on TL/AD).
DET_MODELS=(
    "MedVision-V0 (7B)=$DET_FOLDER/${MV_V0}_CoT"
    "Gemma-4 (31B)=$DET_FOLDER/gemma-4-31B-it"
    "Lingshu (32B)=$DET_FOLDER/lingshu-medical-mllm__Lingshu-32B"
    "Qwen3-VL-Thinking (32B)=$DET_FOLDER/Qwen3-VL-32B-Thinking"
    "MedGemma (27B)=$DET_FOLDER/medgemma-27b-it"
    "MedGemma (4B)=$DET_FOLDER/google__medgemma-4b-it"
    "Qwen2.5-VL (32B)=$DET_FOLDER/Qwen2.5-VL-32B-Instruct"
    "LLaVA-OneVision (72B)=$DET_FOLDER/llava-onevision-qwen2-72b-ov-hf"
    "MiniMax-M3 (428B, int4)=$DET_FOLDER/MiniMax-M3-INT4"
    "InternVL3 (38B)=$DET_FOLDER/InternVL3-38B"
    "Qwen2.5-VL (7B)=$DET_FOLDER/Qwen2.5-VL-7B-Instruct"
    "GLM-4.6V-Flash (9B)=$DET_FOLDER/GLM-4.6V-Flash"
    "Gemma-3 (27B)=$DET_FOLDER/gemma-3-27b-it"
    "HealthGPT (14B)=$DET_FOLDER/HealthGPT-L14"
    "MedDr (40B)=$DET_FOLDER/MedDr__BF16"
    "HuatuoGPT-Vision (34B)=$DET_FOLDER/FreedomIntelligence__HuatuoGPT-Vision-34B"
    "GLM-4.6V (106B)=$DET_FOLDER/GLM-4.6V"
    "Llama-3.2-Vision (11B)=$DET_FOLDER/Llama-3.2-11B-Vision-Instruct"
)
TL_MODELS=(
    "MedVision-V0 (7B)=$TL_FOLDER/${MV_V0}"
    "Gemma-4 (31B)=$TL_FOLDER/gemma-4-31B-it"
    "MiniMax-M3 (428B, int4)=$TL_FOLDER/MiniMax-M3-INT4"
    "GLM-4.6V (106B)=$TL_FOLDER/GLM-4.6V"
    "GLM-4.6V-Flash (9B)=$TL_FOLDER/GLM-4.6V-Flash"
    "Lingshu (32B)=$TL_FOLDER/lingshu-32b"
    "Qwen3-VL-Thinking (32B)=$TL_FOLDER/Qwen3-VL-32B-Thinking"
    "HealthGPT (14B)=$TL_FOLDER/HealthGPT-L14_bugfix-0a4c5e2"
    "Gemma-3 (27B)=$TL_FOLDER/gemma-3-27b-it"
    "LLaVA-OneVision (72B)=$TL_FOLDER/LLaVA-OneVision_bugfix-0a4c5e2"
    "Qwen2.5-VL (7B)=$TL_FOLDER/Qwen2.5-VL-7B-Instruct"
    "InternVL3 (38B)=$TL_FOLDER/InternVL3-38B_bugfix-2eb7706"
    "MedGemma (4B)=$TL_FOLDER/medgemma-4b-it"
    "HuatuoGPT-Vision (34B)=$TL_FOLDER/HuatuoGPT-Vision-34B_bugfix-2eb7706-wStopStrings"
    "MedDr (40B)=$TL_FOLDER/MedDr__BF16"
    "Llama-3.2-Vision (11B)=$TL_FOLDER/Llama-3.2-11B-Vision-Instruct_bugfix-2eb7706"
    "MedGemma (27B)=$TL_FOLDER/medgemma-27b-it"
    "Qwen2.5-VL (32B)=$TL_FOLDER/Qwen2.5-VL-32B-Instruct"
)
AD_MODELS=(
    "MedVision-V0 (7B)=$AD_FOLDER/${MV_V0}"
    "GLM-4.6V-Flash (9B)=$AD_FOLDER/GLM-4.6V-Flash"
    "Gemma-4 (31B)=$AD_FOLDER/gemma-4-31B-it"
    "Qwen3-VL-Thinking (32B)=$AD_FOLDER/Qwen3-VL-32B-Thinking"
    "HealthGPT (14B)=$AD_FOLDER/HealthGPT-L14_bugfix-2eb7706"
    "Lingshu (32B)=$AD_FOLDER/lingshu-32b"
    "MedDr (40B)=$AD_FOLDER/MedDr__BF16"
    "LLaVA-OneVision (72B)=$AD_FOLDER/LLaVA-OneVision_bugfix-0a4c5e2"
    "Gemma-3 (27B)=$AD_FOLDER/gemma-3-27b-it"
    "MedGemma (4B)=$AD_FOLDER/medgemma-4b-it"
    "InternVL3 (38B)=$AD_FOLDER/InternVL3-38B_bugfix-2eb7706"
    "Qwen2.5-VL (7B)=$AD_FOLDER/Qwen2.5-VL-7B-Instruct"
    "GLM-4.6V (106B)=$AD_FOLDER/GLM-4.6V"
    "MiniMax-M3 (428B, int4)=$AD_FOLDER/MiniMax-M3-INT4"
    "MedGemma (27B)=$AD_FOLDER/medgemma-27b-it"
    "Qwen2.5-VL (32B)=$AD_FOLDER/Qwen2.5-VL-32B-Instruct"
    "HuatuoGPT-Vision (34B)=$AD_FOLDER/HuatuoGPT-Vision-34B_bugfix-2eb7706-wStopStrings"
    "Llama-3.2-Vision (11B)=$AD_FOLDER/Llama-3.2-11B-Vision-Instruct_bugfix-2eb7706"
)

PAGE_DIR="${PAGE_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io}"
PER_TARGET_DET="${PER_TARGET_DET:-2}"
PER_TARGET_TL="${PER_TARGET_TL:-2}"
PER_TARGET_AD="${PER_TARGET_AD:-2}"
# Optional hard cap on total cases per task. Unset (default) -> no cap: the per-target
# sweep covers every label. Set it to bound the figure count (some labels then dropped).
PER_TASK_MAX="${PER_TASK_MAX:-200}"
PER_TASK_MAX_ARGS=()
if [ -n "$PER_TASK_MAX" ]; then
    PER_TASK_MAX_ARGS=(--per_task_max "$PER_TASK_MAX")
fi
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
    --per_target_det "$PER_TARGET_DET" \
    --per_target_tl "$PER_TARGET_TL" \
    --per_target_ad "$PER_TARGET_AD" \
    --seed "$SEED" \
    --nonmedvision_topleft \
    "${PER_TASK_MAX_ARGS[@]}" \
    "${REMOVED_ARGS[@]}" \
    "${SKIP_ARGS[@]}"
