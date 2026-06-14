#!/bin/bash
# Used to generate samples for the case viewers in MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Export the API-model PILOT case viewer for the project webpage (medvision-vlm.github.io).
#
# A smaller "pilot study" comparison: frontier API-served VLMs evaluated on a reduced test
# subset, shown side by side with MedVision-V0 on the SAME samples. This wraps the shared
# exporter (export_webpage_cases.py) but writes to a SEPARATE figure folder + manifest so the
# main 13-model case viewer's PNGs and static/js/cases.js are never touched:
#   PNGs     -> <PAGE_DIR>/figure/cases-pilot/<model>/
#   manifest -> <PAGE_DIR>/static/js/cases-pilot.js   (task keys suffixed "-Pilot")
#
# Per task, every model's case list is the SAME samples in the SAME order (the exporter's
# shared-key intersection naturally restricts to the API model's smaller subset), so switching
# models in the viewer compares answers on one case.
#
# Run in the MedVision conda env (needs nibabel + matplotlib + medvision_bm).
#
# Usage:
#   # Full re-render (overwrites every pilot overlay PNG; default):
#   bash export_pilot_cases.sh
#
#   # Append/incremental: reuse PNGs already on disk, only draw NEW cases:
#   SKIP_EXISTING=1 bash export_pilot_cases.sh
#
#   PAGE_DIR=/path PER_DATASET_TL=20 bash export_pilot_cases.sh
#
# Environment-variable knobs (map to export_webpage_cases.py CLI flags):
#   PAGE_DIR             Project page repo root          -> --page_dir
#   PER_DATASET_TL       TL samples / dataset (20)       -> --per_dataset_tl
#   PER_TASK_MAX         Hard cap on samples / task      -> --per_task_max
#   SEED                 Sample-selection seed (1234)    -> --seed
#   REMOVED_SAMPLES_DIR  TL multi-cluster exclusion root -> --removed_samples_dir
#   SKIP_EXISTING=1      Skip re-rendering existing PNGs -> --skip_existing
#
# Hardcoded below (separate from the main viewer, not env-controlled):
#   --cases_dirname cases-pilot   PNGs -> <PAGE_DIR>/figure/cases-pilot/<model>/
#   --cases_js static/js/cases-pilot.js   separate manifest
#   --task_key_suffix=-Pilot      task keys -> "TL-Pilot" (own viewer section)
#   --nonmedvision_topleft        DUAL-ORIGIN mode for non-MedVision TL/AD: both a
#                                 top-left + lower-left overlay & per-origin metrics, with
#                                 an origin toggle in the viewer (V0 stays single-version).
# cases.js (here cases-pilot.js) is always rebuilt in full regardless of SKIP_EXISTING.
#
# IMPORTANT: SKIP_EXISTING trusts on-disk PNGs. If a model's JSONLs were re-generated since
# the last export, run WITHOUT it so stale overlays are redrawn.
#
# Edit the *_MODELS arrays below to add/remove API models ("DisplayName=/path/to/model_dir").
# Currently TL only (Tumor/Lesion Size); add DET_MODELS/AD_MODELS when API results exist.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$MEDVISION_DIR/Results"

# MedVision-V0 = the fullRFT (AD-TL-D, s250) dir the main viewer uses as "MedVision-V0 (7B)".
MV_V0="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250"

# Tumor/Lesion Size models for the pilot viewer. MedVision-V0 first (viewer default).
# Claude-Fable-5 and Gemini-3.1-Pro were evaluated via API on a smaller subset (100 samples
# per task config). To APPEND a new API model without re-rendering the existing models' PNGs,
# add its line below and run with SKIP_EXISTING=1 (see usage above).
TL_MODELS=(
	"MedVision-V0 (7B)=$RESULTS/MedVision-TL-v2-CoT/${MV_V0}"
	"Claude-Fable-5=$RESULTS/MedVision-TL-CoT/Claude-Fable-5"
	"Gemini-3.1-Pro=$RESULTS/MedVision-TL-CoT/Gemini-3.1-Pro"
)

PAGE_DIR="${PAGE_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io}"
PER_DATASET_TL="${PER_DATASET_TL:-20}"
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
    --tl_models "${TL_MODELS[@]}" \
    --page_dir "$PAGE_DIR" \
    --cases_dirname cases-pilot \
    --cases_js static/js/cases-pilot.js \
    --task_key_suffix="-Pilot" \
    --nonmedvision_topleft \
    --per_dataset_tl "$PER_DATASET_TL" \
    --per_task_max "$PER_TASK_MAX" \
    --seed "$SEED" \
    "${REMOVED_ARGS[@]}" \
    "${SKIP_ARGS[@]}"
