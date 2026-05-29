#!/bin/bash
# Compile cross-model comparison grids for AD, TL, and Detection tasks using
# viz_compile_grid.py. Reads pre-generated per-sample subfigures from
# <FIG_DIR>/MedVision-<task>-* and writes compiled_*.png into <FIG_DIR>.
#
# Usage:
#   bash viz_compile_grid_batch.sh
#
# Override defaults via environment variables:
#   FIG_DIR=<path>    Base directory for subfigure inputs and compiled outputs
#                     (default: <MEDVISION_DIR>/Figures)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"

# RFT model folder names (differ by task: detection subfigures use the _CoT suffix)
RFT_MODEL="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250"
RFT_MODEL_DETECT="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250_CoT"

# ── AD ──────────────────────────────────────────────────────────────────────

# AD, RFT model
python "$SCRIPT_DIR/viz_compile_grid.py" \
    --dir_subfigures "$FIG_DIR/MedVision-AD-v2-CoT" \
    --limit_subfigures 24 \
    --dataset_as_row \
    --dataset_as_row_num_row_per_ds 2 \
    --dir_model "$RFT_MODEL" \
    --output "$FIG_DIR/compiled_AD_RFT_model_seed1234.png" \
    --seed 1234

# ── TL ──────────────────────────────────────────────────────────────────────

# TL, RFT model, dataset-as-col
python "$SCRIPT_DIR/viz_compile_grid.py" \
    --dir_subfigures "$FIG_DIR/MedVision-TL-v2-CoT" \
    --limit_subfigures 42 \
    --dir_model "$RFT_MODEL" \
    --output "$FIG_DIR/compiled_TL_RFT_model_seed1234.png" \
    --seed 1234 \
    --dataset_as_col

# TL, RFT model, dataset-as-row
python "$SCRIPT_DIR/viz_compile_grid.py" \
    --dir_subfigures "$FIG_DIR/MedVision-TL-v2-CoT" \
    --limit_subfigures 18 \
    --dir_model "$RFT_MODEL" \
    --output "$FIG_DIR/compiled_TL_RFT_model_seed1234-v2.png" \
    --seed 1234 \
    --dataset_as_row

# ── Detection ───────────────────────────────────────────────────────────────

# Detection, RFT model (18 datasets in detection task)
python "$SCRIPT_DIR/viz_compile_grid.py" \
    --dir_subfigures "$FIG_DIR/MedVision-detect-v2" \
    --limit_subfigures 52 \
    --dir_model "$RFT_MODEL_DETECT" \
    --output "$FIG_DIR/compiled_detect_RFT_model_seed1234.png" \
    --seed 1234 \
    --dataset_as_row \
    --dataset_as_row_num_panel 2
