#!/bin/bash
# Track A · step 3 — MedVision metrics for the pretrained model.
#   detect : bounding box fitted to the predicted mask → IoU, F1, Precision, Recall, MAE
#   tl     : ellipse fitted to the predicted mask     → major/minor axis MAE, MRE
#
# Metrics go to results/<task>/pretrained/, per-sample figures to figures/<task>/pretrained/.
#
# Usage:
#   bash scripts/eval/3_eval.sh
#   TASK=tl bash scripts/eval/3_eval.sh
TASK="${TASK:-detect}"          # detect | tl
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
MODEL=pretrained

python "${ABLATION_DIR}/src/eval_${TASK}.py" \
    --pred_dir   "${ABLATION_DIR}/results/${TASK}/${MODEL}/seg_masks" \
    --npz_dir    "${ABLATION_DIR}/data/test_npz/${TASK}" \
    --output_dir "${ABLATION_DIR}/results/${TASK}/${MODEL}" \
    --fig_dir    "${ABLATION_DIR}/figures/${TASK}/${MODEL}"
