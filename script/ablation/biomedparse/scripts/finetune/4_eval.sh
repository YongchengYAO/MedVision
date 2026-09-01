#!/bin/bash
# Track B · step 4 — MedVision metrics for the fine-tuned model.
# Same evaluation as Track A step 3, written to results/<task>/finetuned/ and
# figures/<task>/finetuned/.
#
# Usage:
#   bash scripts/finetune/4_eval.sh
#   TASK=tl bash scripts/finetune/4_eval.sh
TASK="${TASK:-detect}"          # detect | tl
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
MODEL=finetuned

python "${ABLATION_DIR}/src/eval_${TASK}.py" \
    --pred_dir   "${ABLATION_DIR}/results/${TASK}/${MODEL}/seg_masks" \
    --npz_dir    "${ABLATION_DIR}/data/test_npz/${TASK}" \
    --output_dir "${ABLATION_DIR}/results/${TASK}/${MODEL}" \
    --fig_dir    "${ABLATION_DIR}/figures/${TASK}/${MODEL}"
