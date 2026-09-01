#!/bin/bash
# Track B · step 3 — run a fine-tuned checkpoint on the prepared test set.
#
# The study evaluates the detection-fine-tuned model on BOTH tasks (TASK=detect
# and TASK=tl); there is no T/L-specific fine-tuning. CHECKPOINT defaults to
# last.ckpt — pick a specific epoch (e.g. the best val_loss) to match the paper.
#
# Usage:
#   bash scripts/finetune/3_inference.sh
#   TASK=tl bash scripts/finetune/3_inference.sh
#   CHECKPOINT=models/finetuned-detect/biomedparse_medvision_epoch=03_val_loss=0.4460.ckpt \
#       bash scripts/finetune/3_inference.sh
TASK="${TASK:-detect}"          # detect | tl
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
GPU="${GPU:-0}"
SLICE_BATCH_SIZE=4
CHECKPOINT="${CHECKPOINT:-${ABLATION_DIR}/models/finetuned-detect/last.ckpt}"

python "${ABLATION_DIR}/src/run_inference.py" \
    --checkpoint       "${CHECKPOINT}" \
    --npz_dir          "${ABLATION_DIR}/data/test_npz/${TASK}" \
    --seg_dir          "${ABLATION_DIR}/results/${TASK}/finetuned/seg_masks" \
    --gpu              "${GPU}" \
    --slice_batch_size ${SLICE_BATCH_SIZE} \
    --skip_existing
