#!/bin/bash
# Track A · step 2 — run the pretrained BiomedParse v2 on the prepared test set.
#
# Single-GPU only (the model has no DP/DDP support). SLICE_BATCH_SIZE trades
# speed for VRAM. Re-runs skip samples that already have a prediction.
#
# Usage:
#   bash scripts/eval/2_inference.sh                    # Detection, GPU 0
#   TASK=tl GPU=1 bash scripts/eval/2_inference.sh      # Tumor/Lesion size, GPU 1
TASK="${TASK:-detect}"          # detect | tl
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
GPU="${GPU:-0}"
SLICE_BATCH_SIZE=4

ensure_pretrained_ckpt

python "${ABLATION_DIR}/src/run_inference.py" \
    --checkpoint       "${PRETRAINED_CKPT}" \
    --npz_dir          "${ABLATION_DIR}/data/test_npz/${TASK}" \
    --seg_dir          "${ABLATION_DIR}/results/${TASK}/pretrained/seg_masks" \
    --gpu              "${GPU}" \
    --slice_batch_size ${SLICE_BATCH_SIZE} \
    --skip_existing
