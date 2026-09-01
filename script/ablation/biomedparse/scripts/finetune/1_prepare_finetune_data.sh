#!/bin/bash
# Track B · step 1 — build the fine-tuning set (PNG images + masks + JSON).
#
# Loads the 28 axial Detection *_Train configs from HuggingFace and applies the
# same group split + shuffle(SEED).select(TRAIN_LIMIT) as the MedVision SFT
# pipeline, so BiomedParse is fine-tuned on exactly the samples used for
# MedVision-V0's SFT stage.
#
# Usage:
#   bash scripts/finetune/1_prepare_finetune_data.sh
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

TRAIN_LIMIT=110000
VAL_LIMIT=1000
N_PROCESSES=64

python "${ABLATION_DIR}/src/prepare_finetune_data.py" \
    --tasks_json  "${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json" \
    --output_dir  "${ABLATION_DIR}/data/finetune/detect" \
    --train_limit ${TRAIN_LIMIT} \
    --val_limit   ${VAL_LIMIT} \
    --processes   ${N_PROCESSES}
