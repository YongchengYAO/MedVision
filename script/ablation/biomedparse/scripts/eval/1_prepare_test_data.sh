#!/bin/bash
# Track A · step 1 — export the MedVision test set to BiomedParse .npz format.
#
# Loads the *_Test configs from HuggingFace in native order and keeps the first
# LIMIT_PER_SUBTASK rows per subtask — the same samples the VLM benchmark sees.
#
# Usage:
#   bash scripts/eval/1_prepare_test_data.sh            # Detection
#   TASK=tl bash scripts/eval/1_prepare_test_data.sh    # Tumor/Lesion size
TASK="${TASK:-detect}"          # detect | tl
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
LIMIT_PER_SUBTASK=1000
N_PROCESSES=32

case "${TASK}" in
    detect) TASKS_JSON="${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json" ;;
    tl)     TASKS_JSON="${REPO_ROOT}/tasks_list/tasks_MedVision-TL__train_SFT.json" ;;
    *)      echo "TASK must be 'detect' or 'tl'" >&2; exit 1 ;;
esac

python "${ABLATION_DIR}/src/prepare_test_data_${TASK}.py" \
    --tasks_json        "${TASKS_JSON}" \
    --output_dir        "${ABLATION_DIR}/data/test_npz/${TASK}" \
    --limit_per_subtask ${LIMIT_PER_SUBTASK} \
    -p                  ${N_PROCESSES}
