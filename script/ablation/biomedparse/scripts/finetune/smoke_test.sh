#!/bin/bash
# Track B · smoke test — prepare-finetune-data → finetune → inference → eval on ONE
# dataset (default KiPA22) with tiny limits.
#
# This exercises the full Track B code path; it does NOT produce a meaningful model,
# and a single-dataset pool does NOT reproduce the identical-110k-sample guarantee of
# the real Track B step 1. Everything is written under smoke_test/<dataset>/
# (git-ignored): finetune data in data/finetune/detect/, the checkpoint (~4 GB, only
# last.ckpt is kept) in models/finetuned/, metrics in results/<task>/finetuned/.
# Test NPZ are shared with the Track A smoke test and prepared here if missing.
#
# Prerequisites: `bash setup.sh`, one GPU (training runs bf16-mixed), MedVision_DATA_DIR.
#
# Usage:
#   bash scripts/finetune/smoke_test.sh                      # KiPA22: finetune on detect, eval detect + tl
#   DATASET=KiTS23 EPOCHS=2 GPU=1 bash scripts/finetune/smoke_test.sh
#
# Re-runs reuse existing finetune data, checkpoint and masks; delete
# smoke_test/<dataset>/ to start clean.
# The whole body below is one { } block so bash parses the ENTIRE file before
# executing anything — an edit to this file while a run is in flight (shared
# filesystem, another session) can then never corrupt that run.
{
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
set -euo pipefail

DATASET="${DATASET-KiPA22}"        # one dataset keeps the test small; DATASET= (empty) runs all
TASKS="${TASKS:-detect tl}"        # tasks the fine-tuned checkpoint is evaluated on
LIMIT_PER_SUBTASK="${LIMIT_PER_SUBTASK:-10}"
TRAIN_LIMIT="${TRAIN_LIMIT:-32}"
VAL_LIMIT="${VAL_LIMIT:-8}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE=2
GPU="${GPU:-1}"
N_PROCESSES=8
SLICE_BATCH_SIZE=4
SMOKE_DIR="${ABLATION_DIR}/smoke_test/$(echo "${DATASET:-all}" | tr ',' '+')"
FT_DATA_DIR="${SMOKE_DIR}/data/finetune/detect"
FT_MODEL_DIR="${SMOKE_DIR}/models/finetuned"

fail() { echo "SMOKE TEST FAILED — $*" >&2; exit 1; }
step() { echo; echo "==== [$1] $2 ($(date +%H:%M:%S)) ===="; }
count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l; }

ensure_pretrained_ckpt
[ -s "${PRETRAINED_CKPT}" ] || fail "pretrained checkpoint missing: ${PRETRAINED_CKPT}"
echo "smoke_test dir : ${SMOKE_DIR}"
echo "dataset: ${DATASET:-<all>}   train/val: ${TRAIN_LIMIT}/${VAL_LIMIT}   epochs: ${EPOCHS}   eval tasks: ${TASKS}   GPU: ${GPU}"

PREP_FILTER=()
if [ -n "${DATASET}" ]; then PREP_FILTER=(--filter_dataset "${DATASET}"); fi

step "B 1/4" "prepare fine-tuning data → ${FT_DATA_DIR}"
SECONDS=0
if [ -s "${FT_DATA_DIR}/train.json" ]; then
    echo "-- reusing existing ${FT_DATA_DIR}"
else
    python "${ABLATION_DIR}/src/prepare_finetune_data.py" \
        --tasks_json  "${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json" \
        --output_dir  "${FT_DATA_DIR}" \
        --train_limit "${TRAIN_LIMIT}" \
        --val_limit   "${VAL_LIMIT}" \
        --processes   "${N_PROCESSES}" \
        "${PREP_FILTER[@]}"
fi
for f in train.json val.json; do [ -s "${FT_DATA_DIR}/${f}" ] || fail "missing/empty ${FT_DATA_DIR}/${f}"; done
for d in train train_mask val val_mask; do
    n=$(count "${FT_DATA_DIR}/${d}" '*.png'); [ "${n}" -gt 0 ] || fail "no PNGs in ${FT_DATA_DIR}/${d}"
done
echo "-- train: $(count "${FT_DATA_DIR}/train" '*.png') imgs | val: $(count "${FT_DATA_DIR}/val" '*.png') imgs in ${SECONDS}s"

step "B 2/4" "fine-tune (${EPOCHS} epoch(s), bs ${BATCH_SIZE}, 1 GPU) → ${FT_MODEL_DIR}"
SECONDS=0
CUDA_VISIBLE_DEVICES="${GPU}" python "${ABLATION_DIR}/src/finetune.py" \
    --data_dir    "${FT_DATA_DIR}" \
    --checkpoint  "${PRETRAINED_CKPT}" \
    --output_dir  "${FT_MODEL_DIR}" \
    --batch_size  "${BATCH_SIZE}" \
    --lr          1e-5 \
    --epochs      "${EPOCHS}" \
    --gpus        1 \
    --num_workers 2 \
    --save_top_k  0
CKPT="${FT_MODEL_DIR}/last.ckpt"
[ -s "${CKPT}" ] || fail "fine-tuned checkpoint missing: ${CKPT}"
echo "-- checkpoint $(du -h "${CKPT}" | cut -f1) in ${SECONDS}s"

SUMMARIES=()
for TASK in ${TASKS}; do
    case "${TASK}" in
        detect) TASKS_JSON="${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json"
                SUMMARY_TXT="summary_detection_task.txt"; SUMMARY_JSON="summary_metrics_detect_Task.json" ;;
        tl)     TASKS_JSON="${REPO_ROOT}/tasks_list/tasks_MedVision-TL__train_SFT.json"
                SUMMARY_TXT="summary_tl_task.txt";        SUMMARY_JSON="summary_metrics_tl_Task.json"
                # T/L pins an older annotation version → needs the release ACK (see _env.sh)
                export MedVision_ACK_RELEASE="${MedVision_ACK_RELEASE:-1.4.0}" ;;
        *)      fail "unknown task '${TASK}' (expected detect or tl)" ;;
    esac
    NPZ_DIR="${SMOKE_DIR}/data/test_npz/${TASK}"
    RES_DIR="${SMOKE_DIR}/results/${TASK}/finetuned"
    FIG_DIR="${SMOKE_DIR}/figures/${TASK}/finetuned"

    if [ "$(count "${NPZ_DIR}" '*.npz')" -eq 0 ]; then
        step "${TASK}" "prepare test data (shared with Track A smoke) → ${NPZ_DIR}"
        python "${ABLATION_DIR}/src/prepare_test_data_${TASK}.py" \
            --tasks_json        "${TASKS_JSON}" \
            --output_dir        "${NPZ_DIR}" \
            --limit_per_subtask "${LIMIT_PER_SUBTASK}" \
            -p                  "${N_PROCESSES}" \
            "${PREP_FILTER[@]}"
    fi
    n_npz=$(count "${NPZ_DIR}" '*.npz')
    [ "${n_npz}" -gt 0 ] || fail "${TASK}: no .npz files in ${NPZ_DIR}"

    step "B 3/4 ${TASK}" "inference with ${CKPT##*/} → ${RES_DIR}/seg_masks"
    SECONDS=0
    python "${ABLATION_DIR}/src/run_inference.py" \
        --checkpoint       "${CKPT}" \
        --npz_dir          "${NPZ_DIR}" \
        --seg_dir          "${RES_DIR}/seg_masks" \
        --gpu              "${GPU}" \
        --slice_batch_size "${SLICE_BATCH_SIZE}" \
        --skip_existing
    n_mask=$(count "${RES_DIR}/seg_masks" '*_pred_mask.nii.gz')
    [ "${n_mask}" -eq "${n_npz}" ] || fail "${TASK}: ${n_mask} masks for ${n_npz} npz samples"
    echo "-- ${n_mask} masks in ${SECONDS}s"

    step "B 4/4 ${TASK}" "eval → ${RES_DIR}"
    SECONDS=0
    python "${ABLATION_DIR}/src/eval_${TASK}.py" \
        --pred_dir   "${RES_DIR}/seg_masks" \
        --npz_dir    "${NPZ_DIR}" \
        --tasks_json "${TASKS_JSON}" \
        --output_dir "${RES_DIR}" \
        --fig_dir    "${FIG_DIR}"
    for f in "${SUMMARY_TXT}" "${SUMMARY_JSON}" \
             "eval_biomedparse_medvision_${TASK}_group_summary.csv" \
             "eval_biomedparse_medvision_${TASK}_results.csv" \
             "eval_biomedparse_medvision_${TASK}_metrics_dist.png"; do
        [ -s "${RES_DIR}/${f}" ] || fail "${TASK}: expected output missing or empty: ${f}"
    done
    n_fig=$(find "${FIG_DIR}" -name '*.png' 2>/dev/null | wc -l)
    [ "${n_fig}" -gt 0 ] || fail "${TASK}: no per-sample figures in ${FIG_DIR}"
    echo "-- eval outputs OK, ${n_fig} figures in ${SECONDS}s"
    SUMMARIES+=("${RES_DIR}/${SUMMARY_TXT}")
done

echo
echo "==================== TRACK B SMOKE TEST PASSED ===================="
echo "(metrics from a ${TRAIN_LIMIT}-sample, ${EPOCHS}-epoch model — not meaningful)"
for s in "${SUMMARIES[@]}"; do
    echo "--- ${s#${ABLATION_DIR}/} ---"
    head -8 "${s}"
done

exit 0
}
