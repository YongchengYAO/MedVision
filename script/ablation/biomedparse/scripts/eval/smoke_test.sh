#!/bin/bash
# Track A · smoke test — prepare → inference → eval on a fixed dataset list
# (default: 10 datasets spanning CT/MR/US/PET and both normalization branches),
# a few samples per subtask. Datasets absent from a task's list are skipped
# (e.g. only 6 of the 10 exist for T/L).
#
# Everything is written under smoke_test/<dataset>/ (git-ignored), so the real runs
# in data/, results/ and figures/ are never touched. Fails fast on the first error and prints
# the headline metrics of each task at the end.
#
# Prerequisites: `bash setup.sh` (conda env + pinned upstream), one GPU, and the
# MedVision data directory (MedVision_DATA_DIR, default <repo>/Data). The pretrained
# checkpoint is downloaded into models/ if missing.
#
# Usage:
#   bash scripts/eval/smoke_test.sh                                   # detect + tl, 10 datasets, 10 samples/subtask
#   DATASET=KiPA22 bash scripts/eval/smoke_test.sh                    # single dataset
#   DATASET=BraTS24,KiTS23 TASKS=tl GPU=1 bash scripts/eval/smoke_test.sh
#   DATASET= bash scripts/eval/smoke_test.sh                          # all datasets
#
# Re-runs reuse existing masks (inference skips them); delete smoke_test/ to start clean.
# The whole body below is one { } block so bash parses the ENTIRE file before
# executing anything — an edit to this file while a run is in flight (shared
# filesystem, another session) can then never corrupt that run.
{
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
set -euo pipefail

TASKS="${TASKS:-detect tl}"
DATASET="${DATASET-AMOS22,BraTS24,CAMUS,FeTA24,HNTSMRG24,KiPA22,KiTS23,MSD,OAIZIB-CM,autoPET-III}"   # comma-separated; DATASET= (empty) runs all
LIMIT_PER_SUBTASK="${LIMIT_PER_SUBTASK:-100}"
GPU="${GPU:-0}"
N_PROCESSES=8
SLICE_BATCH_SIZE=4
SMOKE_DIR="${ABLATION_DIR}/smoke_test/$(echo "${DATASET:-all}" | tr ',' '+')"

fail() { echo "SMOKE TEST FAILED — $*" >&2; exit 1; }
step() { echo; echo "==== [$1] $2 ($(date +%H:%M:%S)) ===="; }
count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l; }

ensure_pretrained_ckpt
[ -s "${PRETRAINED_CKPT}" ] || fail "pretrained checkpoint missing: ${PRETRAINED_CKPT}"
echo "smoke_test dir : ${SMOKE_DIR}"
echo "tasks          : ${TASKS}   dataset: ${DATASET:-<all>}   samples/subtask: ${LIMIT_PER_SUBTASK}   GPU: ${GPU}"

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
    RES_DIR="${SMOKE_DIR}/results/${TASK}/pretrained"
    FIG_DIR="${SMOKE_DIR}/figures/${TASK}/pretrained"

    step "${TASK} 1/3" "prepare test data → ${NPZ_DIR}"
    SECONDS=0
    PREP_ARGS=()
    if [ -n "${DATASET}" ]; then PREP_ARGS+=(--filter_dataset "${DATASET}"); fi
    python "${ABLATION_DIR}/src/prepare_test_data_${TASK}.py" \
        --tasks_json        "${TASKS_JSON}" \
        --output_dir        "${NPZ_DIR}" \
        --limit_per_subtask "${LIMIT_PER_SUBTASK}" \
        -p                  "${N_PROCESSES}" \
        "${PREP_ARGS[@]}"
    n_npz=$(count "${NPZ_DIR}" '*.npz')
    [ "${n_npz}" -gt 0 ] || fail "${TASK}: no .npz files produced"
    echo "-- ${n_npz} npz samples in ${SECONDS}s"

    step "${TASK} 2/3" "inference → ${RES_DIR}/seg_masks"
    SECONDS=0
    python "${ABLATION_DIR}/src/run_inference.py" \
        --checkpoint       "${PRETRAINED_CKPT}" \
        --npz_dir          "${NPZ_DIR}" \
        --seg_dir          "${RES_DIR}/seg_masks" \
        --gpu              "${GPU}" \
        --slice_batch_size "${SLICE_BATCH_SIZE}" \
        --skip_existing
    n_mask=$(count "${RES_DIR}/seg_masks" '*_pred_mask.nii.gz')
    [ "${n_mask}" -eq "${n_npz}" ] || fail "${TASK}: ${n_mask} masks for ${n_npz} npz samples"
    echo "-- ${n_mask} masks in ${SECONDS}s"

    step "${TASK} 3/3" "eval → ${RES_DIR}"
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
echo "==================== SMOKE TEST PASSED ===================="
for s in "${SUMMARIES[@]}"; do
    echo "--- ${s#${ABLATION_DIR}/} ---"
    head -8 "${s}"
done

exit 0
}
