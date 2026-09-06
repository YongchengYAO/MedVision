#!/bin/bash
# env_template.sh - shared environment for BiomedParse-ablation launchers.
#
# Adapted from the ablation folder's `scripts/_env.sh` (MedVision repository,
# `script/ablation/biomedparse/`). Source it from your own launcher AFTER setting
# TASK (detect | tl). It resolves the study paths, pins the MedVision annotation
# version, activates the conda env created by the ablation's setup.sh, and
# defines `ensure_pretrained_ckpt` (downloads biomedparse_v2.ckpt, ~4.2 GB).
#
# Prerequisites
#   - ABLATION_DIR : your copy of the ablation folder (contains src/, scripts/, setup.sh)
#   - conda with the env created by `bash "${ABLATION_DIR}/setup.sh"`   (skipped when DRY_RUN=1)
#   - huggingface-cli (installed with huggingface-hub) for the weights download
#
# Knobs (environment variables, all optional except ABLATION_DIR)
#   ABLATION_DIR               ablation folder                (REQUIRED - placeholder below)
#   REPO_ROOT                  MedVision checkout             (default: ${ABLATION_DIR}/../../..)
#   TASK                       detect | tl                    (default: detect)
#   ENV_NAME                   conda env                      (default: biomedparse)
#   BIOMEDPARSE_DIR            upstream BiomedParse checkout  (default: ${ABLATION_DIR}/third_party/BiomedParse)
#   MedVision_DATA_DIR         MedVision data directory       (default: ${REPO_ROOT}/Data)
#                              = the folder holding Datasets/, src/ and .downloaded_datasets.json
#   MedVision_PLANNER_VERSION  annotation version             (default: 1.0.0, as in the paper)
#   MedVision_ACK_RELEASE      T/L release acknowledgement    (default: 1.4.0, exported only when TASK=tl;
#                              must equal the latest release - bump when the dataset is re-released)
#   DRY_RUN=1                  print the resolved knobs, skip conda activation and downloads
#   Machine-specific values go in ${ABLATION_DIR}/scripts/_env.local.sh (git-ignored, sourced if present).
#
# Examples
#   ABLATION_DIR=/path/to/MedVision/script/ablation/biomedparse DRY_RUN=1 bash env_template.sh
#   TASK=tl ABLATION_DIR=/path/to/MedVision/script/ablation/biomedparse DRY_RUN=1 bash env_template.sh
#   # in a launcher:
#   TASK=detect; export ABLATION_DIR=/path/to/MedVision/script/ablation/biomedparse
#   source /path/to/env_template.sh
#   ensure_pretrained_ckpt
#   python "${ABLATION_DIR}/src/run_inference.py" --checkpoint "${PRETRAINED_CKPT}" \
#       --npz_dir "${ABLATION_DIR}/data/test_npz/${TASK}" \
#       --seg_dir "${ABLATION_DIR}/results/${TASK}/pretrained/seg_masks" --gpu 0 --slice_batch_size 4 --skip_existing
#
# Syntax check: bash -n env_template.sh

# ---------------------------------------------------------------- placeholders
ABLATION_DIR="${ABLATION_DIR:-/path/to/MedVision/script/ablation/biomedparse}"   # <-- set me
TASK="${TASK:-detect}"
DRY_RUN="${DRY_RUN:-0}"

# `return` when sourced, `exit` when executed directly.
_bp_is_sourced=0
if [ "${BASH_SOURCE[0]}" != "$0" ]; then _bp_is_sourced=1; fi

_bp_fail() {
    echo "env_template.sh: $*" >&2
    if [ "${_bp_is_sourced}" -eq 1 ]; then return 1; else exit 1; fi
}

if [ ! -d "${ABLATION_DIR}" ]; then
    if [ "${DRY_RUN}" = "1" ]; then
        echo "WARNING: ABLATION_DIR does not exist yet: ${ABLATION_DIR}" >&2
    else
        _bp_fail "ABLATION_DIR not found: ${ABLATION_DIR} (set ABLATION_DIR to your copy of script/ablation/biomedparse)" || return 1 2>/dev/null || exit 1
    fi
fi
case "${TASK}" in
    detect|tl) ;;
    *) _bp_fail "TASK must be 'detect' or 'tl' (got '${TASK}')" || return 1 2>/dev/null || exit 1 ;;
esac

# ------------------------------------------------------------------- paths
if [ -z "${REPO_ROOT:-}" ]; then
    REPO_ROOT="$(cd "${ABLATION_DIR}/../../.." 2>/dev/null && pwd || echo "${ABLATION_DIR}/../../..")"
fi
[ -f "${ABLATION_DIR}/scripts/_env.local.sh" ] && source "${ABLATION_DIR}/scripts/_env.local.sh"
ENV_NAME="${ENV_NAME:-biomedparse}"

export BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-${ABLATION_DIR}/third_party/BiomedParse}"
export MedVision_DATA_DIR="${MedVision_DATA_DIR:-${REPO_ROOT}/Data}"

# Dataset version pin - same convention as the benchmark launchers: every task pins the
# planner to 1.0.0; T/L additionally acknowledges the newest T/L annotation release.
export MedVision_PLANNER_VERSION="${MedVision_PLANNER_VERSION:-1.0.0}"
if [ "${TASK}" = "tl" ]; then
    export MedVision_ACK_RELEASE="${MedVision_ACK_RELEASE:-1.4.0}"
fi

# Task JSONs used by every launcher (prepare / eval)
case "${TASK}" in
    detect) TASKS_JSON="${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json" ;;
    tl)     TASKS_JSON="${REPO_ROOT}/tasks_list/tasks_MedVision-TL__train_SFT.json" ;;
esac

# Pretrained BiomedParse v2 weights (~4.2 GB), downloaded once into models/.
PRETRAINED_CKPT="${ABLATION_DIR}/models/biomedparse_v2.ckpt"
ensure_pretrained_ckpt() {
    if [ -s "${PRETRAINED_CKPT}" ]; then
        return 0
    fi
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[dry-run] would run: huggingface-cli download microsoft/BiomedParse biomedparse_v2.ckpt --local-dir $(dirname "${PRETRAINED_CKPT}")"
        return 0
    fi
    echo "Downloading biomedparse_v2.ckpt from HuggingFace ..."
    huggingface-cli download microsoft/BiomedParse biomedparse_v2.ckpt \
        --local-dir "$(dirname "${PRETRAINED_CKPT}")"
}

# ------------------------------------------------------------- conda env
if [ "${DRY_RUN}" = "1" ]; then
    cat <<EOF
[dry-run] resolved knobs (no conda activation, no downloads)
  ABLATION_DIR              = ${ABLATION_DIR}
  REPO_ROOT                 = ${REPO_ROOT}
  TASK                      = ${TASK}
  TASKS_JSON                = ${TASKS_JSON}
  ENV_NAME                  = ${ENV_NAME}
  BIOMEDPARSE_DIR           = ${BIOMEDPARSE_DIR}
  MedVision_DATA_DIR        = ${MedVision_DATA_DIR}
  MedVision_PLANNER_VERSION = ${MedVision_PLANNER_VERSION}
  MedVision_ACK_RELEASE     = ${MedVision_ACK_RELEASE:-<unset - only exported for TASK=tl>}
  PRETRAINED_CKPT           = ${PRETRAINED_CKPT} ($([ -s "${PRETRAINED_CKPT}" ] && echo present || echo missing))
EOF
else
    if ! command -v conda >/dev/null 2>&1; then
        _bp_fail "conda not found on PATH" || return 1 2>/dev/null || exit 1
    fi
    eval "$(conda shell.bash hook)"
    if ! conda activate "${ENV_NAME}" 2>/dev/null; then
        echo "conda env '${ENV_NAME}' not found - run setup.sh first" >&2
        if [ "${_bp_is_sourced}" -eq 1 ]; then return 1; else exit 1; fi
    fi
fi
