#!/bin/bash
# Sourced by every launcher in scripts/eval/ and scripts/finetune/ — not run directly.
# Resolves the study paths, pins the dataset version, and activates the conda env
# created by setup.sh. Launchers set TASK (detect | tl) BEFORE sourcing this file.
#
# Overrides (environment variables):
#   ENV_NAME                   conda env                      (default: biomedparse)
#   BIOMEDPARSE_DIR            upstream BiomedParse checkout  (default: third_party/BiomedParse)
#   MedVision_DATA_DIR         MedVision data directory       (default: <repo>/Data)
#                              = the folder holding Datasets/, src/ and .downloaded_datasets.json
#   MedVision_PLANNER_VERSION  MedVision annotation version   (default: 1.0.0, as in the paper)
# Machine-specific values go in scripts/_env.local.sh (git-ignored, sourced if present).

ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ABLATION_DIR}/../../.." && pwd)"
[ -f "${ABLATION_DIR}/scripts/_env.local.sh" ] && source "${ABLATION_DIR}/scripts/_env.local.sh"
ENV_NAME="${ENV_NAME:-biomedparse}"

export BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-${ABLATION_DIR}/third_party/BiomedParse}"
export MedVision_DATA_DIR="${MedVision_DATA_DIR:-${REPO_ROOT}/Data}"

# Dataset version pin — same convention as script/benchmark-*/: every task pins the
# planner to 1.0.0; T/L additionally acknowledges the newest T/L annotation release
# (only the T/L annotations changed after 1.0.0, so medvision_ds asks for an explicit ACK;
# the value must equal the latest release — bump it when the dataset is re-released).
export MedVision_PLANNER_VERSION="${MedVision_PLANNER_VERSION:-1.0.0}"
if [ "${TASK:-detect}" = "tl" ]; then
    export MedVision_ACK_RELEASE="${MedVision_ACK_RELEASE:-1.4.0}"
fi

eval "$(conda shell.bash hook)"
if ! conda activate "${ENV_NAME}" 2>/dev/null; then
    echo "conda env '${ENV_NAME}' not found — run setup.sh first" >&2
    exit 1
fi

# Pretrained BiomedParse v2 weights (~4.2 GB), downloaded once into models/.
PRETRAINED_CKPT="${ABLATION_DIR}/models/biomedparse_v2.ckpt"
ensure_pretrained_ckpt() {
    if [ ! -s "${PRETRAINED_CKPT}" ]; then
        echo "Downloading biomedparse_v2.ckpt from HuggingFace ..."
        huggingface-cli download microsoft/BiomedParse biomedparse_v2.ckpt \
            --local-dir "$(dirname "${PRETRAINED_CKPT}")"
    fi
}
