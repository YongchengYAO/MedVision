#!/bin/bash
# ============================================================
# One-time setup for the BiomedParse ablation.
#
#   1. Clone microsoft/BiomedParse (v2 code) into third_party/, pinned to the
#      commit this study was run against.
#   2. Create the `biomedparse` conda env: upstream requirements, detectron2
#      (built from source), and the MedVision dataset package (medvision_ds).
#
# Usage:
#   bash setup.sh
#
# Overrides (environment variables):
#   ENV_NAME             conda env name            (default: biomedparse)
#   MedVision_DATA_DIR   MedVision data directory  (default: <repo>/Data)
# ============================================================
set -euo pipefail

ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ABLATION_DIR}/../../.." && pwd)"

UPSTREAM_URL="https://github.com/microsoft/BiomedParse.git"
UPSTREAM_COMMIT="e02096c03af0d79c6994ffc2d60a49eeb0361e1f"   # v2 branch, 2026-01-20
UPSTREAM_DIR="${ABLATION_DIR}/third_party/BiomedParse"
ENV_NAME="${ENV_NAME:-biomedparse}"
[ -f "${ABLATION_DIR}/scripts/_env.local.sh" ] && source "${ABLATION_DIR}/scripts/_env.local.sh"
export MedVision_DATA_DIR="${MedVision_DATA_DIR:-${REPO_ROOT}/Data}"

# ---------- 1. upstream BiomedParse (pinned) ----------
if [ ! -d "${UPSTREAM_DIR}/.git" ]; then
    echo "Cloning BiomedParse into ${UPSTREAM_DIR} ..."
    git clone "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
fi
git -C "${UPSTREAM_DIR}" checkout --quiet "${UPSTREAM_COMMIT}"
echo "BiomedParse @ $(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD)"

# ---------- 2. conda env ----------
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"

pip install -r "${ABLATION_DIR}/requirements.txt"

# detectron2 is not on PyPI; build once from source (needs nvcc, ~10-20 min).
if ! python -c "import detectron2" 2>/dev/null; then
    echo "Installing detectron2 from source ..."
    pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
fi

# medvision_ds: dataset loaders used by data preparation and evaluation.
PYTHONPATH="${REPO_ROOT}/src" python -m medvision_bm.benchmark.install_medvision_ds \
    --data_dir "${MedVision_DATA_DIR}"

# The dataset-package installer force-reinstalls its own dependencies and can lift
# huggingface-hub / packaging past what transformers 4.40 and lightning 2.3 accept —
# re-apply the pins so the env ends exactly as requirements.txt says.
pip install -r "${ABLATION_DIR}/requirements.txt"
pip check || true

echo "Setup complete. Activate with:  conda activate ${ENV_NAME}"
