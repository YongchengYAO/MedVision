#!/usr/bin/env bash
set -euo pipefail

# Regenerate dataset-info/all_tasks__ds_v{1.2.0,1.3.0,1.4.0}/.
#
# Run oldest-first: the counts are cached by (config, resolved annotation version),
# so the later versions reuse everything the earlier ones already streamed. Only the
# tumour/lesion plans (and the datasets each release adds) are genuinely new work.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data}"

# Counts come from the published dataset. Point DATASET_PATH at a local checkout of the
# dataset repo to count configs that have been added to MedVision.py but not pushed yet.
DATASET_PATH="${DATASET_PATH:-YongchengYAO/MedVision}"

export MedVision_FORCE_INSTALL_CODE="${MedVision_FORCE_INSTALL_CODE:-false}"

# A non-editable medvision_bm in site-packages silently shadows this repo's source, so a
# freshly edited helper is imported as its installed copy instead. Put src first.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for version in 1.2.0 1.3.0 1.4.0; do
    echo "=============== v${version} ==============="
    python "${REPO_ROOT}/script/misc/regen_all_tasks.py" \
        --version "${version}" --data_dir "${DATA_DIR}" --dataset_path "${DATASET_PATH}"
done
