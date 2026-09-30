#!/bin/bash
# Group T/L task metrics by ground-truth major axis length (mm).
# All arguments are forwarded to analyze_TL_by_major_axis.py; see its docstring.
#
# Models must be parsed first:
#   python -m medvision_bm.benchmark.parse_outputs --task_type TL --model_dir <model_dir>
#
# Example (same model on the pre-1.4.0 vs v1.4.0 test set, common datasets only):
#   bash run_analysis.sh \
#       --model_dir Results/MedVision-TL-CoT/Qwen2.5VL-SFT-v140-s16000 \
#                   Results/MedVision-TL-CoT/Qwen2.5VL-SFT-v140-s16000--v140-testset \
#       --datasets autoPET-III BraTS24 HNTSMRG24 KiPA22 \
#       2>&1 | tee tl_by_major_axis__SFT-v140-s16000.log

#
# Uses the medvision_bm and medvision_ds installed in the active environment:
#   python -m medvision_bm.benchmark.install_medvision_ds --data_dir <repo>/Data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# summarize_TL_task imports the vendored lmms_eval (and its deps, e.g. sqlitedict).
# Install only when missing: the installer uses --force-reinstall, which would
# otherwise upgrade medvision_bm's pinned deps (torch, accelerate, ...) on every run.
if ! python -m pip show lmms_eval >/dev/null 2>&1; then
    python -m medvision_bm.benchmark.install_vendored_lmms_eval
fi

python "${SCRIPT_DIR}/analyze_TL_by_major_axis.py" "$@"
