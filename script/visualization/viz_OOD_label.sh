#!/bin/bash
# Plot label clouds of "target @ modality" for the Detection and T/L tasks, one
# 2x2 figure: rows are the task (Detection, T/L), columns are the roster
# (in-distribution, target-OOD). The in-distribution rosters are the ones
# MedVision-V0 was post-trained on; the target-OOD rosters hold targets held out
# from that training.
#
# Targets are resolved from each roster JSON's task keys through the
# medvision_ds benchmark plans and reconciled with label_map_rename, so the
# clouds show fine-grained targets with each upstream dataset's naming variants
# merged. Word size carries no meaning - see the header of viz_label_cloud.py.
#
# Usage:
#   bash viz_label_cloud.sh
#   PYTHON=<path> bash viz_label_cloud.sh
#
# Override defaults via environment variables:
#   PYTHON=<path>       Interpreter (default: python; needs matplotlib + datasets)
#   TASKS_DIR=<path>    Roster JSON root (default: <MEDVISION_DIR>/tasks_list)
#   FIG_DIR=<path>      Output directory for figures (default: <MEDVISION_DIR>/Figures)
#   FIG_NAME=<name>     Output filename (default: fig_label_cloud.pdf)
#   CELL_WIDTH=<N>      Panel width in inches (default: 8)
#   CELL_HEIGHT=<N>     Detection-row panel height in inches (default: 8)
#   TL_CELL_HEIGHT=<N>  T/L-row panel height in inches (default: 2.6); the rows are
#                       sized separately because the T/L rosters hold far fewer targets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
TASKS_DIR="${TASKS_DIR:-$MEDVISION_DIR/tasks_list}"
FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"
FIG_NAME="${FIG_NAME:-fig_OOD_label.pdf}"
CELL_WIDTH="${CELL_WIDTH:-8}"
CELL_HEIGHT="${CELL_HEIGHT:-8}"
TL_CELL_HEIGHT="${TL_CELL_HEIGHT:-2.6}"

# Run medvision_bm against the repo source tree (see viz_radar_grid.sh).
export PYTHONPATH="${MEDVISION_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! "$PYTHON" -c "import medvision_bm.utils.plot_utils, medvision_ds" >/dev/null 2>&1; then
    echo "Error: '$PYTHON' cannot import medvision_bm.utils.plot_utils + medvision_ds." >&2
    echo "       Re-run with an env that has both, e.g. PYTHON=/path/to/env/bin/python bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/viz_OOD_label.py" \
    --detect_indist "$TASKS_DIR/tasks_MedVision-detect-CoT.json" \
    --detect_ood "$TASKS_DIR/OOD/tasks_MedVision-detect-CoT-taskOOD.json" \
    --tl_indist "$TASKS_DIR/tasks_MedVision-TL__train_SFT.json" \
    --tl_ood "$TASKS_DIR/OOD/tasks_MedVision-TL-CoT-taskOOD.json" \
    --fig_dir "$FIG_DIR" \
    --fig_name "$FIG_NAME" \
    --cell_width "$CELL_WIDTH" \
    --cell_height "$CELL_HEIGHT" \
    --tl_cell_height "$TL_CELL_HEIGHT" \
    --save_as_png \
    --save_as_pdf
