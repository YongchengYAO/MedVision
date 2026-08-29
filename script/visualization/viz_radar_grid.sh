#!/bin/bash
# Plot the per-model radar-chart grids: one row per model, six radar cells per
# row (Detection Recall/Precision/F1, AD Angle/Distance MRE, TL MRE), each cell
# overlaying that model's per-sample violin + box plots on every spoke, with a
# label number-to-name mapping block below. The models are split over two
# figures (_part1/_part2). MedVision-V0 is excluded by default (set EXCLUDE=""
# to keep it), and both PDF and PNG are written.
#
# Data sources, configs, and filters match viz_radar_batch.sh's three all-model
# figures (config-detect-CoT / config-AD-CoT / config-TL-CoT).
#
# Usage:
#   bash viz_radar_grid.sh
#   PYTHON=<path> bash viz_radar_grid.sh
#
# Override defaults via environment variables:
#   PYTHON=<path>         Interpreter (default: python; needs matplotlib + torch)
#   RESULTS_DIR=<path>    Root results directory (default: <MEDVISION_DIR>/Results)
#   FIG_DIR=<path>        Output directory for figures (default: <MEDVISION_DIR>/Figures)
#   FIG_NAME=<name>       Output filename (default: fig_radar_grid.pdf)
#   PARSED_DIRNAME=<name> Per-model parsed-records folder (default: llm-parsed_gemma-4-31b;
#                         non-default sources suffix the figure name with __${PARSED_DIRNAME})
#   CELL_INCHES=<N>       Radar cell size in inches (default: 3)
#   EXCLUDE=<substring>   Drop models whose display name contains this
#                         (default: MedVision; EXCLUDE="" keeps every model)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON="${PYTHON:-python}"
RESULTS_DIR="${RESULTS_DIR:-$MEDVISION_DIR/Results}"
FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"
FIG_NAME="${FIG_NAME:-fig_radar_grid.pdf}"
PARSED_DIRNAME="${PARSED_DIRNAME:-llm-parsed_gemma-4-31b}"
CELL_INCHES="${CELL_INCHES:-3}"
EXCLUDE="${EXCLUDE-MedVision}"

# Run medvision_bm against the repo source tree (see viz_benchmark_leaderboard_timeline.sh).
export PYTHONPATH="${MEDVISION_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! "$PYTHON" -c "import medvision_bm.utils.plot_utils" >/dev/null 2>&1; then
    echo "Error: '$PYTHON' cannot import medvision_bm.utils.plot_utils (needs matplotlib + torch)." >&2
    echo "       Re-run with an env that has both, e.g. PYTHON=/path/to/env/bin/python bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

"$PYTHON" "$SCRIPT_DIR/viz_radar_grid.py" \
    --config_detect "$SCRIPT_DIR/config-detect-CoT.yaml" \
    --task_dir_detect "$RESULTS_DIR/MedVision-detect-v2" \
    --config_ad "$SCRIPT_DIR/config-AD-CoT.yaml" \
    --task_dir_ad "$RESULTS_DIR/MedVision-AD-v2-CoT" \
    --config_tl "$SCRIPT_DIR/config-TL-CoT.yaml" \
    --task_dir_tl "$RESULTS_DIR/MedVision-TL-v2-CoT" \
    --parsed_dirname "$PARSED_DIRNAME" \
    --fig_dir "$FIG_DIR" \
    --fig_name "$FIG_NAME" \
    --cell_inches "$CELL_INCHES" \
    --exclude_display "$EXCLUDE" \
    --save_as_png \
    --save_as_pdf
