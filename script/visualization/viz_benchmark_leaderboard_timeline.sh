#!/bin/bash
# Plot benchmark performance against model release date.
#
# One stacked panel per task, all sharing a single release-date axis that is
# labelled only on the bottom panel. Points are read from the three LLM-parsed
# per-task summary files. Colour = task, marker = model series, dotted line =
# same series over time; each point is annotated with its model name.
#
#   Detection  IoU      sample-weighted mean of the ANATOMY and T/L groups
#   T/L size   1/MRE    from the per-model "Weighted Average" line
#   Distance   1/MRE    from the "Distance" cross-dataset group row
#   Angle      1/MRE    from the "Angle" cross-dataset group row
#
# Usage:
#   bash script/visualization/viz_benchmark_leaderboard_timeline.sh
#   PYTHON=<path> FIG_NAME=<name> bash script/visualization/viz_benchmark_leaderboard_timeline.sh
#
# Optional:
#   PYTHON=<path>          Interpreter (default: python; needs matplotlib + torch)
#   AD_SUMMARY=<path>      A/D summary txt
#   TL_SUMMARY=<path>      T/L summary txt
#   DETECT_SUMMARY=<path>  Detection summary txt
#   FIG_DIR=<path>         Output directory (default: <REPO_DIR>/Figures)
#   FIG_NAME=<name>        Output filename (default: leaderboard_timeline.pdf)
#   FIG_WIDTH / FIG_HEIGHT Figure size in inches (default: 10.8 x 18)
#   LAYOUT=<RxC>           Panel grid, e.g. 4x1 (stacked, default) or 1x4 (one row)
#   LABEL_FONTSIZE=<N>     Point-label text size (default: 12)
#   LEGEND_COL=<N>         Columns in the bottom legend (default: 4)
#   LEGEND_FRAC=<f>        Figure-height fraction reserved for it (default: 0.10)
#   TITLE=<text>           Optional figure title
#   SAVE_AS_PNG=1          Write PNG instead of PDF
#   SAVE_AS_PDF=1          With SAVE_AS_PNG=1, write both formats
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Interpreter override, matching the PYTHON= convention used by the other drivers.
PYTHON="${PYTHON:-python}"

# Run medvision_bm against the repo source tree: it is not always installed, and a
# non-editable site-packages copy would otherwise silently shadow src/ with a stale build.
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# plot_utils needs matplotlib, and pulls torch through the package __init__. A bare
# `python` satisfies neither in some envs, so fail early with the fix instead of a traceback.
if ! "$PYTHON" -c "import medvision_bm.utils.plot_utils" >/dev/null 2>&1; then
    echo "Error: '$PYTHON' cannot import medvision_bm.utils.plot_utils (needs matplotlib + torch)." >&2
    echo "       Re-run with an env that has both, e.g. PYTHON=/path/to/env/bin/python bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

AD_SUMMARY="${AD_SUMMARY:-${REPO_DIR}/Results/MedVision-AD-v2-CoT/summary_AD_task__llm-parsed_gemma-4-31b.txt}"
TL_SUMMARY="${TL_SUMMARY:-${REPO_DIR}/Results/MedVision-TL-v2-CoT/summary_TL_task_filtered__llm-parsed_gemma-4-31b.txt}"
DETECT_SUMMARY="${DETECT_SUMMARY:-${REPO_DIR}/Results/MedVision-detect-v2/summary_detection_task__llm-parsed_gemma-4-31b.txt}"

FIG_DIR="${FIG_DIR:-${REPO_DIR}/Figures}"
FIG_NAME="${FIG_NAME:-leaderboard_timeline.pdf}"
FIG_WIDTH="${FIG_WIDTH:-10.8}"
FIG_HEIGHT="${FIG_HEIGHT:-18}"
LAYOUT="${LAYOUT:-4x1}"
LABEL_FONTSIZE="${LABEL_FONTSIZE:-12}"
LEGEND_COL="${LEGEND_COL:-4}"
LEGEND_FRAC="${LEGEND_FRAC:-0.10}"
TITLE="${TITLE:-}"

for f in "$AD_SUMMARY" "$TL_SUMMARY" "$DETECT_SUMMARY"; do
    [ -f "$f" ] || { echo "Error: summary file not found: $f" >&2; exit 1; }
done

PNG_ARG=""
[ -n "${SAVE_AS_PNG:-}" ] && PNG_ARG="--save_as_png"
PDF_ARG=""
[ -n "${SAVE_AS_PDF:-}" ] && PDF_ARG="--save_as_pdf"

"$PYTHON" "$REPO_DIR/script/visualization/viz_benchmark_leaderboard_timeline.py" \
    --ad_summary "$AD_SUMMARY" \
    --tl_summary "$TL_SUMMARY" \
    --detect_summary "$DETECT_SUMMARY" \
    --fig_dir "$FIG_DIR" \
    --fig_name "$FIG_NAME" \
    --fig_width "$FIG_WIDTH" \
    --fig_height "$FIG_HEIGHT" \
    --layout "$LAYOUT" \
    --label_fontsize "$LABEL_FONTSIZE" \
    --legend_col "$LEGEND_COL" \
    --legend_frac "$LEGEND_FRAC" \
    --title "$TITLE" \
    $PNG_ARG \
    $PDF_ARG
