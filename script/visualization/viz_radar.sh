#!/bin/bash
# Plot radar charts comparing model performance across multiple metrics.
# Reads per-task summary JSON files from each model's parsed/ directory and
# renders one polar subplot per metric with each model as a separate trace.
#
# Usage:
#   TASK_TYPE=<type> CONFIG_YAML=<path> TASK_DIR=<path> bash viz_radar.sh
#
# Required:
#   TASK_TYPE=<type>             Task type: AD, TL, or Detection
#   CONFIG_YAML=<path>           YAML file with model_display_name mapping
#   TASK_DIR=<path>              Directory containing model subdirectories
#
# Optional:
#   PARSED_DIRNAME=<name>        Per-model parsed-records folder (default: llm-parsed_gemma-4-31b;
#                                use PARSED_DIRNAME=parsed for the strict-parse records; non-default
#                                sources suffix the figure name with __${PARSED_DIRNAME})
#   FIG_DIR=<path>               Output directory for figures (default: <MEDVISION_DIR>/Figures)
#   FIG_NAME=<name>              Output filename (default: radar_<TASK_TYPE>.pdf)
#   METRICS_LIST=<list>          Space-separated metric names (default: "Precision F1")
#   VERBOSE_MODEL=<list>         Space-separated model names for violin overlay (default: unset)
#   SHOW_SCATTER=1               Overlay jittered scatter on violin (default: unset)
#   SHOW_LABEL_NAME=1            Add label number-to-name panel (default: unset)
#   RADAR_CELL_INCHES=<N>        Width per radar subplot in inches (default: 8)
#   LABEL_COL=<N>                Columns in the label panel (default: unset = auto)
#   LEGEND_COL=<N>               Columns in the model legend (default: unset = auto)
#
# Output formats (resolved by viz_radar.py):
#   - No flags (what viz_radar.sh and viz_radar_batch.sh do) → ["pdf"] — exactly
#     what the line-plots→PDF rule wants.
#   - --save_as_png → ["png"] only — an explicit override for when you need a
#     raster copy.
#   - --save_as_png --save_as_pdf → both files written, one per format, via the
#     extension swap in viz_radar.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TASK_TYPE="${TASK_TYPE:-}"
CONFIG_YAML="${CONFIG_YAML:-}"
TASK_DIR="${TASK_DIR:-}"
PARSED_DIRNAME="${PARSED_DIRNAME:-llm-parsed_gemma-4-31b}"
FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"
FIG_NAME="${FIG_NAME:-}"
METRICS_LIST="${METRICS_LIST:-}"
VERBOSE_MODEL="${VERBOSE_MODEL:-}"
SHOW_SCATTER="${SHOW_SCATTER:-}"
SHOW_LABEL_NAME="${SHOW_LABEL_NAME:-}"
RADAR_CELL_INCHES="${RADAR_CELL_INCHES:-}"
LABEL_COL="${LABEL_COL:-}"
LEGEND_COL="${LEGEND_COL:-}"

if [ -z "$TASK_TYPE" ] || [ -z "$CONFIG_YAML" ] || [ -z "$TASK_DIR" ]; then
    echo "Error: TASK_TYPE, CONFIG_YAML, and TASK_DIR are required."
    echo "  TASK_TYPE=AD CONFIG_YAML=<path> TASK_DIR=<path> bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi

FIG_NAME="${FIG_NAME:-radar_${TASK_TYPE}.pdf}"

METRICS_ARG=()
if [ -n "$METRICS_LIST" ]; then
    read -ra _metrics <<<"$METRICS_LIST"
    METRICS_ARG=(--metrics_list "${_metrics[@]}")
fi

VERBOSE_ARG=()
if [ -n "$VERBOSE_MODEL" ]; then
    read -ra _verbose <<<"$VERBOSE_MODEL"
    VERBOSE_ARG=(--verbose_model "${_verbose[@]}")
fi

SCATTER_ARG=""
[ -n "$SHOW_SCATTER" ] && SCATTER_ARG="--show_scatter"

LABEL_NAME_ARG=""
[ -n "$SHOW_LABEL_NAME" ] && LABEL_NAME_ARG="--show_label_name"

RADAR_INCHES_ARG=""
[ -n "$RADAR_CELL_INCHES" ] && RADAR_INCHES_ARG="--radar_cell_inches $RADAR_CELL_INCHES"

LABEL_COL_ARG=""
[ -n "$LABEL_COL" ] && LABEL_COL_ARG="--label_col $LABEL_COL"

LEGEND_COL_ARG=""
[ -n "$LEGEND_COL" ] && LEGEND_COL_ARG="--legend_col $LEGEND_COL"

python "$SCRIPT_DIR/viz_radar.py" \
    --task_type "$TASK_TYPE" \
    --config_yaml "$CONFIG_YAML" \
    --task_dir "$TASK_DIR" \
    --parsed_dirname "$PARSED_DIRNAME" \
    --fig_dir "$FIG_DIR" \
    --fig_name "$FIG_NAME" \
    "${METRICS_ARG[@]}" \
    "${VERBOSE_ARG[@]}" \
    $SCATTER_ARG \
    $LABEL_NAME_ARG \
    $RADAR_INCHES_ARG \
    $LABEL_COL_ARG \
    $LEGEND_COL_ARG
