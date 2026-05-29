#!/bin/bash
# Plot detection metrics and sample-size distribution per label × box-to-image ratio group.
# Reads per-label CSVs from each model's parsed/ directory and renders a multi-panel figure
# (Recall, Precision, F1 scatter + stacked sample-size bar chart).
#
# Usage:
#   IN_DIR=<path>  bash viz_detection_sampleSize_per_label_x_boxSize.sh
#
# Required:
#   IN_DIR=<path>          Directory containing model subfolders (each with a parsed/ subdirectory)
#
# Optional:
#   OUT_DIR=<path>         Output directory for the figure (default: <MEDVISION_DIR>/Figures)
#   CONFIG_YAML=<path>     YAML file with model_display_name mapping
#                          (default: sibling config-detect-sampleSize-per-label-boxSize.yaml
#                           next to the Python script in src/medvision_bm/benchmark/)
#   ANATOMY_LEVEL=1        Use anatomy-grouped label CSV instead of fine-grained label CSV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

IN_DIR="${IN_DIR:-}"
OUT_DIR="${OUT_DIR:-$MEDVISION_DIR/Figures}"
CONFIG_YAML="${CONFIG_YAML:-}"
ANATOMY_LEVEL="${ANATOMY_LEVEL:-}"

if [ -z "$IN_DIR" ]; then
    echo "Error: IN_DIR is required."
    echo "  IN_DIR=<path/to/task_dir> bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi

CONFIG_ARG=()
[ -n "$CONFIG_YAML" ] && CONFIG_ARG=(--config "$CONFIG_YAML")

LEVEL_ARG=""
[ -n "$ANATOMY_LEVEL" ] && LEVEL_ARG="--anatomy_level"

python -m medvision_bm.benchmark.viz_detection_sampleSize_per_label_x_boxSize \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    "${CONFIG_ARG[@]}" \
    $LEVEL_ARG
