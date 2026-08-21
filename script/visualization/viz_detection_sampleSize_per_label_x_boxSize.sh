#!/bin/bash
# Plot detection metrics and sample-size distribution per label × box-to-image ratio group.
# Reads per-label CSVs from each model's ${PARSED_DIRNAME}/ directory and renders a
# multi-panel figure (Recall, Precision, F1 scatter + stacked sample-size bar chart).
# Generate those CSVs first with:
#   python -m medvision_bm.benchmark.analyze_detection_task_boxsize \
#       --task_dir <path> --parsed_dirname <name> --skip_model_wo_parsed_files
#
# Usage:
#   IN_DIR=<path>  bash viz_detection_sampleSize_per_label_x_boxSize.sh
#
# Required:
#   IN_DIR=<path>          Directory containing model subfolders (each with a
#                          ${PARSED_DIRNAME}/ subdirectory)
#
# Optional:
#   PARSED_DIRNAME=<name>  Per-model parsed-records folder (default: llm-parsed_gemma-4-31b;
#                          use PARSED_DIRNAME=parsed for the strict-parse records; non-default
#                          sources suffix the figure name with __${PARSED_DIRNAME})
#   OUT_DIR=<path>         Output directory for the figure (default: <MEDVISION_DIR>/Figures)
#   CONFIG_YAML=<path>     YAML file with model_display_name mapping
#                          (default: config-detect-sampleSize-per-label-boxSize.yaml
#                           next to this script in script/visualization/)
#   ANATOMY_LEVEL=1        Use anatomy-grouped label CSV instead of fine-grained label CSV

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

IN_DIR="${IN_DIR:-}"
PARSED_DIRNAME="${PARSED_DIRNAME:-llm-parsed_gemma-4-31b}"
OUT_DIR="${OUT_DIR:-$MEDVISION_DIR/Figures}"
CONFIG_YAML="${CONFIG_YAML:-$SCRIPT_DIR/config-detect-sampleSize-per-label-boxSize.yaml}"
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
    --parsed_dirname "$PARSED_DIRNAME" \
    --out_dir "$OUT_DIR" \
    --save_as_pdf \
    "${CONFIG_ARG[@]}" \
    $LEVEL_ARG
