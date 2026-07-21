#!/bin/bash
# Visualize MedVision TL task predictions (model-predicted axes + GT axes + mask contour).
#
# Usage:
#   TASK_DIR=<path>  bash viz_tl_axes.sh
#   MODEL_DIR=<path> bash viz_tl_axes.sh
#
# Required (mutually exclusive — set exactly one):
#   TASK_DIR=<path>              Task folder containing model subdirectories
#   MODEL_DIR=<path>             Single model directory
#
# Optional:
#   FIG_DIR=<path>               Base output directory for figures (default: <MEDVISION_DIR>/Figures)
#   LIMIT_PER_JSONL=<N>          Max samples per JSONL file (default: unset = all)
#   REMOVED_SAMPLES_DIR=<path>   Root dir of per-dataset removed-samples JSONs (default: unset = no filtering)
#
# Output formats (resolved by viz_tl_axes.py):
#   - No flags (what this script does) → ["png"] — default.
#   - --save_as_pdf → ["pdf"] only.
#   - --save_as_png --save_as_pdf → both files written, one per format.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

TASK_DIR="${TASK_DIR:-}"
MODEL_DIR="${MODEL_DIR:-}"
FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"
LIMIT_PER_JSONL="${LIMIT_PER_JSONL:-10}"
REMOVED_SAMPLES_DIR="${REMOVED_SAMPLES_DIR:-}"

if [ -n "$TASK_DIR" ] && [ -n "$MODEL_DIR" ]; then
    echo "Error: set only one of TASK_DIR or MODEL_DIR, not both."
    exit 1
elif [ -z "$TASK_DIR" ] && [ -z "$MODEL_DIR" ]; then
    echo "Error: set TASK_DIR or MODEL_DIR."
    echo "  TASK_DIR=<path/to/task_dir>   bash $(basename "${BASH_SOURCE[0]}")"
    echo "  MODEL_DIR=<path/to/model_dir> bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi

LIMIT_ARG=""
if [ -n "$LIMIT_PER_JSONL" ]; then
    LIMIT_ARG="--limit-per-jsonl $LIMIT_PER_JSONL"
fi

REMOVED_ARG=""
if [ -n "$REMOVED_SAMPLES_DIR" ]; then
    REMOVED_ARG="--removed_samples_dir $REMOVED_SAMPLES_DIR"
fi

INPUT_ARGS=()
[ -n "$TASK_DIR" ] && INPUT_ARGS+=(--task_dir "$TASK_DIR")
[ -n "$MODEL_DIR" ] && INPUT_ARGS+=(--model_dir "$MODEL_DIR")

python "$SCRIPT_DIR/viz_tl_axes.py" \
    "${INPUT_ARGS[@]}" \
    --fig_dir "$FIG_DIR" \
    --save_as_pdf \
    $REMOVED_ARG \
    $LIMIT_ARG
