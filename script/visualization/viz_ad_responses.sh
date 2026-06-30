#!/bin/bash
# Visualize per-sample model responses for the AD (Angle/Distance) benchmark task.
# Produces per-sample figures with Prompt / Response / GT+Metrics panels.
#
# Usage:
#   MODEL_DIR=<path> bash viz_ad_responses.sh
#
# Required:
#   MODEL_DIR=<path>             Single model directory (loops parsed/*.jsonl)
#
# Optional:
#   OUTPUT_DIR=<path>            Base output directory for figures (default: <MEDVISION_DIR>/Figures/viz_responses/AD)
#   LIMIT_PER_JSONL=<N>          Max figures per JSONL file (default: unset = all)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$MEDVISION_DIR/Figures/viz_responses/AD}"

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is not set."
    echo "Usage: MODEL_DIR=<path/to/model_dir> bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi
LIMIT_PER_JSONL="${LIMIT_PER_JSONL:-}"

LIMIT_ARG=""
if [ -n "$LIMIT_PER_JSONL" ]; then
    LIMIT_ARG="--limit_per_jsonl $LIMIT_PER_JSONL"
fi

python "$SCRIPT_DIR/viz_ad_responses.py" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $LIMIT_ARG
