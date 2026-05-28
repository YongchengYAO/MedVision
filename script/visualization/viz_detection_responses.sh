#!/bin/bash
# Visualize per-sample model responses for the Detection (bounding box) benchmark task.
# Produces per-sample figures with Prompt / Response / GT+Metrics panels.
#
# Usage:
#   MODEL_DIR=<path> bash viz_detection_responses.sh
#
# Required:
#   MODEL_DIR=<path>             Single model directory (loops parsed/*.jsonl)
#
# Optional:
#   OUTPUT_DIR=<path>            Base output directory for figures (default: <MEDVISION_DIR>/Figures/viz_responses/detection)
#   LIMIT_PER_JSONL=<N>          Max figures per JSONL file (default: unset = all)
#   REMOVED_SAMPLES_DIR=<path>   Root dir of per-dataset removed-samples JSONs (default: unset = no filtering)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$MEDVISION_DIR/Figures/viz_responses/detection}"

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is not set."
    echo "Usage: MODEL_DIR=<path/to/model_dir> bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi
LIMIT_PER_JSONL="${LIMIT_PER_JSONL:-}"
REMOVED_SAMPLES_DIR="${REMOVED_SAMPLES_DIR:-}"

LIMIT_ARG=""
if [ -n "$LIMIT_PER_JSONL" ]; then
    LIMIT_ARG="--limit_per_jsonl $LIMIT_PER_JSONL"
fi

REMOVED_ARG=""
if [ -n "$REMOVED_SAMPLES_DIR" ]; then
    REMOVED_ARG="--removed_samples_dir $REMOVED_SAMPLES_DIR"
fi

python "$SCRIPT_DIR/viz_detection_responses.py" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $REMOVED_ARG \
    $LIMIT_ARG
