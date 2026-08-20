#!/bin/bash
# Visualize per-sample model responses for the AD (Angle/Distance) benchmark task.
# Produces per-sample figures with Prompt / Response / GT+Metrics panels.
#
# Usage:
#   MODEL_DIR=<path> bash viz_ad_responses.sh
#
# Required:
#   MODEL_DIR=<path>             Single model directory (loops ${PARSED_DIRNAME}/*.jsonl)
#
# Optional:
#   PARSED_DIRNAME=<name>        Per-model parsed-records folder (default: llm-parsed_gemma-4-31b;
#                                use PARSED_DIRNAME=parsed for the strict-parse records)
#   OUTPUT_DIR=<path>            Base output directory for figures (default: <MEDVISION_DIR>/Figures/viz_responses/AD,
#                                suffixed with __${PARSED_DIRNAME} for non-default sources)
#   LIMIT_PER_JSONL=<N>          Max figures per JSONL file (default: unset = all)
#   DPI=<N>                      Figure save DPI, clamped to the 34MP cap (default: 100)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
PARSED_DIRNAME="${PARSED_DIRNAME:-llm-parsed_gemma-4-31b}"
if [ "$PARSED_DIRNAME" = "parsed" ]; then
    DEFAULT_OUTPUT_DIR="$MEDVISION_DIR/Figures/viz_responses/AD"
else
    DEFAULT_OUTPUT_DIR="$MEDVISION_DIR/Figures/viz_responses/AD__${PARSED_DIRNAME}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR is not set."
    echo "Usage: MODEL_DIR=<path/to/model_dir> bash $(basename "${BASH_SOURCE[0]}")"
    exit 1
fi
LIMIT_PER_JSONL="${LIMIT_PER_JSONL:-100}"
DPI="${DPI:-100}"

LIMIT_ARG=""
if [ -n "$LIMIT_PER_JSONL" ]; then
    LIMIT_ARG="--limit_per_jsonl $LIMIT_PER_JSONL"
fi

python "$SCRIPT_DIR/viz_ad_responses.py" \
    --model_dir "$MODEL_DIR" \
    --parsed_dirname "$PARSED_DIRNAME" \
    --output_dir "$OUTPUT_DIR" \
    --save_as_pdf \
    --dpi "$DPI" \
    $LIMIT_ARG
