#!/bin/bash
# Used to generate the detection box-ratio curves on the MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Rebuilds <PAGE_DIR>/static/js/boxratio-data.js, the blob that static/js/boxratio.js draws as
# detection metrics vs box-to-image ratio, one line per model against the random-box baseline —
# the interactive twin of metrics_boxImgRatio-dotline.pdf. It mounts directly BELOW the box-size
# explorer (boxsize.js) on index.md: same question, once per target and once in aggregate.
#
# Reads the per-ratio summaries written by run_analysis.sh. Generate those first:
#   bash script/analyze/detection--target-size/run_analysis.sh \
#       --task_dir <path> --parsed_dirname <name> --skip_model_wo_parsed_files
# That step also produces the random_detection/ baseline this figure plots.
#
# Usage:
#   bash export_detection_performance_per_boxImgRatio_data.sh
#   PARSED_DIRNAME=llm-parsed_gemma-4-31b bash export_detection_performance_per_boxImgRatio_data.sh
#
# Environment-variable knobs (all map to export_detection_performance_per_boxImgRatio_data.py CLI flags):
#   PAGE_DIR         Project page repo root                    -> --page_dir
#   RESULTS_DIR      MedVision Results/ directory              -> --results_dir
#   PARSED_DIRNAME   Per-model subdir to read from ("parsed")  -> --parsed_dirname
#   CONFIG_YAML      YAML with the model_display_name mapping  -> --config
#   OUT              Override output JS path                   -> --out

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PAGE_DIR="${PAGE_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io}"
RESULTS_DIR="${RESULTS_DIR:-$MEDVISION_DIR/Results}"
PARSED_DIRNAME="${PARSED_DIRNAME:-parsed}"

ARGS=(
    --page_dir "$PAGE_DIR"
    --results_dir "$RESULTS_DIR"
    --parsed_dirname "$PARSED_DIRNAME"
)
[ -n "$CONFIG_YAML" ] && ARGS+=(--config "$CONFIG_YAML")
[ -n "$OUT" ] && ARGS+=(--out "$OUT")

PYTHONPATH="$MEDVISION_DIR/src" python "$SCRIPT_DIR/export_detection_performance_per_boxImgRatio_data.py" "${ARGS[@]}"
