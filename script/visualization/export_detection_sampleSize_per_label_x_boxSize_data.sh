#!/bin/bash
# Used to generate the detection box-size explorer on the MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Rebuilds <PAGE_DIR>/static/js/boxsize-data.js, the blob that static/js/boxsize.js draws as
# detection metrics per box-to-image ratio x clinical target — the interactive twin of
# `viz_detection_sampleSize_per_label_x_boxSize.sh` (fig_detection__metrics-boxSize__*.pdf),
# with both label granularities (anatomy group / fine label) in one blob.
#
# Reads the per-label CSVs from each model's ${PARSED_DIRNAME}/ directory. Generate those first:
#   python -m medvision_bm.benchmark.analyze_detection_task_boxsize \
#       --task_dir <path> --parsed_dirname <name> --skip_model_wo_parsed_files
#
# Usage:
#   bash export_detection_sampleSize_per_label_x_boxSize_data.sh
#   PARSED_DIRNAME=llm-parsed_gemma-4-31b bash export_detection_sampleSize_per_label_x_boxSize_data.sh
#
# Environment-variable knobs (all map to export_detection_sampleSize_per_label_x_boxSize_data.py CLI flags):
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

PYTHONPATH="$MEDVISION_DIR/src" python "$SCRIPT_DIR/export_detection_sampleSize_per_label_x_boxSize_data.py" "${ARGS[@]}"
