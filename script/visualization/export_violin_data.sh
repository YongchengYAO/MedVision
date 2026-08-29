#!/bin/bash
# Used to generate the radar violin overlays in the MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Rebuilds <PAGE_DIR>/static/js/violin-data.js, the per-spoke distribution blob that
# static/js/radar.js draws as a violin + box plot along each radar spoke — the interactive
# twin of `viz_radar.py --verbose_model`, for every model rather than one named one.
#
# Re-run this whenever radar-data.js is regenerated: the two blobs are read together and the
# violin spokes are built from the radar's own spokes, so they must come from the same
# Results/ state. Covers every radar on the page, reading the same configs as
# export_radar_data.py (config-{detect,TL,AD}-CoT.yaml and config-TL-pilot-CoT.yaml).
#
# Slow by nature: it re-reads every *_samples_*.jsonl record (~1.5 GB for Detection) to
# recover the distribution behind each radar point. Expect several minutes.
#
# Usage:
#   bash export_violin_data.sh
#   PARSED_DIRNAME=llm-parsed_gemma-4-31b bash export_violin_data.sh
#
# Environment-variable knobs (all map to export_violin_data.py CLI flags):
#   PAGE_DIR         Project page repo root                    -> --page_dir
#   RESULTS_DIR      MedVision Results/ directory              -> --results_dir
#   PARSED_DIRNAME   Per-model subdir to read from ("parsed")  -> --parsed_dirname
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
[ -n "$OUT" ] && ARGS+=(--out "$OUT")

PYTHONPATH="$MEDVISION_DIR/src" python "$SCRIPT_DIR/export_violin_data.py" "${ARGS[@]}"
