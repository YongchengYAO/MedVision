#!/bin/bash
# Used to generate samples for the annotation-preview widget in MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Randomly samples landmark/biometry figure PNGs (A/D + T/L groups) from DATA_DIR and copies
# them into <PAGE_DIR>/figure/annot-preview/<group>/<dataset>/, regenerating
# <PAGE_DIR>/static/js/annot-preview-data.js. Detection figures are out of scope.
#
# Usage:
#   bash export_annotation_preview.sh
#   N_PER_FOLDER=30 bash export_annotation_preview.sh
#
# Environment-variable knobs (all map to export_annotation_preview.py CLI flags):
#   DATA_DIR        Root holding <dataset>/Landmarks-fig* folders -> --data_dir (repo's Data/Datasets)
#   PAGE_DIR         Project page repo root                        -> --page_dir
#   N_PER_FOLDER    Figures sampled from EACH figure folder (20)  -> --n_per_folder
#   SEED             Sample-selection seed (1234)                  -> --seed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$MEDVISION_DIR/Data/Datasets}"
PAGE_DIR="${PAGE_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io}"
N_PER_FOLDER="${N_PER_FOLDER:-20}"
SEED="${SEED:-1234}"

python "$SCRIPT_DIR/export_annotation_preview.py" \
    --data_dir "$DATA_DIR" \
    --page_dir "$PAGE_DIR" \
    --n_per_folder "$N_PER_FOLDER" \
    --seed "$SEED"
