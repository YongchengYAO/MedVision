#!/bin/bash
# Seed the MedVision-V0 demo gallery: write one 504x504 input.png per selected
# benchmark sample plus an examples.json (task metadata + scaledPS spacing + GT
# coords as relative [0,1]) into the demo repo's examples/ directory.
#
# Only the MedVision-V0 model is exported (the demo runs that one model live);
# unlike export_webpage_cases.sh there are no multi-model arrays.
#
# Run in the MedVision conda env (needs nibabel + medvision_bm).
#
# Usage:
#   bash export_demo_gallery.sh
#   OUT_DIR=/path/to/demo/examples PER_SUBTASK=2 bash export_demo_gallery.sh
#   SEED=42 bash export_demo_gallery.sh        # override; default = 1234 (webpage seed)
#   REMOVED_SAMPLES_DIR= bash export_demo_gallery.sh   # disable TL removed-sample filtering
#
# Missing model dirs are skipped gracefully (that task just gets no cases).
# TL removed-sample filtering is ON by default (REMOVED_SAMPLES_DIR -> dataset root); it drops
# the v1.0.0 -> v1.1.0 multi-cluster T/L samples the benchmark excludes (set empty to disable).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$MEDVISION_DIR/Results"

# Per-task eval folders + the MedVision-V0 (fullRFT, AD-TL-D, s250) model dir name.
# Detection uses the _CoT-suffixed folder; TL/AD use the bare name (matches
# export_webpage_cases.sh).
DET_FOLDER="$RESULTS/MedVision-detect-v2"
TL_FOLDER="$RESULTS/MedVision-TL-v2-CoT"
AD_FOLDER="$RESULTS/MedVision-AD-v2-CoT"
MV_V0="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250"

DET_DIR="$DET_FOLDER/${MV_V0}_CoT"
TL_DIR="$TL_FOLDER/${MV_V0}"
AD_DIR="$AD_FOLDER/${MV_V0}"

# Output goes into the standalone demo repo's examples/ folder.
OUT_DIR="${OUT_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-v0-demo/examples}"
# Samples drawn at random (seeded) from EACH subtask (one main results JSONL = one
# dataset/task-config/plane), so every subtask is represented. Total per task =
# PER_SUBTASK x (number of subtasks): detection ~28, TL ~10, AD ~5 subtasks.
PER_SUBTASK="${PER_SUBTASK:-20}"

# Seed: leave unset to use the python default (1234, matching export_webpage_cases so the
# demo gallery shares the project-page case viewer's selection seed). Only pass --seed when
# SEED is explicitly provided.
SEED_ARGS=()
if [ -n "$SEED" ]; then
    SEED_ARGS=(--seed "$SEED")
fi

# TL removed-sample filtering (matches export_webpage_cases.sh / summarize_TL_task.py): drop
# the v1.0.0 -> v1.1.0 multi-cluster T/L samples before selection so the gallery only draws
# from the scored v1.1.0 TL set. Set to your dataset root; unset -> no filtering.
REMOVED_SAMPLES_DIR="${REMOVED_SAMPLES_DIR:-/mnt/vincent-pvc-rwm/Github/MedVision/Data/Datasets}"
REMOVED_ARGS=()
if [ -n "$REMOVED_SAMPLES_DIR" ]; then
    REMOVED_ARGS=(--removed_samples_dir "$REMOVED_SAMPLES_DIR")
fi

for d in "$DET_DIR" "$TL_DIR" "$AD_DIR"; do
    [ -d "$d" ] || echo "WARNING: model dir not found, that task will be skipped: $d"
done

python "$SCRIPT_DIR/export_demo_gallery.py" \
    --det_dir "$DET_DIR" \
    --tl_dir "$TL_DIR" \
    --ad_dir "$AD_DIR" \
    --out_dir "$OUT_DIR" \
    --per_subtask "$PER_SUBTASK" \
    "${SEED_ARGS[@]}" \
    "${REMOVED_ARGS[@]}"
