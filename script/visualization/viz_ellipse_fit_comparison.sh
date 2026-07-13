#!/bin/bash
# Randomly sample anisotropic TL cases and render image-space vs real-space
# ellipse-fit comparison figures (one 1x2 panel per case: no-scaling vs physical
# aspect). See viz_ellipse_fit_comparison.py and doc/ellipse-fitting-image-vs-real-space.md.
#
# Candidate pool = UNION of anisotropic (dataset, taskID) tasks across the
# ds_v1.0.0 Coronal + Sagittal pixel-size summaries in dataset-info/pixel_sizes__ds_v1.0.0/.
# Per-case test-split records come from the HF MedVision per-task configs, loaded
# OFFLINE from local data (reaches all 14 anisotropic tasks, incl. BraTS24-MET,
# HNTSMRG24-preRT, MSD-Colon/Lung/Pancreas — not just the axial-parquet subsets).
#
# Run in the MedVision conda env (needs nibabel + opencv + datasets + medvision_bm).
#
# Usage:
#   bash viz_ellipse_fit_comparison.sh
#   N=4 ORIENTATIONS="sagittal" bash viz_ellipse_fit_comparison.sh
#   SEED=7 MIN_DIV=35 bash viz_ellipse_fit_comparison.sh
#   OUT_DIR=/tmp/ellipse_fit bash viz_ellipse_fit_comparison.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# HF MedVision loader resolves to LOCAL data when these are set (no download).
export MedVision_DATA_DIR="${MedVision_DATA_DIR:-$MEDVISION_DIR/Data}"
export MedVision_PLANNER_VERSION="${MedVision_PLANNER_VERSION:-1.1.0}"
export MedVision_ACK_RELEASE='1.1.1'

N="${N:-10}"                               # figures per orientation
ORIENTATIONS="${ORIENTATIONS:-sagittal coronal}"
MIN_DIV="${MIN_DIV:-10}"                  # min image/real major-axis divergence (deg)
MIN_ANISO="${MIN_ANISO:-1.1}"             # min in-plane spacing ratio (matches py default)

# Seed: leave unset to use the python default (SEED from configs.py). Only pass
# --seed when SEED is explicitly provided.
SEED_ARGS=()
if [ -n "$SEED" ]; then
    SEED_ARGS=(--seed "$SEED")
fi

# Output dir: default is per-orientation under the script's output/ folder (the
# python default). Override OUT_DIR to send all orientations to one folder.
OUT_ARGS=()
if [ -n "$OUT_DIR" ]; then
    OUT_ARGS=(--out-dir "$OUT_DIR")
fi

for orient in $ORIENTATIONS; do
    echo "=== $orient ==="
    python "$SCRIPT_DIR/viz_ellipse_fit_comparison.py" \
        --orientation "$orient" \
        --n "$N" \
        --min-divergence "$MIN_DIV" \
        --min-anisotropy "$MIN_ANISO" \
        "${SEED_ARGS[@]}" \
        "${OUT_ARGS[@]}"
done
