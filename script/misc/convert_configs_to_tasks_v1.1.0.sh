#!/usr/bin/env bash
set -euo pipefail

# Regenerate the standard eval task-list JSONs from the config CSV.
# Counts are computed by streaming each config from the HF MedVision dataset.

export MedVision_PLANNER_VERSION='1.1.0'
export MedVision_ACK_RELEASE='1.1.1'

# Resolve the repo root from this script's location (<repo>/script/misc/<this>.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Paths are overridable via the environment; defaults assume the standard layout.
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data}"
CSV="${CSV:-${REPO_ROOT}/docs/dataset-configs/ConfigurationsList_All.csv}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/tasks_list/all_tasks__ds_v1.1.0}"

run() { python -m medvision_bm.utils.configs_to_tasks --data_dir "${DATA_DIR}" --configs_csv "${CSV}" "$@"; }

# Task list for benchmarking (test split)
# ===============================================
# Detection (BoxSize -> BoxCoordinate), Axial, test, CoT
run --families BoxSize --planes Axial --split test --cot --out "${OUT_DIR}/tasks_MedVision-detect-CoT__Axial__Test.json"

# Detection (BoxSize -> BoxCoordinate), Sagittal, test, CoT
run --families BoxSize --planes Sagittal --split test --cot --out "${OUT_DIR}/tasks_MedVision-detect-CoT__Sagittal__Test.json"

# Detection (BoxSize -> BoxCoordinate), Coronal, test, CoT
run --families BoxSize --planes Coronal --split test --cot --out "${OUT_DIR}/tasks_MedVision-detect-CoT__Coronal__Test.json"


# Tumor/Lesion size, Axial, test, CoT
run --families TumorLesionSize --planes Axial --split test --cot --out "${OUT_DIR}/tasks_MedVision-TL-CoT__Axial__Test.json"

# Tumor/Lesion size, Sagittal, test, CoT
run --families TumorLesionSize --planes Sagittal --split test --cot --out "${OUT_DIR}/tasks_MedVision-TL-CoT__Sagittal__Test.json"

# Tumor/Lesion size, Coronal, test, CoT
run --families TumorLesionSize --planes Coronal --split test --cot --out "${OUT_DIR}/tasks_MedVision-TL-CoT__Coronal__Test.json"


# Angle/Distance (biometrics), all planes, test, CoT
run --families BiometricsFromLandmarks --planes Axial,Coronal,Sagittal --split test --cot --out "${OUT_DIR}/tasks_MedVision-AD-CoT__AllSlices__Test.json"
# ===============================================


# Task list for training (train split, no CoT)
# ===============================================
# Detection (BoxSize -> BoxCoordinate), Axial, train
run --families BoxSize --planes Axial --split train --out "${OUT_DIR}/tasks_MedVision-detect-CoT__Axial__Train.json"

# Detection (BoxSize -> BoxCoordinate), Sagittal, train
run --families BoxSize --planes Sagittal --split train --out "${OUT_DIR}/tasks_MedVision-detect-CoT__Sagittal__Train.json"

# Detection (BoxSize -> BoxCoordinate), Coronal, train
run --families BoxSize --planes Coronal --split train --out "${OUT_DIR}/tasks_MedVision-detect-CoT__Coronal__Train.json"


# Tumor/Lesion size, Axial, train
run --families TumorLesionSize --planes Axial --split train --out "${OUT_DIR}/tasks_MedVision-TL-CoT__Axial__Train.json"

# Tumor/Lesion size, Sagittal, train
run --families TumorLesionSize --planes Sagittal --split train --out "${OUT_DIR}/tasks_MedVision-TL-CoT__Sagittal__Train.json"

# Tumor/Lesion size, Coronal, train
run --families TumorLesionSize --planes Coronal --split train --out "${OUT_DIR}/tasks_MedVision-TL-CoT__Coronal__Train.json"


# Angle/Distance (biometrics), all planes, train
run --families BiometricsFromLandmarks --planes Axial,Coronal,Sagittal --split train --out "${OUT_DIR}/tasks_MedVision-AD-CoT__AllSlices__Train.json"
# ===============================================


# --- Variations (uncomment / adapt as needed) ---
# OOD planes:   --planes Coronal,Sagittal
# Train split:  --split train   (drop --cot for SFT-style lists)
# MaskSize:     --families MaskSize
# Fast naming-only check: add --no-count
