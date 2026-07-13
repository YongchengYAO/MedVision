#!/usr/bin/env bash
set -euo pipefail

# Summarize the local Data/Datasets collection from the benchmark plans (no HF; nibabel only if
# the live labels_map lookup via medvision_ds succeeds).
# Writes dataset_files.jsonl, dataset_summary_{filtered,raw}.json, dataset_summary.csv, and
# dataset_label_stats.csv.

# Resolve the repo root from this script's location (<repo>/script/misc/<this>.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Paths are overridable via the environment; defaults assume the standard layout.
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data}"
# Dataset (biometry) version: selects which T/L plan is counted. Detection/segmentation ship v1.0.0
# only, so only T/L differs across versions. Output dir is version-suffixed.
PLAN_VERSION="${PLAN_VERSION:-1.1.1}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dataset-info/datasets_summary_v${PLAN_VERSION}}"
# Optional: reuse version-invariant Box/segmentation/inventory from this existing summary dir and
# recompute only biometry — skips the multi-GB detection scan (fast version regen).
REUSE_FROM="${REUSE_FROM:-}"

ARGS=(--data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" --plan_version "${PLAN_VERSION}")
[ -n "${REUSE_FROM}" ] && ARGS+=(--reuse_from "${REUSE_FROM}")

python -m medvision_bm.utils.summarize_datasets "${ARGS[@]}" "$@"

# The scan always writes dataset_summary_{filtered,raw}.json (filtered = v1.0.0-filtered
# benchmark counts; raw = the same 3 tasks counted unfiltered).
#
# --- Variations (uncomment / adapt as needed) ---
# Subset:   --datasets KiTS23,Ceph-Biometrics-400
# Pin ds:   --plan_version 1.1.1   (biometry only; measurement sample set differs by version)
# Fast:     --no_detection  skip the large detection plans (drops BoxSize; ~8.5 min vs ~30 min)
# Figures:  --viz    also render dataset_summary.pdf (bar panels), dataset_summary_wordcloud.png,
#                    and the donut in filtered+raw x {2x1, 1x2, compact}:
#                    dataset_summary_rings_{filtered,raw}_{2x1,1x2,compact}.pdf
#                    (2x1/1x2 = the magnified small-dataset panel stacked / side-by-side)
# Fig only: --viz_only  skip the scan; render all figures from the existing dataset_summary_filtered.json
