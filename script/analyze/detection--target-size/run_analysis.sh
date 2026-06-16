#!/bin/bash
# Parse detection results by box-to-image ratio and generate comparison figure.
#
# Usage:
#   bash run_analysis.sh --task_dir <path> [options]
#   bash run_analysis.sh --model_dir <path> [options]
#
# Options:
#   --config <yaml>              YAML mapping model folder → display name
#                                (default: ${SCRIPT_DIR}/config-detect-boxImgRatio.yaml)
#   --out_dir <path>             Output directory for PNG (default: script dir)
#   --limit <N>                  Limit samples per JSONL
#   --skip_model_wo_parsed_files Skip models missing parsed/ (only with --task_dir)
#   --processes, -p <N>           Worker count for parsing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ── Parse CLI args ─────────────────────────────────────────────────────────────
TASK_DIR=""
MODEL_DIR=""
CONFIG="${SCRIPT_DIR}/config-detect-boxImgRatio.yaml"
OUT_DIR="${SCRIPT_DIR}"
LIMIT=""
SKIP_FLAG=""
PROCESSES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_dir)
            TASK_DIR="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --skip_model_wo_parsed_files)
            SKIP_FLAG="--skip_model_wo_parsed_files"
            shift
            ;;
        --processes | -p)
            PROCESSES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${TASK_DIR}" && -z "${MODEL_DIR}" ]]; then
    echo "Error: one of --task_dir or --model_dir is required"
    exit 1
fi
if [[ -n "${TASK_DIR}" && -n "${MODEL_DIR}" ]]; then
    echo "Error: --task_dir and --model_dir are mutually exclusive"
    exit 1
fi
if [[ -n "${SKIP_FLAG}" && -z "${TASK_DIR}" ]]; then
    echo "Error: --skip_model_wo_parsed_files requires --task_dir"
    exit 1
fi

# ── Step 1: Parse results (+ random baseline when --task_dir) ─────────────────
echo "[1/2] Parsing detection results"
ANALYZE_ARGS=()
[[ -n "${TASK_DIR}" ]] && ANALYZE_ARGS+=(--task_dir "${TASK_DIR}")
[[ -n "${MODEL_DIR}" ]] && ANALYZE_ARGS+=(--model_dir "${MODEL_DIR}")
[[ -n "${LIMIT}" ]] && ANALYZE_ARGS+=(--limit "${LIMIT}")
[[ -n "${SKIP_FLAG}" ]] && ANALYZE_ARGS+=("${SKIP_FLAG}")
[[ -n "${PROCESSES}" ]] && ANALYZE_ARGS+=(--processes "${PROCESSES}")

python -m medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random \
    "${ANALYZE_ARGS[@]}"

# ── Step 2: Remap config so viz resolves {model}/parsed paths ─────────────────
echo ""
echo "[2/2] Generating visualization"

if [[ -n "${TASK_DIR}" ]]; then
    VIZ_IN_DIR="${TASK_DIR}"
else
    VIZ_IN_DIR="$(dirname "${MODEL_DIR}")"
fi

TMP_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

python - <<PYEOF
import yaml
with open("${CONFIG}") as f:
    cfg = yaml.safe_load(f)
new_map = {}
for model, display in cfg["model_display_name"].items():
    if model == "random_detection":
        new_map[model] = display
    else:
        new_map[f"{model}/parsed"] = display
with open("${TMP_CONFIG}", "w") as f:
    yaml.dump({"model_display_name": new_map}, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF

mkdir -p "${OUT_DIR}"
python -m medvision_bm.benchmark.viz_detection_performance_per_boxImgRatio \
    --config "${TMP_CONFIG}" \
    --in_dir "${VIZ_IN_DIR}" \
    --out_dir "${OUT_DIR}"

echo ""
echo "Done. Figure saved to: ${OUT_DIR}/metrics_boxImgRatio-dotline.png"
