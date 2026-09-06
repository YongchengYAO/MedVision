#!/usr/bin/env bash
# parse_and_summarize.sh -- run MedVision benchmark step 2 (parse_outputs) and
# step 3 (summarize_{AD,TL,detection}_task) for one task type in one command.
#
# Purpose
#   Turn the raw eval JSONL files of a task directory (or one model directory)
#   into per-sample metrics (parsed/*.jsonl, parsed/*_results.json) and the
#   per-model / cross-model summaries (summary_metrics_*.json,
#   summary_values_*.json, summary_<task>_task.txt). The task type selects the
#   summarizer module and the k numbers expected in <answer>...</answer>.
#
# Prerequisites
#   - medvision_bm importable by the chosen Python (`pip install medvision-bm`
#     or an editable install of the repository);
#   - medvision_ds, torch and transformers importable (both steps import the vendored
#     eval utilities); medvision_ds is installed by
#     `python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>`;
#   - for nMAE (T/L and A/D distance) the NIfTI images must exist at the paths
#     recorded in each record's doc.image_file (otherwise nMAE is NaN, silently).
#   No GPU or network is needed.
#
# Examples
#   parse_and_summarize.sh --task-type TL --task-dir Results/MedVision-TL -p 8 \
#       --removed-samples-dir <data_dir>/Datasets
#   parse_and_summarize.sh --task-type Detection --model-dir Results/MedVision-detect/<model_name> --skip-existing
#   # re-summarize LLM-judge records only (parse step is skipped automatically):
#   parse_and_summarize.sh --task-type AD --task-dir Results/MedVision-AD \
#       --parsed-dirname llm-parsed_gemma-4-31b --resps-key LLM_filtered_resps --skip-model-wo-parsed-files
#   parse_and_summarize.sh --task-type TL --task-dir Results/MedVision-TL --dry-run

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: parse_and_summarize.sh --task-type {AD|TL|Detection} (--task-dir DIR | --model-dir DIR) [options]

Required
  --task-type TYPE              AD | TL | Detection (selects k=1/2/4 numbers and the summarizer)
  --task-dir DIR                task folder holding one sub-folder per model (all models processed)
  --model-dir DIR               a single model folder (mutually exclusive with --task-dir)

Options passed to both steps
  -p, --processes N             worker processes (parse: per JSONL file; summarize: per file or label group)
  --limit N                     first N records per JSONL (parsed files are truncated; summary names get _limit<N>)

parse_outputs options
  --skip-existing               keep already parsed files
  --rm-old                      delete <model>/parsed before parsing
  --skip-parse                  do not run parse_outputs (implied when --parsed-dirname != parsed)

summarize options
  --parsed-dirname NAME         per-model folder to summarize (default parsed; e.g. llm-parsed_gemma-4-31b)
  --resps-key KEY               record key with the prediction (default filtered_resps; LLM_filtered_resps for llm-parsed*/)
  --models NAME [NAME ...]      restrict --task-dir mode to these model folder names (must be the last option)
  --skip-model-wo-parsed-files  skip model folders without the parsed folder (--task-dir only)
  --removed-samples-dir DIR     TL only: <data_dir>/Datasets with per-dataset removed-samples JSON (adds _filtered)
  --removed-samples-filename F  TL only: file name inside each dataset folder
                                (default multi_cluster_samples_v1.0.0_to_v1.1.0.json)

Other
  --python PATH                 interpreter to use (default: python)
  --dry-run                     print the two commands and exit
  -h, --help                    this message
USAGE
}

TASK_TYPE=""; TASK_DIR=""; MODEL_DIR=""; PROCESSES=""; LIMIT=""
SKIP_EXISTING=0; RM_OLD=0; SKIP_PARSE=0
PARSED_DIRNAME="parsed"; RESPS_KEY=""; SKIP_WO=0
REMOVED_DIR=""; REMOVED_FILE=""; PYTHON_BIN="python"; DRY_RUN=0
MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-type) TASK_TYPE="$2"; shift 2 ;;
    --task-dir) TASK_DIR="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    -p|--processes) PROCESSES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --rm-old) RM_OLD=1; shift ;;
    --skip-parse) SKIP_PARSE=1; shift ;;
    --parsed-dirname) PARSED_DIRNAME="$2"; shift 2 ;;
    --resps-key) RESPS_KEY="$2"; shift 2 ;;
    --skip-model-wo-parsed-files) SKIP_WO=1; shift ;;
    --removed-samples-dir) REMOVED_DIR="$2"; shift 2 ;;
    --removed-samples-filename) REMOVED_FILE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --models) shift; while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$TASK_TYPE" in
  AD) SUMMARIZER="medvision_bm.benchmark.summarize_AD_task" ;;
  TL) SUMMARIZER="medvision_bm.benchmark.summarize_TL_task" ;;
  Detection) SUMMARIZER="medvision_bm.benchmark.summarize_detection_task" ;;
  "") echo "Error: --task-type is required (AD | TL | Detection)" >&2; exit 2 ;;
  *) echo "Error: --task-type must be AD, TL or Detection (got '$TASK_TYPE')" >&2; exit 2 ;;
esac
if [[ -z "$TASK_DIR" && -z "$MODEL_DIR" ]]; then echo "Error: one of --task-dir or --model-dir is required" >&2; exit 2; fi
if [[ -n "$TASK_DIR" && -n "$MODEL_DIR" ]]; then echo "Error: --task-dir and --model-dir are mutually exclusive" >&2; exit 2; fi
if [[ $SKIP_WO -eq 1 && -z "$TASK_DIR" ]]; then echo "Error: --skip-model-wo-parsed-files needs --task-dir" >&2; exit 2; fi
if [[ ${#MODELS[@]} -gt 0 && -z "$TASK_DIR" ]]; then echo "Error: --models needs --task-dir" >&2; exit 2; fi
if [[ ( -n "$REMOVED_DIR" || -n "$REMOVED_FILE" ) && "$TASK_TYPE" != "TL" ]]; then
  echo "Error: --removed-samples-dir/--removed-samples-filename are accepted by summarize_TL_task only" >&2; exit 2
fi
if [[ "$PARSED_DIRNAME" != "parsed" && $SKIP_PARSE -eq 0 ]]; then
  echo "[info] --parsed-dirname '$PARSED_DIRNAME' != 'parsed': parse_outputs only writes to parsed/, so the parse step is skipped." >&2
  SKIP_PARSE=1
fi
if [[ "$PARSED_DIRNAME" == llm-parsed* && -z "$RESPS_KEY" ]]; then
  echo "[warn] llm-parsed*/ records carry 'LLM_filtered_resps' and lack 'filtered_resps'; the summarizer will abort unless you pass --resps-key LLM_filtered_resps." >&2
fi

TARGET_ARGS=()
if [[ -n "$TASK_DIR" ]]; then TARGET_ARGS+=(--task_dir "$TASK_DIR"); else TARGET_ARGS+=(--model_dir "$MODEL_DIR"); fi

PARSE_CMD=("$PYTHON_BIN" -m medvision_bm.benchmark.parse_outputs --task_type "$TASK_TYPE" "${TARGET_ARGS[@]}")
[[ -n "$LIMIT" ]] && PARSE_CMD+=(--limit "$LIMIT")
[[ -n "$PROCESSES" ]] && PARSE_CMD+=(--processes "$PROCESSES")
[[ $SKIP_EXISTING -eq 1 ]] && PARSE_CMD+=(--skip_existing)
[[ $RM_OLD -eq 1 ]] && PARSE_CMD+=(--rm_old)

SUM_CMD=("$PYTHON_BIN" -m "$SUMMARIZER" "${TARGET_ARGS[@]}" --parsed_dirname "$PARSED_DIRNAME")
# NOTE: summarize_detection_task honours --resps_key only in --task_dir mode; its --model_dir
# path drops it (summarize_detection_task.py:876-878). Use --task-dir for judge re-parses.
if [[ -n "$RESPS_KEY" ]]; then
  SUM_CMD+=(--resps_key "$RESPS_KEY")
  if [[ "$SUMMARIZER" == *summarize_detection_task* && -n "${MODEL_DIR:-}" ]]; then
    echo "warning: --resps-key is ignored by summarize_detection_task in --model-dir mode; use --task-dir" >&2
  fi
fi
[[ -n "$LIMIT" ]] && SUM_CMD+=(--limit "$LIMIT")
[[ -n "$PROCESSES" ]] && SUM_CMD+=(--processes "$PROCESSES")
[[ $SKIP_WO -eq 1 ]] && SUM_CMD+=(--skip_model_wo_parsed_files)
[[ -n "$REMOVED_DIR" ]] && SUM_CMD+=(--removed_samples_dir "$REMOVED_DIR")
[[ -n "$REMOVED_FILE" ]] && SUM_CMD+=(--removed_samples_filename "$REMOVED_FILE")
[[ ${#MODELS[@]} -gt 0 ]] && SUM_CMD+=(--models "${MODELS[@]}")

echo "[step 2] ${PARSE_CMD[*]}"; [[ $SKIP_PARSE -eq 1 ]] && echo "         (skipped)"
echo "[step 3] ${SUM_CMD[*]}"
if [[ $DRY_RUN -eq 1 ]]; then echo "[dry-run] nothing executed"; exit 0; fi

if ! "$PYTHON_BIN" -c "import medvision_bm" 2>/dev/null; then
  echo "Error: '$PYTHON_BIN' cannot import medvision_bm. Install it (pip install medvision-bm, or pip install -e <repo>) or pass --python." >&2; exit 3
fi
if ! "$PYTHON_BIN" -c "import medvision_ds" 2>/dev/null; then
  echo "Error: '$PYTHON_BIN' cannot import medvision_ds. Both steps import it (via the eval utilities); install with: $PYTHON_BIN -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>" >&2; exit 3
fi
if [[ $SKIP_PARSE -eq 0 ]]; then "${PARSE_CMD[@]}"; fi
"${SUM_CMD[@]}"
