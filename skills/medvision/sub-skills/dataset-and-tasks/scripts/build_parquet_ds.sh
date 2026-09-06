#!/usr/bin/env bash
# build_parquet_ds.sh -- build a small parquet snapshot of MedVision task lists and (optionally)
# render sample figures. Adapted from the repository's dataset-visualization recipe with explicit
# paths, conservative limits and a dry-run.
#
# Purpose
#   Runs `python -m medvision_bm.dataset.build_parquet_ds` for the task lists you pass, writing
#   train.parquet / validation.parquet / test.parquet into --out-dir, then (with --visualize)
#   `python -m medvision_bm.dataset.visualize_samples` on test.parquet.
#
# Prerequisites
#   medvision_bm importable; MedVision_PLANNER_VERSION set; the configs already downloaded into
#   <data_dir> (otherwise the builder downloads them: network + large disk) and
#   <data_dir>/.downloaded_datasets.json present. Visualization imports the vendored lmms_eval
#   task utilities that ship inside medvision_bm.
#
# Usage
#   bash build_parquet_ds.sh --data-dir <data_dir> --out-dir <out_dir> \
#        (--tasks-json-detect F | --tasks-json-tl F | --tasks-json-ad F) [options]
#   Options (defaults are deliberately small):
#     --train-limit-per-subset N   (20)   cap per config while loading the Train pool
#     --test-limit-per-subset N    (10)   cap per config while loading the Test pool
#     --val-limit-per-task N       (5)    validation samples carved from each family (must be > 0)
#     --num-workers N              (1)    parallel config loads
#     --download-mode M            (reuse_dataset_if_exists | reuse_cache_if_exists | force_redownload)
#     --visualize                  also render figures (only when exactly one family is given)
#     --num-samples N              (10)   figures per task type
#     --python <interpreter>       (python)
#     --dry-run                    print the commands, run nothing
#   Task lists must use SFT-style names (no -CoT): tasks_MedVision-*__train_SFT.json.
#   NOTE for --tasks-json-detect: the dataset-info __Train catalogues carry EVAL-namespace keys
#   (..._BoxCoordinate_...), which build_parquet_ds does not rewrite -- it would fail with
#   "BuilderConfig ..._BoxCoordinate_..._Train not found". Use tasks_MedVision-detect__train_SFT.json
#   (already _BoxSize_), or rewrite the keys first with sub-skills/dataset-and-tasks/scripts/list_tasks.py.
set -euo pipefail

usage() { sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

DATA_DIR=""; OUT_DIR=""; TJ_DET=""; TJ_TL=""; TJ_AD=""
TR_SUB=20; TE_SUB=10; VAL_TASK=5; WORKERS=1; DL_MODE="reuse_dataset_if_exists"
VIS=0; NSAMP=10; DRY=0; PY="${PYTHON:-python}"
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --tasks-json-detect) TJ_DET="$2"; shift 2 ;;
    --tasks-json-tl) TJ_TL="$2"; shift 2 ;;
    --tasks-json-ad) TJ_AD="$2"; shift 2 ;;
    --train-limit-per-subset) TR_SUB="$2"; shift 2 ;;
    --test-limit-per-subset) TE_SUB="$2"; shift 2 ;;
    --val-limit-per-task) VAL_TASK="$2"; shift 2 ;;
    --num-workers) WORKERS="$2"; shift 2 ;;
    --download-mode) DL_MODE="$2"; shift 2 ;;
    --visualize) VIS=1; shift ;;
    --num-samples) NSAMP="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$DATA_DIR" ] || { echo "error: --data-dir is required" >&2; exit 2; }
[ -n "$OUT_DIR" ] || { echo "error: --out-dir is required" >&2; exit 2; }
n_fam=0
for f in "$TJ_DET" "$TJ_TL" "$TJ_AD"; do
  if [ -n "$f" ]; then n_fam=$((n_fam+1)); [ -f "$f" ] || { echo "error: not a file: $f" >&2; exit 2; }; fi
done
[ "$n_fam" -gt 0 ] || { echo "error: give at least one of --tasks-json-detect / --tasks-json-tl / --tasks-json-ad" >&2; exit 2; }
[ "$VAL_TASK" -gt 0 ] 2>/dev/null || { echo "error: --val-limit-per-task must be > 0 (the builder asserts limit_val_sample > 0)" >&2; exit 2; }
case "$DL_MODE" in reuse_dataset_if_exists|reuse_cache_if_exists|force_redownload) ;; *) echo "error: bad --download-mode" >&2; exit 2 ;; esac
if [ -z "${MedVision_PLANNER_VERSION:-}" ]; then
  echo "error: MedVision_PLANNER_VERSION is unset (use 'latest', or a pin plus MedVision_ACK_RELEASE)." >&2; exit 2
fi
for f in "$TJ_DET" "$TJ_TL" "$TJ_AD"; do
  if [ -n "$f" ] && grep -q -- '-CoT' "$f"; then
    echo "warning: $f contains '-CoT' names; the parquet builder appends _Train/_Test verbatim and such configs do not exist." >&2
  fi
done

export MedVision_DATA_DIR
MedVision_DATA_DIR="$(cd "$DATA_DIR" 2>/dev/null && pwd || echo "$DATA_DIR")"
[ -f "$MedVision_DATA_DIR/.downloaded_datasets.json" ] || \
  echo "warning: $MedVision_DATA_DIR/.downloaded_datasets.json not found; the builder requires at least one downloaded config." >&2

BUILD=("$PY" -m medvision_bm.dataset.build_parquet_ds
  --parquet_ds_dir "$OUT_DIR"
  --ds_download_mode "$DL_MODE"
  --num_workers_concat_datasets "$WORKERS"
  --train_sample_limit_per_subset "$TR_SUB"
  --test_sample_limit_per_subset "$TE_SUB"
  --val_sample_limit_per_task "$VAL_TASK")
[ -n "$TJ_DET" ] && BUILD+=(--tasks_list_json_path_detect "$TJ_DET")
[ -n "$TJ_TL" ] && BUILD+=(--tasks_list_json_path_TL "$TJ_TL")
[ -n "$TJ_AD" ] && BUILD+=(--tasks_list_json_path_AD "$TJ_AD")

VIS_CMDS=()
if [ "$VIS" -eq 1 ]; then
  if [ "$n_fam" -ne 1 ]; then
    echo "note: --visualize is skipped when several families are mixed in one parquet (per-sample renderers differ); build one --out-dir per family." >&2
  else
    types=()
    [ -n "$TJ_DET" ] && types=(Detection)
    [ -n "$TJ_TL" ] && types=(TL)
    [ -n "$TJ_AD" ] && types=(Angle Distance)
    for t in "${types[@]}"; do
      VIS_CMDS+=("$(printf '%q ' "$PY" -m medvision_bm.dataset.visualize_samples --parquet_ds_path "$OUT_DIR/test.parquet" --fig_dir "$OUT_DIR/figures/$t" --num_samples "$NSAMP" --task_type "$t")")
    done
  fi
fi

echo "[build_parquet_ds] MedVision_DATA_DIR=$MedVision_DATA_DIR  MedVision_PLANNER_VERSION=$MedVision_PLANNER_VERSION"
echo "[build_parquet_ds] build command:"; printf '  %q' "${BUILD[@]}"; echo
for c in "${VIS_CMDS[@]:-}"; do [ -n "$c" ] && { echo "[build_parquet_ds] visualize command:"; echo "  $c"; }; done
if [ "$DRY" -eq 1 ]; then echo "[build_parquet_ds] dry run: nothing executed."; exit 0; fi

mkdir -p "$OUT_DIR"
"${BUILD[@]}"
for c in "${VIS_CMDS[@]:-}"; do [ -n "$c" ] && eval "$c"; done
echo "[build_parquet_ds] done: $OUT_DIR"
