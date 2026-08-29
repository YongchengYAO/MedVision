#!/bin/bash
# Generate the single compiled MedVision ground-truth annotation figure.
#
# Stage 1: viz_gt_annotations.py renders GT-only subfigures straight from the data folder,
#          grouped into one folder per task (Detection / Tumor-Lesion-Size / Distance / Angle).
# Stage 2: viz_compile_grid.py composites them into ONE figure, one labelled row-block per task.
#          --dir_model GT drops the model-label column, so the only row labels are task names.
#
# Grid control:
#   --num_col N            columns per task            (default 6)
#   --num_row_per_task N   rows per task               (default 2)
# The compositor derives its own grid from --limit_subfigures and --dataset_as_row_num_row_per_ds:
#   per_task     = ceil(limit_subfigures / n_tasks)
#   cols_per_task = ceil(per_task / num_row_per_ds)
# so passing limit_subfigures = num_col * num_row_per_task * n_tasks lands exactly on num_col
# columns. n_tasks is counted from the task folders that stage 1 actually filled, so a run
# narrowed with --datasets still gets the requested grid.
#
# The sampling pool must be able to feed that grid: _select_samples draws
# num_col * num_row_per_task subfigures PER TASK, and a task fed by a single dataset (Angle comes
# only from Ceph-Biometrics-400) therefore needs POOL_PER_DATASET >= num_col * num_row_per_task.
# That is the default below; raising the grid raises the pool with it.
#
# The annotation version is a ceiling: at 1.1.1 the detection plans resolve back to v1.0.0
# (no detection plan was ever published at 1.1.x) while T/L resolves to a real v1.1.1.
# plan_utils prints a stderr line for every such fallback.
#
# COMPILE_ONLY=1 skips stage 1 and re-lays out the existing subfigure tree, which makes trying
# layouts cheap (stage 1 costs ~25 min parsing detection plans). It refuses if any task folder
# holds fewer than num_col * num_row_per_task subfigures:
#   COMPILE_ONLY=1 bash script/visualization/viz_gt_annotations.sh --num_col 9 --num_row_per_task 1
#
# Unrecognised args are forwarded to stage 1, e.g.:
#   bash script/visualization/viz_gt_annotations.sh --num_col 4 --num_row_per_task 3
#   bash script/visualization/viz_gt_annotations.sh --show_mask
#   bash script/visualization/viz_gt_annotations.sh --datasets CrossMoDA,KiTS23,Ceph-Biometrics-400
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# Interpreter override, matching the PYTHON= convention used by the judge-sweep drivers.
PYTHON="${PYTHON:-python}"

# Run medvision_bm against the repo source tree: it is not always installed, and a non-editable
# site-packages copy would otherwise silently shadow src/ with a stale build.
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# medvision_ds is vendored under Data/src. Fall back to it ONLY when the package is not
# importable, so a real (possibly newer) install is never shadowed by the in-repo snapshot.
if ! "$PYTHON" -c "import medvision_ds" >/dev/null 2>&1; then
    export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/Data/src"
fi

NUM_COL=6
NUM_ROW_PER_TASK=2

FORWARD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_col)             NUM_COL="$2"; shift 2 ;;
        --num_col=*)           NUM_COL="${1#*=}"; shift ;;
        --num_row_per_task)    NUM_ROW_PER_TASK="$2"; shift 2 ;;
        --num_row_per_task=*)  NUM_ROW_PER_TASK="${1#*=}"; shift ;;
        *)                     FORWARD+=("$1"); shift ;;
    esac
done

for n in "$NUM_COL" "$NUM_ROW_PER_TASK"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "--num_col/--num_row_per_task must be positive integers, got '$n'" >&2; exit 2; }
done

PER_TASK=$(( NUM_COL * NUM_ROW_PER_TASK ))

VERSION="${VERSION:-1.1.1}"
FIG_DIR="${FIG_DIR:-${REPO_DIR}/Figures/GT-annotations}"
OUTPUT="${OUTPUT:-${REPO_DIR}/Figures/medvision_gt_annotations.pdf}"
POOL_PER_DATASET="${POOL_PER_DATASET:-$PER_TASK}"
PDF_IMAGE_DPI="${PDF_IMAGE_DPI:-150}"

echo "Grid: ${NUM_COL} col x ${NUM_ROW_PER_TASK} row per task (${PER_TASK} panels/task), pool ${POOL_PER_DATASET}/dataset"

# COMPILE_ONLY=1 re-lays out the EXISTING subfigure tree without re-rendering it. Stage 1 is
# dominated by parsing ~1.8 GB of detection plans (~25 min), so trying layouts is far cheaper
# this way. Valid only while every task folder still holds >= NUM_COL * NUM_ROW_PER_TASK
# subfigures; the check below refuses rather than silently emitting a half-empty row.
if [[ "${COMPILE_ONLY:-0}" == "1" ]]; then
    echo "COMPILE_ONLY=1: reusing the existing subfigure tree in ${FIG_DIR}"
    # NOTE: an `[[ test ]] && echo` idiom here would return non-zero whenever the test is false,
    # and under `set -e` that kills the script in the SUCCESS case. Use an explicit if, and run
    # the loop in this shell (process substitution) so `short` survives.
    short=""
    while IFS= read -r g; do
        n=$(find "$FIG_DIR/GT/$g" -name '*.pdf' | wc -l)
        if [[ "$n" -lt "$PER_TASK" ]]; then
            short+="  $g has $n subfigures, needs $PER_TASK"$'\n'
        fi
    done < <(find "$FIG_DIR/GT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null)
    if [[ -n "$short" ]]; then
        echo "Existing tree cannot fill a ${NUM_COL}x${NUM_ROW_PER_TASK} grid:" >&2
        echo "$short" >&2
        echo "Re-run without COMPILE_ONLY to regenerate the pool." >&2
        exit 1
    fi
else
    "$PYTHON" script/visualization/viz_gt_annotations.py \
        --version "$VERSION" \
        --fig_dir "$FIG_DIR" \
        --pool_per_dataset "$POOL_PER_DATASET" \
        --save_as_pdf \
        ${FORWARD+"${FORWARD[@]}"}
fi

# Count only task folders stage 1 actually populated, so --datasets runs still size correctly.
N_TASKS=$(find "$FIG_DIR/GT" -mindepth 2 -maxdepth 2 -name '*.pdf' -printf '%h\n' | sort -u | wc -l)
[[ "$N_TASKS" -gt 0 ]] || { echo "No subfigures were rendered under ${FIG_DIR}/GT" >&2; exit 1; }
LIMIT_SUBFIGURES=$(( PER_TASK * N_TASKS ))

"$PYTHON" script/visualization/viz_compile_grid.py \
    --dir_subfigures "$FIG_DIR" \
    --dir_model GT \
    --dataset_as_row \
    --dataset_as_row_num_row_per_ds "$NUM_ROW_PER_TASK" \
    --dataset_order Detection,Tumor-Lesion-Size,Distance,Angle \
    --limit_subfigures "$LIMIT_SUBFIGURES" \
    --input_format pdf \
    --output_format pdf png \
    --pdf_image_dpi "$PDF_IMAGE_DPI" \
    --output "$OUTPUT"

echo "Compiled figure: ${OUTPUT}  (${N_TASKS} tasks x ${NUM_ROW_PER_TASK} rows x ${NUM_COL} col)"
