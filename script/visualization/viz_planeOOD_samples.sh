#!/bin/bash
# Generate the compiled MedVision plane-OOD sample figures (one per task).
#
# MedVision-V0 was SFT'd on AXIAL slices only, so coronal and sagittal are the plane-OOD
# evaluation planes. Each compiled figure shows, per column, ONE volume + target seen in all
# three planes, with the in-distribution axial block on top:
#
#     Axial (ID)      row block  <- top, in-distribution
#     Coronal (OOD)   row block
#     Sagittal (OOD)  row block
#
# Stage 1: viz_planeOOD_samples.py renders GT-only panels straight from the data folder, into
#          <fig_dir>/<task>/<plane>/<dataset>/<pairing-key>.pdf. The pairing key carries no slice
#          index, so the three plane folders hold BYTE-IDENTICAL filenames for one volume+target.
# Stage 2: viz_compile_grid.py composites each task's tree in its DEFAULT layout mode, where the
#          plane occupies the "model" level. That mode builds one ordered sample list and reuses
#          it for every model block, varying only the folder prefix -- which is exactly what puts
#          the three planes of one volume in the same column. It also intersects filenames across
#          the plane folders, so an incomplete triple drops out instead of misaligning a column.
#          Row order comes from sorted(), and "Axial (ID)" < "Coronal (OOD)" < "Sagittal (OOD)",
#          so the in-distribution block lands on top. Blocks are named by the rotated labels
#          (LABEL_FONTSIZE); the compositor's inter-block rules are suppressed, the labels being
#          large enough to mark the divisions on their own.
#
# Grid control:
#   --num_col N            columns                      (default 6)
#   --num_row_per_type N   rows per plane type          (default 1)
# The compiled figure therefore has 3 * num_row_per_type rows, and stage 1 renders exactly
# num_col * num_row_per_type volume+target groups, drawn round-robin across datasets so the
# columns span as many datasets as possible. Because at most that many datasets can contribute,
# --limit_subfigures is never below the dataset count that viz_compile_grid.py insists on.
#
# The annotation version is a ceiling: at 1.1.1 the detection plans resolve back to v1.0.0 (no
# detection plan was ever published at 1.1.x) while T/L resolves to a real v1.1.1. T/L NEEDS
# 1.1.1 -- that release added the sagittal and coronal T/L slices, without which the OOD rows
# would be empty. plan_utils prints a stderr line for every fallback.
#
# COMPILE_ONLY=1 skips stage 1 and re-lays out the existing panel tree, which makes trying
# layouts cheap (stage 1 parses multi-hundred-MB detection plans). It refuses if any plane folder
# holds fewer than num_col * num_row_per_type panels:
#   COMPILE_ONLY=1 bash script/visualization/viz_planeOOD_samples.sh --num_col 8
#
# Unrecognised args are forwarded to stage 1, e.g.:
#   bash script/visualization/viz_planeOOD_samples.sh --num_col 4 --num_row_per_type 2
#   bash script/visualization/viz_planeOOD_samples.sh --tasks Tumor-Lesion-Size --show_mask
#   bash script/visualization/viz_planeOOD_samples.sh --max_plan_mb 300   # skip AbdomenAtlas etc.
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
NUM_ROW_PER_TYPE=1

FORWARD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_col)              NUM_COL="$2"; shift 2 ;;
        --num_col=*)            NUM_COL="${1#*=}"; shift ;;
        --num_row_per_type)     NUM_ROW_PER_TYPE="$2"; shift 2 ;;
        --num_row_per_type=*)   NUM_ROW_PER_TYPE="${1#*=}"; shift ;;
        *)                      FORWARD+=("$1"); shift ;;
    esac
done

for n in "$NUM_COL" "$NUM_ROW_PER_TYPE"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "--num_col/--num_row_per_type must be positive integers, got '$n'" >&2; exit 2; }
done

PER_TASK=$(( NUM_COL * NUM_ROW_PER_TYPE ))

VERSION="${VERSION:-1.1.1}"
FIG_DIR="${FIG_DIR:-${REPO_DIR}/Figures/planeOOD-samples}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/Figures}"
PDF_IMAGE_DPI="${PDF_IMAGE_DPI:-150}"

# The plane labels carry the figure's whole point (which rows are in-distribution), so they run at
# twice viz_compile_grid.py's 11 pt default.
LABEL_FONTSIZE="${LABEL_FONTSIZE:-22}"

echo "Grid: ${NUM_COL} col x ${NUM_ROW_PER_TYPE} row per plane type (${PER_TASK} volume+target groups, $((3 * NUM_ROW_PER_TYPE)) rows)"

if [[ "${COMPILE_ONLY:-0}" == "1" ]]; then
    echo "COMPILE_ONLY=1: reusing the existing panel tree in ${FIG_DIR}"
else
    "$PYTHON" script/visualization/viz_planeOOD_samples.py \
        --version "$VERSION" \
        --fig_dir "$FIG_DIR" \
        --num_col "$NUM_COL" \
        --num_row_per_type "$NUM_ROW_PER_TYPE" \
        --save_as_pdf \
        ${FORWARD+"${FORWARD[@]}"}
fi

# One compiled figure per task tree stage 1 actually populated, so a run narrowed with --tasks
# (or one where a task found no complete triples) still succeeds.
compiled=0
for task_dir in "$FIG_DIR"/*/; do
    [[ -d "$task_dir" ]] || continue
    task="$(basename "$task_dir")"

    # Refuse a half-empty grid rather than silently emitting one. Under COMPILE_ONLY this is the
    # only thing standing between a stale tree and a figure with blank cells.
    short=""
    for plane_dir in "$task_dir"*/; do
        [[ -d "$plane_dir" ]] || continue
        n=$(find "$plane_dir" -name '*.pdf' | wc -l)
        if [[ "$n" -lt "$PER_TASK" ]]; then
            short+="  $(basename "$plane_dir") has $n panels, needs $PER_TASK"$'\n'
        fi
    done
    if [[ -n "$short" ]]; then
        echo "Skipping '${task}': tree cannot fill a ${NUM_COL}x${NUM_ROW_PER_TYPE} grid:" >&2
        echo "$short" >&2
        continue
    fi

    out="${OUTPUT_DIR}/medvision_planeOOD_$(echo "$task" | tr ' ' '-').pdf"
    "$PYTHON" script/visualization/viz_compile_grid.py \
        --dir_subfigures "$task_dir" \
        --limit_subfigures "$PER_TASK" \
        --row_per_model "$NUM_ROW_PER_TYPE" \
        --input_format pdf \
        --output_format pdf png \
        --pdf_image_dpi "$PDF_IMAGE_DPI" \
        --model_label_fontsize "$LABEL_FONTSIZE" \
        --hide_model_separator \
        --output "$out"
    echo "Compiled figure: ${out}  (${task}: $((3 * NUM_ROW_PER_TYPE)) rows x ${NUM_COL} col)"
    compiled=$(( compiled + 1 ))
done

[[ "$compiled" -gt 0 ]] || { echo "No task tree under ${FIG_DIR} could be compiled" >&2; exit 1; }
