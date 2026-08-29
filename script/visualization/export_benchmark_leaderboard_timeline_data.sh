#!/bin/bash
# Used to generate the interactive performance-vs-release-date chart on the MedVision project page:
# *********************************
# https://medvision-vlm.github.io/
# *********************************
#
# Rebuilds <PAGE_DIR>/static/js/timeline-data.js, the blob that static/js/timeline.js draws as
# benchmark accuracy against the date each model's weights first appeared — the interactive twin of
# leaderboard_timeline.pdf (viz_benchmark_leaderboard_timeline.sh). It mounts at the TOP of the Leaderboard
# section on index.md, ahead of the per-task tables, as the section's opening claim.
#
# Reads the same three LLM-parsed per-task summary .txt files the PDF does, and reuses that
# script's parsers, release dates and model-series grouping directly — so a number on the page and
# the same number in the paper cannot drift.
#
#   Detection  IoU      sample-weighted mean of the ANATOMY and T/L groups
#   T/L size   1/MRE    from the per-model "Weighted Average" line
#   Distance   1/MRE    from the "Distance" cross-dataset group row
#   Angle      1/MRE    from the "Angle" cross-dataset group row
#
# Usage:
#   bash script/visualization/export_benchmark_leaderboard_timeline_data.sh
#   PYTHON=<path> PAGE_DIR=<path> bash script/visualization/export_benchmark_leaderboard_timeline_data.sh
#
# Environment-variable knobs (all map to export_benchmark_leaderboard_timeline_data.py CLI flags):
#   PAGE_DIR         Project page repo root                     -> --page_dir
#   PYTHON           Interpreter (needs matplotlib + torch)
#   AD_SUMMARY       A/D summary txt                            -> --ad_summary
#   TL_SUMMARY       T/L summary txt                            -> --tl_summary
#   DETECT_SUMMARY   Detection summary txt                      -> --detect_summary
#   COLOR_CONFIG     Config fixing the shared model palette     -> --color_config
#   OUT              Override output JS path                    -> --out
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

PAGE_DIR="${PAGE_DIR:-/mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io}"
PYTHON="${PYTHON:-python}"

# Run medvision_bm against the repo source tree: it is not always installed, and a non-editable
# site-packages copy would otherwise silently shadow src/ with a stale build.
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The exporter imports viz_benchmark_leaderboard_timeline, which pulls plot_utils -> matplotlib + torch. A
# bare `python` satisfies neither in some envs, so fail early with the fix instead of a traceback.
if ! "$PYTHON" -c "import medvision_bm.utils.plot_utils" >/dev/null 2>&1; then
    echo "Error: '$PYTHON' cannot import medvision_bm.utils.plot_utils (needs matplotlib + torch)." >&2
    echo "       Re-run with an env that has both, e.g. PYTHON=/path/to/env/bin/python bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# Defaults kept byte-identical to viz_benchmark_leaderboard_timeline.sh, so the widget and the PDF are always
# built from one Results/ state.
AD_SUMMARY="${AD_SUMMARY:-${REPO_DIR}/Results/MedVision-AD-v2-CoT/summary_AD_task__llm-parsed_gemma-4-31b.txt}"
TL_SUMMARY="${TL_SUMMARY:-${REPO_DIR}/Results/MedVision-TL-v2-CoT/summary_TL_task_filtered__llm-parsed_gemma-4-31b.txt}"
DETECT_SUMMARY="${DETECT_SUMMARY:-${REPO_DIR}/Results/MedVision-detect-v2/summary_detection_task__llm-parsed_gemma-4-31b.txt}"

for f in "$AD_SUMMARY" "$TL_SUMMARY" "$DETECT_SUMMARY"; do
    [ -f "$f" ] || { echo "Error: summary file not found: $f" >&2; exit 1; }
done

ARGS=(
    --page_dir "$PAGE_DIR"
    --ad_summary "$AD_SUMMARY"
    --tl_summary "$TL_SUMMARY"
    --detect_summary "$DETECT_SUMMARY"
)
[ -n "${COLOR_CONFIG:-}" ] && ARGS+=(--color_config "$COLOR_CONFIG")
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")

"$PYTHON" "$SCRIPT_DIR/export_benchmark_leaderboard_timeline_data.py" "${ARGS[@]}"
