#!/usr/bin/env bash
# End-to-end Clinical Decision Agreement (CDA) analysis over the canonical
# (v2-CoT) result directories, restricted to the config-mapped models.
#
# Usage:  bash run_CDA_analysis.sh [BENCHMARK_DIR]
#   BENCHMARK_DIR defaults to the MedVision repo root inferred from this script.
set -euo pipefail

wd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark_dir="${1:-$(cd "$wd/../../.." && pwd)}"
# One config per task: a model re-run under a bugfix can carry a different
# results-folder name in the AD and TL directories (HealthGPT does).
cfg_ad="$wd/config-AD-CoT.yaml"
cfg_tl="$wd/config-TL-CoT.yaml"

ad_dir="$benchmark_dir/Results/MedVision-AD-v2-CoT"
tl_dir="$benchmark_dir/Results/MedVision-TL-v2-CoT"
kits23_json="$benchmark_dir/Data/Datasets/KiTS23/kits23_clinical.json"

echo "[CDA] benchmark_dir = $benchmark_dir"

# Optional multi-cluster exclusion, matching the T/L benchmark's own filtering
# (summarize_TL_task.py). Unset = no filtering, same default as the rest of the
# repo. Set REMOVED_SAMPLES_DIR=<benchmark_dir>/Data/Datasets to score CDA on the
# same sample set the benchmark reports (renal: 1064 -> 1025 slices), which also
# switches every output filename to its "_filtered" twin.
#
# T/L ONLY. The exclusion list marks slices whose segmentation mask has more than
# one connected component; A/D measurements come from landmark coordinates, so
# there is no mask and no such slice. Passing the flag to the A/D run removed
# nothing yet still wrote a full set of "_filtered" files byte-identical to the
# unfiltered ones (38 of them). The A/D calls below therefore never take it.
# Which parsed-results folder to read inside each model directory:
#   parsed                  regex parser        (prediction in "filtered_resps")
#   llm-parsed_gemma-4-31b  LLM-judge re-parse  (prediction in "LLM_filtered_resps")
# The prefix fixes the prediction field, so the two can never be crossed. Any
# folder starting with "llm-parsed" is accepted, so pointing this at a different
# judge's output needs no code change -- just its real folder name, e.g.
#   CDA_PARSED_DIR=llm-parsed_gemma-4-31b bash run_CDA_analysis.sh
# Per-model outputs are written back into the folder read; task-level reports and
# this run's logs gain a source marker so two sources never overwrite.
CDA_PARSED_DIR="${CDA_PARSED_DIR:-parsed}"
# Ask cda_config for the marker rather than reimplementing it here: the "two
# sources never overwrite each other" guarantee holds only while the shell and
# the Python agree, and this also rejects an unknown source before any work runs.
s="$(cd "$wd" && python -c '
import sys
from cda_config import parsed_source_field, source_suffix
parsed_source_field(sys.argv[1])
print(source_suffix(sys.argv[1]))
' "$CDA_PARSED_DIR")"
echo "[CDA] parsed source = $CDA_PARSED_DIR/${s:+  (marker: $s)}"

REMOVED_SAMPLES_DIR="${REMOVED_SAMPLES_DIR:-}"
REMOVED_SAMPLES_FILENAME="${REMOVED_SAMPLES_FILENAME:-multi_cluster_samples_v1.0.0_to_v1.1.0.json}"
REMOVED_ARGS=()
FILTERED_FLAG=()
if [ -n "$REMOVED_SAMPLES_DIR" ]; then
    REMOVED_ARGS=(--removed_samples_dir "$REMOVED_SAMPLES_DIR"
                  --removed_samples_filename "$REMOVED_SAMPLES_FILENAME")
    FILTERED_FLAG=(--filtered)
    echo "[CDA] removed_samples_dir = $REMOVED_SAMPLES_DIR ($REMOVED_SAMPLES_FILENAME)"
    echo "[CDA] filtering applies to the T/L task only (A/D has no mask clusters)"
fi

# Track 1 — self-consistent agreement (angle proxies live in the AD dir; the
# renal proxy lives in the TL dir).
python "$wd/summarize_CDA_task.py" --task_dir "$ad_dir" --parsed_dirname "$CDA_PARSED_DIR" \
    --config_yaml "$cfg_ad" --skip_model_wo_parsed_files 2>&1 | tee "$wd/cda_summarize_AD-CoT${s}.log"
python "$wd/summarize_CDA_task.py" --task_dir "$tl_dir" --parsed_dirname "$CDA_PARSED_DIR" \
    --config_yaml "$cfg_tl" --skip_model_wo_parsed_files "${REMOVED_ARGS[@]}" 2>&1 | tee "$wd/cda_summarize_TL-CoT${s}.log"

# Track 2 — renal T-stage true-label validation vs KiTS23 pathologic stage.
python "$wd/analyze_CDA_renal_truelabel.py" --task_dir "$tl_dir" --parsed_dirname "$CDA_PARSED_DIR" \
    --kits23_json "$kits23_json" \
    --config_yaml "$cfg_tl" "${REMOVED_ARGS[@]}" 2>&1 | tee "$wd/cda_renal_truelabel_TL-CoT${s}.log"

# Uncertainty — clustered bootstrap 95% CIs + p-values. Must run AFTER the two
# analysis scripts: it reads the per-sample categorizations they persist.
python "$wd/cda_uncertainty.py" --task_dir "$ad_dir" --parsed_dirname "$CDA_PARSED_DIR" \
    --config_yaml "$cfg_ad" 2>&1 | tee "$wd/cda_uncertainty_AD-CoT${s}.log"
python "$wd/cda_uncertainty.py" --task_dir "$tl_dir" --parsed_dirname "$CDA_PARSED_DIR" \
    --config_yaml "$cfg_tl" "${FILTERED_FLAG[@]}" 2>&1 | tee "$wd/cda_uncertainty_TL-CoT${s}.log"
python "$wd/cda_uncertainty.py" --task_dir "$tl_dir" --truelabel --parsed_dirname "$CDA_PARSED_DIR" \
    --config_yaml "$cfg_tl" "${FILTERED_FLAG[@]}" 2>&1 | tee "$wd/cda_uncertainty_truelabel_TL-CoT${s}.log"

# Final report — the one artifact that lives beside the code instead of in the
# gitignored Results/ tree, so the leaderboards survive a fresh clone.
python "$wd/build_CDA_report.py" --parsed_dirname "$CDA_PARSED_DIR" \
    --ad_task_dir "$ad_dir" --ad_config_yaml "$cfg_ad" \
    --tl_task_dir "$tl_dir" --tl_config_yaml "$cfg_tl" \
    "${FILTERED_FLAG[@]}" --out "$wd/CDA_REPORT${s}.md" 2>&1 | tee "$wd/cda_report${s}.log"

# T/L gains the "_filtered" marker; A/D never does (see the note above).
f=""
[ -n "$REMOVED_SAMPLES_DIR" ] && f="_filtered"

echo "[CDA] done. Final report:"
echo "  $wd/CDA_REPORT${s}.md"
echo "[CDA] Underlying canonical reports:"
echo "  $ad_dir/summary_CDA_task${s}_canonical.txt"
echo "  $tl_dir/summary_CDA_task${s}${f}_canonical.txt"
echo "  $tl_dir/summary_CDA_renal_truelabel${s}${f}_canonical.txt"
echo "  $ad_dir/summary_CDA_uncertainty${s}.json"
echo "  $tl_dir/summary_CDA_uncertainty${s}${f}.json"
echo "  $tl_dir/summary_CDA_uncertainty_truelabel${s}${f}.json"
