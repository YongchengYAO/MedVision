#!/usr/bin/env bash
# run_cda.sh -- Clinical Decision Agreement (CDA) end-to-end runner (medvision skill, analysis sub-skill).
#
# Purpose
#   Re-score existing parsed benchmark records through published clinical cutoff tables and report
#   agreement (Cohen's kappa / quadratic-weighted kappa) between the category of the model's
#   measurement and the category of the ground truth. Runs, in order, the bundled
#     cda/summarize_CDA_task.py   (per-sample categorisation + per-proxy agreement, per task dir)
#     cda/cda_uncertainty.py      (clustered bootstrap 95% CIs + one-sided p-values; resamples VOLUMES)
#     cda/build_CDA_report.py     (renders ONE Markdown leaderboard report; recomputes nothing)
#   CPU only; no inference; reads <model_dir>/<parsed_dirname>/*.jsonl and writes JSON/TXT next to
#   them plus the report. Seconds per model.
#
# Prerequisites
#   - A Python with numpy. PyYAML is needed only when --ad-config/--tl-config are given.
#     medvision_bm is NOT required (the CDA modules are self-contained).
#   - Parsed records: Results/<AD task>/<model>/parsed/*.jsonl and/or Results/<TL task>/<model>/parsed/*.jsonl
#     (or an llm-parsed_<judge>/ folder; see --parsed-dirname).
#
# Example
#   bash run_cda.sh \
#     --ad-task-dir Results/MedVision-AD-v2-CoT --ad-config scripts/cda/config-AD-CoT.yaml \
#     --tl-task-dir Results/MedVision-TL-v2-CoT --tl-config scripts/cda/config-TL-CoT.yaml \
#     --removed-samples-dir <data_dir>/Datasets --out CDA_REPORT.md
#   bash run_cda.sh --tl-task-dir Results/MedVision-TL-v2-CoT --dry-run      # one task, print commands only
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cda_dir="$here/cda"

usage() {
cat <<'EOF'
Usage: run_cda.sh [--ad-task-dir DIR] [--tl-task-dir DIR] [options]

At least one of --ad-task-dir / --tl-task-dir is required. The Markdown report
(build_CDA_report.py) needs BOTH; with only one task the runner performs the
agreement + uncertainty steps for that task and skips the report.

Task directories / configs
  --ad-task-dir DIR        A/D results directory (angle proxies: SNA, SNB on Ceph-Biometrics-400)
  --tl-task-dir DIR        T/L results directory (renal AJCC T-category proxy on KiTS23, KiPA22)
  --ad-config YAML         model_display_name config for the A/D dir (template: cda/config-AD-CoT.yaml)
  --tl-config YAML         model_display_name config for the T/L dir (template: cda/config-TL-CoT.yaml)
                           Without a config every subfolder of the task dir is analysed and the
                           task-level report has no "_canonical" marker. A config-listed folder
                           that is missing on disk is a hard error.

Parsed source
  --parsed-dirname NAME    Folder inside each model dir to read: "parsed" (regex parser, default) or
                           any "llm-parsed*" folder (LLM-judge re-parse). The prefix selects the row
                           field holding the prediction (filtered_resps vs LLM_filtered_resps).
                           Task-level outputs gain a marker such as "_llm-parsed-gemma-4-31b".

Sample set (T/L only)
  --removed-samples-dir DIR         <data_dir>/Datasets; drops multi-cluster T/L slices so CDA scores the
                                    same sample set as summarize_TL_task; T/L outputs gain "_filtered".
                                    Never applied to the A/D task (landmarks have no mask clusters).
  --removed-samples-filename NAME   default: multi_cluster_samples_v1.0.0_to_v1.1.0.json

Uncertainty
  --n-boot N               bootstrap resamples (default 4000)
  --seed N                 resampling seed (default: CDA_SEED = 1024 from cda_config.py)

Output / misc
  --out FILE               report path (default: ./CDA_REPORT<source-marker>.md)
  --repo-root DIR          directory against which the report shortens paths (default: cwd)
  --python EXE             interpreter (default: python)
  --dry-run                print the commands and exit 0
  -h, --help               this text
EOF
}

ad_dir=""; tl_dir=""; cfg_ad=""; cfg_tl=""
parsed_dirname="parsed"
removed_dir=""; removed_fname="multi_cluster_samples_v1.0.0_to_v1.1.0.json"
n_boot="4000"; seed=""
out=""; repo_root=""; PY="python"; dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ad-task-dir) ad_dir="$2"; shift 2 ;;
    --tl-task-dir) tl_dir="$2"; shift 2 ;;
    --ad-config) cfg_ad="$2"; shift 2 ;;
    --tl-config) cfg_tl="$2"; shift 2 ;;
    --parsed-dirname) parsed_dirname="$2"; shift 2 ;;
    --removed-samples-dir) removed_dir="$2"; shift 2 ;;
    --removed-samples-filename) removed_fname="$2"; shift 2 ;;
    --n-boot) n_boot="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --repo-root) repo_root="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_cda] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$ad_dir" && -z "$tl_dir" ]]; then
  echo "[run_cda] error: give at least one of --ad-task-dir / --tl-task-dir" >&2; exit 2
fi
for d in "$ad_dir" "$tl_dir"; do
  if [[ -n "$d" && ! -d "$d" && $dry_run -eq 0 ]]; then
    echo "[run_cda] error: task directory not found: $d" >&2; exit 2
  fi
done
for c in "$cfg_ad" "$cfg_tl"; do
  if [[ -n "$c" && ! -f "$c" && $dry_run -eq 0 ]]; then
    echo "[run_cda] error: config not found: $c" >&2; exit 2
  fi
done
if [[ -n "$removed_dir" && ! -d "$removed_dir" && $dry_run -eq 0 ]]; then
  # A non-existent directory would SILENTLY yield unfiltered numbers in "_filtered" files.
  echo "[run_cda] error: --removed-samples-dir does not exist: $removed_dir" >&2; exit 2
fi
for f in cda_config.py cda_stats.py summarize_CDA_task.py cda_uncertainty.py build_CDA_report.py; do
  [[ -f "$cda_dir/$f" ]] || { echo "[run_cda] error: bundled module missing: $cda_dir/$f" >&2; exit 2; }
done

# Ask cda_config for the source marker (also rejects an unknown --parsed-dirname prefix up front).
if ! marker="$(cd "$cda_dir" && "$PY" -c '
import sys
from cda_config import parsed_source_field, source_suffix
parsed_source_field(sys.argv[1])
print(source_suffix(sys.argv[1]))
' "$parsed_dirname" 2>&1)"; then
  echo "[run_cda] error: $marker" >&2; exit 2
fi
[[ -z "$out" ]] && out="CDA_REPORT${marker}.md"

removed_args=(); filtered_flag=()
if [[ -n "$removed_dir" ]]; then
  removed_args=(--removed_samples_dir "$removed_dir" --removed_samples_filename "$removed_fname")
  filtered_flag=(--filtered)
fi
seed_args=(); [[ -n "$seed" ]] && seed_args=(--seed "$seed")
root_args=(); [[ -n "$repo_root" ]] && root_args=(--repo_root "$repo_root")

echo "[run_cda] parsed source = $parsed_dirname/${marker:+  (marker: $marker)}"
[[ -n "$removed_dir" ]] && echo "[run_cda] removed-samples filter (T/L only) = $removed_dir/<dataset>/$removed_fname"

run() {
  echo "+ $*"
  if [[ $dry_run -eq 0 ]]; then "$@"; fi
}

# 1) agreement -- one invocation per task directory, each with its own config
if [[ -n "$ad_dir" ]]; then
  cfg_args=(); [[ -n "$cfg_ad" ]] && cfg_args=(--config_yaml "$cfg_ad")
  run "$PY" "$cda_dir/summarize_CDA_task.py" --task_dir "$ad_dir" --parsed_dirname "$parsed_dirname" \
      "${cfg_args[@]}" --skip_model_wo_parsed_files
fi
if [[ -n "$tl_dir" ]]; then
  cfg_args=(); [[ -n "$cfg_tl" ]] && cfg_args=(--config_yaml "$cfg_tl")
  run "$PY" "$cda_dir/summarize_CDA_task.py" --task_dir "$tl_dir" --parsed_dirname "$parsed_dirname" \
      "${cfg_args[@]}" --skip_model_wo_parsed_files "${removed_args[@]}"
fi

# 2) uncertainty -- must run AFTER step 1 (reads the per-sample categorisations it persists)
if [[ -n "$ad_dir" ]]; then
  cfg_args=(); [[ -n "$cfg_ad" ]] && cfg_args=(--config_yaml "$cfg_ad")
  run "$PY" "$cda_dir/cda_uncertainty.py" --task_dir "$ad_dir" --parsed_dirname "$parsed_dirname" \
      "${cfg_args[@]}" --n_boot "$n_boot" "${seed_args[@]}"
fi
if [[ -n "$tl_dir" ]]; then
  cfg_args=(); [[ -n "$cfg_tl" ]] && cfg_args=(--config_yaml "$cfg_tl")
  run "$PY" "$cda_dir/cda_uncertainty.py" --task_dir "$tl_dir" --parsed_dirname "$parsed_dirname" \
      "${cfg_args[@]}" --n_boot "$n_boot" "${seed_args[@]}" "${filtered_flag[@]}"
fi

# 3) report -- needs both task directories AND both configs (build_CDA_report.py requires them)
if [[ -n "$ad_dir" && -n "$tl_dir" ]]; then
  if [[ -z "$cfg_ad" || -z "$cfg_tl" ]]; then
    echo "[run_cda] note: build_CDA_report.py requires --ad-config and --tl-config; report skipped."
    echo "[run_cda] task-level reports: $ad_dir/summary_CDA_task${marker}*.txt, $tl_dir/summary_CDA_task${marker}*.txt"
  else
    run "$PY" "$cda_dir/build_CDA_report.py" --parsed_dirname "$parsed_dirname" \
        --ad_task_dir "$ad_dir" --ad_config_yaml "$cfg_ad" \
        --tl_task_dir "$tl_dir" --tl_config_yaml "$cfg_tl" \
        "${filtered_flag[@]}" "${root_args[@]}" --out "$out"
    echo "[run_cda] final report: $out"
  fi
else
  echo "[run_cda] note: only one task directory given; the Markdown report needs both (skipped)."
fi

f=""; [[ -n "$removed_dir" ]] && f="_filtered"
c_ad=""; [[ -n "$cfg_ad" ]] && c_ad="_canonical"
c_tl=""; [[ -n "$cfg_tl" ]] && c_tl="_canonical"
echo "[run_cda] task-level outputs:"
[[ -n "$ad_dir" ]] && echo "  $ad_dir/summary_CDA_task${marker}${c_ad}.txt   $ad_dir/summary_CDA_uncertainty${marker}.json"
[[ -n "$tl_dir" ]] && echo "  $tl_dir/summary_CDA_task${marker}${f}${c_tl}.txt   $tl_dir/summary_CDA_uncertainty${marker}${f}.json"
echo "[run_cda] per-model outputs: <model_dir>/$parsed_dirname/summary_{metrics,values}_CDA_Task${f}.json"
[[ $dry_run -eq 1 ]] && echo "[run_cda] dry run: nothing executed."
exit 0
