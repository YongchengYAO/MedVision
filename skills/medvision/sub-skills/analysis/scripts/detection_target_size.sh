#!/usr/bin/env bash
# detection_target_size.sh -- Detection metrics stratified by box-to-image ratio (medvision skill, analysis).
#
# Purpose
#   Step 1  python -m medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random
#           Reads <model>/<parsed_dirname>/*_BoxCoordinate_*.jsonl (per-sample metrics written by
#           parse_outputs), bins every sample by box area / image area (5% bins), writes
#           summary_metrics_per_boxImgRatio_detect_Task.json + summary_values_per_boxImgRatio_detect_Task.json
#           into that folder and, in --task-dir mode, a random-box baseline into <task_dir>/random_detection/
#           (RANDOM_BOX_SIMULATIONS = 100 random boxes per GT box, seed SEED = 1024).
#   Step 2  python -m medvision_bm.benchmark.viz_detection_performance_per_boxImgRatio
#           Plots F1/Precision/Recall vs box-to-image ratio for the config's models
#           -> <out_dir>/metrics_boxImgRatio-dotline.pdf (PNG with --save-as-png).
#           The config keys are rewritten to "<folder>/<parsed_dirname>" (except random_detection) so the
#           viz reads the JSON from the parsed source folder.
#
# Prerequisites
#   - medvision_bm importable by --python (pip install medvision-bm, or --repo-root <checkout> to prepend
#     <checkout>/src to PYTHONPATH), with medvision_ds installed (the analyzer resolves label names from the
#     segmentation benchmark plans) and PyYAML/matplotlib/pandas for the plot.
#   - Detection results already parsed: Results/<detection task>/<model>/parsed/*_BoxCoordinate_*.jsonl.
#   CPU only; the random baseline is the slow part (100 simulated boxes per sample).
#
# Example
#   bash detection_target_size.sh --task-dir Results/MedVision-detect-v2 --config scripts/config-detect-boxImgRatio.yaml \
#        --out-dir Figures/boxImgRatio --skip-model-wo-parsed-files -p 8
#   bash detection_target_size.sh --model-dir Results/MedVision-detect-v2/<model> --skip-viz --dry-run
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
cat <<'EOF'
Usage: detection_target_size.sh (--task-dir DIR | --model-dir DIR) [options]

Input
  --task-dir DIR              Detection task dir; every model subfolder is analysed and a random_detection/
                              baseline is generated from the first model that has --parsed-dirname.
  --model-dir DIR             One model dir only (no random baseline; the plot then covers that model
                              plus any random_detection/ sibling already present).
  --parsed-dirname NAME       Parsed-records subfolder inside each model dir (default: parsed). Use e.g.
                              llm-parsed_gemma-4-31b for LLM-judge re-parsed records; outputs go into that
                              folder so published summaries are never overwritten.
  --skip-model-wo-parsed-files  Skip models lacking the parsed subfolder (--task-dir only). Recommended
                              with an llm-parsed source: a missing folder is fatal otherwise.
  --limit N                   Samples per JSONL (debug).
  -p, --processes N           Worker processes for metric aggregation.

Plot
  --config YAML               model_display_name map (default: config-detect-boxImgRatio.yaml next to this
                              script; edit folder names to match your Results tree).
  --out-dir DIR               Figure directory (default: ./Figures).
  --save-as-png               Also write a PNG (default output is PDF).
  --skip-viz                  Run step 1 only.

Environment
  --python EXE                Interpreter (default: python).
  --repo-root DIR             Prepend DIR/src to PYTHONPATH (use when medvision_bm is not pip-installed).
  --dry-run                   Print the commands and exit 0.
  -h, --help                  This text.
EOF
}

task_dir=""; model_dir=""; parsed_dirname="parsed"; skip_flag=""; limit=""; procs=""
config="$here/config-detect-boxImgRatio.yaml"; out_dir="./Figures"; png=0; skip_viz=0
PY="python"; repo_root=""; dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-dir) task_dir="$2"; shift 2 ;;
    --model-dir) model_dir="$2"; shift 2 ;;
    --parsed-dirname) parsed_dirname="$2"; shift 2 ;;
    --skip-model-wo-parsed-files) skip_flag="--skip_model_wo_parsed_files"; shift ;;
    --limit) limit="$2"; shift 2 ;;
    -p|--processes) procs="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --save-as-png) png=1; shift ;;
    --skip-viz) skip_viz=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    --repo-root) repo_root="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[detection_target_size] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$task_dir" && -z "$model_dir" ]]; then echo "error: one of --task-dir / --model-dir is required" >&2; exit 2; fi
if [[ -n "$task_dir" && -n "$model_dir" ]]; then echo "error: --task-dir and --model-dir are mutually exclusive" >&2; exit 2; fi
if [[ -n "$skip_flag" && -z "$task_dir" ]]; then echo "error: --skip-model-wo-parsed-files requires --task-dir" >&2; exit 2; fi
if [[ $dry_run -eq 0 ]]; then
  for d in "$task_dir" "$model_dir"; do [[ -z "$d" || -d "$d" ]] || { echo "error: directory not found: $d" >&2; exit 2; }; done
  [[ $skip_viz -eq 1 || -f "$config" ]] || { echo "error: config not found: $config" >&2; exit 2; }
fi
if [[ -n "$repo_root" ]]; then export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"; fi

run() { echo "+ $*"; if [[ $dry_run -eq 0 ]]; then "$@"; fi; }

if [[ $dry_run -eq 0 ]]; then
  "$PY" -c "import medvision_bm" 2>/dev/null || {
    echo "error: medvision_bm is not importable by '$PY'. pip install medvision-bm (mind the torch/transformers pins)" >&2
    echo "       or pass --repo-root <checkout> to use its src/ tree." >&2; exit 3; }
fi

echo "[1/2] box-to-image-ratio metrics (source: $parsed_dirname)"
args=(--parsed_dirname "$parsed_dirname")
[[ -n "$task_dir" ]] && args+=(--task_dir "$task_dir")
[[ -n "$model_dir" ]] && args+=(--model_dir "$model_dir")
[[ -n "$limit" ]] && args+=(--limit "$limit")
[[ -n "$skip_flag" ]] && args+=("$skip_flag")
[[ -n "$procs" ]] && args+=(--processes "$procs")
run "$PY" -m medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random "${args[@]}"

if [[ $skip_viz -eq 1 ]]; then echo "[2/2] skipped (--skip-viz)"; exit 0; fi

echo "[2/2] figure"
in_dir="${task_dir:-$(dirname "$model_dir")}"
tmp_cfg="$(mktemp "${TMPDIR:-/tmp}/boxImgRatio-config.XXXXXX.yaml")"
trap 'rm -f "$tmp_cfg"' EXIT
remap='
import sys, yaml
src, dst, parsed = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
new = {}
for model, display in cfg["model_display_name"].items():
    new[model if model == "random_detection" else f"{model}/{parsed}"] = display
yaml.dump({"model_display_name": new}, open(dst, "w"), default_flow_style=False, allow_unicode=True, sort_keys=False)
'
echo "+ $PY -c <remap config keys to <model>/$parsed_dirname> $config -> $tmp_cfg"
if [[ $dry_run -eq 0 ]]; then "$PY" -c "$remap" "$config" "$tmp_cfg" "$parsed_dirname"; fi
viz=(--config "$tmp_cfg" --in_dir "$in_dir" --out_dir "$out_dir")
# --save_as_png alone REPLACES the pdf (viz_detection_performance_per_boxImgRatio.py:313
# builds `formats` from whichever flags are on), so ask for both to match the help text.
[[ $png -eq 1 ]] && viz+=(--save_as_png --save_as_pdf)
if [[ $dry_run -eq 0 ]]; then mkdir -p "$out_dir"; fi
run "$PY" -m medvision_bm.benchmark.viz_detection_performance_per_boxImgRatio "${viz[@]}"
echo "done. figure: $out_dir/metrics_boxImgRatio-dotline.pdf$( [[ $png -eq 1 ]] && echo ' (+ .png)' )"
[[ $dry_run -eq 1 ]] && echo "dry run: nothing executed."
exit 0
