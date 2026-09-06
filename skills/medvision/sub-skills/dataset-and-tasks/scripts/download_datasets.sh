#!/usr/bin/env bash
# download_datasets.sh -- wrapper around `python -m medvision_bm.benchmark.download_datasets`.
#
# Purpose
#   Download (and build the Arrow cache for) the MedVision configs named by a task-list
#   JSON or a config CSV, with the checks the raw CLI does not do:
#     * refuses to start when MedVision_PLANNER_VERSION is unset (the loader would fail later);
#     * trims whitespace/newlines from HF_TOKEN and SYNAPSE_TOKEN (tokens copied from secret
#       stores often carry a trailing newline, which yields HTTP 401);
#     * strips task-variant suffixes ("-CoT", "-CoT-scaledPS", ...) after the plane token from a
#       task-list JSON into a temporary copy, because dataset configs never carry them and
#       tasks_to_configs() does not remove them;
#     * prints the exact command and stops with --dry-run.
#
# Prerequisites
#   medvision_bm importable by the chosen Python (default: `python` on PATH); network access;
#   disk for the WHOLE source dataset of every config (any config downloads the entire dataset).
#   Gated datasets: FeTA24 -> SYNAPSE_TOKEN; SKM-TEA / ToothFairy2 -> MedVision_SKMTEA_HF_ID /
#   MedVision_ToothFairy2_HF_ID + HF_TOKEN; AbdomenAtlas1.0Mini -> accepted terms + HF_TOKEN.
#
# Usage
#   bash download_datasets.sh --data-dir <data_dir> --tasks-json <tasks.json> [--split test] [--dry-run]
#   bash download_datasets.sh --data-dir <data_dir> --configs-csv <ConfigurationsList_Test.csv> [--split test]
#   Options: --force-download-data   (sets MedVision_FORCE_DOWNLOAD_DATA=true; debug only)
#            --python <interpreter>  (default: python)
#            --dry-run               (print the command, do not run)
#
# Example
#   export MedVision_PLANNER_VERSION=latest
#   bash download_datasets.sh --data-dir ./Data --tasks-json tasks_MedVision-TL-CoT.json --dry-run
set -euo pipefail

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

DATA_DIR=""; TASKS_JSON=""; CONFIGS_CSV=""; SPLIT="test"; FORCE=0; DRY=0; PY="${PYTHON:-python}"
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tasks-json) TASKS_JSON="$2"; shift 2 ;;
    --configs-csv) CONFIGS_CSV="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --force-download-data) FORCE=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$DATA_DIR" ] || { echo "error: --data-dir is required" >&2; exit 2; }
if [ -n "$TASKS_JSON" ] && [ -n "$CONFIGS_CSV" ]; then echo "error: give --tasks-json OR --configs-csv, not both" >&2; exit 2; fi
if [ -z "$TASKS_JSON" ] && [ -z "$CONFIGS_CSV" ]; then echo "error: one of --tasks-json / --configs-csv is required" >&2; exit 2; fi
case "$SPLIT" in train|test) ;; *) echo "error: --split must be train or test" >&2; exit 2 ;; esac
[ -z "$TASKS_JSON" ] || [ -f "$TASKS_JSON" ] || { echo "error: not a file: $TASKS_JSON" >&2; exit 2; }
[ -z "$CONFIGS_CSV" ] || [ -f "$CONFIGS_CSV" ] || { echo "error: not a file: $CONFIGS_CSV" >&2; exit 2; }

# The loader raises "MedVision: annotation version selection required" without this.
if [ -z "${MedVision_PLANNER_VERSION:-}" ]; then
  echo "error: MedVision_PLANNER_VERSION is unset. Use 'latest' for new work, or '1.0.0' plus" >&2
  echo "       MedVision_ACK_RELEASE=1.4.0 to reproduce the leaderboard annotations." >&2
  exit 2
fi

# Sanitise tokens copied from secret stores (trailing newline => HTTP 401).
for var in HF_TOKEN SYNAPSE_TOKEN; do
  if [ -n "${!var:-}" ]; then
    cleaned="$(printf '%s' "${!var}" | tr -d '[:space:]')"
    export "$var=$cleaned"
  fi
done

# Strip variant suffixes after the plane token from a task list (configs never carry them).
TASKS_ARG="$TASKS_JSON"
if [ -n "$TASKS_JSON" ]; then
  TMP_JSON="$(mktemp "${TMPDIR:-/tmp}/medvision_tasks.XXXXXX.json")"
  n_changed="$("$PY" - "$TASKS_JSON" "$TMP_JSON" <<'PYEOF'
import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
rx = re.compile(r"^(.*?_(?:Axial|Coronal|Sagittal))(.*)$")
tasks = json.load(open(src))
out, changed = {}, 0
for name, count in tasks.items():
    m = rx.match(name)
    base = m.group(1) if m else name
    if base != name:
        changed += 1
    out[base] = count
json.dump(out, open(dst, "w"), indent=2)
print(changed)
PYEOF
)"
  if [ "$n_changed" != "0" ]; then
    echo "[download_datasets] $n_changed task name(s) carried a variant suffix (e.g. -CoT); using stripped copy: $TMP_JSON"
    TASKS_ARG="$TMP_JSON"
  else
    rm -f "$TMP_JSON"
  fi
fi

CMD=("$PY" -m medvision_bm.benchmark.download_datasets --data_dir "$DATA_DIR" --split "$SPLIT")
if [ -n "$TASKS_ARG" ]; then CMD+=(--tasks_json "$TASKS_ARG"); else CMD+=(--configs_csv "$CONFIGS_CSV"); fi
[ "$FORCE" -eq 1 ] && CMD+=(--force_download_data)

echo "[download_datasets] MedVision_PLANNER_VERSION=$MedVision_PLANNER_VERSION MedVision_ACK_RELEASE=${MedVision_ACK_RELEASE:-<unset>}"
echo "[download_datasets] note: every config downloads the WHOLE source dataset (all planes and both splits);"
echo "                    --split only selects which Arrow build is materialised."
echo "[download_datasets] command:"
printf '  %q' "${CMD[@]}"; echo
if [ "$DRY" -eq 1 ]; then echo "[download_datasets] dry run: nothing executed."; exit 0; fi
exec "${CMD[@]}"
