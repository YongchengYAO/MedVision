#!/usr/bin/env bash
#
# LLM-as-Judge output parsing -- the single entry point, prep through reports.
#
# Runs the whole pipeline in order, stopping at the first failed gate. The steps
# and what each one costs are listed under "Steps" below, and by --list. They come
# from one table in the script (STEP_TABLE), so this help and --list cannot drift
# apart -- the previous copy in this comment had gone stale, still claiming the
# pilot ran on TL only after the loop was generalised to every task.
#
# Usage:
# Any cwd works -- the script re-roots itself to the repo:
#   bash script/llm-parsing/run_llm_parsing.sh            # everything, in order
#   bash script/llm-parsing/run_llm_parsing.sh --fresh    # FULL NEW RE-JUDGE
#   bash script/llm-parsing/run_llm_parsing.sh --yes      # (-y) no prompts
#   bash script/llm-parsing/run_llm_parsing.sh smoke      # just that step
#   bash script/llm-parsing/run_llm_parsing.sh --from full  # that step onward
#   bash script/llm-parsing/run_llm_parsing.sh --list     # what would run, only
#   bash script/llm-parsing/run_llm_parsing.sh --judges   # readers you can pick
#   bash script/llm-parsing/run_llm_parsing.sh --judge gemma-4-31b   # a 2nd reader
#
# --fresh is the one-command full re-judge: it additionally deletes every
# llm-parsed*/ directory in `prep`, so no record from a previous roster survives
# into the new reports. Without it those directories are reported and kept, because
# Stage 2 rewrites them file by file and cannot remove what it no longer produces.
# Still destructive, so still behind the same confirmation prompt.
#
# --judge picks WHICH reader runs. Only gemma-4-31b is registered today, so the
# flag is rarely needed -- but every reader writes to its OWN
# judge-out_<task>_<reader>.jsonl and llm-parsed_<reader>/, so a second reader can
# be added and share the same Results tree and the same queues without touching
# the first one's output. That is the point of the mechanism: re-reading one
# corpus with two readers is how you find out whether a recovery rate is a
# property of the responses or of the reader. It also means --fresh under one
# reader leaves the others alone. `judge_config.py --list` shows what is
# registered.
#
# Readers can need DIFFERENT environments and may not share a venv (Gemma-4 wants
# transformers>=5.5, which a vLLM declaring transformers<5 cannot host). Build the
# right one and point PYTHON at it:
#   bash script/llm-parsing/setup_judge_env.sh --judge gemma-4-31b
#
# Environment (no path below is machine-specific; all have working defaults):
#   JUDGE           reader key, same as --judge.
#   JUDGE_MODEL     an explicit checkpoint, overriding the registry's hub id --
#                   e.g. a local mirror of the same weights.
#   JUDGE_MODEL_HF  upstream id (default google/gemma-4-31B-it). vLLM fetches it
#                   on first load, so HF_HOME needs room for ~62 GB.
#   MEDVISION_DS_SRC  the medvision_ds checkout, if it is not already importable
#                   and not found by the short upward search.
#   PYTHON        interpreter holding torch + vllm. A GPU pod usually has several
#                 python3 on PATH and the first one wins, which is how a two-A100
#                 pod reports "no CUDA device". Create one with
#                 `bash script/llm-parsing/setup_judge_env.sh`.
#   TASKS         default "TL AD Detection"
#   TASK_DIR_<task>    re-point one task at another Results tree (task = TL, AD
#                 or Detection). ROSTER_YAML_<task> swaps its roster YAML in the
#                 same move. How run_llm_parsing_ood.sh reuses this pipeline for
#                 the OOD splits; unset means the main benchmark tree + the
#                 paper roster, exactly as before.
#   TP            GPUs ONE engine spans (tensor parallel), overriding the reader's
#                 registry value. Default 1: with NUM_SHARDS=1 below, the whole
#                 run stays on a single GPU unless BOTH knobs are raised. Total
#                 GPUs used = NUM_SHARDS x TP. Raise TP to FIT the weights, not
#                 to go faster: shards share nothing and scale near-linearly,
#                 while TP adds an all-reduce per layer. The case that needs it
#                 is a dense model whose weights crowd out the KV cache: a 31B
#                 dense model is ~62 GB and leaves an 80 GB card almost nothing
#                 to batch with, hence the registry's TP=2. A small or sparse
#                 model belongs at TP=1 with the extra cards running extra shards.
#   NUM_SHARDS    default 1 (single GPU). e.g. TP=2 NUM_SHARDS=2 uses four GPUs.
#   GPU_NUM       total GPUs to use -- the one-knob spelling of the layout above.
#                 NUM_SHARDS is derived as GPU_NUM / TP (must divide evenly), so
#                 GPU_NUM=4 TP=2 is two shards of two cards, and GPU_NUM=4 alone
#                 is four single-card shards. An explicit NUM_SHARDS wins over
#                 it; with neither set the single-GPU default stands.
#   PROCS         CPU worker count for Stages 0/2/3 (default 32)
#   MOCK=1        substitute the regex stand-in for the judge. Exercises every
#                 gate on a CPU box. NEVER for reported numbers.
#   YES=1         skip the confirmation prompt in `prep` (same as --yes/-y)
#
# Resumable throughout: Stage 1 output is keyed by qid, so re-running skips
# finished work. Kill it and restart at will.

set -euo pipefail

# This script lives in script/llm-parsing/ but every data path below (Results/,
# Data/, unit-test/) is repo-root-relative, so re-root to the repo -- two levels up
# from here -- no matter where the caller invoked it from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLM=script/llm-parsing

# --------------------------------------------------------------- environment --
PY="${PYTHON:-python3}"
# Whether the caller CHOSE this interpreter or got the fallback. Worth tracking
# because "the first python3 on PATH is the wrong one" is this pipeline's single
# most recurrent failure, and it never says so itself -- it surfaces as a missing
# module, or as "no CUDA device" on a pod with four. An unset PYTHON turns those
# into a one-variable fix, but only if the diagnosis is printed.
PY_EXPLICIT=1; [ -n "${PYTHON:-}" ] || PY_EXPLICIT=0
TASKS="${TASKS:-TL AD Detection}"
PROCS="${PROCS:-32}"
# Single-GPU by default: one shard (NUM_SHARDS=1) spanning one card (TP=1),
# overriding the reader's registry TP -- judge_env.sh, sourced below, adopts a
# pre-set TP as JUDGE_TP. A ~62 GB reader on one 80 GB card keeps only ~10 GB
# for KV cache, so this trades throughput for footprint; raise BOTH knobs for a
# multi-GPU layout (TP=2 NUM_SHARDS=2 = four GPUs). Set here rather than in
# run_judge so smoke, the sweep and the OOD driver all see one resolution.
TP="${TP:-1}"
# GPU_NUM is the total-GPU spelling of the same layout: NUM_SHARDS derived as
# GPU_NUM / TP. Resolved here, before the default below, so an explicit
# NUM_SHARDS still wins and every consumer (smoke, sweep, OOD driver) sees one
# answer. Divisibility is enforced rather than rounded: silently dropping the
# remainder would leave paid-for cards idle while claiming to use them.
if [ -z "${NUM_SHARDS:-}" ] && [ -n "${GPU_NUM:-}" ]; then
  if ! [ "${GPU_NUM}" -ge 1 ] 2>/dev/null; then
    printf '\n\033[1;31mFAILED: GPU_NUM=%s is not a positive integer\033[0m\n' "${GPU_NUM}" >&2; exit 1
  fi
  if [ $(( GPU_NUM % TP )) -ne 0 ]; then
    printf '\n\033[1;31mFAILED: GPU_NUM=%s is not divisible by TP=%s -- each shard spans TP GPUs, so the total must be a multiple of TP\033[0m\n' "${GPU_NUM}" "${TP}" >&2; exit 1
  fi
  NUM_SHARDS=$(( GPU_NUM / TP ))
fi
NUM_SHARDS="${NUM_SHARDS:-1}"
MOCK="${MOCK:-0}"
# Set by --fresh. Makes `prep` additionally delete every llm-parsed*/ directory,
# which is what turns "full new re-judge" into one command.
FRESH="${FRESH:-0}"
PILOT_LIMIT="${PILOT_LIMIT:-100}"
SMOKE_ROWS="${SMOKE_ROWS:-200}"
# Grammar-constrained decoding. Empty lets run_judge_vllm choose its own default.
# xgrammar builds a per-token bitmask on the CPU, and the schema's free-form
# "span" string makes that mask expensive -- measured here at ~1 core busy with
# the GPU at 0%, i.e. decode fully serialized behind the CPU. The v2 prompt shows
# the object outright, so the grammar is a belt-and-braces measure rather than
# the thing producing valid JSON; --min_valid_rate is what actually guards it.
STRUCTURED="${STRUCTURED:-}"

# Space-separated prompt fingerprints to accept BESIDES the current one, when a
# judge-out file legitimately holds rows from more than one prompt. That happens
# on every repair pass: --max_tokens is part of the fingerprint, so raising the
# budget for the broken rows restamps them and leaves the good rows on the old
# stamp. Without this, stage2 refuses the file and the driver cannot reproduce
# reports that were in fact produced correctly. Extra entries are inert -- the
# list is a whitelist, so one variable covers every task.
#
# The default is the COMPLETE stamp history of the production judge-outs, which
# is why `analyze` reproduces out of the box even though the current prompt
# (max_tokens=4096 default) postdates every row on disk:
#   09ee44a311e85670  TL         @1024      dd3c7fb2d50255db  TL         @3072
#   02a90adc517baab6  AD:dist    @1024      5cfdd478b855e9e4  AD:dist    @3072
#   a3fd2c4a87a9ce4e  AD:angle   @1024      56de1c8d5a7719b0  AD:angle   @3072
#   49c9fbdcf069aef9  Detection  @256       54e8d9ef545df4a6  Detection  @512
# A stamp NOT in this list still aborts, which is the point of the check.
# The 2026-08-08 skeleton-placeholder fix ("<number>" instead of [0.0]) changed
# the rendered prompt: the on-disk @4096 queues no longer match the current
# code, so `run_judge` aborts until stage 0 rebuilds them. Expected; rebuild.
ACCEPT_FP="${ACCEPT_FP:-09ee44a311e85670 dd3c7fb2d50255db 02a90adc517baab6 5cfdd478b855e9e4 a3fd2c4a87a9ce4e 56de1c8d5a7719b0 49c9fbdcf069aef9 54e8d9ef545df4a6}"

say()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

# --judge is read HERE, ahead of the option loop at the bottom, because
# judge_env.sh resolves the checkpoint at source time and every step function reads
# the result. The option loop still consumes the flag, so it does not fall through
# into SELECT and get mistaken for a step name.
_ji=1
while [ "${_ji}" -le $# ]; do
  if [ "${!_ji}" = "--judge" ]; then
    _jv=$((_ji + 1))
    [ "${_jv}" -le $# ] || die "--judge needs a value (see: ${PY} ${LLM}/judge_config.py --list)"
    JUDGE="${!_jv}"
    break
  fi
  _ji=$((_ji + 1))
done
unset _ji _jv

# Judge checkpoint resolution and HF secret hygiene, shared verbatim with
# test-sweep.sh. Sourced rather than duplicated: the judge_model string is stamped
# into judge-out and resume compares provenance, so the two entry points resolving
# differently aborts a legitimate resume.
. "${LLM}/judge_env.sh"

# Stage 3 imports medvision_bm from this checkout, not the installed package: the
# installed one predates --resps_key and would reject the flag.
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

# medvision_ds ships as a separate checkout and the summarizers import it
# transitively, so Stage 3 needs it on the path. Resolved rather than hardcoded --
# an absolute path here pinned the whole pipeline to one machine.
#
# Called from just before the step loop, and only when a selected step actually
# reaches Stage 3. Checking eagerly for every invocation would block `smoke` on a
# box that has no medvision_ds and needs none; checking lazily inside run_analysis
# would surface the failure AFTER the 13-hour judge, which is worse.
resolve_medvision_ds() {
  if "${PY}" -c 'import medvision_ds' >/dev/null 2>&1; then return 0; fi

  local cand="" base p c
  if [ -n "${MEDVISION_DS_SRC:-}" ]; then
    [ -d "${MEDVISION_DS_SRC}/medvision_ds" ] \
      || die "MEDVISION_DS_SRC=${MEDVISION_DS_SRC} contains no medvision_ds/ package."
    cand="${MEDVISION_DS_SRC}"
  else
    # Search upward from the MAIN checkout rather than $PWD. In a git worktree the
    # root sits several levels below the repository it belongs to, so a search
    # anchored at the worktree walks .claude/worktrees/ and never reaches the
    # sibling checkouts where medvision_ds actually lives.
    base="$(git rev-parse --git-common-dir 2>/dev/null || echo .git)"
    base="$(cd "$(dirname "${base}")" 2>/dev/null && pwd || pwd)"
    p="${base}"
    for _ in 1 2 3 4 5; do
      p="$(dirname "${p}")"
      [ "${p}" = "/" ] && break
      for c in "${p}/MedVision/src" "${p}/medvision_ds/src"; do
        if [ -d "${c}/medvision_ds" ]; then cand="${c}"; break 2; fi
      done
    done
  fi

  [ -n "${cand}" ] || die "medvision_ds is not importable and no checkout was found.
       Stage 3 needs it: the task summarizers import it transitively.
       Install it:      ${PY} -m medvision_bm.benchmark.install_medvision_ds --data_dir Data
       or point at it:  MEDVISION_DS_SRC=/path/to/MedVision/src bash ${LLM}/$(basename "$0")"

  export PYTHONPATH="${PYTHONPATH}:${cand}"
  "${PY}" -c 'import medvision_ds' >/dev/null 2>&1 \
    || die "found ${cand}, but medvision_ds still does not import under ${PY}."
  note "medvision_ds  : ${cand}"
}

# Stages 2-4 import medvision_bm, whose package __init__ pulls `datasets` and whose
# summarizers pull `nibabel`. Stage 1 needs neither, so a judge env missing them
# builds cleanly, passes setup verification, runs the whole GPU sweep and dies on
# the FIRST LINE of Stage 2 -- the most expensive possible moment to learn this.
#
# Probes the REAL import that apply_judge performs, not a list of module names.
# A list is a second copy of the dependency set and drifts; this cannot. It is also
# why the medvision_ds check beside it was not enough: medvision_ds imports fine
# WITHOUT datasets, so that probe passed and Stage 2 still failed.
#
# Args: $1 "hard" to abort, anything else to warn only.
# Shared remedy text for "this interpreter cannot do X". Split out because the right
# advice depends on WHY, not on which stage noticed. Two different problems produce
# an import failure here and the wrong remedy makes things worse: if PYTHON is unset
# the interpreter is merely the first python3 on PATH, and "pip install into it"
# puts the packages somewhere that is not where the run happens, papering over the
# mismatch instead of fixing it. So never print a pip command against an interpreter
# nobody chose.
#
# Args: $1 the install hint to use when the interpreter WAS chosen deliberately.
_interpreter_remedy() {
  if [ "${PY_EXPLICIT}" = "0" ]; then
    printf '%s' "PYTHON is NOT set, so this is whichever python3 came first on PATH:
             $(command -v "${PY}" 2>/dev/null || echo '<not found>')
       That is probably not the environment you built for the judge, and installing
       into it would hide the mismatch rather than fix it. Point the pipeline at the
       judge environment instead:
           export PYTHON=<judge-env>/bin/python
       Build one with:  bash ${LLM}/setup_judge_env.sh --judge ${JUDGE_KEY}
       (it installs the CPU-stage dependencies too)"
  else
    printf '%s' "$1"
  fi
}

# Stage 1 needs vLLM. Checked here, up front, for the same reason as the Stage-2/4
# probe below -- and they are genuinely INDEPENDENT: an interpreter can satisfy one
# and not the other. Conda base imports medvision_bm happily and has no vllm; a bare
# judge venv is the exact reverse. Probing only one leaves the other to surface
# mid-run, after the queues are built or after the sweep.
#
# test-sweep.sh has a far more detailed GPU preflight (driver/CUDA/device list) and
# keeps it -- this is only the cheap "is vllm even here" gate, hoisted ahead of the
# CPU work so a wrong interpreter costs seconds instead of a queue build.
check_judge_imports() {
  if [ "${MOCK}" = "1" ]; then return 0; fi   # the stand-in loads no model
  # Written as `if ! ...; then die`, NOT `if ...; then return 0`. The early-return
  # form reads fine and silently skipped the GPU probe below on every interpreter
  # that DOES have vllm -- i.e. on exactly the environments this check is for.
  local err
  if ! err="$("${PY}" -c 'import vllm' 2>&1)"; then
    die "Stage 1 cannot import vllm under ${PY}.
       ${err##*$'\n'}
       $(_interpreter_remedy "That interpreter has no vllm. Build the judge
       environment and point PYTHON at it:
           bash ${LLM}/setup_judge_env.sh --judge ${JUDGE_KEY}")"
  fi

  # ...and that there is a GPU to run it on. Deliberately minimal: test-sweep.sh
  # owns the real diagnosis (driver vs interpreter vs pod spec) and keeps it. The
  # only job here is ORDER. Without this, `pilot` on a GPU-less box builds three
  # limit-100 queues -- 50,400 rows for Detection alone -- and only then reaches
  # test-sweep's check, so the wrong pod costs minutes of CPU before saying so.
  if [ "${SKIP_GPU_CHECK:-0}" = "1" ]; then return 0; fi
  if ! err="$("${PY}" -c 'import torch; n=torch.cuda.device_count(); assert n, "no CUDA device"; torch.zeros(1, device="cuda")' 2>&1)"; then
    die "Stage 1 needs a GPU and this interpreter cannot use one.
       ${err##*$'\n'}
       interpreter : ${PY}
       Stages that need no GPU work fine here (prep, stage0, analyze); run the
       judge on a GPU pod, or override once you are sure:  SKIP_GPU_CHECK=1
       test-sweep.sh prints the full driver/pod diagnosis if you need it."
  fi
}

check_analysis_imports() {
  # Probes BOTH stages' entry imports, because they are not the same import and the
  # narrower one passes while the wider one fails. Stage 2 needs `cal_metrics`;
  # Stage 3 imports the task summarizer, which reaches medvision_utils ->
  # lmms_eval.tasks -> lmms_eval.utils and drags in the whole harness (90
  # third-party modules). Probing only cal_metrics let a run get through Stage 2 and
  # die at Stage 3 on `pytz` -- the same shape as the medvision_ds probe passing
  # while Stage 2 died on `datasets`. Probe what the stage actually does.
  #
  # Only the summarizers for the SELECTED tasks: a TASKS=TL run must not be blocked
  # by a dependency that only the Detection summarizer pulls.
  #
  # medvision_ds.utils.{benchmark_planner,preprocess_utils} are probed too, and they
  # are NOT reachable by importing the summarizers. Stage 3 resolves a per-dataset
  # benchmark plan with importlib.import_module WHILE processing records, so those
  # dependencies surface partway through the stage -- wrapped as "Error loading
  # benchmark plan for <dataset>", which reads like corrupt data rather than a
  # missing package. All 17 plan modules import exactly these two, so probing them
  # covers the lazy path without importing 17 modules.
  local err mods="" t
  for t in ${TASKS}; do mods="${mods}import medvision_bm.benchmark.$(summarizer "${t}");"; done
  mods="${mods}import medvision_ds.utils.benchmark_planner;"
  mods="${mods}import medvision_ds.utils.preprocess_utils;"
  if err="$("${PY}" -c "from medvision_bm.utils.parse_utils import cal_metrics;${mods}" 2>&1)"; then
    return 0
  fi
  local last="${err##*$'\n'}"

  local remedy
  remedy="$(_interpreter_remedy "Stage 1 does not need this, so the judge environment
       can look healthy and still fail here. Install the CPU-stage dependencies:
           ${PY} -m pip install -r ${LLM}/requirements-cpu-stages.txt")"

  if [ "${1:-hard}" = "hard" ]; then
    die "Stages 2-4 cannot import medvision_bm under ${PY}.
       ${last}
       ${remedy}"
  fi
  printf '\n\033[1;33m  WARNING\033[0m Stages 2-4 will fail under this interpreter:\n'
  note "  ${PY}: ${last}"
  note "  Stage 1 (the GPU sweep) does not need it and will run, but the reports"
  note "  afterwards will not. Fix it now rather than after the sweep."
  printf '  %s\n' "${remedy}"
}

# TASK_DIR_<task> re-points a task at a different Results tree -- how
# run_llm_parsing_ood.sh reuses this whole pipeline for the OOD splits without a
# second copy of it. Defaults are the main benchmark trees, as before. The same
# variables are honoured by test-sweep.sh and unit-test/llm-parsing/test-8.py,
# so every stage of one run resolves the same tree.
task_dir() {
  case "$1" in
    TL)        echo "${TASK_DIR_TL:-Results/MedVision-TL-v2-CoT}" ;;
    AD)        echo "${TASK_DIR_AD:-Results/MedVision-AD-v2-CoT}" ;;
    Detection) echo "${TASK_DIR_Detection:-Results/MedVision-detect-v2}" ;;
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
}
# ROSTER_YAML_<task> overrides the roster the same way (the OOD splits are judged
# over a 3-model roster, not the paper's 18). Empty means "use judge_config's
# default for the task": the stages then fall back to DEFAULT_ROSTER_YAML
# themselves, so the default table stays in one place instead of growing a shell
# copy that can drift.
roster_yaml() {
  case "$1" in
    TL)        echo "${ROSTER_YAML_TL:-}" ;;
    AD)        echo "${ROSTER_YAML_AD:-}" ;;
    Detection) echo "${ROSTER_YAML_Detection:-}" ;;
  esac
}
summarizer() {
  case "$1" in
    TL) echo summarize_TL_task ;;
    AD) echo summarize_AD_task ;;
    Detection) echo summarize_detection_task ;;
  esac
}
# TL is the only task with a removed-samples filter; passing it elsewhere errors.
extra_sum() { [ "$1" = "TL" ] && echo "--removed_samples_dir Data/Datasets" || true; }

# Resolve the GPU list into the named array. Honours a pre-set
# CUDA_VISIBLE_DEVICES so this cooperates with a scheduler that already narrowed
# the pod: setting CUDA_VISIBLE_DEVICES=$S in a child would REPLACE the parent's
# filter and re-index against physical GPUs, so `CUDA_VISIBLE_DEVICES=2,3 ...`
# would otherwise silently land on physical GPUs 0 and 1. Same rule as
# test-sweep.sh, which owns the sweep's copy.
resolve_devices() {
  local -n _out="$1"
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _out <<< "${CUDA_VISIBLE_DEVICES}"
  else
    local n
    n="$(${PY} -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
    [ "${n}" -ge 1 ] 2>/dev/null || n=1
    mapfile -t _out < <(seq 0 $((n - 1)))
  fi
}

# ---------------------------------------------------------------------- prep --
# The only destructive step, and it is required rather than tidy-up: every queue
# and judge-out on disk predates the current prompt. Stage 1 refuses to resume an
# output file whose prompt stamp differs -- rows are skipped by qid, and qid has
# no prompt component, so a stale file would otherwise report every row already
# done, make zero model calls, and ship the previous prompt's answers.
# Every judge-out file belonging to the SELECTED reader, and no other reader's.
# The default reader's suffix is empty, so the obvious `judge-out_TL*` also matches
# judge-out_TL_gemma-4-31b.jsonl and would archive a second reader's thirteen-hour
# sweep during a run that has nothing to do with it. Anchoring on `.jsonl` -- and
# listing the _limit variant separately -- makes each reader's set exact. Archives
# (.v1) and shards (.n4.shard0) hang off those stems, hence the trailing `*`.
judge_out_ls() {
  local T="$1" d
  d="$(task_dir "${T}")"
  ls "${d}/judge-out_${T}${JUDGE_SUFFIX}"*.jsonl* 2>/dev/null || true
}

# Same rule for Stage 2's output directories: each reader owns
# llm-parsed<reader_suffix>/, so a --fresh run under one reader cannot delete
# another's records.
#
# Both globs are anchored on JUDGE_SUFFIX, which judge_config guarantees is
# non-empty for every registered reader. That is what makes them safe: an empty
# suffix would widen these to `judge-out_<task>*` and `llm-parsed*`, matching every
# reader at once -- turning --fresh into a much bigger delete than it counted and
# printed. See the warning in judge_config.JUDGE_MODELS; test-11 pins it.
#
# Scoped to TASKS via task_dir(), like judge_out_ls above. The previous
# whole-tree glob (Results/*/*/llm-parsed...) predates the OOD splits: once
# several trees can hold one reader's records, `--fresh` on one split must not
# delete the main sweep's 774 llm-parsed directories along with it.
llm_parsed_ls() {
  local T
  for T in ${TASKS}; do
    ls -d "$(task_dir "${T}")"/*/llm-parsed"${JUDGE_SUFFIX}"* 2>/dev/null || true
  done
}

step_prep() {
  say "prep -- retire artifacts from the previous prompt (${JUDGE_KEY})"
  local outs queues n_out n_q
  # Anything already retired is skipped, so re-running prep is a no-op rather than
  # producing judge-out_TL.jsonl.v1.v1.v1 -- which it did, before this filter.
  # Scoped to TASKS, like every other step. prep used to glob the whole Results
  # tree, so `TASKS=TL bash ...` retired AD and Detection output too -- a blast
  # radius that did not narrow with the documented knob that narrows the run.
  local T
  outs=""; queues=""
  for T in ${TASKS}; do
    outs="${outs}$(judge_out_ls "${T}" | grep -v '\.v[0-9]\+$' || true)
"
    queues="${queues}$(ls "$(task_dir "${T}")"/judge-queue_"${T}"*.jsonl 2>/dev/null || true)
"
  done
  n_out=$(printf '%s' "${outs}"   | grep -c . || true)
  n_q=$(printf '%s'   "${queues}" | grep -c . || true)

  # Counted BEFORE the prompt, so --fresh can state its real blast radius rather
  # than delete 54 directories the user was never shown. `|| true` is load-bearing
  # for the same reason as the two counters above: an unmatched glob leaves `ls`
  # exiting 2 and pipefail carries it into the assignment.
  local n_parsed=0
  if [ "${FRESH}" = "1" ]; then
    n_parsed=$(llm_parsed_ls | grep -c . || true)
  fi

  # --fresh must be able to act even when there is nothing to retire: after an
  # aborted run the judge-out is already .v1 and the queues are already gone, yet
  # the llm-parsed*/ directories -- the whole point of the flag -- remain.
  if [ "${n_out}" = "0" ] && [ "${n_q}" = "0" ] && [ "${n_parsed}" = "0" ]; then
    note "nothing to retire"; return 0
  fi
  note "judge-out files to move aside (kept, suffixed .v1): ${n_out}"
  note "queue files to delete (pure derived data):          ${n_q}"
  if [ "${FRESH}" = "1" ]; then
    note "llm-parsed*/ directories to DELETE (--fresh):       ${n_parsed}"
  fi
  if [ "${YES:-0}" != "1" ]; then
    printf '\n  Proceed? [y/N] '
    read -r reply </dev/tty || reply=n
    case "${reply}" in [yY]*) ;; *) die "prep declined -- nothing changed" ;; esac
  fi
  # Moved, not deleted: these files are the evidence for why the v1 sweep failed.
  # Never overwrite an existing archive -- a second prep cycle targets a fresh
  # judge-out whose .v1 destination is already occupied by the first cycle's,
  # and a plain `mv` would destroy exactly the evidence this step exists to keep.
  for p in ${outs}; do
    [ -e "$p" ] || continue
    local dest="$p.v1" i=2
    while [ -e "${dest}" ]; do dest="$p.v${i}"; i=$((i + 1)); done
    mv -- "$p" "${dest}"
  done
  for p in ${queues}; do rm -f -- "$p"; done
  rm -rf Results/*/.judge-shards_* 2>/dev/null || true

  if [ "${FRESH}" = "1" ] && [ "${n_parsed}" -gt 0 ]; then
    # Deleted from the SAME listing that was counted and shown, so the confirmation
    # prompt cannot describe one blast radius while the removal performs another.
    # It could before: the count came from a glob and the removal from
    # `find Results -type d ...`, and in a git worktree Results is a SYMLINK to the
    # main tree -- find defaults to -P and will not follow a symlink named as a
    # starting point, so it matched nothing and exited 0 while the count reported
    # 54. Glob expansion does traverse the symlink, which is why the listing works
    # and is now the only path. (`find -L` would have fixed the symlink and broken
    # something worse: it follows symlinks found INSIDE the tree too, which is far
    # too wide for an `rm -rf`.)
    llm_parsed_ls | while IFS= read -r d; do
      [ -n "${d}" ] || continue
      rm -rf -- "${d}"
    done
    note "deleted ${n_parsed} llm-parsed${JUDGE_SUFFIX}*/ directories"
  fi
  note "done"

  # Reported, not removed. Stage 2 rewrites these file by file, so a directory
  # left from an earlier roster can keep records no current run would produce --
  # but they are also the previous results, and deleting those is the user's call.
  # `|| true` is load-bearing, like the three counters above. Under
  # `set -euo pipefail` a glob with no match leaves `ls` exiting 2, pipefail
  # carries that through `wc -l`, the command substitution inherits it and the
  # bare assignment fails -- killing the driver HERE, after the destructive mv/rm
  # above and before stage0, with no FAILED banner. (Written `local old=$(...)`
  # the builtin's own status would have masked it; the split is what exposed it.)
  local old
  old=$(llm_parsed_ls | grep -c . || true)
  if [ "${old}" -gt 0 ]; then
    printf '\n'
    note "NOTE ${old} existing llm-parsed${JUDGE_SUFFIX}*/ directories were left in place."
    note "     Stage 2 overwrites them per file; remove them first if the roster"
    note "     or the file set has changed since they were written. Either re-run"
    note "     with --fresh, which does exactly this, or by hand:"
    # The trailing slash is REQUIRED and is not cosmetic -- see the --fresh branch
    # above for why `find Results` silently matches nothing in a worktree. This
    # by-hand form is deliberately reader-agnostic: someone typing it wants
    # everything gone, and --fresh is the reader-scoped path.
    note "       find Results/ -type d -name 'llm-parsed*' -exec rm -rf {} +"
  fi
}

# -------------------------------------------------------------------- stage0 --
step_stage0() {
  say "stage0 -- queues + regex baseline (CPU)"
  local RY
  for T in ${TASKS}; do
    note "building ${T} ..."
    RY="$(roster_yaml "${T}")"
    "${PY}" "${LLM}/build_judge_queue.py" --task_type "${T}" \
        --task_dir "$(task_dir "${T}")" -p "${PROCS}" \
        ${RY:+--config_yaml "${RY}"} \
      || die "stage0 ${T}: a gate failed (see the [GATE FAIL] line above)"
  done
}

# --------------------------------------------------------------------- smoke --
# GATE G8. The v1 sweep returned 3 valid rows out of 43,938 and ran for thirteen
# GPU-hours because nothing checked what was coming back. This is that check, and
# it costs minutes.
step_smoke() {
  say "smoke -- GATE G8: does the judge return the required schema?"
  local T Q OUT DIR
  T=$(set -- ${TASKS}; echo "$1")
  Q="$(task_dir "${T}")/judge-queue_${T}.jsonl"
  [ -f "${Q}" ] || die "smoke: missing ${Q} -- run stage0 first"
  DIR="$(mktemp -d)"; OUT="${DIR}/smoke_${T}.jsonl"

  local MOCK_ARG=""; [ "${MOCK}" = "1" ] && MOCK_ARG="--mock"
  # One process per GPU, exactly as the sweep runs. Loading a 39 GB checkpoint
  # takes ~15 minutes whether one card or four are used, so a single-GPU smoke
  # would pay that cost, leave the other cards idle, and still not exercise the
  # launch path the long run depends on.
  local -a DEV; resolve_devices DEV
  # devices/TP, not devices: one shard spans JUDGE_TP GPUs. Identical to the rule
  # test-sweep.sh applies, because this launches the same way.
  local N="${NUM_SHARDS:-$(( ${#DEV[@]} / JUDGE_TP ))}"
  [ "${N}" -ge 1 ] || N=1
  # Same over-subscription guard test-sweep.sh applies to the identical launch
  # pattern. Without it the `${DEV[$S]:-$S}` fallback below resolves an
  # out-of-range shard to a RAW physical ordinal outside the allocation, so the
  # gate that exists to prevent wasted GPU time OOMs on a card it was never given.
  # ...but NOT in MOCK mode. The stand-in loads no model and touches no GPU, and
  # the documented CPU-box invocation (`MOCK=1 bash run_llm_parsing.sh`) resolves
  # an EMPTY device list, so an unguarded comparison aborts the one configuration
  # that has nothing to over-subscribe.
  if [ "${MOCK}" != "1" ] && [ $(( N * JUDGE_TP )) -gt "${#DEV[@]}" ]; then
    die "smoke: NUM_SHARDS=${N} x TP=${JUDGE_TP} = $(( N * JUDGE_TP )) GPUs, but only
       ${#DEV[@]} are visible [${DEV[*]}]. Each shard loads its own copy of the judge
       at gpu_memory_utilization=0.90; overlapping them will OOM.
       Lower NUM_SHARDS, lower TP, or expose more GPUs."
  fi
  note "sharding the smoke across ${N} shard(s) x TP ${JUDGE_TP}: [${DEV[*]}]"

  if [ "${N}" -le 1 ] || [ "${MOCK}" = "1" ]; then
    "${PY}" "${LLM}/run_judge_vllm.py" --queue "${Q}" --out "${OUT}" \
        --model "${JUDGE_MODEL}" --judge "${JUDGE_KEY}" \
        --tensor_parallel_size "${JUDGE_TP}" \
        --limit_rows "${SMOKE_ROWS}" --keep_raw \
        ${STRUCTURED:+--structured ${STRUCTURED}} ${MOCK_ARG} \
      || die "smoke: the judge did not clear --min_valid_rate. Read the printed
       example final_answer before spending GPU hours. Do NOT run 'full'."
  else
    local -a pids=(); local S rc=0 SHARD_DEV
    for S in $(seq 0 $((N - 1))); do
      # A slice of the device list, shared with test-sweep.sh via judge_env.sh.
      SHARD_DEV="$(judge_shard_devices "${S}" "${JUDGE_TP}" "${DEV[@]}")" \
        || die "smoke: not enough GPUs for shard ${S} at TP=${JUDGE_TP}"
      CUDA_VISIBLE_DEVICES="${SHARD_DEV}" "${PY}" "${LLM}/run_judge_vllm.py" \
        --queue "${Q}" --out "${OUT}.shard${S}" --shard "${S}" --num_shards "${N}" \
        --model "${JUDGE_MODEL}" --judge "${JUDGE_KEY}" \
        --tensor_parallel_size "${JUDGE_TP}" \
        --limit_rows "${SMOKE_ROWS}" --keep_raw \
        ${STRUCTURED:+--structured ${STRUCTURED}} &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}" || rc=1; done
    [ "${rc}" = "0" ] || die "smoke: a shard failed -- see the abort message above.
       Do NOT run 'full' until the judge clears this gate."
    # Enumerate explicitly and require every shard file, exactly as test-sweep.sh
    # does: a glob orders shard10 before shard2, and a shard that exited 0 having
    # written nothing would otherwise yield a short file the assertions below
    # would score as if it were complete.
    local -a parts=(); for S in $(seq 0 $((N - 1))); do
      [ -f "${OUT}.shard${S}" ] || die "smoke: missing ${OUT}.shard${S}"
      parts+=("${OUT}.shard${S}")
    done
    cat "${parts[@]}" > "${OUT}"
  fi
  "${PY}" - "${OUT}" <<'PY' || die "smoke: gate assertions failed"
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
c = collections.Counter(r["judge_status"] for r in rows)
rate = c["ok"] / max(len(rows), 1)
print(f"  {len(rows)} rows, valid={rate:.1%}  {dict(c)}")
assert rows, "no rows written"
assert rate >= 0.95, f"valid rate {rate:.1%} below 95%"
ok = [r for r in rows if r["judge_status"] == "ok"]
present = [r for r in ok if (r.get("final_answer") or {}).get("status") == "present"]
print(f"  status present={len(present)}  no_conclusion={len(ok)-len(present)}")
spans = [r["final_answer"]["span"] for r in present if r["final_answer"].get("span")]
assert spans, "no span was ever quoted -- span verification would reject everything"
print(f"  example span: {spans[0][:90]!r}")
PY
  note "G8 PASSED"
}

# ------------------------------------------------------------ judge (stage 1) --
# Delegates to test-sweep.sh, which owns the GPU preflight, the device-list
# resolution and the shard/merge machinery.
run_judge() {
  local limit="$1"
  # Exported rather than written as a command prefix. A prefix assignment must be
  # literal at parse time: `${STRUCTURED:+STRUCTURED=none} bash ...` expands to a
  # WORD in command position, so bash looks for a command called "STRUCTURED=none"
  # and reports "command not found". Only-if-set is preserved by guarding the
  # export, which also keeps test-sweep.sh's own defaults intact when unset.
  export TASKS LIMIT="${limit}" MOCK JUDGE_MODEL
  # The sweep names its output files after the reader and stamps the reader onto
  # every row, so it must be told which one this is -- judge_env.sh would
  # otherwise re-resolve the DEFAULT reader in the child and write one reader's
  # answers into another's file.
  export JUDGE="${JUDGE_KEY}" TP="${JUDGE_TP}"
  export PYTHON="${PY}"
  [ -n "${STRUCTURED}" ]      && export STRUCTURED
  [ -n "${NUM_SHARDS:-}" ]    && export NUM_SHARDS
  bash "${LLM}/test-sweep.sh" || die "stage1 (limit=${limit:-none}) failed"
}

# --------------------------------------------------- analyze (stages 2, 3, 4) --
run_analysis() {
  local limit="$1" T TD SUM PD LIM SUF JOUT fp ROSTER RY ROSTER_SRC AFP=""
  for fp in ${ACCEPT_FP}; do AFP="${AFP} --accept_prompt_fp ${fp}"; done
  for T in ${TASKS}; do
    TD="$(task_dir "${T}")"; SUM="$(summarizer "${T}")"; RY="$(roster_yaml "${T}")"
    # Reader suffix before the limit suffix, matching judge_config.judge_out_filename
    # and llm_parsed_dirname. These are built here rather than queried per task to
    # keep the loop free of interpreter starts, so they are the one place that has
    # to be re-checked if that naming ever changes.
    if [ -n "${limit}" ]; then LIM="--limit ${limit}"; SUF="${JUDGE_SUFFIX}_limit${limit}"; PD="llm-parsed${JUDGE_SUFFIX}-limit${limit}"
    else                       LIM="";                 SUF="${JUDGE_SUFFIX}";               PD="llm-parsed${JUDGE_SUFFIX}"; fi
    # The MOCK infix must match test-sweep.sh, or `MOCK=1 run_llm_parsing.sh`
    # judges into judge-out_<T>.MOCK.jsonl and then dies here looking for a file
    # Stage 1 never wrote -- which would defeat the whole point of the stand-in,
    # namely exercising Stages 2-4 on a CPU box.
    MOCK_INFIX=""
    [ "${MOCK}" = "1" ] && MOCK_INFIX=".MOCK"
    JOUT="${TD}/judge-out_${T}${SUF}${MOCK_INFIX}.jsonl"

    [ -s "${JOUT}" ] || die "analyze ${T}: ${JOUT} is missing or empty.
       Stage 1 has not produced judge output for this task at this limit yet.
       Run the judge first:  bash ${LLM}/$(basename "$0") --from $([ -n "${limit}" ] && echo pilot || echo full)"

    say "stage2 ${T} -- span-verify + merge -> ${PD}/"
    "${PY}" "${LLM}/apply_judge.py" --task_type "${T}" --task_dir "${TD}" \
        --judge_out "${JOUT}" --judge "${JUDGE_KEY}" ${LIM} -p "${PROCS}" ${AFP} \
        ${RY:+--config_yaml "${RY}"} \
      || die "stage2 ${T}"

    # The roster, as --models for the two summarizer runs below. Those summarizers
    # have no roster concept of their own: they walk every directory under
    # --task_dir, and a results tree holds far more than the study reports on --
    # measured 2026-08-15, 57 dirs under MedVision-TL-v2-CoT against 18 in the
    # roster (39 superseded bugfix variants, training checkpoints, baselines like
    # random_detection).
    #
    # stage3 was already landing on the roster, but only by accident: it reads
    # llm-parsed<reader>/, which only Stage 2 creates and Stage 2 IS roster-scoped,
    # so --skip_model_wo_parsed_files removed the rest. stage3b reads parsed/, which
    # every directory has, so nothing filtered it at all. Passing --models to both
    # makes one rule explicit instead of one accident and one gap.
    # A ROSTER_YAML_<task> override wins; otherwise judge_config's default, as the
    # Python stages themselves resolve it.
    if [ -n "${RY}" ]; then ROSTER_SRC="'${RY}'"; else ROSTER_SRC="DEFAULT_ROSTER_YAML['${T}']"; fi
    ROSTER="$("${PY}" -c "import sys;sys.path.insert(0,'${LLM}');from judge_io import load_roster;from judge_config import DEFAULT_ROSTER_YAML;print(' '.join(load_roster(${ROSTER_SRC})))")"       || die "stage3 ${T}: could not read the roster"
    [ -n "${ROSTER}" ] || die "stage3 ${T}: roster resolved to nothing"

    say "stage3 ${T} -- format-robust metrics via the existing summarizer"
    # --resps_key is not optional: llm-parsed records have the strict key removed,
    # and without the flag every record is skipped (loudly, by assert_resps_key).
    "${PY}" -m "medvision_bm.benchmark.${SUM}" --task_dir "${TD}" \
        --parsed_dirname "${PD}" --resps_key LLM_filtered_resps \
        --models ${ROSTER} \
        ${LIM} -p "${PROCS}" --skip_model_wo_parsed_files $(extra_sum "${T}") \
      || die "stage3 ${T}"

    say "stage3b ${T} -- strict baseline on the SAME rows, for the diff"
    "${PY}" -m "medvision_bm.benchmark.${SUM}" --task_dir "${TD}" \
        --models ${ROSTER} \
        ${LIM} -p "${PROCS}" --skip_model_wo_parsed_files $(extra_sum "${T}") \
      || die "stage3b ${T}"

    say "stage4 ${T} -- answer modes + judge validity"
    "${PY}" "${LLM}/summarize_judge_task.py" --task_type "${T}" --task_dir "${TD}" \
        --judge "${JUDGE_KEY}" ${LIM} ${RY:+--config_yaml "${RY}"} || die "stage4 ${T}"

    say "invariants ${T} -- record-level asserts over what was just written"
    # MOCK is passed explicitly: test-8 checks provenance in BOTH directions, and
    # `analyze` can be invoked without going through run_judge (which is the only
    # other place MOCK is exported).
    # TASKS is narrowed to the task being analysed: the gate is per-task but the
    # test scans every task tree, so a subset run would otherwise be judged
    # against trees this invocation never wrote.
    MOCK="${MOCK}" TASKS="${T}" "${PY}" unit-test/llm-parsing/test-8.py "${PD}" \
      || die "record invariants ${T}"
  done
}

step_pilot() {
  say "pilot -- ${PILOT_LIMIT} rows per file, end to end"
  # The pilot needs its OWN queue: --limit reaches the filename
  # (judge-queue_TL_limit100.jsonl) precisely so a pilot can never be mistaken for
  # a full run, which means stage0's unlimited queue does not satisfy it.
  local RY
  for T in ${TASKS}; do
    note "building the limit-${PILOT_LIMIT} queue for ${T} ..."
    RY="$(roster_yaml "${T}")"
    "${PY}" "${LLM}/build_judge_queue.py" --task_type "${T}" \
        --task_dir "$(task_dir "${T}")" --limit "${PILOT_LIMIT}" -p "${PROCS}" \
        ${RY:+--config_yaml "${RY}"} \
      || die "pilot ${T}: stage0 gate failed"
  done
  run_judge "${PILOT_LIMIT}"
  run_analysis "${PILOT_LIMIT}"
  cat <<EOF

  Read the pilot before committing 13 GPU-hours:
    - dSR should be large and POSITIVE for the known offenders
      (Qwen2.5-VL-32B, Llama-3.2-11B) and ~0.0 for MedVision-V0.
    - judge-vs-regex agreement should be >= 99%.
    - judge invalid + span-unverified should both be near 0.
EOF
}

step_full() {
  say "full -- the whole roster (46,379 / 39,140 / 437,349 rows)"
  run_judge ""
}

step_analyze() { say "analyze -- Stages 2-4, full sweep"; run_analysis ""; }

# ---------------------------------------------------------------------- main --
# The steps, in execution order, as DATA: name | what it does | what it costs.
# --list, --help and the runner all read this one table, so a step cannot be
# described one way in the help and behave another way in the run.
STEP_TABLE=(
  "prep|move aside stale judge output, delete stale queues|DESTRUCTIVE, seconds"
  "stage0|build the queues; replay the strict parser as a gate|CPU, minutes"
  "smoke|GATE G8 -- prove the judge returns the required schema|GPU, minutes"
  "pilot|PILOT_LIMIT rows per file, every task, end to end + invariants|GPU, ~30 min"
  "full|Stage 1 (the judge) over the whole roster, every task|GPU, ~13 h"
  "analyze|Stages 2-4: verify, merge, reports, record invariants|CPU, ~1 h"
)
STEPS=(); for _r in "${STEP_TABLE[@]}"; do STEPS+=("${_r%%|*}"); done; unset _r

# Renders STEP_TABLE. Used by --list and by --help, so they cannot disagree.
list_steps() {
  local row n w c rule8 rule62
  # Generated, not typed: a hand-written rule silently stops matching the column
  # width the moment a description grows.
  rule8=$(printf '%*s' 8 '' | tr ' ' '-')
  rule62=$(printf '%*s' 62 '' | tr ' ' '-')
  printf '%-9s %-62s %s\n' "STEP" "WHAT IT DOES" "COST"
  printf '%-9s %-62s %s\n' "${rule8}" "${rule62}" "------------"
  for row in "${STEP_TABLE[@]}"; do
    IFS='|' read -r n w c <<< "${row}"
    printf '%-9s %-62s %s\n' "${n}" "${w}" "(${c})"
  done
}

# Prints the leading comment block, stopping at the first non-comment line, then
# the step table. A fixed line range (the previous '2,40p') leaks code into --help
# the moment the header grows by a line, which moving this file did.
usage() {
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"
  echo "Steps:"
  list_steps | sed 's/^/  /'
}

SELECT=(); FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    -y|--yes) YES=1; shift ;;
    --judge) shift 2 ;;   # already consumed by the pre-scan above
    --list) list_steps; exit 0 ;;
    --judges) "${PY}" "${LLM}/judge_config.py" --list; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) SELECT+=("$1"); shift ;;
  esac
done

if [ -n "${FROM}" ]; then
  seen=0
  for s in "${STEPS[@]}"; do
    [ "${s}" = "${FROM}" ] && seen=1
    [ "${seen}" = "1" ] && SELECT+=("${s}")
  done
  [ ${#SELECT[@]} -gt 0 ] || die "--from ${FROM}: unknown step. One of: ${STEPS[*]}"
fi
[ ${#SELECT[@]} -eq 0 ] && SELECT=("${STEPS[@]}")

echo "=============================================================="
echo " tasks       : ${TASKS}"
echo " judge model : ${JUDGE_MODEL}"
if [ "${PY_EXPLICIT}" = "1" ]; then
  echo " interpreter : ${PY} ($(${PY} -V 2>&1))"
else
  echo " interpreter : ${PY} ($(${PY} -V 2>&1)) -- PYTHON unset, first on PATH"
  echo "               -> $(command -v "${PY}" 2>/dev/null || echo 'not found')"
fi
echo " steps       : ${SELECT[*]}"
[ "${MOCK}" = "1" ]  && echo " MOCK MODE   : regex stand-in -- never report these numbers"
[ "${FRESH}" = "1" ] && echo " FRESH       : prep will DELETE every llm-parsed*/ directory"
echo "=============================================================="

# Stages 2-4 import medvision_ds AND medvision_bm. Resolve/probe both here, before
# anything expensive runs, rather than discovering the gap after the 13-hour judge.
#
# Two probes, because they fail independently: medvision_ds is a checkout that has
# to be FOUND, while medvision_bm is installed but needs `datasets`/`nibabel` that
# Stage 1 never touches. Only the first existed, and it passes in an environment
# where Stage 2 dies on its first import -- which is exactly what happened.
case " ${SELECT[*]} " in
  *" smoke "*|*" pilot "*|*" full "*) check_judge_imports ;;
esac
case " ${SELECT[*]} " in
  *" analyze "*|*" pilot "*)
    resolve_medvision_ds
    check_analysis_imports hard
    ;;
  *" full "*|*" smoke "*)
    # Stage 1 only. It genuinely does not need these, so this must not abort a
    # GPU-only run -- but saying nothing means the sweep finishes and the reports
    # then fail, which is the whole failure mode being closed here.
    resolve_medvision_ds >/dev/null 2>&1 || true
    check_analysis_imports warn
    ;;
esac

for s in "${SELECT[@]}"; do
  case "${s}" in
    prep|stage0|smoke|pilot|full|analyze) "step_${s}" ;;
    *) die "unknown step '${s}'. One of: ${STEPS[*]}" ;;
  esac
done

say "done: ${SELECT[*]}"
