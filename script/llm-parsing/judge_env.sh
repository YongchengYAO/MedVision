#!/usr/bin/env bash
#
# Shared judge-model and secret resolution. SOURCED by run_llm_parsing.sh and
# test-sweep.sh; never executed on its own.
#
# Why this file exists
# --------------------
# Both entry points must agree on JUDGE_MODEL. That string is stamped into every
# judge-out row and `load_done` compares provenance on resume, so when the driver
# auto-detected a local checkpoint while a bare test-sweep.sh defaulted to the hub
# id, the two stamped different strings into the SAME output file and a perfectly
# good resume aborted. That used to be two copies of the same block plus a comment
# asking them to stay in sync. It is now one file, so agreement is structural.
#
# Callers cd to the repo root before sourcing (both do, derived from BASH_SOURCE),
# because everything downstream resolves Results/ and .cache/ relative to it.

# Which reader to run. The table of readers lives in judge_config.JUDGE_MODELS and
# is QUERIED here rather than mirrored, for the same reason this file exists at
# all: a second copy of a table is a second thing to keep aligned, and the failure
# when it drifts is a resume that aborts on provenance. `judge_config --list` shows
# what is registered. Costs one interpreter start with no torch import.
JUDGE="${JUDGE:-}"
_JUDGE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A JUDGE_MODEL_HF the caller set by hand outranks the registry (a local mirror of
# the same weights, typically). Captured before the eval, which would overwrite it.
_judge_hf_override="${JUDGE_MODEL_HF:-}"
if ! _judge_reg="$("${PYTHON:-python3}" "${_JUDGE_ENV_DIR}/judge_config.py" \
                     --shell ${JUDGE:+--judge "${JUDGE}"} 2>&1)"; then
  printf '%s\n' "${_judge_reg}" >&2
  exit 1
fi
# Exit 0 with no output is a real failure mode, not a hypothetical: any wrapper
# standing in for PYTHON that answers every call with success prints nothing here,
# every JUDGE_* stays unset, and `set -u` then reports an unbound variable from a
# line that has nothing to do with the cause. Say what actually went wrong.
if [ -z "${_judge_reg}" ]; then
  printf 'FAILED: the judge registry returned nothing.\n' >&2
  printf '        %s %s --shell produced no output.\n' \
    "${PYTHON:-python3}" "${_JUDGE_ENV_DIR}/judge_config.py" >&2
  printf '        Is PYTHON a real interpreter, and is judge_config.py present?\n' >&2
  exit 1
fi
eval "${_judge_reg}"
unset _judge_reg
if [ -n "${_judge_hf_override}" ]; then JUDGE_MODEL_HF="${_judge_hf_override}"; fi
unset _judge_hf_override

# An explicit JUDGE_MODEL always wins -- it is how a caller points at a local
# mirror of the weights, or at a checkpoint this resolution knows nothing about.
# Otherwise the hub id from the registry is the answer.
#
# This used to choose between an upstream quantized release and a locally
# converted bf16 copy based on the compute capability of GPU 0, which meant one
# campaign could stamp two different JUDGE_MODEL strings into the same output
# file. That was legal only because provenance was normalised inside the
# checkpoint. No registered reader ships quantized weights now, so the switch, the
# converted copy and the normalisation are all gone -- if a per-pod switch is ever
# reintroduced, it needs that normalisation back with it, or a resume aborts the
# moment a run moves between node types.
JUDGE_MODEL="${JUDGE_MODEL:-${JUDGE_MODEL_HF}}"

# GPUs per shard. TP=<n> overrides the registry for one run -- the registry value
# is a capacity floor derived from the parameter count, not a measurement, so it is
# the kind of number a run legitimately needs to adjust.
if [ -n "${TP:-}" ]; then JUDGE_TP="${TP}"; fi
case "${JUDGE_TP}" in
  ''|*[!0-9]*|0) printf 'FAILED: TP must be a positive integer (got %s)\n' "${JUDGE_TP}" >&2; exit 1 ;;
esac

export JUDGE_MODEL JUDGE_MODEL_HF
export JUDGE_KEY JUDGE_SUFFIX JUDGE_ENV_HINT
export JUDGE_TP

# The GPUs one shard owns, as a CUDA_VISIBLE_DEVICES value.
#
# Shards are the DATA-parallel axis (each strides the queue) and JUDGE_TP is the
# TENSOR-parallel axis inside one shard, so shard S owns devices
# [S*TP, S*TP+TP) of the resolved list. At TP=1 this degenerates to one device per
# shard, which is the shape this pipeline ran under before TP existed.
#
# Lives here, in the file both entry points source, because the driver's `smoke`
# and test-sweep.sh launch the same way and a disagreement about this arithmetic
# would put two engines on one GPU. That OOMs, but only after both have loaded.
#
# Args: $1 shard index, $2 GPUs per shard, $3.. the resolved device list.
judge_shard_devices() {
  local shard="$1" tp="$2"; shift 2
  local -a all=("$@")
  local out="" i
  for i in $(seq 0 $((tp - 1))); do
    local d="${all[$((shard * tp + i))]:-}"
    [ -n "${d}" ] || return 1
    out="${out}${out:+,}${d}"
  done
  printf '%s' "${out}"
}

# Pod secrets arrive with a trailing newline often enough to be worth handling
# here: vLLM 0.11 reads the raw env var and httpx rejects "Bearer <token>\n" as an
# illegal header, killing every shard at engine init with a message about the
# model being unavailable. Tokens are stripped of ALL whitespace; the cache paths
# only of newlines, since a directory name may legitimately contain a space.
#
# Written as `if`, not `[ ... ] && export`. Under `set -eu` the && form leaves the
# loop's exit status at 1 whenever the last variable checked happens to be unset,
# and this file is SOURCED -- so that status becomes the sourcing script's, at the
# top of the run, with no error message. The same shape already cost this pipeline
# one silent abort; the `if` form always returns 0.
for _v in HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN; do
  if [ -n "${!_v:-}" ]; then
    export "${_v}=$(printf '%s' "${!_v}" | tr -d '[:space:]')"
  fi
done
for _v in HF_HOME HF_HUB_CACHE; do
  if [ -n "${!_v:-}" ]; then
    export "${_v}=$(printf '%s' "${!_v}" | tr -d '\r\n')"
  fi
done
unset _v
true
