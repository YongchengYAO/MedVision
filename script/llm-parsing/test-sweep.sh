#!/usr/bin/env bash
#
# Stage 1 of the LLM-as-Judge pipeline: run the judge over the pre-built queues.
#
# Stage 0 (build_judge_queue.py) must have run first -- this script only consumes
# its queues. Stages 2-4 run afterwards on CPU via `run_llm_parsing.sh analyze`.
#
# Normally invoked by run_llm_parsing.sh, not by hand. Direct use (S below is
# script/llm-parsing/test-sweep.sh):
#
#   bash $S                       # all tasks, one shard per visible GPU
#   TASKS="TL" bash $S            # one task
#   NUM_SHARDS=4 bash $S          # force 4-way data parallelism
#   CUDA_VISIBLE_DEVICES=2,3 bash $S   # use only those two GPUs
#   LIMIT=100 bash $S             # pilot queues (judge-queue_TL_limit100.jsonl)
#   MOCK=1 bash $S                # CPU regex stand-in; NOT a judge
#   STRUCTURED=auto bash $S       # restore grammar-constrained decoding
#
# GPU scaling: one independent vLLM process per shard, each spanning JUDGE_TP
# GPUs. Data parallelism across shards is the efficient axis -- separate processes
# share nothing and resume independently after a preemption -- while TP exists only
# to fit weights that do not fit one card. NUM_SHARDS=1 skips the sharding
# machinery entirely.
#
# Resumable: output rows are keyed by qid, so re-running skips finished work and
# never re-judges a response it has already seen. Kill it and restart at will.

set -euo pipefail

# Derived from this script's own location, not hardcoded: the queue/output paths
# below are repo-root-relative, and an absolute path pinned this file to one
# checkout (it named a specific worktree, so a clone or a second worktree silently
# swept the WRONG Results/ tree).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L=script/llm-parsing

# Judge checkpoint resolution and HF secret hygiene. The two entry points used to
# disagree -- the driver auto-detected a local bf16 checkpoint while a bare run of
# this script defaulted to the hub id -- so a sweep started with one and resumed
# with the other stamped two different judge_model strings into the SAME output
# file, and resume (which compares provenance) aborted a perfectly good run. That
# was two copies of one block plus a comment asking them to stay in sync; it is
# now one sourced file, so they cannot drift.
. "${L}/judge_env.sh"

TASKS="${TASKS:-TL AD Detection}"
LIMIT="${LIMIT:-}"
MOCK="${MOCK:-0}"
# Grammar-constrained decoding is ON by default. It was 'none' for the v1 sweep,
# which left nothing enforcing the response shape -- the reader then invented its
# own schema on 43,935 of 43,938 TL rows and every one was discarded. The v2 prompt
# shows the object explicitly so 'none' should also work, but the default is the
# belt-and-braces setting; --min_valid_rate catches it either way.
STRUCTURED="${STRUCTURED:-auto}"

# The interpreter that must have torch + vllm. A GPU pod usually has several
# python3 binaries on PATH (system, conda base, the env that actually holds
# vllm), and the first one wins -- which is how a two-A100 pod can report
# "no CUDA device". Override with PYTHON=/path/to/python.
PY="${PYTHON:-python3}"
SUFFIX=""
[ -n "${LIMIT}" ] && SUFFIX="_limit${LIMIT}"

# (HF secret hygiene now lives in judge_env.sh, sourced above: a newline in
# HF_TOKEN makes the Authorization header invalid, and huggingface_hub surfaces
# that as "Can't load the configuration of '<the judge model>'" -- which reads
# like a missing or gated model rather than a malformed credential.)

# TASK_DIR_<task> re-points a task at a different Results tree -- how the OOD
# splits are judged. Must stay in step with run_llm_parsing.sh's copy:
# stage 0 builds the queue where the driver's task_dir() points, and this script
# has to find it there.
task_dir() {
  case "$1" in
    TL)        echo "${TASK_DIR_TL:-Results/MedVision-TL-v2-CoT}" ;;
    AD)        echo "${TASK_DIR_AD:-Results/MedVision-AD-v2-CoT}" ;;
    Detection) echo "${TASK_DIR_Detection:-Results/MedVision-detect-v2}" ;;
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
}

# ---------------------------------------------------------------- preflight --
# Both default to the REAL-run values; the mock branch overrides them. Declared
# here rather than only inside the branch because `set -u` makes an unset
# MOCK_INFIX fatal at the OUT assignment -- on the non-mock path, i.e. production.
MOCK_ARG=""
MOCK_INFIX=""
# Declared empty for the same reason: the mock branch never resolves a device list,
# and `"${DEVICES[@]}"` on a never-assigned array is fatal under `set -u`.
DEVICES=()
if [ "${MOCK}" = "1" ]; then
  MOCK_ARG="--mock"
  # Honour an explicit NUM_SHARDS so the shard/merge path stays testable on a CPU
  # box; default to 1 because the stand-in has nothing to parallelise.
  NUM_SHARDS="${NUM_SHARDS:-1}"
  echo "!! MOCK MODE: deterministic regex stand-in, not a judge. Never report these numbers."
  # Computed HERE, not inline in the OUT assignment below. Written as
  #   OUT="...$([ "${MOCK}" = "1" ] && echo .MOCK).jsonl"
  # the failing test makes the command substitution exit 1, a lone assignment
  # takes its last substitution's status, and `set -e` kills the script -- so
  # every REAL sweep died silently at that line while MOCK=1 sailed through.
  # (`[ ... ] && VAR=...` as a standalone command is safe: bash exempts a failing
  # command that precedes `&&`. Only the substitution-in-assignment form is not.)
  MOCK_INFIX=".MOCK"
else
  # Distinguish the three ways this can fail. Collapsing them into one message
  # ("no CUDA device") sent a previous run chasing a hardware problem on a pod
  # that had two A100s and simply the wrong python3 first on PATH.
  echo "interpreter: ${PY} ($(${PY} -V 2>&1))"
  # The allocation is the load-bearing part. device_count() only ENUMERATES the
  # driver; it does not initialize CUDA, so a torch built against a newer CUDA
  # than the driver supports still reports a plausible count here and fails much
  # later inside the engine ("The NVIDIA driver on your system is too old (found
  # version 12080)") -- after the queue is built, from a raw traceback. Touching
  # a tensor forces _lazy_init(), so that mismatch lands in this probe, where the
  # message below already names the right cause: the interpreter.
  probe="$(${PY} -c 'import torch; n=torch.cuda.device_count(); (torch.zeros(1, device="cuda") if n else None); print(n); print(torch.__version__, torch.version.cuda)' 2>&1)" || probe_failed=1

  if [ "${probe_failed:-0}" = "1" ] || ! printf '%s' "${probe}" | head -1 | grep -qE '^[0-9]+$'; then
    echo "ERROR: could not query CUDA through ${PY}." >&2
    echo "       ${PY} said:" >&2
    printf '%s\n' "${probe}" | sed 's/^/         /' >&2
    echo "       This is an INTERPRETER problem, not necessarily a GPU one." >&2
    echo "       Point PYTHON at the env that has torch+vllm, e.g.:" >&2
    echo "         PYTHON=\$(conda run -n <env> which python) bash test-sweep.sh" >&2
    echo "         PYTHON=/opt/conda/envs/vllm/bin/python bash test-sweep.sh" >&2
    exit 1
  fi

  N_GPU="$(printf '%s' "${probe}" | head -1)"
  if [ "${N_GPU}" -eq 0 ]; then
    echo "ERROR: torch imports but reports 0 CUDA devices." >&2
    echo "       torch: $(printf '%s' "${probe}" | sed -n 2p)" >&2
    echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}" >&2
    if [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ]; then
      echo "       /dev/nvidia* IS present, so the GPUs are attached to the pod." >&2
      echo "       Likely: a CPU-only torch build, or CUDA_VISIBLE_DEVICES set empty." >&2
    else
      echo "       /dev/nvidia* is ABSENT -- the container has no GPU attached." >&2
      echo "       Check the pod spec requests nvidia.com/gpu and uses the NVIDIA runtime." >&2
    fi
    echo "       Override once you are sure: SKIP_GPU_CHECK=1 bash test-sweep.sh" >&2
    [ "${SKIP_GPU_CHECK:-0}" = "1" ] || exit 1
    N_GPU=1
    echo "       SKIP_GPU_CHECK=1 set -- continuing anyway." >&2
  fi

  ${PY} -c "import vllm" 2>/dev/null || {
    echo "ERROR: vllm not importable by ${PY}." >&2
    echo "       The judge needs a newer vLLM than the repo's parse env pins." >&2
    echo "       If vllm lives in another env: PYTHON=/path/to/python bash test-sweep.sh" >&2
    exit 1
  }
  # Resolve the device list ONCE. Honour a pre-set CUDA_VISIBLE_DEVICES so the
  # script cooperates with a scheduler that already restricted the pod: setting
  # CUDA_VISIBLE_DEVICES=$S in a child would REPLACE the parent's filter and
  # re-index against physical GPUs, so `CUDA_VISIBLE_DEVICES=2,3 bash ...` would
  # otherwise silently land on physical GPUs 0 and 1.
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  else
    mapfile -t DEVICES < <(seq 0 $((N_GPU - 1)))
  fi
  # Shards are the data-parallel axis and JUDGE_TP the tensor-parallel axis inside
  # each, so the default is devices/TP -- NOT devices. At TP=1 that is one shard per
  # GPU exactly as before; at TP=2 on four cards it is two shards of two GPUs.
  NUM_SHARDS="${NUM_SHARDS:-$(( ${#DEVICES[@]} / JUDGE_TP ))}"
  [ "${NUM_SHARDS}" -ge 1 ] || NUM_SHARDS=1

  # The engine needs JUDGE_TP GPUs, so the ceiling is on shards x TP. Checked
  # before any weights load: two engines sharing a card co-fit at
  # gpu_memory_utilization=0.90 only long enough to OOM, and six hours in that is
  # expensive.
  if [ $(( NUM_SHARDS * JUDGE_TP )) -gt "${#DEVICES[@]}" ]; then
    echo "ERROR: NUM_SHARDS=${NUM_SHARDS} x TP=${JUDGE_TP} = $(( NUM_SHARDS * JUDGE_TP )) GPUs," >&2
    echo "       but only ${#DEVICES[@]} are visible [${DEVICES[*]}]." >&2
    echo "       Each shard loads its own copy of ${JUDGE_KEY} at" >&2
    echo "       gpu_memory_utilization=0.90; overlapping them will OOM." >&2
    echo "       Lower NUM_SHARDS, lower TP, or expose more GPUs." >&2
    exit 1
  fi
  # Leftover GPUs are silently idle otherwise, and idle GPUs on a 13-hour sweep are
  # the kind of thing worth one line of output.
  if [ $(( ${#DEVICES[@]} - NUM_SHARDS * JUDGE_TP )) -gt 0 ]; then
    echo "NOTE: $(( ${#DEVICES[@]} - NUM_SHARDS * JUDGE_TP )) GPU(s) will sit idle" \
         "(${#DEVICES[@]} visible, ${NUM_SHARDS} shards x TP ${JUDGE_TP})." >&2
    echo "      Raise NUM_SHARDS to use them, if they divide evenly by TP." >&2
  fi
  echo "GPUs visible: ${N_GPU} [${DEVICES[*]}]   shards: ${NUM_SHARDS} x TP ${JUDGE_TP}"
echo "judge: ${JUDGE_KEY}  model: ${JUDGE_MODEL}"
fi

for T in ${TASKS}; do
  TD="$(task_dir "$T")"
  Q="${TD}/judge-queue_${T}${SUFFIX}.jsonl"
  [ -f "${Q}" ] || { echo "ERROR: missing queue ${Q} -- run build_judge_queue.py first." >&2; exit 1; }
done

# --------------------------------------------------------------------- run --
for T in ${TASKS}; do
  # NOTE: TD is recomputed per task and used only inside this iteration. An
  # earlier version let it leak out of the loop, so a following shard block wrote
  # TL shards into the Detection directory.
  TD="$(task_dir "$T")"
  Q="${TD}/judge-queue_${T}${SUFFIX}.jsonl"
  # MOCK output goes to its OWN path. It used to share the production filename,
  # and because run_judge_vllm resumes on (prompt_fp, qid) -- never on
  # judge_model -- a single MOCK=1 run left every qid marked done, so the next
  # REAL sweep made zero model calls, exited 0, and fed the regex stand-in's
  # verdicts to Stages 2-4 with nothing downstream able to tell.
  # JUDGE_SUFFIX (from judge_env.sh) keeps two readers apart. Empty for the
  # default reader, so every file already on disk keeps its name. The QUEUE
  # deliberately gets no suffix: it is built from the prompt, not the reader,
  # and both readers answering the identical queue is what makes the two
  # reports comparable.
  OUT="${TD}/judge-out_${T}${JUDGE_SUFFIX}${SUFFIX}${MOCK_INFIX}.jsonl"

  echo
  echo "=============================================================="
  echo " ${T}: $(wc -l < "${Q}") queued rows -> ${OUT}"
  echo "=============================================================="

  if [ "${NUM_SHARDS}" -le 1 ]; then
    "${PY}" "${L}/run_judge_vllm.py" \
      --queue "${Q}" --out "${OUT}" --model "${JUDGE_MODEL}" \
      --judge "${JUDGE_KEY}" --tensor_parallel_size "${JUDGE_TP}" \
      --structured "${STRUCTURED}" ${MOCK_ARG}
  else
    # One independent LLM() process per GPU. Data parallelism, not tensor
    # parallelism: a 3.6B-active MoE gains almost nothing from TP at this size,
    # and separate processes resume independently after a preemption.
    # A shard file left over from a run with a DIFFERENT NUM_SHARDS covers a
    # different stride, so merging it in would duplicate qids. Refuse rather than
    # silently produce a merged file longer than the queue.
    # The shard COUNT is part of the filename. Without it the guard below could
    # only compare names, so LOWERING NUM_SHARDS was caught while RAISING it was
    # not: old shard0/shard1 remain legal names but cover the coarser i%2 stride,
    # a superset of the new i%4, and the merge emitted those rows twice.
    expected=()
    for S in $(seq 0 $((NUM_SHARDS - 1))); do expected+=("${OUT}.n${NUM_SHARDS}.shard${S}"); done
    stale=""
    for f in "${OUT}".shard* "${OUT}".n*.shard*; do
      [ -e "${f}" ] || continue
      # Retired files are not shards. `prep` renames superseded output to *.v1,
      # and the glob above matches judge-out_TL.jsonl.shard0.v1 -- which is an
      # archive of a previous prompt's run, not a stride of this one. Treating it
      # as stale made the guard refuse a perfectly clean start.
      # Any .v<N> archive, not just .v1: prep now falls back to .v2/.v3 rather
      # than clobbering an existing archive, and an archived SHARD file
      # (judge-out_TL.jsonl.n4.shard0.v2) still matches the shard glob above. With
      # only `*.v1` exempted it was reported as a stale shard and Stage 1 refused
      # to start -- telling the user to delete exactly the evidence prep preserves.
      # .reparsed is exempted for trees written by an earlier version, which named
      # the re-parse marker "<file>.reparsed" -- that matched this glob, so
      # re-parsing a SHARD hard-aborted Stage 1 before any judging, reporting a
      # two-line text marker as a stale shard of a different stride. The marker is
      # a dotfile now and no longer matches at all; this keeps old trees working.
      case "${f}" in *.v[0-9]*|*.stale|*.tmp|*.reparsed) continue ;; esac
      keep=0
      for e in "${expected[@]}"; do [ "${f}" = "${e}" ] && keep=1; done
      [ "${keep}" -eq 0 ] && stale="${stale} ${f}"
    done
    if [ -n "${stale}" ]; then
      echo "ERROR: shard files from a previous run with a different NUM_SHARDS:" >&2
      for f in ${stale}; do echo "         ${f}" >&2; done
      echo "       They cover a different stride and would duplicate rows." >&2
      echo "       If NUM_SHARDS is unchanged and these are legacy names from" >&2
      echo "       before the count was added to the filename, RENAME them --" >&2
      echo "         for f in ${OUT}.shard*; do" >&2
      echo "           mv \"\$f\" \"\${f/.shard/.n${NUM_SHARDS}.shard}\"; done" >&2
      echo "       -- which preserves the partial sweep. Otherwise remove them," >&2
      echo "       or re-run with the NUM_SHARDS they were produced under." >&2
      exit 1
    fi

    pids=()
    for S in $(seq 0 $((NUM_SHARDS - 1))); do
      # A SLICE of the device list, not DEVICES[S]: at TP>1 one shard owns several
      # GPUs. judge_shard_devices does the arithmetic once, in the file both entry
      # points source, so `smoke` and the sweep cannot disagree about ownership and
      # overlap two engines onto one card.
      # The mock stand-in resolves no device list (it touches no GPU) but the
      # sharded path is still exercised on CPU boxes, so it falls back to the shard
      # index -- which is what the previous `${DEVICES[$S]:-$S}` did. Outside mock
      # an empty result means the arithmetic and the guard disagree, which must
      # abort rather than quietly co-locate two engines.
      if ! SHARD_DEV="$(judge_shard_devices "${S}" "${JUDGE_TP}" "${DEVICES[@]}")"; then
        [ "${MOCK}" = "1" ] || {
          echo "ERROR: not enough GPUs for shard ${S} at TP=${JUDGE_TP}" >&2; exit 1; }
        SHARD_DEV="${S}"
      fi
      CUDA_VISIBLE_DEVICES="${SHARD_DEV}" "${PY}" "${L}/run_judge_vllm.py" \
        --queue "${Q}" --out "${OUT}.n${NUM_SHARDS}.shard${S}" \
        --shard "${S}" --num_shards "${NUM_SHARDS}" \
        --model "${JUDGE_MODEL}" --judge "${JUDGE_KEY}" \
        --tensor_parallel_size "${JUDGE_TP}" \
        --structured "${STRUCTURED}" ${MOCK_ARG} &
      pids+=($!)
    done
    fail=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "ERROR: shard ${i} failed" >&2; fail=1; }
    done
    [ "${fail}" -eq 0 ] || { echo "ERROR: ${T} incomplete; shard files kept for resume." >&2; exit 1; }

    # Merge is derived, never authoritative: shard files ARE the resume state, so
    # they are kept. Enumerate explicitly (a glob orders shard10 before shard2)
    # and write to a temp, so an interrupted merge cannot leave a truncated file.
    for f in "${expected[@]}"; do
      [ -f "${f}" ] || { echo "ERROR: missing ${f}" >&2; exit 1; }
    done
    cat "${expected[@]}" > "${OUT}.tmp"
    # The merge overwrites OUT unconditionally, but OUT is also where every
    # out-of-band repair lands: Stage 1 --redo_invalid APPENDS to it and
    # reparse_judge_out.py rewrites it, and neither touches the shard files. So
    # refuse to overwrite when OUT holds a qid the shards do not -- that row
    # would be destroyed with no error anywhere.
    if [ -f "${OUT}" ]; then
      # argv: merged, shard-concatenation, then every shard file (for mtimes).
      "${PY}" - "${OUT}" "${OUT}.tmp" "${expected[@]}" <<'PY' || { rm -f "${OUT}.tmp"; exit 1; }
import collections, json, os, sys
def counts(p):
    c = collections.Counter()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c[json.loads(line).get("qid")] += 1
            except json.JSONDecodeError:
                continue
    return c
merged, shards, shard_files = sys.argv[1], sys.argv[2], sys.argv[3:]
old, new = counts(merged), counts(shards)
lost = [q for q, n in old.items() if new.get(q, 0) < n]

# Out-of-band work is detected by an explicit MARKER, not by comparing verdicts.
# Comparing verdicts cannot tell a reparse from an ordinary re-judge: the judge is
# not reproducible run to run (DESIGN section 10 -- 12.8% agreement on re-judged
# rows), so a shard legitimately re-judging a row routinely yields a different
# verdict. A verdict comparison aborted FINISHED multi-GPU sweeps and blamed a
# reparse that never happened.
#
# The marker is compared against the MERGED FILE, not against the shards. A
# re-parse rewrites the merged file and never touches a shard, so anchoring to the
# shards meant any later shard write -- even an in-place torn-tail truncation that
# adds no rows -- pushed them past the marker and the guard went silent, which is
# precisely when a repair is in flight. Anchoring to the merged file states the
# real question ("did OUT's current bytes come from a re-parse?") and is
# self-clearing: once a legitimate merge rewrites OUT, OUT is newer than the
# marker and the guard stands down on its own.
marker = os.path.join(os.path.dirname(os.path.abspath(merged)),
                      "." + os.path.basename(merged) + ".reparsed")
reparsed = (os.path.exists(marker)
            and os.path.getmtime(marker) >= os.path.getmtime(merged))

if lost or reparsed:
    if lost:
        print(f"ERROR: refusing to merge -- {len(lost):,} qid(s) have more rows in "
              f"{merged} than the shard files provide.", file=sys.stderr)
    if reparsed:
        print(f"ERROR: refusing to merge -- {marker} is at least as new as "
              f"{merged}, so that file's current bytes came from a re-parse and "
              f"the shards never produced them.", file=sys.stderr)
    print("       Back up the merged file before re-running, or re-judge with "
          "NUM_SHARDS=1 so the merged file IS the resume state.", file=sys.stderr)
    print(f"       If the re-parse is superseded, delete {marker} and re-run.",
          file=sys.stderr)
    sys.exit(1)
PY
    fi
    mv "${OUT}.tmp" "${OUT}"
  fi

  # Every queued row must have exactly one judge record.
  n_q=$(wc -l < "${Q}"); n_o=$(wc -l < "${OUT}")
  if [ "${n_q}" -ne "${n_o}" ]; then
    echo "WARNING: ${T} queue has ${n_q} rows but output has ${n_o}." >&2
    echo "         Re-run to fill gaps (it resumes); investigate if it persists." >&2
  fi

  echo "${T}: $(wc -l < "${OUT}") judge rows written"
done

echo
echo "=============================================================="
echo " Stage 1 complete. Next (CPU, no GPU needed):"
if [ -n "${LIMIT}" ]; then
  # `analyze` is hardwired to the FULL sweep and would look for judge-out_<T>.jsonl,
  # not the _limit${LIMIT} file this run produced. `pilot` is the limit-aware path,
  # and re-entering it is cheap: the queue rebuild is idempotent (qids are content
  # hashes) and the judge finds every row already done.
  echo "   TASKS=\"${TASKS}\" PILOT_LIMIT=${LIMIT} bash ${L}/run_llm_parsing.sh pilot"
else
  echo "   TASKS=\"${TASKS}\" bash ${L}/run_llm_parsing.sh analyze"
fi
echo "=============================================================="
