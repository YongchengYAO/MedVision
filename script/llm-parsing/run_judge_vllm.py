"""Stage 1 -- run the judge model over the queue (GPU).

Offline vLLM ``LLM()``, matching the repo's existing convention (see
``lmms_eval/models/vllm_glm4v.py:52`` -- "We run vLLM OFFLINE (LLM.chat), not the
OpenAI server"). No server lifecycle to manage, and continuous batching gives far
better throughput than fanning HTTP requests at one.

Resume: every completed row is keyed by ``cache_key`` (a hash of the response text
AND the prompt fingerprint) and stamped with ``prompt_fp``. Re-running skips
finished rows; a row written under a different prompt is refused rather than
reused, because rows are skipped by ``qid`` and ``qid`` has no prompt component.

Two abort gates run before any long sweep can waste time: the queue's prompt stamp
must match the prompt this process renders, and the judge must return the required
schema on its first rows. The second gate exists because its absence cost thirteen
GPU-hours -- see ``gate_valid_rate``.

A ``--mock`` mode runs the same pipeline with a deterministic regex extractor
instead of the model. It exists so Stages 2-4 can be exercised end to end on a
CPU box. It is NOT a judge and must never be used for reported numbers -- every
row it writes is stamped ``judge_model: "mock"``.

Usage:
    python run_judge_vllm.py --task_type TL \\
        --queue Results/MedVision-TL-v2-CoT/judge-queue_TL_limit100.jsonl \\
        --out   Results/MedVision-TL-v2-CoT/judge-out_TL_limit100.jsonl
"""

import argparse
import collections
import functools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_config import (  # noqa: E402
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MODEL_DEFAULT,
    JUDGE_MODELS,
    JUDGE_SEED,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    CHUNK_ROWS_DEFAULT,
    DEFAULT_JUDGE_MAX_TOKENS,
    MIN_VALID_RATE_DEFAULT,
    TASK_SPECS,
    VALID_RATE_PROBE_ROWS,
    judge_entry,
    resolve_judge_key,
)
from judge_decode import parse_judge_json, validate_judge_obj  # noqa: E402
from judge_io import find_numbers  # noqa: E402
from judge_prompts import (  # noqa: E402
    build_messages,
    build_schema,
    short_prompt_fp,
)


# --------------------------------------------------------------------------
# Mock judge (CPU only; test fixture)
# --------------------------------------------------------------------------

# Wrappers observed in the real corpus, most specific first. Used ONLY by --mock.
_MOCK_PATTERNS = [
    r"<answer>(.*?)</answer>",
    r"<final-answer>(.*?)</final-answer>",
    r"\\boxed\{([^{}]*)\}",
    r"<step-\d+-answer>(.*?)</step-\d+-answer>",
    r"\*\*Answer:?\*\*\s*([^\n]*)",
    r"(?:Final Answer|Answer)\s*:?\s*([^\n]*)",
]


def _mock_judge(response, task_type):
    """Deterministic stand-in for the judge, for CPU testing of Stages 2-4.

    Returns only ``present`` or ``no_conclusion`` -- the same two-way contract the
    real judge answers under. Every row it writes is stamped ``judge_model: "mock"``
    so its output can never be mistaken for a real sweep.
    """
    arity = TASK_SPECS[task_type]["arity"]
    for pat in _MOCK_PATTERNS:
        for m in reversed(list(re.finditer(pat, response or "", re.DOTALL))):
            inner = m.group(1)
            nums = find_numbers(inner)
            if len(nums) >= arity:
                span = m.group(0)[:200]
                return {
                    "final_answer": {
                        "status": "present",
                        "span": span,
                        "values": [float(x) for x in find_numbers(span)[-arity:]],
                    },
                    "steps": [],
                }
    return {
        "final_answer": {"status": "no_conclusion", "span": "", "values": []},
        "steps": [],
    }


# --------------------------------------------------------------------------
# Queue / cache
# --------------------------------------------------------------------------


def load_queue(path):
    """Read the Stage 0 queue."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _expected_fps(rows):
    """Return ``{step_key: prompt_fp}`` for this run, verifying the queue's stamps.

    Recomputed from the current prompt rather than trusted from the queue, then
    compared against what the queue carries. A queue built before a prompt edit has
    ``cache_key`` values derived from the old prompt; judging it now would attribute
    old-prompt cache keys to new-prompt answers and make the next resume incoherent.

    Args:
        rows: Queue rows for this run.

    Returns:
        dict: ``{step_key: prompt_fp}``.

    Raises:
        SystemExit: If any queue row's stamp disagrees with the current prompt.
    """
    expected, mismatched = {}, {}
    for r in rows:
        step_key = r.get("step_key")
        if step_key not in expected:
            expected[step_key] = short_prompt_fp(r["task_type"], step_key)
        if r.get("prompt_fp") != expected[step_key] and step_key not in mismatched:
            mismatched[step_key] = (r.get("prompt_fp"), expected[step_key])
    if mismatched:
        detail = "\n".join(
            f"    step_key={k!r}: queue has {got!r}, current prompt is {want!r}"
            for k, (got, want) in mismatched.items()
        )
        raise SystemExit(
            f"\n[stage1] ABORT the queue was built for a different prompt.\n"
            f"{detail}\n\n"
            f"  Rebuild it before judging:\n"
            f"      python build_judge_queue.py --task_type {rows[0]['task_type']} ...\n"
        )
    return expected


@functools.lru_cache(maxsize=None)
def _model_key(model):
    """Normalise a judge-model identifier for provenance comparison.

    A local checkpoint can be named many ways that mean the same weights -- with
    or without a trailing slash, relative, or through a symlink -- and comparing
    the raw strings turned any of those into "a different judge" and aborted the
    resume. Filesystem paths collapse to their realpath; a hub id (or the "mock"
    sentinel) is not a path and passes through unchanged.

    Normalisation stops at the filesystem. It once also collapsed a locally
    CONVERTED checkpoint onto the model it was converted from, read out of a
    marker file inside the directory, so that a campaign could span pods that
    wanted different materializations of one release. No registered reader ships
    weights that need converting, so there is nothing left to collapse -- and the
    narrow version was the safe one: two genuinely different checkpoints must
    still compare unequal, since the failure mode of an over-eager normaliser is
    silently accepting another model's answers on resume.

    Cached: this runs once per row over ~500K rows, and ``realpath`` is a syscall.

    Args:
        model: The ``judge_model`` string from a row, or this run's ``--model``.

    Returns:
        str: A comparison key. ``""`` for None, so a row written before
        ``judge_model`` was persisted compares unequal to any real model and is
        correctly treated as stale.
    """
    if not model:
        return ""
    m = str(model).rstrip("/")
    if not os.path.exists(m):
        return m
    return os.path.realpath(m)


def _repair_torn_tail(out_path):
    """Drop an unterminated final line before appending to ``out_path``.

    Rows are appended to a buffered handle and fsynced only every --chunk_rows, so
    a kill leaves a partial JSON line with no trailing newline. Appending onto it
    welds two half-records into one permanently unparseable line: ``load_done``
    skips it silently, and the only completeness check downstream is ``wc -l`` of
    the merged file, which a welded line satisfies -- so exactly one judge record
    goes missing with nothing reporting it.

    Truncating at the last newline costs at most the one row that was in flight,
    which the resume then re-judges.

    Args:
        out_path: Judge-output JSONL to repair in place. Missing/empty is a no-op.

    Returns:
        int: Bytes discarded.
    """
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return 0
    with open(out_path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return 0
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Scan back for the last newline; the tail after it is the torn fragment.
        window = min(size, 1 << 20)
        f.seek(size - window)
        tail = f.read(window)
        nl = tail.rfind(b"\n")
        keep = (size - window) + nl + 1 if nl >= 0 else 0
        f.truncate(keep)
        lost = size - keep
    print(f"[stage1] repaired a torn tail in {out_path}: discarded {lost:,} "
          f"byte(s) of an unterminated final row (it will be re-judged)")
    return lost


def load_done(out_path, expected_fps, accept_fps=(), redo_invalid=False, expect_model=None):
    """Index prior output by qid (what is written) and by cache_key (what is known).

    These are deliberately two different things, and conflating them is a bug.
    ``cache_key`` hashes the RESPONSE TEXT plus the prompt fingerprint, so two
    queue rows with identical responses share one key -- measured at ~9.5% of TL
    rows. The output, however, is one row per ``qid``.

    Resuming on cache_key alone would therefore skip a row whose twin happened to
    be written before the interruption, silently leaving that qid with no judge
    record. Resuming on qid alone would be correct but would re-run the model on
    responses already judged.

    So: skip by ``qid``, and satisfy a repeat response from ``by_key`` without
    calling the model. That is sound because the judge is a pure function of
    (response text, prompt) -- it never sees the sample, the image, or the target.

    THE PROMPT STAMP IS WHAT MAKES THAT SOUND
    -----------------------------------------
    "Pure function of (response text, prompt)" only helps if the prompt is the same
    one. ``qid = hash(model, file, doc_id)`` has no prompt component, and ``main``
    filters on qid BEFORE consulting ``by_key`` -- so editing the prompt, rebuilding
    the queue (which correctly changes every ``cache_key``), and rerunning against
    an existing output file would report every row already done, make zero model
    calls, and ship the previous prompt's answers under the new prompt's version
    string. The ``cache_key`` invalidation is real but unreachable.

    Rows are therefore counted as done only when their ``prompt_fp`` matches the
    fingerprint this run will actually use. Mismatches are counted and returned so
    the caller can refuse to continue.

    Args:
        out_path: Existing judge-output JSONL, if any.
        expected_fps: ``{step_key: prompt_fp}`` for this run.

    Returns:
        tuple: ``(done_qids, by_key, n_stale, seen)`` where ``seen`` is
        ``{"fps": set, "models": set}`` -- what the FILE actually holds, so an
        abort can name the mismatch instead of only what was expected.
    """
    done_qids, by_key, n_stale = set(), {}, 0
    seen = {"fps": set(), "models": set()}
    if not os.path.exists(out_path):
        return done_qids, by_key, n_stale, seen
    # A REPAIR PASS legitimately mixes two prompts in one file. Raising max_tokens
    # changes the fingerprint, but a row judged under the smaller budget that
    # PARSED is still a valid answer to the same question -- the larger budget only
    # removes a truncation that never applied to it. --accept_prompt_fp whitelists
    # those, and --redo_invalid then re-judges exactly the rows the old budget
    # broke. Without both, repairing 3,000 rows would mean re-judging 81,000.
    current = set(expected_fps.values()) | set(accept_fps)
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen["fps"].add(row.get("prompt_fp"))
            seen["models"].add(row.get("judge_model"))
            if row.get("prompt_fp") not in current:
                n_stale += 1
                continue
            # Provenance must match too, or one judge's answers satisfy another's
            # resume. prompt_fingerprint hashes the rendered prompt but NOT the
            # model that answers it, and resume skips by (prompt_fp, qid) -- so
            # rerunning with a different judge found every qid done, made zero
            # model calls and exited 0. That was true for the --mock stand-in
            # (which stamps judge_model "mock") and equally true for real-vs-real,
            # i.e. swapping one reader's checkpoint for another's.
            #
            # Compared on a NORMALISED key, not the raw string. An exact
            # comparison made a trailing slash, a relative path or a symlink read
            # as a different judge and hard-aborted a legitimate resume, and the
            # two supported entry points can genuinely name the same weights
            # differently (a caller-set JUDGE_MODEL pointing at a local mirror, a
            # bare test-sweep.sh run defaulting to the hub id). Normalising the
            # filesystem forms fixes the accidental mismatches; a genuinely
            # different checkpoint still aborts, which is the point.
            if _model_key(row.get("judge_model")) != _model_key(expect_model):
                n_stale += 1
                continue
            if redo_invalid and row.get("judge_status") != "ok":
                continue  # leave it outstanding so this pass re-judges it
            done_qids.add(row.get("qid"))
            key = row.get("cache_key")
            # LAST wins, matching apply_judge.load_judge_index. First-wins made a
            # resume fan a SUPERSEDED verdict onto new qids: after --redo_invalid
            # appends a better answer for a cache_key, the original invalid row
            # sits earlier in the file and would be the one reused.
            if key is not None:
                by_key[key] = row
    return done_qids, by_key, n_stale, seen


def _raw_to_keep(raw, status, args):
    """Decide what raw judge text to persist for one row.

    Persists FULL raw on judge-invalid rows by default. Those are the only rows a
    future decoder change can act on, and without the text a fix costs a GPU pass
    instead of a CPU re-parse -- which is exactly what happened here: the original
    production sweeps ran without ``--keep_raw``, so 23,958 of the 24,582 invalid
    rows now on disk carry no text and can never be re-examined offline. Only the
    later repair passes (624 rows) kept theirs.

    Uncapped on purpose. The previous 2,000-character cap made a truncated
    response indistinguishable from one that merely ran past the cap, so a
    question answerable by reading an integer needed a GPU run instead.

    Args:
        raw: The judge's full generated text.
        status: ``"ok"`` or ``"invalid"``.
        args: Parsed CLI args.

    Returns:
        str | None: Text to store, or ``None`` to store none.
    """
    if not (args.keep_raw or (status == "invalid" and not args.no_raw_on_invalid)):
        return None
    return raw if args.raw_max_chars <= 0 else raw[: args.raw_max_chars]


def _out_row(q, obj, status, reason, judge_model, raw=None, raw_len=None):
    """Assemble one judge-output row, carrying the keys Stage 2 indexes on."""
    row = {
        "qid": q["qid"],
        "cache_key": q["cache_key"],
        # Identifies WHICH prompt produced this answer. Without it a resume cannot
        # distinguish "already judged" from "judged under a different prompt".
        "prompt_fp": q.get("prompt_fp"),
        "model": q["model"],
        "file": q["file"],
        "doc_id": q["doc_id"],
        "task_type": q["task_type"],
        "judge_status": status,
        "judge_reason": reason,
        "judge_model": judge_model,
    }
    if obj:
        row["final_answer"] = obj.get("final_answer")
        if obj.get("steps") is not None:
            row["steps"] = obj.get("steps")
    # Recorded even when the text is not, so "did this response hit the token
    # budget?" is answerable from the output alone.
    if raw_len is not None:
        row["raw_len"] = raw_len
    if raw is not None:
        row["raw"] = raw
    return row


# --------------------------------------------------------------------------
# GPU path
# --------------------------------------------------------------------------


def _check_hardware():
    """Report the GPU, and fail loudly if CUDA is visible but unusable.

    Warns rather than aborts on a missing device: the ``--mock`` stand-in runs the
    same code path on a CPU box.
    """
    try:
        import torch
    except Exception:
        print("[stage1] torch unavailable; skipping hardware check")
        return
    if not torch.cuda.is_available():
        print("[stage1] WARNING no CUDA device visible")
        return
    # First real CUDA initialization in the process. is_available() and
    # device_count() only enumerate the driver, so an interpreter whose torch was
    # built against a newer CUDA than the driver supports gets this far and then
    # raises from _lazy_init() -- as a bare traceback, after the queue is built.
    # Name the actual cause instead: on a multi-env GPU pod the first python3 on
    # PATH is usually the wrong one, and that is a one-variable fix.
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except RuntimeError as e:
        raise SystemExit(
            f"\n[stage1] ABORT CUDA is visible but will not initialize under this "
            f"interpreter.\n"
            f"  interpreter : {sys.executable}\n"
            f"  torch       : {torch.__version__} (built for CUDA "
            f"{torch.version.cuda})\n"
            f"  torch said  : {e}\n\n"
            f"  A 'driver is too old' message here means the torch BUILD wants a "
            f"newer CUDA\n"
            f"  than the driver provides -- the driver is fine, the interpreter is "
            f"wrong. A GPU\n"
            f"  pod usually has several python3, and the first on PATH wins.\n"
            f"  Point the pipeline at the env holding vllm + a matching torch:\n"
            f"      PYTHON=/path/to/env/bin/python bash "
            f"script/llm-parsing/run_llm_parsing.sh\n"
        ) from None
    print(f"[stage1] GPU: {name} (sm_{major}{minor})")


def _build_sampling_params(task_type, step_key, SamplingParams, structured="auto"):
    """Build SamplingParams, attempting structured output across vLLM API renames.

    vLLM 0.11 renamed ``guided_decoding=GuidedDecodingParams(json=...)`` to
    ``structured_outputs=StructuredOutputsParams(json=...)``. We try the new name,
    then the old, then give up -- the tolerant parser in ``judge_decode`` makes
    structured output an optimization, not a requirement.

    ``structured="none"`` skips both attempts: xgrammar's per-token bitmask for
    the schema's free-form ``span`` string is CPU-bound and serializes every
    decode step, so a throughput-minded launcher can opt out entirely.
    """
    base = dict(
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        seed=JUDGE_SEED,
        max_tokens=TASK_SPECS[task_type]["max_tokens"],
    )
    if structured != "none":
        schema = build_schema(task_type, step_key)

        try:
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(**base,
                                  structured_outputs=StructuredOutputsParams(json=schema)), "structured_outputs"
        except Exception:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams

            return SamplingParams(**base,
                                  guided_decoding=GuidedDecodingParams(json=schema)), "guided_decoding"
        except Exception:
            pass
        print("[stage1] NOTE structured output unavailable; relying on prompt + tolerant parser")
    return SamplingParams(**base), "none"


def run_gpu(rows, args):
    """Run the real judge over ``rows``, yielding output rows."""
    from vllm import LLM, SamplingParams

    entry = judge_entry(args.judge)
    _check_hardware()
    # GPUs this ONE engine spans. The sweep's parallelism is primarily across
    # processes (one shard per engine, striding the queue), so this is the inner
    # axis and should be the smallest value that fits the weights with room left
    # for a usable KV cache -- see the topology note in judge_config. It used to be
    # hardcoded to 1, which is correct for a 13 GB MoE and unusable for a dense
    # 31B: 62 GB of weights on an 80 GB card leave almost nothing to batch with.
    tp = args.tensor_parallel_size or entry["tensor_parallel"]
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    # Left to vLLM unless the caller overrides it. Worth setting when a checkpoint
    # lands on a kernel whose dtype choice costs throughput -- vLLM says so itself
    # in that case ("You are running Marlin kernel with bf16 on GPUs before SM90.
    # You can consider change to fp16 to achieve better performance.").
    if args.dtype:
        llm_kwargs["dtype"] = args.dtype
    llm = LLM(**llm_kwargs)

    # Per-reader chat-template arguments. Gemma's template declares no
    # reasoning_effort, and transformers 5.x (which Gemma-4 requires) is less
    # forgiving about unknown template kwargs than the 4.x line this pipeline was
    # written against. Passed as **kwargs only when
    # non-empty, so a reader that declares nothing gets a call identical to the one
    # it would get with no switch at all.
    #
    # NOTE the deliberate asymmetry with the prompt fingerprint: reasoning_effort
    # stays inside prompt_fingerprint for every reader, because it describes the
    # prompt this code authors, not who is asked. Making the fingerprint
    # reader-dependent would invalidate every queue on disk. See judge_config.
    chat_extra = dict(entry["chat_kwargs"])
    print(f"[stage1] judge={args.judge} ({args.model}); tensor_parallel_size={tp}; "
          f"chat_template_kwargs={chat_extra or '{}'}")

    # Group by step_key: sampling params (schema, max_tokens) differ per group.
    by_key = {}
    for r in rows:
        by_key.setdefault(r.get("step_key"), []).append(r)

    for step_key, group in by_key.items():
        task_type = group[0]["task_type"]
        sp, mode = _build_sampling_params(task_type, step_key, SamplingParams,
                                          structured=args.structured)
        print(f"[stage1] step_key={step_key!r}: {len(group):,} rows, structured={mode}")

        expects_steps = step_key is not None

        # Submitted in chunks, NOT as one call per group. `llm.chat` returns only
        # when every prompt it was given is finished, so handing it the whole
        # group makes the generator silent until the last row completes -- and a
        # generator that yields nothing writes nothing. A sweep interrupted after
        # judging 2,000 of 19,000 TL rows persisted exactly zero of them, and the
        # qid-resume that makes this pipeline restartable had nothing to resume
        # from. Chunking bounds that loss to one chunk.
        #
        # Continuous batching is unaffected: vLLM schedules across whatever is in
        # flight, and a chunk of this size keeps the engine saturated.
        # Budget-caused truncation never fails loudly on its own -- it surfaces
        # weeks later as an inflated judge-invalid rate (TL at 1024, Detection at
        # 256, one GPU repair pass each). finish_reason is the direct signal, so
        # count it and SAY it.
        # The FIRST chunk is deliberately small. gate_valid_rate decides on the
        # first VALID_RATE_PROBE_ROWS rows, but it can only see rows this
        # generator has yielded, and llm.chat returns nothing until the whole
        # chunk has decoded -- so with a 2,000-row first chunk the "instant"
        # shape check actually cost 2,000 completions. Probing first makes a
        # failed gate cost what its docstring claims.
        bounds = []
        probe = min(VALID_RATE_PROBE_ROWS, args.chunk_rows, len(group))
        if probe and probe < len(group):
            bounds.append((0, probe))
            bounds += [(s, min(s + args.chunk_rows, len(group)))
                       for s in range(probe, len(group), args.chunk_rows)]
        else:
            bounds = [(s, min(s + args.chunk_rows, len(group)))
                      for s in range(0, len(group), args.chunk_rows)]

        n_cut = 0
        for start, stop in bounds:
            chunk = group[start:stop]
            batch = [
                build_messages(task_type, step_key, r["response"])[0] for r in chunk
            ]
            chat_kw = {"chat_template_kwargs": chat_extra} if chat_extra else {}
            outputs = llm.chat(batch, sampling_params=sp, **chat_kw)
            n_cut += sum(
                1 for out in outputs
                if out.outputs and out.outputs[0].finish_reason == "length"
            )
            yield from _rows_from_outputs(chunk, outputs, expects_steps, args)
        if n_cut:
            print(f"[stage1] WARNING step_key={step_key!r}: {n_cut:,}/{len(group):,} "
                  f"completions hit the max_tokens={sp.max_tokens} budget. "
                  f"A few is normal (degenerate repetition loops, ~0.3%); more "
                  f"than that means the budget is truncating real answers and "
                  f"the judge-invalid rate for this run cannot be trusted.")


def _rows_from_outputs(chunk, outputs, expects_steps, args):
    """Turn one chunk's vLLM outputs into judge-output rows."""
    for r, out in zip(chunk, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            obj, reason = parse_judge_json(raw)
            if obj is None:
                yield _out_row(r, None, "invalid", reason, args.model,
                               raw=_raw_to_keep(raw, "invalid", args), raw_len=len(raw))
                continue
            ok, vreason = validate_judge_obj(obj, expects_steps)
            if not ok:
                yield _out_row(r, obj, "invalid", vreason, args.model,
                               raw=_raw_to_keep(raw, "invalid", args), raw_len=len(raw))
                continue
            yield _out_row(r, obj, "ok", "ok", args.model,
                           raw=_raw_to_keep(raw, "ok", args), raw_len=len(raw))


def run_mock(rows, args):
    """Run the deterministic stand-in over ``rows``, yielding output rows."""
    print("[stage1] MOCK MODE -- regex stand-in, not a judge. "
          "Do not use these rows for reported numbers.")
    for r in rows:
        obj = _mock_judge(r["response"], r["task_type"])
        yield _out_row(r, obj, "ok", "mock", "mock")


def gate_valid_rate(row_iter, min_rate, probe_rows):
    """Withhold output until the judge has proved it returns the right schema.

    The first ``probe_rows`` rows are BUFFERED, not written. If the fraction with
    ``judge_status == "ok"`` is below ``min_rate``, this raises ``SystemExit`` and
    nothing has been written -- so a failed run leaves no partial output file for a
    later resume to mistake for progress.

    This gate exists because its absence cost a full sweep. The v1 run produced 3
    valid rows out of 43,938 on TL and 0 out of 37,080 on AD, and ran to completion
    over thirteen GPU-hours because nothing looked at what was coming back. The
    judge's extraction was correct throughout; it simply wrapped it in a schema of
    its own invention, which is exactly the failure a shape check catches instantly
    and a metric check never would.

    Args:
        row_iter: Iterator of judge-output rows.
        min_rate: Minimum fraction that must validate.
        probe_rows: How many rows to buffer before deciding.

    Yields:
        dict: The same rows, once the gate has passed.

    Raises:
        SystemExit: If the valid rate over the probe is below ``min_rate``.
    """
    buffered, n_ok = [], 0
    for row in row_iter:
        buffered.append(row)
        n_ok += 1 if row.get("judge_status") == "ok" else 0
        if len(buffered) < probe_rows:
            continue

        rate = n_ok / len(buffered)
        if rate < min_rate:
            reasons = collections.Counter(
                r.get("judge_reason") for r in buffered
                if r.get("judge_status") != "ok"
            )
            sample = next(
                (r for r in buffered if r.get("judge_status") != "ok"), {}
            )
            raise SystemExit(
                f"\n[stage1] ABORT judge schema gate failed after "
                f"{len(buffered):,} rows.\n"
                f"  valid: {n_ok}/{len(buffered)} = {rate:.1%}  "
                f"(required >= {min_rate:.0%})\n"
                f"  top failure reasons: {dict(reasons.most_common(5))}\n"
                f"  example bad final_answer: "
                f"{json.dumps(sample.get('final_answer'))[:200]}\n\n"
                f"  Nothing was written. The judge is not returning the required\n"
                f"  object. Check the OUTPUT section of the system prompt and try\n"
                f"  --structured auto before spending GPU hours on this.\n"
                f"  Re-run with --min_valid_rate 0 to override deliberately."
            )
        print(f"[stage1] [gate ok] judge schema valid on {n_ok}/{len(buffered)} "
              f"= {rate:.1%} of the first rows")
        break

    yield from buffered
    yield from row_iter


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # No --task_type: it comes from the queue rows, which is the only source that
    # cannot disagree with the data being judged.
    p.add_argument("--queue", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=JUDGE_MODEL_DEFAULT)
    p.add_argument("--judge", default=None, choices=sorted(JUDGE_MODELS),
                   help="Which reader --model is, deciding the chat-template kwargs "
                        "and the weight-format expectations. Inferred from --model "
                        "when that is a known hub id or a checkpoint converted from "
                        "one; required when it is neither, because guessing wrong is "
                        "silent. Does NOT choose the output path -- pass --out.")
    p.add_argument("--max_model_len", type=int, default=JUDGE_MAX_MODEL_LEN)
    p.add_argument("--tensor_parallel_size", type=int, default=None,
                   help="GPUs this ONE engine spans; the sweep runs one engine per "
                        "shard, so total GPUs = shards x this. Default: the reader's "
                        "registry value. Raise it to FIT the weights, not to go "
                        "faster -- across-shard data parallelism scales better than "
                        "TP, which adds an all-reduce per layer. A dense model whose "
                        "weights crowd out the KV cache is the case that needs it.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--dtype", default=None,
                   help="vLLM dtype override, e.g. float16. Left to vLLM by "
                        "default; set it when the engine reports that another "
                        "dtype would pick a faster kernel on this GPU.")
    p.add_argument("--shard", type=int, default=0, help="This shard's index (data parallelism)")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit_rows", type=int, default=None,
                   help="Cap queue rows to an evenly spaced sample spanning the "
                        "whole queue (not the head, which is one model)")
    p.add_argument("--keep_raw", action="store_true",
                   help="Persist raw text on EVERY row, not just invalid ones "
                        "(audit a run). Invalid rows keep raw by default.")
    p.add_argument("--no_raw_on_invalid", action="store_true",
                   help="Do not persist raw text on judge-invalid rows. Saves "
                        "~45 MB across the sweep and makes the next decoder fix "
                        "cost a GPU pass instead of a CPU re-parse.")
    p.add_argument("--raw_max_chars", type=int, default=0,
                   help="Truncate persisted raw text to N characters (0 = full). "
                        "A cap hides whether a response was truncated.")
    p.add_argument("--structured", choices=["auto", "none"], default="auto",
                   help="'none' skips grammar-constrained decoding (CPU-bound "
                        "on this schema); the tolerant parser absorbs the risk")
    p.add_argument("--accept_prompt_fp", action="append", default=[],
                   metavar="FP",
                   help="Treat rows carrying this prompt stamp as done even though "
                        "the current prompt differs. For repair passes that raise "
                        "--max_tokens: valid rows are kept, so only the broken ones "
                        "are re-judged. Repeatable.")
    p.add_argument("--redo_invalid", action="store_true",
                   help="Re-judge rows whose judge_status is not 'ok', even if they "
                        "would otherwise count as done. Pair with --accept_prompt_fp.")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Override the per-task decode budget (TASK_SPECS max_tokens). "
                        "Too small truncates the judge's own JSON mid-object, which "
                        "shows up as no_json_object/json_decode_error rather than as "
                        "an obvious error. Changing it changes the prompt fingerprint, "
                        "so a queue built at another budget is correctly refused.")
    p.add_argument("--chunk_rows", type=int, default=CHUNK_ROWS_DEFAULT,
                   help="Rows per llm.chat() call. Bounds how much judged work an "
                        "interruption can lose, since a chat() call yields nothing "
                        "until all of its prompts finish.")
    p.add_argument("--min_valid_rate", type=float, default=MIN_VALID_RATE_DEFAULT,
                   help="Abort if fewer than this fraction of the first "
                        f"{VALID_RATE_PROBE_ROWS} rows return the required schema. "
                        "0 disables the gate.")
    p.add_argument("--mock", action="store_true",
                   help="CPU stand-in for pipeline testing; NOT a judge")
    args = p.parse_args()

    # Settle which reader this is before anything reads the queue. Under --mock the
    # model is the "mock" sentinel, which belongs to no reader; resolve_judge_key
    # returns the default key for it so the mock exercises the ordinary path, and
    # the mock never reaches run_gpu anyway.
    args.judge = resolve_judge_key(args.model, args.judge)

    if args.max_tokens:
        # Mutated before the queue is read: the fingerprint, the sampling params
        # and the stale-queue check all consult TASK_SPECS, and they must agree.
        for spec in TASK_SPECS.values():
            spec["max_tokens"] = args.max_tokens
        print(f"[stage1] max_tokens overridden to {args.max_tokens}")
        # max_tokens is INSIDE prompt_fingerprint, and build_judge_queue.py has no
        # matching flag -- so a queue stamped under this budget can only exist if
        # DEFAULT_JUDGE_MAX_TOKENS was this value when the queue was built. Say so
        # here, where it is actionable, instead of letting _expected_fps abort with
        # a fingerprint mismatch that reads like a corrupted queue.
        if args.max_tokens != DEFAULT_JUDGE_MAX_TOKENS:
            print(f"[stage1] NOTE --max_tokens {args.max_tokens} differs from "
                  f"DEFAULT_JUDGE_MAX_TOKENS={DEFAULT_JUDGE_MAX_TOKENS}. The budget is "
                  f"part of the prompt fingerprint, so this run will ABORT unless the "
                  f"queue was built while judge_config.py carried the same value. To "
                  f"change the budget for a new sweep, edit DEFAULT_JUDGE_MAX_TOKENS "
                  f"and rebuild the queues (Stage 0); this flag is for re-judging rows "
                  f"whose queue already carries the matching stamp.")

    rows = load_queue(args.queue)
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
        print(f"[stage1] shard {args.shard}/{args.num_shards}: {len(rows):,} rows")
    if args.limit_rows and len(rows) > args.limit_rows:
        # An EVENLY SPACED sample, not the head. build_judge_queue writes the
        # queue in roster order then filename order, so rows[:N] falls entirely
        # inside roster model 0 -- which on every task is MedVision-V0 at ~0.00%
        # regex-fail, i.e. the single model whose responses are already
        # well-formed. A gate sampled there is reading the easiest rows in the
        # corpus. Striding spans every model, dataset and response length, and
        # stays deterministic (no RNG, so the smoke sample is reproducible).
        stride = len(rows) / args.limit_rows
        rows = [rows[int(i * stride)] for i in range(args.limit_rows)]
        print(f"[stage1] --limit_rows {args.limit_rows}: evenly spaced sample "
              f"across the queue (every ~{stride:.0f}th row)")
    if not rows:
        print("[stage1] queue is empty; nothing to do")
        return

    # The queue was stamped by Stage 0 with the prompt it was built for. Verify
    # that stamp against the prompt THIS process would render: a queue built before
    # a prompt edit carries cache_keys that no longer describe its own prompt, and
    # reusing it would mix two prompts inside one output file.
    expected_fps = _expected_fps(rows)

    _repair_torn_tail(args.out)
    done_qids, by_key, n_stale, _seen = load_done(
        args.out, expected_fps,
        accept_fps=args.accept_prompt_fp, redo_invalid=args.redo_invalid,
        expect_model=("mock" if args.mock else args.model))
    if n_stale:
        raise SystemExit(
            f"\n[stage1] ABORT {args.out} holds {n_stale:,} row(s) written under a "
            f"different prompt, or by a different judge model (including the\n"
            f"  --mock stand-in).\n"
            f"  this run expects : prompt_fp {sorted(set(expected_fps.values()))}, "
            f"model {'mock' if args.mock else args.model}\n"
            f"  the file holds   : prompt_fp "
            f"{sorted(str(x) for x in _seen['fps'])[:4]}, "
            f"model {sorted(str(x) for x in _seen['models'])[:4]}\n"
            f"  Resuming would keep those answers: rows are skipped by qid, and qid\n"
            f"  does not depend on the prompt, so the stale rows would count as done\n"
            f"  and the model would never be called for them.\n\n"
            f"  Move the file aside and start fresh:\n"
            f"      mv {args.out} {args.out}.stale\n"
        )
    todo = [r for r in rows if r["qid"] not in done_qids]

    # One model call per DISTINCT response, one output row per qid. Identical
    # responses recur often (~9.5% of TL rows), and the judge is a pure function of
    # the response text, so a repeat can be filled from an answer we already have.
    distinct = {}
    for r in todo:
        distinct.setdefault(r["cache_key"], r)
    fresh = [r for k, r in distinct.items() if k not in by_key]

    print(f"[stage1] {len(rows):,} rows | {len(done_qids):,} already written | "
          f"{len(todo):,} outstanding | {len(distinct):,} distinct responses | "
          f"{len(fresh):,} model calls")
    if not todo:
        print("[stage1] nothing to do")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    t0 = time.time()
    n = 0
    emitted = set()

    with open(args.out, "a") as f:

        def emit(row):
            nonlocal n
            f.write(json.dumps(row) + "\n")
            emitted.add(row["qid"])
            n += 1

        # 1. Model calls, one per distinct unseen response.
        runner = run_mock if args.mock else run_gpu
        if fresh:
            produced = runner(fresh, args)
            if args.min_valid_rate > 0 and not args.mock:
                produced = gate_valid_rate(
                    produced, args.min_valid_rate,
                    min(VALID_RATE_PROBE_ROWS, len(fresh)),
                )
            for row in produced:
                by_key.setdefault(row.get("cache_key"), row)
                emit(row)
                # Flushed (and fsynced) every chunk so an interrupted run keeps
                # what it judged. Buffering to 20,000 rows meant ~37 minutes of
                # work vanished on any kill -- and the pod running this changed
                # three times in one session.
                if n % args.chunk_rows == 0:
                    f.flush()
                    os.fsync(f.fileno())
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"[stage1] {n:,}/{len(todo):,} "
                          f"({time.time()-t0:.0f}s, {rate:.1f} rows/s)", flush=True)

        # 2. Fan those answers out to every remaining qid with the same response.
        #    Covers both repeats within this run and rows whose twin was written
        #    before an interruption -- the case that a cache_key-only resume drops.
        n_fanned = 0
        for r in todo:
            if r["qid"] in emitted:
                continue
            prior = by_key.get(r["cache_key"])
            if prior is None:
                continue  # the model failed on that response; leave it for a rerun
            row = dict(prior)
            row.update({"qid": r["qid"], "model": r["model"], "file": r["file"],
                        "doc_id": r["doc_id"], "cache_key": r["cache_key"],
                        # Strip any marker the prior row already carries. by_key is
                        # last-wins, and a fanned row sits AFTER the judged one, so
                        # a resume would otherwise fan from an already-marked row
                        # and grow "+cached+cached..." once per resume -- which
                        # judge_decision._judge_reason embeds verbatim into
                        # judge_invalid:<reason> on invalid rows.
                        "judge_reason": ((prior.get("judge_reason") or "ok")
                                         .split("+cached")[0] + "+cached")})
            emit(row)
            n_fanned += 1

    print(f"[stage1] wrote {n:,} rows ({n - n_fanned:,} judged, {n_fanned:,} "
          f"reused) in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
