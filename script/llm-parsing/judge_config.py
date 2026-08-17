"""LLM-as-Judge output parsing — task specs and run configuration.

Self-contained configuration for the judge-based parsing analysis. This folder
does not import ``medvision_bm`` for its own logic (mirroring
``script/analyze/clinical-decision-analysis/``), with one deliberate exception:
``apply_judge.py`` imports ``cal_metrics`` so judge-parsed samples are scored by
the *same* code that produced the published numbers. Re-implementing a metric
here would silently make the strict and judge columns incomparable.

THE JUDGE DOES EXACTLY TWO JOBS
-------------------------------
Job A -- final answer. Does the response state a final answer? If so, extract
    exactly ``k`` numbers (``TASK_SPECS[task]["arity"]``). Robust to any wrapper.

Job B -- intermediate results (TL and AD only). The benchmark prompts prescribe a
    fixed step structure; the judge extracts each step's numbers or marks the step
    absent. Detection is excluded: it has a single step whose answer *is* the
    final answer.

The judge is NOT asked for units, coordinate space, origin convention, or whether
a response was TRUNCATED. Those are cheaper and more reliable to derive
deterministically, and asking a model for them adds a failure mode that buys
nothing.

Truncation is the instructive case. It is a property of *why generation stopped* --
which lives in the run config, not in the text -- so no amount of reading the
response can settle it. Measured on T/L: the ``HuatuoGPT-Vision-34B`` run that set
``stop_strings=["</answer>"]`` left 84% of its responses without that stop string
and 64% ending mid-word; ``Qwen2.5-VL-32B`` ran with no stop string and never came
within half of its 4,096-token ceiling, so its failures are ordinary EOS. The two
look identical to a reader. Asking the judge to choose produced confident
mislabelling, so it is no longer asked.

Two cautions on that example, both learned later. (1) It is task-specific: the
same model on Detection shows a truncation signature on only 0.3% of samples, so
"HuatuoGPT is budget-limited" is not a property of the model. (2) It is now
historical -- every launcher sets an explicit output-token budget as of commit
09206a2. Deciding truncation still requires the generation config, which is the
point that survives.
"""

import os

# --- Judge identity -------------------------------------------------------
# Stamped into every output file. vLLM is not bitwise deterministic across batch
# compositions even at temperature=0, so we do not claim reproducible sampling --
# we release the judge output JSONL as an artifact. That makes the *identity* of
# the producing pipeline the thing that must be recorded, exactly as an annotation
# version identifies annotation data.
#
# That identity is `prompt_fp`, a hash of the fully rendered prompt (see
# judge_prompts.short_prompt_fp). There is deliberately no hand-maintained version
# string beside it: a constant someone must remember to bump is exactly the thing
# that goes stale when a prompt is edited, and it would then disagree with the
# hash about which prompt produced a row.

# Sampling. reasoning_effort="low" because this is extraction, not reasoning. It is
# part of prompt_fingerprint and so cannot move, but no CURRENT reader is sent it --
# only readers whose chat template declares the key receive it (see chat_kwargs).
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_SEED = 1024  # mirrors medvision_bm.utils.configs.SEED
JUDGE_REASONING_EFFORT = "low"
# Must fit the WORST prompt plus the full decode budget, or vLLM silently caps
# generation at the window instead of at max_tokens and those rows get less budget
# than configured -- on exactly the long, number-dense responses most likely to
# need it. The response window (below) is in CHARACTERS while this bound is in
# TOKENS, and number-dense CoT tokenises far below 4 chars/token, so the window
# does NOT bound this. Measured 2026-08-13 with the judge's own tokenizer over the
# FULL queues: worst prompt 5,316 (TL) / 5,222 (AD) tokens, against a comment that
# had long claimed "~2.8K". 5,316 + 4,096 = 9,412 > 8,192, so rows were already
# being clipped. (A first pass sampling the first 4,000 rows gave only 4,225 --
# queues are grouped by model, so a prefix covers 2 of 18 models. Sample the whole
# queue or not at all.) 12288 clears the real worst case with headroom.
# NOT part of prompt_fingerprint, so changing it does not invalidate any queue.
JUDGE_MAX_MODEL_LEN = 12288

# --- Judge model registry -------------------------------------------------
# Which reader runs. The table survives having a single entry because the judge is
# a measurement instrument: "how much apparent failure was formatting?" is only
# credible if the answer does not turn on one model's idiosyncrasies, and the
# cheapest available cross-check is re-reading the same queues with a different
# reader. The prompts, the span verification and the metrics are all reader-
# independent, so two reports would differ ONLY by the reader. Adding one back is
# an entry here rather than a hunt through five modules:
#
#   hf_id        upstream checkpoint.
#   chat_kwargs  extra chat_template_kwargs for LLM.chat. A template that does not
#                know a key ignores it at best and raises at worst, so each entry
#                declares only what its own template reads.
#   tensor_parallel  GPUs ONE engine spans. See the topology note below -- this is
#                a capacity floor, not a speed knob.
#   out_suffix   appended to judge-out_<task>.jsonl and to llm-parsed/. MUST be
#                non-empty -- see the warning below.
#   env          the environment this reader needs. Quoted in errors because
#                readers can be mutually exclusive: Gemma-4 requires transformers
#                >= 5.5.0 (eval__gemma4.install_transformers_for_gemma4), which a
#                vLLM release declaring `transformers<5` cannot host. One venv per
#                reader; setup_judge_env.sh takes --judge.
#
# WHY out_suffix IS NOT COSMETIC. apply_judge.load_judge_index keys rows by
# (evaluated_model, file) -> doc_id and takes LAST WINS. That "model" is the
# benchmarked VLM, not the judge, so two readers' rows in one judge-out file do
# not collide loudly -- they interleave, and whichever reader wrote a given doc_id
# last silently supplies that record's verdict. A report built from such a file is
# a blend of two instruments with nothing anywhere saying so. Separate filenames
# make that impossible by construction; assert_single_judge_model in apply_judge.py
# catches the hand-run paths that bypass the driver.
#
# WHY *EVERY* SUFFIX IS NON-EMPTY, INCLUDING THE DEFAULT'S. An earlier reader
# (retired 2026-08-17) wrote the unsuffixed names -- judge-out_<task>.jsonl and
# llm-parsed/. While that was legal, `judge-out_TL*` and `llm-parsed*` matched
# EVERY reader at once, so the driver carried a special case to stop --fresh
# deleting one reader's records while retiring another's. With no empty suffix
# registered, every glob is anchored on a reader name by construction and the
# special case is gone.
#
# Keep it that way. The danger is not hypothetical bookkeeping: out_suffix is the
# ONLY thing separating two readers on disk, so registering one with out_suffix=""
# both re-widens every driver glob and puts that reader's rows in the same
# judge-out file as another's, where load_judge_index blends them silently.
# test-11 pins it.
#
# GPU TOPOLOGY: TENSOR PARALLEL *WITHIN* A SHARD, DATA PARALLEL *ACROSS* SHARDS.
# The sweep runs one vLLM process per shard, each taking every Nth queue row, and
# `tensor_parallel` says how many GPUs one of those processes spans. Total GPUs
# used = NUM_SHARDS * tensor_parallel.
#
# Raise it only to fit the weights, never to go faster. Data parallelism is the
# efficient axis here: the shards share nothing, so they scale near-linearly and
# each resumes independently after a preemption, while TP inserts an all-reduce
# after every layer. A sparse model whose per-layer compute is small next to that
# all-reduce should stay at tensor_parallel=1 and let the extra cards run extra
# shards.
#
# A dense 31B model inverts the argument, and not marginally. ~62 GB of bf16
# weights against an 80 GB card at gpu_memory_utilization=0.90 (72 GB) leaves ~10 GB
# for the KV cache; at JUDGE_MAX_MODEL_LEN=12288 that is a handful of concurrent
# sequences, so the engine is capacity-bound and throughput collapses -- and on any
# smaller card it simply OOMs during load. TP=2 halves the weights per GPU and
# hands the freed ~30 GB to the KV cache, which is where the batching comes from.
# The eval path reaches the same conclusion from the other direction: it runs
# Gemma-4 at tensor_parallel_size=num_processes (eval__gemma4.py), i.e. pure TP.
#
# These are capacity floors derived from parameter counts, not measurements.
# Override per run with TP=<n> / --tensor_parallel_size.
#
# WHAT IS DELIBERATELY *NOT* HERE: anything that reaches prompt_fingerprint. The
# fingerprint identifies the PROMPT and is judge-independent on purpose, which is
# what lets a new reader consume the queues already on disk instead of forcing a
# 1.1 GB rebuild. Judge identity is carried separately, by the judge_model stamp on
# every row, and load_done checks the two independently. Note the asymmetry this
# creates for reasoning_effort: the constant above stays in the fingerprint (the
# prompt was authored for it) even though no CURRENT reader is sent it, while
# chat_kwargs decides only who actually receives it. Moving JUDGE_REASONING_EFFORT
# into this table would invalidate every queue on disk.
JUDGE_MODELS = {
    "gemma-4-31b": {
        "hf_id": "google/gemma-4-31B-it",
        # Ships plain bf16, so there is no checkpoint preparation step: vLLM
        # fetches the weights on first load. 31B dense at bf16 is ~62 GB, which
        # does not fit one 80 GB card beside a usable KV cache -- hence TP=2.
        # Gemma templates take no reasoning_effort. Passing it would be inert at
        # best; transformers 5.x is stricter about unknown template kwargs than
        # the 4.x line this pipeline was built on, so it is simply not sent.
        "chat_kwargs": {},
        # ~62 GB of bf16 weights. TP=1 leaves ~10 GB for the KV cache on an 80 GB
        # card and OOMs on anything smaller; TP=2 leaves ~41 GB per GPU, which is
        # where the concurrency comes from. Starting value from the parameter
        # count, not a measurement -- raise it if the engine reports a small KV
        # cache, and note vLLM requires it to divide the attention-head count.
        "tensor_parallel": 2,
        "out_suffix": "_gemma-4-31b",
        "env": ("vllm==0.19.0 (requirements-gemma-4-31b.txt) then transformers==5.10.2 "
                "(requirements-gemma-4-31b-post.txt, a SECOND pip pass)"),
        "requirements": "requirements-gemma-4-31b.txt",
        # A SECOND pip pass. Two pins here contradict phase-1 metadata --
        # vllm 0.19 declares `transformers<5` and torch 2.10 pins nccl with `==`
        # -- and pip resolves one requirements file as a single constraint set,
        # so expressing them there is ResolutionImpossible, not merely untidy.
        # A later invocation resolves against the INSTALLED set, warns, and
        # proceeds. Same two-phase shape as eval__gemma4.py.
        "post_requirements": "requirements-gemma-4-31b-post.txt",
        "torch_pin": "torch==2.10.0",
        # Not a floor with slack: Gemma-4's config declares transformers_version
        # 5.5.0.dev0 and no 4.x release can load it. setup_judge_env.sh checks the
        # major it built against, so a resolver that quietly lands on the 4.x line
        # fails at setup rather than at the first chat() call after a 62 GB
        # download.
        "transformers_major": 5,
    },
}

# The only registered reader, and the one every CURRENT artifact on disk was
# produced by. Its out_suffix is non-empty on purpose -- see the warning above.
JUDGE_DEFAULT_KEY = "gemma-4-31b"
JUDGE_MODEL_DEFAULT = JUDGE_MODELS[JUDGE_DEFAULT_KEY]["hf_id"]

# Response windowing. Answers live at the tail, so the tail gets the larger share.
# Applied only when the response exceeds RESPONSE_WINDOW_TRIGGER characters
# (<5% of records); the elision is marked explicitly so the judge does not mistake
# a windowed response for a truncated one.
RESPONSE_WINDOW_TRIGGER = 8000
RESPONSE_WINDOW_HEAD = 2000
RESPONSE_WINDOW_TAIL = 6000
RESPONSE_ELISION_MARKER = "\n...[{n} characters elided]...\n"

# Span contract. The judge quotes; a regex transcribes. See apply_judge.py.
MAX_ANSWER_SPAN_CHARS = 200
MAX_STEP_SPAN_CHARS = 120
MIN_SPAN_CONTEXT_CHARS = 12

# --- Task specs -----------------------------------------------------------
# "arity"       -- how many numbers the final answer must contain (Job A). This is
#                  the same k the strict parser uses (parse_outputs.py:201-216), and
#                  it MUST stay in sync: a different k would change which samples
#                  count as parsed and make coverage incomparable.
# "description" -- shown to the judge so it knows what a final answer looks like.
# "max_tokens"  -- decode budget. Too small truncates the JSON mid-object, which
#                  surfaces as a judge-invalid rate rather than as an obvious error,
#                  so this is part of the prompt fingerprint.

# One generous default for every task. Budget-caused truncation cost two GPU
# repair passes to diagnose (TL at 1024, then Detection at 256), because it
# never fails loudly -- it just inflates judge-invalid. A large budget is nearly
# free: decoding stops at EOS, so only rows that NEED the room pay for it, plus
# the ~0.3% degenerate repetition loops that run to whatever cap exists.
# Headroom check: worst MEASURED prompt 5,316 tokens + 4096 fits
# JUDGE_MAX_MODEL_LEN=12288 (see the note there; 4,225 was an earlier bad sample
# over one model's prefix and must not be quoted). Changing this value moves EVERY
# prompt fingerprint and therefore forces a queue rebuild -- see judge_prompts.
DEFAULT_JUDGE_MAX_TOKENS = 4096

TASK_SPECS = {
    "TL": {
        "arity": 2,
        "description": (
            "two numbers: the major axis length and the minor axis length "
            "of an ellipse, in millimetres, in that order"
        ),
        # History: 1024 was measured to be too small (8.70% judge-invalid; TL
        # asks for five verbatim spans plus JSON scaffolding); re-judging at
        # 3072 rescued 76% of the failures. Now on the shared generous default.
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
    },
    "AD": {
        "arity": 1,
        "description": (
            "a single number: either a distance in millimetres or an angle in degrees"
        ),
        # Four spans rather than five, and correspondingly a lower failure rate
        # (5.02%) -- but the same cause, so the same remedy.
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
    },
    "Detection": {
        "arity": 4,
        "description": (
            "four numbers: the bounding-box coordinates "
            "x_lower_left, y_lower_left, x_upper_right, y_upper_right"
        ),
        # History: 256 was measured to be too small (88.9% of residual failures
        # sat at EXACTLY 256 completion tokens; the controlled repair at 512
        # recovered 98.1% vs a 63.0% same-budget control). Detection's measured
        # requirement is tiny -- p99.5 = 327 tokens -- so the shared default is
        # pure headroom here; only degenerate repetition loops (~0.3%) ever
        # decode past ~350 tokens.
        "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
    },
}

# --- Job A status and the derived answer-mode enum ------------------------
# The judge reports one of two statuses: an answer is stated, or it is not. Neither
# "absent" (the v1 value) nor "truncated" (briefly used, then withdrawn -- see the
# module docstring) is accepted. Rejecting them is what makes a judge-output file
# from either of those prompts unusable rather than silently reinterpretable.
FINAL_ANSWER_STATUSES = ("present", "no_conclusion")
STEP_STATUSES = ("present", "absent")

# Per-record category written to llm-parsed/*.jsonl as LLM_judge_answer_mode.
# "conclusion_in_format" is decided by the STRICT parser, not the judge: it means
# the published regex already succeeded. Everything else is the judge's contribution.
# There is deliberately no "truncation" mode: see the module docstring. A response
# that was cut off before stating an answer lands in "no_conclusion" -- the record
# says an answer was not stated, which is true, and does not claim to know why.
ANSWER_MODES = (
    "conclusion_in_format",  # <answer>..</answer> with the right arity
    "conclusion_off_format",  # judge found a span-verified answer the regex missed
    "no_conclusion",  # no answer was stated
    "undetermined",  # judge unusable AND the regex failed
)
SUCCESS_MODES = ("conclusion_in_format", "conclusion_off_format")

# --- Job B step schemas ---------------------------------------------------
# Mirrors the reasoning steps the benchmark prompts prescribe
# (medvision_bm/sft/sft_prompts.py: COT_INSTRUCT_TL_NORM, COT_INSTRUCT_DISTANCE,
# COT_INSTRUCT_ANGLE). "n_values" is how many numbers that step must yield:
# a coordinate pair is 2, a two-endpoint line is 4, a scalar is 1.
#
# AD is keyed by biometric_profile.metric_type, because the distance and angle
# prompts prescribe *different* step contents under the same task type.
STEP_SPECS = {
    "TL": [
        {"index": 1, "n_values": 4, "what": "the two endpoints of the MAJOR axis, as (x1,y1),(x2,y2)"},
        {"index": 2, "n_values": 4, "what": "the two endpoints of the MINOR axis, as (x1,y1),(x2,y2)"},
        {"index": 3, "n_values": 1, "what": "the computed physical length of the major axis"},
        {"index": 4, "n_values": 1, "what": "the computed physical length of the minor axis"},
    ],
    "AD:distance": [
        {"index": 1, "n_values": 2, "what": "the coordinates of landmark 1, as (x1,y1)"},
        {"index": 2, "n_values": 2, "what": "the coordinates of landmark 2, as (x2,y2)"},
        {"index": 3, "n_values": 1, "what": "the computed physical distance"},
    ],
    "AD:angle": [
        {"index": 1, "n_values": 4, "what": "the two endpoints of LINE 1, as (x1,y1),(x2,y2)"},
        {"index": 2, "n_values": 4, "what": "the two endpoints of LINE 2, as (x1,y1),(x2,y2)"},
        {"index": 3, "n_values": 1, "what": "the computed angle"},
    ],
}


def step_spec_key(task_type, metric_type=None):
    """Return the ``STEP_SPECS`` key for a sample, or ``None`` if it has no steps.

    Args:
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        metric_type: For AD, ``doc["biometric_profile"]["metric_type"]``
            (``"distance"`` or ``"angle"``). Ignored otherwise.

    Returns:
        str | None: Key into ``STEP_SPECS``, or ``None`` when the task defines no
        intermediate steps (Detection) or the AD metric type is unrecognised.
    """
    if task_type == "TL":
        return "TL"
    if task_type == "AD":
        if metric_type in ("distance", "angle"):
            return f"AD:{metric_type}"
        return None
    return None


# --- Output naming --------------------------------------------------------
# The limit belongs in the *directory* name, not only the report name. Otherwise a
# --limit pilot leaves N-row files in llm-parsed/ and a later full summarizer run
# over that directory silently reports on N rows per file with no error. The strict
# pipeline gets away with this because it regenerates parsed/ on the next full run;
# regenerating a GPU sweep is not free.
LLM_PARSED_DIRNAME = "llm-parsed"

# Records in llm-parsed/ have "filtered_resps" REMOVED and this key in its place, so
# the summarizers must be told which key to read (--resps_key). The removal is the
# point: a record that carries both would let a reader silently score the wrong one.
RESPS_KEY_STRICT = "filtered_resps"
RESPS_KEY_LLM = "LLM_filtered_resps"

SUMMARY_FILENAME_JUDGE_METRICS = "summary_metrics_judge_Task.json"

# Stems excluded from every parsed/*.jsonl discovery glob in this repo. New
# analysis outputs MUST be added here and to the existing summarizers' filters,
# or they get picked up as sample files and corrupt downstream metrics.
EXCLUDED_JSONL_STEMS = ("_proc_acc", "_eq_acc", "_judge")


def judge_entry(judge=None):
    """Return the ``JUDGE_MODELS`` entry for ``judge``.

    Args:
        judge: A registry key, or ``None`` for the default reader.

    Returns:
        dict: The registry entry.

    Raises:
        SystemExit: If ``judge`` is not a registered key. Loud rather than
            defaulted: an unrecognised key that silently fell back to the default
            would take the default's out_suffix with it and write a second
            reader's rows into the first reader's files.
    """
    key = judge or JUDGE_DEFAULT_KEY
    if key not in JUDGE_MODELS:
        raise SystemExit(
            f"\nABORT unknown judge {key!r}.\n"
            f"  known : {', '.join(sorted(JUDGE_MODELS))}\n"
            f"  Add an entry to JUDGE_MODELS in judge_config.py to introduce a new one.\n"
        )
    return JUDGE_MODELS[key]


def resolve_judge_key(model, explicit=None):
    """Map a checkpoint identifier to its ``JUDGE_MODELS`` key.

    ``--model`` names weights; the key names behaviour (which chat kwargs to send,
    whether the checkpoint needs converting, where the output goes). Usually the
    weights determine the key, so this infers it -- but it never GUESSES, because
    every consequence of a wrong key is silent: the wrong out_suffix mixes two
    readers into one file, and the wrong chat kwargs change what the reader was
    asked without changing any recorded provenance.

    Recognised, in order: an explicit key; an exact hub id; a local directory
    whose basename starts with a key.

    Args:
        model: The ``--model`` string, or ``None``.
        explicit: The ``--judge`` value, if the caller passed one. Wins outright.

    Returns:
        str: A key into ``JUDGE_MODELS``.

    Raises:
        SystemExit: If nothing matches, naming the flag that fixes it.
    """
    if explicit:
        judge_entry(explicit)  # validates
        return explicit
    if not model or model == "mock":
        return JUDGE_DEFAULT_KEY
    m = str(model).rstrip("/")
    by_hf = {v["hf_id"]: k for k, v in JUDGE_MODELS.items()}
    if m in by_hf:
        return by_hf[m]
    if os.path.isdir(m):
        base = os.path.basename(os.path.realpath(m))
        for k in sorted(JUDGE_MODELS, key=len, reverse=True):
            if base.startswith(k):
                return k
    raise SystemExit(
        f"\nABORT cannot tell which judge {model!r} is.\n"
        f"  known : {', '.join(sorted(JUDGE_MODELS))}\n\n"
        f"  The key decides where output is written and which chat-template kwargs\n"
        f"  are sent, so it is not guessed. Name it:  --judge <key>\n"
    )


def judge_suffix(judge=None):
    """Return the filename suffix that keeps two readers' output apart."""
    return judge_entry(judge)["out_suffix"]


def llm_parsed_dirname(limit=None, judge=None):
    """Return the output directory name for judge-parsed records.

    Args:
        limit: The ``--limit`` value in force, or ``None`` for a full run.
        judge: A ``JUDGE_MODELS`` key, or ``None`` for the default reader.

    Returns:
        str: ``"llm-parsed"`` plus the reader's suffix, plus ``-limit{N}`` on a
        limited run -- e.g. ``"llm-parsed_gemma-4-31b"``. Every registered reader
        has a non-empty suffix, so the bare ``"llm-parsed"`` is never produced.
    """
    name = f"{LLM_PARSED_DIRNAME}{judge_suffix(judge)}"
    if limit is None:
        return name
    return f"{name}-limit{int(limit)}"


def limit_suffix(limit=None):
    """Return the filename suffix encoding the active ``--limit``.

    Args:
        limit: The ``--limit`` value in force, or ``None`` for a full run.

    Returns:
        str: ``""`` for a full run, ``"_limit{N}"`` otherwise. Matches the suffix
        convention already used by the summarizers.
    """
    if limit is None:
        return ""
    return f"_limit{int(limit)}"


def queue_filename(task_type, limit=None):
    """Return the Stage 0 queue filename for a task."""
    return f"judge-queue_{task_type}{limit_suffix(limit)}.jsonl"


def judge_out_filename(task_type, limit=None, judge=None):
    """Return the Stage 1 judge-output filename for a task.

    The reader suffix precedes the limit suffix, so every file belonging to one
    reader shares the prefix ``judge-out_<task><reader_suffix>`` and the driver can
    archive one reader with a single glob. That prefix is unambiguous only because
    no registered reader has an empty suffix -- an empty one would make
    ``judge-out_TL*`` match every reader at once.
    """
    return f"judge-out_{task_type}{judge_suffix(judge)}{limit_suffix(limit)}.jsonl"


# --- Stage 1 safety -------------------------------------------------------
# Minimum fraction of judge rows that must validate against the schema before the
# sweep is allowed to continue past its first batch. The v1 sweep returned 3 valid
# rows out of 43,938 and ran for thirteen hours anyway, because nothing checked.
# Rows per llm.chat() call. vLLM returns only when every prompt in the call has
# finished, so this is also the unit of durability: an interrupted sweep loses
# at most one chunk. 2,000 keeps the engine saturated while capping the loss at
# a few minutes instead of the whole task.
CHUNK_ROWS_DEFAULT = 2000

MIN_VALID_RATE_DEFAULT = 0.95
VALID_RATE_PROBE_ROWS = 200


# --- Roster configs -------------------------------------------------------
# The paper's evaluated-model roster. Results/ additionally holds superseded
# duplicates (e.g. HealthGPT-L14 *and* HealthGPT-L14_bugfix-0a4c5e2); judging all
# of them triples GPU cost and risks quoting numbers computed over the wrong dirs.
#
# 2026-08-19: the fullSFT checkpoint (MedVision__fullSFT__...__v2) was appended to
# all three rosters AFTER its own view-dir campaign finished, by merging that
# campaign's judge-out and queue rows into the main trees (same prompt stamps, so
# no re-judge). The rosters therefore hold 18 paper models + 1 late addition, and
# EXPECTED_ROSTER_COUNTS below includes its rows.
#
# The roster YAMLs live IN THIS DIRECTORY, copied from script/visualization/ (where
# the plotting scripts read them from). Two consequences worth knowing:
#
# - Resolved against __file__, not the working directory. These are configuration,
#   so a stage launched from anywhere must find the same roster; a CWD-relative
#   path resolves to nothing and the Stage 0 gate then reports "no such directory"
#   for all 18 models, which reads as a broken results tree rather than a bad path.
# - A roster edit upstream must be copied in. The Stage 0 count gate
#   (EXPECTED_ROSTER_COUNTS) is what makes that omission loud instead of silent:
#   a roster that gained or lost a model no longer produces the expected response
#   count, and the build aborts before any GPU time is spent.
_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ROSTER_YAML = {
    "TL": os.path.join(_HERE, "config-TL-CoT.yaml"),
    "AD": os.path.join(_HERE, "config-AD-CoT.yaml"),
    "Detection": os.path.join(_HERE, "config-detect-CoT.yaml"),
}

DEFAULT_TASK_DIR = {
    "TL": "Results/MedVision-TL-v2-CoT",
    "AD": "Results/MedVision-AD-v2-CoT",
    "Detection": "Results/MedVision-detect-v2",
}

# Expected response counts over the roster, used as a Stage 0 self-check. A
# mismatch means roster resolution is wrong -- fix it before spending GPU time.
# 18 paper models + the fullSFT checkpoint: per-model rows are 2,441 (TL), 2,060
# (AD), 23,071 (Detection) -- except fullSFT Detection at 22,071 (its eval lacks
# BCV15_BoxCoordinate_Task01_Axial.jsonl, 27 files not 28; real, not a bug).
EXPECTED_ROSTER_COUNTS = {"TL": 46379, "AD": 39140, "Detection": 437349}


# --- Shell interface ------------------------------------------------------
# judge_env.sh needs the registry, and the registry is Python. Re-typing the table
# in bash would recreate exactly the drift judge_env.sh was written to end: the two
# entry points disagreeing about which checkpoint they are running, discovered only
# when a resume aborts. So the shell asks this module instead, and there is still
# one table. Cheap enough to call unconditionally -- this file imports only `os`,
# so there is no torch startup here.
def _emit_shell(judge):
    """Print `NAME=value` lines describing one judge, for `eval` in bash."""
    import shlex

    key = judge or JUDGE_DEFAULT_KEY
    e = judge_entry(key)
    for name, value in (
        ("JUDGE_KEY", key),
        ("JUDGE_MODEL_HF", e["hf_id"]),
        ("JUDGE_SUFFIX", e["out_suffix"]),
        # GPUs per shard. The shell needs it to carve CUDA_VISIBLE_DEVICES into
        # groups and to default NUM_SHARDS to devices/TP rather than devices.
        ("JUDGE_TP", str(e["tensor_parallel"])),
        ("JUDGE_ENV_HINT", e["env"]),
        # For setup_judge_env.sh. The venv name carries out_suffix for the same
        # reason the output files do: building the second reader's environment
        # must not overwrite the first reader's working one.
        ("JUDGE_REQUIREMENTS", e["requirements"]),
        ("JUDGE_POST_REQUIREMENTS", e["post_requirements"]),
        ("JUDGE_TORCH_PIN", e["torch_pin"]),
        ("JUDGE_TRANSFORMERS_MAJOR", str(e["transformers_major"])),
        ("JUDGE_ENV_BASENAME", f"judge-env{e['out_suffix']}"),
    ):
        print(f"{name}={shlex.quote(value)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Query the judge-model registry.")
    ap.add_argument("--judge", default=None, help="registry key (default: %s)" % JUDGE_DEFAULT_KEY)
    ap.add_argument("--shell", action="store_true", help="emit NAME=value lines for eval")
    ap.add_argument("--list", action="store_true", help="list the registered judges")
    a = ap.parse_args()
    if a.list:
        width = max(len(k) for k in JUDGE_MODELS)
        for k in sorted(JUDGE_MODELS):
            mark = "*" if k == JUDGE_DEFAULT_KEY else " "
            print(f" {mark} {k:<{width}}  {JUDGE_MODELS[k]['hf_id']:<26}  {JUDGE_MODELS[k]['env']}")
    else:
        _emit_shell(a.judge)
