"""test-11: the judge registry, and the naming rules that keep readers apart.

The pipeline resolves its reader through judge_config.JUDGE_MODELS. Only
gemma-4-31b is registered today, but the table is what makes adding a second
reader safe, and three invariants carry that -- none of which fails loudly:

1. THE REGISTERED READER'S NAMES DO NOT MOVE. Every current artifact on disk --
   109 llm-parsed_gemma-4-31b*/ directories and their judge-out files -- is named
   from out_suffix. A rename would simply stop finding the existing corpus, and a
   rebuild is thirteen GPU-hours.

2. TWO READERS NEVER SHARE AN OUTPUT PATH. apply_judge.load_judge_index keys rows
   by (benchmarked_model, file) -> doc_id and takes last-wins; the judge is not in
   the key. Two readers in one file therefore interleave silently and produce a
   report blended from two instruments. Nothing downstream can detect it, so it
   has to be impossible by construction.

3. NO READER HAS AN EMPTY out_suffix. An earlier reader (retired 2026-08-17) wrote
   the unsuffixed judge-out_<task>.jsonl and llm-parsed/. While that was legal the
   driver needed a special case to keep `judge-out_TL*` and `llm-parsed*` from
   matching everything at once; with every suffix non-empty those globs are
   anchored by construction and the special case is gone. Registering
   out_suffix="" would re-widen every glob AND collapse invariant 2, since the
   suffix is the only thing keeping two readers in separate files.

Also pinned: the prompt fingerprint stays judge-INDEPENDENT. That is what lets a
new reader consume the queues already on disk instead of forcing a 1.1 GB rebuild,
and it is one careless edit away from being lost.

Run from the repo root:
    PYTHONPATH=src python unit-test/llm-parsing/test-11.py
"""

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from judge_config import (
    JUDGE_DEFAULT_KEY,
    JUDGE_MODEL_DEFAULT,
    JUDGE_MODELS,
    JUDGE_REASONING_EFFORT,
    judge_entry,
    judge_out_filename,
    judge_suffix,
    llm_parsed_dirname,
    resolve_judge_key,
)
from judge_prompts import prompt_fingerprint

TASKS = ("TL", "AD", "Detection")
failures = []


def check(name, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + extra) if extra and not ok else ''}")
    if not ok:
        failures.append(name)


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except SystemExit:
        return True
    except Exception:
        return False
    return False


print("test-11: judge registry")

# --- registry shape -------------------------------------------------------
REQUIRED = {"hf_id", "chat_kwargs", "out_suffix", "env",
            "requirements", "post_requirements", "torch_pin", "transformers_major",
            "tensor_parallel"}
check("every registered judge declares every field",
      all(REQUIRED <= set(v) for v in JUDGE_MODELS.values()),
      str({k: sorted(REQUIRED - set(v)) for k, v in JUDGE_MODELS.items()}))
check("gemma-4-31B-it is registered",
      JUDGE_MODELS.get("gemma-4-31b", {}).get("hf_id") == "google/gemma-4-31B-it")
check("the default key is registered", JUDGE_DEFAULT_KEY in JUDGE_MODELS)

# Invariant 3, at its source. NOT a tidiness rule: an empty suffix makes the
# driver's judge_out_ls/llm_parsed_ls globs (`judge-out_<task>*`, `llm-parsed*`)
# match every reader at once AND the unsuffixed archives of the retired reader, so
# --fresh would delete far more than it counted and printed.
empty = [k for k, v in JUDGE_MODELS.items() if not v["out_suffix"]]
check("NO judge has an empty out_suffix (keeps every driver glob anchored)",
      empty == [], str(empty))
suffixes = [v["out_suffix"] for v in JUDGE_MODELS.values()]
check("out_suffixes are unique", len(set(suffixes)) == len(suffixes), str(suffixes))
check("hub ids are unique",
      len({v["hf_id"] for v in JUDGE_MODELS.values()}) == len(JUDGE_MODELS))
# Each reader needs its own venv, so each needs its own requirements file present.
check("every judge's requirements file exists",
      all(os.path.isfile(os.path.join("script/llm-parsing", v["requirements"]))
          for v in JUDGE_MODELS.values()),
      str([v["requirements"] for v in JUDGE_MODELS.values()]))
check("every declared post-requirements file exists",
      all(os.path.isfile(os.path.join("script/llm-parsing", v["post_requirements"]))
          for v in JUDGE_MODELS.values() if v["post_requirements"]),
      str([v["post_requirements"] for v in JUDGE_MODELS.values()]))
# Readers get one venv EACH, and the reason is that their pins can be mutually
# unsatisfiable. With one reader registered there is nothing to compare, so pin the
# fact that actually forces the split: gemma-4-31b needs the 5.x line, which a vLLM
# declaring transformers<5 cannot host.
check("gemma-4-31b pins the transformers 5.x line",
      JUDGE_MODELS["gemma-4-31b"]["transformers_major"] == 5)
majors = [v["transformers_major"] for v in JUDGE_MODELS.values()]
if len(JUDGE_MODELS) > 1:
    check("judges pinning different transformers majors cannot share a venv",
          len(set(majors)) > 1 or len({v["requirements"] for v in JUDGE_MODELS.values()})
          == len(JUDGE_MODELS))

# The CPU stages (2-4) run under the SAME interpreter as the GPU judge and import
# medvision_bm, which needs datasets/nibabel. Nothing in Stage 1 touches those, so
# an env without them builds, verifies, sweeps for 13 GPU-hours and dies on Stage
# 2's first line. One shared file, installed for every reader.
CPU_REQS = "script/llm-parsing/requirements-cpu-stages.txt"
check("the shared CPU-stage requirements file exists", os.path.isfile(CPU_REQS))
if os.path.isfile(CPU_REQS):
    body = open(CPU_REQS).read()
    names = [ln.split("#")[0].strip().split("==")[0].split(">")[0].split("<")[0].lower()
             for ln in body.splitlines() if ln.split("#")[0].strip()]
    # Measured 2026-08-15: Stage 3 imports the task summarizers, which reach
    # lmms_eval.utils and load 90 third-party modules. These are the ones the judge
    # venvs lacked, each of which cost a pipeline run to discover one at a time.
    NEEDED = {"datasets", "nibabel", "pytz", "loguru", "sqlitedict", "evaluate",
              "sacrebleu", "scikit-learn", "scipy", "numexpr", "pandas", "av",
              "soundfile",
              # Reached only LAZILY, by importlib.import_module of a per-dataset
              # benchmark plan partway through Stage 3 -- invisible to any probe
              # that merely imports the summarizers, and reported as
              # "Error loading benchmark plan for <dataset>" rather than as a
              # missing package.
              "matplotlib", "simpleitk", "pynrrd"}
    check("it covers the measured medvision_bm/lmms_eval import chain",
          NEEDED <= set(names), f"missing {sorted(NEEDED - set(names))}")
    # It is installed into BOTH reader venvs, which disagree about transformers by
    # design. A pin on any package in that fight would silently break one of them.
    check("it does not pin anything the reader environments fight over",
          not ({"transformers", "vllm", "torch"} & set(names)), str(names))


# --- requirements files must be RESOLVABLE, not merely correct-looking ----
# The bug this guards: `vllm==0.19.0` and `transformers==5.10.2` were written into
# ONE requirements file. vLLM 0.19 declares `transformers<5`, and pip resolves a
# single file as a single constraint set -- so that is not an untidy pin, it is
# ResolutionImpossible and nothing installs at all. The override only works as a
# LATER pip invocation, which is what post_requirements is for. The comment in the
# file had the constraint right and the file itself contradicted it, so the guard
# has to read the files rather than the prose.
def pins(path):
    """{package: version} from a requirements file, ignoring comments/ranges."""
    out = {}
    with open(os.path.join("script/llm-parsing", path)) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "==" in line:
                name, _, ver = line.partition("==")
                out[name.strip().lower()] = ver.strip()
    return out


for key, v in sorted(JUDGE_MODELS.items()):
    base = pins(v["requirements"])
    post = pins(v["post_requirements"]) if v["post_requirements"] else {}

    # A package pinned in both files is a contradiction by construction: the whole
    # reason phase 2 exists is to disagree with phase 1.
    check(f"[{key}] no package is pinned in BOTH phases",
          not (set(base) & set(post)), str(sorted(set(base) & set(post))))

    # transformers must be pinned exactly once, at the major the registry claims.
    where = {**base, **post}
    tf = where.get("transformers")
    check(f"[{key}] transformers is pinned, at the registry's major",
          tf is not None and int(tf.split(".")[0]) == v["transformers_major"],
          f"pinned {tf}, registry says {v['transformers_major']}.x")

    # The load-bearing one. No vLLM release yet permits transformers 5.x, so a
    # reader needing 5.x may not pin it beside vllm -- that file cannot resolve.
    if v["transformers_major"] >= 5:
        check(f"[{key}] transformers 5.x is NOT pinned beside vllm (unresolvable)",
              not ("transformers" in base and "vllm" in base),
              f"both in {v['requirements']} -- pip would fail with ResolutionImpossible")
        check(f"[{key}] transformers 5.x IS pinned in the override pass",
              "transformers" in post)

# --- invariant 1: the registered reader's names have not moved -------------
# These are the names the 109 directories on disk already use. Changing any of
# them orphans the corpus silently -- Stage 2 would simply write a new tree beside
# the old one and Stage 3 would report on whichever it was pointed at.
check("the default judge is google/gemma-4-31B-it",
      JUDGE_MODEL_DEFAULT == "google/gemma-4-31B-it")
check("default judge -> _gemma-4-31b suffix",
      judge_suffix() == "_gemma-4-31b" == judge_suffix(JUDGE_DEFAULT_KEY))
check("llm-parsed_gemma-4-31b/ is the full-run directory",
      llm_parsed_dirname() == "llm-parsed_gemma-4-31b"
      and llm_parsed_dirname(None, JUDGE_DEFAULT_KEY) == "llm-parsed_gemma-4-31b")
check("llm-parsed_gemma-4-31b-limit100/ is the limited-run directory",
      llm_parsed_dirname(100) == "llm-parsed_gemma-4-31b-limit100")
check("judge-out_<task>_gemma-4-31b.jsonl is the full-run output",
      all(judge_out_filename(t) == f"judge-out_{t}_gemma-4-31b.jsonl" for t in TASKS))
check("judge-out_<task>_gemma-4-31b_limit100.jsonl is the limited-run output",
      all(judge_out_filename(t, 100) == f"judge-out_{t}_gemma-4-31b_limit100.jsonl"
          for t in TASKS))
# The reader suffix must precede the limit suffix, or one reader's files stop
# sharing a single prefix and the driver cannot archive them with one glob.
check("the reader suffix precedes the limit suffix",
      all(judge_out_filename(t, 100).index("_gemma-4-31b")
          < judge_out_filename(t, 100).index("_limit100") for t in TASKS))
# Gemma's template declares no reasoning_effort, and transformers 5.x raises on
# unknown template kwargs rather than ignoring them.
check("gemma-4 sends no chat-template kwargs",
      judge_entry("gemma-4-31b")["chat_kwargs"] == {})

# --- invariant 2: no two readers share a path -----------------------------
outs = [judge_out_filename(t, lim, k)
        for t in TASKS for lim in (None, 100) for k in JUDGE_MODELS]
check("no judge-out filename is reused across judges/limits",
      len(set(outs)) == len(outs), str(sorted(outs)))
dirs = [llm_parsed_dirname(lim, k) for lim in (None, 100) for k in JUDGE_MODELS]
check("no llm-parsed directory is reused across judges/limits",
      len(set(dirs)) == len(dirs), str(sorted(dirs)))

# Invariant 3, at the point where it actually bites: the driver globs
# `judge-out_<task>${JUDGE_SUFFIX}*` and `llm-parsed${JUDGE_SUFFIX}*`. Simulated
# against the names an EMPTY-suffix reader produces -- the shape the retired
# reader wrote, and the shape any future misregistration would write.
UNSUFFIXED = ["judge-out_TL.jsonl", "judge-out_TL.jsonl.v1", "judge-out_TL_limit100.jsonl"]
for key in JUDGE_MODELS:
    sfx = judge_suffix(key)
    glob_prefix = f"judge-out_TL{sfx}"
    check(f"[{key}] its own full-run output matches its own glob",
          judge_out_filename("TL", None, key).startswith(glob_prefix))
    check(f"[{key}] its own limited output matches its own glob",
          judge_out_filename("TL", 100, key).startswith(glob_prefix))
    check(f"[{key}] unsuffixed judge-out names do not match its glob",
          not any(n.startswith(glob_prefix) for n in UNSUFFIXED), str(UNSUFFIXED))
    check(f"[{key}] llm-parsed{sfx}* does not match a bare llm-parsed/",
          not "llm-parsed".startswith(f"llm-parsed{sfx}")
          and not "llm-parsed-limit100".startswith(f"llm-parsed{sfx}"))

# --- key resolution -------------------------------------------------------
check("a hub id resolves to its key",
      resolve_judge_key("google/gemma-4-31B-it") == "gemma-4-31b")
check("--judge outranks --model",
      resolve_judge_key("google/gemma-4-31B-it", JUDGE_DEFAULT_KEY) == JUDGE_DEFAULT_KEY)
check("the mock sentinel resolves to the default",
      resolve_judge_key("mock") == JUDGE_DEFAULT_KEY)
# Negative controls. An unknown checkpoint must ABORT rather than default -- and
# resolution comes ONLY from the registry, so an unregistered id (including any
# retired reader's) lands here. A wrong key silently sends the wrong chat kwargs
# and writes to the wrong directory, so guessing is never the safe fallback.
check("an unknown checkpoint aborts instead of defaulting",
      raises(resolve_judge_key, "some-org/some-model"))
check("an unknown --judge aborts", raises(resolve_judge_key, None, "not-a-judge"))
check("an unknown key aborts in judge_entry too", raises(judge_entry, "not-a-judge"))
check("an unknown key aborts in the filename helpers",
      raises(llm_parsed_dirname, None, "not-a-judge")
      and raises(judge_out_filename, "TL", None, "not-a-judge"))

with tempfile.TemporaryDirectory() as td:
    plain = os.path.join(td, "gemma-4-31b-mirror")
    os.makedirs(plain)
    check("an unmarked directory resolves by name prefix",
          resolve_judge_key(plain) == "gemma-4-31b")
    unknown = os.path.join(td, "mystery-weights")
    os.makedirs(unknown)
    check("an unmarked, unrecognised directory aborts",
          raises(resolve_judge_key, unknown))

# --- the mixing guard -----------------------------------------------------
from apply_judge import assert_single_judge_model  # noqa: E402


def write_out(path, models):
    with open(path, "w") as f:
        for i, m in enumerate(models):
            f.write(json.dumps({"qid": f"q{i}", "judge_model": m,
                                "model": "SomeVLM", "file": "a.jsonl",
                                "doc_id": i, "judge_status": "ok"}) + "\n")


with tempfile.TemporaryDirectory() as td:
    one = os.path.join(td, "one.jsonl")
    write_out(one, ["google/gemma-4-31B-it"] * 4)
    check("a single-judge file passes", not raises(assert_single_judge_model, one))

    # The case this guard exists for: a hand-run repair pass appending one reader's
    # rows onto another's file. Nothing downstream can detect the blend.
    mixed = os.path.join(td, "mixed.jsonl")
    write_out(mixed, ["google/gemma-4-31B-it", "some-org/another-reader"])
    check("a two-judge file ABORTS", raises(assert_single_judge_model, mixed))

    # Not a mixture: one reader named two ways. _model_key collapses filesystem
    # spellings of the same weights -- a trailing slash, a relative path, a symlink
    # -- because an exact string comparison turned any of those into "a different
    # judge" and hard-aborted a legitimate resume.
    mirror = os.path.join(td, "gemma-4-31b-mirror")
    os.makedirs(mirror)
    link = os.path.join(td, "gemma-link")
    os.symlink(mirror, link)
    spanned = os.path.join(td, "spanned.jsonl")
    write_out(spanned, [mirror, link, mirror + "/"])
    check("one local checkpoint named three ways is NOT reported as a mixture",
          not raises(assert_single_judge_model, spanned))

# --- GPU topology ---------------------------------------------------------
# Shards are the data-parallel axis; tensor_parallel is how many GPUs ONE shard's
# engine spans. Total GPUs = shards x TP. Two things must hold, and the second one
# fails expensively: a wrong grouping puts two engines on one card, which co-fit at
# gpu_memory_utilization=0.90 only long enough to OOM after both have loaded.
check("every tensor_parallel is a positive int",
      all(isinstance(v["tensor_parallel"], int) and v["tensor_parallel"] >= 1
          for v in JUDGE_MODELS.values()),
      str({k: v["tensor_parallel"] for k, v in JUDGE_MODELS.items()}))
check("gemma-4-31b asks for more than one GPU per engine (~62 GB of bf16 weights)",
      judge_entry("gemma-4-31b")["tensor_parallel"] > 1)

# judge_shard_devices lives in judge_env.sh because BOTH entry points launch shards
# and a disagreement about which GPUs a shard owns is silent until the OOM. Drive
# the real shell function rather than reimplementing its arithmetic here.
import subprocess  # noqa: E402


def shard_devices(shard, tp, devices):
    """Call the shell helper; returns None when it refuses (too few GPUs)."""
    script = (
        'set -euo pipefail\n'
        '. script/llm-parsing/judge_env.sh >/dev/null 2>&1\n'
        f'judge_shard_devices {shard} {tp} {" ".join(str(d) for d in devices)}\n'
    )
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


check("TP=1 gives one GPU per shard (unchanged default-reader behaviour)",
      [shard_devices(s, 1, [0, 1, 2, 3]) for s in range(4)] == ["0", "1", "2", "3"])
check("TP=2 on 4 GPUs gives 2 shards of 2, with no GPU in both",
      [shard_devices(s, 2, [0, 1, 2, 3]) for s in range(2)] == ["0,1", "2,3"])
check("TP=4 on 4 GPUs gives a single shard spanning all of them",
      shard_devices(0, 4, [0, 1, 2, 3]) == "0,1,2,3")
check("a pre-set CUDA_VISIBLE_DEVICES is honoured, not re-indexed",
      [shard_devices(s, 2, [4, 5, 6, 7]) for s in range(2)] == ["4,5", "6,7"])
check("asking for a shard beyond the device list FAILS instead of wrapping",
      shard_devices(2, 2, [0, 1, 2, 3]) is None)
check("TP larger than the device list FAILS instead of over-subscribing",
      shard_devices(0, 4, [0, 1]) is None)

# --- the fingerprint stays judge-independent ------------------------------
# If this ever fails, every queue on disk and all ~453K judge rows are invalidated
# by whatever edit caused it.
fps = {t: prompt_fingerprint(t, {"TL": "TL", "AD": "AD:distance", "Detection": None}[t])
       for t in TASKS}
check("no judge key or hub id leaks into the prompt fingerprint",
      not any(tok in fp
              for fp in fps.values()
              for tok in list(JUDGE_MODELS) + [v["hf_id"] for v in JUDGE_MODELS.values()]),
      "a reader-specific token reached prompt_fingerprint")
# This matters MORE now that no registered reader is sent reasoning_effort. The
# tempting cleanup -- "nothing uses it, drop it" -- would move every fingerprint on
# disk and invalidate the 1.1 GB of queues. It describes the prompt this code
# authors, not who is asked.
check("reasoning_effort is still IN the fingerprint for every task",
      all(json.loads(fp).get("reasoning_effort") == JUDGE_REASONING_EFFORT
          for fp in fps.values()))
check("no registered reader is actually SENT reasoning_effort",
      all("reasoning_effort" not in v["chat_kwargs"] for v in JUDGE_MODELS.values()))

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-11: all checks passed")
