"""test-4: naming and arity invariants that keep judge output separable and comparable.

Two failure modes this guards against, both silent:

1. A ``--limit`` pilot writing into the same directory or report name as a full
   run. The published pipeline gets away with reusing names because it regenerates
   ``parsed/`` on the next full run; regenerating a GPU sweep is not free, so a
   100-row pilot left in ``llm-parsed/`` would be summarized as if it were the
   whole corpus.
2. The judge's expected value count drifting from the strict parser's ``k``. That
   would change which samples count as parsed and make coverage incomparable.

Run from the repo root:
    python unit-test/llm-parsing/test-4.py
"""

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from judge_config import (
    STEP_SPECS,
    TASK_SPECS,
    ANSWER_MODES,
    judge_out_filename,
    FINAL_ANSWER_STATUSES,
    RESPS_KEY_LLM,
    RESPS_KEY_STRICT,
    SUCCESS_MODES,
    step_spec_key,
    limit_suffix,
    llm_parsed_dirname,
    queue_filename,
    step_spec_key,
)

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print("test-4: naming and arity invariants")

# --- limit must reach the DIRECTORY name, not just the report name --------
# Expressed against the reader's own base name rather than a literal: which reader
# is registered is judge_config's business (test-11 pins the actual names), while
# the invariant here -- the limit reaches the DIRECTORY, not just the report -- is
# reader-independent and must survive a reader swap.
_base = llm_parsed_dirname(None)
check("full run -> the reader's bare directory", _base.startswith("llm-parsed"))
check("limit run -> <base>-limit100", llm_parsed_dirname(100) == f"{_base}-limit100")
check("limit dir differs from full dir", llm_parsed_dirname(None) != llm_parsed_dirname(100))
check("distinct limits -> distinct dirs", llm_parsed_dirname(100) != llm_parsed_dirname(200))
check("limit suffix empty for full run", limit_suffix(None) == "")
check("limit suffix for pilot", limit_suffix(100) == "_limit100")

for t in ("TL", "AD", "Detection"):
    check(f"{t}: queue name carries the limit",
          queue_filename(t, 100) != queue_filename(t, None))
    check(f"{t}: judge-out name carries the limit",
          judge_out_filename(t, 100) != judge_out_filename(t, None))
names = {queue_filename(t, l) for t in ("TL", "AD", "Detection") for l in (None, 100)}
check("all queue names unique across task x limit", len(names) == 6)

# --- summarizer report qualifier ------------------------------------------
from medvision_bm.benchmark.summarize_TL_task import _parsed_dir_suffix as tl_suffix
from medvision_bm.benchmark.summarize_AD_task import _parsed_dir_suffix as ad_suffix
from medvision_bm.benchmark.summarize_detection_task import _parsed_dir_suffix as dt_suffix

for nm, fn in (("TL", tl_suffix), ("AD", ad_suffix), ("Detection", dt_suffix)):
    check(f"{nm}: default parsed dir adds no qualifier", fn("parsed") == "")
    check(f"{nm}: judge dir adds a qualifier",
          fn("llm-parsed-limit100") == "__llm-parsed-limit100")

# --- arity must equal the strict parser's k -------------------------------
# parse_outputs.py sets k per task type (AD=1, TL=2, Detection=4). Read it from
# the source rather than restating it, so this test breaks if that file changes.
src = pathlib.Path("src/medvision_bm/benchmark/parse_outputs.py").read_text()
strict_k = {}
for task, pat in (
    ("AD", r'task_type == "AD":\s*\n\s*target_nums = (\d+)'),
    ("TL", r'task_type == "TL":\s*\n\s*target_nums = (\d+)'),
    ("Detection", r'task_type == "Detection":\s*\n\s*target_nums = (\d+)'),
):
    m = re.search(pat, src)
    if m:
        strict_k[task] = int(m.group(1))
check(f"read k from parse_outputs.py: {strict_k}", len(strict_k) == 3)
for task, k in strict_k.items():
    check(f"{task}: judge arity {TASK_SPECS[task]['arity']} == strict k {k}",
          TASK_SPECS[task]["arity"] == k)

# --- step specs -----------------------------------------------------------
check("TL routes to the TL step spec", step_spec_key("TL") == "TL")
check("AD distance routes correctly", step_spec_key("AD", "distance") == "AD:distance")
check("AD angle routes correctly", step_spec_key("AD", "angle") == "AD:angle")
check("AD with unknown metric type has no steps", step_spec_key("AD", "weird") is None)
check("Detection has no steps", step_spec_key("Detection") is None)
# has_steps was removed: step_spec_key() is the single router, and a second
# declaration of the same fact is a second place for it to go stale.
check("Detection has no step spec", step_spec_key("Detection") is None)
check("TL has a step spec", step_spec_key("TL") == "TL")
check("resps keys are distinct", RESPS_KEY_STRICT != RESPS_KEY_LLM)
check("llm resps key is LLM_filtered_resps", RESPS_KEY_LLM == "LLM_filtered_resps")
check("success modes are a subset of answer modes",
      set(SUCCESS_MODES) < set(ANSWER_MODES))
check("four answer modes", len(ANSWER_MODES) == 4)
check("no truncation mode", "truncation" not in ANSWER_MODES)
# Both withdrawn statuses stay rejected: "absent" (v1) and "truncated" (the
# short-lived three-way version). Truncation depends on the generation config,
# not the response text, so the judge is not asked for it.
check("v1 'absent' is not a valid status", "absent" not in FINAL_ANSWER_STATUSES)
check("'truncated' is not a valid status", "truncated" not in FINAL_ANSWER_STATUSES)
check("two final-answer statuses", len(FINAL_ANSWER_STATUSES) == 2)

check("TL prescribes 4 steps", len(STEP_SPECS["TL"]) == 4)
check("AD distance prescribes 3 steps", len(STEP_SPECS["AD:distance"]) == 3)
check("AD angle prescribes 3 steps", len(STEP_SPECS["AD:angle"]) == 3)
for key, specs in STEP_SPECS.items():
    check(f"{key}: step indices are 1..n in order",
          [s["index"] for s in specs] == list(range(1, len(specs) + 1)))
    check(f"{key}: coordinate steps are even-valued, scalars are 1",
          all(s["n_values"] in (1, 2, 4) for s in specs))

# TL steps 3/4 are scalar lengths; steps 1/2 are two endpoints each.
check("TL step 1 is two endpoints (4 numbers)", STEP_SPECS["TL"][0]["n_values"] == 4)
check("TL step 3 is a scalar", STEP_SPECS["TL"][2]["n_values"] == 1)
check("AD distance step 1 is one point (2 numbers)",
      STEP_SPECS["AD:distance"][0]["n_values"] == 2)

# --- token budget must fit Job B output -----------------------------------
# TL emits up to 5 spans (1 answer + 4 steps); 256 tokens would truncate the JSON
# mid-object and show up as a spuriously high judge-invalid rate.
# Every task shares one GENEROUS budget by design. Per-task "right-sized"
# budgets caused two silent truncation incidents (TL at 1024, Detection at 256),
# each costing a GPU repair pass to diagnose -- decoding stops at EOS, so
# headroom is nearly free and a floor is the property worth asserting.
from judge_config import DEFAULT_JUDGE_MAX_TOKENS, JUDGE_MAX_MODEL_LEN
check("a single shared decode budget",
      all(TASK_SPECS[t]["max_tokens"] == DEFAULT_JUDGE_MAX_TOKENS for t in TASK_SPECS))
check("the shared budget is generous (>= 4096)", DEFAULT_JUDGE_MAX_TOKENS >= 4096)
# 3000 was a guess that measurement falsified, and 4225 was a BAD measurement:
# queues are grouped by model, so the first 4,000 rows cover 2 of 18 roster models.
# Over the FULL queues with the judge's own tokenizer the worst prompt is 5,316
# tokens (TL), 5,222 (AD). The window constants are in CHARS and cannot bound this;
# only a token measurement over the whole corpus can.
WORST_PROMPT_TOKENS = 5316
check("budget + worst MEASURED prompt fits the model window",
      DEFAULT_JUDGE_MAX_TOKENS + WORST_PROMPT_TOKENS <= JUDGE_MAX_MODEL_LEN,
      f"{DEFAULT_JUDGE_MAX_TOKENS} + {WORST_PROMPT_TOKENS} > {JUDGE_MAX_MODEL_LEN}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-4: all checks passed")
