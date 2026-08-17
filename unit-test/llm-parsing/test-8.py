"""test-8: invariants that must hold over REAL llm-parsed/ records on disk.

test-7 proves the decision function is right in isolation. This one proves the
records Stage 2 actually wrote agree with it, over whatever output exists.

The invariant that does the most work is the strict-superset identity:

    the strict parser succeeded  =>  LLM_filtered_resps == filtered_resps

It holds for EVERY model, not only a format-compliant one, because the strict-first
rule is unconditional. That makes it a merge-bug detector rather than a finding: if
a single strict-success row's value moved, something in Stage 2 is rewriting
published numbers and no amount of judge quality can excuse it.

Also checks the schema contract that ``--resps_key`` depends on: the strict key
must be GONE. If both keys were present, a summarizer run without the flag would
silently score the published values while claiming to report judge-parsed ones.

Usage (from the repo root). Pass a directory name; defaults to the registered
reader's full-run directory (llm_parsed_dirname(), e.g. llm-parsed_gemma-4-31b):
    python unit-test/llm-parsing/test-8.py [llm-parsed_gemma-4-31b-limit100]

Skips cleanly (exit 0) when no such directory exists yet.
"""

import glob
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from judge_config import (
    ANSWER_MODES,
    RESPS_KEY_LLM,
    RESPS_KEY_STRICT,
    SUCCESS_MODES,
    llm_parsed_dirname,
)

# Metric blocks Stage 2 is allowed to drop, because it cannot recompute them and
# a stale strict-parser value beside recomputed ones is worse than an absent key.
#
# nMAE belongs here: _compute_nmae returns None when neither the canonical
# diagonal helper nor the strict MAE/nMAE pair can supply an image diagonal, and
# apply_judge then pops the key so the summarizer recomputes it (DESIGN section 3).
# Omitting it made a DOCUMENTED drop fail this test -- and run_llm_parsing.sh runs
# test-8 as the `invariants` gate with `|| die`, so an environment without the
# lmms_eval import would abort the whole task on intended behaviour.
DROPPABLE = {"MAE", "MRE", "avgMRE", "nMAE"}

# Derived, not a literal: the driver runs this as its `invariants` gate with no
# argument, and a hardcoded default would check the wrong reader's directory (or
# none at all) the moment the registered reader changes.
DIRNAME = sys.argv[1] if len(sys.argv) > 1 else llm_parsed_dirname()
# "llm-parsed-limit100" -> 100. The row-count invariant has to know the cap,
# because a limited run is SUPPOSED to hold fewer records than its source.
_m = re.search(r"-limit(\d+)$", DIRNAME)
LIMIT = int(_m.group(1)) if _m else None
# Set by the driver (and by a hand-run `MOCK=1 ...`). Decides which DIRECTION the
# provenance check runs in -- see the mock check below.
MOCK_RUN = os.environ.get("MOCK") == "1"
# The driver's invariants gate is PER TASK, but this test scans all three task
# trees. Under `TASKS=<subset> MOCK=1` the untouched trees still hold real-judge
# records and failed the mock assertion for a task the run never touched. Honour
# the same TASKS variable the driver uses; unset means all tasks, as before.
_TASKS_ENV = os.environ.get("TASKS", "").split()
# TASK_DIR_<task> re-points a task at a different Results tree -- the same
# variables run_llm_parsing.sh and test-sweep.sh honour, so the invariants gate
# scans the tree the run actually wrote (the OOD splits) rather than always the
# main benchmark tree.
TASK_DIRS = {
    "TL": os.environ.get("TASK_DIR_TL", "Results/MedVision-TL-v2-CoT"),
    "AD": os.environ.get("TASK_DIR_AD", "Results/MedVision-AD-v2-CoT"),
    "Detection": os.environ.get("TASK_DIR_Detection", "Results/MedVision-detect-v2"),
}

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print(f"test-8: llm-parsed record invariants (dir={DIRNAME!r})")

found_any = False
for task, task_dir in TASK_DIRS.items():
    if _TASKS_ENV and task not in _TASKS_ENV:
        continue
    files = sorted(glob.glob(os.path.join(task_dir, "*", DIRNAME, "*_samples_*.jsonl")))
    files = [f for f in files if not any(x in os.path.basename(f)
                                         for x in ("_proc_acc", "_eq_acc", "_judge"))]
    if not files:
        continue
    found_any = True

    n = 0
    bad_strict_key = bad_llm_key = 0
    bad_mode = bad_sr_mode = bad_sr_value = 0
    bad_superset = bad_steps = bad_order = bad_dropped = 0
    n_mock = bad_dup = bad_count = 0
    n_strict_ok = 0
    modes = dict.fromkeys(ANSWER_MODES, 0)

    for f in files:
        src = f.replace(os.sep + DIRNAME + os.sep, os.sep + "parsed" + os.sep)
        strict_by_doc, strict_order = {}, {}
        if os.path.exists(src):
            with open(src) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    s = json.loads(line)
                    vals = s.get(RESPS_KEY_STRICT) or [""]
                    strict_by_doc[s["doc_id"]] = str(vals[0]) if vals else ""
                    # Key order with the strict key renamed in place -- what the
                    # output is supposed to look like.
                    strict_order[s["doc_id"]] = [
                        RESPS_KEY_LLM if k == RESPS_KEY_STRICT else k for k in s
                    ]

        # I9: exactly one llm-parsed record per source record. DESIGN section 12
        # credited this test with enforcing it, but nothing here counted rows or
        # collected doc_ids -- every per-record check is a doc_id LOOKUP, which a
        # duplicate satisfies twice. Both halves are now actually asserted.
        seen_docs = set()
        n_file = 0

        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                n_file += 1
                _doc = r.get("doc_id")
                if _doc in seen_docs:
                    bad_dup += 1
                seen_docs.add(_doc)

                if RESPS_KEY_STRICT in r:
                    bad_strict_key += 1
                vals = r.get(RESPS_KEY_LLM)
                if not (isinstance(vals, list) and len(vals) == 1):
                    bad_llm_key += 1
                    continue
                written = vals[0]

                mode = r.get("LLM_judge_answer_mode")
                if mode not in ANSWER_MODES:
                    bad_mode += 1
                    continue
                modes[mode] += 1

                sr = (r.get("LLM_judge_SR") or {}).get("success")
                if sr is not (mode in SUCCESS_MODES):
                    bad_sr_mode += 1
                if sr is not bool(written.strip()):
                    bad_sr_value += 1

                doc = r.get("doc_id")
                if doc in strict_by_doc:
                    strict = strict_by_doc[doc]
                    if strict.strip():
                        n_strict_ok += 1
                        if written != strict or mode != "conclusion_in_format":
                            bad_superset += 1
                    # The surviving source keys must appear in the SAME RELATIVE
                    # ORDER, which is what "renamed in place" means. A prefix
                    # comparison would be wrong: Stage 2 deliberately drops metric
                    # blocks it cannot recompute -- AD's legacy MAE/MRE (an older
                    # lmms-eval schema) and Detection's avgMRE (which
                    # cal_metrics_detection_task never produces). Dropping those is
                    # the point; carrying a strict-parser number into a record whose
                    # other metrics came from the judge is mixed provenance.
                    kept = [k for k in strict_order[doc] if k in r]
                    if [k for k in r if k in strict_order[doc]] != kept:
                        bad_order += 1
                    # and only the documented blocks may go missing
                    dropped = set(strict_order[doc]) - set(r)
                    if dropped - DROPPABLE:
                        bad_dropped += 1

                # Provenance. `run_judge_vllm --mock` substitutes a regex
                # stand-in for the judge; its rows are stamped judge_model
                # "mock" and apply_judge copies that onto the record. A mock
                # sweep must never be mistaken for a real one -- the numbers
                # are meaningless. (Records written before judge_model was
                # propagated simply carry None, which is not "mock".)
                if ((r.get("LLM_judge") or {}).get("judge_model")) == "mock":
                    n_mock += 1

                has_steps = "LLM_judge_steps" in r
                if task == "Detection" and has_steps:
                    bad_steps += 1
                if task in ("TL", "AD") and not has_steps:
                    bad_steps += 1

        # One record out per record in -- but a --limit run writes only the first
        # N per file BY DESIGN (apply_judge passes limit to iter_records), and the
        # limit lives in the DIRECTORY NAME. Comparing against the full source
        # made every pilot fail, and run_llm_parsing.sh runs this as the pilot's
        # `invariants` gate with `|| die`, so the gate that clears a run before 13
        # GPU-hours was the thing blocking it.
        expected_rows = len(strict_by_doc)
        if LIMIT is not None:
            expected_rows = min(LIMIT, expected_rows)
        if strict_by_doc and n_file != expected_rows:
            bad_count += 1

    print(f"\n  [{task}] {len(files)} file(s), {n:,} records")
    check(f"{task}: strict key removed from every record", bad_strict_key == 0,
          f"{bad_strict_key} records still carry {RESPS_KEY_STRICT!r}")
    check(f"{task}: {RESPS_KEY_LLM} present, one element", bad_llm_key == 0)
    check(f"{task}: every mode is declared", bad_mode == 0)
    check(f"{task}: LLM_judge_SR agrees with the mode", bad_sr_mode == 0)
    check(f"{task}: LLM_judge_SR agrees with the written answer", bad_sr_value == 0)
    check(f"{task}: strict successes preserved verbatim ({n_strict_ok:,} rows)",
          bad_superset == 0, f"{bad_superset} rows diverge")
    check(f"{task}: surviving source keys keep their relative order",
          bad_order == 0, f"{bad_order} rows reordered")
    check(f"{task}: only the documented metric blocks were dropped",
          bad_dropped == 0, f"{bad_dropped} rows dropped something else")
    check(f"{task}: Job B present iff the task prescribes steps", bad_steps == 0)
    check(f"{task}: I9 -- no duplicated doc_id in any file", bad_dup == 0,
          f"{bad_dup} duplicate doc_id(s)")
    check(f"{task}: I9 -- one record per source record", bad_count == 0,
          f"{bad_count} file(s) differ in row count from parsed/")
    # Provenance must be CONSISTENT with how the run was made, in both directions.
    # An unconditional "no mock records" check made the documented CPU-only
    # workflow (`MOCK=1 bash run_llm_parsing.sh`) impossible to finish: the driver
    # runs this test as the invariants gate with `|| die`, and under MOCK=1 every
    # record is stamped mock by construction, so the gate failed 100% of the time.
    if MOCK_RUN:
        check(f"{task}: mock run -- every record is stamped mock", n_mock == n,
              f"{n - n_mock} of {n} records are NOT mock; a mock run must not mix "
              f"with real judge output")
    else:
        check(f"{task}: no record came from the --mock stand-in", n_mock == 0,
              f"{n_mock} records carry judge_model='mock' -- these are regex "
              f"stand-in verdicts, not judge output; never report them")
    print(f"    modes: " + ", ".join(f"{k}={v:,}" for k, v in modes.items() if v))

    check(f"{task}: no record claims a truncation mode",
          "truncation" not in modes and
          all(r != "truncation" for r in modes))

if not found_any:
    print(f"\n  no {DIRNAME}/ directories found -- nothing to check (not a failure)")
    sys.exit(0)

print()
if failures:
    print(f"FAILED ({len(failures)}): {failures}")
    sys.exit(1)
print("test-8: all checks passed")
