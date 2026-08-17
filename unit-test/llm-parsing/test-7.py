"""test-7: the decision table, row by row, plus the invariants it must satisfy.

``decide_answer`` is the only place that decides what a judged sample's answer is,
so every consumer inherits its mistakes. The table is small enough to enumerate
completely -- 2 strict outcomes x 5 judge outcomes -- and this test enumerates it.

The invariants matter more than the individual rows:

  * ``strict ok  =>  the published value is kept, unchanged, always.``
    This is what makes a judge failure cost a recovery rather than corrupt a
    number, and it is what turns the "a compliant model is unaffected" gate into
    an identity that holds for the whole roster.
  * ``SR  <=>  mode is a conclusion_*``  and  ``SR  <=>  a non-empty answer``.
    Three spellings of one fact; if they can disagree, a report can contradict the
    records it summarizes.

The regression inputs at the end are the ACTUAL malformed shapes the v1 reader
returned during that sweep -- ``final_answer`` as a bare string, a float, a list,
and ``None``. They must all land in ``undetermined`` rather than crash.

Run from the repo root:
    python unit-test/llm-parsing/test-7.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from judge_config import ANSWER_MODES, SUCCESS_MODES
from judge_decision import decide_answer, is_success

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print("test-7: decision table")

STRICT = "10.0,20.0"
JUDGED = "30.0,40.0"


def row(judge_status=None, fa_status=None, present=True):
    """Build a Stage 1 output row, or None to model 'no judge record'."""
    if not present:
        return None
    r = {"judge_status": judge_status}
    if judge_status == "ok":
        r["final_answer"] = {"status": fa_status, "span": "s", "values": [30.0, 40.0]}
    else:
        r["judge_reason"] = "missing_final_answer"
    return r


OK_VERIFIED = {"ok": True, "pred": JUDGED, "reason": "ok", "numbers": ["30.0", "40.0"]}
NOT_VERIFIED = {"ok": False, "pred": "", "reason": "values_not_in_span", "numbers": []}

# (#, strict_pred, judge_row, verified, expected_pred, expected_mode)
TABLE = [
    (1, STRICT, row("ok", "present"), OK_VERIFIED, STRICT, "conclusion_in_format"),
    (2, STRICT, row("ok", "present"), NOT_VERIFIED, STRICT, "conclusion_in_format"),
    (3, STRICT, row("ok", "no_conclusion"), None, STRICT, "conclusion_in_format"),
    (4, STRICT, row("invalid"), None, STRICT, "conclusion_in_format"),
    (5, STRICT, row(present=False), None, STRICT, "conclusion_in_format"),
    (6, "", row("ok", "present"), OK_VERIFIED, JUDGED, "conclusion_off_format"),
    (7, "", row("ok", "present"), NOT_VERIFIED, "", "undetermined"),
    (8, "", row("ok", "no_conclusion"), None, "", "no_conclusion"),
    (9, "", row("invalid"), None, "", "undetermined"),
    (10, "", row(present=False), None, "", "undetermined"),
]

check("table enumerates 2 strict outcomes x 5 judge outcomes", len(TABLE) == 10)

# Truncation is a property of the generation config, not the response text, so it
# is not a mode the judge can assign. A judge that returns the withdrawn
# "truncated" status is rejected upstream and must land in undetermined, never in
# a category of its own.
_p, _m, _r = decide_answer("", {"judge_status": "ok",
                                "final_answer": {"status": "truncated",
                                                 "span": "s", "values": [1.0]}}, None)
check("withdrawn 'truncated' status has no mode of its own", _m == "undetermined",
      f"got {_m!r}")
check("no mode is named truncation", "truncation" not in ANSWER_MODES)
check("four answer modes", len(ANSWER_MODES) == 4)

seen_modes = set()
for n, strict, jrow, ver, want_pred, want_mode in TABLE:
    pred, mode, reason = decide_answer(strict, jrow, ver)
    seen_modes.add(mode)
    check(f"row {n:>2}: pred", pred == want_pred, f"got {pred!r}, want {want_pred!r}")
    check(f"row {n:>2}: mode", mode == want_mode, f"got {mode!r}, want {want_mode!r}")
    check(f"row {n:>2}: reason is non-empty", bool(reason))
    check(f"row {n:>2}: mode is a declared mode", mode in ANSWER_MODES)

check("the table exercises every declared mode", seen_modes == set(ANSWER_MODES),
      f"missing {set(ANSWER_MODES) - seen_modes}")

# --- invariants -----------------------------------------------------------
for n, strict, jrow, ver, want_pred, want_mode in TABLE:
    pred, mode, _ = decide_answer(strict, jrow, ver)
    check(f"row {n:>2}: SR <=> conclusion_*",
          is_success(mode) == (mode in SUCCESS_MODES))
    check(f"row {n:>2}: SR <=> non-empty answer",
          is_success(mode) == bool(pred.strip()))
    if strict.strip():
        check(f"row {n:>2}: strict success is preserved verbatim", pred == strict)
        check(f"row {n:>2}: strict success is always in-format",
              mode == "conclusion_in_format")

# The judge cannot revise a published number under ANY judge outcome. Stated
# separately from the per-row asserts because it is the property the whole
# fail-safe design rests on.
strict_rows = [t for t in TABLE if t[1].strip()]
check("no judge outcome can change a strict-parsed answer",
      all(decide_answer(s, j, v)[0] == s for _, s, j, v, _, _ in strict_rows))
check("all five judge outcomes were tried against a strict success",
      len(strict_rows) == 5)

# --- regression: the real v1 malformed shapes -----------------------------
# These are the actual objects the v1 reader returned when the prompt described
# the schema without showing it. validate_judge_obj stamps them "invalid" upstream, so
# decide_answer sees judge_status != "ok" -- but it must also survive being handed
# the raw shapes, because a future decoder change could let one through.
V1_SHAPES = [
    ("bare string", {"judge_status": "ok", "final_answer": "(40.542, 28.799)"}),
    ("float", {"judge_status": "ok", "final_answer": 40.542}),
    ("list", {"judge_status": "ok", "final_answer": [58.854, 43.159]}),
    ("None", {"judge_status": "ok", "final_answer": None}),
    ("missing key", {"judge_status": "ok"}),
    ("values-only dict", {"judge_status": "ok",
                          "final_answer": {"values": ["29.426", "20.881"]}}),
    ("invalid status", {"judge_status": "invalid",
                        "judge_reason": "missing_final_answer"}),
]
for label, jrow in V1_SHAPES:
    try:
        pred, mode, reason = decide_answer("", jrow, None)
        ok = (pred == "") and (mode == "undetermined") and bool(reason)
    except Exception as e:  # a crash here would take down a 415K-row sweep
        ok, mode = False, f"raised {type(e).__name__}: {e}"
    check(f"v1 shape survives: {label}", ok, str(mode))
    # And the same shape must not be able to overwrite a strict success.
    try:
        pred2, mode2, _ = decide_answer(STRICT, jrow, None)
        ok2 = pred2 == STRICT and mode2 == "conclusion_in_format"
    except Exception as e:
        ok2 = False
    check(f"v1 shape cannot overwrite a strict success: {label}", ok2)

print()
if failures:
    print(f"FAILED ({len(failures)}): {failures[:8]}")
    sys.exit(1)
print("test-7: all checks passed")
