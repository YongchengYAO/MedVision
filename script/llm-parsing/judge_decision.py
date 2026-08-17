"""The one place that decides what a judged sample's answer is.

Stage 2, the Stage 4 statistics and the unit tests all call ``decide_answer``, so
the decision table exists exactly once. Reimplementing it per consumer is how the
"format failure" and "genuine non-answer" columns of a report drift apart from the
records they claim to summarise.

The strict-first rule
---------------------
Whenever the published regex parser succeeded, its answer wins -- unconditionally,
whatever the judge said. The judge can only ADD recoveries; it can never revise a
number that is already published.

Two things follow, and both are load-bearing:

1. ``LLM_filtered_resps == filtered_resps`` on every strict-success row, for every
   model. The "a format-compliant model is unaffected" gate is therefore an
   identity that holds across the whole roster, not a property special to
   MedVision-V0 -- so any violation is a merge bug, never a finding.
2. A judge failure can only ever cost a *recovery*. It cannot corrupt a published
   value, which is what lets the judge-invalid rate be a quality metric rather
   than a correctness risk.

The table
---------
Inputs: whether the strict parser produced an answer; whether a judge row exists;
its ``judge_status``; its ``final_answer.status``; and whether ``verify_span``
accepted the quoted span at the task's arity. Two strict outcomes x five judge
outcomes = ten rows, all enumerated in ``DECISION_TABLE_DOC`` below and asserted
one-for-one in ``unit-test/llm-parsing/test-7.py``.
"""

from judge_config import ANSWER_MODES, SUCCESS_MODES  # noqa: F401  (re-exported)

# Kept as data so the test can iterate it and the README can quote it without
# either drifting from the implementation.
DECISION_TABLE_DOC = """
 # | strict | judge row | judge_status | final_answer.status | span ok | pred     | mode
---+--------+-----------+--------------+---------------------+---------+----------+----------------------
 1 |  ok    | present   | ok           | present             |  yes    | strict   | conclusion_in_format
 2 |  ok    | present   | ok           | present             |  no     | strict   | conclusion_in_format
 3 |  ok    | present   | ok           | no_conclusion       |   -     | strict   | conclusion_in_format
 4 |  ok    | present   | invalid      |  -                  |   -     | strict   | conclusion_in_format
 5 |  ok    | ABSENT    |  -           |  -                  |   -     | strict   | conclusion_in_format
 6 | fail   | present   | ok           | present             |  yes    | verified | conclusion_off_format
 7 | fail   | present   | ok           | present             |  no     | ""       | undetermined
 8 | fail   | present   | ok           | no_conclusion       |   -     | ""       | no_conclusion
 9 | fail   | present   | invalid      |  -                  |   -     | ""       | undetermined
10 | fail   | ABSENT    |  -           |  -                  |   -     | ""       | undetermined
"""


def decide_answer(strict_pred, judge_row, verified):
    """Resolve one sample's answer, category and reason.

    Args:
        strict_pred: The published regex parser's prediction for this sample --
            a comma-joined number string, or ``""`` when it failed.
        judge_row: The Stage 1 output row for this sample, or ``None`` when the
            judge produced no row for it (rows 5 and 10).
        verified: The ``judge_verify.verify_span`` result dict for this sample, or
            ``None`` when verification was not attempted (no judge row, an invalid
            row, or a non-``present`` status). A dict with ``ok=True`` carries the
            regex-transcribed ``pred``.

    Returns:
        tuple: ``(pred, mode, reason)``.

        - ``pred``: what to write to ``LLM_filtered_resps[0]``.
        - ``mode``: one of ``ANSWER_MODES``.
        - ``reason``: a short diagnostic string for ``LLM_judge.reason``. It
          records what the judge did even on strict-success rows, where it had no
          influence on ``pred`` -- that is what makes judge-vs-regex agreement
          measurable over the ~450K rows the regex already handles.
    """
    strict_ok = bool((strict_pred or "").strip())
    reason = _judge_reason(judge_row, verified)

    # Rows 1-5: the published answer stands, whatever the judge concluded.
    if strict_ok:
        return strict_pred, "conclusion_in_format", reason

    # Rows 9 and 10: nothing usable came back from the judge.
    if judge_row is None:
        return "", "undetermined", reason
    if judge_row.get("judge_status") != "ok":
        return "", "undetermined", reason

    status = _final_answer_status(judge_row)

    # Row 8: the judge read the response and found no answer in it. It is not asked
    # WHY -- a response cut off mid-working and one that simply declines both land
    # here, because separating them needs the generation config, not the text.
    if status == "no_conclusion":
        return "", "no_conclusion", reason

    # Row 6: a recovery, but only once the span verifier has re-derived the numbers
    # from the quoted text. Row 7: the judge claimed an answer it could not point at.
    if status == "present" and verified is not None and verified.get("ok"):
        return verified["pred"], "conclusion_off_format", reason
    return "", "undetermined", reason


def _final_answer_status(judge_row):
    """Return ``final_answer.status``, or ``None`` if the shape is not an object.

    Defensive on purpose. ``validate_judge_obj`` should already have stamped a
    non-object ``final_answer`` as invalid, but the v1 sweep produced bare strings
    (``"(40.542, 28.799)"``), bare floats and bare lists on ~90% of rows, and a
    ``.get`` on any of those raises. A crash 300,000 rows into a Detection sweep
    costs hours; returning ``None`` costs one row, which then lands in
    ``undetermined`` exactly as an unusable judge answer should.
    """
    fa = judge_row.get("final_answer")
    return fa.get("status") if isinstance(fa, dict) else None


def _judge_reason(judge_row, verified):
    """Summarise what the judge contributed, independent of what was decided."""
    if judge_row is None:
        return "no_judge_record"
    if judge_row.get("judge_status") != "ok":
        return f"judge_invalid:{judge_row.get('judge_reason') or 'unknown'}"
    status = _final_answer_status(judge_row)
    if status == "no_conclusion":
        return "judge_no_conclusion"
    if verified is None:
        return "span_not_checked"
    if verified.get("ok"):
        return "ok"
    return f"span_{verified.get('reason') or 'unverified'}"


def is_success(mode):
    """Return whether a mode counts as a successfully extracted answer.

    ``LLM_judge_SR.success`` is exactly this, and ``LLM_filtered_resps[0] != ""``
    is exactly this too -- three spellings of one fact, asserted against each other
    in ``test-7.py`` and again over real records in ``test-8.py``.
    """
    return mode in SUCCESS_MODES
