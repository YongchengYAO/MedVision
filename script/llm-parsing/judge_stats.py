"""Stage 4 metric formulas: failure decomposition and judge validity.

Scope discipline
----------------
This module deliberately defines NO new model-quality metric. Format-robust
MAE/MRE/SR/IoU come from the existing summarizers reading ``llm-parsed*/`` with
``--resps_key LLM_filtered_resps`` (Stage 3). What lives here is only:

  A. Failure decomposition -- splitting the strict parser's failures into
     "an answer was stated, the regex just missed it" versus "no answer was
     stated".
  B. Judge validity -- evidence that the judge can be trusted at all.

Job B (intermediate reasoning) is counted only as *extraction coverage*: how
often the judge located each step. That is a property of the extraction, not a
score of the model, and it is reported as such.

Denominator convention follows ``clinical-decision-analysis/cda_stats.py``, the
one place in the repo that already reports parsed-only and coverage-adjusted
numbers side by side. Every rate here names its denominator explicitly.
"""


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_config import ANSWER_MODES  # noqa: E402


def _rate(num, den):
    """Return ``num/den``, or NaN when the denominator is zero."""
    return (num / den) if den else float("nan")


def decompose_failures(records):
    """Split strict-parser failures into format failures and genuine non-answers.

    A strict failure means the response had no ``<answer>`` block holding *k*
    numbers. The judge then tells us which of two very different things happened:

      * **format failure** -- an answer WAS stated, just not in the required
        wrapper (``\\boxed{...}``, ``**Answer:**``, plain prose). Charging these
        to measurement ability is exactly the confound W3 names.
      * **genuine non-answer** -- no value is stated anywhere: empty output, a
        refusal, or working that stops part-way. These are reported together on
        purpose; telling "ran out of room" from "declined to answer" needs the
        generation settings, not the text.

    Args:
        records: Iterable of ``llm-parsed*`` records carrying an ``LLM_judge``
            block and an ``LLM_judge_answer_mode``.

    Returns:
        dict: Counts and rates. ``format_failure_share`` is over strict failures
        (how much of the apparent failure was format), while
        ``*_rate`` fields are over all samples.
    """
    n = n_strict_ok = n_judge_ok = 0
    by_mode = {m: 0 for m in ANSWER_MODES}

    for rec in records:
        n += 1
        j = rec.get("LLM_judge") or {}
        strict_ok = bool((j.get("strict_pred") or "").strip())
        judge_ok = bool((rec.get("LLM_judge_SR") or {}).get("success"))
        n_strict_ok += strict_ok
        n_judge_ok += judge_ok
        mode = rec.get("LLM_judge_answer_mode")
        if mode in by_mode:
            by_mode[mode] += 1

    n_strict_fail = n - n_strict_ok
    # A recovery is exactly conclusion_off_format -- the strict-first rule means no
    # other mode can produce an answer the regex did not already have.
    n_recovered = by_mode["conclusion_off_format"]
    return {
        "n_samples": n,
        "n_strict_parsed": n_strict_ok,
        "n_judge_parsed": n_judge_ok,
        "n_strict_failed": n_strict_fail,
        "n_format_failure": n_recovered,
        # ONLY no_conclusion. This is the one number in the report attributed to
        # the MODEL, so it must not absorb the judge's own failures: the residual
        # (n_strict_fail - n_recovered) also contains `undetermined`, which means
        # the judge was unusable or its span was rejected -- nothing about whether
        # the model stated a value. Charging those to the model inflated the column
        # by the judge's failure rate. `n_undetermined` reports them separately.
        "n_genuine_non_answer": by_mode["no_conclusion"],
        "success_rate_strict": _rate(n_strict_ok, n),
        "success_rate_judge": _rate(n_judge_ok, n),
        "delta_success_rate": _rate(n_judge_ok, n) - _rate(n_strict_ok, n),
        # Of everything the strict parser rejected, what share was recoverable?
        "format_failure_share": _rate(n_recovered, n_strict_fail),
        "non_answer_rate": _rate(by_mode["no_conclusion"], n),
        # The non-answers, split by whether the judge could speak to them at all.
        # Deliberately NOT split into truncated-vs-declined: that depends on the
        # generation config, not the response text, so the judge is not asked.
        "n_no_conclusion": by_mode["no_conclusion"],
        "n_undetermined": by_mode["undetermined"],
        "no_conclusion_rate": _rate(by_mode["no_conclusion"], n),
        "undetermined_rate": _rate(by_mode["undetermined"], n),
        "by_mode": by_mode,
    }


def judge_validity(records):
    """Measure whether the judge can be trusted, using the regex as ground truth.

    The regex successes are a large, effectively-correct reference set: 33,854 on
    TL, 27,355 on AD, 387,910 on Detection (Stage 0 baselines), and on the rows it
    accepts the strict parser is near-perfect. So agreement here is a real
    reliability measurement, not a formality -- and it is free, which is the main
    reason to judge 100% of the corpus rather than only the failures.

    Agreement compares ``strict_pred`` against ``LLM_judge.judge_pred``, which is
    the judge's OWN span-verified extraction -- deliberately not the value that was
    written to ``LLM_filtered_resps``. Under the strict-first rule the written value
    equals ``strict_pred`` on every row in this reference set, so comparing against
    it would report 100% agreement by construction.

    Args:
        records: Iterable of ``llm-parsed*`` records carrying an ``LLM_judge``
            block.

    Returns:
        dict: Agreement rate over regex successes, plus the rates at which the
        judge was unusable (invalid JSON, span not verified, record missing).
    """
    n_ref = agree = 0
    n_value_disagree = n_empty_disagree = 0
    empty_reasons = {}
    disagreements = []
    n = n_invalid = n_unverified = n_missing = 0
    # Span verification is the anti-hallucination mechanism, so WHY a span was
    # rejected is reportable evidence rather than noise: each reason is a distinct
    # way the judge claimed an answer it could not point at.
    span_reasons = {}

    for rec in records:
        n += 1
        j = rec.get("LLM_judge") or {}
        reason = j.get("reason") or ""
        # Health is read from the reason, not from the answer: the strict-first
        # rule means a strict-success row keeps its value no matter how badly the
        # judge did, so the mode alone cannot reveal an unusable judge.
        n_invalid += reason.startswith("judge_invalid")
        n_missing += reason == "no_judge_record"
        if reason.startswith("span_"):
            n_unverified += 1
            # "span_" + verify_span's own reason, e.g. span_span_not_found.
            key = reason[len("span_"):]
            span_reasons[key] = span_reasons.get(key, 0) + 1

        strict = (j.get("strict_pred") or "").strip()
        if not strict:
            continue  # regex failed: no reference to agree with
        n_ref += 1
        if strict == (j.get("judge_pred") or "").strip():
            agree += 1
        elif (j.get("judge_pred") or "").strip():
            # The EXAMPLES list holds only rows where the judge actually read a
            # number that differs. A row whose judge was invalid, missing or
            # span-rejected carries judge_pred == "" by construction, and letting
            # those in filled the 20-slot audit list with "the judge said nothing",
            # crowding out the value disagreements a reviewer opens it to find.
            if len(disagreements) < 20:
                disagreements.append(
                    {
                        "doc_id": rec.get("doc_id"),
                        "strict": strict,
                        "judge": j.get("judge_pred"),
                        "reason": j.get("reason"),
                    }
                )
            n_value_disagree += 1
        else:
            # ...but they are still disagreements and must not vanish. Filtering
            # them out of the examples ALSO removed the only trace of
            # judge_no_conclusion on a strict success -- the judge validly read the
            # response and found no answer where the regex found one -- which is
            # not covered by judge_invalid_rate or span_unverified_rate. Counted by
            # reason so the volume survives even though the examples do not.
            n_empty_disagree += 1
            empty_reasons[j.get("reason") or "unknown"] = (
                empty_reasons.get(j.get("reason") or "unknown", 0) + 1)

    return {
        "n_reference": n_ref,
        "n_agree": agree,
        "agreement_rate": _rate(agree, n_ref),
        # The two halves of (n_reference - n_agree), so no disagreement is lost:
        # one where the judge read a DIFFERENT number, one where it read none.
        "n_value_disagreement": n_value_disagree,
        "n_no_value_disagreement": n_empty_disagree,
        "no_value_disagreement_reasons": empty_reasons,
        "judge_invalid_rate": _rate(n_invalid, n),
        "span_unverified_rate": _rate(n_unverified, n),
        "judge_missing_rate": _rate(n_missing, n),
        # The anti-hallucination mechanism, made auditable. Every count here is a
        # record where the judge asserted an answer and the span it quoted did not
        # support it -- so the pipeline threw the answer away rather than trust the
        # transcription. A pipeline that never rejects anything is one whose
        # verification is not doing work.
        "n_span_rejected": n_unverified,
        "span_rejection_reasons": dict(
            sorted(span_reasons.items(), key=lambda kv: -kv[1])
        ),
        "disagreement_examples": disagreements,
    }


def step_extraction_coverage(records, task_type=None):
    """Count how often the judge located each prescribed reasoning step.

    This is EXTRACTION coverage -- a property of the judge, not a score of the
    model. Job B output is persisted for later use, not scored here.

    Args:
        records: Iterable of ``llm-parsed*`` records, some with ``LLM_judge_steps``.

    Returns:
        dict | None: Per-step present counts, or ``None`` for tasks without steps.
    """
    per_step = {}
    n_eligible = n_all = n_any = 0
    for rec in records:
        steps = rec.get("LLM_judge_steps")
        if not steps:
            continue
        n_eligible += 1
        # The step index alone does NOT identify the step on AD: STEP_SPECS splits
        # it into AD:distance and AD:angle because the two prompts prescribe
        # different step contents, so index 1 is "landmark 1" (2 values) for
        # distance but "the endpoints of LINE 1" (4 values) for angle. Both metric
        # types live in one model directory, so keying on the index alone averaged
        # two unrelated extractions of different difficulty into one rate.
        #
        # Gated on task_type, NOT on the presence of metric_type. TL records also
        # carry a biometric_profile.metric_type -- as a 1-element LIST
        # (["distance"]) where AD stores a plain string -- so a presence test
        # relabelled every TL step "distance:N" while doing nothing for AD, the
        # exact inverse of the intent. TL prescribes one step schema, so its index
        # already identifies the step.
        metric = None
        if task_type == "AD":
            metric = ((rec.get("doc") or {}).get("biometric_profile") or {}).get("metric_type")
            if isinstance(metric, list):
                metric = metric[0] if metric else None
        present = 0
        for s in steps:
            idx = (f"{metric}:{s.get('index')}" if metric else s.get("index"))
            slot = per_step.setdefault(idx, {"present": 0, "total": 0})
            slot["total"] += 1
            if s.get("status") == "present":
                slot["present"] += 1
                present += 1
        n_all += present == len(steps)
        n_any += present > 0
    if not n_eligible:
        return None
    return {
        "n_eligible": n_eligible,
        "all_steps_extracted_rate": _rate(n_all, n_eligible),
        "any_step_extracted_rate": _rate(n_any, n_eligible),
        # Keys are ints for TL and "<metric_type>:<index>" strings for AD, so sort
        # on the string form to keep mixed keys orderable.
        "per_step_rate": {
            k: _rate(v["present"], v["total"])
            for k, v in sorted(per_step.items(), key=lambda kv: str(kv[0]))
        },
    }


def length_stratification(records, n_bins=4):
    """Compare response lengths of judge-recovered vs originally-parsed samples.

    Threat to validity this addresses: strict failures are systematically LONGER
    (AD 4,502 vs 2,440 chars overall). If recovered samples are disproportionately
    long-CoT, and long CoT correlates with harder cases, then format-robust error
    is computed over a harder subset -- a new survivorship story pointing the
    other way. Reporting the two length distributions lets a reader check.

    Args:
        records: Iterable of ``llm-parsed*`` records.
        n_bins: Number of quantile bins to report.

    Returns:
        dict: Mean/median length for each group, and quantile cut points.
    """
    import numpy as np

    def _resp_len(rec):
        try:
            r = rec["resps"][0][0]
            if isinstance(r, list):
                r = r[0] if r else ""
            return len(r or "")
        except Exception:
            return 0

    parsed, recovered = [], []
    for rec in records:
        mode = rec.get("LLM_judge_answer_mode")
        if mode == "conclusion_in_format":
            parsed.append(_resp_len(rec))
        elif mode == "conclusion_off_format":
            recovered.append(_resp_len(rec))

    def _stat(xs):
        if not xs:
            return {"n": 0, "mean": float("nan"), "median": float("nan")}
        a = np.asarray(xs, dtype=float)
        return {"n": len(xs), "mean": float(a.mean()), "median": float(np.median(a))}

    out = {"originally_parsed": _stat(parsed), "judge_recovered": _stat(recovered)}
    if parsed:
        qs = np.quantile(np.asarray(parsed, dtype=float),
                         [i / n_bins for i in range(1, n_bins)])
        out["parsed_quantiles"] = [float(q) for q in qs]
    return out
