"""Stage 4 -- failure decomposition and judge-validity report.

Reads the judge-parsed records written by Stage 2 and reports the two things the
existing summarizers cannot: WHY the strict parser failed, and whether the judge
is trustworthy.

Format-robust MAE/MRE/SR/IoU are NOT computed here. Those come from running the
existing summarizers against the same directory (Stage 3), so the judge column and
the published column share one code path.

Outputs:
  <model_dir>/<parsed_dirname>/summary_metrics_judge_Task[_limitN].json
  <task_dir>/summary_judge_task[_limitN]__<parsed_dirname>.txt

Usage:
    python summarize_judge_task.py --task_type TL \\
        --task_dir Results/MedVision-TL-v2-CoT --limit 100
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_config import (  # noqa: E402
    ANSWER_MODES,
    DEFAULT_ROSTER_YAML,
    DEFAULT_TASK_DIR,
    EXCLUDED_JSONL_STEMS,
    JUDGE_MODELS,
    SUMMARY_FILENAME_JUDGE_METRICS,
    limit_suffix,
    llm_parsed_dirname,
)
from judge_io import iter_records, list_sample_files, load_roster  # noqa: E402
from judge_stats import (  # noqa: E402
    decompose_failures,
    judge_validity,
    length_stratification,
    step_extraction_coverage,
)


def _model_records(model_dir, parsed_dirname, limit):
    """Yield every judge-parsed record for one model."""
    for f in list_sample_files(model_dir, parsed_dirname, EXCLUDED_JSONL_STEMS):
        for rec in iter_records(f, limit=limit):
            yield rec


def summarize_model(model_dir, parsed_dirname, limit, task_type=None):
    """Compute all Stage 4 blocks for one model directory.

    The records are materialized once and shared by all four statistics. Streaming
    them separately per statistic read every file four times, which on a full
    Detection sweep is 4x the disk for no benefit -- one model's share of the corpus
    is ~23K records and is further bounded by ``--limit``.
    """
    records = list(_model_records(model_dir, parsed_dirname, limit))
    failures = decompose_failures(records)
    if not failures["n_samples"]:
        return None
    validity = judge_validity(records)
    # task_type gates the AD-only step split; see judge_stats.
    steps = step_extraction_coverage(records, task_type)
    lengths = length_stratification(records)
    return {
        "failure_decomposition": failures,
        "judge_validity": validity,
        "step_extraction_coverage": steps,
        "response_length": lengths,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_type", required=True, choices=["TL", "AD", "Detection"])
    p.add_argument("--task_dir", default=None)
    p.add_argument("--config_yaml", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge", default=None, choices=sorted(JUDGE_MODELS),
                   help="Which reader's records to report on. Default: the reader "
                        "every published artifact came from.")
    p.add_argument("--parsed_dirname", default=None,
                   help="Default: llm-parsed[<judge>][-limitN] derived from "
                        "--judge and --limit")
    args = p.parse_args()

    task_dir = args.task_dir or DEFAULT_TASK_DIR[args.task_type]
    roster_yaml = args.config_yaml or DEFAULT_ROSTER_YAML[args.task_type]
    parsed_dirname = args.parsed_dirname or llm_parsed_dirname(args.limit, args.judge)
    roster = load_roster(roster_yaml)

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"========== JUDGE REPORT: {args.task_type} ==========")
    emit(f"task_dir       : {task_dir}")
    emit(f"parsed_dirname : {parsed_dirname}")
    emit(f"limit          : {args.limit}")
    emit()
    emit("FAILURE DECOMPOSITION -- what the strict parser's failures actually were")
    emit(f"{'model':<44}{'SR strict':>10}{'SR judge':>10}{'dSR':>8}"
         f"{'fmt-fail':>10}{'non-answer':>12}")
    emit("-" * 94)

    all_metrics = {}
    tot = {"n": 0, "s": 0, "j": 0, "rec": 0, "na": 0}
    for model in roster:
        model_dir = os.path.join(task_dir, model)
        if not os.path.isdir(model_dir):
            continue
        res = summarize_model(model_dir, parsed_dirname, args.limit, args.task_type)
        if res is None:
            continue
        all_metrics[model] = res
        f = res["failure_decomposition"]
        tot["n"] += f["n_samples"]
        tot["s"] += f["n_strict_parsed"]
        tot["j"] += f["n_judge_parsed"]
        tot["rec"] += f["n_format_failure"]
        # Accumulated, not re-derived. The TOTAL used to print the residual
        # n - strict - recovered, which is no_conclusion + undetermined -- so once
        # the per-model column became no_conclusion alone, the TOTAL stopped being
        # the sum of its own column and silently re-added the judge's failures to
        # a model-attributed number (AD 3,689 vs 3,516; TL 4,533 vs 3,840).
        tot["na"] += f["n_genuine_non_answer"]
        emit(f"{model[:43]:<44}"
             f"{f['success_rate_strict']*100:>9.1f}%"
             f"{f['success_rate_judge']*100:>9.1f}%"
             f"{f['delta_success_rate']*100:>+7.1f}"
             f"{f['n_format_failure']:>10,}"
             f"{f['n_genuine_non_answer']:>12,}")

        out_dir = os.path.join(model_dir, parsed_dirname)
        os.makedirs(out_dir, exist_ok=True)
        fn = SUMMARY_FILENAME_JUDGE_METRICS.removesuffix(".json") + limit_suffix(args.limit) + ".json"
        with open(os.path.join(out_dir, fn), "w") as fh:
            json.dump(res, fh, indent=2)

    if tot["n"]:
        emit("-" * 94)
        emit(f"{'TOTAL':<44}{tot['s']/tot['n']*100:>9.1f}%{tot['j']/tot['n']*100:>9.1f}%"
             f"{(tot['j']-tot['s'])/tot['n']*100:>+7.1f}{tot['rec']:>10,}"
             f"{tot['na']:>12,}")

    emit()
    emit("ANSWER MODE -- the four categories LLM_judge_answer_mode assigns")
    emit("(the two conclusion_* modes count as a successful answer; the rest do not.")
    emit(" no-concl does not distinguish 'ran out of room' from 'declined' -- that")
    emit(" needs the generation config, so the judge is not asked to guess it)")
    emit(f"{'model':<44}{'in-format':>11}{'off-format':>12}"
         f"{'no-concl':>10}{'undet':>8}")
    emit("-" * 85)
    g_mode = {m: 0 for m in ANSWER_MODES}
    for model, res in all_metrics.items():
        bm = res["failure_decomposition"]["by_mode"]
        for m in ANSWER_MODES:
            g_mode[m] += bm.get(m, 0)
        emit(f"{model[:43]:<44}"
             f"{bm['conclusion_in_format']:>11,}{bm['conclusion_off_format']:>12,}"
             f"{bm['no_conclusion']:>10,}{bm['undetermined']:>8,}")
    if tot["n"]:
        emit("-" * 85)
        emit(f"{'TOTAL':<44}"
             f"{g_mode['conclusion_in_format']:>11,}{g_mode['conclusion_off_format']:>12,}"
             f"{g_mode['no_conclusion']:>10,}{g_mode['undetermined']:>8,}")

    emit()
    emit("JUDGE VALIDITY -- agreement with the strict parser on ITS successes")
    emit("(the strict parser is near-perfect on the rows it accepts, so this is a")
    emit(" real reliability check; it is free only because we judge 100% of rows)")
    # The last two columns are over ALL records, not over n_ref. Printed
    # unlabelled beside a rate that IS over n_ref, they read as sharing its
    # denominator -- and on TL the two differ by 1.30x (43,938 vs 33,854), so a
    # reader cannot decompose 1 - agreement into disagreement vs unavailability.
    emit("(agree% is over n_ref; invalid% and unverif% are over ALL records)")
    emit(f"{'model':<44}{'n_ref':>9}{'agree':>9}{'agree%':>9}"
         f"{'invalid%':>10}{'unverif%':>10}")
    emit("-" * 89)
    g_ref = g_agree = 0
    for model, res in all_metrics.items():
        v = res["judge_validity"]
        g_ref += v["n_reference"]
        g_agree += v["n_agree"]
        emit(f"{model[:43]:<44}{v['n_reference']:>9,}{v['n_agree']:>9,}"
             f"{v['agreement_rate']*100:>8.2f}%"
             f"{v['judge_invalid_rate']*100:>9.2f}%"
             f"{v['span_unverified_rate']*100:>9.2f}%")
    if g_ref:
        emit("-" * 89)
        emit(f"{'TOTAL':<44}{g_ref:>9,}{g_agree:>9,}{g_agree/g_ref*100:>8.3f}%")

    emit()
    emit("SPAN VERIFICATION -- answers the judge asserted but could not point at")
    emit("(the judge quotes; the benchmark's own _NUM_RE re-derives the numbers from")
    emit(" that quote. A mismatch means the answer is DISCARDED rather than trusted,")
    emit(" which is what makes a hallucinated digit structurally impossible)")
    g_rej = 0
    g_reasons = {}
    for model, res in all_metrics.items():
        v = res["judge_validity"]
        g_rej += v.get("n_span_rejected", 0)
        for k, c in (v.get("span_rejection_reasons") or {}).items():
            g_reasons[k] = g_reasons.get(k, 0) + c
    if g_rej:
        emit(f"{'rejection reason':<44}{'count':>10}{'share':>10}")
        emit("-" * 64)
        for k, c in sorted(g_reasons.items(), key=lambda kv: -kv[1]):
            emit(f"{k:<44}{c:>10,}{c/g_rej*100:>9.1f}%")
        emit("-" * 64)
        emit(f"{'TOTAL rejected':<44}{g_rej:>10,}"
             f"{(g_rej/tot['n']*100 if tot['n'] else 0):>9.2f}% of records")
    else:
        emit("  no span was rejected -- verify this is real and not a disabled check")

    # Job B is persisted, not scored: report only how often the judge LOCATED a
    # step, which is a property of the extraction rather than of the model.
    any_steps = any(r["step_extraction_coverage"] for r in all_metrics.values())
    if any_steps:
        emit()
        emit("JOB B EXTRACTION COVERAGE -- how often the judge located each step")
        emit("(extraction diagnostic, NOT a model score; step values are persisted")
        emit(" in LLM_judge_steps for downstream use)")
        emit(f"{'model':<44}{'eligible':>10}{'all steps':>11}{'any step':>10}")
        emit("-" * 75)
        for model, res in all_metrics.items():
            s = res["step_extraction_coverage"]
            if not s:
                continue
            emit(f"{model[:43]:<44}{s['n_eligible']:>10,}"
                 f"{s['all_steps_extracted_rate']*100:>10.1f}%"
                 f"{s['any_step_extracted_rate']*100:>9.1f}%")

    emit()
    emit("RESPONSE LENGTH -- recovered vs originally-parsed (survivorship check)")
    emit("(strict failures are systematically longer; if recovered rows are much")
    emit(" longer, format-robust error is computed over a harder subset)")
    emit(f"{'model':<44}{'parsed mean':>13}{'recovered mean':>16}")
    emit("-" * 73)
    for model, res in all_metrics.items():
        L = res["response_length"]
        a, b = L["originally_parsed"], L["judge_recovered"]
        if not b["n"]:
            continue
        emit(f"{model[:43]:<44}{a['mean']:>13,.0f}{b['mean']:>16,.0f}")

    suffix = limit_suffix(args.limit)
    out_path = os.path.join(task_dir, f"summary_judge_task{suffix}__{parsed_dirname}.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
