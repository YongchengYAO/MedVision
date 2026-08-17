"""Stage 2 -- verify judge spans, merge, and write the llm-parsed records.

This is a translation-and-scoring step. It invents no metric: scoring goes through
``medvision_bm.utils.parse_utils.cal_metrics`` -- the same function that produced the
published numbers -- so the strict and judge columns differ only in which values
were scored.

Output: ``<model_dir>/llm-parsed[-limitN]/<same filename>.jsonl``. Each record is
its ``parsed/`` source with:

  - ``filtered_resps``  REMOVED, replaced in place by ``LLM_filtered_resps``
  - every derived metric block recomputed from that value, or dropped
  - ``LLM_judge_answer_mode``  which of four categories this sample fell into
  - ``LLM_judge_SR``           ``{"success": bool}``, the repo's per-record idiom
  - ``LLM_judge``              reason, both predictions, and the quoted span
  - ``LLM_judge_steps``        Job B, TL and AD only

The strict key is removed rather than kept alongside, so a consumer cannot
silently score the wrong one. The summarizers therefore need
``--resps_key LLM_filtered_resps``; forgetting it is a hard error, not an empty
report (see ``parse_utils.assert_resps_key``).

Fail-safe direction: whenever the strict parser succeeded its value is kept
verbatim, whatever the judge said. The judge can only ADD recoveries -- see
``judge_decision`` for why that makes a judge failure cost a recovery rather than
corrupt a published number.

Usage:
    python apply_judge.py --task_type TL \\
        --task_dir Results/MedVision-TL-v2-CoT \\
        --judge_out Results/MedVision-TL-v2-CoT/judge-out_TL_limit100.jsonl \\
        --limit 100 -p 32
"""

import argparse
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)
if os.path.isdir(_REPO_SRC):
    sys.path.insert(0, _REPO_SRC)

import numpy as np  # noqa: E402

from judge_config import (  # noqa: E402
    ANSWER_MODES,
    DEFAULT_ROSTER_YAML,
    DEFAULT_TASK_DIR,
    EXCLUDED_JSONL_STEMS,
    JUDGE_MODELS,
    SUCCESS_MODES,
    RESPS_KEY_LLM,
    RESPS_KEY_STRICT,
    STEP_SPECS,
    TASK_SPECS,
    llm_parsed_dirname,
    step_spec_key,
)
from judge_decision import decide_answer, is_success  # noqa: E402
from judge_prompts import short_prompt_fp  # noqa: E402
from judge_io import (  # noqa: E402
    extract_response,
    iter_records,
    list_sample_files,
    load_roster,
)
from judge_verify import verify_span  # noqa: E402

# Scoring MUST come from the benchmark's own implementation. Re-deriving MAE/IoU
# here would let the judge column drift from the published column for reasons that
# have nothing to do with parsing.
from medvision_bm.utils.parse_utils import cal_metrics, convert_numpy_to_python  # noqa: E402

# nMAE needs the physical image diagonal. The canonical helper lives in the
# lmms_eval tree and drags in heavy deps (loguru et al.) that are present in the
# real parse env but not in a bare interpreter. Three tiers, best first.
try:
    from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
        _compute_physical_diagonal,
    )

    _HAVE_DIAGONAL = True
except Exception:  # pragma: no cover - env-dependent
    _compute_physical_diagonal = None
    _HAVE_DIAGONAL = False


def _diagonal_from_strict(rec):
    """Recover the image's physical diagonal from the strict record, if possible.

    ``nMAE = MAE / diagonal``, and the diagonal is a property of the image, not of
    the prediction. So when the strict parse produced both numbers we can invert
    them and reuse the diagonal for the judge's MAE -- no NIfTI read, no heavy
    import. Returns ``None`` when the strict parse failed (exactly the records the
    judge recovers), which is why this is only the second tier.
    """
    try:
        mae = float(rec.get("avgMAE", {}).get("MAE"))
        nmae = float(rec.get("nMAE", {}).get("NMAE"))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(mae) or not np.isfinite(nmae) or nmae == 0:
        return None
    return mae / nmae


def _compute_nmae(rec_out, strict_rec, task_type, mae, jsonl_file):
    """Return the ``nMAE`` block for a judge-scored record, or ``None`` to omit it.

    Omitting is safe: ``summarize_TL_task``/``summarize_AD_task`` compute nMAE
    themselves when the key is absent. Writing it here just saves them the work.
    """
    if task_type not in ("TL", "AD") or mae is None or not np.isfinite(mae):
        return {"NMAE": float("nan"), "success": False}

    doc = rec_out.get("doc") or {}
    metric_type = (
        (doc.get("biometric_profile") or {}).get("metric_type", "")
        if task_type == "AD"
        else "distance"
    )
    if metric_type != "distance":
        return {"NMAE": float("nan"), "success": False}

    # Tier 1: the canonical helper.
    if _HAVE_DIAGONAL:
        is_scaled_ps = "scaledPS" in os.path.basename(jsonl_file)
        scale_mode = (
            ("anisotropic" if task_type == "AD" else "uniform") if is_scaled_ps else None
        )
        doc_meta = {
            "image_file": doc.get("image_file"),
            "slice_dim": doc.get("slice_dim"),
            "slice_idx": doc.get("slice_idx"),
            "taskID": doc.get("taskID"),
            "label": doc.get("label"),
            "image_size_2d": doc.get("image_size_2d"),
        }
        try:
            diagonal = _compute_physical_diagonal(
                doc_meta,
                scale_mode=scale_mode,
                explicit_scale=rec_out.get("pixel_size_scale"),
            )
            return {"NMAE": float(mae) / diagonal, "success": True}
        except Exception:
            pass

    # Tier 2: invert the strict record's own MAE/nMAE pair.
    diagonal = _diagonal_from_strict(strict_rec)
    if diagonal:
        return {"NMAE": float(mae) / diagonal, "success": True}

    # Tier 3: let the summarizer do it.
    return None


def _apply_steps(response, judge_steps, step_key):
    """Verify and transcribe Job B step extractions.

    Returns:
        tuple: ``(steps_out, n_present)``. ``steps_out`` has one entry per
        prescribed step, always in index order, so a consumer can rely on shape.
    """
    if step_key is None:
        return None, 0
    specs = STEP_SPECS[step_key]
    by_index = {}
    for s in judge_steps or []:
        # isinstance first: a step entry is not guaranteed to be an object. The v1
        # sweep produced bare strings here, and ``.get`` on one raises -- which is
        # a crash mid-sweep rather than one skipped step.
        if not isinstance(s, dict):
            continue
        try:
            by_index[int(s.get("index"))] = s
        except (TypeError, ValueError):
            continue

    out = []
    n_present = 0
    for spec in specs:
        j = by_index.get(spec["index"])
        if not j or j.get("status") != "present":
            out.append({"index": spec["index"], "status": "absent", "reason": "judge_absent",
                        "values": []})
            continue
        v = verify_span(response, j.get("span"), j.get("values"),
                        expected_arity=spec["n_values"])
        if v["ok"]:
            n_present += 1
            out.append({"index": spec["index"], "status": "present", "reason": "ok",
                        "tier": v.get("tier", "exact"),
                        "values": [float(x) for x in v["numbers"]], "span": j.get("span")})
        else:
            out.append({"index": spec["index"], "status": "absent", "reason": v["reason"],
                        "values": []})
    return out, n_present


def _process_file(jsonl_file, task_type, limit, judge_by_doc, out_dir):
    """Apply judge results to one sample file and write the llm-parsed record.

    Args:
        jsonl_file: Source ``parsed/*.jsonl`` path.
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        limit: Max records per file, or ``None``.
        judge_by_doc: ``{doc_id: judge_record}`` for this file.
        out_dir: Destination ``llm-parsed[-limitN]`` directory.

    Returns:
        dict: Per-file counters.
    """
    arity = TASK_SPECS[task_type]["arity"]
    out_path = os.path.join(out_dir, os.path.basename(jsonl_file))
    stats = defaultdict(int)
    modes = defaultdict(int)
    reasons = defaultdict(int)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as out:
        for rec in iter_records(jsonl_file, limit=limit):
            stats["n_total"] += 1
            response = extract_response(rec) or ""
            strict_stored = rec.get(RESPS_KEY_STRICT) or [""]
            strict_pred = str(strict_stored[0]) if strict_stored else ""
            stats["n_strict_ok"] += 1 if strict_pred.strip() else 0

            j = judge_by_doc.get(rec.get("doc_id"))
            step_key = step_spec_key(
                task_type, (rec.get("doc") or {}).get("biometric_profile", {}).get("metric_type")
            )

            # Verification is attempted only where it can mean anything: the judge
            # claimed an answer and is asked to point at it.
            verified = None
            if j is not None and j.get("judge_status") == "ok":
                # isinstance, not `or {}`: the v1 sweep returned bare strings,
                # floats and lists here, and every one of those is truthy, so
                # `or {}` would not fire and `.get` would raise mid-sweep.
                fa = j.get("final_answer")
                fa = fa if isinstance(fa, dict) else {}
                if fa.get("status") == "present":
                    verified = verify_span(
                        response, fa.get("span"), fa.get("values"), expected_arity=arity
                    )

            pred, mode, reason = decide_answer(strict_pred, j, verified)
            success = is_success(mode)

            modes[mode] += 1
            reasons[reason] += 1
            stats["n_judge_ok"] += 1 if success else 0
            if not strict_pred.strip() and success:
                stats["n_recovered"] += 1

            steps_out, n_steps_present = _apply_steps(response, j.get("steps") if j else None,
                                                      step_key)
            if step_key is not None:
                stats["n_step_eligible"] += 1
                stats["n_step_all_present"] += 1 if n_steps_present == len(STEP_SPECS[step_key]) else 0
                stats["n_step_any_present"] += 1 if n_steps_present else 0

            # --- rewrite the record ---
            # The strict key is REMOVED and the LLM key takes its slot, preserving
            # the source key order. Keeping both would let a reader score the wrong
            # one silently; renaming in place keeps a diff against parsed/ readable.
            rec = _rename_resps_key(rec, pred)

            # Every stored metric is derived from the prediction, so every stored
            # metric is recomputed -- or dropped. Carrying a strict-parser number
            # into a record whose other metrics came from the judge is the kind of
            # mixed provenance nobody catches by eye.
            # Snapshot the STRICT metric blocks before they are overwritten below.
            # _compute_nmae tier 2 recovers the image diagonal by inverting the
            # strict MAE/nMAE pair, so it needs the record as the strict pipeline
            # left it. Passing the post-overwrite record made tier 2 an algebraic
            # no-op -- diagonal = mae_judge/nmae_stale, hence
            # mae_judge/diagonal == nmae_stale -- silently copying the stale
            # eval-time nMAE onto a judge-recovered row.
            strict_metrics = {"avgMAE": rec.get("avgMAE") or {}, "nMAE": rec.get("nMAE") or {}}

            m = cal_metrics({RESPS_KEY_STRICT: [pred], "target": rec["target"]}, task_type)
            rec["avgMAE"] = m["avgMAE"]
            rec["SuccessRate"] = m["SuccessRate"]
            if task_type == "Detection":
                rec["avgIoU"] = m["avgIoU"]
                rec["F1"] = m["F1"]
                rec["Precision"] = m["Precision"]
                rec["Recall"] = m["Recall"]
                # cal_metrics_detection_task never produces an MRE; a stale one
                # inherited from parsed/ would be the only un-recomputed metric here.
                rec.pop("avgMRE", None)
            else:
                rec["avgMRE"] = m["avgMRE"]
                nmae = _compute_nmae(rec, strict_metrics, task_type, m["avgMAE"]["MAE"],
                                     jsonl_file)
                if nmae is None:
                    rec.pop("nMAE", None)
                else:
                    rec["nMAE"] = nmae
                # AD records carry legacy top-level MAE/MRE blocks from an older
                # lmms-eval schema. They are not recomputed anywhere, so they must go.
                rec.pop("MAE", None)
                rec.pop("MRE", None)

            rec["LLM_judge_answer_mode"] = mode
            rec["LLM_judge_SR"] = {"success": success}
            rec["LLM_judge"] = {
                "reason": reason,
                "strict_pred": strict_pred,
                # The judge's OWN span-verified extraction, independent of what
                # was decided. It must NOT be the decided value: the strict-first
                # rule makes the decided value equal strict_pred on every
                # strict-success row, so comparing that to strict_pred would report
                # 100% judge-regex agreement by construction and measure nothing.
                # The decided value is LLM_filtered_resps; this is the evidence.
                "judge_pred": verified["pred"] if (verified and verified["ok"]) else "",
                "judge_span": _judge_span(j),
                # Provenance travels with the record. Without it a --mock sweep was
                # invisible downstream: decide_answer returns reason "ok" for a mock
                # row, so nothing in llm-parsed recorded that a regex stand-in --
                # not the judge -- produced the answer, and test-8's documented
                # mock check had nothing to test.
                "judge_model": (j or {}).get("judge_model"),
            }
            # Which tier of span verification accepted the extraction. "exact" is
            # the original strict contract; "value_anchor" means the values were
            # located in the response but the quoted span was not -- the audit
            # trail for exactly how much trust each recovery leans on.
            if verified and verified["ok"]:
                rec["LLM_judge"]["verify_tier"] = verified.get("tier", "exact")
            if steps_out is not None:
                rec["LLM_judge_steps"] = steps_out

            out.write(json.dumps(rec, default=convert_numpy_to_python) + "\n")
    os.replace(tmp, out_path)

    result = dict(stats)
    result["modes"] = dict(modes)
    result["reasons"] = dict(reasons)
    result["file"] = os.path.basename(jsonl_file)
    return result


def _rename_resps_key(rec, pred):
    """Return ``rec`` with ``filtered_resps`` replaced by ``LLM_filtered_resps``.

    Rebuilt rather than mutated so the LLM key lands in the *slot* the strict key
    occupied. A diff of ``llm-parsed/`` against ``parsed/`` then shows one renamed
    line plus the recomputed metrics, instead of a whole-record reordering that
    hides what actually changed.

    Args:
        rec: The source ``parsed/`` record.
        pred: The resolved prediction string (``""`` when none).

    Returns:
        dict: A new record. ``filtered_resps`` is absent; ``LLM_filtered_resps``
        holds ``[pred]``.
    """
    out = {}
    for k, v in rec.items():
        if k == RESPS_KEY_STRICT:
            out[RESPS_KEY_LLM] = [pred]
        else:
            out[k] = v
    # A source record without the strict key should be impossible -- parse_outputs
    # writes it on every record -- but appending is better than dropping the answer.
    if RESPS_KEY_LLM not in out:
        out[RESPS_KEY_LLM] = [pred]
    return out


def _judge_span(judge_row):
    """Return the span the judge quoted, or ``""`` when it quoted nothing.

    Persisted so a reviewer can see WHERE a recovered number came from without
    re-reading the response. This is the audit trail for every
    ``conclusion_off_format`` row.
    """
    if not judge_row:
        return ""
    fa = judge_row.get("final_answer")
    if not isinstance(fa, dict):
        return ""
    span = fa.get("span")
    return span if isinstance(span, str) else ""


# Set once per worker by ``_init_worker`` and read by ``_process_file_star``.
# NOT bound into the mapped callable: Pool pickles ``func`` with every TASK CHUNK
# it queues, not once per worker, so binding the index meant serialising a 222 MB
# blob 126 times for Detection (~761 s of parent-side pickling) -- which made
# `-p 32` SLOWER than `-p 1`. An initializer sends it once per process instead.
_JUDGE_INDEX = {}


def _init_worker(judge_index):
    """Pool initializer: publish the judge index into this worker once."""
    global _JUDGE_INDEX
    _JUDGE_INDEX = judge_index


def _process_file_star(job, task_type, limit, judge_index=None):
    """multiprocessing.Pool adapter for ``_process_file``.

    ``judge_index`` defaults to the per-worker global so the serial path (which
    passes it explicitly) and the pooled path share one function.
    """
    jsonl_file, model, out_dir = job
    index = _JUDGE_INDEX if judge_index is None else judge_index
    key = (model, os.path.basename(jsonl_file))
    os.makedirs(out_dir, exist_ok=True)
    return _process_file(jsonl_file, task_type, limit, index.get(key, {}), out_dir)


def assert_judge_out_prompt(judge_out_path, task_type, accept_fps=()):
    """Refuse a judge-output file produced under a different prompt.

    Stage 1 has this guard on its own resume path, but Stage 2 is a separate entry
    point that can be pointed at any file -- and pointing it at last week's output
    is exactly the mistake worth catching. Without this, a v1 file is consumed
    happily: its ``final_answer`` is a bare string, every row degrades to
    ``undetermined``, and the run reports a catastrophic judge failure that is
    really just the wrong input file.

    Args:
        judge_out_path: Stage 1 output JSONL.
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.

    Raises:
        SystemExit: If any row's ``prompt_fp`` is absent or does not match a
            fingerprint the current prompt produces for this task.
    """
    expected = {
        short_prompt_fp(task_type, k)
        for k in {"TL": ["TL"], "AD": ["AD:distance", "AD:angle", None],
                  "Detection": [None]}[task_type]
    }
    # A repair pass that raised --max_tokens leaves two stamps in one file: the
    # rows that parsed under the old budget, and the re-judged ones. Both are
    # answers to the same question, so the caller can whitelist the old stamp --
    # explicitly, so the mixing is a decision rather than an accident.
    expected |= set(accept_fps)
    seen = set()
    with open(judge_out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line).get("prompt_fp"))
            except json.JSONDecodeError:
                continue
            if len(seen) > 4:  # enough to characterise the file
                break
    stale = seen - expected
    if stale:
        raise SystemExit(
            f"\n[stage2] ABORT {judge_out_path} was written under a different "
            f"prompt.\n"
            f"  rows carry : {sorted(str(s) for s in stale)}\n"
            f"  current    : {sorted(expected)}\n\n"
            f"  Its answers do not correspond to the prompt this code renders, so\n"
            f"  merging them would attribute one prompt's extractions to another.\n"
            f"  Move it aside and re-run the judge:\n"
            f"      mv {judge_out_path} {judge_out_path}.stale\n"
        )


def assert_single_judge_model(judge_out_path):
    """Refuse a judge-output file written by more than one reader.

    ``load_judge_index`` keys rows by ``(model, file) -> doc_id`` where ``model``
    is the BENCHMARKED VLM, not the judge, and takes last-wins. Two readers' rows
    in one file therefore do not collide loudly -- they interleave, and each
    doc_id silently takes whichever reader happened to write it last. The report
    that comes out is a blend of two instruments, is internally consistent, and
    says nothing about it anywhere. This is the only check that would notice.

    The driver cannot produce such a file: ``judge_out_filename`` gives each
    reader its own name. This guards the paths that bypass the driver -- a
    hand-run repair pass with ``--out`` pointed at the previous reader's file, or
    a ``cat`` of two sweeps.

    Compared on the normalised key ``run_judge_vllm._model_key`` uses, so a hub id
    and a local mirror of the same weights still count as one reader.

    Args:
        judge_out_path: Stage 1 output JSONL.

    Raises:
        SystemExit: If rows carry more than one distinct reader.
    """
    from run_judge_vllm import _model_key

    seen = {}
    with open(judge_out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line).get("judge_model")
            except json.JSONDecodeError:
                continue
            seen.setdefault(_model_key(m), m)
            if len(seen) > 1:
                break
    if len(seen) > 1:
        raise SystemExit(
            f"\n[stage2] ABORT {judge_out_path} holds rows from more than one judge.\n"
            f"  readers : {sorted(str(v) for v in seen.values())}\n\n"
            f"  Records are indexed by the BENCHMARKED model, not the judge, and the\n"
            f"  last row for a doc_id wins -- so these would be merged into a single\n"
            f"  report with no indication that two readers produced it.\n"
            f"  Split the file by judge_model, or re-run each reader to its own --out\n"
            f"  (the driver names them apart automatically).\n"
        )


def load_judge_index(judge_out_path):
    """Load judge output and index it by ``(model, file)`` then ``doc_id``.

    LAST WINS per ``doc_id``, which is what makes a repair pass work: Stage 1
    appends re-judged rows to the end of the same file and the later verdict
    supersedes the earlier one.

    That also makes the merge order load-bearing. Repaired rows must be APPENDED
    to the existing output; ``cat new old > merged`` puts the stale rows last and
    silently reinstates exactly the answers the repair was run to replace, with no
    error anywhere -- the file is well-formed and every qid is present.

    Args:
        judge_out_path: Stage 1 output JSONL.

    Returns:
        dict: ``{(model, file): {doc_id: record}}``.
    """
    index = defaultdict(dict)
    n = 0
    with open(judge_out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index[(row.get("model"), row.get("file"))][row.get("doc_id")] = row
            n += 1
    print(f"[stage2] loaded {n:,} judge records over {len(index):,} files")
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_type", required=True, choices=["TL", "AD", "Detection"])
    p.add_argument("--task_dir", default=None)
    p.add_argument("--config_yaml", default=None)
    p.add_argument("--judge_out", required=True, help="Stage 1 output JSONL")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--accept_prompt_fp", action="append", default=[], metavar="FP",
                   help="Additional prompt stamp to accept in --judge_out. For "
                        "repair passes that re-judged only part of a file at a "
                        "different --max_tokens. Repeatable.")
    p.add_argument("--judge", default=None, choices=sorted(JUDGE_MODELS),
                   help="Which reader produced --judge_out. Selects the output "
                        "directory (llm-parsed<suffix>/) so two readers' records "
                        "never overwrite each other. Default: the reader every "
                        "published artifact came from.")
    p.add_argument("--processes", "-p", type=int, default=None)
    args = p.parse_args()

    task_dir = args.task_dir or DEFAULT_TASK_DIR[args.task_type]
    roster_yaml = args.config_yaml or DEFAULT_ROSTER_YAML[args.task_type]
    roster = load_roster(roster_yaml)
    out_dirname = llm_parsed_dirname(args.limit, args.judge)

    assert_judge_out_prompt(args.judge_out, args.task_type, args.accept_prompt_fp)
    assert_single_judge_model(args.judge_out)
    judge_index = load_judge_index(args.judge_out)

    jobs = []
    for model in roster:
        model_dir = os.path.join(task_dir, model)
        if not os.path.isdir(model_dir):
            continue
        out_dir = os.path.join(model_dir, out_dirname)
        for f in list_sample_files(model_dir, "parsed", EXCLUDED_JSONL_STEMS):
            jobs.append((f, model, out_dir))

    print(f"[stage2] {len(jobs)} files -> */{out_dirname}/")
    if not _HAVE_DIAGONAL:
        print("[stage2] note: physical-diagonal helper unavailable; nMAE falls back "
              "to strict-record inversion, then to the summarizer")

    if args.processes and args.processes > 1 and len(jobs) > 1:
        # The index goes through the INITIALIZER (once per worker), never through
        # the mapped callable (once per task chunk). chunksize=1 keeps the queued
        # task tiny -- it is now just a (path, model, out_dir) tuple.
        pooled = partial(_process_file_star, task_type=args.task_type, limit=args.limit)
        with multiprocessing.Pool(args.processes, initializer=_init_worker,
                                  initargs=(judge_index,)) as pool:
            results = pool.map(pooled, jobs, chunksize=1)
    else:
        worker = partial(_process_file_star, task_type=args.task_type,
                         limit=args.limit, judge_index=judge_index)
        results = [worker(j) for j in jobs]

    agg = defaultdict(int)
    modes = defaultdict(int)
    reasons = defaultdict(int)
    for r in results:
        for k, v in r.items():
            if k in ("modes", "reasons", "file"):
                continue
            agg[k] += v
        for k, v in r["modes"].items():
            modes[k] += v
        for k, v in r["reasons"].items():
            reasons[k] += v

    total = agg["n_total"]
    print(f"\n{'records':<28}{total:>12,}")
    print(f"{'strict parsed':<28}{agg['n_strict_ok']:>12,}  "
          f"({agg['n_strict_ok']/total*100 if total else 0:.1f}%)")
    print(f"{'judge parsed':<28}{agg['n_judge_ok']:>12,}  "
          f"({agg['n_judge_ok']/total*100 if total else 0:.1f}%)")
    print(f"{'recovered by judge':<28}{agg['n_recovered']:>12,}")
    if agg.get("n_step_eligible"):
        e = agg["n_step_eligible"]
        print(f"{'steps: all present':<28}{agg['n_step_all_present']:>12,}  "
              f"({agg['n_step_all_present']/e*100:.1f}%)")
        print(f"{'steps: any present':<28}{agg['n_step_any_present']:>12,}  "
              f"({agg['n_step_any_present']/e*100:.1f}%)")
    print("\nLLM_judge_answer_mode:")
    for mode in ANSWER_MODES:
        n = modes.get(mode, 0)
        flag = "OK  " if mode in SUCCESS_MODES else "fail"
        print(f"  {flag} {mode:<26}{n:>12,}  "
              f"({n/total*100 if total else 0:5.1f}%)")
    print("\nreasons:")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26}{v:>12,}")


if __name__ == "__main__":
    main()
