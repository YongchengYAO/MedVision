"""Stage 0 -- resolve the model roster, replay the regex baseline, emit the judge queue.

No LLM runs here. This stage exists to make the expensive stage cheap to trust:
it pins exactly which rows the judge will see, records what the strict parser got
on those same rows, and self-checks the record counts against known values before
any GPU time is spent.

Outputs (into ``--out_dir``, default ``--task_dir``):
  judge-queue_{task}{_limitN}.jsonl     one row per response, self-contained
  judge-baseline_{task}{_limitN}.json   per-model strict-parser stats + counts

Usage:
    python build_judge_queue.py --task_type TL \\
        --task_dir  Results/MedVision-TL-v2-CoT \\
        --limit 100 -p 32

The roster defaults to this directory's own ``config-{TL,AD,detect}-CoT.yaml``;
pass ``--config_yaml`` only to judge a roster other than the paper's.
"""

import argparse
import json
import multiprocessing
import os
import sys
import tempfile
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_config import (  # noqa: E402
    DEFAULT_ROSTER_YAML,
    DEFAULT_TASK_DIR,
    EXCLUDED_JSONL_STEMS,
    EXPECTED_ROSTER_COUNTS,
    RESPS_KEY_STRICT,
    TASK_SPECS,
    limit_suffix,
    queue_filename,
    step_spec_key,
)
from judge_io import (  # noqa: E402
    content_hash,
    dataset_from_filename,
    extract_last_k_nums_within_answer_tag,
    extract_response,
    iter_records,
    list_sample_files,
    load_roster,
)
from judge_prompts import (  # noqa: E402
    _windowed_response,
    prompt_fingerprint,
    short_prompt_fp,
)


def _metric_type_of(record, task_type):
    """Return the AD metric type (``"distance"``/``"angle"``) for a record."""
    if task_type != "AD":
        return None
    profile = (record.get("doc") or {}).get("biometric_profile") or {}
    return profile.get("metric_type")


def _process_file(jsonl_file, task_type, limit, model, shard_dir, fingerprints,
                  dry_run=False):
    """Build queue rows and baseline stats for one sample file.

    Args:
        jsonl_file: Path to a ``parsed/*.jsonl`` file.
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        limit: Max records per file, or ``None``.
        model: Model directory name.
        shard_dir: Directory to write this file's queue shard into.
        fingerprints: ``{step_key_or_None: prompt_fingerprint}`` for cache keying.

    Returns:
        dict: Baseline stats for this file.
    """
    arity = TASK_SPECS[task_type]["arity"]
    dataset = dataset_from_filename(jsonl_file)
    base = os.path.basename(jsonl_file)
    shard_path = os.path.join(shard_dir, base)

    n_total = 0
    n_regex_ok = 0
    n_stored_mismatch = 0
    n_empty_response = 0
    n_windowed = 0
    len_sum = 0

    rows = []
    for rec in iter_records(jsonl_file, limit=limit):
        response = extract_response(rec) or ""
        regex_pred = extract_last_k_nums_within_answer_tag(response, arity)

        # Cross-check our replayed baseline against what the published pipeline
        # actually stored. A mismatch means this module has drifted from
        # parse_utils.py and every downstream comparison would be invalid.
        stored = rec.get(RESPS_KEY_STRICT) or [""]
        stored_pred = str(stored[0]) if stored else ""
        if stored_pred.strip() != regex_pred.strip():
            n_stored_mismatch += 1

        step_key = step_spec_key(task_type, _metric_type_of(rec, task_type))
        # Recorded, not acted on. A windowed response is the one case where the
        # judge is shown less than the model wrote, so any later analysis of the
        # non-answers needs to be able to exclude these rows.
        _, was_windowed = _windowed_response(response)

        n_total += 1
        n_regex_ok += 1 if regex_pred else 0
        n_empty_response += 0 if response.strip() else 1
        n_windowed += 1 if was_windowed else 0
        len_sum += len(response)

        fingerprint, fingerprint_stamp = fingerprints.get(step_key, ("", ""))
        rows.append(
            {
                "qid": content_hash(model, base, rec.get("doc_id")),
                "task_type": task_type,
                "model": model,
                "file": base,
                "dataset": dataset,
                "doc_id": rec.get("doc_id"),
                "step_key": step_key,
                "regex_pred": regex_pred,
                "response_chars": len(response),
                "was_windowed": was_windowed,
                "response": response,
                "cache_key": content_hash(response, fingerprint),
                "prompt_fp": fingerprint_stamp,
            }
        )

    # --dry_run means "count and gate only". Writing the shard anyway materialised
    # the whole queue -- every row including the full response text -- inside the
    # task dir before deleting it, which is the disk cost the flag exists to avoid.
    # The counters above are already accumulated, so the gates lose nothing.
    if not dry_run:
        tmp = shard_path + ".tmp"
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        os.replace(tmp, shard_path)

    return {
        "model": model,
        "file": base,
        "dataset": dataset,
        "shard": shard_path,
        "n_total": n_total,
        "n_regex_ok": n_regex_ok,
        "n_regex_fail": n_total - n_regex_ok,
        "n_stored_mismatch": n_stored_mismatch,
        "n_empty_response": n_empty_response,
        "n_windowed": n_windowed,
        "mean_response_chars": (len_sum / n_total) if n_total else 0.0,
    }


def _process_file_star(job, task_type, limit, shard_dir, fingerprints, dry_run=False):
    """multiprocessing.Pool adapter for ``_process_file``.

    Args:
        job: ``(jsonl_file, model)`` tuple.
        task_type, limit, shard_dir, fingerprints: forwarded to ``_process_file``.
    """
    jsonl_file, model = job
    return _process_file(jsonl_file, task_type, limit, model, shard_dir, fingerprints,
                         dry_run=dry_run)


def _fingerprints_for(task_type):
    """Return ``{step_key: (full_fingerprint, short_stamp)}`` for this task.

    ``step_spec_key`` returns ``"TL"`` for TL, ``"AD:distance"``/``"AD:angle"``/
    ``None`` for AD, and ``None`` for Detection. Precomputing them once keeps the
    per-record cache-key computation cheap.

    The full fingerprint feeds ``cache_key`` (so an edited prompt produces new
    keys); the short stamp is written onto the row (so Stage 1 can detect an output
    file produced under a different prompt).
    """
    keys = {
        "TL": ["TL"],
        "AD": ["AD:distance", "AD:angle", None],
        "Detection": [None],
    }[task_type]
    return {
        k: (prompt_fingerprint(task_type, k), short_prompt_fp(task_type, k))
        for k in keys
    }


def build_queue(task_type, task_dir, roster_yaml, limit, processes, out_dir, dry_run):
    """Resolve the roster, replay the baseline, and write the judge queue.

    Args:
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        task_dir: Result directory holding one subdirectory per model.
        roster_yaml: Path to the evaluated-model config YAML.
        limit: Max records per file, or ``None``.
        processes: Worker count for the file-level pool.
        out_dir: Where to write the queue and baseline.
        dry_run: If True, count only; do not write the queue.

    Returns:
        dict: The baseline summary that was written (or would be).
    """
    roster = load_roster(roster_yaml)
    fingerprints = _fingerprints_for(task_type)

    # Roster gating is absolute: exactly the models named in the YAML, each read
    # from its own parsed/ directory.
    #
    # There is deliberately NO fallback to globbing <model>/*.jsonl when parsed/ is
    # empty. Stage 0 used to have one and Stage 2 never did, so a model discovered
    # via the fallback got judged and then silently produced no output -- 100% of
    # its GPU time wasted with nothing to show it. A roster model without parsed/
    # is a setup error and is now fatal here, where it costs seconds.
    jobs = []
    missing = []
    for model in roster:
        model_dir = os.path.join(task_dir, model)
        if not os.path.isdir(model_dir):
            missing.append((model, "no such directory"))
            continue
        files = list_sample_files(model_dir, "parsed", EXCLUDED_JSONL_STEMS)
        if not files:
            missing.append((model, "no sample files in parsed/"))
            continue
        for f in files:
            jobs.append((f, model))

    # Paths are resolved BEFORE the first gate so that every failure path can
    # retract, not just the two that happen to run after the queue is written.
    suffix = limit_suffix(limit)
    queue_path = os.path.join(out_dir, queue_filename(task_type, limit))
    baseline_path = os.path.join(out_dir, f"judge-baseline_{task_type}{suffix}.json")

    def _retract(why):
        """Delete anything Stage 1 could consume, and say so.

        The invariant is "a failed Stage 0 leaves nothing consumable". The roster
        gate used to return before the queue was even named, so a queue left by an
        EARLIER successful Stage 0 survived a failed one byte-identical -- and
        Stage 1 would judge it without complaint.
        """
        if dry_run:
            return
        for p in (queue_path, baseline_path):
            try:
                os.remove(p)
                print(f"[gate] removed {p} -- {why}")
            except FileNotFoundError:
                pass

    print(f"[stage0] {task_type}: {len(roster)} in roster, "
          f"{len(roster) - len(missing)} resolved, {len(jobs)} files")
    if missing:
        print(f"\n[GATE FAIL] {len(missing)} roster model(s) could not be resolved "
              f"under {task_dir}:")
        for model, why in missing:
            print(f"    {model}  --  {why}")
        print("\nEvery model named in the roster YAML must have its own parsed/ "
              "directory. Fix the roster or the results tree; do not proceed.")
        _retract("the roster gate failed; a stale queue must not survive it")
        return None, False

    n_written = 0

    # Shards live in a TemporaryDirectory so an interrupt cannot leave a
    # .judge-shards_*/ behind. One such orphan already exists on disk from a killed
    # run, and because it sits inside the task dir it looks like a model directory
    # to any naive scan.
    with tempfile.TemporaryDirectory(
        prefix=f".judge-shards_{task_type}{suffix}_", dir=out_dir
    ) as shard_dir:
        worker = partial(
            _process_file_star,
            task_type=task_type,
            limit=limit,
            shard_dir=shard_dir,
            fingerprints=fingerprints,
            dry_run=dry_run,
        )

        if processes and processes > 1 and len(jobs) > 1:
            with multiprocessing.Pool(processes) as pool:
                stats = pool.map(worker, jobs)
        else:
            stats = [worker(j) for j in jobs]

        # Merge shards into one queue file, preserving roster then filename order so
        # a rerun produces a byte-identical queue.
        order = {m: i for i, m in enumerate(roster)}
        stats.sort(key=lambda s: (order.get(s["model"], 1 << 30), s["file"]))

        if not dry_run:
            tmp = queue_path + ".tmp"
            with open(tmp, "w") as out:
                for s in stats:
                    with open(s["shard"], "r") as sh:
                        for line in sh:
                            out.write(line)
                            n_written += 1
            os.replace(tmp, queue_path)

    # --- per-model rollup ---
    per_model = {}
    for s in stats:
        m = per_model.setdefault(
            s["model"],
            {"n_total": 0, "n_regex_ok": 0, "n_regex_fail": 0,
             "n_stored_mismatch": 0, "n_empty_response": 0, "n_windowed": 0,
             "n_files": 0},
        )
        m["n_files"] += 1
        for k in ("n_total", "n_regex_ok", "n_regex_fail",
                  "n_stored_mismatch", "n_empty_response", "n_windowed"):
            m[k] += s[k]
    for m in per_model.values():
        m["regex_fail_rate"] = m["n_regex_fail"] / m["n_total"] if m["n_total"] else 0.0

    total = sum(m["n_total"] for m in per_model.values())
    total_fail = sum(m["n_regex_fail"] for m in per_model.values())
    total_mismatch = sum(m["n_stored_mismatch"] for m in per_model.values())

    summary = {
        "task_type": task_type,
        "task_dir": os.path.abspath(task_dir),
        "roster_yaml": os.path.abspath(roster_yaml),
        "limit": limit,
        "n_models_in_roster": len(roster),
        "n_models_present": len(roster),
        # Recorded so a queue can be matched to the prompt that will judge it.
        # run_judge_vllm refuses to resume an output file whose rows carry a
        # different stamp -- see load_done there.
        "prompt_fp": {str(k): short for k, (_full, short) in fingerprints.items()},
        "n_files": len(jobs),
        "n_responses": total,
        "n_regex_fail": total_fail,
        "regex_fail_rate": total_fail / total if total else 0.0,
        "n_stored_mismatch": total_mismatch,
        "queue_path": None if dry_run else os.path.abspath(queue_path),
        "n_queue_rows": n_written,
        "per_model": per_model,
    }

    # baseline_path was resolved above, alongside queue_path, so _retract can name
    # both from any failure path.
    if not dry_run:
        with open(baseline_path, "w") as f:
            json.dump(summary, f, indent=2)

    # --- report ---
    print(f"\n{'model':<62} {'responses':>9} {'regex-fail':>11} {'rate':>7}")
    for model in roster:
        if model not in per_model:
            continue
        m = per_model[model]
        print(f"{model:<62} {m['n_total']:>9,} {m['n_regex_fail']:>11,} "
              f"{m['regex_fail_rate']*100:>6.1f}%")
    print(f"\n{'TOTAL':<62} {total:>9,} {total_fail:>11,} "
          f"{(total_fail/total*100 if total else 0):>6.1f}%")

    # --- gates ---
    ok = True
    if total_mismatch:
        print(f"\n[GATE FAIL] replayed strict parser disagrees with the stored "
              f"filtered_resps on {total_mismatch:,} records. judge_io has drifted "
              f"from medvision_bm.utils.parse_utils -- fix before proceeding.")
        ok = False
    else:
        print(f"\n[gate ok] replayed strict parser matches stored filtered_resps on all "
              f"{total:,} records")

    if limit is None:
        expected = EXPECTED_ROSTER_COUNTS.get(task_type)
        # The expected counts describe the main roster over the DEFAULT task
        # directory only. An OOD split (a TASK_DIR_<task> override) reuses the
        # task type with a different --task_dir and a 3-model roster, so its
        # total can never match -- gating it would make every OOD build abort on
        # numbers that were never a promise about that tree.
        if expected is not None and os.path.abspath(task_dir) != os.path.abspath(
                DEFAULT_TASK_DIR[task_type]):
            print(f"[gate n/a] non-default task_dir {task_dir}; the full-roster "
                  f"count check applies only to {DEFAULT_TASK_DIR[task_type]}")
        elif expected is not None:
            if total == expected:
                print(f"[gate ok] response count {total:,} matches expected {expected:,}")
            else:
                print(f"[GATE FAIL] response count {total:,} != expected {expected:,}. "
                      f"Roster resolution is wrong -- fix before spending GPU time.")
                ok = False
    else:
        print(f"[gate n/a] --limit {limit} in force; full-roster count check skipped")

    if not ok:
        # The queue and baseline are written above, before these gates can be
        # evaluated, so a [GATE FAIL] used to leave a fully consumable queue on
        # disk -- and Stage 1 would happily judge it. Every gate exits through the
        # same retraction, so the abort is real rather than advisory.
        _retract("a failed Stage 0 must leave nothing Stage 1 can consume")
    elif not dry_run:
        print(f"\nqueue    -> {queue_path} ({n_written:,} rows)")
        print(f"baseline -> {baseline_path}")
    else:
        print("\n[dry-run] no queue written")

    return summary, ok


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_type", required=True, choices=["TL", "AD", "Detection"])
    p.add_argument("--task_dir", default=None,
                   help="Result dir with one subdir per model (default: per task_type)")
    p.add_argument("--config_yaml", default=None,
                   help="Evaluated-model roster YAML (default: per task_type)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max records per JSONL file, first-N by doc_id ascending")
    p.add_argument("--processes", "-p", type=int, default=None)
    p.add_argument("--out_dir", default=None, help="Default: --task_dir")
    p.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                   help="Count and gate only; do not write the queue")
    return p.parse_args()


def main():
    args = parse_args()
    task_dir = args.task_dir or DEFAULT_TASK_DIR[args.task_type]
    roster_yaml = args.config_yaml or DEFAULT_ROSTER_YAML[args.task_type]
    out_dir = args.out_dir or task_dir
    os.makedirs(out_dir, exist_ok=True)

    _, ok = build_queue(
        task_type=args.task_type,
        task_dir=task_dir,
        roster_yaml=roster_yaml,
        limit=args.limit,
        processes=args.processes,
        out_dir=out_dir,
        dry_run=args.dry_run,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
