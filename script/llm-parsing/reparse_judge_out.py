"""Re-run the decoder and validator over an existing judge-out file, on CPU.

A decoder or validator change is worth nothing until it reaches the 496K rows
already on disk. Re-judging them costs a GPU pass; re-parsing them costs minutes,
because ``run_judge_vllm._out_row`` persists what it decoded:

* rows carrying ``raw``          -- full re-parse: ``parse_judge_json`` then
  ``validate_judge_obj``. Only rows written with raw persistence can take this
  path, which is why ``_raw_to_keep`` now keeps raw on invalid rows by default.
* rows carrying ``final_answer`` -- re-VALIDATION only. Enough for a validator
  change, and the reason the skeleton-echo fix reached 3,407 stranded Detection
  rows without a GPU.
* rows with neither              -- untouched. ``no_json_object`` rows written
  before raw persistence hold nothing to re-read; they need a re-judge.

Refuses to write when any row moves ok -> invalid. That direction means the change
under test destroyed a verdict the pipeline had already accepted, and no recovery
rate justifies it.

Reads the file LINE BY LINE, so the counts it prints are per line, not per answer.
Stage 1 appends, and ``apply_judge.load_judge_index`` resolves duplicates
last-wins, so a file that has been through a repair pass holds superseded lines
that no longer affect any metric. Those lines are counted here and are not
outstanding work: after the max_tokens repair this over-reported T/L's invalid
rows as 5,130 where Stage 1 correctly found 1,306. Trust Stage 1's "outstanding"
for what remains to judge; trust these counts only for what a re-parse changes.

Usage (from the repo root):
    python script/llm-parsing/reparse_judge_out.py \\
        --in  Results/MedVision-detect-v2/judge-out_Detection.jsonl \\
        --out Results/MedVision-detect-v2/judge-out_Detection.jsonl.new

Then replace the original once the table reads as expected. Stage 2 reads only
the file it is pointed at, so nothing downstream moves until you do.

WRITING IN PLACE IS THE GUARDED PATH. On success this tool drops a marker beside
its --out file, and test-sweep.sh refuses to overwrite a merged judge-out whose
marker is at least as new as the file itself -- which is what stops a later
sharded sweep from silently reverting the re-parse. That protection follows the
--out path, so:

  --out <merged>            in place; the merged file is guarded immediately.
  --out <merged>.new + mv   the marker describes the SIDE file, and the mv leaves
                            the merged file unguarded. Re-run in place, or accept
                            that a sharded re-run can overwrite it.

Never name --out "<merged>.reparsed": that collides with nothing now, but earlier
versions used it as the marker name, so old trees may still hold one.
"""

import argparse
import collections
import json


def _marker_path(out_path):
    """Return the re-parse marker path for ``out_path``.

    A dotfile beside the output. Shared verbatim with test-sweep.sh's merge guard;
    if you change it, change it there too -- the two are the only readers.
    """
    import os as _os

    d, base = _os.path.split(_os.path.abspath(out_path))
    return _os.path.join(d, "." + base + ".reparsed")
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from judge_decode import parse_judge_json, validate_judge_obj

# Detection prescribes no intermediate steps; T/L and A/D do.
_STEP_TASKS = ("TL", "AD")


def reparse_row(row):
    """Re-decide one row's status from whatever it preserved.

    Args:
        row: A judge-output record.

    Returns:
        tuple: ``(new_status, new_reason, new_obj, path)`` where ``path`` is one
        of ``"raw"``, ``"validate"``, ``"skipped"``. ``new_obj`` is ``None`` when
        nothing was re-decoded.
    """
    expects_steps = row.get("task_type") in _STEP_TASKS
    # Stage 1 makes ONE model call per distinct response and fills the repeats
    # from it, marking those rows "+cached". That is provenance, not a verdict:
    # re-deciding the verdict must not quietly relabel a cached row as a fresh
    # one. Carried through every branch below.
    tag = "+cached" if str(row.get("judge_reason", "")).endswith("+cached") else ""

    raw = row.get("raw")
    # TRUNCATED raw is worse than no raw: re-deciding from a prefix of the text
    # cannot reproduce a verdict made on the whole of it, and it manufactures
    # phantom regressions (an early --keep_raw ran with a 2,000-char cap, and on
    # those files every stored "ok" row whose raw hit the cap re-decides to
    # invalid). Fall back to re-VALIDATION when the stored text is incomplete.
    if raw and row.get("raw_len") is not None and len(raw) < row["raw_len"]:
        raw = None
    if raw:
        obj, reason = parse_judge_json(raw)
        if obj is None:
            return "invalid", reason + tag, None, "raw"
        ok, vreason = validate_judge_obj(obj, expects_steps)
        return ("ok" if ok else "invalid"), ("ok" if ok else vreason) + tag, obj, "raw"

    fa = row.get("final_answer")
    if fa is not None:
        # Rebuild the object as the validator saw it. ``steps`` is stored beside
        # final_answer rather than inside it, so it has to go back in.
        obj = {"final_answer": fa}
        if row.get("steps") is not None:
            obj["steps"] = row["steps"]
        ok, vreason = validate_judge_obj(obj, expects_steps)
        return ("ok" if ok else "invalid"), ("ok" if ok else vreason) + tag, obj, "validate"

    return row.get("judge_status"), row.get("judge_reason"), None, "skipped"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", required=True, help="judge-out_*.jsonl to re-parse")
    p.add_argument("--out", dest="out_path", required=True, help="Where to write the result")
    p.add_argument("--allow_regressions", action="store_true",
                   help="Write even if rows moved ok -> invalid. For investigating "
                        "a regression, never for producing reported numbers.")
    p.add_argument("--allow_value_changes", action="store_true",
                   help="Write even if an ok row's final_answer/steps CHANGED "
                        "under the new decoder. Worse than a regression -- a "
                        "silently different published value -- so it aborts by "
                        "default. Only for deliberate, reviewed decoder changes.")
    args = p.parse_args()

    transitions = collections.Counter()
    paths = collections.Counter()
    new_reasons = collections.Counter()
    echoes = collections.Counter()
    regressed = []
    value_changed = []
    out_lines = []
    last_by_qid = {}
    n = 0

    n_torn = 0
    with open(args.in_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A killed sweep leaves one unterminated final line -- the exact
                # condition run_judge_vllm._repair_torn_tail exists to clean up.
                # Every other consumer of these files tolerates it
                # (apply_judge.load_judge_index, assert_judge_out_prompt), so
                # dying with a traceback here made the CPU-only repair path the
                # single tool that could not open a file it was built to repair.
                n_torn += 1
                continue
            n += 1
            old_status = row.get("judge_status")
            old_fa = row.get("final_answer")
            old_steps = row.get("steps")
            status, reason, obj, path = reparse_row(row)
            paths[path] += 1
            transitions[(old_status, status)] += 1
            # Also track the LAST line per qid. apply_judge.load_judge_index is
            # last-wins, so a superseded duplicate that re-decodes invalid -> ok
            # changes no downstream verdict; counting lines reports it as a
            # recovery anyway. This is the number a user reads to decide whether
            # to keep the reparsed file, so report the effective one too.
            last_by_qid[row.get("qid")] = (old_status, status)
            if status == "invalid":
                new_reasons[reason] += 1
            if old_status == "ok" and status == "invalid":
                if len(regressed) < 5:
                    regressed.append((row.get("qid"), row.get("judge_reason"), reason))
            # The failure mode WORSE than a regression: the row stays "ok" but the
            # re-decode read a DIFFERENT object out of the same raw. Nothing else
            # in the pipeline would ever notice -- Stage 2 trusts an ok row -- so
            # a decoder change that shifts values must be caught right here.
            if (path == "raw" and old_status == "ok" and status == "ok"
                    and obj is not None
                    and (obj.get("final_answer") != old_fa
                         or obj.get("steps") != old_steps)):
                if len(value_changed) < 5:
                    value_changed.append((row.get("qid"), old_fa, obj.get("final_answer")))
                else:
                    value_changed.append(None)

            if path != "skipped":
                row["judge_status"] = status
                row["judge_reason"] = reason
                if obj is not None:
                    row["final_answer"] = obj.get("final_answer")
                    # Set and CLEAR in the same branch: a re-decode that yields no
                    # steps must not leave the superseded parse's steps beside a
                    # fresh final_answer -- apply_judge would build LLM_judge_steps
                    # from an object that no longer exists.
                    if obj.get("steps") is not None:
                        row["steps"] = obj.get("steps")
                    elif path == "raw":
                        row.pop("steps", None)
            # Placeholder-echo census: all-zero values are the pre-2026-08-08
            # skeleton's own placeholder quoted back (see the all-zero branch in
            # validate_judge_obj). Counted per status so the echo rate stays a
            # visible judge-quality signal instead of being silently absorbed.
            fa = row.get("final_answer")
            if (isinstance(fa, dict) and isinstance(fa.get("values"), list)
                    and fa["values"]
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and v == 0 for v in fa["values"])):
                echoes[str(fa.get("status"))] += 1
            out_lines.append(json.dumps(row))

    print(f"[reparse] {args.in_path}")
    print(f"  rows: {n:,}   re-parsed from raw: {paths['raw']:,}   "
          f"re-validated: {paths['validate']:,}   untouched: {paths['skipped']:,}")
    print("  transitions (old -> new):")
    for (old, new), c in sorted(transitions.items(), key=lambda kv: -kv[1]):
        arrow = "  " if old == new else "<-" if new == "ok" else "!!"
        print(f"    {arrow} {str(old):8s} -> {str(new):8s} {c:,}")
    if new_reasons:
        print("  remaining invalid reasons:")
        for r, c in new_reasons.most_common(8):
            print(f"       {r:34s} {c:,}")
    if echoes:
        print("  all-zero values (skeleton echo) by status:")
        for s, c in echoes.most_common():
            print(f"       {s:34s} {c:,}")

    if value_changed:
        print(f"  ok rows whose final_answer/steps CHANGED: {len(value_changed):,}")

    n_regressed = transitions.get(("ok", "invalid"), 0)
    if n_regressed and not args.allow_regressions:
        print(f"\n  ABORT: {n_regressed:,} rows moved ok -> invalid; nothing written.")
        for qid, was, now in regressed:
            print(f"    qid={qid} was={was!r} now={now!r}")
        return 1
    if value_changed and not args.allow_value_changes:
        print(f"\n  ABORT: {len(value_changed):,} ok row(s) decoded to a DIFFERENT "
              f"object under the new decoder; nothing written.")
        for entry in value_changed[:5]:
            if entry:
                qid, was, now = entry
                print(f"    qid={qid}\n      was {was!r}\n      now {now!r}")
        return 1

    tmp = args.out_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, args.out_path)
    # Marker for test-sweep.sh's merge guard. A sharded sweep rebuilds the merged
    # file from its shards and would silently revert this re-parse; the guard
    # cannot detect that by comparing verdicts, because the judge is not
    # reproducible run to run and a legitimate re-judge changes verdicts too.
    #
    # The name is deliberate on two counts. It is a DOTFILE beside the output, not
    # "<out>.reparsed": that suffix is what this module's own usage block hands to
    # --out, so a side-file re-parse would have left a JSONL data file where the
    # guard expects a marker (aborting clean sweeps), and it would also have been
    # swallowed by test-sweep.sh's "<out>.n*.shard*" stale-shard glob when a SHARD
    # was re-parsed. A leading dot shares neither prefix.
    with open(_marker_path(args.out_path), "w") as fh:
        fh.write(f"reparsed from {args.in_path}\n{len(out_lines)} rows\n")
    eff = sum(1 for o, s in last_by_qid.values() if o == "invalid" and s == "ok")
    print(f"\n  wrote {len(out_lines):,} rows -> {args.out_path}")
    print(f"  recovered: {eff:,} qid(s) invalid -> ok  (last-wins, i.e. what "
          f"Stage 2 will actually see)")
    print(f"             {transitions.get(('invalid', 'ok'), 0):,} LINES changed "
          f"invalid -> ok, incl. superseded duplicates")
    if n_torn:
        print(f"  NOTE {n_torn} unparseable line(s) skipped (torn tail); they are "
              f"NOT carried into {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
