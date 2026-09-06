#!/usr/bin/env python3
"""Read-only inspector for a MedVision ``Results/<task_tag>/`` tree.

Purpose
    Answer "where is this evaluation?" without opening files by hand: per model directory it
    reports the raw lmms_eval outputs (``*_samples_<task>.jsonl`` + ``*_results.json``), the
    crash-safe ``response_cache/<task>_rank<N>.jsonl`` shards, whether ``parsed/`` and any
    ``llm-parsed_<judge>/`` directories exist, and how the on-disk outputs compare with the
    ``completed_tasks/completed_tasks_<task_tag>.json`` tracker and the task list JSON.

Prerequisites
    Python >= 3.8, standard library only. Never modifies anything.

Examples
    python check_results_tree.py --results-dir <repo>/Results/MedVision-detect-CoT --repo-root <repo>
    python check_results_tree.py --results-dir Results/MedVision-TL-CoT --model MyModel --show-tasks
    python check_results_tree.py --results-dir Results/MedVision-AD-CoT \
        --completed-json completed_tasks/completed_tasks_MedVision-AD-CoT.json \
        --tasks-json tasks_list/tasks_MedVision-AD-CoT.json --json --strict

Exit codes
    0 report produced (and, with --strict, nothing inconsistent); 1 --strict found gaps
    (missing outputs / outputs not marked complete / duplicate output files); 2 bad arguments
    or the results directory does not exist.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RANK_RE = re.compile(r"^(?P<task>.+)_rank(?P<rank>\d+)\.jsonl$")


def die(msg: str, code: int = 2) -> None:
    print(f"[check_results_tree] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def task_from_samples_filename(name: str) -> str | None:
    """``<prefix>_samples_<task>.jsonl`` -> task. The tool-use entry point writes
    ``<task>_samples_0.jsonl`` instead, so a purely numeric suffix means the task is the prefix."""
    if not name.endswith(".jsonl") or "_samples_" not in name:
        return None
    prefix, suffix = name[: -len(".jsonl")].split("_samples_", 1)
    return prefix if suffix.isdigit() else suffix


def count_lines(path: Path) -> int:
    n = 0
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                n += chunk.count(b"\n")
        # a final line without newline still counts
        with open(path, "rb") as f:
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    n += 1
    except OSError:
        return -1
    return n


def load_json(path: Path | None):
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        die(f"{path} is not valid JSON: {e}", 2)


def inspect_model(mdir: Path, expected: list[str] | None, completed: dict, count: bool) -> dict:
    samples: dict[str, list[str]] = {}
    results_json = 0
    for p in sorted(mdir.iterdir()):
        if p.is_file():
            if p.name.endswith("_results.json"):
                results_json += 1
            t = task_from_samples_filename(p.name)
            if t:
                samples.setdefault(t, []).append(p.name)
    sample_lines = {}
    if count:
        for t, files in samples.items():
            sample_lines[t] = sum(count_lines(mdir / f) for f in files)

    cache: dict[str, dict] = {}
    cache_dir = mdir / "response_cache"
    if cache_dir.is_dir():
        for p in sorted(cache_dir.iterdir()):
            m = RANK_RE.match(p.name)
            if not m:
                continue
            ent = cache.setdefault(
                m.group("task"), {"ranks": 0, "cached_responses": 0 if count else None}
            )
            ent["ranks"] += 1
            if count:
                ent["cached_responses"] += max(count_lines(p), 0)

    parsed_dir = mdir / "parsed"
    parsed = None
    if parsed_dir.is_dir():
        files = list(parsed_dir.iterdir())
        parsed = {
            "jsonl": sum(1 for f in files if f.suffix == ".jsonl"),
            "summary_files": sorted(f.name for f in files if f.name.startswith("summary_")),
        }
    llm_parsed = sorted(p.name for p in mdir.iterdir() if p.is_dir() and p.name.startswith("llm-parsed_"))

    done = sorted(t for t, v in (completed.get(mdir.name) or {}).items() if v)
    with_output = sorted(samples)
    rep = {
        "model": mdir.name,
        "samples_jsonl_files": sum(len(v) for v in samples.values()),
        "results_json_files": results_json,
        "tasks_with_output": with_output,
        "sample_lines_per_task": sample_lines if count else None,
        "duplicate_output_tasks": sorted(t for t, v in samples.items() if len(v) > 1),
        "response_cache": {"present": cache_dir.is_dir(), "tasks": cache} if cache_dir.is_dir() else {"present": False, "tasks": {}},
        "parsed": parsed,
        "llm_parsed_dirs": llm_parsed,
        "completed_tasks_marked": done,
        "marked_complete_without_output": sorted(set(done) - set(with_output)),
        "output_not_marked_complete": sorted(set(with_output) - set(done)),
    }
    if expected is not None:
        exp = set(expected)
        rep["expected_tasks"] = len(expected)
        rep["missing_output_tasks"] = sorted(exp - set(with_output))
        rep["unexpected_output_tasks"] = sorted(set(with_output) - exp)
        rep["cache_only_tasks"] = sorted((set(cache) & exp) - set(with_output))  # in-flight / interrupted tasks
    return rep


def print_text(report: dict, show_tasks: bool) -> None:
    print(f"Results dir : {report['results_dir']}")
    print(f"Task tag    : {report['task_tag']}")
    print(f"Completed   : {report['completed_json'] or '(not found)'}")
    print(f"Task list   : {report['tasks_json'] or '(not found)'}" + (f"  [{report['expected_tasks']} tasks]" if report.get("expected_tasks") else ""))
    if report["task_level_files"]:
        print("Task-level  : " + ", ".join(report["task_level_files"]))
    print(f"Models      : {len(report['models'])}\n")
    for m in report["models"]:
        rc = m["response_cache"]
        flags = []
        if m.get("missing_output_tasks"):
            flags.append(f"missing={len(m['missing_output_tasks'])}")
        if m["output_not_marked_complete"]:
            flags.append(f"not-marked={len(m['output_not_marked_complete'])}")
        if m["marked_complete_without_output"]:
            flags.append(f"marked-no-output={len(m['marked_complete_without_output'])}")
        if m["duplicate_output_tasks"]:
            flags.append(f"duplicates={len(m['duplicate_output_tasks'])}")
        if m.get("cache_only_tasks"):
            flags.append(f"in-flight={len(m['cache_only_tasks'])}")
        status = "OK" if not flags else "; ".join(flags)
        print(f"[{m['model']}]  {status}")
        print(f"    outputs : {m['samples_jsonl_files']} samples jsonl / {m['results_json_files']} results json"
              f" / tasks with output {len(m['tasks_with_output'])} / marked complete {len(m['completed_tasks_marked'])}")
        cache_txt = "absent" if not rc["present"] else f"{len(rc['tasks'])} task(s), {sum(v['ranks'] for v in rc['tasks'].values())} shard file(s)"
        if rc["present"] and any(v.get("cached_responses") for v in rc["tasks"].values()):
            cache_txt += f", {sum(v['cached_responses'] for v in rc['tasks'].values())} cached responses"
        print(f"    cache   : {cache_txt}")
        p = m["parsed"]
        if p is None:
            parsed_txt = "absent"
        else:
            summaries = ", ".join(p["summary_files"]) or "none"
            parsed_txt = "{} jsonl, summaries: {}".format(p["jsonl"], summaries)
        print(f"    parsed/ : {parsed_txt}")
        print(f"    judge   : {', '.join(m['llm_parsed_dirs']) or 'none'}")
        if show_tasks:
            for t in m["tasks_with_output"]:
                n = (m.get("sample_lines_per_task") or {}).get(t)
                c = rc["tasks"].get(t, {}).get("cached_responses")
                mark = "done" if t in m["completed_tasks_marked"] else "NOT marked"
                print(f"      - {t}: {n if n is not None else '?'} samples, cache {c if c is not None else '-'}, {mark}")
            for t in m.get("missing_output_tasks", []):
                c = rc["tasks"].get(t, {}).get("cached_responses")
                print(f"      - {t}: NO OUTPUT" + (f" (cache has {c} responses -> interrupted mid-task)" if c else ""))
        print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Inspect a MedVision Results/<task_tag> tree (read-only).")
    p.add_argument("--results-dir", required=True, type=Path, help="Results/<task_tag> directory")
    p.add_argument("--model", action="append", help="only this model directory name (repeatable)")
    p.add_argument("--completed-json", type=Path, help="completed_tasks_<task_tag>.json (default: derived from --repo-root or ../../completed_tasks)")
    p.add_argument("--tasks-json", type=Path, help="task-list JSON to compare against (default: <repo-root>/tasks_list/tasks_<task_tag>.json if present)")
    p.add_argument("--repo-root", type=Path, help="MedVision checkout used to derive completed_tasks/ and tasks_list/ defaults")
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    p.add_argument("--show-tasks", action="store_true", help="list per-task rows in the text report")
    p.add_argument("--no-line-counts", action="store_true", help="skip counting lines in sample/cache jsonl files (faster on huge trees)")
    p.add_argument("--strict", action="store_true", help="exit 1 if any selected model has missing outputs, outputs not marked complete, or duplicate output files")
    args = p.parse_args(argv)

    rdir = args.results_dir.resolve()
    if not rdir.is_dir():
        die(f"results directory does not exist: {rdir}")
    task_tag = rdir.name
    roots = [r for r in (args.repo_root, rdir.parent.parent) if r is not None]

    completed_path = args.completed_json
    if completed_path is None:
        for r in roots:
            cand = Path(r) / "completed_tasks" / f"completed_tasks_{task_tag}.json"
            if cand.is_file():
                completed_path = cand
                break
    tasks_path = args.tasks_json
    if tasks_path is None:
        for r in roots:
            cand = Path(r) / "tasks_list" / f"tasks_{task_tag}.json"
            if cand.is_file():
                tasks_path = cand
                break
    completed = load_json(completed_path) or {}
    if not isinstance(completed, dict):
        die(f"completed-tasks file is not a JSON object: {completed_path}")
    tasks_obj = load_json(tasks_path)
    expected = list(tasks_obj.keys()) if isinstance(tasks_obj, dict) else None

    model_dirs = sorted(d for d in rdir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if args.model:
        wanted = set(args.model)
        missing = wanted - {d.name for d in model_dirs}
        if missing:
            die(f"model directory not found under {rdir}: {', '.join(sorted(missing))}")
        model_dirs = [d for d in model_dirs if d.name in wanted]

    report = {
        "results_dir": str(rdir),
        "task_tag": task_tag,
        "completed_json": str(completed_path) if completed_path and Path(completed_path).is_file() else None,
        "tasks_json": str(tasks_path) if tasks_path and Path(tasks_path).is_file() else None,
        "expected_tasks": len(expected) if expected is not None else None,
        "task_level_files": sorted(f.name for f in rdir.iterdir() if f.is_file() and (f.name.startswith("summary_") or f.name.startswith("judge-"))),
        "models": [inspect_model(d, expected, completed, not args.no_line_counts) for d in model_dirs],
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text(report, args.show_tasks)

    if args.strict:
        bad = any(
            m.get("missing_output_tasks") or m["output_not_marked_complete"] or m["duplicate_output_tasks"]
            for m in report["models"]
        )
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
