#!/usr/bin/env python3
"""list_task_yamls.py -- inventory and consistency check for MedVision task YAMLs.

Purpose
    Every MedVision evaluation task is a pair of YAML files under the vendored
    ``lmms_eval/tasks/<Dataset>/``:
      * a BASE yaml (``<Dataset>_<TaskType>_base[-CoT|-VP|...].yaml``) that carries
        ``include``/``tag``/``dataset_path``/``output_type``, the four ``!function utils.*``
        hooks and ``metric_list``; it has NO ``task:`` key, so lmms_eval never registers it;
      * one TASK yaml per subtask that sets only ``include:``, ``task:`` and ``dataset_name:``.
    A task only actually RUNS when its ``task:`` name is listed in a task-list JSON
    (``tasks_list/*.json``), which is the sole authority for what the pipeline evaluates --
    most shipped YAMLs are unused.

    This script walks the task tree, prints per dataset which YAMLs are bases and which are
    tasks (with ``task``, ``dataset_name`` and the tag inherited from the base), cross-checks
    them against any task-list JSONs you pass, and reports problems: a broken ``include:``
    target, a duplicate ``task:`` name, a task YAML with no ``dataset_name``, a task-list
    entry that has no YAML at all, and (informational) YAMLs no list references.

    Parsing is deliberately line-based -- the base YAMLs contain ``!function`` tags that
    ``yaml.safe_load`` rejects -- so no PyYAML is required and nothing is imported from
    lmms_eval.

Prerequisites
    Python >= 3.8, standard library only. The task tree is located from (in order):
    ``--tasks-dir``, ``--repo-root``/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks,
    the installed ``medvision_bm`` package, or an importable ``lmms_eval`` package.

Usage
    list_task_yamls.py                                   # full inventory
    list_task_yamls.py --dataset BraTS24                 # one dataset only
    list_task_yamls.py --tasks-json <dir>/tasks_MedVision-TL-CoT.json
    list_task_yamls.py --tasks-json a.json b.json --unused   # also list unreferenced tasks
    list_task_yamls.py --json > inventory.json
    list_task_yamls.py --repo-root <repo> --dataset Ceph-Biometrics-400

Exit codes
    0 = no problems; 1 = at least one problem (broken include, duplicate task name, missing
    dataset_name, task-list entry without a YAML); 2 = the task tree could not be located.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

TASKS_REL = os.path.join("medvision_lmms_eval", "lmms_eval", "tasks")
IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
# Shared includes that are not dataset task templates.
SHARED_DIR = "medvision"


# --------------------------------------------------------------------------- location
def _is_tasks_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isdir(os.path.join(path, SHARED_DIR))


def locate_tasks_dir(repo_root: Optional[str], explicit: Optional[str]) -> Optional[str]:
    # An explicitly given location must be right; never fall back and silently inspect a
    # different tree than the caller asked for.
    if explicit:
        return os.path.abspath(explicit) if _is_tasks_dir(explicit) else None
    if repo_root:
        cand = os.path.join(repo_root, "src", "medvision_bm", TASKS_REL)
        return os.path.abspath(cand) if _is_tasks_dir(cand) else None
    candidates: List[str] = []
    try:
        import medvision_bm  # type: ignore

        candidates.append(os.path.join(os.path.dirname(medvision_bm.__file__), TASKS_REL))
    except Exception:  # noqa: BLE001 - not installed / broken install
        pass
    try:
        import importlib.util

        spec = importlib.util.find_spec("lmms_eval")
        if spec is not None and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                candidates.append(os.path.join(loc, "tasks"))
    except Exception:  # noqa: BLE001
        pass
    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, SHARED_DIR)):
            return cand
    return None


# --------------------------------------------------------------------------- yaml-lite
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s*-\s+(.*?)\s*$")

SCALAR_KEYS = (
    "task",
    "dataset_name",
    "dataset_path",
    "test_split",
    "output_type",
    "doc_to_text",
    "doc_to_visual",
    "doc_to_target",
    "process_results",
    "group",
)


def parse_yaml_lite(path: str) -> Dict[str, object]:
    """Extract the handful of top-level keys this tool needs, tolerating `!function` tags."""
    out: Dict[str, object] = {}
    includes: List[str] = []
    tags: List[str] = []
    metrics: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.split("#", 1)[0].rstrip() if not raw.strip().startswith("#") else ""
        if not line.strip():
            i += 1
            continue
        if raw[:1] in (" ", "\t", "-"):  # nested content, handled by the block readers below
            i += 1
            continue
        m = _KEY_RE.match(line)
        if not m:
            i += 1
            continue
        key, value = m.group(1), m.group(2).strip()

        if key == "include":
            if value:
                includes.append(value.strip("'\" "))
            else:
                j = i + 1
                while j < len(lines):
                    item = _LIST_ITEM_RE.match(lines[j].split("#", 1)[0])
                    if item is None:
                        break
                    includes.append(item.group(1).strip("'\" "))
                    j += 1
                i = j - 1
        elif key == "tag":
            if value:
                tags.extend([t.strip() for t in value.strip("[]").split(",") if t.strip()])
            else:
                j = i + 1
                while j < len(lines):
                    item = _LIST_ITEM_RE.match(lines[j].split("#", 1)[0])
                    if item is None:
                        break
                    tags.append(item.group(1).strip("'\" "))
                    j += 1
                i = j - 1
        elif key == "metric_list":
            j = i + 1
            while j < len(lines):
                sub = lines[j].split("#", 1)[0]
                if sub.strip() and not (sub[:1] in (" ", "\t", "-")):
                    break
                mm = re.match(r"^\s*-?\s*metric:\s*(.+?)\s*$", sub)
                if mm:
                    metrics.append(mm.group(1).strip("'\" "))
                j += 1
            i = j - 1
        elif key in SCALAR_KEYS and value:
            out[key] = value.strip("'\" ")
        i += 1

    out["include"] = includes
    out["tag"] = tags
    out["metric_list"] = metrics
    return out


def resolve_include(yaml_path: str, include_value: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(yaml_path), include_value))


# --------------------------------------------------------------------------- inventory
def scan(tasks_dir: str, dataset_filter: Optional[str]) -> Tuple[Dict[str, Dict], List[str]]:
    """Return {dataset: {"bases": [...], "tasks": [...], "dir": path}} and a problem list."""
    datasets: Dict[str, Dict] = {}
    problems: List[str] = []
    seen_task_names: Dict[str, str] = {}

    for root, dirs, files in os.walk(tasks_dir):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS)
        rel = os.path.relpath(root, tasks_dir)
        if rel == ".":
            continue
        dataset = rel.split(os.sep)[0]
        if dataset == SHARED_DIR or dataset.startswith("_"):
            continue
        if dataset_filter and dataset != dataset_filter:
            continue
        for fname in sorted(files):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(root, fname)
            cfg = parse_yaml_lite(path)
            entry = datasets.setdefault(dataset, {"dir": root, "bases": [], "tasks": []})

            includes = cfg.get("include") or []
            resolved_includes = []
            for inc in includes:
                target = resolve_include(path, inc)
                resolved_includes.append({"raw": inc, "path": target, "exists": os.path.isfile(target)})
                if not os.path.isfile(target):
                    problems.append(f"{dataset}/{fname}: include target not found -> {inc}")

            if "task" in cfg:
                # tag is normally inherited from the included base yaml
                tags = list(cfg.get("tag") or [])
                if not tags:
                    for inc in resolved_includes:
                        if inc["exists"]:
                            tags.extend(parse_yaml_lite(inc["path"]).get("tag") or [])
                rec = {
                    "file": fname,
                    "task": cfg["task"],
                    "dataset_name": cfg.get("dataset_name"),
                    "dataset_path_override": cfg.get("dataset_path"),
                    "tags": tags,
                    "includes": [i["raw"] for i in resolved_includes],
                    "include_ok": all(i["exists"] for i in resolved_includes),
                }
                entry["tasks"].append(rec)
                ds_path = cfg.get("dataset_path")
                if not cfg.get("dataset_name"):
                    if not ds_path:
                        problems.append(
                            f"{dataset}/{fname}: task '{cfg['task']}' sets neither dataset_name nor dataset_path"
                        )
                    elif "/" not in str(ds_path):
                        # A bare config-looking string in dataset_path REPLACES the HF repo id
                        # (dataset_path: YongchengYAO/MedVision) instead of selecting a config.
                        problems.append(
                            f"{dataset}/{fname}: task '{cfg['task']}' overrides dataset_path with "
                            f"{ds_path!r} (no '/'), which looks like a dataset_name typo"
                        )
                if cfg["task"] in seen_task_names:
                    problems.append(
                        f"duplicate task name '{cfg['task']}' in {dataset}/{fname} "
                        f"(already defined by {seen_task_names[cfg['task']]})"
                    )
                else:
                    seen_task_names[cfg["task"]] = f"{dataset}/{fname}"
            else:
                entry["bases"].append(
                    {
                        "file": fname,
                        "tags": list(cfg.get("tag") or []),
                        "dataset_path": cfg.get("dataset_path"),
                        "test_split": cfg.get("test_split"),
                        "output_type": cfg.get("output_type"),
                        "doc_to_text": cfg.get("doc_to_text"),
                        "doc_to_target": cfg.get("doc_to_target"),
                        "process_results": cfg.get("process_results"),
                        "metrics": cfg.get("metric_list") or [],
                        "includes": [i["raw"] for i in resolved_includes],
                    }
                )
    return datasets, problems


def load_task_lists(paths: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Return {list_path: [task names]} plus IO problems."""
    listed: Dict[str, List[str]] = {}
    problems: List[str] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            problems.append(f"task list {p}: cannot read ({exc})")
            continue
        if isinstance(data, dict):
            listed[p] = list(data.keys())
        elif isinstance(data, list):
            listed[p] = [str(x) for x in data]
        else:
            problems.append(f"task list {p}: expected a JSON object or array")
    return listed, problems


# --------------------------------------------------------------------------- report
def build_report(tasks_dir: str, dataset_filter: Optional[str], task_list_paths: List[str]) -> Dict[str, object]:
    datasets, problems = scan(tasks_dir, dataset_filter)
    listed, list_problems = load_task_lists(task_list_paths)
    problems.extend(list_problems)

    all_yaml_tasks = {t["task"]: ds for ds, e in datasets.items() for t in e["tasks"]}
    referenced: Dict[str, List[str]] = {}
    for lst, names in listed.items():
        for name in names:
            referenced.setdefault(name, []).append(os.path.basename(lst))
            if name not in all_yaml_tasks:
                problems.append(f"task list {os.path.basename(lst)}: '{name}' has no task YAML")

    for ds, entry in datasets.items():
        for t in entry["tasks"]:
            t["referenced_by"] = referenced.get(t["task"], [])

    return {
        "tasks_dir": tasks_dir,
        "dataset_filter": dataset_filter,
        "task_lists": {os.path.basename(k): len(v) for k, v in listed.items()},
        "datasets": datasets,
        "dataset_count": len(datasets),
        "base_yaml_count": sum(len(e["bases"]) for e in datasets.values()),
        "task_yaml_count": sum(len(e["tasks"]) for e in datasets.values()),
        "referenced_task_count": sum(1 for n in all_yaml_tasks if n in referenced),
        "problems": problems,
    }


def print_text(report: Dict[str, object], show_unused: bool) -> None:
    print(f"tasks dir : {report['tasks_dir']}")
    if report["task_lists"]:
        for name, n in report["task_lists"].items():  # type: ignore[union-attr]
            print(f"task list : {name} ({n} names)")
    print()
    datasets: Dict[str, Dict] = report["datasets"]  # type: ignore[assignment]
    for ds in sorted(datasets):
        entry = datasets[ds]
        print(f"== {ds}  ({len(entry['bases'])} base, {len(entry['tasks'])} task yaml)")
        for b in entry["bases"]:
            tag = ",".join(b["tags"]) or "-"
            print(f"   [base] {b['file']}")
            print(f"          tag={tag}  doc_to_text={b['doc_to_text']}  metrics={','.join(b['metrics']) or '-'}")
        for t in entry["tasks"]:
            refs = ",".join(t["referenced_by"]) if t["referenced_by"] else ""
            mark = "USED" if refs else ("unused" if show_unused else "")
            if not refs and not show_unused and report["task_lists"]:
                continue
            print(f"   [task] {t['task']}")
            print(f"          dataset_name={t['dataset_name']}  tag={','.join(t['tags']) or '-'}"
                  + (f"  {mark}{(': ' + refs) if refs else ''}" if report["task_lists"] else ""))
        print()
    print(f"datasets={report['dataset_count']}  base yaml={report['base_yaml_count']}  "
          f"task yaml={report['task_yaml_count']}  referenced by the given lists={report['referenced_task_count']}")
    if report["problems"]:  # type: ignore[index]
        print("\nPROBLEMS:")
        for p in report["problems"]:  # type: ignore[index]
            print(f"  - {p}")
    else:
        print("\nNo problems: every include target exists, task names are unique and every "
              "task-list entry resolves to a YAML.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-dir", default=None, help="Explicit path to lmms_eval/tasks.")
    ap.add_argument("--repo-root", default=None, help="MedVision checkout root (uses <root>/src/medvision_bm/...).")
    ap.add_argument("--tasks-json", nargs="+", default=[], metavar="JSON",
                    help="One or more task-list JSONs (tasks_list/*.json) to cross-check against.")
    ap.add_argument("--dataset", default=None, help="Limit the inventory to one dataset folder (e.g. BraTS24).")
    ap.add_argument("--unused", action="store_true",
                    help="With --tasks-json, also print task YAMLs that no list references (default: hide them).")
    ap.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    args = ap.parse_args(argv)

    tasks_dir = locate_tasks_dir(args.repo_root, args.tasks_dir)
    if tasks_dir is None:
        print("ERROR: could not locate lmms_eval/tasks. Pass --tasks-dir or --repo-root <repo>, "
              "or install medvision_bm.", file=sys.stderr)
        return 2

    report = build_report(tasks_dir, args.dataset, args.tasks_json)
    if args.dataset and not report["datasets"]:
        print(f"ERROR: no dataset folder named {args.dataset!r} under {tasks_dir}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text(report, show_unused=args.unused or not args.tasks_json)
    return 1 if report["problems"] else 0  # type: ignore[index]


if __name__ == "__main__":
    sys.exit(main())
