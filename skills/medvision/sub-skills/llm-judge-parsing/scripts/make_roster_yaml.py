#!/usr/bin/env python3
"""Write an LLM-judge roster YAML from the model directories of a Results tree.

Purpose
-------
The MedVision LLM-as-judge pipeline decides WHICH models to judge from a roster
YAML holding one mapping:

    model_display_name:
      "<model directory name under Results/<task_tag>>": "<label used in reports>"

Each key must be a directory directly under the Results tree that contains the
benchmark's ``parsed/*.jsonl`` records; Stage 0 otherwise aborts with
``[GATE FAIL] ... no such directory`` or ``no sample files in parsed/``. This
script builds that mapping from a tree and skips everything that is not a
judgeable model directory: plain files, judge artifacts (``judge-queue_*``,
``judge-out_*``, ``judge-baseline_*``, ``.judge-shards_*``), ``summary_*``,
``llm-parsed*`` and ``_archive*`` entries, hidden entries, and -- unless
``--no-require-parsed`` is given -- directories without ``parsed/*.jsonl``.

Prerequisites: standard library only. Scalars are written JSON-quoted, which
PyYAML's ``safe_load`` (what the pipeline's ``load_roster`` uses) reads back as
plain strings, so keys with dots, colons or ``__`` are safe.

Examples
--------
  python make_roster_yaml.py --results-dir <repo>/Results/My-TL-tree --dry-run
  python make_roster_yaml.py --results-dir Results/My-TL-tree \\
      --include-glob 'Qwen*' --exclude-glob '*_bugfix-*' \\
      --display-name-map '{"Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL (7B)"}' \\
      --out config-my-roster.yaml
  python make_roster_yaml.py --results-dir Results/My-TL-tree \\
      --display-name-map names.json --out config-my-roster.yaml

Then point one task at it (needs a repository checkout and the judge env):
  TASKS="TL" TASK_DIR_TL=Results/My-TL-tree ROSTER_YAML_TL=/abs/path/config-my-roster.yaml \\
  PYTHON=<judge-env>/bin/python bash <repo>/script/llm-parsing/run_llm_parsing.sh stage0 smoke full analyze

Exit codes: 0 success (also for --dry-run); 1 no model directory qualified;
3 bad arguments (missing tree, unreadable display-name map, --out missing).
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_SKIP_PREFIXES = ("judge-", ".judge-shards_", "llm-parsed", "summary_", "_archive", ".")


def load_display_map(spec: Optional[str]) -> Dict[str, str]:
    """Accept inline JSON or a path to a JSON file mapping dir name -> label."""
    if not spec:
        return {}
    text = spec
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as fh:
            text = fh.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--display-name-map is neither a JSON object nor a readable JSON file: {exc}")
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                             for k, v in data.items()):
        raise SystemExit("--display-name-map must be a flat JSON object of string -> string")
    return data


def scan_tree(results_dir: str, include: List[str], exclude: List[str],
              require_parsed: bool) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Return (model dir names kept, [(skipped, why)], [(kept-with-warning, why)])."""
    kept: List[str] = []
    skipped: List[Tuple[str, str]] = []
    warnings: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(results_dir), key=str.lower):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            skipped.append((name, "not a directory"))
            continue
        if name.startswith(_SKIP_PREFIXES):
            skipped.append((name, "judge/summary/archive artifact, not a model directory"))
            continue
        if include and not any(fnmatch.fnmatchcase(name, g) for g in include):
            skipped.append((name, "does not match any --include-glob"))
            continue
        if any(fnmatch.fnmatchcase(name, g) for g in exclude):
            skipped.append((name, "matches an --exclude-glob"))
            continue
        n_parsed = len(glob.glob(os.path.join(path, "parsed", "*.jsonl")))
        if n_parsed == 0:
            if require_parsed:
                skipped.append((name, "no parsed/*.jsonl (run parse_outputs first, or pass --no-require-parsed)"))
                continue
            warnings.append((name, "kept without parsed/*.jsonl; Stage 0 will refuse it"))
        kept.append(name)
    return kept, skipped, warnings


def render_yaml(models: List[str], display: Dict[str, str], header: str) -> str:
    lines = [f"# {header}", "", "model_display_name:"]
    for m in models:
        label = display.get(m, m)
        lines.append(f"  {json.dumps(m)}: {json.dumps(label)}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a judge roster YAML (model_display_name map) from a Results tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n", 1)[-1])
    ap.add_argument("--results-dir", required=True,
                    help="Results/<task_tag> tree holding one directory per model")
    ap.add_argument("--include-glob", action="append", default=[], metavar="GLOB",
                    help="keep only directory names matching this glob (repeatable; OR-ed)")
    ap.add_argument("--exclude-glob", action="append", default=[], metavar="GLOB",
                    help="drop directory names matching this glob (repeatable)")
    ap.add_argument("--display-name-map", default=None, metavar="JSON",
                    help="inline JSON object or path to a JSON file: {\"<dir>\": \"<label>\"}; "
                         "unmapped directories use their own name as the label")
    ap.add_argument("--no-require-parsed", action="store_true",
                    help="keep directories that have no parsed/*.jsonl (Stage 0 will still refuse them)")
    ap.add_argument("--out", default=None, help="roster YAML to write (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="print the YAML to stdout; write nothing")
    ap.add_argument("--header", default=None,
                    help="comment line at the top of the file (default names the tree)")
    args = ap.parse_args(argv)

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"error: --results-dir {args.results_dir} is not a directory", file=sys.stderr)
        return 3
    if not args.dry_run and not args.out:
        print("error: --out is required unless --dry-run is given", file=sys.stderr)
        return 3
    try:
        display = load_display_map(args.display_name_map)
    except SystemExit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    models, skipped, warned = scan_tree(results_dir, args.include_glob, args.exclude_glob,
                                        not args.no_require_parsed)
    for name, why in skipped:
        print(f"skip  {name}: {why}", file=sys.stderr)
    for name, why in warned:
        print(f"WARN  {name}: {why}", file=sys.stderr)
    unmapped = [m for m in display if m not in models]
    for m in unmapped:
        print(f"note  display-name-map key {m!r} is not a kept model directory", file=sys.stderr)
    if not models:
        print("error: no model directory qualified; nothing written", file=sys.stderr)
        return 1

    qualifier = "" if warned else " with parsed/ records"
    header = args.header or (f"Judge roster for {os.path.basename(results_dir)}: "
                             f"{len(models)} model director{'y' if len(models) == 1 else 'ies'}"
                             f"{qualifier}")
    text = render_yaml(models, display, header)
    if args.dry_run:
        sys.stdout.write(text)
    else:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {out} ({len(models)} models)", file=sys.stderr)
    print(f"kept  {len(models)} model(s); skipped {len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
