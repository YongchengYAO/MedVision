#!/usr/bin/env python3
"""list_registered_models.py -- registry consistency check for MedVision VLM wrappers.

Purpose
    A MedVision model must be wired at (at least) three places before it can run the
    Tumor/Lesion-size and Angle/Distance tasks:
      1. the key -> class entry in ``lmms_eval/models/__init__.py::AVAILABLE_MODELS``,
      2. a module ``lmms_eval/models/<key>.py`` whose class carries ``@register_model("<key>")``,
      3. a branch in ``lmms_eval/tasks/medvision/medvision_utils.py::get_resized_img_shape``
         (a missing branch raises ``ValueError: <key> is not recognised/supported`` at
         prompt-build time).
    This script reads the three sites STATICALLY (``ast`` + regex, nothing is imported from
    the vendored ``lmms_eval`` besides locating it) and reports every registered key with its
    class, module status and dispatch branch, plus the dispatch-only aliases (SFT
    ``model_family_name`` strings such as ``qwen25vl``), the commented-out registry entries,
    and every mismatch. Each dispatch branch is classified with the repository's own strategy
    letters: A = fixed perceived size, B = probed/computed per image, C = API rule imported
    lazily from the model file.

Prerequisites
    Python >= 3.8, standard library only. The vendored ``lmms_eval`` tree is located from (in
    order): ``--lmms-eval-dir``, ``--repo-root``/src/medvision_bm/medvision_lmms_eval/lmms_eval,
    the installed ``medvision_bm`` package, or an importable ``lmms_eval`` package.

Usage
    list_registered_models.py                      # human-readable report
    list_registered_models.py --json               # machine-readable
    list_registered_models.py --repo-root <repo>   # a checkout instead of the installed package
    list_registered_models.py --expect vllm_mymodel  # checklist for a key you are adding

Exit codes
    0 = consistent (or --expect key fully wired); 1 = a registered key lacks a module,
    decorator or dispatch branch (or --expect key is incomplete); 2 = could not locate the
    vendored lmms_eval tree or parse its files.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, List, Optional

MODELS_INIT_REL = os.path.join("models", "__init__.py")
MEDVISION_UTILS_REL = os.path.join("tasks", "medvision", "medvision_utils.py")


# --------------------------------------------------------------------------- location
def _is_lmms_eval_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MODELS_INIT_REL)) and os.path.isfile(
        os.path.join(path, MEDVISION_UTILS_REL)
    )


def locate_lmms_eval_dir(repo_root: Optional[str], explicit: Optional[str]) -> Optional[str]:
    # An explicitly given location must be right; never fall back and silently inspect a
    # different tree than the caller asked for.
    if explicit:
        return os.path.abspath(explicit) if _is_lmms_eval_dir(explicit) else None
    if repo_root:
        cand = os.path.join(repo_root, "src", "medvision_bm", "medvision_lmms_eval", "lmms_eval")
        return os.path.abspath(cand) if _is_lmms_eval_dir(cand) else None
    candidates: List[str] = []
    try:
        import medvision_bm  # type: ignore

        candidates.append(os.path.join(os.path.dirname(medvision_bm.__file__), "medvision_lmms_eval", "lmms_eval"))
    except Exception:  # noqa: BLE001 - ImportError or a broken install; fall through
        pass
    try:
        import importlib.util

        spec = importlib.util.find_spec("lmms_eval")
        if spec is not None and spec.submodule_search_locations:
            candidates.extend(list(spec.submodule_search_locations))
    except Exception:  # noqa: BLE001
        pass
    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.isfile(os.path.join(cand, MODELS_INIT_REL)) and os.path.isfile(os.path.join(cand, MEDVISION_UTILS_REL)):
            return cand
    return None


# --------------------------------------------------------------------------- registry
def parse_available_models(models_init_path: str) -> Dict[str, str]:
    with open(models_init_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=models_init_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AVAILABLE_MODELS":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise ValueError("AVAILABLE_MODELS is not a dict literal")
                    return {str(k): str(v) for k, v in value.items()}
    raise ValueError("AVAILABLE_MODELS assignment not found")


_COMMENTED_ENTRY_RE = re.compile(r'^\s*#\s*"([A-Za-z0-9_]+)"\s*:\s*"([A-Za-z0-9_.]+)"\s*,?\s*$')


def parse_commented_out_models(models_init_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(models_init_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _COMMENTED_ENTRY_RE.match(line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def check_model_module(lmms_eval_dir: str, key: str, class_name: str) -> Dict[str, object]:
    path = os.path.join(lmms_eval_dir, "models", f"{key}.py")
    info: Dict[str, object] = {"module_path": path, "module_exists": os.path.isfile(path), "register_decorator": False, "class_defined": False}
    if info["module_exists"]:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        info["register_decorator"] = re.search(r'@register_model\(\s*["\']' + re.escape(key) + r'["\']', src) is not None
        info["class_defined"] = re.search(r"^class\s+" + re.escape(class_name) + r"\s*\(", src, re.M) is not None
    return info


# --------------------------------------------------------------------------- dispatch
def _keys_from_test(test: ast.AST) -> Optional[List[str]]:
    """Return the model_name literals compared in an `if`/`elif` test, or None if not literal."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        comp = test.comparators[0]
        if isinstance(test.ops[0], ast.In) and isinstance(comp, (ast.List, ast.Tuple)):
            keys = []
            for elt in comp.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    keys.append(elt.value)
                else:
                    return None
            return keys
        if isinstance(test.ops[0], ast.Eq) and isinstance(comp, ast.Constant) and isinstance(comp.value, str):
            return [comp.value]
    return None


def _classify_branch(body: List[ast.stmt]) -> Dict[str, object]:
    imports: List[str] = []
    probes: List[str] = []
    fixed: Optional[List[int]] = None
    returns_content = False
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and sub.module and sub.module.startswith("lmms_eval.models"):
                imports.append(sub.module + ":" + ",".join(a.name for a in sub.names))
            elif isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if name and (name.startswith("_process_img_") or name == "_padsquare_clip_content_hw"):
                    probes.append(name)
            elif isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "img_shape_resized_hw" and isinstance(sub.value, (ast.List, ast.Tuple)):
                        try:
                            fixed = [int(ast.literal_eval(e)) for e in sub.value.elts]
                        except Exception:  # noqa: BLE001
                            fixed = None
                    if isinstance(tgt, ast.Name) and tgt.id == "img_shape_content_hw":
                        returns_content = True
                    if isinstance(tgt, ast.Tuple) and any(isinstance(e, ast.Name) and e.id == "img_shape_content_hw" for e in tgt.elts):
                        returns_content = True
    if imports:
        strategy = "C"
    elif probes:
        strategy = "B"
    elif fixed is not None:
        strategy = "A"
    else:
        strategy = "?"
    return {
        "strategy": strategy,
        "probe_functions": sorted(set(probes)),
        "lazy_imports": imports,
        "fixed_hw": fixed,
        "separate_content_shape": returns_content,
    }


def parse_dispatch(medvision_utils_path: str) -> Dict[str, object]:
    with open(medvision_utils_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=medvision_utils_path)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_resized_img_shape":
            fn = node
            break
    if fn is None:
        raise ValueError("get_resized_img_shape not found")

    branches: List[Dict[str, object]] = []
    unparsed_tests: List[str] = []
    fallthrough_raises = False
    probe_defs = sorted(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("_process_img_"))

    # find the first `if` whose test mentions model_name, then follow the elif chain
    head: Optional[ast.If] = None
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and "model_name" in ast.dump(stmt.test):
            head = stmt
            break
    node = head
    while node is not None:
        keys = _keys_from_test(node.test)
        if keys is None:
            unparsed_tests.append(ast.unparse(node.test) if hasattr(ast, "unparse") else "<test>")
            keys = []
        info = _classify_branch(node.body)
        info["keys"] = keys
        info["lineno"] = node.lineno
        branches.append(info)
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else:
            fallthrough_raises = any(isinstance(s, ast.Raise) for s in node.orelse)
            node = None

    key_to_branch: Dict[str, Dict[str, object]] = {}
    for b in branches:
        for k in b["keys"]:  # type: ignore[union-attr]
            key_to_branch[k] = b
    return {
        "function_lineno": fn.lineno,
        "branches": branches,
        "key_to_branch": key_to_branch,
        "unparsed_tests": unparsed_tests,
        "fallthrough_raises": fallthrough_raises,
        "probe_function_defs": probe_defs,
        "returns_tuple": "img_shape_content_hw" in src,
    }


# --------------------------------------------------------------------------- report
def build_report(lmms_eval_dir: str) -> Dict[str, object]:
    models_init = os.path.join(lmms_eval_dir, MODELS_INIT_REL)
    mv_utils = os.path.join(lmms_eval_dir, MEDVISION_UTILS_REL)
    registry = parse_available_models(models_init)
    commented = parse_commented_out_models(models_init)
    dispatch = parse_dispatch(mv_utils)
    key_to_branch: Dict[str, Dict[str, object]] = dispatch["key_to_branch"]  # type: ignore[assignment]

    rows = []
    problems: List[str] = []
    for key, cls in registry.items():
        mod = check_model_module(lmms_eval_dir, key, cls)
        br = key_to_branch.get(key)
        row = {
            "key": key,
            "class": cls,
            "module_exists": mod["module_exists"],
            "register_decorator": mod["register_decorator"],
            "class_defined": mod["class_defined"],
            "dispatch": br is not None,
            "strategy": br["strategy"] if br else None,
            "probe_functions": br["probe_functions"] if br else [],
            "lazy_imports": br["lazy_imports"] if br else [],
            "fixed_hw": br["fixed_hw"] if br else None,
            "separate_content_shape": br["separate_content_shape"] if br else None,
            "branch_lineno": br["lineno"] if br else None,
        }
        rows.append(row)
        if not mod["module_exists"]:
            problems.append(f"{key}: module models/{key}.py missing")
        else:
            if not mod["register_decorator"]:
                problems.append(f"{key}: models/{key}.py has no @register_model(\"{key}\")")
            if not mod["class_defined"]:
                problems.append(f"{key}: class {cls} not defined in models/{key}.py")
        if br is None:
            problems.append(f"{key}: no branch in get_resized_img_shape (TL/AD prompts would raise ValueError)")

    dispatch_only = sorted(k for k in key_to_branch if k not in registry)
    return {
        "lmms_eval_dir": lmms_eval_dir,
        "models_init": models_init,
        "medvision_utils": mv_utils,
        "registered": rows,
        "registered_count": len(rows),
        "dispatch_only_aliases": dispatch_only,
        "commented_out_registry_entries": commented,
        "dispatch": {
            "function_lineno": dispatch["function_lineno"],
            "branch_count": len(dispatch["branches"]),
            "fallthrough_raises": dispatch["fallthrough_raises"],
            "returns_tuple_canvas_and_content": dispatch["returns_tuple"],
            "probe_function_defs": dispatch["probe_function_defs"],
            "unparsed_tests": dispatch["unparsed_tests"],
        },
        "problems": problems,
    }


def expect_checklist(report: Dict[str, object], key: str) -> List[str]:
    reg = {r["key"]: r for r in report["registered"]}  # type: ignore[index]
    lines = []
    missing = []
    if key in reg:
        r = reg[key]
        lines.append(f"[OK]      AVAILABLE_MODELS['{key}'] = '{r['class']}'")
        lines.append(("[OK]      " if r["module_exists"] else "[MISSING] ") + f"models/{key}.py")
        lines.append(("[OK]      " if r["register_decorator"] else "[MISSING] ") + f'@register_model("{key}") in models/{key}.py')
        lines.append(("[OK]      " if r["class_defined"] else "[MISSING] ") + f"class {r['class']} in models/{key}.py")
        lines.append(("[OK]      " if r["dispatch"] else "[MISSING] ") + "branch in get_resized_img_shape()")
        missing = [l for l in lines if l.startswith("[MISSING]")]
    else:
        lines.append(f"[MISSING] AVAILABLE_MODELS has no key '{key}'")
        aliases = report["dispatch_only_aliases"]  # type: ignore[index]
        lines.append(("[OK]      " if key in aliases else "[MISSING] ") + "branch in get_resized_img_shape()")
        missing = [l for l in lines if l.startswith("[MISSING]")]
    lines.append("Also required (not checkable here): benchmark/eval__<key>.py, three launchers, requirements file, resize test.")
    return lines + ([f"RESULT: {len(missing)} site(s) missing"] if missing else ["RESULT: fully wired at the three code sites"])


def print_text(report: Dict[str, object]) -> None:
    print(f"lmms_eval dir : {report['lmms_eval_dir']}")
    print(f"registry      : {report['models_init']}")
    print(f"dispatch      : {report['medvision_utils']} (get_resized_img_shape @ line {report['dispatch']['function_lineno']}, "  # type: ignore[index]
          f"{report['dispatch']['branch_count']} branches, else-raises={report['dispatch']['fallthrough_raises']}, "  # type: ignore[index]
          f"returns (canvas, content)={report['dispatch']['returns_tuple_canvas_and_content']})")  # type: ignore[index]
    print()
    hdr = f"{'key':<26}{'class':<26}{'module':<8}{'@reg':<6}{'class?':<8}{'branch':<8}{'strat':<6}rule"
    print(hdr)
    print("-" * len(hdr))
    for r in report["registered"]:  # type: ignore[index]
        if r["strategy"] == "A":
            rule = f"fixed {r['fixed_hw']}"
        elif r["strategy"] == "B":
            rule = ",".join(r["probe_functions"]) or "probe"
            if r["separate_content_shape"]:
                rule += " (+content shape)"
        elif r["strategy"] == "C":
            rule = ";".join(r["lazy_imports"])
        else:
            rule = "-"
        print(f"{r['key']:<26}{r['class']:<26}{'yes' if r['module_exists'] else 'NO':<8}{'yes' if r['register_decorator'] else 'NO':<6}"
              f"{'yes' if r['class_defined'] else 'NO':<8}{'yes' if r['dispatch'] else 'NO':<8}{(r['strategy'] or '-'):<6}{rule}")
    print()
    print(f"registered keys           : {report['registered_count']}")
    print(f"dispatch-only aliases     : {', '.join(report['dispatch_only_aliases']) or '(none)'}  "  # type: ignore[index]
          "(SFT model_family_name strings / HF-backend aliases; not runnable via --model)")
    print(f"commented-out registry    : {', '.join(report['commented_out_registry_entries']) or '(none)'}")  # type: ignore[index]
    print(f"_process_img_* defined    : {', '.join(report['dispatch']['probe_function_defs'])}")  # type: ignore[index]
    if report["dispatch"]["unparsed_tests"]:  # type: ignore[index]
        print(f"non-literal branch tests  : {report['dispatch']['unparsed_tests']}")  # type: ignore[index]
    print()
    if report["problems"]:  # type: ignore[index]
        print("PROBLEMS:")
        for p in report["problems"]:  # type: ignore[index]
            print(f"  - {p}")
    else:
        print("No mismatches: every registered key has a module, a matching @register_model decorator and a dispatch branch.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=None, help="MedVision checkout root (uses <root>/src/medvision_bm/medvision_lmms_eval/lmms_eval).")
    ap.add_argument("--lmms-eval-dir", default=None, help="Explicit path to the vendored lmms_eval package directory.")
    ap.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    ap.add_argument("--expect", metavar="KEY", default=None, help="Print a wiring checklist for KEY and exit 1 if any code site is missing.")
    args = ap.parse_args(argv)

    lmms_eval_dir = locate_lmms_eval_dir(args.repo_root, args.lmms_eval_dir)
    if lmms_eval_dir is None:
        print("ERROR: could not locate the vendored lmms_eval tree. Pass --repo-root <repo> or --lmms-eval-dir, or install medvision_bm.", file=sys.stderr)
        return 2
    try:
        report = build_report(lmms_eval_dir)
    except (SyntaxError, ValueError, OSError) as exc:
        print(f"ERROR: failed to parse the registry/dispatch sources: {exc}", file=sys.stderr)
        return 2

    if args.expect:
        lines = expect_checklist(report, args.expect)
        if args.json:
            print(json.dumps({"expect": args.expect, "checklist": lines}, indent=2))
        else:
            print("\n".join(lines))
        return 0 if lines[-1].startswith("RESULT: fully") else 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text(report)
    return 1 if report["problems"] else 0  # type: ignore[index]


if __name__ == "__main__":
    sys.exit(main())
