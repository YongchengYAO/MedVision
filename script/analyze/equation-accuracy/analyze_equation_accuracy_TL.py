"""
Analyze equation computing accuracy for T/L (Tumor/Lesion size) task model responses.

For each sample, extracts the equations written in step-3-reasoning (major axis)
and step-4-reasoning (minor axis), evaluates them in Python, and compares the
Python-evaluated results with the model's step-k-answer values.

    equation_MRE = abs(model_answer - python_eval) / (|python_eval| + 1e-15)

This measures arithmetic correctness independent of GT: did the model correctly
compute the distance formula it wrote down?

Usage:
    python analyze_equation_accuracy_TL.py \\
        --task_dir /path/to/MedVision-TL-v2-CoT \\
        [--model_dir <model_dir>] \\
        [--jsonl path/to/explicit.jsonl ...] \\
        [--output_suffix _eq_acc]
"""

import argparse
import ast
import json
import math as _math
import operator as _op
import re
import sys
from pathlib import Path

import numpy as np

from medvision_bm.utils.configs import label_map_rename
from medvision_bm.utils.parse_utils import (
    get_labelsMap_imgModality_from_biometry_benchmark_plan,
    get_targetLabel_imgModality_from_biometry_benchmark_plan,
)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _cal_equation_MRE(model_val, python_val):
    return float(abs(model_val - python_val) / (abs(python_val) + 1e-15))


# ---------------------------------------------------------------------------
# AST-based expression evaluator (restricted to numeric math only)
# ---------------------------------------------------------------------------

_BINOPS = {
    ast.Add:  _op.add,
    ast.Sub:  _op.sub,
    ast.Mult: _op.mul,
    ast.Div:  _op.truediv,
    ast.Pow:  _op.pow,
}

_MATH_FUNCS = {
    "sqrt":    _math.sqrt,
    "acos":    _math.acos,
    "asin":    _math.asin,
    "atan":    _math.atan,
    "atan2":   _math.atan2,
    "degrees": _math.degrees,
    "abs":     abs,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
    if isinstance(node, ast.BinOp):
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        fn = _BINOPS.get(type(node.op))
        if fn is not None:
            return fn(left, right)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr   # math.sqrt -> "sqrt"
        elif isinstance(node.func, ast.Name):
            fn_name = node.func.id     # abs -> "abs"
        else:
            raise ValueError(f"Unsupported call node: {ast.dump(node)}")
        fn = _MATH_FUNCS.get(fn_name)
        if fn is None:
            raise ValueError(f"Disallowed function: {fn_name!r}")
        args = [_eval_node(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _compute_expr(py_expr):
    """Parse and compute a numeric math expression via ast.parse."""
    tree = ast.parse(py_expr, mode="eval")
    return _eval_node(tree.body)


# ---------------------------------------------------------------------------
# Equation parsing utilities
# ---------------------------------------------------------------------------

_FUNC_MAP = {
    "arccos":  "math.acos",
    "arcsin":  "math.asin",
    "arctan2": "math.atan2",
    "arctan":  "math.atan",
    "sqrt":    "math.sqrt",
}


def _extract_func_call(text, func_name):
    """Extract the last balanced func_name(...) expression from text."""
    idx = text.rfind(func_name + "(")
    if idx == -1:
        return None
    start = idx + len(func_name)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return None


def _convert_abs_notation(expr):
    """Convert |...| absolute-value bars to abs(...). Handles one nesting level."""
    result = []
    open_abs = False
    for ch in expr:
        if ch == "|":
            if not open_abs:
                result.append("abs(")
                open_abs = True
            else:
                result.append(")")
                open_abs = False
        else:
            result.append(ch)
    return "".join(result)


def _to_python_expr(raw):
    """Convert math-notation distance formula to a Python-evaluable expression string."""
    expr = _convert_abs_notation(raw)
    # Parenthesize bare numbers (possibly negative) before ^ to fix Python precedence.
    expr = re.sub(r"(-?\d+(?:\.\d+)?)\^", r"(\1)**", expr)
    # Remaining ^ (after closing paren: (...)^2 -> (...)**2)
    expr = re.sub(r"\^", r"**", expr)
    for fn, py_fn in _FUNC_MAP.items():
        expr = expr.replace(fn + "(", py_fn + "(")
    return expr


# ---------------------------------------------------------------------------
# Step block extraction
# ---------------------------------------------------------------------------

_NNR  = r"\d+(?:\.\d+)?"
_NNRG = rf"({_NNR})"


def _extract_reasoning_block(solution, k):
    """Return the content of <step-k-reasoning>...</step-k-reasoning>."""
    m = re.search(
        rf"<step-{k}-reasoning>(.*?)</step-{k}-reasoning>", solution, re.DOTALL
    )
    return m.group(1).strip() if m else None


def _extract_step_answer(solution, k):
    """Return the first numeric value inside <step-k-answer>...</step-k-answer>."""
    m = re.search(
        rf"<step-{k}-answer>.*?{_NNRG}.*?</step-{k}-answer>", solution, re.DOTALL
    )
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-sample analyzer
# ---------------------------------------------------------------------------

def analyze_tl_sample(solution):
    """Evaluate the distance equations in steps 3 and 4 vs model's stated answers."""
    result = {"metric_type": "tl"}

    for step, axis in [(3, "major"), (4, "minor")]:
        model_val = _extract_step_answer(solution, step)
        reasoning = _extract_reasoning_block(solution, step)

        result[f"step{step}_model_answer"] = model_val
        result[f"step{step}_raw_expr"]     = None
        result[f"step{step}_python_eval"]  = None
        result[f"step{step}_equation_MRE"] = None

        if reasoning is None:
            continue

        raw = _extract_func_call(reasoning, "sqrt")
        result[f"step{step}_raw_expr"] = raw
        if raw is None or model_val is None:
            continue

        try:
            py_expr = _to_python_expr(raw)
            py_val  = _compute_expr(py_expr)
            result[f"step{step}_python_eval"]  = float(py_val)
            result[f"step{step}_equation_MRE"] = _cal_equation_MRE(model_val, py_val)
        except Exception as e:
            result[f"step{step}_eval_error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Process a single JSONL file
# ---------------------------------------------------------------------------

def process_jsonl(jsonl_path, output_suffix):
    jsonl_path = Path(jsonl_path)
    out_path   = jsonl_path.with_name(jsonl_path.stem + output_suffix + ".jsonl")

    n_total = n_tl = n_gt_fail = n_parse_fail = n_success = 0
    results = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            n_total += 1

            doc      = sample.get("doc", {})
            doc_id   = sample.get("doc_id")
            solution = sample.get("resps", [[""]])[0]
            if isinstance(solution, list):
                solution = solution[0] if solution else ""

            task_type = doc.get("taskType", "")
            if "Tumor" not in task_type and "Lesion" not in task_type:
                n_gt_fail += 1
                results.append({
                    "doc_id": doc_id,
                    "error": f"unexpected taskType: {task_type!r}",
                })
                continue

            n_tl += 1
            record = {
                "doc_id":    doc_id,
                "dataset":   doc.get("dataset_name"),
                "taskID":    doc.get("taskID"),
                "taskType":  task_type,
                "image_file": doc.get("image_file"),
                "slice_dim": doc.get("slice_dim"),
            }

            try:
                record.update(analyze_tl_sample(solution))
                if all(
                    record.get(f"step{k}_equation_MRE") is not None
                    for k in (3, 4)
                ):
                    n_success += 1
            except Exception as e:
                n_parse_fail += 1
                record["error"] = str(e)

            results.append(record)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def _stats(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return None, None, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    success_str = (
        f", success_rate={100*n_success/n_tl:.1f}% {n_success}/{n_tl}"
        if n_tl > 0 else ""
    )
    print(f"\n[{jsonl_path.name}]")
    print(
        f"  Total: {n_total}  "
        f"(tl={n_tl}, task_type_fail={n_gt_fail}, parse_fail={n_parse_fail}"
        f"{success_str})"
    )
    for key, label in [
        ("step3_equation_MRE", "Step3 equation MRE (major axis)"),
        ("step4_equation_MRE", "Step4 equation MRE (minor axis)"),
    ]:
        if n_tl == 0:
            continue
        mean, sd, n = _stats(key)
        fail     = n_tl - n
        mean_str = f"{mean:.4f}" if mean is not None else "nan"
        sd_str   = f"{sd:.4f}"   if sd   is not None else "nan"
        print(f"  {label}: mean={mean_str} ± sd={sd_str} (n={n}, fail={fail})")
    print(f"  Output: {out_path}")
    return results


# ---------------------------------------------------------------------------
# Path discovery helpers
# ---------------------------------------------------------------------------

def _collect_from_model_dir(model_dir):
    paths = []
    parsed_dir = Path(model_dir) / "parsed"
    if parsed_dir.is_dir():
        for p in sorted(parsed_dir.glob("*.jsonl")):
            if "_eq_acc" not in p.stem:
                paths.append(p)
    return paths


def _collect_from_task_dir(task_dir):
    paths = []
    for model_dir in sorted(Path(task_dir).iterdir()):
        paths.extend(_collect_from_model_dir(model_dir))
    return paths


def _collect_from_jsonl_args(jsonl_args):
    paths = []
    for pattern in jsonl_args:
        if "*" in pattern:
            paths.extend(sorted(Path(".").glob(pattern)))
        else:
            paths.append(Path(pattern))
    return [p for p in paths if p.exists()]


# ---------------------------------------------------------------------------
# Per-model aggregation and cross-model summary
# ---------------------------------------------------------------------------

SUMMARY_EQ_ACC_TL_METRICS_FILENAME = "summary_eq_acc_TL_metrics.json"

_IMGMOD_MAP = {"MRI": "MR", "CT": "CT", "ultrasound": "US", "X-ray": "XR", "PET": "PET"}
_SLICE_MAP = {0: "S", 1: "C", 2: "A"}


def _get_tl_label(record):
    """Derive anatomy label key from a TL record (requires dataset, taskID, slice_dim)."""
    dataset = record.get("dataset")
    task_id = record.get("taskID")
    slice_dim = record.get("slice_dim")
    if dataset is None or task_id is None or slice_dim is None:
        return None
    try:
        label, _ = get_targetLabel_imgModality_from_biometry_benchmark_plan(dataset, int(task_id))
        labels_map, img_modality = get_labelsMap_imgModality_from_biometry_benchmark_plan(dataset, int(task_id))
        label_name = labels_map.get(str(label))
        if label_name is None:
            return None
        new_label = label_map_rename.get(label_name)
        if new_label is None:
            return None
        img_mod = _IMGMOD_MAP.get(img_modality, img_modality)
        slicetype = _SLICE_MAP.get(int(slice_dim))
        if slicetype is None:
            return None
        return f"{new_label} @ {img_mod} ({slicetype})"
    except Exception:
        return None


def _aggregate_by_label_TL(all_results):
    """Aggregate per-sample results by anatomy label; return {label: averaged equation MREs}."""
    grouped = {}
    for r in all_results:
        if r.get("error"):
            continue
        label = _get_tl_label(r)
        if label is None:
            continue
        if label not in grouped:
            grouped[label] = {"eq3": [], "eq4": [], "n_samples": 0}
        g = grouped[label]
        g["n_samples"] += 1
        v3 = r.get("step3_equation_MRE")
        v4 = r.get("step4_equation_MRE")
        if v3 is not None: g["eq3"].append(v3)
        if v4 is not None: g["eq4"].append(v4)

    def _avg(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        label: {
            "step3_avg_equation_MRE": _avg(g["eq3"]),
            "step4_avg_equation_MRE": _avg(g["eq4"]),
            "n_samples": g["n_samples"],
            "n_valid_3": len(g["eq3"]),
            "n_valid_4": len(g["eq4"]),
        }
        for label, g in grouped.items()
    }


def _process_model_dir(model_dir, output_suffix):
    """Process all JSONL files in model's parsed/ dir; save per-label summary JSON."""
    model_dir = Path(model_dir)
    parsed_dir = model_dir / "parsed"
    if not parsed_dir.is_dir():
        print(f"[skip] no parsed/ dir: {model_dir}")
        return None
    jsonl_paths = [p for p in sorted(parsed_dir.glob("*.jsonl")) if output_suffix not in p.stem]
    if not jsonl_paths:
        print(f"[skip] no JSONL files in: {parsed_dir}")
        return None

    print(f"\nProcessing model: {model_dir.name}")
    all_results = []
    for jp in jsonl_paths:
        all_results.extend(process_jsonl(jp, output_suffix))

    summary = _aggregate_by_label_TL(all_results)
    out_path = parsed_dir / SUMMARY_EQ_ACC_TL_METRICS_FILENAME
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] per-label summary → {out_path}")
    return summary


def _print_cross_model_summaries_TL(task_dir):
    """Read per-model summary JSONs, print label table, save summary TXT."""
    task_dir = Path(task_dir)
    out_path = task_dir / "summary_eq_acc_TL_task.txt"
    lines = []

    def _p(text):
        print(text)
        lines.append(text)

    _p("\n\n========== MODEL SUMMARIES (Equation Accuracy - TL Task) ==========\n")

    for model_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        summary_json = model_dir / "parsed" / SUMMARY_EQ_ACC_TL_METRICS_FILENAME
        if not summary_json.exists():
            continue
        with open(summary_json) as f:
            metrics = json.load(f)

        _p(f"\nModel: {model_dir.name}")

        total_n = 0
        wsum3 = wsum4 = 0.0
        wcount3 = wcount4 = 0

        for label, lm in metrics.items():
            n = lm.get("n_samples", 0)
            if n <= 0:
                continue
            total_n += n
            v3 = lm.get("step3_avg_equation_MRE", float("nan"))
            v4 = lm.get("step4_avg_equation_MRE", float("nan"))
            if v3 is not None and not np.isnan(v3):
                wsum3 += v3 * n; wcount3 += n
            if v4 is not None and not np.isnan(v4):
                wsum4 += v4 * n; wcount4 += n

        avg3 = wsum3 / wcount3 if wcount3 > 0 else float("nan")
        avg4 = wsum4 / wcount4 if wcount4 > 0 else float("nan")
        _p(f"Weighted Average → Step3_eq_MRE: {avg3:.4f}, Step4_eq_MRE: {avg4:.4f} (Total Samples: {total_n})")

        _p("\nLabel-specific metrics:")
        _p(f"{'Label':<52} | {'S3_eq_MRE':<12} | {'S4_eq_MRE':<12} | {'Valid3':<7} | {'Valid4':<7} | {'Samples':<8}")
        _p("-" * 110)
        for label, lm in sorted(metrics.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True):
            _p(
                f"{label:<52} | "
                f"{lm.get('step3_avg_equation_MRE', float('nan')):<12.4f} | "
                f"{lm.get('step4_avg_equation_MRE', float('nan')):<12.4f} | "
                f"{lm.get('n_valid_3', 0):<7} | "
                f"{lm.get('n_valid_4', 0):<7} | "
                f"{lm.get('n_samples', 0):<8}"
            )
        _p("\n" + "=" * 100 + "\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze equation computing accuracy for T/L task JSONL files."
    )
    parser.add_argument(
        "--task_dir", default=None,
        help="Task results directory; each immediate subdir is a model folder with parsed/ subdir.",
    )
    parser.add_argument(
        "--model_dir", default=None,
        help="Single model directory containing a parsed/ subfolder with JSONL files.",
    )
    parser.add_argument(
        "--jsonl", nargs="+", default=None,
        help="One or more explicit JSONL file paths (or glob patterns) to analyze.",
    )
    parser.add_argument(
        "--output_suffix", default="_eq_acc",
        help="Suffix appended before .jsonl in the output filename (default: _eq_acc).",
    )
    args = parser.parse_args()

    if args.task_dir is None and args.model_dir is None and args.jsonl is None:
        parser.error("Provide at least one of --task_dir, --model_dir, or --jsonl.")

    if args.jsonl:
        paths = _collect_from_jsonl_args(args.jsonl)
        if not paths:
            print("Error: no valid JSONL files found.", file=sys.stderr)
            sys.exit(1)
        for jp in paths:
            process_jsonl(jp, args.output_suffix)

    if args.model_dir:
        print(f"[Info] Processing model dir: {args.model_dir}")
        _process_model_dir(args.model_dir, args.output_suffix)

    if args.task_dir:
        model_dirs = sorted(d for d in Path(args.task_dir).iterdir() if d.is_dir())
        print(f"[Info] Discovered {len(model_dirs)} model dir(s) under: {args.task_dir}")
        for model_dir in model_dirs:
            _process_model_dir(model_dir, args.output_suffix)
        _print_cross_model_summaries_TL(args.task_dir)


if __name__ == "__main__":
    main()
