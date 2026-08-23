"""
Analyze equation computing accuracy for A/D (Angle & Distance) task model responses.

For each sample, extracts the equation written in step-3-reasoning, evaluates it
in Python, and compares the Python-evaluated result with the model's step-3-answer.

    equation_MRE = abs(model_answer - python_eval) / (|python_eval| + 1e-15)

This measures arithmetic correctness independent of GT: did the model correctly
compute the equation it wrote down, given the landmark coordinates it estimated?

Usage:
    python analyze_equation_accuracy_AD.py \\
        --task_dir /path/to/MedVision-AD-v2-CoT \\
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

from medvision_bm.utils.configs import AD_NEAR_ZERO_GT_THRESHOLD
from medvision_bm.utils.tool_execution import safe_exec_python

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def _cal_equation_MRE(model_val, python_val):
    return float(abs(model_val - python_val) / (abs(python_val) + 1e-15))


# ---------------------------------------------------------------------------
# AST-based expression evaluator (restricted to numeric math only)
# ---------------------------------------------------------------------------

_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
}

_MATH_FUNCS = {
    "sqrt": _math.sqrt,
    "acos": _math.acos,
    "asin": _math.asin,
    "atan": _math.atan,
    "atan2": _math.atan2,
    "degrees": _math.degrees,
    "abs": abs,
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
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        fn = _BINOPS.get(type(node.op))
        if fn is not None:
            return fn(left, right)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr  # math.sqrt -> "sqrt"
        elif isinstance(node.func, ast.Name):
            fn_name = node.func.id  # abs -> "abs"
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
    "arccos": "math.acos",
    "arcsin": "math.asin",
    "arctan2": "math.atan2",
    "arctan": "math.atan",
    "sqrt": "math.sqrt",
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


def _to_python_expr(raw, wrap_degrees=False):
    """Convert math-notation string to a Python-evaluable expression string."""
    expr = _convert_abs_notation(raw)
    # Parenthesize bare numbers (possibly negative) before ^ to fix Python precedence.
    # -24.000^2 must become (-24.000)**2, not -(24.000**2).
    expr = re.sub(r"(-?\d+(?:\.\d+)?)\^", r"(\1)**", expr)
    # Remaining ^ (e.g., after closing paren: (...)^2 -> (...)**2)
    expr = re.sub(r"\^", r"**", expr)
    for fn, py_fn in _FUNC_MAP.items():
        expr = expr.replace(fn + "(", py_fn + "(")
    if wrap_degrees:
        expr = f"math.degrees({expr})"
    return expr


# ---------------------------------------------------------------------------
# Step block extraction
# ---------------------------------------------------------------------------

_NNR = r"\d+(?:\.\d+)?"
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
# Tooluse helpers
# ---------------------------------------------------------------------------


def _extract_tool_call_code(solution):
    m = re.search(r"<tool_call>(.*?)</tool_call>", solution, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(1))
        return (parsed.get("arguments") or {}).get("code")
    except (json.JSONDecodeError, AttributeError):
        return None


def _exec_tool_and_parse(solution):
    """Execute <tool_call> code; return (python_eval, model_answer) from <answer> tag."""
    code = _extract_tool_call_code(solution)
    if code is None:
        return None, None
    stdout = safe_exec_python(code)
    if not stdout or stdout.startswith("ERROR:"):
        return None, None
    m_ans = re.search(r"<answer>(.*?)</answer>", solution, re.DOTALL)
    if not m_ans:
        return None, None
    py_nums = [
        s.replace(",", "")
        for s in re.findall(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", stdout
        )
    ]
    ans_nums = [
        s.replace(",", "")
        for s in re.findall(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?",
            m_ans.group(1),
        )
    ]
    if not py_nums or not ans_nums:
        return None, None
    try:
        return float(py_nums[0]), float(ans_nums[0])
    except ValueError:
        return None, None


# ---------------------------------------------------------------------------
# Per-sample analyzers
# ---------------------------------------------------------------------------


def analyze_distance_sample(solution):
    """Evaluate the distance equation in step-3-reasoning vs step-3-answer."""
    reasoning = _extract_reasoning_block(solution, 3)
    model_val = _extract_step_answer(solution, 3)

    result = {
        "metric_type": "distance",
        "step3_model_answer": model_val,
        "step3_raw_expr": None,
        "step3_python_eval": None,
        "step3_equation_MRE": None,
    }
    if reasoning is None:
        return result

    raw = _extract_func_call(reasoning, "sqrt")
    result["step3_raw_expr"] = raw
    if raw is None or model_val is None:
        py_val, tu_val = _exec_tool_and_parse(solution)
        if py_val is not None and tu_val is not None:
            result["step3_model_answer"] = tu_val
            result["step3_python_eval"] = py_val
            result["step3_equation_MRE"] = _cal_equation_MRE(tu_val, py_val)
        return result

    try:
        py_expr = _to_python_expr(raw)
        py_val = _compute_expr(py_expr)
        result["step3_python_eval"] = float(py_val)
        result["step3_equation_MRE"] = _cal_equation_MRE(model_val, py_val)
    except Exception as e:
        result["step3_eval_error"] = str(e)
    return result


def analyze_angle_sample(solution):
    """Evaluate the arccos equation in step-3-reasoning vs step-3-answer (degrees)."""
    reasoning = _extract_reasoning_block(solution, 3)
    model_val = _extract_step_answer(solution, 3)

    result = {
        "metric_type": "angle",
        "step3_model_answer": model_val,
        "step3_raw_expr": None,
        "step3_python_eval": None,
        "step3_equation_MRE": None,
    }
    if reasoning is None:
        return result

    raw = _extract_func_call(reasoning, "arccos")
    result["step3_raw_expr"] = raw
    if raw is None or model_val is None:
        py_val, tu_val = _exec_tool_and_parse(solution)
        if py_val is not None and tu_val is not None:
            result["step3_model_answer"] = tu_val
            result["step3_python_eval"] = py_val
            result["step3_equation_MRE"] = _cal_equation_MRE(tu_val, py_val)
        return result

    try:
        py_expr = _to_python_expr(raw, wrap_degrees=True)
        py_val = _compute_expr(py_expr)
        result["step3_python_eval"] = float(py_val)
        result["step3_equation_MRE"] = _cal_equation_MRE(model_val, py_val)
    except Exception as e:
        result["step3_eval_error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Process a single JSONL file
# ---------------------------------------------------------------------------


def process_jsonl(jsonl_path, output_suffix):
    jsonl_path = Path(jsonl_path)
    out_path = jsonl_path.with_name(jsonl_path.stem + output_suffix + ".jsonl")

    n_total = n_distance = n_angle = n_parse_fail = 0
    results = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            n_total += 1

            doc = sample.get("doc", {})
            doc_id = sample.get("doc_id")
            solution = sample.get("resps", [[""]])[0]
            if isinstance(solution, list):
                solution = solution[0] if solution else ""

            bp = doc.get("biometric_profile", {})
            metric_type = bp.get("metric_type", "")

            record = {
                "doc_id": doc_id,
                "dataset": doc.get("dataset_name"),
                "metric_type": metric_type,
                "metric_key": bp.get("metric_key"),
                "image_file": doc.get("image_file"),
            }

            try:
                if metric_type == "distance":
                    n_distance += 1
                    record.update(analyze_distance_sample(solution))
                elif metric_type == "angle":
                    n_angle += 1
                    record.update(analyze_angle_sample(solution))
                else:
                    n_parse_fail += 1
                    record["error"] = f"unknown metric_type: {metric_type!r}"
            except Exception as e:
                n_parse_fail += 1
                record["error"] = str(e)

            results.append(record)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def _stats(key, mtype):
        vals = [
            r[key]
            for r in results
            if r.get("metric_type") == mtype and r.get(key) is not None
        ]
        if not vals:
            return None, None, 0
        return float(np.mean(vals)), float(np.std(vals)), len(vals)

    print(f"\n[{jsonl_path.name}]")
    print(
        f"  Total: {n_total}  "
        f"(distance={n_distance}, angle={n_angle}, parse_fail={n_parse_fail})"
    )
    for key, label, mtype in [
        ("step3_equation_MRE", "Step3 equation MRE (distance)", "distance"),
        ("step3_equation_MRE", "Step3 equation MRE (angle)", "angle"),
    ]:
        n_mtype = n_distance if mtype == "distance" else n_angle
        if n_mtype == 0:
            continue
        mean, sd, n = _stats(key, mtype)
        fail = n_mtype - n
        mean_str = f"{mean:.4f}" if mean is not None else "nan"
        sd_str = f"{sd:.4f}" if sd is not None else "nan"
        print(
            f"  [{mtype}] {label}: mean={mean_str} ± sd={sd_str} (n={n}, fail={fail})"
        )
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
            if "_eq_acc" not in p.stem and "_proc_acc" not in p.stem:
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

SUMMARY_EQ_ACC_AD_METRICS_FILENAME = "summary_eq_acc_AD_metrics.json"
SUMMARY_EQ_ACC_AD_MODEL_FILENAME = "summary_eq_acc_AD_model.txt"


def _get_ad_label(record):
    dataset = record.get("dataset", "")
    metric_type = record.get("metric_type", "")
    metric_key = record.get("metric_key", "")
    if not (dataset and metric_type and metric_key):
        return None
    return f"{dataset}_{metric_type}_{metric_key}"


def _aggregate_by_label_AD(all_results):
    """Aggregate per-sample results by label; return {label: averaged equation MRE}."""
    grouped = {}
    for r in all_results:
        label = _get_ad_label(r)
        if label is None:
            continue
        if label not in grouped:
            grouped[label] = {
                "metric_type": r.get("metric_type"),
                "eq_mre": [],
                "n_samples": 0,
                "n_ignored": 0,
            }
        g = grouped[label]
        g["n_samples"] += 1
        v = r.get("step3_equation_MRE")
        if v is not None:
            py_val = r.get("step3_python_eval")
            if py_val is not None and py_val < AD_NEAR_ZERO_GT_THRESHOLD:
                g["n_ignored"] += 1
            else:
                g["eq_mre"].append(v)

    def _avg(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        label: {
            "metric_type": g["metric_type"],
            "step3_avg_equation_MRE": _avg(g["eq_mre"]),
            "n_samples": g["n_samples"],
            "n_valid": len(g["eq_mre"]),
            "n_ignored": g["n_ignored"],
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
    jsonl_paths = [
        p
        for p in sorted(parsed_dir.glob("*.jsonl"))
        if "_eq_acc" not in p.stem and "_proc_acc" not in p.stem
    ]
    if not jsonl_paths:
        print(f"[skip] no JSONL files in: {parsed_dir}")
        return None

    print(f"\nProcessing model: {model_dir.name}")
    all_results = []
    for jp in jsonl_paths:
        all_results.extend(process_jsonl(jp, output_suffix))

    summary = _aggregate_by_label_AD(all_results)
    out_path = parsed_dir / SUMMARY_EQ_ACC_AD_METRICS_FILENAME
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] per-label summary → {out_path}")
    _print_model_summary_AD(model_dir, summary)
    return summary


def _print_model_summary_AD(model_dir, summary):
    """Write a weighted-average + group + label-table summary TXT for a single model."""
    model_dir = Path(model_dir)
    out_path = model_dir / SUMMARY_EQ_ACC_AD_MODEL_FILENAME
    lines = []

    def _p(text):
        print(text)
        lines.append(text)

    _p(f"\nModel: {model_dir.name}")

    total_n = wsum = wcount = 0
    groups = {"FeTA-Distance": [], "Ceph-Angle": [], "Ceph-Distance": [], "Other": []}
    for label, lm in summary.items():
        n_valid = lm.get("n_valid", 0)
        if n_valid <= 0:
            continue
        total_n += lm.get("n_samples", 0)
        groups[_group_classify_AD(label)].append(lm)
        v = lm.get("step3_avg_equation_MRE", float("nan"))
        if v is not None and not np.isnan(v):
            wsum += v * n_valid
            wcount += n_valid

    overall = wsum / wcount if wcount > 0 else float("nan")
    _p(
        f"Weighted Average → Step3_eq_MRE: {overall:.4f} (Valid: {wcount}, Total: {total_n})"
    )

    _p("\nGroup averages:")
    _p(f"{'Group':<15} | {'Step3_eq_MRE':<14} | {'Valid':<8} | {'Samples':<8}")
    _p("-" * 53)
    for gname, glist in _group_rows_AD(groups):
        ga = _calc_group_avg_AD(glist)
        _p(
            f"{gname:<15} | {ga['step3_avg_equation_MRE']:<14.4f} | {ga.get('n_valid', 0):<8} | {ga['n_samples']:<8}"
        )

    _p("\nLabel-specific metrics:")
    _p(
        f"{'Label':<50} | {'Type':<8} | {'Step3_eq_MRE':<14} | {'Valid':<6} | {'Ignored':<7} | {'Samples':<8}"
    )
    _p("-" * 112)
    for label, lm in sorted(
        summary.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True
    ):
        _p(
            f"{label:<50} | "
            f"{lm.get('metric_type', ''):<8} | "
            f"{lm.get('step3_avg_equation_MRE', float('nan')):<14.4f} | "
            f"{lm.get('n_valid', 0):<6} | "
            f"{lm.get('n_ignored', 0):<7} | "
            f"{lm.get('n_samples', 0):<8}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [saved] model summary → {out_path}")


def _group_classify_AD(label):
    if "FeTA24_distance" in label:
        return "FeTA-Distance"
    if "Ceph-Biometrics-400_angle" in label:
        return "Ceph-Angle"
    if "Ceph-Biometrics-400_distance" in label:
        return "Ceph-Distance"
    return "Other"


def _calc_group_avg_AD(label_metrics_list):
    def _wavg(key):
        s, n = 0.0, 0
        for m in label_metrics_list:
            v = m.get(key, float("nan"))
            if v is not None and not np.isnan(v):
                s += v * m.get("n_valid", 0)
                n += m.get("n_valid", 0)
        return s / n if n > 0 else float("nan")

    return {
        "step3_avg_equation_MRE": _wavg("step3_avg_equation_MRE"),
        "n_valid": sum(m.get("n_valid", 0) for m in label_metrics_list),
        "n_samples": sum(m.get("n_samples", 0) for m in label_metrics_list),
    }


def _group_rows_AD(groups):
    """Ordered (display name, label list) rows, incl. cross-dataset aggregates.

    The last two rows aggregate across datasets by metric type, matching the
    "Distance"/"Angle" grouping that split_ad_labels() in
    script/visualization/viz_radar.py uses.
    """
    return (
        ("FeTA-Distance", groups["FeTA-Distance"]),
        ("Ceph-Angle", groups["Ceph-Angle"]),
        ("Ceph-Distance", groups["Ceph-Distance"]),
        ("Distance", groups["FeTA-Distance"] + groups["Ceph-Distance"]),
        ("Angle", groups["Ceph-Angle"]),
    )


def _print_cross_model_summaries_AD(task_dir):
    """Read per-model summary JSONs, print group/label tables, save summary TXT."""
    task_dir = Path(task_dir)
    out_path = task_dir / "summary_eq_acc_AD_task.txt"
    lines = []

    def _p(text):
        print(text)
        lines.append(text)

    _p("\n\n========== MODEL SUMMARIES (Equation Accuracy - AD Task) ==========\n")

    for model_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        summary_json = model_dir / "parsed" / SUMMARY_EQ_ACC_AD_METRICS_FILENAME
        if not summary_json.exists():
            continue
        with open(summary_json) as f:
            metrics = json.load(f)

        _p(f"\nModel: {model_dir.name}")

        total_n = 0
        wsum, wcount = 0.0, 0
        groups = {
            "FeTA-Distance": [],
            "Ceph-Angle": [],
            "Ceph-Distance": [],
            "Other": [],
        }

        for label, lm in metrics.items():
            n_valid = lm.get("n_valid", 0)
            if n_valid <= 0:
                continue
            total_n += lm.get("n_samples", 0)
            groups[_group_classify_AD(label)].append(lm)
            v = lm.get("step3_avg_equation_MRE", float("nan"))
            if v is not None and not np.isnan(v):
                wsum += v * n_valid
                wcount += n_valid

        overall = wsum / wcount if wcount > 0 else float("nan")
        _p(
            f"Weighted Average → Step3_eq_MRE: {overall:.4f} (Valid: {wcount}, Total: {total_n})"
        )

        _p("\nGroup averages:")
        _p(f"{'Group':<15} | {'Step3_eq_MRE':<14} | {'Valid':<8} | {'Samples':<8}")
        _p("-" * 53)
        for gname, glist in _group_rows_AD(groups):
            ga = _calc_group_avg_AD(glist)
            _p(
                f"{gname:<15} | {ga['step3_avg_equation_MRE']:<14.4f} | {ga.get('n_valid', 0):<8} | {ga['n_samples']:<8}"
            )

        _p("\nLabel-specific metrics:")
        _p(
            f"{'Label':<50} | {'Type':<8} | {'Step3_eq_MRE':<14} | {'Valid':<6} | {'Ignored':<7} | {'Samples':<8}"
        )
        _p("-" * 112)
        for label, lm in sorted(
            metrics.items(), key=lambda x: x[1].get("n_samples", 0), reverse=True
        ):
            _p(
                f"{label:<50} | "
                f"{lm.get('metric_type', ''):<8} | "
                f"{lm.get('step3_avg_equation_MRE', float('nan')):<14.4f} | "
                f"{lm.get('n_valid', 0):<6} | "
                f"{lm.get('n_ignored', 0):<7} | "
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
        description="Analyze equation computing accuracy for A/D task JSONL files."
    )
    parser.add_argument(
        "--task_dir",
        default=None,
        help="Task results directory; each immediate subdir is a model folder with parsed/ subdir.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Single model directory containing a parsed/ subfolder with JSONL files.",
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=None,
        help="One or more explicit JSONL file paths (or glob patterns) to analyze.",
    )
    parser.add_argument(
        "--output_suffix",
        default="_eq_acc",
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
        print(
            f"[Info] Discovered {len(model_dirs)} model dir(s) under: {args.task_dir}"
        )
        for model_dir in model_dirs:
            _process_model_dir(model_dir, args.output_suffix)
        _print_cross_model_summaries_AD(args.task_dir)


if __name__ == "__main__":
    main()
