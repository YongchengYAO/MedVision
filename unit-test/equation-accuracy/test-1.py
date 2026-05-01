print("=== Test 1: _extract_func_call, _convert_abs_notation, _to_python_expr ===")
print("Objective : Verify the three equation-parsing helpers in analyze_equation_accuracy_AD.py.")
print("Expected  :")
print("  _extract_func_call  -> extracts last balanced func(...) with nested parens")
print("  _convert_abs_notation -> |...| -> abs(...) replacement")
print("  _to_python_expr     -> ^ to **, negative exponent fix, function name mapping")
import importlib.util, pathlib, re, math

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # worktree root

def _load(script_path):
    spec = importlib.util.spec_from_file_location("_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

script = _ROOT / "script/dev_analyze/analyze_equation_accuracy_AD.py"
ad = _load(script)

# --- _extract_func_call ---
# Simple: one sqrt call
text1 = "result = sqrt(1.0 + 2.0) = 3.0"
got = ad._extract_func_call(text1, "sqrt")
assert got == "sqrt(1.0 + 2.0)", f"simple: {got!r}"
print(f"  extract simple sqrt  : {got!r}  PASS")

# Two sqrts: symbolic formula first, numeric formula second.
# rfind picks the LAST "sqrt(" which is the numeric one — exactly what we want.
text2 = "sqrt(((x-x1)*W*pw)^2) = sqrt(((3.0*1.0*1.0)^2)) = 3.0"
got2 = ad._extract_func_call(text2, "sqrt")
assert got2 == "sqrt(((3.0*1.0*1.0)^2))", f"two-sqrt rfind: {got2!r}"
print(f"  two-sqrt rfind last  : {got2!r}  PASS")

# arccos with nested sqrt inside
text3 = "arccos(|3.0*4.0| / (sqrt(9.0) * sqrt(16.0)))"
got3 = ad._extract_func_call(text3, "arccos")
assert got3 == "arccos(|3.0*4.0| / (sqrt(9.0) * sqrt(16.0)))", f"arccos: {got3!r}"
print(f"  extract arccos       : {got3!r}  PASS")

# Missing function: returns None
assert ad._extract_func_call("no function here", "sqrt") is None
print(f"  missing func         : None  PASS")

# Unbalanced parens: returns None (depth never hits 0)
assert ad._extract_func_call("sqrt(1 + (2", "sqrt") is None
print(f"  unbalanced parens    : None  PASS")

# --- _convert_abs_notation ---
# Simple |...|
expr_abs = "|3.0 + 4.0|"
got_abs = ad._convert_abs_notation(expr_abs)
assert got_abs == "abs(3.0 + 4.0)", f"abs simple: {got_abs!r}"
print(f"  abs simple           : {got_abs!r}  PASS")

# Inside a larger expression
expr_abs2 = "x / |a*b + c*d|"
got_abs2 = ad._convert_abs_notation(expr_abs2)
assert got_abs2 == "x / abs(a*b + c*d)", f"abs in expr: {got_abs2!r}"
print(f"  abs in expr          : {got_abs2!r}  PASS")

# No bars: identity
expr_noabs = "sqrt(1.0 + 2.0)"
assert ad._convert_abs_notation(expr_noabs) == expr_noabs
print(f"  no bars (identity)   : PASS")

# --- _to_python_expr ---
# Exponent after closing paren: (...)^2 -> (...)**2
expr_pow = "sqrt((0.5 * 10.0)^2 + (0.3 * 8.0)^2)"
got_pow = ad._to_python_expr(expr_pow)
expected_pow = "math.sqrt((0.5 * 10.0)**2 + (0.3 * 8.0)**2)"
assert got_pow == expected_pow, f"pow paren: {got_pow!r}"
print(f"  (...)^2 conversion   : {got_pow!r}  PASS")

# Bare negative number before ^: -24.0^2 -> (-24.0)**2
expr_neg = "sqrt(-24.0^2 + 1.8^2)"
got_neg = ad._to_python_expr(expr_neg)
assert "(-24.0)**2" in got_neg and "(1.8)**2" in got_neg, f"neg pow: {got_neg!r}"
print(f"  -24.0^2 conversion   : {got_neg!r}  PASS")

# Function name mapping: arccos -> math.acos, sqrt -> math.sqrt
expr_fn = "arccos(sqrt(0.5))"
got_fn = ad._to_python_expr(expr_fn)
assert got_fn == "math.acos(math.sqrt(0.5))", f"fn map: {got_fn!r}"
print(f"  arccos+sqrt mapping  : {got_fn!r}  PASS")

# wrap_degrees=True wraps in math.degrees(...)
expr_deg = "arccos(0.5)"
got_deg = ad._to_python_expr(expr_deg, wrap_degrees=True)
assert got_deg == "math.degrees(math.acos(0.5))", f"wrap_degrees: {got_deg!r}"
print(f"  wrap_degrees         : {got_deg!r}  PASS")

# arctan2 replaced before arctan (order matters)
expr_at2 = "arctan2(1.0, 1.0)"
got_at2 = ad._to_python_expr(expr_at2)
assert got_at2 == "math.atan2(1.0, 1.0)", f"arctan2: {got_at2!r}"
print(f"  arctan2 (not arctan) : {got_at2!r}  PASS")

print("OK")
