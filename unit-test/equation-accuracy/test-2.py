print("=== Test 2: _compute_expr (AST-based expression evaluator) ===")
print("Objective : Verify the AST evaluator handles all numeric math patterns seen in")
print("            model outputs: nested sqrt, arccos+degrees, negative exponents,")
print("            abs() bars, and arithmetic.")
print("Expected  :")
print("  Each expression evaluates to the correct mathematical value.")
import importlib.util, pathlib, math

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # worktree root

def _load(script_path):
    spec = importlib.util.spec_from_file_location("_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ad = _load(_ROOT / "script/analyze/equation-accuracy/analyze_equation_accuracy_AD.py")
tl = _load(_ROOT / "script/analyze/equation-accuracy/analyze_equation_accuracy_TL.py")

TOL = 1e-9

def check(label, expr_raw, expected):
    # Convert through _to_python_expr then _compute_expr (AD version)
    py_expr = ad._to_python_expr(expr_raw)
    got = ad._compute_expr(py_expr)
    assert abs(got - expected) < TOL, f"{label}: got {got}, expected {expected}"
    print(f"  {label}: {got:.6f} == {expected:.6f}  PASS")

def check_deg(label, expr_raw, expected_deg):
    py_expr = ad._to_python_expr(expr_raw, wrap_degrees=True)
    got = ad._compute_expr(py_expr)
    assert abs(got - expected_deg) < 1e-6, f"{label}: got {got}, expected {expected_deg}"
    print(f"  {label}: {got:.6f} deg == {expected_deg:.6f} deg  PASS")

# --- Basic arithmetic ---
check("1 + 2", "1.0 + 2.0", 3.0)
check("5 * 3", "5.0 * 3.0", 15.0)
check("10 / 4", "10.0 / 4.0", 2.5)
check("2 ** 3 (via (2.0)^3)", "(2.0)^3", 8.0)

# --- sqrt ---
check("sqrt(4.0)", "sqrt(4.0)", 2.0)
check("sqrt(9.0 + 16.0)", "sqrt(9.0 + 16.0)", 5.0)  # 3-4-5 triangle

# --- Distance formula: sqrt((dx*W*pw)^2 + (dy*H*ph)^2) ---
# Use known values: dx=0.066, W=504, pw=0.476; dy=-0.265, H=504, ph=0.384
dx, W, pw = 0.066, 504.0, 0.476
dy, H, ph = -0.265, 504.0, 0.384
expected_dist = math.sqrt((dx * W * pw)**2 + (dy * H * ph)**2)
dist_expr = f"sqrt((({dx} * {W} * {pw})^2) + (({dy} * {H} * {ph})^2))"
check("distance formula", dist_expr, expected_dist)

# --- Negative bare number before ^ ---
# -24.0^2 should be +576, not -576 (Python precedence fix)
check("(-24.0)^2 -> 576", "sqrt(-24.0^2)", 24.0)   # sqrt(576)=24
check("positive+negative", "sqrt(2.8^2 + -24.0^2)", math.sqrt(2.8**2 + 24.0**2))

# --- abs() from |...| conversion ---
abs_expr = ad._convert_abs_notation("|2.800*-24.000 + 54.200*1.800|")
abs_py = ad._to_python_expr(abs_expr)
expected_abs = abs(2.800 * -24.000 + 54.200 * 1.800)
got_abs = ad._compute_expr(abs_py)
assert abs(got_abs - expected_abs) < TOL, f"abs: {got_abs} vs {expected_abs}"
print(f"  |A·B| expression     : {got_abs:.4f} == {expected_abs:.4f}  PASS")

# --- arccos + degrees ---
# arccos(0.5) = 60 degrees
check_deg("arccos(0.5) = 60 deg", "arccos(0.5)", 60.0)
# arccos(0.0) = 90 degrees
check_deg("arccos(0.0) = 90 deg", "arccos(0.0)", 90.0)
# arccos(1.0) = 0 degrees
check_deg("arccos(1.0) = 0 deg",  "arccos(1.0)", 0.0)

# --- Full angle formula: arccos(|A·B| / (||A||·||B||)) ---
# A=(3,4), B=(0,5): dot=20, ||A||=5, ||B||=5, cos=0.8, angle=36.87 deg
Ax, Ay, Bx, By = 3.0, 4.0, 0.0, 5.0
expected_angle_deg = math.degrees(math.acos(abs(Ax*Bx + Ay*By) / (math.sqrt(Ax**2 + Ay**2) * math.sqrt(Bx**2 + By**2))))
angle_expr = f"arccos(|{Ax}*{Bx} + {Ay}*{By}| / (sqrt({Ax}^2 + {Ay}^2) * sqrt({Bx}^2 + {By}^2)))"
check_deg("full angle formula", angle_expr, expected_angle_deg)

# --- TL script _compute_expr gives same result (shared implementation) ---
tl_result = tl._compute_expr(tl._to_python_expr("sqrt(3.0^2 + 4.0^2)"))
assert abs(tl_result - 5.0) < TOL, f"TL sqrt: {tl_result}"
print(f"  TL _compute_expr     : sqrt(3^2+4^2)={tl_result:.1f}  PASS")

print("OK")
