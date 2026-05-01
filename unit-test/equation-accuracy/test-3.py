print("=== Test 3: analyze_distance_sample (AD distance pipeline) ===")
print("Objective : Verify the full distance pipeline using synthetic CoT strings.")
print("Expected  :")
print("  correct arithmetic   -> equation_MRE ~0")
print("  deliberate error     -> equation_MRE > 0 and reflects the arithmetic mistake")
print("  missing tags         -> all step3_* fields are None")
print("  model matches Python -> MRE = 0 exactly (integer inputs)")
import importlib.util, pathlib, math

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # worktree root

def _load(script_path):
    spec = importlib.util.spec_from_file_location("_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ad = _load(_ROOT / "script/dev_analyze/analyze_equation_accuracy_AD.py")

# ---- Build synthetic solution strings ----

def make_dist_solution(reasoning_body, answer_val):
    return (
        f"<think>"
        f"<step-1-reasoning>...</step-1-reasoning><step-1-answer>(0.5, 0.5)</step-1-answer>"
        f"<step-2-reasoning>...</step-2-reasoning><step-2-answer>(0.6, 0.4)</step-2-answer>"
        f"<step-3-reasoning>{reasoning_body}</step-3-reasoning>"
        f"<step-3-answer>The distance: {answer_val}.</step-3-answer>"
        f"</think>"
        f"<answer>{answer_val}</answer>"
    )

# --- Case 1: Perfect arithmetic (model answer == Python eval) ---
# Use exact-integer inputs so there is zero floating-point drift
# sqrt((3.0 * 1.0 * 1.0)^2 + (4.0 * 1.0 * 1.0)^2) = sqrt(9+16) = 5
reasoning_ok = (
    "distance = sqrt(((3.0 * 1.0 * 1.0)^2) + ((4.0 * 1.0 * 1.0)^2)) = 5.0"
)
solution_ok = make_dist_solution(reasoning_ok, "5.0")
res_ok = ad.analyze_distance_sample(solution_ok)
assert res_ok["step3_model_answer"] == 5.0, f"model_answer: {res_ok['step3_model_answer']}"
assert abs(res_ok["step3_python_eval"] - 5.0) < 1e-9, f"python_eval: {res_ok['step3_python_eval']}"
assert res_ok["step3_equation_MRE"] < 1e-9, f"MRE: {res_ok['step3_equation_MRE']}"
print(f"  perfect arithmetic   : model={res_ok['step3_model_answer']:.3f}  "
      f"python={res_ok['step3_python_eval']:.3f}  MRE={res_ok['step3_equation_MRE']:.6f}  PASS")

# --- Case 2: Model makes an arithmetic error (says 6.0 instead of 5.0) ---
solution_err = make_dist_solution(reasoning_ok, "6.0")
res_err = ad.analyze_distance_sample(solution_err)
assert res_err["step3_model_answer"] == 6.0
assert abs(res_err["step3_python_eval"] - 5.0) < 1e-9
expected_mre = abs(6.0 - 5.0) / 5.0
assert abs(res_err["step3_equation_MRE"] - expected_mre) < 1e-9, \
    f"MRE {res_err['step3_equation_MRE']} vs {expected_mre}"
print(f"  arithmetic error     : model=6.0  python=5.0  "
      f"MRE={res_err['step3_equation_MRE']:.4f} == {expected_mre:.4f}  PASS")

# --- Case 3: Real-world-style formula from the actual model output ---
# sqrt(((0.587 - 0.521) * 504 * 0.476)^2 + ((0.334 - 0.599) * 504 * 0.384)^2)
dx, W, pw = 0.587 - 0.521, 504.0, 0.476
dy, H, ph = 0.334 - 0.599, 504.0, 0.384
true_val = math.sqrt((dx * W * pw)**2 + (dy * H * ph)**2)
reasoning_real = (
    f"distance = sqrt((({dx} * {W} * {pw})^2) + (({dy} * {H} * {ph})^2)) = {true_val:.3f}"
)
# Model claims 56.889; true value may differ due to rounding
solution_real = make_dist_solution(reasoning_real, "56.889")
res_real = ad.analyze_distance_sample(solution_real)
assert res_real["step3_python_eval"] is not None, "python_eval should not be None"
assert abs(res_real["step3_python_eval"] - true_val) < 1e-6, \
    f"python_eval {res_real['step3_python_eval']} vs true {true_val}"
print(f"  real formula         : python_eval={res_real['step3_python_eval']:.4f}  "
      f"true={true_val:.4f}  model=56.889  "
      f"MRE={res_real['step3_equation_MRE']:.6f}  PASS")

# --- Case 4: Missing step-3-reasoning tag -> all None ---
solution_no_tag = (
    "<think>"
    "<step-3-answer>The distance: 5.0.</step-3-answer>"
    "</think>"
)
res_no_tag = ad.analyze_distance_sample(solution_no_tag)
assert res_no_tag["step3_raw_expr"] is None
assert res_no_tag["step3_python_eval"] is None
assert res_no_tag["step3_equation_MRE"] is None
print(f"  missing reasoning    : all None  PASS")

# --- Case 5: No sqrt in reasoning -> raw_expr=None, python_eval=None ---
solution_no_sqrt = make_dist_solution("I cannot compute this.", "5.0")
res_no_sqrt = ad.analyze_distance_sample(solution_no_sqrt)
assert res_no_sqrt["step3_raw_expr"] is None
assert res_no_sqrt["step3_python_eval"] is None
assert res_no_sqrt["step3_equation_MRE"] is None
print(f"  no sqrt in reasoning : raw_expr=None  PASS")

# --- Case 6: metric_type field is set correctly ---
assert res_ok["metric_type"] == "distance"
print(f"  metric_type          : 'distance'  PASS")

print("OK")
