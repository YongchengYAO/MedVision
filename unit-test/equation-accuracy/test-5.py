print("=== Test 5: analyze_tl_sample (TL pipeline, steps 3 & 4) + edge cases ===")
print("Objective : Verify the TL two-step pipeline (major + minor axis equations)")
print("            and edge cases shared by both AD and TL scripts.")
print("Expected  :")
print("  steps 3 and 4 both evaluated independently")
print("  step 3 correct, step 4 wrong -> correct MREs per step")
print("  symbolic sqrt before numeric sqrt -> rfind picks correct (last) one")
print("  disallowed function -> step_eval_error recorded, MRE=None")
import importlib.util, pathlib, math

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # worktree root

def _load(script_path):
    spec = importlib.util.spec_from_file_location("_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

tl = _load(_ROOT / "script/dev_analyze/analyze_equation_accuracy_TL.py")
ad = _load(_ROOT / "script/dev_analyze/analyze_equation_accuracy_AD.py")

def make_tl_solution(r3, a3, r4, a4):
    return (
        "<think>"
        "<step-1-reasoning>...</step-1-reasoning>"
        "<step-1-answer>(0.4, 0.5), (0.6, 0.5)</step-1-answer>"
        "<step-2-reasoning>...</step-2-reasoning>"
        "<step-2-answer>(0.5, 0.4), (0.5, 0.6)</step-2-answer>"
        f"<step-3-reasoning>{r3}</step-3-reasoning>"
        f"<step-3-answer>The major axis length: {a3}.</step-3-answer>"
        f"<step-4-reasoning>{r4}</step-4-reasoning>"
        f"<step-4-answer>The minor axis length: {a4}.</step-4-answer>"
        "</think>"
        f"<answer>({a3}, {a4})</answer>"
    )

# ---- Case 1: Both steps correct ----
# Step 3: sqrt((3*1*1)^2 + (4*1*1)^2) = 5.0
# Step 4: sqrt((5*1*1)^2 + (12*1*1)^2) = 13.0
r3_ok = "major_axis_length = sqrt(((3.0*1.0*1.0)^2) + ((4.0*1.0*1.0)^2)) = 5.0"
r4_ok = "minor_axis_length = sqrt(((5.0*1.0*1.0)^2) + ((12.0*1.0*1.0)^2)) = 13.0"
sol_ok = make_tl_solution(r3_ok, "5.0", r4_ok, "13.0")
res_ok = tl.analyze_tl_sample(sol_ok)
assert abs(res_ok["step3_python_eval"] - 5.0) < 1e-9, f"step3_py: {res_ok['step3_python_eval']}"
assert abs(res_ok["step4_python_eval"] - 13.0) < 1e-9, f"step4_py: {res_ok['step4_python_eval']}"
assert res_ok["step3_equation_MRE"] < 1e-9, f"step3_MRE: {res_ok['step3_equation_MRE']}"
assert res_ok["step4_equation_MRE"] < 1e-9, f"step4_MRE: {res_ok['step4_equation_MRE']}"
print(f"  steps 3&4 correct    : MRE3={res_ok['step3_equation_MRE']:.6f}  "
      f"MRE4={res_ok['step4_equation_MRE']:.6f}  PASS")

# ---- Case 2: Step 3 correct, step 4 model error (says 14.0 instead of 13.0) ----
sol_err4 = make_tl_solution(r3_ok, "5.0", r4_ok, "14.0")
res_err4 = tl.analyze_tl_sample(sol_err4)
assert abs(res_err4["step3_equation_MRE"]) < 1e-9, "step3 should be 0"
expected_mre4 = abs(14.0 - 13.0) / 13.0
assert abs(res_err4["step4_equation_MRE"] - expected_mre4) < 1e-9, \
    f"step4_MRE {res_err4['step4_equation_MRE']} vs {expected_mre4}"
print(f"  step4 error (14!=13) : MRE4={res_err4['step4_equation_MRE']:.4f} == "
      f"{expected_mre4:.4f}  PASS")

# ---- Case 3: Symbolic sqrt appears BEFORE numeric sqrt -> rfind picks last ----
# The model writes "formula = sqrt(symbolic) = sqrt(numeric) = result"
# _extract_func_call uses rfind so it gets the LAST sqrt (numeric)
r3_two_sqrts = (
    "formula = sqrt(((x2-x1)*W*pw)^2 + ((y2-y1)*H*ph)^2)"
    " = sqrt(((0.200*504*1.0)^2) + ((0.300*504*1.0)^2))"
    " = 181.044"
)
true_val_two = math.sqrt((0.200*504*1.0)**2 + (0.300*504*1.0)**2)
r4_simple = f"sqrt(((3.0*1.0*1.0)^2) + ((4.0*1.0*1.0)^2)) = 5.0"
sol_two = make_tl_solution(r3_two_sqrts, "181.044", r4_simple, "5.0")
res_two = tl.analyze_tl_sample(sol_two)
assert res_two["step3_python_eval"] is not None
assert abs(res_two["step3_python_eval"] - true_val_two) < 1e-3, \
    f"two-sqrt: {res_two['step3_python_eval']} vs {true_val_two}"
print(f"  two sqrts (rfind)    : python_eval={res_two['step3_python_eval']:.3f}  "
      f"true={true_val_two:.3f}  PASS")

# ---- Case 4: Missing step-4 tag -> step4 all None ----
sol_no_step4 = (
    "<think>"
    f"<step-3-reasoning>{r3_ok}</step-3-reasoning>"
    f"<step-3-answer>5.0</step-3-answer>"
    "</think>"
)
res_no4 = tl.analyze_tl_sample(sol_no_step4)
assert res_no4["step4_raw_expr"] is None
assert res_no4["step4_python_eval"] is None
assert res_no4["step4_equation_MRE"] is None
print(f"  missing step-4 tag   : step4_* all None  PASS")

# ---- Case 5: Disallowed function (e.g. 'log') -> eval_error, MRE=None ----
r3_bad_fn = "sqrt(log(5.0))"  # log not in _MATH_FUNCS for TL (wait, it IS in AD but not TL?)
# Actually checking: _MATH_FUNCS in both scripts: sqrt, acos, asin, atan, atan2, degrees, abs
# 'log' is NOT in _MATH_FUNCS -> should raise ValueError
r4_ok2 = "sqrt(((3.0*1.0*1.0)^2) + ((4.0*1.0*1.0)^2))"
sol_bad_fn = make_tl_solution(r3_bad_fn, "5.0", r4_ok2, "5.0")
res_bad = tl.analyze_tl_sample(sol_bad_fn)
assert res_bad["step3_python_eval"] is None, f"should be None: {res_bad['step3_python_eval']}"
assert res_bad["step3_equation_MRE"] is None
assert "step3_eval_error" in res_bad, "should have step3_eval_error key"
print(f"  disallowed function  : eval_error='{res_bad['step3_eval_error']}'  MRE=None  PASS")

# ---- Case 6: _cal_equation_MRE boundary (python_eval ~0) ----
# When python_val is nearly 0, denominator guard (1e-15) prevents division by zero
mre = ad._cal_equation_MRE(1.0, 0.0)
assert mre == 1.0 / 1e-15, f"zero denominator guard: {mre}"
print(f"  zero denominator     : MRE={mre:.2e} (uses 1e-15 guard)  PASS")

# ---- Case 7: metric_type is 'tl' ----
assert res_ok["metric_type"] == "tl"
print(f"  metric_type          : 'tl'  PASS")

print("OK")
