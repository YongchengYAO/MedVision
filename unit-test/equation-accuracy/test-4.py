print("=== Test 4: analyze_angle_sample (AD angle pipeline) ===")
print("Objective : Verify the full angle pipeline including arccos extraction,")
print("            |...| absolute-value conversion, and degrees wrapping.")
print("Expected  :")
print("  correct arithmetic   -> equation_MRE ~0")
print("  arithmetic error     -> equation_MRE reflects the mistake")
print("  no arccos in text    -> raw_expr=None")
import importlib.util, pathlib, math

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent  # worktree root

def _load(script_path):
    spec = importlib.util.spec_from_file_location("_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ad = _load(_ROOT / "script/analyze/equation-accuracy/analyze_equation_accuracy_AD.py")

# ---- Helpers ----

def make_angle_solution(reasoning_body, answer_deg):
    return (
        f"<think>"
        f"<step-1-reasoning>...</step-1-reasoning><step-1-answer>(0.3, 0.4), (0.4, 0.5)</step-1-answer>"
        f"<step-2-reasoning>...</step-2-reasoning><step-2-answer>(0.3, 0.4), (0.5, 0.3)</step-2-answer>"
        f"<step-3-reasoning>{reasoning_body}</step-3-reasoning>"
        f"<step-3-answer>The angle: {answer_deg}.</step-3-answer>"
        f"</think>"
        f"<answer>{answer_deg}</answer>"
    )

# ---- Case 1: Perfect arithmetic — A=(3,4), B=(0,5) ----
# dot=20, ||A||=5, ||B||=5, arccos(20/25)=arccos(0.8)=36.87 deg
Ax, Ay, Bx, By = 3.0, 4.0, 0.0, 5.0
true_deg = math.degrees(math.acos(abs(Ax*Bx + Ay*By) / (math.sqrt(Ax**2 + Ay**2) * math.sqrt(Bx**2 + By**2))))
# Model correctly states the true value
reasoning_ok = (
    f"angle = arccos(|{Ax}*{Bx} + {Ay}*{By}| / "
    f"(sqrt({Ax}^2 + {Ay}^2) * sqrt({Bx}^2 + {By}^2)))"
    f" = {true_deg:.3f} degrees"
)
solution_ok = make_angle_solution(reasoning_ok, f"{true_deg:.3f}")
res_ok = ad.analyze_angle_sample(solution_ok)
assert res_ok["step3_python_eval"] is not None, "python_eval None"
assert abs(res_ok["step3_python_eval"] - true_deg) < 1e-6, \
    f"python_eval {res_ok['step3_python_eval']} vs {true_deg}"
assert res_ok["step3_equation_MRE"] < 1e-4, \
    f"MRE {res_ok['step3_equation_MRE']} (expected near-0; model answer is 3dp rounded)"
print(f"  perfect arithmetic   : python_eval={res_ok['step3_python_eval']:.4f} deg  "
      f"true={true_deg:.4f} deg  MRE={res_ok['step3_equation_MRE']:.2e}  PASS")

# ---- Case 2: Model makes arithmetic error (states wrong degrees) ----
wrong_deg = true_deg + 10.0
answer_str = f"{wrong_deg:.3f}"
solution_err = make_angle_solution(reasoning_ok, answer_str)
res_err = ad.analyze_angle_sample(solution_err)
# expected_mre uses the EXTRACTED model value (rounded string), same as the pipeline
extracted_model_val = float(answer_str)
expected_mre = abs(extracted_model_val - true_deg) / abs(true_deg)
assert abs(res_err["step3_equation_MRE"] - expected_mre) < 1e-9, \
    f"MRE {res_err['step3_equation_MRE']} vs {expected_mre}"
print(f"  arithmetic error     : model={extracted_model_val:.3f}  python={res_err['step3_python_eval']:.3f}  "
      f"MRE={res_err['step3_equation_MRE']:.4f}  PASS")

# ---- Case 3: Perpendicular vectors (angle=90 deg) ----
# A=(1,0), B=(0,1): dot=0, ||A||=||B||=1, arccos(0)=90
Ax2, Ay2, Bx2, By2 = 1.0, 0.0, 0.0, 1.0
true_deg2 = 90.0
reasoning_perp = (
    f"arccos(|{Ax2}*{Bx2} + {Ay2}*{By2}| / "
    f"(sqrt({Ax2}^2 + {Ay2}^2) * sqrt({Bx2}^2 + {By2}^2)))"
    f" = 90.0 degrees"
)
solution_perp = make_angle_solution(reasoning_perp, "90.0")
res_perp = ad.analyze_angle_sample(solution_perp)
assert abs(res_perp["step3_python_eval"] - 90.0) < 1e-6, \
    f"perpendicular: {res_perp['step3_python_eval']}"
assert res_perp["step3_equation_MRE"] < 1e-9
print(f"  perpendicular (90deg): python_eval={res_perp['step3_python_eval']:.1f}  MRE=0  PASS")

# ---- Case 4: Negative Bx component (-24.0^2 precedence) ----
# Same as real model example
Ax3, Ay3, Bx3, By3 = 2.800, 54.200, -24.000, 1.800
true_deg3 = math.degrees(math.acos(
    abs(Ax3*Bx3 + Ay3*By3) /
    (math.sqrt(Ax3**2 + Ay3**2) * math.sqrt(Bx3**2 + By3**2))
))
reasoning_neg = (
    f"arccos(|{Ax3}*{Bx3} + {Ay3}*{By3}| / "
    f"(sqrt({Ax3}^2 + {Ay3}^2) * sqrt({Bx3}^2 + {By3}^2)))"
)
solution_neg = make_angle_solution(reasoning_neg, "86.241")  # model's (wrong) answer
res_neg = ad.analyze_angle_sample(solution_neg)
assert res_neg["step3_python_eval"] is not None
# Verify Python got the RIGHT result (not -576 from precedence bug)
assert abs(res_neg["step3_python_eval"] - true_deg3) < 1e-6, \
    f"negative Bx: {res_neg['step3_python_eval']} vs {true_deg3}"
print(f"  negative Bx (real ex): python_eval={res_neg['step3_python_eval']:.3f} deg  "
      f"model=86.241  MRE={res_neg['step3_equation_MRE']:.4f}  PASS")

# ---- Case 5: Missing arccos in reasoning -> raw_expr=None ----
solution_no_arccos = make_angle_solution("I cannot compute this.", "45.0")
res_no_arccos = ad.analyze_angle_sample(solution_no_arccos)
assert res_no_arccos["step3_raw_expr"] is None
assert res_no_arccos["step3_python_eval"] is None
print(f"  no arccos in text    : raw_expr=None  PASS")

# ---- Case 6: metric_type is angle ----
assert res_ok["metric_type"] == "angle"
print(f"  metric_type          : 'angle'  PASS")

print("OK")
