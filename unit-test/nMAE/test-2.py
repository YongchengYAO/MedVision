print("=== Test 2: aggregate_results_NMAE ===")
print("Objective : Verify the aggregator averages only success=True entries, silently")
print("            excludes success=False (angle or parse-failure samples), and returns NaN")
print("            when no valid results exist. No dataset required.")
print("Expected  :")
print("  all success   -> mean of all NMAE values")
print("  mixed         -> mean of success=True only; False entries excluded")
print("  all fail      -> NaN")
print("  empty list    -> NaN")
import sys, pathlib, math, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))

from lmms_eval.tasks.medvision.medvision_utils import aggregate_results_NMAE

# All successes: average of known values
results_all_ok = [
    {"NMAE": 0.10, "success": True},
    {"NMAE": 0.20, "success": True},
    {"NMAE": 0.30, "success": True},
]
got = aggregate_results_NMAE(results_all_ok)
expected = (0.10 + 0.20 + 0.30) / 3
assert abs(got - expected) < 1e-9, f"all-success mean wrong: {got} vs {expected}"
print(f"  all-success (n=3)       : mean={got:.4f}  = (0.10+0.20+0.30)/3={expected:.4f}  PASS")

# Mixed: NaN entries (success=False) must be excluded
results_mixed = [
    {"NMAE": 0.10, "success": True},
    {"NMAE": float("nan"), "success": False},   # angle sample or parse failure
    {"NMAE": 0.30, "success": True},
    {"NMAE": float("nan"), "success": False},
]
got_mixed = aggregate_results_NMAE(results_mixed)
expected_mixed = (0.10 + 0.30) / 2
assert abs(got_mixed - expected_mixed) < 1e-9, f"mixed mean wrong: {got_mixed} vs {expected_mixed}"
print(f"  mixed (n=4, 2 excluded) : mean={got_mixed:.4f}  = (0.10+0.30)/2={expected_mixed:.4f}  PASS")

# All fail: should return NaN
results_all_fail = [
    {"NMAE": float("nan"), "success": False},
    {"NMAE": float("nan"), "success": False},
]
got_fail = aggregate_results_NMAE(results_all_fail)
assert math.isnan(got_fail), f"all-fail should return NaN, got {got_fail}"
print(f"  all-fail (n=2)          : NaN  (no valid entries)  PASS")

# Empty list: should return NaN
got_empty = aggregate_results_NMAE([])
assert math.isnan(got_empty), f"empty list should return NaN, got {got_empty}"
print(f"  empty list              : NaN  PASS")

print("OK")
