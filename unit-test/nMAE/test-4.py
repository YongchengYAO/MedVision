print("=== Test 4: AD nMAE suppression ===")
print("Objective : Verify nMAE is suppressed for angle samples (success=False, NMAE=NaN)")
print("            and computed correctly for distance samples (success=True, NMAE=0 for")
print("            perfect prediction). Suppression must not affect MAE or SuccessRate.")
print("Expected  :")
print("  angle samples   -> nMAE success=False, NMAE=NaN; MAE+SuccessRate unaffected")
print("  distance samples-> nMAE success=True, NMAE=0 (perfect prediction)")
print("Note: requires dataset and NIfTI files (MedVision_DATA_DIR must be set).")
import sys, pathlib, math, random, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset
from lmms_eval.tasks.medvision.medvision_utils import (
    process_results_BiometricsFromLandmarks,
    doc_to_target_BiometricsFromLandmarks,
)

# --- Angle dataset: nMAE must always be suppressed ---
print("\nChecking Ceph-Angle samples (all must have nMAE success=False)...")
ds_angle = load_dataset(
    "YongchengYAO/MedVision",
    "Ceph-Biometrics-400_BiometricsFromLandmarks_Angle_Task01_Sagittal_Test",
    split="test",
    trust_remote_code=True,
)
random.seed(42)
angle_indices = random.sample(range(len(ds_angle)), 5)
for idx in angle_indices:
    doc = ds_angle[idx]
    bp = doc["biometric_profile"]
    target_val = float(np.array(doc_to_target_BiometricsFromLandmarks(doc)).flatten()[0])
    # The parser only reads numbers inside <answer></answer>; a bare string is a parse failure.
    target_str = f"<answer>{target_val:.4f}</answer>"
    out = process_results_BiometricsFromLandmarks(doc, [target_str])
    nmae = out["nMAE"]
    assert not nmae["success"], f"angle sample {idx}: nMAE should have success=False, got {nmae}"
    assert math.isnan(nmae["NMAE"]), f"angle sample {idx}: nMAE NMAE should be NaN, got {nmae['NMAE']}"
    print(f"  angle ds[{idx}]: {bp['metric_key']!r}  type={bp['metric_type']}  nMAE=NaN  success=False  PASS")

# --- Distance dataset: nMAE must be computed and finite ---
print("\nChecking Ceph-Distance samples (all must have nMAE success=True and NMAE=0 for perfect pred)...")
ds_dist = load_dataset(
    "YongchengYAO/MedVision",
    "Ceph-Biometrics-400_BiometricsFromLandmarks_Distance_Task01_Sagittal_Test",
    split="test",
    trust_remote_code=True,
)
dist_indices = random.sample(range(len(ds_dist)), 5)
for idx in dist_indices:
    doc = ds_dist[idx]
    bp = doc["biometric_profile"]
    target_val = float(np.array(doc_to_target_BiometricsFromLandmarks(doc)).flatten()[0])
    target_str = f"<answer>{target_val:.4f}</answer>"
    out = process_results_BiometricsFromLandmarks(doc, [target_str])
    nmae = out["nMAE"]
    assert nmae["success"], f"distance sample {idx}: nMAE should have success=True, got {nmae}"
    assert math.isfinite(nmae["NMAE"]), f"distance sample {idx}: nMAE should be finite, got {nmae['NMAE']}"
    # With perfect prediction nMAE should be 0
    assert abs(nmae["NMAE"]) < 1e-6, f"distance sample {idx}: perfect pred nMAE={nmae['NMAE']}, expected 0"
    print(f"  dist  ds[{idx}]: {bp['metric_key']!r}  type={bp['metric_type']}  nMAE={nmae['NMAE']:.6f} (perfect->0)  PASS")

# --- MAE and SuccessRate flags must be unaffected by angle suppression ---
print("\nVerifying that nMAE suppression does not affect MAE/SuccessRate for angle samples...")
for idx in angle_indices[:3]:
    doc = ds_angle[idx]
    target_val = float(np.array(doc_to_target_BiometricsFromLandmarks(doc)).flatten()[0])
    target_str = f"<answer>{target_val:.4f}</answer>"
    out = process_results_BiometricsFromLandmarks(doc, [target_str])
    assert out["MAE"]["success"], f"angle sample {idx}: MAE success should be True"
    assert out["SuccessRate"]["success"], f"angle sample {idx}: SuccessRate should be True"
    assert not out["nMAE"]["success"], f"angle sample {idx}: nMAE success should be False"
    print(f"  angle ds[{idx}]: MAE success=True  nMAE success=False  (suppression isolated)  PASS")

print("OK")
