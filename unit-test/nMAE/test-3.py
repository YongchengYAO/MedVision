print("=== Test 3: process_results_TumorLesionSize nMAE ===")
print("Objective : Verify nMAE is computed correctly for TL samples.")
print("            Three cases per sample: perfect prediction, known +10mm error, parse failure.")
print("Expected  :")
print("  perfect pred  -> nMAE = 0.0")
print("  +10mm on major-> mean_abs_err = mean([|10|, |0|]) = 5.0mm  =>  nMAE = 5.0 / diagonal")
print("  parse failure -> success=False, NMAE=NaN")
print("Note: requires dataset and NIfTI files (MedVision_DATA_DIR must be set).")
import sys, pathlib, math, random, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset
from lmms_eval.tasks.medvision.medvision_utils import (
    process_results_TumorLesionSize,
    _compute_physical_diagonal,
    doc_to_target_TumorLesionSize,
)

ds = load_dataset(
    "YongchengYAO/MedVision",
    "BraTS24_TumorLesionSize_Task04_Axial_Test",
    split="test",
    trust_remote_code=True,
)

random.seed(42)
sample_indices = random.sample(range(len(ds)), 3)

for idx in sample_indices:
    doc = ds[idx]
    target = doc_to_target_TumorLesionSize(doc)          # [major, minor] in mm
    # The parser only reads numbers inside <answer></answer>; a bare string is a parse failure.
    target_str = "<answer>" + ",".join(f"{v:.4f}" for v in target) + "</answer>"
    diagonal = _compute_physical_diagonal(doc, scale_mode=None, explicit_scale=None)
    print(f"  sample ds[{idx}]  major={target[0]:.2f}mm  minor={target[1]:.2f}mm  diagonal={diagonal:.2f}mm")

    # Perfect prediction: nMAE must be exactly 0
    out_perfect = process_results_TumorLesionSize(doc, [target_str])
    nmae_perfect = out_perfect["nMAE"]
    assert nmae_perfect["success"], f"sample {idx}: perfect pred should have success=True"
    assert abs(nmae_perfect["NMAE"]) < 1e-6, f"sample {idx}: perfect pred nMAE={nmae_perfect['NMAE']}, expected 0"
    print(f"    perfect pred : nMAE={nmae_perfect['NMAE']:.6f}  (0.0 / {diagonal:.2f}mm)  PASS")

    # Known error: perturb major axis by +10mm.
    # mean_abs_err = mean([|pred_major - gt_major|, |pred_minor - gt_minor|])
    #              = mean([10.0, 0.0]) = 5.0mm  =>  nMAE = 5.0 / diagonal
    perturbed = [target[0] + 10.0, target[1]]
    perturbed_str = "<answer>" + ",".join(f"{v:.4f}" for v in perturbed) + "</answer>"
    out_perturbed = process_results_TumorLesionSize(doc, [perturbed_str])
    nmae_perturbed = out_perturbed["nMAE"]
    assert nmae_perturbed["success"], f"sample {idx}: perturbed pred should have success=True"
    expected_nmae = 5.0 / diagonal   # mean([10, 0]) = 5mm
    assert abs(nmae_perturbed["NMAE"] - expected_nmae) < 1e-6, (
        f"sample {idx}: perturbed nMAE={nmae_perturbed['NMAE']:.6f} vs expected={expected_nmae:.6f}"
    )
    print(f"    +10mm error  : nMAE={nmae_perturbed['NMAE']:.6f}  expected=5.0/{diagonal:.2f}={expected_nmae:.6f}  PASS")

# Parse failure: nMAE success must be False
out_fail = process_results_TumorLesionSize(ds[sample_indices[0]], ["not a number"])
assert not out_fail["nMAE"]["success"], "parse failure should give nMAE success=False"
assert math.isnan(out_fail["nMAE"]["NMAE"]), "parse failure should give nMAE=NaN"
print("  parse failure  : success=False, NMAE=NaN  (unparseable response -> no metric)  PASS")

print("OK")
