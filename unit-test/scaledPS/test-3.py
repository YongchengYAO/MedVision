# TL GT scaling analytical check: doc_to_target_scaledPS == stored * S
print("Test 3: TL ground-truth scaling — doc_to_target_TumorLesionSize_scaledPS(doc) must equal stored major/minor axis lengths × S (tolerance 1e-6).")
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset
from lmms_eval.tasks.medvision.medvision_utils import (
    doc_to_target_TumorLesionSize_scaledPS,
    _get_pixel_size_scale_factor,
)

ds = load_dataset(
    "YongchengYAO/MedVision",
    "BraTS24_TumorLesionSize_Task04_Axial_Test",
    split="test",
    trust_remote_code=True,
)

for i in range(3):
    doc = ds[i]
    S = _get_pixel_size_scale_factor(doc, "uniform")
    stored_major = float(np.array(doc["biometric_profile"]["metric_value_major_axis"]).flatten()[0])
    stored_minor = float(np.array(doc["biometric_profile"]["metric_value_minor_axis"]).flatten()[0])
    scaled = doc_to_target_TumorLesionSize_scaledPS(doc)
    assert abs(scaled[0] - stored_major * S) < 1e-6, f"major mismatch: {scaled[0]} vs {stored_major * S}"
    assert abs(scaled[1] - stored_minor * S) < 1e-6, f"minor mismatch: {scaled[1]} vs {stored_minor * S}"
    print(f"sample {i}: S={S:.4f}  major {stored_major:.3f} -> {scaled[0]:.3f}  minor {stored_minor:.3f} -> {scaled[1]:.3f}")

print("OK")
