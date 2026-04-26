# AD angle invariance: isotropic scaling leaves angle unchanged; anisotropic changes it.
print("Test 4: AD angle invariance — isotropic scaling (S_h == S_w) must leave the computed angle exactly unchanged; anisotropic scaling (S_h ≠ S_w) must change it.")
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset
from lmms_eval.tasks.medvision.medvision_utils import (
    create_doc_to_target_BiometricsFromLandmarks_scaledPS,
)
from medvision_ds.datasets.Ceph_Biometrics_400 import preprocess_biometry
import lmms_eval.tasks.medvision.medvision_utils as mu

ds = load_dataset(
    "YongchengYAO/MedVision",
    "Ceph-Biometrics-400_BiometricsFromLandmarks_Angle_Task01_Sagittal_Test",
    split="test",
    trust_remote_code=True,
)

# create the target function once; monkeypatching the module-level helper affects all calls
target_fn = create_doc_to_target_BiometricsFromLandmarks_scaledPS(preprocess_biometry)

# stored GT has small rounding differences vs runtime recomputation (~0.02% relative)
GT_TOL = 0.1   # degrees — lenient tolerance for recomputed vs stored GT
for i in range(3):
    doc = ds[i]
    stored = float(np.array(doc["biometric_profile"]["metric_value"]).flatten()[0])

    # isotropic S=1: should reproduce stored GT within rounding tolerance
    mu._get_pixel_size_scale_factor = lambda d, mode: (1.0, 1.0)
    angle_iso1 = target_fn(doc)

    # isotropic S=2: must be exactly equal to iso_S=1 (same code path, only S differs)
    mu._get_pixel_size_scale_factor = lambda d, mode: (2.0, 2.0)
    angle_iso2 = target_fn(doc)

    # anisotropic S_h=1, S_w=3: angle should differ from stored GT
    mu._get_pixel_size_scale_factor = lambda d, mode: (1.0, 3.0)
    angle_aniso = target_fn(doc)

    diff_iso1 = abs(angle_iso1 - stored)
    print(
        f"sample {i}: stored={stored:.4f}  "
        f"iso_S=1={angle_iso1:.4f}(diff={diff_iso1:.1e})  "
        f"iso_S=2={angle_iso2:.4f}(iso_diff={abs(angle_iso2 - angle_iso1):.1e})  "
        f"aniso(1,3)={angle_aniso:.4f}"
    )
    assert diff_iso1 < GT_TOL, f"sample {i}: iso S=1 diff vs stored too large: {diff_iso1}"
    assert angle_iso2 == angle_iso1, f"sample {i}: iso S=2 != iso S=1 (not invariant)"
    assert abs(angle_aniso - stored) > GT_TOL, f"sample {i}: anisotropic angle unexpectedly matched stored GT"

print("OK: isotropic scaling leaves angle unchanged; anisotropic changes it")
