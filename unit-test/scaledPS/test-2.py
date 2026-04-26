# Force isotropic S=1 and recompute distance from landmarks; compare to stored GT.
print("Test 2: AD distance recomputation sanity — with S=(1,1) forced, landmark-based distance recomputation must match the stored ground-truth value within numerical tolerance.")
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path("src/medvision_bm/medvision_lmms_eval").resolve()))
sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset
from lmms_eval.tasks.medvision.medvision_utils import (
    _load_nifti_2d,
    create_doc_to_target_BiometricsFromLandmarks_scaledPS,
)
from medvision_ds.datasets.Ceph_Biometrics_400 import preprocess_biometry
import medvision_bm.sft.sft_utils as sft_utils
import lmms_eval.tasks.medvision.medvision_utils as mu

# Monkeypatch: force isotropic S=1
mu._get_pixel_size_scale_factor = lambda doc, mode, **kw: (1.0, 1.0) if mode == "anisotropic" else 1.0

ds = load_dataset(
    "YongchengYAO/MedVision",
    "Ceph-Biometrics-400_BiometricsFromLandmarks_Distance_Task01_Sagittal_Test",
    split="test",
    trust_remote_code=True,
)
target_fn = create_doc_to_target_BiometricsFromLandmarks_scaledPS(preprocess_biometry)
for i in range(3):
    doc = ds[i]
    stored = float(np.array(doc["biometric_profile"]["metric_value"]).flatten()[0])
    recomputed = target_fn(doc)
    print(f"sample {i}: stored={stored:.4f}  recomputed_S=1={recomputed:.4f}  diff={abs(stored-recomputed):.4e}")
