print("=== Test 5: scaledPS diagonal scale consistency ===")
print("Objective : Verify _compute_physical_diagonal uses the same scale factor S as")
print("            the prompt's pixel_size_text for scaledPS variants. Uses regex exec")
print("            to bypass lmms_eval import chain; loads real HuggingFace docs.")
print("Expected  :")
print("  TL uniform    : diag_uniform / diag_noscale == S  (ratio exactly matches scale factor)")
print("                  diag_uniform == sqrt((H*px_h*S)^2 + (W*px_w*S)^2)")
print("  AD anisotropic: diag_aniso == sqrt((H*px_h*S_h)^2 + (W*px_w*S_w)^2)")
print("                  when |S_h-S_w| > 0.01: diag_aniso != diag_uniform")
print("Note: requires dataset and NIfTI files (MedVision_DATA_DIR must be set).")
import functools, hashlib, os, re, pathlib, sys, math, random, numpy as np
import nibabel as nib
from scipy.ndimage import zoom

sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("Data/src").resolve()))

from datasets import load_dataset

# ── Load helpers via regex exec — no lmms_eval package imports ──────────────
src = pathlib.Path(
    "src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py"
).read_text()

scope = {"np": np, "os": os, "nib": nib, "zoom": zoom, "hashlib": hashlib, "functools": functools}
exec(re.search(r"^_SCALED_PS_LOW.*?_SCALED_PS_HIGH\s*=\s*[0-9.]+", src, re.S | re.M).group(0), scope)
exec(re.search(r"^def _load_nifti_2d.*?^(?=def [A-Za-z_])", src, re.S | re.M).group(0), scope)
exec(re.search(r"^def _get_pixel_size_scale_factor.*?^(?=(?:@|def )[A-Za-z_])", src, re.S | re.M).group(0), scope)
exec(re.search(r"^@functools\.lru_cache.*?^def _compute_physical_diagonal.*?^(?=def [A-Za-z_])", src, re.S | re.M).group(0), scope)

load_nifti = scope["_load_nifti_2d"]
get_scale  = scope["_get_pixel_size_scale_factor"]
diag_fn    = scope["_compute_physical_diagonal"]

# ── TL scaledPS: uniform mode ────────────────────────────────────────────────
print("\nTL scaledPS (uniform):")
ds_tl = load_dataset(
    "YongchengYAO/MedVision",
    "BraTS24_TumorLesionSize_Task04_Axial_Test",
    split="test",
    trust_remote_code=True,
)
random.seed(42)
tl_indices = random.sample(range(len(ds_tl)), 3)
for i in tl_indices:
    doc = ds_tl[i]
    S = get_scale(doc, "uniform")
    H, W = doc["image_size_2d"]

    px_hw, _ = load_nifti(doc["image_file"], doc["slice_dim"], doc["slice_idx"])
    raw_px_h, raw_px_w = float(px_hw[0]), float(px_hw[1])

    # For uniform mode, diagonal_uniform = S * diagonal_noscale exactly.
    diag_uniform  = diag_fn(doc, scale_mode="uniform", explicit_scale=None)
    diag_noscale  = diag_fn(doc, scale_mode=None, explicit_scale=None)
    got_ratio     = diag_uniform / diag_noscale
    assert abs(got_ratio - S) < 1e-6, (
        f"sample {i}: diag_uniform/diag_noscale={got_ratio:.6f} != S={S:.6f}"
    )

    # Also verify against the closed-form formula directly.
    expected = math.sqrt((H * raw_px_h * S) ** 2 + (W * raw_px_w * S) ** 2)
    assert abs(diag_uniform - expected) < 1e-6, (
        f"sample {i}: diag_uniform={diag_uniform:.6f} vs formula={expected:.6f}"
    )
    print(f"  TL ds[{i}]: size={H}x{W}px  raw_px=[{raw_px_h:.3f},{raw_px_w:.3f}]mm"
          f"  S={S:.4f}  diag_uniform={diag_uniform:.2f}mm  ratio={got_ratio:.6f}  PASS")

# ── AD scaledPS: anisotropic mode ────────────────────────────────────────────
print("\nAD scaledPS (anisotropic):")
ds_ad = load_dataset(
    "YongchengYAO/MedVision",
    "Ceph-Biometrics-400_BiometricsFromLandmarks_Distance_Task01_Sagittal_Test",
    split="test",
    trust_remote_code=True,
)
ad_indices = random.sample(range(len(ds_ad)), 3)
for i in ad_indices:
    doc = ds_ad[i]
    S_h, S_w = get_scale(doc, "anisotropic")
    H, W = doc["image_size_2d"]

    px_hw, _ = load_nifti(doc["image_file"], doc["slice_dim"], doc["slice_idx"])
    raw_px_h, raw_px_w = float(px_hw[0]), float(px_hw[1])

    diag_aniso   = diag_fn(doc, scale_mode="anisotropic", explicit_scale=None)
    diag_noscale = diag_fn(doc, scale_mode=None, explicit_scale=None)

    # Verify against closed-form formula.
    expected = math.sqrt((H * raw_px_h * S_h) ** 2 + (W * raw_px_w * S_w) ** 2)
    assert abs(diag_aniso - expected) < 1e-6, (
        f"sample {i}: diag_aniso={diag_aniso:.6f} vs formula={expected:.6f}"
    )

    # Anisotropic != uniform unless S_h == S_w (confirm non-trivial scaling).
    diag_uniform = diag_fn(doc, scale_mode="uniform", explicit_scale=None)
    S_u = get_scale(doc, "uniform")
    if abs(S_h - S_w) > 0.01:   # only assert when axes differ meaningfully
        assert abs(diag_aniso - diag_uniform) > 1e-4, (
            f"sample {i}: anisotropic diagonal unexpectedly equals uniform diagonal"
        )

    print(f"  AD ds[{i}]: size={H}x{W}px  raw_px=[{raw_px_h:.3f},{raw_px_w:.3f}]mm"
          f"  S_h={S_h:.4f} S_w={S_w:.4f}  diag_aniso={diag_aniso:.2f}mm  PASS")

print("OK")
