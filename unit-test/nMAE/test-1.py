print("=== Test 1: _compute_physical_diagonal formula ===")
print("Objective : Verify the physical diagonal formula for all scale modes using synthetic")
print("            (patched) pixel sizes. _read_nifti_zooms is monkeypatched — no files needed.")
print("Expected  :")
print("  None         -> sqrt((H*px_h)^2 + (W*px_w)^2)")
print("  'uniform'    -> S * unscaled_diagonal  (S applied equally to both axes)")
print("  'anisotropic'-> sqrt((H*px_h*S_h)^2 + (W*px_w*S_w)^2)")
print("  bad mode     -> ValueError")
import functools, re, math, pathlib, numpy as np

src = pathlib.Path("src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py").read_text()

# Pull constants and helpers into a fresh scope.
# Regex stops at '@' or 'def' at line start to avoid capturing decorator tails.
scope = {"np": np, "functools": functools}
exec(re.search(r"^_SCALED_PS_LOW.*?_SCALED_PS_HIGH\s*=\s*[0-9.]+", src, re.S | re.M).group(0), scope)
exec(re.search(r"^def _get_pixel_size_scale_factor.*?^(?=(?:@|def )[A-Za-z_])", src, re.S | re.M).group(0), scope)
exec(re.search(r"^def _compute_physical_diagonal.*?^(?=def [A-Za-z_])", src, re.S | re.M).group(0), scope)

# Patch _read_nifti_zooms so no real file is needed.
# _compute_physical_diagonal with slice_dim=2 uses voxel_size[0]=px_h, voxel_size[1]=px_w.
FAKE_PX_H, FAKE_PX_W = 0.5, 0.7        # mm
FAKE_H, FAKE_W = 256, 320               # pixels
scope["_read_nifti_zooms"] = lambda path: (FAKE_PX_H, FAKE_PX_W, 1.0)

diag_fn = scope["_compute_physical_diagonal"]

doc = {
    "image_file": "/tmp/fake.nii.gz",
    "slice_dim": 2,
    "slice_idx": 0,
    "image_size_2d": [FAKE_H, FAKE_W],
    "taskID": 1,
    "label": "1",
}

# scale_mode=None: s_h = s_w = 1
expected_no_scale = math.sqrt((FAKE_H * FAKE_PX_H) ** 2 + (FAKE_W * FAKE_PX_W) ** 2)
got = diag_fn(doc, scale_mode=None, explicit_scale=None)
assert abs(got - expected_no_scale) < 1e-9, f"scale_mode=None mismatch: {got} vs {expected_no_scale}"
print(f"  None        : {got:.4f}mm = sqrt(({FAKE_H}*{FAKE_PX_H})^2+({FAKE_W}*{FAKE_PX_W})^2)={expected_no_scale:.4f}mm  PASS")

# scale_mode="uniform": single S applied to both axes -> diagonal = S * unscaled
S = scope["_get_pixel_size_scale_factor"](doc, "uniform")
expected_uniform = math.sqrt((FAKE_H * FAKE_PX_H * S) ** 2 + (FAKE_W * FAKE_PX_W * S) ** 2)
got_uniform = diag_fn(doc, scale_mode="uniform", explicit_scale=None)
assert abs(got_uniform - expected_uniform) < 1e-9, f"uniform mismatch: {got_uniform} vs {expected_uniform}"
assert abs(got_uniform - S * expected_no_scale) < 1e-9, "uniform diagonal should equal S * unscaled diagonal"
print(f"  uniform     : {got_uniform:.4f}mm  ratio/S={got_uniform/expected_no_scale:.6f}/{S:.6f}  (S*unscaled={S*expected_no_scale:.4f})  PASS")

# scale_mode="anisotropic": independent S_h, S_w
S_h, S_w = scope["_get_pixel_size_scale_factor"](doc, "anisotropic")
expected_aniso = math.sqrt((FAKE_H * FAKE_PX_H * S_h) ** 2 + (FAKE_W * FAKE_PX_W * S_w) ** 2)
got_aniso = diag_fn(doc, scale_mode="anisotropic", explicit_scale=None)
assert abs(got_aniso - expected_aniso) < 1e-9, f"anisotropic mismatch: {got_aniso} vs {expected_aniso}"
print(f"  anisotropic : {got_aniso:.4f}mm  S_h={S_h:.4f} S_w={S_w:.4f}  formula={expected_aniso:.4f}mm  PASS")

# bad scale_mode raises ValueError
try:
    diag_fn(doc, scale_mode="bad", explicit_scale=None)
    assert False, "should have raised ValueError"
except ValueError:
    print("  bad mode    : ValueError raised  PASS")

print("OK")
