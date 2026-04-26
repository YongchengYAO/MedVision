print("Test 1: Scale-factor determinism — _get_pixel_size_scale_factor returns identical values on repeated calls for both 'uniform' and 'anisotropic' modes; no dataset required.")
import hashlib, re, pathlib, numpy as np

src = pathlib.Path("src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py").read_text()
scope = {"np": np, "hashlib": hashlib}
exec(re.search(r"^_SCALED_PS_LOW.*?_SCALED_PS_HIGH\s*=\s*[0-9.]+", src, re.S | re.M).group(0), scope)
exec(re.search(r"^def _get_pixel_size_scale_factor.*?^(?=def [A-Za-z_])", src, re.S | re.M).group(0), scope)
gf = scope["_get_pixel_size_scale_factor"]
doc = {"image_file": "/tmp/x.nii.gz", "slice_dim": 2, "slice_idx": 120, "taskID": 1, "label": "1"}
assert gf(doc, "uniform") == gf(doc, "uniform"), "uniform not deterministic"
assert gf(doc, "anisotropic") == gf(doc, "anisotropic"), "aniso not deterministic"
print("uniform:", gf(doc, "uniform"))
print("aniso :", gf(doc, "anisotropic"))
print("OK")
