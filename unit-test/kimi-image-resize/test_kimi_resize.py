"""
Tests for the MoonViT / Kimi image-resize rule used by the Kimi model integration.

The rule + cap table live in ONE place -- lmms_eval/models/kimi.py -- and the task layer
(medvision_utils.get_resized_img_shape) imports kimi_resized_hw() from there, so there is no
second copy to keep in sync. These tests import the shipped functions directly.

The core guarantee these tests enforce: the shape kimi_resized_hw() returns is a FIXED POINT
of MoonViT's own navit_resize_image -- feeding it back through the (re-implemented) server
algorithm leaves it unchanged with ZERO padding (scale==1.0, pad==0). That is what makes
"image size stated in the prompt == canvas the model perceives" hold, so the pixel->mm
arithmetic in TL/AD prompts stays valid.

NOTE: importing lmms_eval.models.kimi transitively pulls `transformers` (via lmms_eval.api),
so run this in an env with the eval deps installed (e.g. the `eval-kimi` conda env), not a
bare Python.

Run: pytest unit-test/kimi-image-resize/test_kimi_resize.py   (or: python test_kimi_resize.py)
"""

import os
import sys

# Make the vendored lmms_eval importable as a top-level package, as it is at eval time.
# __file__ is unit-test/kimi-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))

from lmms_eval.models.kimi import (  # noqa: E402
    SUPPORTED_MODEL_CAPS,
    _moonvit_navit_resize,
    kimi_image_caps,
    kimi_resized_hw,
)

_MODELS = ["kimi-k2.6"]
_SIZES = [(512, 512), (1024, 1024), (4000, 3000), (3000, 800), (2400, 1900),
          (1900, 2400), (333, 777), (8000, 8000), (28, 28), (1, 1)]


def _is_28_grid(h, w):
    return h % 28 == 0 and w % 28 == 0


def _is_fixed_point(h, w, model):
    """The sent image must be a true fixed point of MoonViT's navit_resize_image:
    the server applies scale==1.0 (no downscale) and pad==0 (no canvas enlargement)."""
    caps = kimi_image_caps(model)
    sh, sw, pad_h, pad_w, _ = _moonvit_navit_resize(h, w, caps)
    return (sh, sw, pad_h, pad_w) == (h, w, 0, 0)


def _within_patch_budget(h, w, model):
    caps = kimi_image_caps(model)
    return (w // caps.patch_size) * (h // caps.patch_size) <= caps.in_patch_limit


def test_small_image_floors_to_28_grid():
    # A within-budget slice (scale=1.0) is floored DOWN to the 28-px grid so MoonViT's pad-up
    # step is a no-op (512 -> 504 = 28*18, 1024 -> 1008 = 28*36).
    assert kimi_resized_hw(512, 512, "kimi-k2.6") == (504, 504)
    assert kimi_resized_hw(1024, 1024, "kimi-k2.6") == (1008, 1008)


def test_outputs_are_fixed_points():
    # The whole point: every output must be a true fixed point of MoonViT (28-grid, within
    # budget, within the 7168 px per-side cap) so the server neither downscales nor pads.
    caps = kimi_image_caps("kimi-k2.6")
    side_px_cap = caps.patch_limit_on_one_side * caps.patch_size  # 7168
    for m in _MODELS:
        for h, w in _SIZES:
            nh, nw = kimi_resized_hw(h, w, m)
            assert _is_28_grid(nh, nw), f"{m} {h}x{w} -> {nh}x{nw} not on 28-grid"
            assert _within_patch_budget(nh, nw, m), f"{m} {h}x{w} -> {nh}x{nw} over patch budget"
            assert max(nh, nw) <= side_px_cap, f"{m} {h}x{w} -> {nh}x{nw} over 7168 px side cap"
            assert _is_fixed_point(nh, nw, m), f"{m} {h}x{w} -> {nh}x{nw} is not a MoonViT fixed point"


def test_near_boundary_nonsquare_trim():
    # A near-budget NON-square image: floor-to-28 alone leaves it a few patches over
    # in_patch_limit (the budget uses integer floor div), so the trim loop must pull it back
    # under budget -- otherwise the server would re-downscale and break the fixed point.
    for h, w in [(2400, 1900), (1900, 2400), (2400, 2400), (3000, 2000)]:
        nh, nw = kimi_resized_hw(h, w, "kimi-k2.6")
        assert _within_patch_budget(nh, nw, "kimi-k2.6")
        assert _is_fixed_point(nh, nw, "kimi-k2.6")


def test_large_image_downscaled_within_budget():
    # 4000x4000 must be downscaled to fit the 16384-patch budget and land on the 28-grid.
    nh, nw = kimi_resized_hw(4000, 4000, "kimi-k2.6")
    assert _is_28_grid(nh, nw)
    assert _within_patch_budget(nh, nw, "kimi-k2.6")
    assert nh == nw  # square stays square
    # near-maximal: close to the 16384-patch budget (flooring to 28 shaves a little)
    caps = kimi_image_caps("kimi-k2.6")
    assert (nw // caps.patch_size) * (nh // caps.patch_size) > caps.in_patch_limit * 0.90


def test_aspect_ratio_preserved_within_grid():
    # 3000x800 (h x w): aspect ratio preserved within 28-grid rounding.
    nh, nw = kimi_resized_hw(3000, 800, "kimi-k2.6")
    assert _is_28_grid(nh, nw)
    assert _is_fixed_point(nh, nw, "kimi-k2.6")
    assert abs((nw / nh) - (800 / 3000)) < 0.02


def test_never_upscaled():
    for h, w in [(28, 28), (100, 50), (504, 504), (1, 1)]:
        nh, nw = kimi_resized_hw(h, w, "kimi-k2.6")
        assert nh <= max(28, h) and nw <= max(28, w)
        # minimum side is one 28-grid cell
        assert nh >= 28 and nw >= 28


def test_openrouter_model_code_normalization():
    # OpenRouter id "moonshotai/kimi-k2.6" must resolve to the same caps/result as "kimi-k2.6".
    assert kimi_resized_hw(4000, 3000, "moonshotai/kimi-k2.6") == kimi_resized_hw(4000, 3000, "kimi-k2.6")
    assert kimi_image_caps("moonshotai/kimi-k2.6") == kimi_image_caps("kimi-k2.6")


def test_caps_table_is_moonvit_caps():
    assert SUPPORTED_MODEL_CAPS
    for caps in SUPPORTED_MODEL_CAPS.values():
        # MoonViTCaps NamedTuple: (patch_size, merge_kernel_size, in_patch_limit, patch_limit_on_one_side)
        assert len(caps) == 4 and all(isinstance(x, int) for x in caps)
        assert caps.patch_size * caps.merge_kernel_size == 28


def test_unsupported_model_raises():
    # Unverified models must fail loudly (not silently fall back to a wrong budget).
    for bad in ["kimi-k2.5", "kimi-k2.7-code", "moonshotai/kimi-k99", "gpt-4o", ""]:
        try:
            kimi_image_caps(bad)
            raise AssertionError(f"kimi_image_caps({bad!r}) should have raised")
        except ValueError:
            pass
        try:
            kimi_resized_hw(512, 512, bad)
            raise AssertionError(f"kimi_resized_hw({bad!r}) should have raised")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
