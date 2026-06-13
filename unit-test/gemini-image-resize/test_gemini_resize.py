"""
Tests for the Gemini image-resize rule used by the unified Gemini model integration.

The rule + supported-model table live in ONE place -- lmms_eval/models/gemini.py -- and the
task layer (medvision_utils.get_resized_img_shape) imports gemini_resized_hw() from there, so
there is no second copy to keep in sync. These tests import the shipped functions directly.

The Gemini rule is PASS-THROUGH (identity) with a single >3072-px long-edge guard: unlike
Claude, Gemini does not pad the canvas for in-cap images (its tiling is crop-based and the
spatial contract is normalized to the input image), so no grid rounding is needed and the
resize ratio is 1.0 for every MedVision slice.

NOTE: importing lmms_eval.models.gemini transitively pulls `transformers` (via lmms_eval.api),
so run this in an env with the eval deps installed (e.g. the `eval-gemini` conda env), not a
bare Python. The google-genai / openai SDKs are NOT needed (they are imported lazily).

Run: pytest unit-test/gemini-image-resize/test_gemini_resize.py   (or: python <this file>)
"""

import os
import sys

# Make the vendored lmms_eval importable as a top-level package, as it is at eval time.
# __file__ is unit-test/gemini-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))

from lmms_eval.models.gemini import (  # noqa: E402
    SUPPORTED_MODEL_CAPS,
    gemini_image_caps,
    gemini_resized_hw,
)


def test_medvision_sizes_pass_through():
    # The core property: every MedVision slice (max 1935x2400) is sent unchanged, so the
    # stated image size and pixel size are the literal truth (resize ratio 1.0).
    for model in ["gemini-3.1-pro-preview", "gemini-2.5-pro"]:
        for h, w in [(512, 512), (256, 256), (240, 155), (1935, 2400), (2400, 1935), (3072, 3072)]:
            assert gemini_resized_hw(h, w, model) == (h, w), f"{model} {h}x{w} not pass-through"


def test_cap_binding_downscale():
    # Only >3072 long edges are downscaled (pre-empting the server's own downscale+pad path).
    h, w = gemini_resized_hw(4000, 3000, "gemini-3.1-pro-preview")
    assert max(h, w) <= 3072
    assert h == 3072  # long edge lands exactly on the cap
    # aspect ratio preserved within integer flooring
    assert abs(w / h - 3000 / 4000) < 0.01


def test_outputs_always_fit_cap():
    sizes = [(512, 512), (4000, 3000), (3000, 8000), (3073, 3073), (1, 1), (10000, 100)]
    for model in SUPPORTED_MODEL_CAPS:
        for h, w in sizes:
            nh, nw = gemini_resized_hw(h, w, model)
            cap = gemini_image_caps(model)["long_edge_cap"]
            assert max(nh, nw) <= cap, f"{model} {h}x{w} -> {nh}x{nw} exceeds cap {cap}"
            assert nh >= 1 and nw >= 1


def test_never_upscaled():
    for model in ["gemini-3.1-pro-preview", "gemini-2.5-flash"]:
        for h, w in [(28, 28), (100, 50), (3072, 3072)]:
            nh, nw = gemini_resized_hw(h, w, model)
            assert nh <= h and nw <= w


def test_openrouter_model_code_normalization():
    # OpenRouter IDs ("google/gemini-2.5-pro") must resolve to the same entry as the bare code.
    assert gemini_resized_hw(4000, 3000, "google/gemini-2.5-pro") == gemini_resized_hw(4000, 3000, "gemini-2.5-pro")
    assert gemini_image_caps("google/gemini-3.1-pro-preview") == gemini_image_caps("gemini-3.1-pro-preview")


def test_dated_suffixes_use_longest_prefix():
    # Dated/preview variants share the base model's image handling (longest prefix wins),
    # and "gemini-2.5-flash-lite" must not be shadowed by the shorter "gemini-2.5-flash".
    assert gemini_image_caps("gemini-2.5-flash-preview-09-2025")["series"] == "2.5"
    assert gemini_image_caps("gemini-2.5-flash-lite")["series"] == "2.5"
    assert gemini_image_caps("google/gemini-2.5-flash-lite-preview-09-2025")["series"] == "2.5"


def test_unsupported_model_raises():
    # Unverified models must fail loudly (not silently fall back to a wrong rule).
    # NOTE: "gemini-3-pro-preview" was retired 2026-03-09 and was deliberately NOT added;
    # "gemini-3-pro" never existed as a stable code.
    for bad in ["gemini-3-pro", "gemini-3-pro-preview", "gemini-1.5-pro", "gpt-4o", "google/gemini-9.9-pro", ""]:
        try:
            gemini_image_caps(bad)
            raise AssertionError(f"gemini_image_caps({bad!r}) should have raised")
        except ValueError:
            pass
        # gemini_resized_hw delegates to gemini_image_caps, so it raises too.
        try:
            gemini_resized_hw(512, 512, bad)
            raise AssertionError(f"gemini_resized_hw({bad!r}) should have raised")
        except ValueError:
            pass


def test_caps_table_entries_are_well_formed():
    # The single table should map each model to {"series": "2.5"|"3", "long_edge_cap": int},
    # and both series must be represented.
    assert SUPPORTED_MODEL_CAPS
    series_seen = set()
    for caps in SUPPORTED_MODEL_CAPS.values():
        assert caps["series"] in ("2.5", "3")
        assert isinstance(caps["long_edge_cap"], int) and caps["long_edge_cap"] > 0
        series_seen.add(caps["series"])
    assert series_seen == {"2.5", "3"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
