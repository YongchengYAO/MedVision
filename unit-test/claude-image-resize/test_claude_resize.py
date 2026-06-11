"""
Tests for the Anthropic image-resize rule used by the Claude model integration.

The rule + cap table live in ONE place -- lmms_eval/models/claude.py -- and the task layer
(medvision_utils.get_resized_img_shape) imports anthropic_resized_hw() from there, so there is
no second copy to keep in sync. These tests import the shipped functions directly.

NOTE: importing lmms_eval.models.claude transitively pulls `transformers` (via lmms_eval.api),
so run this in an env with the eval deps installed (e.g. the `eval-claude` conda env), not a
bare Python.

Run: pytest tests/test_claude_resize.py   (or: python tests/test_claude_resize.py)
"""

import os
import sys

# Make the vendored lmms_eval importable as a top-level package, as it is at eval time.
# __file__ is unit-test/claude-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))

from lmms_eval.models.claude import (  # noqa: E402
    SUPPORTED_MODEL_CAPS,
    anthropic_image_caps,
    anthropic_resized_hw,
)


def _assert_fits_caps(h, w, long_edge_cap, max_tokens):
    assert max(h, w) <= long_edge_cap, f"long edge {max(h, w)} > {long_edge_cap}"
    assert (h * w) / 750.0 <= max_tokens + 1, f"tokens {(h * w) / 750.0:.1f} > {max_tokens}"


def _is_28_grid(h, w):
    return h % 28 == 0 and w % 28 == 0


def test_small_image_snaps_to_28_grid():
    # Even a small slice (within caps, scale=1.0) is rounded DOWN to the 28-px grid so
    # Claude adds no bottom/right padding (512 -> 504 = 28*18, 1024 -> 1008 = 28*36).
    assert anthropic_resized_hw(512, 512, "claude-fable-5") == (504, 504)
    assert anthropic_resized_hw(1024, 1024, "claude-fable-5") == (1008, 1008)
    assert anthropic_resized_hw(512, 512, "claude-haiku-4-5") == (504, 504)


def test_token_cap_binding_high_res():
    # 4000x3000 on Fable 5: token cap (4784) binds before the 2576 px long edge
    h, w = anthropic_resized_hw(4000, 3000, "claude-fable-5")
    _assert_fits_caps(h, w, 2576, 4784)
    assert _is_28_grid(h, w)
    # near-maximal: close to the token cap (>0.90; flooring to 28 can shave up to ~5%)
    assert (h * w) / 750.0 > 4784 * 0.90
    # aspect ratio preserved within 28-grid rounding
    assert abs(w / h - 3000 / 4000) < 0.01


def test_long_edge_binding_high_res():
    # 3000x800 on Fable 5: long edge (2576) binds, token count stays below the cap
    h, w = anthropic_resized_hw(3000, 800, "claude-fable-5")
    assert h == 2576  # 2576 = 28*92, already on the grid
    _assert_fits_caps(h, w, 2576, 4784)
    assert _is_28_grid(h, w)
    assert abs(w / h - 800 / 3000) < 0.01


def test_standard_tier_caps():
    # 2000x2000 on Haiku: 1568-token cap binds (sqrt(1568*750) ~= 1084 px square)
    h, w = anthropic_resized_hw(2000, 2000, "claude-haiku-4-5")
    _assert_fits_caps(h, w, 1568, 1568)
    assert _is_28_grid(h, w)
    assert (h * w) / 750.0 > 1568 * 0.90
    assert h == w


def test_outputs_are_always_on_28_grid():
    # The whole point of the fix: every output is a multiple of 28 so Claude never pads.
    models = ["claude-fable-5", "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]
    sizes = [(512, 512), (4000, 3000), (3000, 800), (2000, 2000), (333, 777), (8000, 8000), (1, 1)]
    for m in models:
        for h, w in sizes:
            nh, nw = anthropic_resized_hw(h, w, m)
            assert _is_28_grid(nh, nw), f"{m} {h}x{w} -> {nh}x{nw} not on 28-grid"
            # also confirm it fits the model's caps (so Claude does not resize again)
            cap_long, cap_tok = anthropic_image_caps(m)
            _assert_fits_caps(nh, nw, cap_long, cap_tok)


def test_openrouter_model_code_normalization():
    # OpenRouter IDs ("anthropic/claude-opus-4.8") must select the high-res caps
    assert anthropic_resized_hw(4000, 3000, "anthropic/claude-opus-4.8") == anthropic_resized_hw(4000, 3000, "claude-opus-4-8")
    assert anthropic_resized_hw(2000, 2000, "anthropic/claude-sonnet-4.6") == anthropic_resized_hw(2000, 2000, "claude-sonnet-4-6")


def test_never_upscaled():
    for model_code in ["claude-fable-5", "claude-haiku-4-5"]:
        for h, w in [(28, 28), (100, 50), (1568, 1568)]:
            nh, nw = anthropic_resized_hw(h, w, model_code)
            assert nh <= h and nw <= w


def test_caps_table_is_non_empty_tuples():
    # The single cap table should map each model to a (long_edge_px, max_tokens) pair.
    assert SUPPORTED_MODEL_CAPS
    for caps in SUPPORTED_MODEL_CAPS.values():
        assert isinstance(caps, tuple) and len(caps) == 2


def test_unsupported_model_raises():
    # Unverified models must fail loudly (not silently fall back to a wrong cap).
    for bad in ["claude-opus-9-9", "gpt-4o", "anthropic/claude-opus-99.9", ""]:
        try:
            anthropic_image_caps(bad)
            raise AssertionError(f"anthropic_image_caps({bad!r}) should have raised")
        except ValueError:
            pass
        # anthropic_resized_hw delegates to anthropic_image_caps, so it raises too.
        try:
            anthropic_resized_hw(512, 512, bad)
            raise AssertionError(f"anthropic_resized_hw({bad!r}) should have raised")
        except ValueError:
            pass


def test_fast_and_date_suffixes_supported():
    # Suffixed ids (Fast Mode, date-pinned) share the base model's vision encoder/caps.
    assert anthropic_image_caps("claude-opus-4-8-fast") == (2576, 4784)
    assert anthropic_image_caps("anthropic/claude-opus-4.8-fast") == (2576, 4784)
    assert anthropic_image_caps("claude-haiku-4-5-20251001") == (1568, 1568)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
