"""
Tests for the OpenAI image-resize rule used by the OpenAI (GPT) model integration.

The rule + cap table live in ONE place -- lmms_eval/models/openai.py -- and the task layer
(medvision_utils.get_resized_img_shape) imports openai_resized_hw() from there, so there is
no second copy to keep in sync. These tests import the shipped functions directly.

NOTE: importing lmms_eval.models.openai transitively pulls `transformers` (via lmms_eval.api),
so run this in an env with the eval deps installed (e.g. the `eval-openai` conda env), not a
bare Python.

Run: pytest unit-test/openai-image-resize/test_openai_resize.py
     (or: python unit-test/openai-image-resize/test_openai_resize.py)
"""

import os
import sys

# Make the vendored lmms_eval importable as a top-level package, as it is at eval time.
# __file__ is unit-test/openai-image-resize/<this>; three dirnames up -> repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "src/medvision_bm/medvision_lmms_eval"))

from lmms_eval.models.openai import (  # noqa: E402
    SUPPORTED_MODEL_CAPS,
    openai_image_caps,
    openai_resized_hw,
)


def _is_32_grid(h, w):
    return h % 32 == 0 and w % 32 == 0


def _patch_count(h, w):
    # both sides are 32-aligned in our outputs, so this is exact (no ceil needed)
    return (h // 32) * (w // 32)


def test_patch_small_image_snaps_to_32_grid():
    # A within-budget image is sent unchanged by the server, but a non-32-aligned one
    # would be covered by overhanging edge patches ("a patch may extend beyond the image
    # boundary"), enlarging the perceived patch grid. Rounding DOWN to the 32-px grid
    # removes the overhang (512 stays, 520 -> 512, 1000 -> 992 = 32*31).
    assert openai_resized_hw(512, 512, "gpt-5.5") == (512, 512)
    assert openai_resized_hw(520, 520, "gpt-5.5") == (512, 512)
    assert openai_resized_hw(1000, 1000, "gpt-5.4-mini") == (992, 992)


def test_patch_budget_binding_flagship():
    # 1800x2400 on gpt-5.5 (high detail: 2500 patches / 2048 px): the patch budget binds
    h, w = openai_resized_hw(1800, 2400, "gpt-5.5")
    assert _is_32_grid(h, w)
    assert _patch_count(h, w) <= 2500
    assert max(h, w) <= 2048
    # near-maximal: budget well used (>90%; flooring to 32 can shave a few %)
    assert _patch_count(h, w) > 2500 * 0.90
    # aspect ratio preserved within 32-grid rounding
    assert abs(w / h - 2400 / 1800) < 0.05


def test_patch_max_dimension_binding_flagship():
    # 1000x3000 on gpt-5.5: the 2048-px max dimension binds before the patch budget
    h, w = openai_resized_hw(1000, 3000, "gpt-5.5")
    assert w == 2048  # 2048 = 32*64, already on the grid
    assert _is_32_grid(h, w)
    assert _patch_count(h, w) <= 2500
    assert abs(w / h - 3000 / 1000) < 0.1


def test_patch_mini_tier_budget():
    # 1800x2400 on gpt-5.4-mini (1536-patch budget, no px cap)
    h, w = openai_resized_hw(1800, 2400, "gpt-5.4-mini")
    assert _is_32_grid(h, w)
    assert _patch_count(h, w) <= 1536
    assert _patch_count(h, w) > 1536 * 0.90
    # mini budget is tighter than the flagship's, so the output is smaller
    fh, fw = openai_resized_hw(1800, 2400, "gpt-5.5")
    assert h < fh and w < fw


def test_patch_outputs_always_on_32_grid_and_within_caps():
    models = ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-mini", "gpt-5-nano", "o4-mini"]
    sizes = [(512, 512), (4000, 3000), (3000, 800), (2000, 2000), (333, 777), (8000, 8000), (1, 1)]
    for m in models:
        _, budget, max_dim = openai_image_caps(m)
        for h, w in sizes:
            nh, nw = openai_resized_hw(h, w, m)
            assert _is_32_grid(nh, nw), f"{m} {h}x{w} -> {nh}x{nw} not on 32-grid"
            assert _patch_count(nh, nw) <= budget, f"{m} {h}x{w} -> {nh}x{nw} over budget"
            assert max_dim is None or max(nh, nw) <= max_dim


def test_tile_small_image_unchanged():
    # Tile family: an image with long edge <= 2048 and short edge <= 768 satisfies both
    # server resize conditions, so it is sent unchanged (no grid floor -- OpenAI documents
    # no padding to tile boundaries).
    assert openai_resized_hw(512, 512, "gpt-4o") == (512, 512)
    assert openai_resized_hw(600, 800, "gpt-4o") == (600, 800)
    assert openai_resized_hw(768, 2048, "gpt-4.1") == (768, 2048)


def test_tile_short_edge_binding():
    # 4000x3000 on gpt-4o: the 768-px short edge binds (scale 0.256)
    h, w = openai_resized_hw(4000, 3000, "gpt-4o")
    assert (h, w) == (1024, 768)
    assert abs(w / h - 3000 / 4000) < 0.01


def test_tile_long_edge_binding():
    # 3000x900 on gpt-4o: the 2048-px long edge binds (900 * 2048/3000 = 614 < 768)
    h, w = openai_resized_hw(3000, 900, "gpt-4o")
    assert h == 2048
    assert w <= 768
    assert abs(w / h - 900 / 3000) < 0.01


def test_openrouter_model_code_normalization():
    # OpenRouter IDs ("openai/gpt-5.5") must select the same caps as the bare OpenAI id
    assert openai_resized_hw(1800, 2400, "openai/gpt-5.5") == openai_resized_hw(1800, 2400, "gpt-5.5")
    # gpt-5.5-pro shares the gpt-5.5 vision family/caps (live-probed 2026-06-13)
    assert openai_image_caps("openai/gpt-5.5-pro") == openai_image_caps("gpt-5.5") == ("patch", 2500, 2048)
    assert openai_resized_hw(4000, 3000, "openai/gpt-4o") == openai_resized_hw(4000, 3000, "gpt-4o")


def test_date_snapshot_suffix_supported():
    # Dated snapshot ids share the base model's vision rule (e.g. "gpt-4o-2024-11-20")
    assert openai_image_caps("gpt-4o-2024-11-20") == openai_image_caps("gpt-4o")
    assert openai_image_caps("openai/gpt-5.5-2026-01-15") == openai_image_caps("gpt-5.5")


def test_never_upscaled():
    for model_code in ["gpt-5.5", "gpt-5.4-mini", "gpt-4o"]:
        for h, w in [(32, 32), (100, 50), (512, 512), (768, 768)]:
            nh, nw = openai_resized_hw(h, w, model_code)
            assert nh <= h and nw <= w


def test_caps_table_shape():
    # Each entry maps a model to (rule_family, cap_a, cap_b)
    assert SUPPORTED_MODEL_CAPS
    for family, cap_a, cap_b in SUPPORTED_MODEL_CAPS.values():
        assert family in ("patch", "tile")
        assert isinstance(cap_a, int)
        assert cap_b is None or isinstance(cap_b, int)


def test_unsupported_model_raises():
    # Unverified models must fail loudly (not silently fall back to a wrong rule). The
    # exact-match rule is the point of several of these: under prefix matching "gpt-5.6"/
    # "gpt-5-pro" would inherit a sibling's caps. "gpt-5" base is intentionally unlisted
    # (its tile-vs-patch family is unconfirmed across doc reads), so it must raise too.
    for bad in ["gpt-5", "gpt-5.6", "gpt-5-pro", "gpt-5-chat-latest", "gpt-4.1-mini", "claude-fable-5", "openai/gpt-99", ""]:
        try:
            openai_image_caps(bad)
            raise AssertionError(f"openai_image_caps({bad!r}) should have raised")
        except ValueError:
            pass
        # openai_resized_hw delegates to openai_image_caps, so it raises too.
        try:
            openai_resized_hw(512, 512, bad)
            raise AssertionError(f"openai_resized_hw({bad!r}) should have raised")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
