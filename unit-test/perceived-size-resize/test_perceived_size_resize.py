"""Geometry checks for the perceived-size bugfix in `get_resized_img_shape`.

`get_resized_img_shape` returns TWO shapes: `(perceived_canvas_hw, content_hw)`.
  - perceived_canvas_hw -> the resized+PADDED shape the encoder sees -> stated **image size** in the prompt.
  - content_hw          -> the pre-pad / resize-only shape -> per-axis **pixel-size** ratio
                           (`resize_ratio_axis = content_axis / original_axis`).
For non-padding models the two are equal. For PADDING models (LLaVA-OneVision letterbox, CLIP-336
pad-to-square, Llama-3.2 tile-pad) the content is strictly inside the padded canvas on the padded axis,
so the two differ and BOTH must be right: image size = canvas, pixel size = content scale.

This test exercises the *real* library functions the fixed probes compose (no GPU, no model download),
and `pytest.skip`s if the eval deps are not installed.

Run inside an eval env (e.g. eval-internvl3 / eval-llava-onevision), or:
    pytest unit-test/perceived-size-resize/test_perceived_size_resize.py -v

Covers:
  - InternVL3 (stretch, no pad: perceived == content): `get_optimal_tiled_canvas` -> (448*rows, 448*cols).
    Old fixed [448,448] is correct ONLY for a 1x1 grid (squares <=633px); wrong for non-square AND large
    squares (768^2 -> 896^2, 1024^2 -> 1344^2).
  - LLaVA-OneVision (letterbox): perceived = `select_best_resolution` canvas; content = `get_patch_output_size`.
    The two differ for non-square inputs.
  - CLIP-336 trio (pad-to-square): perceived = 336x336; content = uniform `336/max(H,W)` (`_padsquare_clip_content_hw`).
  - Llama-3.2 (tile-pad): perceived = ceil(content/560)*560 (padded tile canvas); content = pre-pad resize.
"""

import math
import os
import sys

import pytest

# Make the vendored lmms_eval importable (repo root is three dirnames up from this file).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src/medvision_bm/medvision_lmms_eval"))

# llava-onevision-qwen2-*-ov-hf image_grid_pinpoints: multiples of 384, 1..6 per side, as (H, W).
ONEVISION_PINPOINTS = [[384 * i, 384 * j] for i in range(1, 7) for j in range(1, 7)]
TILE = 448  # InternVL3 vision_config.image_size
INTERNVL_MIN, INTERNVL_MAX = 1, 13  # max_dynamic_patch(12) + 1 for thumbnail (engine-faithful)


# ----------------------------- InternVL3 (stretch into tile grid) -----------------------------

# (H, W) -> expected NEW perceived (H, W) = (448*rows, 448*cols)
INTERNVL_EXPECTED = {
    (256, 256): (448, 448),    # small square -> 1x1 grid (old [448,448] correct)
    (512, 512): (448, 448),    # small square -> 1x1 grid (old correct)
    (768, 768): (896, 896),    # LARGE square -> 2x2 grid (old [448,448] WRONG)
    (1024, 1024): (1344, 1344),  # LARGE square -> 3x3 grid (old WRONG)
    (1935, 2400): (1344, 1792),  # Ceph-Biometrics-400 (landscape) -> cols=4, rows=3
    (182, 218): (1344, 1792),    # BraTS24-Task04
    (160, 384): (896, 2240),     # wide
    (160, 512): (448, 1344),
}


@pytest.mark.parametrize("orig_hw,expected_hw", list(INTERNVL_EXPECTED.items()))
def test_internvl3_canvas(orig_hw, expected_hw):
    got = pytest.importorskip(
        "transformers.models.got_ocr2.image_processing_got_ocr2"
    )
    get_optimal_tiled_canvas = got.get_optimal_tiled_canvas
    H, W = orig_hw
    cols, rows = get_optimal_tiled_canvas((H, W), (TILE, TILE), INTERNVL_MIN, INTERNVL_MAX)
    new_hw = (TILE * rows, TILE * cols)
    assert new_hw == expected_hw, f"{orig_hw}: got {new_hw}, expected {expected_hw}"


def test_internvl3_old_correct_only_for_1x1_grid():
    """The old fixed [448,448] is right ONLY for inputs that resolve to a 1x1 grid.

    That is small squares (<=633px) AND small near-square images -- NOT all squares: large squares
    (768^2, 1024^2) tile and are mis-scaled by the old code, same as non-square inputs.
    """
    got = pytest.importorskip("transformers.models.got_ocr2.image_processing_got_ocr2")
    get_optimal_tiled_canvas = got.get_optimal_tiled_canvas
    old = (448, 448)

    def new(H, W):
        c, r = get_optimal_tiled_canvas((H, W), (TILE, TILE), INTERNVL_MIN, INTERNVL_MAX)
        return (TILE * r, TILE * c)

    # small squares: old == new (old was correct)
    for s in (256, 512, 632):
        assert new(s, s) == old, f"small square {s} should be 1x1 -> {old}"
    # large squares: old != new (old was WRONG even for square)
    for s in (768, 1024):
        assert new(s, s) != old, f"large square {s} should tile, not {old}"
    # non-square: old != new
    for hw in ((1935, 2400), (182, 218), (160, 384)):
        assert new(*hw) != old, f"non-square {hw} should differ from {old}"


# ----------------------------- LLaVA-OneVision (letterbox -> pre-pad content) -----------------------------

# (H, W) -> (expected_content_hw, expected_padded_canvas_hw)
ONEVISION_EXPECTED = {
    (256, 256): ((384, 384), (384, 384)),     # square: content == canvas (old correct)
    (512, 512): ((768, 768), (768, 768)),     # square: content == canvas (old correct)
    (160, 384): ((160, 384), (384, 384)),     # non-square: content < canvas on padded axis
    (1935, 2400): ((1858, 2304), (1920, 2304)),  # Ceph: content height 1858 < canvas 1920
    (182, 218): ((321, 384), (384, 384)),
}


@pytest.mark.parametrize("orig_hw,expected", list(ONEVISION_EXPECTED.items()))
def test_onevision_content_vs_canvas(orig_hw, expected):
    ipu = pytest.importorskip("transformers.image_processing_utils")
    select_best_resolution = ipu.select_best_resolution
    get_patch_output_size = ipu.get_patch_output_size
    iu = pytest.importorskip("transformers.image_utils")
    np = pytest.importorskip("numpy")

    expected_content, expected_canvas = expected
    H, W = orig_hw
    img = np.zeros((H, W, 3), dtype=np.uint8)
    best_hw = select_best_resolution((H, W), ONEVISION_PINPOINTS)  # the OLD (canvas) return
    content_hw = tuple(get_patch_output_size(img, best_hw, iu.ChannelDimension.LAST))  # the FIX
    assert tuple(best_hw) == expected_canvas, f"{orig_hw}: canvas {tuple(best_hw)} != {expected_canvas}"
    assert content_hw == expected_content, f"{orig_hw}: content {content_hw} != {expected_content}"
    if H != W:
        assert content_hw != tuple(best_hw), f"{orig_hw}: content must differ from padded canvas"


def test_onevision_content_is_aspect_preserving():
    """For non-square inputs the content scale is ~uniform on both axes (min-scale + ceil rounding)."""
    ipu = pytest.importorskip("transformers.image_processing_utils")
    iu = pytest.importorskip("transformers.image_utils")
    np = pytest.importorskip("numpy")
    H, W = 160, 384
    img = np.zeros((H, W, 3), dtype=np.uint8)
    best_hw = ipu.select_best_resolution((H, W), ONEVISION_PINPOINTS)
    ch, cw = ipu.get_patch_output_size(img, best_hw, iu.ChannelDimension.LAST)
    assert abs(ch / H - cw / W) < 0.02  # same scale within rounding


# ----------------------------- CLIP-336 trio (pad-to-square -> uniform content) -----------------------------

CLIP_EXPECTED = {
    (336, 336): (336, 336),
    (256, 256): (336, 336),   # square: uniform (old [336,336] correct)
    (512, 512): (336, 336),
    (160, 384): (140, 336),   # non-square: short axis shrinks (old WRONG)
    (1935, 2400): (271, 336),
    (182, 218): (281, 336),
}


@pytest.mark.parametrize("orig_hw,expected_hw", list(CLIP_EXPECTED.items()))
def test_clip336_uniform_content(orig_hw, expected_hw):
    try:
        from lmms_eval.tasks.medvision.medvision_utils import _padsquare_clip_content_hw
    except Exception as e:  # heavy module imports (torch, etc.) may be absent
        pytest.skip(f"medvision_utils import failed: {e}")
    import numpy as np

    H, W = orig_hw
    img = np.zeros((H, W), dtype=np.uint8)
    got = _padsquare_clip_content_hw(img, 336)
    assert got == expected_hw, f"{orig_hw}: got {got}, expected {expected_hw}"
    if H != W:
        assert got != (336, 336), f"{orig_hw}: non-square must differ from old [336,336]"


def test_clip336_reference_formula():
    """Reference check independent of the import (documents the contract)."""
    def ref(H, W, size=336):
        m = max(H, W)
        return (round(H * size / m), round(W * size / m))
    for hw, exp in CLIP_EXPECTED.items():
        assert ref(*hw) == exp, f"{hw}: ref {ref(*hw)} != {exp}"


# ----------------------------- Llama-3.2 (tile-pad: perceived canvas = ceil(content/560)*560) -----------------------------

def test_llama_perceived_canvas_formula():
    """The padded tile canvas (stated image size) is ceil(content/tile)*tile per axis; content drives pixel size."""
    TILE = 560
    def canvas(content_hw):
        return (math.ceil(content_hw[0] / TILE) * TILE, math.ceil(content_hw[1] / TILE) * TILE)
    # content 233x560 (from a 160x384 input, upscaled aspect-fit) -> 1x1 tile canvas 560x560
    assert canvas((233, 560)) == (560, 560)
    # content fully fills one tile -> unchanged
    assert canvas((560, 560)) == (560, 560)
    # multi-tile content -> ceils up to the tile grid
    assert canvas((900, 560)) == (1120, 560)
    assert canvas((560, 1100)) == (560, 1120)
    # padded axis: canvas strictly exceeds content -> image_size != content (the whole point of the split)
    assert canvas((233, 560)) != (233, 560)


# ----------------------------- two-value contract: perceived vs content -----------------------------

def test_perceived_differs_from_content_only_when_padded():
    """For padding models perceived (image size) != content (pixel basis) on non-square; for stretch they match."""
    ipu = pytest.importorskip("transformers.image_processing_utils")
    iu = pytest.importorskip("transformers.image_utils")
    got = pytest.importorskip("transformers.models.got_ocr2.image_processing_got_ocr2")
    np = pytest.importorskip("numpy")

    H, W = 160, 384
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # OneVision (letterbox): perceived (canvas) != content
    canvas = tuple(ipu.select_best_resolution((H, W), ONEVISION_PINPOINTS))
    content = tuple(ipu.get_patch_output_size(img, canvas, iu.ChannelDimension.LAST))
    assert canvas == (384, 384) and content == (160, 384) and canvas != content
    # InternVL3 (stretch): perceived == content (no pad)
    cols, rows = got.get_optimal_tiled_canvas((H, W), (TILE, TILE), INTERNVL_MIN, INTERNVL_MAX)
    internvl_canvas = (TILE * rows, TILE * cols)
    assert internvl_canvas == (896, 2240)  # content fills the stretched canvas; perceived == content


if __name__ == "__main__":
    # Reference-only smoke run (no transformers needed): prints the documented contract values.
    def ref_clip(H, W, size=336):
        m = max(H, W)
        return (round(H * size / m), round(W * size / m))
    print("CLIP-336 reference content sizes:")
    for hw in CLIP_EXPECTED:
        print(f"  {hw} -> {ref_clip(*hw)}")
