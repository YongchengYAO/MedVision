# Perceived-Size Bugfix: image size & pixel scale in measurement-task prompts

**Date:** 2026-06-14
**File fixed:** `src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py` — `get_resized_img_shape` (+ the SFT call sites in `sft/sft_utils.py`)
**Test:** `unit-test/perceived-size-resize/test_perceived_size_resize.py`
**Models fixed:** `vllm_llava_onevision`, `vllm_internvl3`, `llava_med`, `huatuogpt_vision`, `healthgpt_l14`, `vllm_llama_3_2_vision`
**Tasks affected:** the measurement tasks only — `TumorLesionSize` (TL), `MaskSize`, `BiometricsFromLandmarks` (AD), and all CoT / `woMedImg` / `wVisualPrompt` / `scaledPS` variants. **Detection (`BoxCoordinate`) does NOT call `get_resized_img_shape` and is unaffected.**
**Inputs:** **non-square** slices (and, for InternVL3, large squares >633 px). Square ≤633 px datasets are unaffected.

---

## 1. The core principle — two numbers, two sources

A measurement prompt states an **image size** and a **pixel size**, and the model converts a measured structure to millimetres. These are **two different quantities** and, once a processor pads, they come from **two different shapes**:

- **image size** must equal the **resized-and-padded canvas the encoder actually perceives** (so the model's spatial frame matches what it sees).
- **pixel size** must reflect **resize only** — `pixel / (content/original)` — because padding adds no physical extent; the model measures real structures inside the content, never across black padding.

`get_resized_img_shape` therefore returns **two shapes**: `(perceived_canvas_hw, content_hw)`. The consumer uses `perceived_canvas_hw` for the stated image size and `content_hw` for the per-axis pixel ratio:

```python
img_shape_resized_hw, img_shape_content_hw = get_resized_img_shape(...)   # (perceived, content)
resized_img_h, resized_img_w = img_shape_resized_hw                       # perceived canvas
resize_ratio_h = img_shape_content_hw[0] / original_height                # content/resize-only
resize_ratio_w = img_shape_content_hw[1] / original_width
adjusted_pixel_* = pixel_* / resize_ratio_*
image_size_text = f"...{resized_img_w} x {resized_img_h}..."              # perceived canvas
pixel_size_text = f"...{adjusted_pixel_*}..."                             # resize-only
```

For **non-padding** models the two shapes are identical (the split is a no-op). They diverge only when the processor **pads**.

> Two wrong tests to avoid: (a) "physical extent is conserved" (`size × pixel = original × pixel`) holds for *any* returned value, so it can't tell right from wrong. (b) Using one shape for both fields — the original code stated the perceived canvas but derived the pixel size from it (pixel wrong); a naive "return content" fix corrects the pixel size but then states the content as the image size (image size wrong). Both fields are only correct when the two shapes are supplied separately.

### The decisive per-model test: does the processor pad?

| Processor behavior | perceived canvas vs content | image_size / pixel_size source |
|---|---|---|
| **Pad** — letterbox, pad-to-square (`expand2square`), tile-pad | canvas > content on the padded axis | image_size = **canvas**; pixel = **content** |
| **Stretch / pure resize** — content fills output | canvas == content | both from the single output |

### Worked example (160×384 slice, pixel 0.5 mm; CLIP-336 pad-to-square)

`expand2square` → 384², resize 384→336 ⇒ uniform content scale `0.875`; the encoder perceives a **336×336** canvas with content occupying **140×336** (top) + black padding below.
- **Correct:** image size = **336×336** (perceived), pixel = 0.5/0.875 = **0.571** (content scale). A structure of true height `h_mm` occupies `0.875·h_mm/0.5` perceived px → `×0.571 = h_mm`. ✓ — and the stated 336×336 matches what the model sees.
- **Wrong-pixel** (old): image size 336×336 (right) but pixel = 0.5 (canvas ratio `336/160`) → height under-estimated ~2.4×.
- **Wrong-image-size** (naive content-only fix): pixel 0.571 (right) but image size stated as 140×336 — contradicts the 336×336 the encoder sees, skewing any frame-relative reasoning.

---

## 2. How this was validated

Each branch was checked against the library version it actually installs (`requirements/requirements_eval_*.txt`), because for a served model the *engine* does the real resize. The two-number contract and the per-model perceived/content shapes were validated by subagents against pinned sources; geometry values are reproduced by a reference implementation in the unit test. Installed pins: OneVision/InternVL3 `transformers==4.57.1` + `vllm==0.10.0`; Llama-3.2 `tf 4.55.2` + `vllm 0.10.2`; llava_med `tf 4.37.2`; huatuogpt_vision `tf 4.40.0`.

---

## 3. Previous behavior → bug → fix (per affected model)

Each padding model now returns `(perceived_canvas, content)`; non-padding models return one shape used for both.

### 3.1 `vllm_llava_onevision` — letterbox (resize-then-pad)
- **Previous:** returned `select_best_resolution(img_PIL.size, ...)` (single) → image size = canvas (correct) but pixel ratio used the **padded** canvas (wrong on the padded axis); also passed PIL `(W,H)` where `select_best_resolution` expects `(H,W)` (axis transpose, hidden by swap-symmetric `image_grid_pinpoints`).
- **Fix:** `_process_img_llavaonevision` returns `(best_resolution_hw, get_patch_output_size(...))` = (perceived canvas, pre-pad content), with the `(H,W)` order corrected. image_size = canvas; pixel = content.
- 160×384 → perceived `384×384`, content `160×384`.

### 3.2 `vllm_internvl3` — dynamic tiling (stretch, no pad)
- **Previous:** hardcoded `[448,448]` — wrong canvas for any input not on a 1×1 grid (non-square, and large squares >633 px).
- **Fix:** `_process_img_internvl3` computes the tiling canvas via `transformers ... get_optimal_tiled_canvas` → `(448·rows, 448·cols)`. Stretch fills the canvas, so perceived == content (single value used for both). 1935×2400 (H×W) → `(1344,1792)`; 768²→896²; 512²/256²→448².

### 3.3 `llava_med`, `huatuogpt_vision`, `healthgpt_l14` — pad-to-square then CLIP-336
- **Previous:** hardcoded `[336,336]` (single) → image size = 336² (correct) but pixel ratio used the padded 336² (wrong on the short axis).
- **Fix:** branch returns perceived `[336,336]` and content `_padsquare_clip_content_hw(img, 336)` = `(round(H·336/max(H,W)), round(W·336/max(H,W)))`. image_size = 336²; pixel = content. 160×384 → perceived `336×336`, content `140×336`.

### 3.4 `vllm_llama_3_2_vision` — tile-pad
- **Previous:** `_process_img_llama_3_2_vision` returned the **pre-pad** resize (single) → pixel correct, but image size stated the pre-pad content while the encoder sees the padded `num_tiles·560` tile canvas (image size wrong all along — pre-existing).
- **Fix:** probe returns `(canvas, content)` where `canvas = ceil(content/tile)·tile` per axis (tile=560). image_size = canvas; pixel = content. 160×384 → content `233×560`, perceived `560×560`.

---

## 4. Branches unchanged (perceived == content)

`vllm_qwen25vl` / `_tooluse` / `lingshu` / `vllm_qwen3vl` (smart_resize, no pad), `vllm_gemma4` (floor-48, no spatial pad), `vllm_gemma3` / `medgemma` (stretch to 896²), `meddr` (stretch to 448²), `claude` / `gemini` / `openai` (client pre-resize to a fixed point; sent == perceived == content). The function returns one shape for both fields — already correct.

---

## 5. Edge cases & callers

- **All 20 callers use the return identically** — the benchmark `doc_to_text_*` (16 sites) and SFT `sft_utils.py` (4 sites): image size from `perceived`, pixel ratio from `content`. The return is never used to reshape an image. SFT only ever passes non-padding families (`qwen25vl`/`qwen2_5_vl`, `medgemma`), so `perceived == content` there (no behavior change).
- **Squares:** OneVision / CLIP trio are unchanged on squares (perceived == content == fixed). InternVL3 is correct only for squares ≤633 px (a 1×1 grid); 768²/1024² tile.
- **InternVL `downsample_ratio`** is post-ViT token compression, not a spatial change.
- **`reshape_image_hw`** is `None` for these off-the-shelf evals (raw slices); SFT forces square 512² (perceived == content).

---

## 6. Status and follow-up

- Code fix + unit test implemented; both files `py_compile` clean; geometry reproduced by the test's reference implementation (the test `pytest.skip`s where `transformers` is absent).
- **Prior results affected** (re-eval needed): TL/AD runs for the six models on padded inputs — non-square slices (AD Ceph-Biometrics-400 1935×2400; TL BraTS24-Task04 182×218) plus, for InternVL3, large squares (HNTSMRG24 768², BraTS24-Task05 672²/704²). Detection and square ≤633 px results are valid.
- See [`Model-Image-Processing.md`](Model-Image-Processing.md) for the corrected per-model pipeline reference.
