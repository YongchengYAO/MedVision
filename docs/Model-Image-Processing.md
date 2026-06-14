# Model Image Processing in MedVision

How each benchmark model's *perceived* image resolution is determined, and how the image-size / pixel-size statements in the quantitative-task prompts are kept consistent with it.

> Companion to the [New models guide](New-Models-Guide.md). This page documents the per-model image-processing pipeline as implemented **today**. For the history and rationale of the perceived-size correctness fixes (LLaVA-OneVision, InternVL3, the CLIP-336 trio), see [`Perceived-Size-Bugfix.md`](Perceived-Size-Bugfix.md).

## Why this matters

MedVision's quantitative tasks — Tumor/Lesion size (**TL**) and Angle/Distance (**AD**) — put the **image size** and **pixel size** into the text prompt, and the model must do the pixel→mm arithmetic itself. These numbers must describe the resolution the model's vision encoder **actually perceives after its internal resize** — not the raw NIfTI slice size. If they don't, the model reasons against a different scale than the ground truth assumes and every measurement is wrong.

Because every VLM resizes differently (fixed square, "smart resize", tile grids, server-side API rules), the perceived size is model-specific. The invariant is maintained **independently for each axis** (height and width):

```
stated_size_axis × stated_pixel_size_axis == original_size_axis × original_pixel_size_axis   (physical extent, mm)
```

The prompt states **two** quantities, and once a processor **pads** they come from **two different shapes**, so `get_resized_img_shape` returns both — `(perceived_canvas_hw, content_hw)`:

- **image size** = the **resized-and-padded canvas the encoder perceives** (`perceived_canvas_hw`), so the model's spatial frame matches what it sees.
- **pixel size** = adjusted by the **resize-only** scale (`resize_ratio_axis = content_axis / original_axis`, from `content_hw`), because padding adds no physical extent — the model measures real structures inside the content, never across black padding.

For a processor that **stretches/resizes** (content fills the output) the two shapes are identical. For one that **pads** (letterbox or pad-to-square) `perceived_canvas > content` on the padded axis, and the split keeps both fields correct. (Conserving physical extent alone is not a sufficient test — it holds for any returned value.)

**Detection tasks do not need any of this**: `doc_to_text_BoxCoordinate*` asks for relative `[0,1]` coordinates and never calls `get_resized_img_shape()`. The per-model machinery below applies only to the measurement (TL/AD) prompts.

## How it works

Pipeline at prompt-build time (per sample), in [`medvision_utils.py`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py):

1. **Optional input reshape** — if `reshape_image_hw` is passed via `--model_args` (parsed in `evaluator.py` and injected into `lmms_eval_specific_kwargs`), the 2D slice is reshaped at NIfTI load (`_load_nifti_2d(new_shape_hw=...)`) for **both** `doc_to_visual` and `doc_to_text`, so the probe and the model always see the same input.
2. **Perceived-size lookup** — `get_resized_img_shape(model_name, img_2d_raw, extra_kwargs)` dispatches on `model_name` (the `--model` CLI key) and returns `(perceived_canvas_hw, content_hw)` — the padded canvas the encoder sees (for the stated image size) and the resize-only content shape (for the pixel ratio). For non-padding models the two are equal. Unknown names **raise** — deliberately loud, so a new model cannot silently run with a wrong scale.
3. **Per-axis pixel-size adjustment** — each TL/AD `doc_to_text` computes `resize_ratio_h`, `resize_ratio_w` **independently** and divides the pixel sizes by them. This conserves physical extent per axis and absorbs *anisotropic* resizes (several models below are only approximately aspect-preserving).
4. The prompt states `The image size is {W} pixels (width) x {H} pixels (height).` and the adjusted pixel sizes.

`model_name` / `model_hf` are injected at run time from `--model` / `--model_args model_hf=...` (no task-YAML edits needed). The same branches also accept SFT aliases (`qwen25vl`, `gemma4`, `llama_3_2_vision`, …) because SFT training reuses `get_resized_img_shape()`.

## Summary table

One row per active key in [`AVAILABLE_MODELS`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/models/__init__.py). Strategy: **A** = fixed perceived size (processor stretches to a fixed square; content fills it), **B** = input-dependent perceived size (computed/probed per image), **C** = API rule owned by the model file (Claude/OpenAI: client pre-resize to a *fixed point* — an image the server's own resize leaves unchanged; Gemini: pass-through), **D** = no TL/AD branch.

| Model key | Strategy | Perceived-size rule |
|---|---|---|
| `vllm_qwen25vl` | B | smart-resize, 28-divisible; probe reads `image_grid_thw × patch_size` |
| `vllm_qwen25vl_tooluse` | B | same branch as `vllm_qwen25vl` |
| `vllm_qwen3vl` | B | same mechanism (`image_grid_thw × patch_size`) |
| `lingshu` | B | same Qwen2VL smart-resize; processor runs client-side |
| `vllm_llama_3_2_vision` | B | aspect-fit onto ≤`max_image_tiles` 560×560 tile canvas; image_size = padded tile canvas, pixel = pre-pad content |
| `vllm_llava_onevision` | B | letterbox into anyres 384-px-tile canvas; image_size = padded canvas, pixel = pre-pad content |
| `vllm_gemma4` | B | patch-grid extent from `image_position_ids` (floor-48 resize, ≤2520 patches) |
| `vllm_internvl3` | B | dynamic tiling → `448·cols × 448·rows` canvas (stretch, no pad) |
| `vllm_gemma3` | A | fixed 896×896 (stretch) |
| `medgemma` | A | fixed 896×896 (stretch) |
| `meddr` | A | fixed 448×448 (single-tile stretch) |
| `llava_med` | B | pad-to-square then CLIP-336; image_size = 336×336, pixel = content `336/max(H,W)` |
| `huatuogpt_vision` | B | pad-to-square then CLIP-336; image_size = 336×336, pixel = content `336/max(H,W)` |
| `healthgpt_l14` | B | expand2square then CLIP-336; image_size = 336×336, pixel = content `336/max(H,W)` |
| `claude` | C | client pre-resize to fixed point: `min(1, cap/long_edge, √(tokens·750/area))` then per-side floor-28 |
| `gemini` | C | pass-through (sent size == perceived canvas); only >3072 px long edge is downscaled |
| `openai` | C | client pre-resize to fixed point: patch models budget-fit then floor-32; tile models `min(1, 2048/long, 768/short)` |
| `healthgpt_xl32` | D | no branch → raises on TL/AD; Detection-safe |

Commented-out registry keys (not in the benchmark): `qwen2_5_vl`, `internvl3` (HF backends, replaced by vLLM variants), `llava_onevision` (HF backend; its alias is still accepted by the branch), `llama4`, and `biomedgpt` — the latter is the only model that force-resizes client-side to a fixed 480×480 (BICUBIC) in its own file (`biomedgpt.py:64-65`) and has no TL/AD branch.

## Official documentation

Durable, human-readable references for each model's image-preprocessing behavior, to read alongside the source references in each strategy section. Where a model card omits the resize math, the architectural-base docs page is the authoritative anchor (noted inline). Links verified 2026-06-14.

| Model(s) | Official docs | What it confirms |
|---|---|---|
| `vllm_qwen25vl`, `vllm_qwen25vl_tooluse` | [transformers qwen2_5_vl](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) | `Qwen2VLImageProcessor` with `min_pixels`/`max_pixels` driving smart-resize (28-multiple). |
| `vllm_qwen3vl` | [transformers qwen3_vl](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) | Same `image_grid_thw × patch_size` dynamic-resolution lineage. |
| `lingshu` | [Lingshu-7B card](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) · resize anchor: [qwen2_5_vl](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl) | Card loads via `Qwen2_5_VLForConditionalGeneration` + `qwen_vl_utils`; resize rule lives in the Qwen2.5-VL docs. |
| `vllm_llama_3_2_vision` | [transformers mllama](https://huggingface.co/docs/transformers/model_doc/mllama) | `MllamaImageProcessor`: aspect-fit into ≤`max_image_tiles` 560×560 tiles, then pad; `aspect_ratio_ids`. |
| `vllm_llava_onevision` | [transformers llava_onevision](https://huggingface.co/docs/transformers/model_doc/llava_onevision) | Anyres tiling over a SigLIP encoder (resize-then-pad letterbox). |
| `vllm_gemma4` | [transformers gemma4](https://huggingface.co/docs/transformers/model_doc/gemma4) | **States in prose** the divisible-by-48 rule (patch 16 × pooling 3) + variable-resolution patch budget. |
| `vllm_gemma3` | [transformers gemma3](https://huggingface.co/docs/transformers/model_doc/gemma3) | `Gemma3ImageProcessor`, `do_pan_and_scan`, 896 crop size. |
| `medgemma` | [medgemma-4b-it card](https://huggingface.co/google/medgemma-4b-it) · processor: [gemma3](https://huggingface.co/docs/transformers/model_doc/gemma3) | Card: images "normalized to 896×896 … encoded to 256 tokens"; Gemma3 docs for the processor mechanics. |
| `meddr` | [MedDr_0401 card](https://huggingface.co/Sunanhe/MedDr_0401) · 448 base: [internvl](https://huggingface.co/docs/transformers/model_doc/internvl) | Card has no resize text; the 448 InternViT base resolution is documented via the InternVL lineage. |
| `vllm_internvl3` | [InternVL3-38B card](https://huggingface.co/OpenGVLab/InternVL3-38B) | **Documents** dynamic 448-tile preprocessing (`dynamic_preprocess`, `min/max_dynamic_patch`, `use_thumbnail`). |
| `llava_med`, `huatuogpt_vision`, `healthgpt_l14` | [openai/clip-vit-large-patch14-336 card](https://huggingface.co/openai/clip-vit-large-patch14-336) · [transformers clip](https://huggingface.co/docs/transformers/model_doc/clip) | Shared encoder; `preprocessor_config.json` = resize-336 + center-crop. Per-model square-padding (`pad`/`expand2square`) is repo-config, see GitHub source. |
| `claude` | [Anthropic vision](https://platform.claude.com/docs/en/docs/build-with-claude/vision) | 28×28-patch tokenization, long-edge cap, resize-preserving-aspect then pad to a 28-multiple. |
| `gemini` | [Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding) | `media_resolution` control, 768×768-crop tiling, 258-token base for ≤384-px images. |
| `openai` | [OpenAI images & vision](https://developers.openai.com/api/docs/guides/images-vision) | Patch-based (gpt-5.x, 32-px patches) vs tile-based (GPT-4o/o-series, 512-px tiles) families; `detail` modes; token budgets. |

## Strategy A — fixed perceived size

The model's processor **stretches** every image to one fixed square (no aspect preservation, no padding), so the content fills the canvas and `get_resized_img_shape` returns a hardcoded constant. The anisotropic per-axis ratio is physically real and handled by the per-axis pixel-size adjustment. Each branch keeps a dynamic probe (`_process_img_<model>`) commented out "for debugging only", used to confirm the fixed square.

| Models | Size | Basis |
|---|---|---|
| `vllm_gemma3`, `medgemma` | [896, 896] | `Gemma3ImageProcessor` stretch-to-896 (`do_pan_and_scan` is `null`/off in both checkpoints; MedVision passes no override). |
| `meddr` | [448, 448] | `Sunanhe/MedDr_0401`: `pad2square: false` and no dynamic-tiling fields → single-tile `Resize((448,448))` stretch. |

Notes:
- **`do_pan_and_scan` is off by default** for `gemma3`/`medgemma`; if it were enabled the processor would emit multiple 896 crops plus a global view, which the single constant would no longer describe. MedVision never sets it.
- **MedDr is *not* the InternVL3 case.** Despite sharing InternVL-chat lineage, `MedDr_0401`'s config omits `dynamic_image_size`/`max_dynamic_patch`/`use_thumbnail`, so tiling is off; it is a plain single-tile 448 stretch.
- None of these wrappers resize client-side; the raw (optionally `reshape_image_hw`-reshaped) image goes to the engine/processor, which applies the fixed resize internally.

## Strategy B — input-dependent perceived size

The perceived size depends on the input, so the branch computes it per image — either by probing the real image processor, by composing the processor's own geometry helpers, or (for the pad-to-square CLIP models) in closed form. Each branch returns `(perceived_canvas_hw, content_hw)`: the padded canvas the encoder sees (→ stated image size) and the resize-only content (→ pixel ratio). For the non-padding probes below (Qwen family, Gemma 4) the two are identical; the padding probes (Llama-3.2, LLaVA-OneVision, CLIP-336 trio) return them distinctly.

- **Qwen2.5-VL family** (`vllm_qwen25vl`, `vllm_qwen25vl_tooluse`, `lingshu`; SFT alias `qwen25vl`) — `_process_img_qwen25vl` / `_process_img_lingshu`: `Qwen2VLImageProcessor.smart_resize` rounds each side to a multiple of `patch_size(14) × merge_size(2) = 28` and resizes (no spatial padding). Recovered from the processor's patch-grid output as `image_grid_thw[1:3] × patch_size`. The probe runs the real processor, and (for `lingshu`) literally client-side.
- **Qwen3-VL** (`vllm_qwen3vl`; alias `qwen3vl`) — `_process_img_qwen3vl`: same `image_grid_thw × patch_size` mechanism, config-driven.
- **Llama-3.2-Vision** (`vllm_llama_3_2_vision`; alias `llama_3_2_vision`) — `_process_img_llama_3_2_vision`: a `MllamaImageProcessor` subclass replays preprocessing up to (and only) the resize step to get the aspect-fitted **pre-pad content** on the optimal canvas of up to `max_image_tiles` 560×560 tiles. The processor then pads to the tile grid, so the probe returns `perceived = ceil(content/560)·560` per axis (the padded tile canvas the encoder sees → image size) and `content` (the resize-only shape → pixel ratio; aspect is carried separately via `aspect_ratio_ids`). vLLM 0.10.2 uses `MllamaProcessor` server-side.
- **LLaVA-OneVision** (`vllm_llava_onevision`; alias `llava_onevision`) — `_process_img_llavaonevision`: the processor **letterboxes** — `select_best_resolution` picks an anyres canvas (multiples of 384), the image is aspect-preserving-resized into it (`get_patch_output_size`, min-scale + ceil), **then padded**. The probe returns **both**: `perceived = best_resolution_hw` (the padded canvas → image size) and `content = get_patch_output_size(np.array(img), best_hw, ChannelDimension.LAST)` (pre-pad → pixel ratio), passing the original size as `(H, W)` (`img_PIL.size[::-1]`) per the helper's contract. Both helpers are from `transformers.image_processing_utils`; vLLM 0.10.0 runs this same HF processor server-side.
- **Gemma 4** (`vllm_gemma4`; alias `gemma4`) — `_process_img_gemma4`: Gemma4's processor outputs a flattened, sequence-padded patch list (`pixel_values: [batch, max_patches, patch²·3]`, defaults `max_patches = 280·3² = 2520`, patch dim `16²·3 = 768` — **constants, never the image size**). The probe recovers `(H, W)` from the extent of the valid (non `-1`) entries of `image_position_ids × patch_size`. The resize is only approximately aspect-preserving: one scale factor fits the patch budget, then each side is independently floored to a multiple of 48 (`pooling_kernel_size × patch_size`); upscaling allowed; no spatial padding. The per-axis pixel-size adjustment absorbs the ≤1-step anisotropy.
- **InternVL3** (`vllm_internvl3`; alias `internvl3`) — `_process_img_internvl3`: the engine applies **dynamic tiling** — it picks the `(cols, rows)` grid (`cols·rows ≤ max_dynamic_patch + 1` for the thumbnail) whose aspect best matches the input, **stretches** the image to `448·cols × 448·rows` (a pure `image.resize`, no padding), and splits it into 448² tiles (+ a 448² thumbnail when >1 tile). The probe returns `(448·rows, 448·cols)` computed by `transformers.models.got_ocr2.image_processing_got_ocr2.get_optimal_tiled_canvas` (InternVL reuses GOT-OCR2's tiler; grid selection is identical to vLLM 0.10.0's), with `image_size`/`min_dynamic_patch`/`max_dynamic_patch`/`use_thumbnail` read from the checkpoint config. Because the stretch fills the canvas, the canvas *is* the content; the per-axis ratio is genuinely anisotropic. The fixed 448² applies only when the grid is 1×1 — squares ≤633 px and small near-square images; larger inputs tile (e.g. 768²→896², 1935×2400 (H×W)→`(1344, 1792)`). `downsample_ratio` is post-ViT token compression and does not change the spatial canvas.
- **CLIP-336 trio** (`llava_med`, `huatuogpt_vision`, `healthgpt_l14`) — shared helper `_padsquare_clip_content_hw`: each model **pads the slice to a square** (`image_aspect_ratio: "pad"` → `expand2square`, or an explicit `expand2square()`) **before** the CLIP-ViT-L/14-336 resize + center-crop. The encoder perceives a fixed **336×336** canvas; the content occupies a sub-region (uniform scale `336/max(H,W)`) with the rest padding. The branch returns `perceived = [336, 336]` (→ image size) and `content = (round(H·336/max(H,W)), round(W·336/max(H,W)))` (closed form → pixel ratio; no library helper exposes the pre-pad content). Squares are unchanged (perceived == content == 336²); non-square inputs get the correct uniform per-axis pixel size while still stating the 336² canvas the model sees. (`healthgpt_xl32` shares the CLIP-336 tower but has no TL/AD branch — see Strategy D.)

## Strategy C — API rules (Claude, Gemini, OpenAI)

For API models there is no local processor to probe; the provider resizes server-side. The providers need **different** client-side treatments, because their pipelines differ in the one property that matters — whether the perceived canvas can diverge from the sent image:

- **Claude pads the canvas** (bottom/right to a multiple of 28 px) → the sent image must be made a *fixed point* of the pipeline (pre-resize to the 28-grid), or every relative coordinate is skewed.
- **Gemini crops/resamples but never enlarges the canvas** (≤3072 px) → the sent image already *is* the perceived canvas; pass-through is the faithful rule and a Claude-style grid trick would be pointless.
- **OpenAI's patch-based models behave like Claude**: a non-32-aligned image gets an extra row/column of edge patches that overhang the boundary ("a patch may extend beyond the image boundary"), enlarging the perceived grid → fixed-point pre-resize to the 32-grid; its tile-based models document no padding, so only the no-resize downscale is needed.

### Claude

`claude` makes the sent image a **fixed point** of the provider's pipeline:

- `anthropic_resized_hw()` in [`claude.py`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/models/claude.py) computes `scale = min(1.0, long_edge_cap / max(h,w), sqrt(max_image_tokens · 750 / (h·w)))`, then floors each side to a multiple of 28 (min 28). `_encode_image()` **pre-resizes client-side** with this formula, so the server's downscale *and* its bottom/right pad-to-28 are both no-ops: the size we send, the size Claude perceives, and the size stated in the prompt are all identical (**sent == perceived == stated**).
- Per-model caps are explicit in `SUPPORTED_MODEL_CAPS` (high-res `(2576, 4784)` for Fable 5 / Opus 4.8 / 4.7; standard `(1568, 1568)` for Opus 4.6/4.5, Sonnet 4.6/4.5, Haiku 4.5); unknown model codes **raise** at three points (class init, per-image encode, per-prompt probe). OpenRouter ids (`anthropic/claude-opus-4.8`) are normalized before lookup.
- The model file is the **single source of truth**: the `claude` branch in `get_resized_img_shape` lazily imports `anthropic_resized_hw` from `lmms_eval.models.claude` (no mirrored copy), so the size stated in the prompt always matches the image actually sent. The formula and caps are guarded by [`unit-test/claude-image-resize/test_claude_resize.py`](../unit-test/claude-image-resize/test_claude_resize.py). [`unit-test/claude-image-resize/check_claude_count_tokens.py`](../unit-test/claude-image-resize/check_claude_count_tokens.py) verifies the token formula empirically against the live API.

### Gemini

Covered by [`gemini.py`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/models/gemini.py) (key `gemini`, providers `google`/`openrouter`), for both the Gemini 2.5 models (`gemini-2.5-pro/-flash/-flash-lite`) and the Gemini 3 models (`gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`).

#### In plain terms

- **What we do to the image:** we send it to Gemini **as-is, at its original size** — no resizing. The only exception is a safety net for very large pictures (longer side over 3072 pixels), which we shrink while keeping the shape. No MedVision image is that big (the largest is 1935×2400), so in practice the picture is always sent untouched.
- **Why the exact image size matters:** the prompt gives the model the image size and pixel size and asks it to derive a target's physical size in millimetres, so the stated size must match the resolution the model actually sees — otherwise the conversion is wrong. Relative coordinates (positions normalized per axis) are a separate concern: they survive any rescale, but **padding** breaks them, since blank borders add extent the content doesn't occupy. Sending the image unchanged, and confirming Gemini adds no padding, keeps both correct.
- **What was unclear, and our assumption:** Google's documentation does **not** fully specify what Gemini does to an image internally, especially Gemini 3. The concern is padding (e.g. blank borders to square the image), which would invalidate the stated size and shift the model's relative coordinates. We assume Gemini keeps the image as sent and reports coordinates against it.
- **How we confirmed it:** live tests against the real API (2026-06-12). We sent images with markers at known relative positions across square, wide, tall, and the realistic medical aspect ratio and confirmed the model returned them correctly, and we measured how Gemini bills each image to rule out hidden padding. Both held for Gemini 2.5 Pro and 3.1 Pro.

#### Details

- **Why "send as-is" is right (and different from Claude):** Claude quietly adds blank padding to the bottom/right of an image, which enlarges the picture the model sees and shifts its relative coordinates — so for Claude we have to pre-shape the image to cancel that out. Gemini does **not** add such padding for images up to 3072 pixels. Instead it samples the picture for detail (Gemini 2.5 chops it into ~768-pixel tiles plus one shrunk overview; Gemini 3 resamples it to a fixed detail budget — a target number of image tokens). Google documents that the model reports positions **relative to the picture we send** ([image understanding](https://ai.google.dev/gemini-api/docs/image-understanding)), and the Google SDK does not resize the picture before upload. So the size we send, the size the model sees, and the size we state in the prompt are all the same.
- **The "very large image" safety net:** the one image change Google documents is that *"larger images are scaled down and padded to fit a maximum resolution of 3072×3072"* ([Firebase input requirements](https://firebase.google.com/docs/ai-logic/input-file-requirements)). The word "padded" is not explained, so our shrink-first safety net keeps every image under 3072 and this server-side step never runs.
- **Detail setting is fixed to "high" (`google` provider):** leaving Gemini's detail setting (`media_resolution`) unset makes Gemini 2.5 send back only a **tiny thumbnail of the whole image** regardless of its real size — which would make small structures impossible to measure. So we pin it to `"high"`. (For Gemini 3 the detail budget is fixed and the default already equals "high", so pinning it just makes runs reproducible.) The detail setting can't be controlled through OpenRouter, so prefer the `google` provider for 2.5 measurement runs.
- **Safe-by-default model list:** `SUPPORTED_MODEL_CAPS` lists the model codes we've checked; an unknown code **stops with an error** rather than guessing (note: `gemini-3-pro-preview` was retired 2026-03-09 — use `gemini-3.1-pro-preview`). The resize rule lives in one place and the prompt-building code reuses it.
- **OpenRouter and setup notes:** OpenRouter forwards the image to Google without documented changes but gives no detail-setting control; code execution and structured-output features are `google`-only. The `gemini` extra needs `google-genai>=2.8.0`.

#### What the docs leave unclear, and how we settled each

| # | What's unclear | If we're wrong | How we settled it |
|---|---|---|---|
| 1 | What "padded" does to images over 3072 px | If it pads to a square, every reported position is off | **Avoided it**: we shrink large images first, so this step never runs |
| 2 | What Gemini 2.5 does with its detail setting unset | Every image seen as a tiny thumbnail → can't measure small structures | **Disproved by test**: the default *is* a thumbnail → we fix the setting to "high" |
| 3 | How Gemini 3 turns its detail budget into pixels | If it reshapes to a square, positions are off on non-square images | **Confirmed by test**: positions come back correct on every shape |
| 4 | How the model maps tile detail back to the whole image | Relying on undocumented internals | We rely only on the documented result; the test confirms it holds |
| 5 | How partial edge tiles are handled | Affects detail only, not the overall positions | Covered by the position test passing |

#### What we tested against the live API (2026-06-12)

Three checks in [`unit-test/gemini-image-resize/`](../unit-test/gemini-image-resize/), each built so a wrong assumption fails loudly:

| Check | What it confirms | Result |
|---|---|---|
| [`test_gemini_resize.py`](../unit-test/gemini-image-resize/test_gemini_resize.py) | The send-as-is rule behaves correctly (no API needed): small images sent unchanged, large shrunk keeping shape, never enlarged, unknown/retired codes rejected | **all pass** |
| [`check_gemini_count_tokens.py`](../unit-test/gemini-image-resize/check_gemini_count_tokens.py) | Gemini isn't secretly reshaping to a square (cost would be constant) | Costs **vary with the real image size** → no hidden reshaping |
| [`check_gemini_coordinate_frame.py`](../unit-test/gemini-image-resize/check_gemini_coordinate_frame.py) | The model reports positions against the picture we send | **2.5 Pro and 3.1 Pro both correct on every shape** |

Re-run any check with `python unit-test/gemini-image-resize/<check>.py --model <code>` in an environment with `GEMINI_API_KEY` and the eval dependencies.

### OpenAI

Covered by [`openai.py`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/models/openai.py) (key `openai`, providers `openai`/`openrouter`), for two rule families OpenAI documents ([images-vision guide](https://developers.openai.com/api/docs/guides/images-vision), verified 2026-06-12): **patch-based** models (`gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-mini/nano`, `gpt-5-mini/nano`, `o4-mini`) and **tile-based** models (`gpt-4o`, `gpt-4.1`, `o3`). The default benchmark model is **`gpt-5.5-pro`** (patch-based; shares the `gpt-5.5` vision family/caps). `gpt-5` **base** is deliberately **not** in the cap table — two doc-validation passes disagreed on whether it is tile- or patch-based, so rather than risk a wrong scale it raises.

#### In plain terms

- **What we do to the image:** we pre-shape it so the model sees exactly the picture we send, then state that same size in the prompt — but the two families need different shaping:
  - *Patch models (default `gpt-5.5-pro`, and `gpt-5.5`):* the model chops the picture into a grid of 32-pixel squares. We shrink the image just enough to fit the model's "patch budget" and then trim each side down to a whole number of 32-pixel squares, so the model neither shrinks it further nor adds filler squares.
  - *Tile models (`gpt-4o`, `gpt-4.1`, `o3`):* the model wants the longer side ≤ 2048 px and the shorter side ≤ 768 px. We shrink to meet both; anything already smaller is sent untouched. We never enlarge.
- **Why the exact image size matters:** the stated size must match the resolution the model actually sees — otherwise the pixel→mm conversion is wrong. Relative positions survive a clean rescale, but **padding/enlarging** breaks them.
- **What was unclear, and our assumption:** OpenAI publishes the *token math* but does **not** spell out whether it pads images to grid/tile boundaries or ever enlarges a small one. We assume each model perceives exactly the grid/size we pre-shape to, and confirmed it.
- **How we confirmed it:** live tests against the real API (2026-06-12, via OpenRouter). A patch model billed exactly the patch grid we predicted (constant across image shapes), and a tile model billed exactly **1 tile** for a 512×512 image — proving it was neither enlarged to 768² nor padded.

#### Details

- **Why two families (and why patch behaves like Claude):** patch models tokenize on a 32-px grid and the docs note *"a patch may extend beyond the image boundary"* — a non-32-aligned image gets an extra overhanging row/column. So for patch models we make the image a **fixed point**: `scale = min(1, √(budget·32²/area), maxdim/long_edge)`, then floor each side to a multiple of 32 (min 32). That cancels both the server downscale and the overhang (**sent == perceived == stated**). Tile models document **no** padding to 512-px tile boundaries, so they need only the no-enlarge downscale.
- **Budgets (verbatim from the docs):** patch — `gpt-5.5`/`gpt-5.4` at `detail:"high"` = **2500 patches / 2048 px** max dimension; mini/nano/`o4-mini` = **1536 patches**. Tile — fit 2048×2048, then shortest side 768 px, then 512-px tiles.
- **We floor each axis independently to 32** (stricter than the docs' single uniform shrink): the un-floored patch count equals the budget, and flooring down only lowers it, so our output is **always ≤ budget and on the 32-grid** with no overhang. Confirmed empirically by the live token probe.
- **Detail fixed to `"high"`** (the model class raises on anything else): the server default differs per model, so leaving it implicit would silently change the perceived resolution. (`detail:"low"` would force a fixed 512×512 thumbnail.)
- **Exact-match model codes:** normalization strips the OpenRouter `openai/` prefix and a trailing `-YYYY-MM-DD` snapshot suffix; matching is **exact**, so an unverified sibling can't silently inherit caps. Unknown codes **raise**. The resize rule lives only in `openai.py`; the `openai` branch lazily imports `openai_resized_hw` (no mirrored copy).
- **Request path and OpenRouter:** one Chat Completions format for both providers (`max_completion_tokens` official vs `max_tokens` via OpenRouter); OpenRouter forwards the image to OpenAI without documented changes (re-confirmed by the token-count probe matching the official tables).

#### What the docs leave unclear, and how we settled each

| # | What's unclear | How we settled it |
|---|---|---|
| 1 | Whether an in-budget patch image is sent unchanged | **Confirmed by test**: gpt-5.5 bills exactly the predicted patch grid |
| 2 | What *"a patch may extend beyond the image boundary"* implies | **Neutralized by design**: we floor each side to 32, so no patch overhangs |
| 3 | Whether the tile 768-px step ever **upscales** a small image | **Disproved by test**: raw 512² billed 1 tile (unchanged), not 4 |
| 4 | Whether partial edge tiles are pixel-padded | Affects detail only; covered by the no-enlarge result |
| 5 | Whether the per-model multiplier (≈1.2 on gpt-5.5) changes resolution or only cost | **Confirmed cost-only**: the measured/predicted ratio is constant across shapes |
| 6 | Whether OpenRouter transforms the image before forwarding | **Confirmed by test**: token counts via OpenRouter match the official tables |
| 7 | Whether `gpt-5` **base** is tile- or patch-based | **Removed**: left out of the cap table so it **raises**; add only after a live token probe settles its family |

#### What we tested against the live API (2026-06-12)

Two checks in [`unit-test/openai-image-resize/`](../unit-test/openai-image-resize/):

| Check | What it confirms | Result |
|---|---|---|
| [`test_openai_resize.py`](../unit-test/openai-image-resize/test_openai_resize.py) | The resize formula behaves correctly (no API needed): patch outputs 32-aligned, budget/max-dim binding, tile no-resize fixed point, never enlarge, id normalization, unknown codes rejected | **all 13 pass** |
| [`check_openai_count_tokens.py`](../unit-test/openai-image-resize/check_openai_count_tokens.py) | OpenAI isn't secretly padding/resizing, and OpenRouter forwards untouched | **Patch ratio constant ≈1.20** (cost-only) → geometry honest; **tile billed 1 tile** → not upscaled; OpenRouter counts match official tables |

Re-run with `python unit-test/openai-image-resize/check_openai_count_tokens.py --provider openrouter` (or `--provider openai`) in an environment with the API key and the eval dependencies.

## Strategy D — no TL/AD branch

These keys have no branch in `get_resized_img_shape()`. That is **harmless for Detection** (which never calls it) but means TL/AD either raises or was never run:

- **`healthgpt_xl32`** — model file exists (CLIP-336 tower, like L14), but no branch → TL/AD would raise. No XL32 benchmark scripts exist, so the raise is latent. If a branch is ever added it should mirror the CLIP-336 trio (pad-to-square content, `_padsquare_clip_content_hw(img, 336)`).

Two former orphan keys were removed from the registry: `healthgpt` (no `models/healthgpt.py` module existed — `eval__healthgpt.py` resolves to `healthgpt_l14`/`healthgpt_xl32`) and `llama_vision` (its module lives only at `models/dev/deprecated/llama_vision.py`; the active Llama key is `vllm_llama_3_2_vision`).

## Coverage checklist

Every active `AVAILABLE_MODELS` key → section: `claude`, `gemini`, `openai` → [C](#strategy-c--api-rules-claude-gemini-openai); `healthgpt_xl32` → [D](#strategy-d--no-tlad-branch); `vllm_gemma3`, `medgemma`, `meddr` → [A](#strategy-a--fixed-perceived-size); `vllm_qwen25vl`, `vllm_qwen25vl_tooluse`, `vllm_qwen3vl`, `lingshu`, `vllm_llama_3_2_vision`, `vllm_llava_onevision`, `vllm_gemma4`, `vllm_internvl3`, `llava_med`, `huatuogpt_vision`, `healthgpt_l14` → [B](#strategy-b--input-dependent-perceived-size). Commented-out keys (`qwen2_5_vl`, `internvl3`, `llava_onevision`, `llama4`, `biomedgpt`) → footnote under the [summary table](#summary-table).
