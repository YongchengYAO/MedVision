# Model Image Processing in MedVision

How each benchmark model's *perceived* image resolution is determined, and how the image-size / pixel-size statements in the quantitative-task prompts are kept consistent with it.

> Companion to the [New models guide](New-Models-Guide.md). This page documents what the code does **today**, per model; the guide explains how to wire up a new model. All claims below were validated against the repo code (file references given); facts that rest only on code comments or external sources are marked accordingly.

## Why this matters

MedVision's quantitative tasks (Tumor/Lesion size, Angle/Distance) put the **image size** and **pixel size** into the text prompt, and the model must do the pixel→mm arithmetic itself. These numbers must describe the resolution the model's vision encoder **actually perceives after its internal resize** — not the raw NIfTI slice size. If they don't, the model reasons against a different scale than the ground truth assumes and every measurement is wrong.

Because every VLM resizes differently (fixed square, "smart resize", tile grids, server-side API rules), the perceived size is model-specific. The invariant maintained per axis is:

```
stated_image_size × stated_pixel_size == original_size × original_pixel_size   (physical extent, mm)
```

**Detection tasks do not need any of this**: `doc_to_text_BoxCoordinate*` asks for relative `[0,1]` coordinates and never calls `get_resized_img_shape()`. The per-model machinery below applies only to TL/AD (and other measurement) prompts.

## How it works

Pipeline at prompt-build time (per sample), in [`medvision_utils.py`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py):

1. **Optional input reshape** — if `reshape_image_hw` is passed via `--model_args` (parsed in `evaluator.py` and injected into `lmms_eval_specific_kwargs`), the 2D slice is reshaped at NIfTI load (`_load_nifti_2d(new_shape_hw=...)`) for **both** `doc_to_visual` and `doc_to_text`, so the probe and the model always see the same input.
2. **Perceived-size lookup** — `get_resized_img_shape(model_name, img_2d_raw, extra_kwargs)` (line ~1187) dispatches on `model_name` (the `--model` CLI key) and returns the post-resize `(H, W)`. Unknown names **raise** — deliberately loud, so a new model cannot silently run with a wrong scale.
3. **Per-axis pixel-size adjustment** — each TL/AD `doc_to_text` computes `resize_ratio_h`, `resize_ratio_w` **independently** and divides the pixel sizes by them (e.g. lines ~1339–1352). This conserves physical extent per axis and absorbs *anisotropic* resizes (several models below are only approximately aspect-preserving).
4. The prompt states `The image size is {W} pixels (width) x {H} pixels (height).` and the adjusted pixel sizes.

`model_name` / `model_hf` are injected at run time from `--model` / `--model_args model_hf=...` (no task-YAML edits needed). The same branches also accept SFT aliases (`qwen25vl`, `gemma4`, `llama_3_2_vision`, …) because SFT training reuses `get_resized_img_shape()` with `model_family_name` (see NOTE 1/2 in the function header).

## Summary table

One row per active key in [`AVAILABLE_MODELS`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/models/__init__.py). Strategy: **A** = fixed perceived size, **B** = dynamic probe of the real image processor, **C** = API resize formula (client pre-resize), **D** = no TL/AD branch.

Verdict legend (pinned-source validation, 2026-06-11): ✅ CORRECT · ⚠ CONDITIONAL (correct only for some inputs/configs) · ❌ WRONG (confirmed mismatch vs the pinned engine) · ◽ not re-validated this round (prior code-trace only).

| Model key | Strategy | Perceived-size rule | Client pre-resize? | Verdict (pinned source) |
|---|---|---|---|---|
| `vllm_qwen25vl` | B | smart-resize, 28-divisible (`image_grid_thw × patch_size`) | no | ◽ code-traced |
| `vllm_qwen25vl_tooluse` | B | same branch as `vllm_qwen25vl` | no | ◽ code-traced |
| `vllm_qwen3vl` | B | same mechanism (`image_grid_thw × patch_size`) | no | ◽ code-traced |
| `lingshu` | B | same Qwen2VL smart-resize; processor runs client-side | no | ◽ code-traced |
| `vllm_llama_3_2_vision` | B | aspect-fit onto ≤`max_image_tiles` 560×560 tile canvas (pre-pad) | no | ✅ tf 4.55.2 / vLLM 0.10.2¹ |
| `vllm_llava_onevision` | B | anyres canvas via `select_best_resolution` (384-px tiles) | no | ❌ tf 4.53.2 / vLLM 0.10.0² |
| `vllm_gemma4` | B | patch-grid extent from `image_position_ids` (floor-48 resize, ≤2520 patches) | no | ◽ vs transformers `main`³ |
| `vllm_gemma3` | A | fixed 896×896 | no | ⚠ Gemma3ImageProcessor⁵ |
| `medgemma` | A | fixed 896×896 | no | ⚠ Gemma3ImageProcessor⁵ |
| `meddr` | A | fixed 448×448 (single-tile stretch) | no | ✅ MedDr_0401 config⁶ |
| `vllm_internvl3` | A | fixed 448×448 — only when vLLM's dynamic tiling picks a 1×1 grid | no | ❌ vLLM 0.10.0⁴ |
| `llava_med` | A | fixed 336×336 (pad-to-square then CLIP-336) | no | ✅ tf 4.37.2 / config⁷ |
| `huatuogpt_vision` | A | fixed 336×336 (pad-to-square then CLIP-336) | no | ✅ tf 4.40.0 / config⁷ |
| `healthgpt_l14` | A | fixed 336×336 (expand2square then CLIP-336) | no (but pads to square) | ✅ CLIP-336 / code⁷ |
| `claude` | C | `min(1, cap/long_edge, √(tokens·750/area))` then per-side floor-28 | **yes** (`_encode_image`) | ✅ code-traced + tests |
| `gemini__2_5` | D | none — raw PIL to Google SDK, server-side resize | no | ◽ code-traced (Detection-only) |
| `gemini__2_5_woTool` | D | none — identical image path to `gemini__2_5` | no | ◽ code-traced (Detection-only) |
| `healthgpt_xl32` | D | no branch → raises on TL/AD; Detection-safe | no (expand2square) | ◽ code-traced (latent) |

¹ probe returns the post-resize, **pre-pad** content size, which is exactly what the encoder perceives (pad is neutral filler) — RESOLVED-CORRECT, see [open question 3](#caveats--open-questions).
² confirmed WRONG for non-square images: `(W,H)`/`(H,W)` transpose **and** letterbox-vs-stretch mismatch — see [open questions 1–2](#caveats--open-questions).
³ gemma4 was validated against transformers `main` (2026-06-11); the model pins `transformers>=5.5.0` — not re-validated at that exact tag this round (out of scope).
⁴ vLLM 0.10.0 (the pinned engine) applies InternVL dynamic tiling (≤12 tiles + thumbnail) per the model config — WRONG for any multi-tile (non-square) input, see [open question 4](#caveats--open-questions).
⁵ `do_pan_and_scan` is off by default in both checkpoints, so Gemma3ImageProcessor stretches to 896×896 (correct); CONDITIONAL because enabling pan&scan would emit multiple 896 crops — see [open question 7](#caveats--open-questions).
⁶ `Sunanhe/MedDr_0401` config has **no** dynamic-tiling fields (unlike InternVL3) and `pad2square: false`, so it is a single-tile `Resize((448,448))` stretch — correct.
⁷ all three pad the image to a square **before** the CLIP-336 resize+center-crop, so the crop discards nothing and 336×336 is correct (refutes the feared crop-corrupts-pixel-math case) — see [open question 7](#caveats--open-questions).

Commented-out registry keys (not in the benchmark): `qwen2_5_vl`, `internvl3` (HF backends, replaced by vLLM variants), `llava_onevision` (HF backend; its alias is still accepted by the branch), `llama4`, and `biomedgpt` — the latter is the only model that force-resizes client-side to a fixed 480×480 (BICUBIC) in its own file (`biomedgpt.py:64-65`) and has no TL/AD branch.

## Strategy A — fixed perceived size

The model's processor resizes every image to one fixed square, so `get_resized_img_shape` returns a hardcoded constant. Each branch keeps a dynamic probe (`_process_img_<model>`) commented out "for debugging only", which was used to confirm the constant.

| Models | Size | Basis (from code comments, not fetched) |
|---|---|---|
| `vllm_gemma3`, `medgemma` | [896, 896] | Gemma3ImageProcessor stretch-to-896 (pan&scan off by default) — ⚠ CONDITIONAL |
| `meddr` | [448, 448] | `Sunanhe/MedDr_0401`: no dynamic-tiling fields, `pad2square: false` → single-tile `Resize((448,448))` stretch — ✅ |
| `vllm_internvl3` | [448, 448] | InternVL3 **tile** size, but vLLM 0.10.0 dynamic-tiles into a `448·cols × 448·rows` canvas — ❌ wrong for non-square inputs, see [open question 4](#caveats--open-questions) |
| `llava_med`, `huatuogpt_vision`, `healthgpt_l14` | [336, 336] | CLIP-ViT-L/14-336 + pad-to-square (`image_aspect_ratio: "pad"` / `expand2square`) — ✅ |

Notes:
- **The three CLIP-336 models pad to square before the processor.** `llava_med` and `huatuogpt_vision` set `image_aspect_ratio: "pad"` (their `process_images` applies `expand2square` first); `healthgpt_l14` calls `expand2square()` explicitly. Because the input is already square, CLIP's resize-to-336 + center-crop-336 crops nothing, so 336×336 is correct and the pixel-size math is not corrupted by cropping. (The physical content occupies a sub-region of the padded square — the standard scaledPS approximation used benchmark-wide, not a per-model bug.)
- **MedDr is *not* the InternVL3 case.** Despite sharing InternVL-chat lineage, `MedDr_0401`'s config omits `dynamic_image_size`/`max_dynamic_patch`/`use_thumbnail`, so tiling is off; it is a plain single-tile 448 stretch.
- None of these wrappers resize client-side; the raw (optionally `reshape_image_hw`-reshaped) image goes to the engine/processor, which applies the fixed resize internally.

## Strategy B — dynamic processor probe

The processor's output size depends on the input, so `doc_to_text` probes the **real** image processor (`AutoImageProcessor.from_pretrained(extra_kwargs["model_hf"])`) on each slice and reads back the resized shape. Probe ≡ model input holds because the wrappers send the raw image (base64 PNG) and the engine applies the same HF processor internally — and for `lingshu`, the processor literally runs client-side in the wrapper, the strongest case.

- **Qwen2.5-VL family** (`vllm_qwen25vl`, `vllm_qwen25vl_tooluse`, `lingshu`; SFT alias `qwen25vl`) — `_process_img_qwen25vl` / `_process_img_lingshu`: smart resize to dimensions divisible by `patch_size(14) × merge_size(2) = 28`; recovered as `image_grid_thw[1:3] × patch_size`.
- **Qwen3-VL** (`vllm_qwen3vl`; alias `qwen3vl`) — `_process_img_qwen3vl`: same `image_grid_thw × patch_size` mechanism, config-driven (no hardcoded grid constant; the branch carries no preprocessor-config citation, unlike its siblings).
- **Llama-3.2-Vision** (`vllm_llama_3_2_vision`; alias `llama_3_2_vision`) — `_process_img_llama_3_2_vision`: a `MllamaImageProcessor` subclass replays preprocessing up to (and only) the resize step, returning the aspect-fitted size on the optimal canvas of up to `max_image_tiles` 560×560 tiles. **✅ Validated CORRECT** against transformers 4.55.2 (vLLM 0.10.2's floor) + vLLM 0.10.2: the post-resize size *is* the content the encoder perceives; the later pad to `num_tiles·560` is neutral filler and aspect ratio is carried separately — see [open question 3](#caveats--open-questions).
- **LLaVA-OneVision** (`vllm_llava_onevision`; alias `llava_onevision`) — `_process_img_llavaonevision`: returns `select_best_resolution(image size, image_grid_pinpoints)`, the anyres canvas built from 384-px tiles. **❌ Validated WRONG** for non-square inputs against transformers 4.53.2 (vLLM 0.10.0's floor) + vLLM 0.10.0 on two counts: (1) it passes PIL `(W,H)` where `select_best_resolution` expects `(H,W)`, transposing the axes; (2) the real processor *letterboxes* (aspect-preserving resize then pad), not the pure stretch the probe assumes — see [open questions 1–2](#caveats--open-questions).
- **Gemma 4** (`vllm_gemma4`; alias `gemma4`) — `_process_img_gemma4`: Gemma4's processor outputs a flattened, sequence-padded patch list (`pixel_values: [batch, max_patches, patch²·3]`, with defaults `max_patches = 280·3² = 2520`, patch dim `16²·3 = 768` — **constants, never the image size**). The probe recovers `(H, W)` from the extent of the valid (non `-1`) entries of `image_position_ids` × `patch_size`. The resize is only approximately aspect-preserving: one scale factor fits the patch budget, then each side is independently floored to a multiple of 48 (`pooling_kernel_size × patch_size`); upscaling allowed; no spatial padding. Verified against the HF `image_processing_gemma4.py` source (transformers `main`, 2026-06-11; the model pins `transformers>=5.5.0`, not re-checked at that exact tag). The per-axis pixel-size adjustment absorbs the ≤1-step anisotropy.

## Strategy C — API formula (Claude)

For API models there is no local processor to probe; the provider resizes server-side. `claude` solves this by making the sent image a **fixed point** of the provider's pipeline:

- `anthropic_resized_hw()` in [`claude.py`](../src/medvision_bm/medvision_lmms_eval/lmms_eval/models/claude.py) computes `scale = min(1.0, long_edge_cap / max(h,w), sqrt(max_image_tokens · 750 / (h·w)))`, then floors each side to a multiple of 28 (min 28). `_encode_image()` **pre-resizes client-side** with this formula, so the server's downscale *and* its bottom/right pad-to-28 are both no-ops: sent == perceived == stated.
- Per-model caps are explicit in `SUPPORTED_MODEL_CAPS` (high-res `(2576, 4784)` for Fable 5 / Opus 4.8 / 4.7; standard `(1568, 1568)` for Opus 4.6/4.5, Sonnet 4.6/4.5, Haiku 4.5); unknown model codes **raise** at three points (class init, per-image encode, per-prompt probe). OpenRouter ids (`anthropic/claude-opus-4.8`) are normalized before lookup.
- The model file is the **single source of truth**: the `claude` branch in `get_resized_img_shape` lazily imports `anthropic_resized_hw` from `lmms_eval.models.claude` (no mirrored copy), so the size stated in the prompt always matches the image actually sent. The formula and caps are guarded by [`unit-test/claude-image-resize/test_claude_resize.py`](../unit-test/claude-image-resize/test_claude_resize.py) (raise-on-unknown, 28-grid, never-upscale, cap-binding, OpenRouter-normalization, suffix-resolution tests). [`unit-test/claude-image-resize/check_claude_count_tokens.py`](../unit-test/claude-image-resize/check_claude_count_tokens.py) verifies the token formula empirically against the live API.

## Strategy D — no TL/AD branch

These keys have no branch in `get_resized_img_shape()`. That is **harmless for Detection** (which never calls it) but means TL/AD either raises or was never run:

- **`gemini__2_5`, `gemini__2_5_woTool`** — raw PIL images go straight to the Google SDK; the code never computes or states a perceived size. Gemini was evaluated **Detection-only** (the only result dirs are `Results/MedVision-detect/gemini-2.5-*`; the only eval scripts are `script/benchmark-detect/eval__gemini2_5_*`), so the missing branch never mattered. The two model files are byte-identical except registration; the w/wo-tool difference lives in the eval drivers. If a Gemini TL/AD run were attempted today it would fail even before the branch raise: the empty `gemini__2_5:` blocks in the AD/TL base YAMLs make `lmms_eval_specific_kwargs` `None`, which `task.py` trips over at init.
- **`healthgpt_xl32`** — model file exists (CLIP-336 tower, like L14), but no branch → TL/AD would raise. No XL32 benchmark scripts exist, so the raise is latent. If a branch is ever added it should be [336, 336] (same tower as L14).

Two former orphan keys were removed from the registry: `healthgpt` (no `models/healthgpt.py` module existed — `eval__healthgpt.py` resolves to `healthgpt_l14`/`healthgpt_xl32`) and `llama_vision` (its module lives only at `models/dev/deprecated/llama_vision.py`; the active Llama key is `vllm_llama_3_2_vision`).

## Caveats & open questions

Each item was validated against the **pinned** library version for that model (transformers floats to vLLM's lower-bound pin: OneVision → tf 4.53.2 via vLLM 0.10.0; Llama → tf 4.55.2 via vLLM 0.10.2). Verdicts dated 2026-06-11. No code was changed — confirmed-wrong items carry a recommended fix for a follow-up.

1. **`CONFIRMED-WRONG` — LLaVA-OneVision probe transposes H/W for non-square images.** `_process_img_llavaonevision` passes `img_PIL.size` — PIL's `(width, height)` — as `select_best_resolution`'s `original_size`, but [transformers v4.53.2](https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/image_processing_utils.py) documents the function as `(height, width)` in **and** out, and the real `LlavaOnevisionImageProcessor.get_image_patches` (v4.53.2) calls it with `get_image_size(image)` = `(height, width)` — which is what vLLM 0.10.0 runs server-side (confirmed via its `get_hf_processor`/`_call_hf_processor`). The checkpoint's 36 `image_grid_pinpoints` (`[384·i, 384·j]`, i,j ∈ 1..6) are exactly swap-symmetric, so the probe returns precisely the *transposed* canvas: the two axes are mislabeled relative to the per-axis pixel sizes. (Earlier note used v4.56.1 — re-validated at the pinned floor 4.53.2; same result.) **Fix:** pass `img_PIL.size[::-1]`.
2. **`CONFIRMED-WRONG` — LLaVA-OneVision treats the canvas as a stretch, but the processor letterboxes.** The real processor (v4.53.2, used by vLLM 0.10.0 server-side) does aspect-preserving resize via `get_patch_output_size` (the *min* scale factor) **then pads** into the chosen canvas (`_resize_for_patching` → `_pad_for_patching`). The probe returns the full canvas dims and the caller divides each axis's pixel size by its own canvas/orig ratio — overstating the scale on the padded axis. **Fix:** return the aspect-preserving content size (`get_patch_output_size(orig, best_resolution)`), or apply one uniform `min`-scale ratio to both axes; don't treat padding as real-world content. (Combine with the fix in #1.) Impact: all recorded OneVision TL/AD runs sent raw slices, so non-square datasets (e.g. Ceph-Biometrics-400 sagittal 1935×2400) are affected on both counts; square slices (512² CT/MRI, 256² FeTA) are not.
3. **`RESOLVED-CORRECT` — Llama-3.2-Vision pre-pad shape is the right perceived size.** Against [transformers v4.55.2](https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/mllama/image_processing_mllama.py) (vLLM 0.10.2's floor): `resize` fits the image aspect-preserving into the tiled canvas and returns `new_height × new_width` (the real content extent); `pad` only zero-fills up to `num_tiles·560`, and aspect ratio is conveyed separately via `aspect_ratio_ids`. So the probe's pre-pad shape is exactly what the encoder perceives — returning the padded canvas would wrongly attribute physical extent to filler. vLLM 0.10.2 confirmed to use `MllamaProcessor` server-side.
4. **`CONFIRMED-WRONG` (conditional) — InternVL3's fixed 448² is wrong for any multi-tile (non-square) input.** Against [vLLM 0.10.0 `models/internvl.py`](https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/model_executor/models/internvl.py) + the [InternVL3-38B config](https://huggingface.co/OpenGVLab/InternVL3-38B/blob/main/config.json) (`dynamic_image_size: true`, `min/max_dynamic_patch: 1/12`, `use_thumbnail: true`, tile 448 → resolved bounds `min_num=1, max_num=13`): the engine picks the `(cols, rows)` grid (`cols·rows ≤ 13`) whose aspect ratio best matches the input, stretches to `448·cols × 448·rows`, splits into 448² tiles (+ a 448² thumbnail when >1 tile). **Two corrections to the earlier draft of this note:** (a) there is **no "≤633²" size threshold** — a *perfectly square* input gives `ratio_diff = 0` for the (1,1) grid, which always wins the strict `<` comparison regardless of size, so every square slice → 1×1 → 448² (correct); the area tie-break only fires between *different* ratios that tie on aspect, never demoting a perfect (1,1) match. (b) raw Ceph 1935×2400 (aspect 0.806) picks **cols=3 × rows=4** (not "4×3") → canvas 1344×1792 (+thumbnail); `(4,5)=0.8` is excluded because 4·5=20 > 13. The fixed 448² is correct **only** for inputs that resolve to a 1×1 grid (all exactly-square inputs; some near-square ones). **Fix:** replace the constant with a probe that replicates `resolve_internvl_min_max_num` + `get_internvl_target_ratios` + `find_closest_aspect_ratio` and returns `(448·rows, 448·cols)`; or pass `mm_processor_kwargs={"max_dynamic_patch": 1}` to force 448² (changes effective resolution vs. prior runs).
5. **`CONFIRMED` — probe≡engine is not achievable for InternVL3 with the current fixed constant or a single-tile probe.** vLLM 0.10.0 runs InternVL's dynamic tiling server-side via its own `BaseInternVLProcessor`/`dynamic_preprocess_internvl`; neither the fixed `[448,448]` nor the commented-out `_process_img_internvl3` (which reads `pixel_values.shape[-2:]`, always 448² *per tile*) can match it. A faithful probe must replicate the grid selection (see #4). For the other vLLM Strategy-B models the general assumption still holds — vLLM applies the same HF processor server-side (confirmed for OneVision and Mllama via `get_hf_processor`); the residual risk is only a `**kwargs`/`mm_processor_kwargs` override silently desyncing probe and engine.
6. **`RESOLVED` — Claude `model_hf` ↔ `model` consistency is guaranteed by the eval script.** `eval__claude.py` sets both `model=` and `model_hf=` from the *same* `anthropic_model_code` variable (lines 264, 266), so the prompt's caps and the request target cannot diverge on the supported path. The "nothing cross-checks at runtime" caveat applies only to hand-written `model_args`.
7. **`RESOLVED` — fixed-size constants validated against pinned processors/configs (no wrong-pixel-math case found).** `llava_med` (tf 4.37.2), `huatuogpt_vision` (tf 4.40.0) and `healthgpt_l14` pad the image to a square *before* the CLIP-336 resize+center-crop (`image_aspect_ratio: "pad"` / `expand2square`), so the crop discards nothing and 336² is correct — refuting the earlier worry that center-crop would corrupt the resize-to-fill math. `meddr` 448² is a single-tile stretch (no tiling fields in its config). `gemma3`/`medgemma` 896² are **CONDITIONAL**: `do_pan_and_scan` is `null` (off) in both checkpoints so the processor stretches to 896² (correct), but enabling pan&scan would emit multiple 896 crops + a global view, making the single constant per-crop-only; MedVision passes no pan&scan override, so the default path holds. (Doc comments: `healthgpt_l14`'s branch still cites no source link, and `huatuogpt_vision`'s cites the `-hf` checkpoint while the wrapper defaults to the non-`-hf` one — both confirmed to resolve to the same CLIP-336/pad behavior.)
8. **`CONFIRMED` — minor comment/docstring bugs (cosmetic, no runtime effect):** the `vllm_gemma3` branch comment (`medvision_utils.py:1223`) says "HealthGPT" where it means Gemma3; `vllm_gemma3.py:46` docstring claims default `"Qwen/Qwen2.5-VL-3B-Instruct"` (actual default `"google/gemma-3-27b-it"`); `vllm_qwen25vl.py:46` docstring claims that same default but `model_hf` is actually a required positional with no default; the `vllm_qwen3vl` branch carries no explanatory comment or config citation, unlike its siblings.

## Coverage checklist

Every active `AVAILABLE_MODELS` key → section: `claude` → [C](#strategy-c--api-formula-claude); `gemini__2_5`, `gemini__2_5_woTool`, `healthgpt_xl32` → [D](#strategy-d--no-tlad-branch); `vllm_gemma3`, `medgemma`, `meddr`, `vllm_internvl3`, `llava_med`, `huatuogpt_vision`, `healthgpt_l14` → [A](#strategy-a--fixed-perceived-size); `vllm_gemma4`, `vllm_qwen25vl`, `vllm_qwen25vl_tooluse`, `vllm_qwen3vl`, `lingshu`, `vllm_llama_3_2_vision`, `vllm_llava_onevision` → [B](#strategy-b--dynamic-processor-probe). Commented-out keys (`qwen2_5_vl`, `internvl3`, `llava_onevision`, `llama4`, `biomedgpt`) → footnote under the [summary table](#summary-table).
