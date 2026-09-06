# Perceived Image Size: the `get_resized_img_shape` dispatch

## Why this function exists

Tumor/Lesion-size (T/L) and Angle/Distance (A/D) prompts state the image size and the pixel
size, and the model must do the pixel -> mm arithmetic itself. Those numbers have to describe
the canvas the vision encoder **actually perceives after the model's own internal resize** —
not the raw NIfTI slice. Detection prompts ask for relative coordinates in `[0, 1]`, which the
model normalises by the canvas it perceives, so the same requirement applies there. A wrong
perceived size does not crash anything; it silently scales every measurement.

`lmms_eval/tasks/medvision/medvision_utils.py::get_resized_img_shape(model_name, img_2d_raw,
extra_kwargs)` is the single dispatch point. It is called from the T/L, A/D and MaskSize `create_doc_to_text_*`
factories (and their CoT / visual-prompt / scaledPS variants). **Detection prompt builders never
call it** — a detection answer is relative coordinates, so no perceived size enters the prompt.

## Signature and contract

```python
img_shape_resized_hw, img_shape_content_hw = get_resized_img_shape(model_name, img_2d_raw, extra_kwargs)
```

- `model_name` — the `--model` key, injected into `lmms_eval_specific_kwargs` by the evaluator.
  It is `assert`ed non-`None`. In SFT/RFT the same function is called with a *family* alias
  (`model_family_name`), which is why several branches list two or more strings.
- `img_2d_raw` — the H x W numpy slice (already reshaped if `reshape_image_hw` was passed).
- `extra_kwargs` — `lmms_eval_specific_kwargs`; the probe branches read `extra_kwargs["model_hf"]`
  to load the real image processor.

The **return is a pair**:

| element | meaning | used for |
| --- | --- | --- |
| `img_shape_resized_hw` | the **perceived canvas** (after resize *and* any padding) | the sentence `The image size is <W> pixels (width) x <H> pixels (height).` |
| `img_shape_content_hw` | the **content** shape (resize only, before padding) | the per-axis pixel-size rescale |

They are identical unless the model letterboxes. The prompt arithmetic is:

```python
resize_ratio_h = img_shape_content_hw[0] / original_height
resize_ratio_w = img_shape_content_hw[1] / original_width
adjusted_pixel_height = pixel_height / resize_ratio_h
adjusted_pixel_width  = pixel_width  / resize_ratio_w
```

Returning the padded canvas as the content shape would inflate the short axis's pixel size on
non-square inputs — the exact bug the pair exists to prevent.

An unknown `model_name` falls through to
`raise ValueError(f"[Error] {model_name} is not recognised/supported.")`.

## Strategy classes

- **A — fixed perceived size.** The processor always produces the same canvas; the branch
  hard-codes it. No processor download at prompt-build time.
- **B — input-dependent.** The branch calls a `_process_img_<model>()` probe that loads the real
  image processor from `extra_kwargs["model_hf"]` and reads back the resized shape.
- **C — API rule.** The branch does a **function-local** import of `<provider>_resized_hw()` from
  the provider's model file and calls it. The cap table and the formula live only there, so the
  prompt-side size and the image actually sent can never drift, and the SFT path never loads the
  model layer or a vendor SDK.

## Every branch (verified against the shipped source; 19 branches)

| # | `model_name` values accepted | class | canvas rule | content rule |
| --- | --- | --- | --- | --- |
| 1 | `qwen3vl`, `vllm_qwen3vl` | B | `_process_img_qwen3vl` — `image_grid_thw[1:] * patch_size` | = canvas |
| 2 | `vllm_minimax_m3`, `minimax_m3` | B | `_process_img_minimax_m3` — smart resize to a multiple of 28 under `max_pixels=451584` (~672x672); custom remote code (`trust_remote_code=True`) | = canvas |
| 3 | `vllm_glm4v`, `glm4v` | B | `_process_img_glm4v` — GLM-4V processor, multiple of 28, grid in patch units | = canvas |
| 4 | `vllm_qwen25vl`, `vllm_qwen25vl_tooluse`, `qwen25vl` | B | `_process_img_qwen25vl` — multiple of `patch_size(14) * merge_size(2) = 28` | = canvas |
| 5 | `lingshu` | B | `_process_img_lingshu` — same Qwen2-VL processor, multiple of 28 | = canvas |
| 6 | `vllm_llama_3_2_vision`, `llama_3_2_vision` | B | `_process_img_llama_3_2_vision` — Mllama tiling into 560x560 tiles; returns **(canvas, content)** | pre-pad content |
| 7 | `vllm_llava_onevision`, `llava_onevision` | B | `_process_img_llavaonevision` — anyres canvas from `select_best_resolution` (multiples of 384), then pad; returns **(canvas, content)**; single-image input only | pre-pad content from `get_patch_output_size` |
| 8 | `vllm_gemma3`, `gemma3` | A | **`[896, 896]`** — pure stretch, no pad, pan-&-scan off by default | = canvas |
| 9 | `vllm_gemma4`, `gemma4` | B | `_process_img_gemma4` — variable-resolution: one scale factor fits the image into at most `max_soft_tokens * pooling_kernel_size^2 = 280 * 9 = 2520` patches of 16x16 px, then each side is floored independently to a multiple of `pooling_kernel_size * patch_size = 48`. Aspect ratio only approximately preserved; small images are **upscaled** | = canvas |
| 10 | `medgemma` | A | **`[896, 896]`** | = canvas |
| 11 | `meddr` | A | **`[448, 448]`** | = canvas |
| 12 | `llava_med` | A + probe | canvas **`[336, 336]`**; content `_padsquare_clip_content_hw(img, 336)` | `round(H*336/max(H,W)), round(W*336/max(H,W))` |
| 13 | `vllm_internvl3`, `internvl3` | B | `_process_img_internvl3` — dynamic tiling (**not** a fixed 448x448): stretch to `image_size*cols x image_size*rows`, split into tiles + thumbnail | = canvas |
| 14 | `huatuogpt_vision` | A + probe | canvas **`[336, 336]`**; content `_padsquare_clip_content_hw(img, 336)` | as row 12 |
| 15 | `healthgpt` | A + probe | canvas **`[336, 336]`**; content `_padsquare_clip_content_hw(img, 336)` | as row 12 |
| 16 | `claude` | C | `from lmms_eval.models.claude import anthropic_resized_hw` | = canvas |
| 17 | `gemini` | C | `from lmms_eval.models.gemini import gemini_resized_hw` | = canvas |
| 18 | `openai` | C | `from lmms_eval.models.openai import openai_resized_hw` | = canvas |
| 19 | `kimi` | C | `from lmms_eval.models.kimi import kimi_resized_hw` | = canvas |

Rows 12, 14 and 15 are the "pad to square, then CLIP-336 resize + centre crop" family
(`expand2square` / `image_aspect_ratio="pad"`): the canvas is fixed, but the content occupies a
sub-region whose scale is the uniform factor `336 / max(H, W)`.

## Probe functions defined in `medvision_utils.py`

`_process_img_gemma3`, `_process_img_gemma4`, `_process_img_glm4v`, `_process_img_healthgpt_L14`,
`_process_img_huatuogpt_vision`, `_process_img_internvl3`, `_process_img_lingshu`,
`_process_img_llama_3_2_vision`, `_process_img_llavamed`, `_process_img_llavaonevision`,
`_process_img_meddr`, `_process_img_medgemma`, `_process_img_minimax_m3`,
`_process_img_qwen25vl`, `_process_img_qwen3vl`, plus the shared helper
`_padsquare_clip_content_hw(img_2d_raw, size)`.

Several probes exist but are **commented out** in their branch (`gemma3`, `medgemma`, `meddr`,
`llava_med`, `huatuogpt_vision`, `healthgpt_L14`) — they are kept as debugging cross-checks for the
hard-coded fixed sizes. (`internvl3` is **not** among them: its probe is called live, because the
model tiles dynamically and a fixed 448x448 would be wrong.) Do not delete them when editing a branch.

## Aliases that exist only here

`gemma3`, `gemma4`, `glm4v`, `internvl3`, `llama_3_2_vision`, `llava_onevision`, `minimax_m3`,
`qwen25vl`, `qwen3vl` appear in branch conditions but are **not** keys of `AVAILABLE_MODELS`.
They are the SFT/RFT `model_family_name` strings (see `medvision_bm.sft.sft_utils`). Passing one
as `--model` fails at model construction, not here. When you add a family that will also be
fine-tuned, list both the benchmark key and the family alias in the same branch.

Run `scripts/list_registered_models.py` to print this classification live for the installed
tree, and `--expect <key>` to check one key's wiring.

## Minimal probe pattern (strategy B)

```python
def _process_img_<key>(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]                 # injected from --model_args model_hf=...
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor([img_PIL])
    image_grid_thw = processed_visual["image_grid_thw"][0]
    patch_size = img_processor.patch_size
    return (int(image_grid_thw[1]) * patch_size, int(image_grid_thw[2]) * patch_size)
```

Cast tensor grid entries with `int(...)`: transformers 5.x returns 0-dim tensors, which would
render as `tensor(392)` inside the prompt and propagate into the pixel-size arithmetic.

## Validating a new branch

1. Static wiring: `python scripts/list_registered_models.py --expect <key>` (exit 0 = all three
   code sites present).
2. Behaviour: call `get_resized_img_shape` on synthetic square **and** non-square slices and
   compare against the checkpoint's `preprocessor_config.json` or a direct `AutoImageProcessor`
   call. The repository keeps these as `unit-test/<model>-image-resize/` and
   `unit-test/perceived-size-resize/`; `scripts/scaffold_new_model.py` writes an equivalent stub.
3. End to end: run two samples of one T/L task and read the printed
   `Original image size (HxW): ...; Resized image size (HxW): ...` line and the
   `The image size is ...` / `The pixel size for this image is ...` sentences in the logged prompt.
   A non-square slice is the discriminating case — a square one hides per-axis errors.
