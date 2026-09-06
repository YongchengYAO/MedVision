# Perceived Image Size, the Pixel-Size Invariant, and Output Token Budgets

Two run-time settings decide whether a MedVision evaluation measures what it claims to measure: the **image/pixel size
stated in the prompt** and the **output token budget**. Both are model-specific and both fail silently when wrong — a
wrong pixel size makes every millimetre answer wrong while the response still looks perfectly formatted, and a short
budget truncates the response before `</answer>` so a correct measurement scores as a parse failure.

---

## Part 1 — the pixel-size invariant

### Why it exists

T/L and A/D prompts state the image size and the pixel size (physical spacing) and ask the model to do the pixel→mm
arithmetic itself. Those two numbers must describe the picture the model **actually sees after its own internal
resize**, not the raw NIfTI slice. Every VLM resizes differently (fixed square, "smart resize" to a patch multiple,
tile grids, provider-side API rules), so the correct numbers are model-specific.

The invariant, enforced **per axis**:

```
stated_size_axis × stated_pixel_size_axis  ==  original_size_axis × original_pixel_size_axis      (mm)
```

**Detection prompts need none of this.** They ask for relative `[0, 1]` coordinates and never call the perceived-size
lookup. A wrong image-processing branch therefore breaks T/L and A/D silently while Detection still looks fine.

### Two shapes, two uses

`get_resized_img_shape(model_name, img_2d_raw, extra_kwargs)` (in the vendored engine's
`lmms_eval/tasks/medvision/medvision_utils.py`) returns `(perceived_canvas_hw, content_hw)`:

- **perceived canvas** → the **image size** stated in the prompt: the resized *and padded* canvas the encoder sees.
- **content** → the **pixel-size** adjustment: the resize-only shape, because padding adds no physical extent.

For processors that stretch or resize without padding the two are identical; for processors that letterbox or
pad-to-square (LLaVA-OneVision, Llama-3.2-Vision, and the CLIP-336 trio LLaVA-Med / HuatuoGPT-Vision / HealthGPT) they
differ, and the split is what keeps both numbers right. Conserving physical extent alone is *not* a sufficient test —
it holds for any returned value.

### How the dispatch is wired (nothing to edit in a task YAML)

1. If `--reshape_image_hw` was passed, the 2D slice is reshaped at NIfTI load for **both** `doc_to_visual` and
   `doc_to_text`, so the probe and the model see the same input.
2. `evaluator.py` parses `--model_args`, and injects `model_hf`, the `--model` key (as `model_name`) and the
   normalised `reshape_image_hw` into every task's `lmms_eval_specific_kwargs`.
3. The T/L and A/D `doc_to_text` functions call `get_resized_img_shape(model_name, …)`, compute `resize_ratio_h` and
   `resize_ratio_w` independently, and divide the pixel sizes by them.
4. An unregistered key raises `ValueError: [Error] <model_name> is not recognised/supported.` — deliberately loud, so a
   new model cannot silently run at the wrong scale.

The same branches accept the bare (non-`vllm_`-prefixed) SFT aliases (`qwen25vl`, `qwen3vl`, `gemma3`, `gemma4`,
`glm4v`, `minimax_m3`, `internvl3`, `llava_onevision`, `llama_3_2_vision`), because SFT training reuses the function.

### Three strategies

| Strategy | Meaning | Keys |
|---|---|---|
| **A — fixed** | processor stretches every image to the same square; the answer is a constant | `vllm_gemma3` and `medgemma` (896×896), `meddr` (448×448) |
| **B — input-dependent** | computed or probed per image | `vllm_qwen25vl`, `vllm_qwen25vl_tooluse`, `vllm_qwen3vl`, `lingshu`, `vllm_minimax_m3`, `vllm_glm4v` (28-multiple smart resize, recovered as `image_grid_thw × patch_size`); `vllm_gemma4` (floor-48 resize under a patch budget, recovered from `image_position_ids`); `vllm_internvl3` (dynamic 448-tile canvas, stretch); `vllm_llama_3_2_vision` (aspect-fit onto ≤N 560×560 tiles, then pad); `vllm_llava_onevision` (anyres 384-tile letterbox); `llava_med`, `huatuogpt_vision`, `healthgpt` (pad-to-square then CLIP-336) |
| **C — API rule owned by the model file** | the client pre-resizes to a *fixed point* of the provider pipeline, so sent == perceived == stated | `claude` (28-grid within per-model caps), `openai` (32-grid patch models; no-enlarge downscale for tile models), `kimi` (MoonViT patch budget then floor-28), `gemini` (pass-through; only >3072 px long edge is downscaled) |

Practical consequences:

- **Non-square inputs.** The padding models return distinct perceived/content shapes; do not assume a single square
  constant for LLaVA-OneVision, Llama-3.2-Vision or the CLIP-336 trio. Historical probes for LLaVA-OneVision and
  InternVL3 were wrong for non-square T/L and A/D slices; the current code splits the two shapes explicitly. If you
  add or change a branch, validate it on a non-square slice, not only on a square one.
- **API cap tables are the single source of truth.** `claude`, `openai` and `kimi` keep their resize formula and their
  per-model caps in the model file, and `get_resized_img_shape` lazily imports that same function — so the prompt-side
  size can never drift from the image actually sent. Unknown model codes raise.
- **Gemini's detail setting.** With `--media_resolution` unset the SDK default makes Gemini 2.5 return a ~258-token
  thumbnail of the whole slice, which destroys small-structure measurement; the wrapper pins `high`. The setting is
  not controllable through OpenRouter, so prefer the direct `google` provider for measurement runs.
- **`--reshape_image_hw`** happens *before* all of this. The API launchers and MedVision-V0 use `512x512`; open-weight
  launchers otherwise leave it unset. Changing it changes the prompt text, so it also invalidates the response cache.

---

## Part 2 — output token budgets

### Three resolution channels

The wrappers resolve the budget in this order of precedence:

1. **Task-YAML `generation_kwargs.max_new_tokens`** — would win, but **no MedVision task YAML declares
   `generation_kwargs` at all**; the harness fills in `{"do_sample": False}` (`api/task.py:149-155`) and
   deliberately leaves `until` unset.
2. **`model_args` `max_new_tokens` / `max_tokens`** — injected by the launcher from `--max_new_tokens` (local, default
   4096) or `--max_tokens` (API, default 16000). **This is the channel that decides in practice.**
3. **Wrapper default** — the same values, reached only when a wrapper is invoked without the driver.

A fourth, unwanted layer — a third-party library's internal default — is reachable only when a wrapper fails to set a
budget at all. That was the HuatuoGPT-Vision bug: upstream `HuatuoChatbot` hardcodes `max_new_tokens=512`, so every
HuatuoGPT-Vision run before the 2026-08-08 fix (commit `09206a2`) generated under a silent 512-token cap. Post-fix runs
use 4096 and are **not comparable** with pre-fix outputs; response caches keyed only on the prompt must be cleared
before re-evaluating.

### Effective budgets

Everything inherits 4096 (local) / 16000 (API) unless the launcher overrides it. The documented overrides:

| Model | Budget | Reason |
|---|---|---|
| MiniMax-M3, MiniMax-M3-INT4 | 16384 (all three tasks) | its `<mm:think>` chain alone exceeds 4096, so `</answer>` is never emitted and the stop string never fires; 16384 fits under `max_model_len=32768` |
| MedGemma-27B | 16000 on T/L and A/D | measured budget exhaustion: ~29.5 % of A/D-CoT samples ended at exactly 4096 with no conclusion |
| GLM-4.6V, GLM-4.6V-Flash | 16000 on A/D | verbose reasoning |
| Llama-3.2-Vision | 16000 on Detection | degenerate repetition loops rode to the cap |
| GPT-5.5, GPT-5.5-Pro | **4096** (below the API default) | explicit in the launcher; note the budget is shared with hidden reasoning tokens |
| `vllm_qwen25vl_tooluse` | fixed 512 + 64 | deliberate per-phase caps (`<think>`+`<tool_call>`, then `<answer>`), not configurable |

`llava_med` is the one wrapper that ignores a per-request budget entirely: it generates with its constructor value only
(and hardcodes `min_new_tokens=16`). The `model_args` channel still works.

### Diagnosing a budget problem

Symptoms: a low SuccessRate with responses that read as correct but stop mid-sentence, or that never contain
`</answer>`. Confirm before changing anything:

1. Open a few failing records in `Results/<task_tag>/<model_name>/<ts>_samples_<task>.jsonl` and check whether the
   response ends abruptly with no closing tag.
2. Compare their token length against the budget the run used (the launcher value, not the module default).
3. If most failures sit exactly at the cap, raise the budget — and remember that raising `max_new_tokens` beyond a
   model's `max_model_len` minus prompt-and-image tokens will fail at engine level instead.
4. **Delete that task's `response_cache` shards before re-running.** The cache key hashes the prompt, not the budget,
   so the truncated responses would be replayed.
5. If the failures are *format* problems rather than truncation (answers in `\boxed{}`, prose instead of tags), the
   fix is the LLM-judge re-parse, not a bigger budget: `../../llm-judge-parsing/SKILL.md`.

### Stop strings

The vendored engine injects **no** stop sequence. Upstream `lmms_eval` defaulted `until` to the few-shot delimiter
`"\n\n"`, which truncated CoT output at the first blank line; MedVision removed that (`api/task.py:142-149`), so with
`until` unset generation terminates on the model's **EOS token**, and the wrappers never forward a task config's
`until` as a decoding stop (`models/vllm_qwen25vl.py:287-293`). String-level stops apply only when you pass
`--stop_strings`, and the matched string is kept in the output (`include_stop_str_in_output=True`).
`--stop_strings '</answer>'` therefore supplies an explicit terminator for a reasoning model that would otherwise
run to `--max_new_tokens`. The launchers do this for Qwen3-VL, Gemma-4, GLM-4.6V(-Flash), MiniMax-M3,
HuatuoGPT-Vision and LLaVA-Med. Do not pass it to the OpenAI reasoning models, which may reject a stop parameter.
