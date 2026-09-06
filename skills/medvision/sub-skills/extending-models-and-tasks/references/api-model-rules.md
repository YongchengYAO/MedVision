# API models: caps tables, client-side pre-resize, keys and pins

An API model is wired exactly like a local one (registry entry, `generate_until`, dispatch
branch, eval entry point, three launchers — see `add-a-model.md`). What is fundamentally
different is **image size handling**, and that is the part most likely to silently corrupt
results.

## The core constraint

MedVision's quantitative tasks put the image size and the pixel size into the prompt and ask the
model to do the pixel -> mm arithmetic itself; detection asks for **relative coordinates in
`[0, 1]`**. Both are normalised by the canvas the model's vision encoder actually perceives after
the provider's server-side processing. For a local model you probe the image processor. For an
API model **there is no local processor to probe** — so the rule has to be reproduced from the
provider's documentation and applied client-side.

## Rule 1 — the cap table lives ONLY in the model file

`lmms_eval/models/<provider>.py` owns:

- `SUPPORTED_MODEL_CAPS` — an **enumerated** table, one entry per model code;
- `_normalize_model_code()` — strips the gateway prefix (`anthropic/`, `google/`, `openai/`,
  `moonshotai/`) and any suffix the provider appends, so the direct and gateway ids resolve to
  one entry;
- `<provider>_image_caps(model_code)` — table lookup, **raises** on a miss;
- `<provider>_resized_hw(img_h, img_w, model_code)` — the resize formula.

The task layer does **not** re-implement anything. Its branch is a function-local import:

```python
elif model_name == "claude":
    from lmms_eval.models.claude import anthropic_resized_hw
    img_h, img_w = img_2d_raw.shape[:2]
    model_code = (extra_kwargs or {}).get("model_hf") or ""
    img_shape_resized_hw = anthropic_resized_hw(img_h, img_w, model_code)
```

Two consequences worth stating explicitly:

- **One source of truth.** The size stated in the prompt and the image actually sent are computed
  by the same function, so they cannot drift. A second copy of the table is the bug this design
  exists to prevent.
- **The import is function-local on purpose.** SFT and RFT also call `get_resized_img_shape`, but
  never with an API model name; a module-level import would drag the model layer and the vendor
  SDK into every data-preparation run.

The model class also calls `<provider>_image_caps(model)` **in `__init__`**, so an unsupported
code fails at construction, before a single sample is spent.

## Rule 2 — an unknown model code must raise, never fall back

There is deliberately no default entry. Models in the *same family* differ:

| provider | tiering that actually differs |
| --- | --- |
| Anthropic | high-resolution vision (2576 px long edge / 4784 image tokens) on some flagship codes; standard (1568 px / 1568 tokens) on the rest — including other members of the same generation |
| OpenAI | two rule *families*: `patch` (32x32 patches, `tokens = ceil(w/32) * ceil(h/32)`, budget 2500 patches + 2048 px on flagships, 1536 patches on mini/nano/o4-mini) vs `tile` (fit 2048x2048, then shortest side 768, then 512-px tiles). Matching is **exact**, not prefix, so an unverified sibling cannot inherit a wrong rule; one base code is deliberately absent because two doc passes disagreed on its family |
| Gemini | `2.5` series uses pan-&-scan tiling; `3.x` series uses `media_resolution` token budgets. Both share the 3072 px input-scaling cap |
| Moonshot/Kimi | MoonViT `media_proc_cfg` budget is per checkpoint (`in_patch_limit` 16384 on the current model, 4x smaller on an earlier one) |

A silent fallback would emit a wrong pixel size for an unverified model and corrupt every
measurement, invisibly. So the lookup raises a `ValueError` whose message names the doc URL to
check and the table to edit. Adding a model = reading the vendor's vision documentation and
adding one row.

## Rule 3 — pre-resize client-side so resize AND pad are no-ops

The image is resized in `_encode_image()` before base64/PNG encoding, using the same
`<provider>_resized_hw()` the prompt uses.

The Anthropic formula is the canonical example:

```python
scale = min(1.0,
            long_edge_cap / max(img_h, img_w),
            math.sqrt(max_image_tokens * 750.0 / (img_h * img_w)))
new_h, new_w = floor_to_multiple_28(img_h * scale), floor_to_multiple_28(img_w * scale)
```

with `floor_to_multiple_28(x) = max(28, (int(x) // 28) * 28)`. Rounding each side **down** to a
multiple of 28 makes the image a **fixed point** of the provider's pipeline: it is already within
both caps (so the server does not resize) and already on the grid (so the server does not pad).
`sent image == perceived canvas == stated size`. It never upscales, and per-axis sizes are fine
because the prompt's pixel size is adjusted per axis.

The wrapper asserts the invariant right after resizing:

```python
assert new_h % 28 == 0 and new_w % 28 == 0
assert max(new_h, new_w) <= long_edge_cap and (new_h * new_w) / 750.0 <= max_img_tokens + 1
```

Not every provider needs the same shape of fix:

- **Anthropic** pads bottom/right to a multiple of 28 -> floor-to-28 fixed point (grid 28).
- **Kimi / MoonViT** pads up to a multiple of `patch_size * merge_kernel_size = 28` after a
  downscale-only NaViT resize -> floor-to-28 fixed point, with the area budget
  `in_patch_limit` and a per-side cap `patch_limit_on_one_side * patch_size`.
- **OpenAI** patch models floor to the **32**-px grid; tile models are fitted to
  `2048` long / `768` short edges.
- **Gemini** is **pass-through**: it does crop-based detail sampling (768x768 tiles plus a global
  low-res view, or a fixed token-budget resample) and does not pad canvases below 3072 px. The
  documented coordinate contract is normalised to the *input* image dimensions, so
  sent == perceived. The only client-side guard is the 3072 px long-edge downscale that pre-empts
  the documented "scaled down and padded" server path. No MedVision slice (largest 1935x2400)
  reaches it.

Where a provider's internal geometry is undocumented (the Gemini 3 `media_resolution` budgets;
the hosted Kimi endpoint's equivalence to the open-weights processor), the assumption is stated
in the module docstring and guarded by a coordinate-frame probe in
`unit-test/<provider>-image-resize/` rather than assumed silently.

## Why padding is not harmless

MedVision asks for coordinates in `[0, 1]` — `coordinate / canvas dimension` — and MedVision's
coordinate convention puts the **origin at the lower-left**. If the provider pads the bottom and
right of the image:

- the canvas the model normalises by is **larger** than the image you sent and larger than the
  size stated in the prompt, so every relative coordinate is scaled by a constant factor
  (a systematic skew, not noise);
- the bottom padding sits exactly **on the origin**, so the offset is worst where the reference
  is.

The same padded canvas also invalidates the stated image size for the T/L and A/D pixel -> mm
arithmetic. Nothing errors; the metrics simply come out biased. Grid alignment removes the
failure mode instead of correcting for it.

## Providers, auth and reasoning parameters

Each API wrapper supports its native endpoint plus an OpenAI-compatible gateway, selected by a
`provider` model-arg and surfaced as `--api_provider` on the eval entry point. For Claude:

- `anthropic` (default) — direct API, key from `ANTHROPIC_API_KEY`, Anthropic Messages format,
  `thinking={"type": "adaptive"}`;
- `openrouter` — `https://openrouter.ai/api/v1`, key from `OPENROUTER_API_KEY`, model ids like
  `anthropic/claude-fable-5`, reasoning via `extra_body={"reasoning": {"enabled": True}}`.

Practical rules the repository records:

- **Sanitise API keys.** Pod/k8s-injected secrets can carry a trailing newline, which is an
  illegal HTTP header value. The wrapper calls `.strip()` on the env var; the launchers also
  `tr -d '\n'`. Both layers matter — the launcher fixes the exported value for the whole process
  tree, the wrapper protects a direct invocation.
- **Adaptive thinking only** on the newest Claude tiers: `budget_tokens` and sampling parameters
  (`temperature` / `top_p` / `top_k`) are rejected with a 400, and an explicit
  `thinking:{"type":"disabled"}` also 400s — omit the parameter to disable instead.
- **Do not retry 400s.** The `backoff` decorator uses a `giveup` predicate on
  `status_code == 400`: those are deterministic invalid requests, and retrying them burns credit.
  (Some gateways reserve the full `max_tokens` as credit up front, so a large budget on a
  low-balance account fails with a payment error rather than a model error.)
- **Batch size is 1.** API launchers set `batch_size=1` and `sample_limit=100` (the pilot-study
  size) rather than the 1000 used for open weights.
- Lazy-import the SDK inside `prepare_model()` so only the provider actually used must be
  installed.

## The `transformers` pin in API extras

The API wrapper itself does not need `transformers` — but the lmms_eval framework imports it. If
the extra leaves it unpinned, pip resolves a newer `transformers` that imports `is_offline_mode`
from `huggingface_hub`, which the pinned `huggingface_hub==0.36.0` no longer provides:

```
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

Every API extra therefore pins the validated version explicitly, e.g.

```toml
claude = [
    "anthropic",
    "openai",
    "transformers==4.57.1",   # validated with huggingface-hub==0.36.0
]
gemini = ["google-genai>=2.8.0", "openai", "transformers==4.57.1"]
openai = ["openai", "transformers==4.57.1"]
kimi   = ["openai", "transformers==4.57.1"]
```

The pinned `huggingface_hub` version is itself asserted by the entry point via
`ensure_hf_hub_installed(hf_hub_version="0.36.0")`, and the frozen
`requirements/requirements_eval_<key>.txt` repeats both pins.

## Adding a new API model — condensed

1. Read the provider's official **vision / image-understanding** documentation and extract: the
   token formula, the per-model token budget, the per-model pixel cap, and whether the server
   **pads**.
2. Add one `SUPPORTED_MODEL_CAPS` row per model code, with a comment naming the doc page. Do not
   add a default.
3. Implement `<provider>_image_caps()` (raises on a miss) and `<provider>_resized_hw()`
   (downscale-only, then align to the provider's grid if — and only if — the server pads).
4. Call `_encode_image()` from `generate_until`, asserting the caps/grid invariant.
5. Add the lazy-import branch to `get_resized_img_shape()`.
6. Pin `transformers` in the extra; add the frozen requirements file.
7. Write `unit-test/<key>-image-resize/test_<key>_resize.py`: on-grid outputs, within caps, never
   upscales, aspect ratio preserved, unknown codes raise, gateway ids normalise to the same
   result as direct ids. Add a **credential-gated** live token-count probe confirming that an
   on-grid image incurs no extra image tokens from padding — that is the only end-to-end proof
   that the fixed point holds.
8. Only then spend real credit: run a handful of samples and read the logged prompt.

`scripts/scaffold_new_model.py --kind api` emits all of this as TODO-marked skeletons, including
the lazy-import dispatch branch and the offline test file.
