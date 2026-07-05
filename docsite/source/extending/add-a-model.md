# Adding a new model

MedVision runs its benchmark on a vendored fork of `lmms-eval` that ships inside the package at `src/medvision_bm/medvision_lmms_eval/`. Wiring in a new vision-language model means teaching that fork two things: how to construct and call the model, and how to report the resolution the model *actually perceives* so the measurement prompts stay physically consistent.

This page is an orientation — the exhaustive, code-level guides live in the repository:

:::{seealso}
- [`docs/New-Models-Guide.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/New-Models-Guide.md) — the exhaustive, code-level walkthrough.
- [`docs/Model-Image-Processing.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/Model-Image-Processing.md) — the per-model image-processing recipes.
:::

Five things change when you add a model. Each is covered below at a high level, followed by a checklist.

## 1. The model registry

Every runnable model is a class under `lmms_eval/models/` — one file per model — listed in the `AVAILABLE_MODELS` dictionary in `lmms_eval/models/__init__.py`. The dictionary maps the CLI key you pass to `--model` onto the class that implements it:

```python
AVAILABLE_MODELS = {
    "claude": "Claude",
    "gemini": "Gemini",
    "kimi": "Kimi",
    "vllm_qwen25vl": "VLLM_Qwen25VL",
    # ...
}
```

The class is registered with the `@register_model` decorator, whose argument must be the same key:

```python
from lmms_eval.api.registry import register_model

@register_model("vllm_qwen25vl")
class VLLM_Qwen25VL(lmms):
    ...
```

Keep the three strings aligned — the dictionary key, the decorator argument, and (by convention) the class name. A `vllm_` prefix signals the model runs through the vLLM engine rather than a raw `transformers` backend.

:::{tip}
The list is deliberately explicit. Keys that are commented out (`qwen2_5_vl`, `internvl3`, `biomedgpt`, …) exist in the tree but are not part of the published benchmark.
:::

## 2. The `generate_until()` contract

Each model subclasses the framework's `lmms` base and must implement:

```python
def generate_until(self, requests) -> List[str]:
    ...
```

It receives a batch of request objects — each carrying the assembled prompt, the visual inputs, and the generation kwargs — and returns one decoded string per request, in order. That is the entire inference surface the harness relies on. However you load the model (vLLM, `transformers`, or an HTTP client), `generate_until()`'s only job is to turn requests into text completions.

## 3. Perceived image size (the essential part)

The two quantitative tasks — Tumour/Lesion size and Angle/Distance — write the image size and pixel spacing *into the prompt* and ask the model to convert pixels to millimetres itself. Those numbers must describe the image **after the model's own internal resize**, not the raw slice. Detection is exempt: it asks for relative `[0, 1]` coordinates and never touches this logic.

The single entry point is `get_resized_img_shape()` in `lmms_eval/tasks/medvision/medvision_utils.py`:

```python
get_resized_img_shape(model_name, img_2d_raw, extra_kwargs)
# -> (perceived_canvas_hw, content_hw)
```

It dispatches on `model_name` and returns **two** shapes:

- `perceived_canvas_hw` — the (possibly padded) canvas the encoder sees, used for the *image size* stated in the prompt.
- `content_hw` — the resized content shape, used to calculate the adjusted *pixel size* (padding does not change the physical extent).

For models that stretch or smart-resize without padding the two shapes are identical; for models that pad (letterbox, pad-to-square) they differ, and returning them separately keeps both prompt numbers correct. Add a branch for your new key. Unknown keys **raise Error** — a model must never run against a scale nobody verified.

This is where the physical-units invariant from [Dataset concepts](../dataset/concepts.md) is enforced: physical extent (size × spacing) is conserved independently per axis, so a wrong perceived size silently corrupts every measurement. Your branch either returns a constant (fixed-square processors), probes the real processor via `AutoImageProcessor.from_pretrained(extra_kwargs["model_hf"], ...)`, or computes the geometry in closed form. [`docs/Model-Image-Processing.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/Model-Image-Processing.md) documents the exact recipe used for every current model.

## 4. Runtime injection of `model_name` / `model_hf`

You do **not** edit the base task YAMLs to add a model. At run time, `evaluator.py` parses the CLI and injects two values into the task's `lmms_eval_specific_kwargs`:

- `model_name` — from `--model`; must match a registry key *and* a `get_resized_img_shape()` branch.
- `model_hf` — from `--model_args model_hf=<HF_ID>`; the checkpoint id used to load the image processor for dynamic-resize probes.

The eval driver and shell launcher assemble those flags for you, so the wiring flows:

```text
eval__<model>.sh   →   eval__<model>.py   →   evaluator.py   →   get_resized_img_shape()
 (--model_name,          (--model,              (injects
  --model_hf_id)          --model_args)          model_name, model_hf)
```

The shared `tasks/medvision/lmms_eval_specific_kwargs.yaml` only needs an explicit block when a model wants parameters *beyond* `model_name`/`model_hf` — for example, HealthGPT also passes its base/vision checkpoints and HLoRA settings there.

## 5. API models own their own resize rule

Client-hosted models — `claude`, `gemini`, `openai`, `kimi` — use the same registry / `generate_until()` / driver wiring, but there is no local processor to probe: the provider resizes server-side. So each of these model files carries its own resize formula plus a `SUPPORTED_MODEL_CAPS` table (per-model token and pixel limits), and the matching branch in `get_resized_img_shape()` **lazily imports that function** instead of re-implementing it. That keeps the size stated in the prompt and the image actually sent from ever drifting apart.

The rules differ by provider. Claude, OpenAI's patch family, and Kimi pre-resize client-side to a *fixed point* of the server pipeline — flooring each side to a 28- or 32-pixel grid so the server's downscale *and* its pad step both become no-ops. Gemini is pass-through: it never enlarges the canvas below its 3072-px cap, so the sent image already equals what the model perceives. Unknown model codes raise everywhere, forcing whoever adds one to confirm its caps against the provider's vision docs first.

:::{warning}
Providers that **pad** the canvas (Claude, Kimi, OpenAI patch models) skew every relative coordinate if the sent and perceived sizes diverge. The fixed-point pre-resize is what prevents this. Do not add an API model without reading the provider's vision documentation and the caveats in [`docs/Model-Image-Processing.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/Model-Image-Processing.md).
:::

## 6. Driver script and dependencies

Two more pieces make the model runnable end to end:

- **Eval driver** — `src/medvision_bm/benchmark/eval__<model>.py`, mirroring an existing one (`eval__gemini.py` for API models, `eval__qwen3_vl.py` for vLLM). It turns the launcher's flags into the `python -m medvision_bm.benchmark.eval__<model>` invocation. Per-task shell launchers go under `script/benchmark-{detect,TL,AD}/`.
- **Dependencies** — a per-model pin file at `requirements/requirements_eval_<model>.txt`, plus the matching extras group under `[project.optional-dependencies]` in `medvision_lmms_eval/pyproject.toml`. Version pins matter: several models need a specific `transformers` (and, for vLLM models, a specific vLLM) or the framework import fails outright.

## Checklist

```text
[ ] lmms_eval/models/<model>.py            new class, @register_model("<key>"), generate_until()
[ ] lmms_eval/models/__init__.py           add "<key>": "<ClassName>" to AVAILABLE_MODELS
[ ] tasks/medvision/medvision_utils.py     branch in get_resized_img_shape() (+ _process_img_<model> if dynamic)
[ ] (API only) model file                  SUPPORTED_MODEL_CAPS + resize fn; imported by get_resized_img_shape()
[ ] src/medvision_bm/benchmark/eval__<model>.py       driver, mirrors an existing one
[ ] script/benchmark-{detect,TL,AD}/eval__<model>__*.sh    one launcher per task family
[ ] requirements/requirements_eval_<model>.txt + pyproject extras    pinned deps
```

Once wired, run the model exactly like any built-in one — see [Running evaluations](../benchmarking/running-evaluations.md). For the full, code-level procedure — including the image-processing strategy tables and the unit tests that guard each resize rule — read [`docs/New-Models-Guide.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/New-Models-Guide.md) and [`docs/Model-Image-Processing.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/Model-Image-Processing.md) in the repository.
