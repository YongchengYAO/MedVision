# Adding New Models to MedVision



## Codebase Architecture

```
├── src
	├── medvision_bm 
		├── medvision_lmms_eval
			├── lmms_eval
				├── models
					├── __init__.py # [1]
					├── vllm_qwen25vl.py # [2]
					├── <other-model>
```

- [1] New models registration should be added to the dictionary `AVAILABLE_MODELS`.

  Currently supported models:

  ```python
  AVAILABLE_MODELS = {
      # Claude (Anthropic API or OpenRouter)
      "claude": "Claude",
      # Gemini (Google Gemini API or OpenRouter; 2.5 + 3 series)
      "gemini": "Gemini",
      # Kimi (Moonshot Open Platform or OpenRouter; K2.6 multimodal)
      "kimi": "Kimi",
      # Gemma3
      "vllm_gemma3": "VLLM_Gemma3",
      # Gemma4
      "vllm_gemma4": "VLLM_Gemma4",
      # HealthGPT
      "healthgpt": "HealthGPT",
      # HuatuoGPT-Vision
      "huatuogpt_vision": "HuatuoGPT_Vision",
      # InternVL3
      "vllm_internvl3": "VLLM_InternVL3",
      # "internvl3": "InternVL3",
      # Lingshu
      "lingshu": "Lingshu",
      # Llama
      "vllm_llama_3_2_vision": "VLLM_Llama_3_2_Vision",
      # "llama4": "Llama4",
      # LLaVA-Med
      "llava_med": "LLaVA_Med",
      # LLaVA-OneVision
      # "llava_onevision": "Llava_OneVision",
      "vllm_llava_onevision": "VLLM_Llava_OneVision",
      # MedDr
      "meddr": "MedDr",
      # MedGemma
      "medgemma": "MedGemma",
      # MiniMax-M3 (vision-language; 428B sparse MoE, ~23B active)
      "vllm_minimax_m3": "VLLM_MiniMaxM3",
      # GLM-4.6V (GLM-4.6V MoE + GLM-4.6V-Flash dense; one vLLM wrapper for both)
      "vllm_glm4v": "VLLM_GLM4V",
      # OpenAI (official OpenAI API or OpenRouter)
      "openai": "OpenAI_GPT",
      # Qwen2.5-VL
      # "qwen2_5_vl": "Qwen2_5_VL",
      "vllm_qwen25vl": "VLLM_Qwen25VL",
      "vllm_qwen25vl_tooluse": "VLLM_Qwen25VL_ToolUse",
      # Qwen3-VL
      "vllm_qwen3vl": "VLLM_Qwen3VL",
      # BiomedGPT
      # "biomedgpt": "BiomedGPT",
  }
  ```

  - `vllm_*` means using vLLM inference engine

- [2] Model file

## Steps

1. Add a model file like `vllm_qwen25vl.py` and add model to `AVAILABLE_MODELS` in `__init__.py`


> [!TIP]
>
> 1. Match model class name and registry name in model file and those in `__init__.py`
> 2. Define `generate_until()` in model file

2. Implement a model-specific image processing function for your model in  [`src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py`](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py)

   ```python
   def get_resized_img_shape(model_name, img_2d_raw, extra_kwargs):
       # NOTE 1: usage in MedVision benchamrk 
       # The model_name is the same as the key in AVAILABLE_MODELS. If you add new models, the strings in the if conditions below should be consistent with the keys in AVAILABLE_MODELS.
       # NOTE 2: usage in SFT training 
       # When this function get_resized_img_shape() is not used in the MedVision benchmark, for example, if it is used for SFT model training,
       # the model_name could be different from AVAILABLE_MODELS. For example, we use the model name "vllm_qwen25vl" to refer to the 
       # vllm inference backend of Qwen2.5VL in the MedVision benchmark. While in SFT code (e.g., check the usage of model_family_name in medvision_bm.sft.sft_utils for more details), we can use "qwen25vl" as "model_family_name" 
       # NOTE 3: TODO/Maintainance: Supported model_name is hardcoded, could be improved 
       # For either case, the model_name should be consistent with the string used in the if conditions in this function to ensure the correct image processing method is called to get the resized image shape for pixel size adjustment. 

       assert model_name is not None, "[Error] model_name cannot be None. Please provide a valid model_name to get_resized_img_shape()."

       # Get reshaped image size so that we can adjust the pixel size dynamically
       if model_name in ["qwen3vl", "vllm_qwen3vl"]:
           img_shape_resized_hw = _process_img_qwen3vl(img_2d_raw, extra_kwargs) 
       elif model_name in ["vllm_qwen25vl", "vllm_qwen25vl_tooluse", "qwen25vl"]:
           # NOTE: Qwen2.5-VL resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
           # Preprocessor config: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/blob/main/preprocessor_config.json
           # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
           img_shape_resized_hw = _process_img_qwen25vl(img_2d_raw, extra_kwargs)
       elif model_name == "lingshu":
           # NOTE: Lingshu resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
           # Preprocessor config: https://huggingface.co/lingshu-medical-mllm/Lingshu-32B/blob/main/preprocessor_config.json
           # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
           img_shape_resized_hw = _process_img_lingshu(img_2d_raw, extra_kwargs)
       elif model_name in ["vllm_llama_3_2_vision", "llama_3_2_vision"]:
           # NOTE: Llama-3.2-Vision dynamically resize the image to a shape that can fit in patches of size [560, 560].
           # Preprocessor config: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/preprocessor_config.json
           # Image processor - MllamaImageProcessor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py#L536
           img_shape_resized_hw = _process_img_llama_3_2_vision(img_2d_raw, extra_kwargs)
       elif model_name in ["vllm_llava_onevision", "llava_onevision"]:
           # NOTE: Llava-OneVision dynamically resize the image to a shape that can fit in patches of size [384,384]
           # NOTE: The current probing method only work for single image input, as padding is enabled for multiple image inputs
           # Preprocessor config: https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf/blob/main/preprocessor_config.json
           # Image processor - LlavaOnevisionImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/llava_onevision/image_processing_llava_onevision.py#L108
           img_shape_resized_hw = _process_img_llavaonevision(img_2d_raw, extra_kwargs)
       elif model_name in ["vllm_gemma3", "gemma3"]:
           # NOTE: HealthGPT resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
           # Preprocessor config: https://huggingface.co/google/gemma-3-27b-it/blob/main/preprocessor_config.json
           # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
           img_shape_resized_hw = [896, 896]
           # img_shape_resized_hw = _process_img_gemma3(img_2d_raw, extra_kwargs)  # for debugging only
       elif model_name == "medgemma":
           # NOTE: Medgemma resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
           # Preprocessor config: https://huggingface.co/google/medgemma-4b-it/blob/main/preprocessor_config.json
           # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
           img_shape_resized_hw = [896, 896]
           # img_shape_resized_hw = _process_img_medgemma(img_2d_raw, extra_kwargs) # for debugging only
       elif model_name == "meddr":
           # NOTE: MedDr resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
           # Check the fixed size in the model config: https://huggingface.co/Sunanhe/MedDr_0401/blob/main/config.json
           img_shape_resized_hw = [448, 448]
           # img_shape_resized_hw = _process_img_meddr(img_2d_raw, extra_kwargs) # for debugging only
       elif model_name == "llava_med":
           # NOTE: Llava-Med resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
           # Check the fixed size in the model config: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/main/config.json
           img_shape_resized_hw = [336, 336]
           # img_shape_resized_hw = _process_img_llavamed(img_2d_raw, extra_kwargs) # for debugging only
       elif model_name in ["vllm_internvl3", "internvl3"]:
           # NOTE: InternVL3 resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
           # Preprocessor config: https://huggingface.co/OpenGVLab/InternVL3-38B/blob/main/preprocessor_config.json
           # Image processor - CLIPImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/clip/image_processing_clip.py#L54
           img_shape_resized_hw = [448, 448]
           # img_shape_resized_hw = _process_img_internvl3(img_2d_raw, extra_kwargs)  # for debugging only
       elif model_name == "huatuogpt_vision":
           # NOTE: HuatuoGPT-Vision resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
           # The fixed size is configured in the "shortest_edge" in image processor: https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B-hf/blob/main/preprocessor_config.json
           # Image processor - CLIPImageProcessor:
           img_shape_resized_hw = [336, 336]
           # img_shape_resized_hw = _process_img_huatuogpt_vision(img_2d_raw, extra_kwargs)  # for debugging only
       elif model_name == "healthgpt":
           # NOTE: HealthGPT resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
           img_shape_resized_hw = [336, 336]
           # img_shape_resized_hw = _process_img_healthgpt_L14(img_2d_raw, extra_kwargs)  # for debugging only
       else:
           raise ValueError(f"[Error] {model_name} is not recognised/supported.")
       return img_shape_resized_hw
   ```

   > [!NOTE]
   > The listing above is **abridged**: it shows the branch shapes, not every branch. The dispatch in
   > [`medvision_utils.py`](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py)
   > is the source of truth and also covers Qwen3-VL, Gemma-4, GLM-4.6V, MiniMax-M3 and the API
   > models (Claude, Gemini, OpenAI, Kimi). Read it before adding a branch, and check your new key
   > against `AVAILABLE_MODELS` — a key registered without a matching branch raises
   > `[Error] <model_name> is not recognised/supported.` on the T/L and A/D tasks while Detection
   > still runs.

   > [!TIP]
   >
   > **Function**:
   >
   > This function is used to align the physical spacing information (i.e., pixel size) in the text prompt with the images perceived by the model. 
   >
   > **Motivation**:
   >
   > Each VLM has their own image processor, which has very different behavior –  while some resize images to a predefined fixed size, the other may adopt a dynamic resize (“smart resize”) strategy. Since our task is scale-sensitive, we need to assure that the pixel and image size in the prompt is correct. If you just read the image and pixel size from the original image, and put this info in the prompt, it could mislead the model since the actual input image has been resized internally.
   >
   > **Strategy**:
   >
   > - For model with fixed input size, set the input image size
   > - For dynamic processing model, use the image processor to process each image and get the new image size. The image processor is loaded from `extra_kwargs["model_hf"]`, which is supplied at run time via `--model_args model_hf=<HF_ID>` (no task YAML edit needed). See the next step.

3. No per-model edit to the base task yaml files is needed. `model_name` and `model_hf` are injected into `lmms_eval_specific_kwargs` at run time by the evaluator from the CLI arguments:

   - `model_name` comes from `--model`. It must match a key in `AVAILABLE_MODELS` **and** a branch condition in `get_resized_img_shape()`.
   - `model_hf` (the HF model ID used to load the image processor for dynamic-resize models) comes from `--model_args model_hf=<HF_ID>`.

   See the injection in [`evaluator.py`](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms_eval/lmms_eval/evaluator.py):

   ```python
   # lmms_eval/evaluator.py
   _parsed_model_args = simple_parse_args_string(cli_args.model_args) ...
   _model_arg_model_hf = _parsed_model_args.get("model_hf", None)   # from --model_args model_hf=...
   _model_name = cli_args.model ...                                 # from --model
   ...
   task.lmms_eval_specific_kwargs["model_hf"] = _model_arg_model_hf
   task.lmms_eval_specific_kwargs["model_name"] = _model_name
   ```

   The benchmark eval scripts wire these up for you. For example, an eval `.sh` passes `--model_hf_id` and `--model_name` to `eval__<model>.py`, which builds `--model_args model_hf=...` and `--model <name>`:

   ```text
   eval__<model>.sh   --model_hf_id <HF_ID>  --model_name <name>
        │
        ▼
   eval__<model>.py   --model <name>  --model_args "model_hf=<HF_ID>,..."
        │
        ▼
   evaluator.py       injects model_name / model_hf into lmms_eval_specific_kwargs
        │
        ▼
   get_resized_img_shape() / _process_img_*()
   ```

   The shared [`tasks/medvision/lmms_eval_specific_kwargs.yaml`](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/lmms_eval_specific_kwargs.yaml) (included by every base task yaml) only needs an explicit entry when a model requires **extra** parameters beyond `model_name`/`model_hf`. For example, HealthGPT-L14 needs its base/vision models and HLoRA config:

   ```yaml
   lmms_eval_specific_kwargs:
     healthgpt:
       model: "healthgpt"
       base_model_hf: "microsoft/phi-4"
       vision_model_hf: "openai/clip-vit-large-patch14-336"
       model_dtype: "FP16"
       hlora_r: 32
       hlora_alpha: 64
       hlora_dropout: 0
       hlora_nums: 4
       instruct_template: "phi4_instruct"
     default:
       pre_prompt: ""
       post_prompt: ""
     dataset:
   ```

---

## API Models (Claude, Gemini, …)

API models follow the same wiring as local models (register in `AVAILABLE_MODELS`, implement `generate_until()`, add an eval `.py` + `.sh`), but the **image-size / pixel-size handling is fundamentally different** — and it is the part most likely to silently corrupt results, so read this before adding one.

### The core constraint

MedVision's quantitative tasks (Tumor/Lesion size, Angle/Distance) put the **image size** and **pixel size** into the prompt; the model must do the pixel→mm arithmetic itself. Those numbers must match the resolution the model's vision encoder **actually perceives after its internal resize** — not the raw NIfTI slice size. If they don't, the model reasons against a different scale than the ground truth assumes and every measurement is wrong.

For **local HF models** you probe the image processor (`_process_img_*` loads `AutoImageProcessor` from `extra_kwargs["model_hf"]`). For **API models there is no local processor to probe** — the provider resizes server-side. So you must:

1. **Read the provider's official vision/image docs** and find its exact resize rule and per-model limits (image-token cap, max resolution). E.g. Anthropic: an image uses ≈ `w·h/750` tokens, downscaled aspect-preserving to fit both an image-token cap and a long-edge px cap, **then padded on the bottom/right to a multiple of 28 px**.
   - Anthropic vision: <https://platform.claude.com/docs/en/build-with-claude/vision>

   > [!CAUTION]
   > **Padding is not harmless for MedVision.** The benchmark asks the model for **relative coordinates in [0, 1]** (coordinate ÷ canvas dimension). The model normalizes by the canvas it actually perceives — including any padding the provider adds. If you state the *unpadded* size in the prompt but the model perceives a *padded* canvas, every relative coordinate is skewed (and with the lower-left origin convention, bottom padding sits right on the origin). The robust fix is to make the pre-processed image a **fixed point of the provider's vision pipeline**: round each side **down to a multiple of 28** so the provider's resize *and* pad steps are both no-ops, and the content fills the whole perceived canvas. Then `sent image == perceived canvas == stated size`. This is the same grid-alignment used for Qwen2.5-VL; verify empirically with `unit-test/claude-image-resize/check_claude_count_tokens.py` (a 28-grid image must incur **no** extra image tokens from padding).

2. **Enumerate supported model codes explicitly in the model file**, grouped by cap tier — do **not** use a generic default. Models in the *same family* can differ: Anthropic high-resolution vision (2576 px / 4784 tokens) is available on Claude Fable 5, Opus 4.8 and Opus 4.7, but **not** on Opus 4.6/4.5 or the Sonnet/Haiku tiers (1568 px / 1568 tokens). An unrecognized model code must **raise**, forcing whoever adds it to verify the docs first. See `SUPPORTED_MODEL_CAPS` and `anthropic_image_caps()` in [`lmms_eval/models/claude.py`](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms_eval/lmms_eval/models/claude.py):

   ```python
   _HIGH_RES = (2576, 4784)       # (long_edge_cap_px, max_image_tokens)
   _STANDARD_RES = (1568, 1568)
   SUPPORTED_MODEL_CAPS = {
       "claude-fable-5":   _HIGH_RES,
       "claude-opus-4-8":  _HIGH_RES,
       "claude-opus-4-7":  _HIGH_RES,
       "claude-opus-4-6":  _STANDARD_RES,
       "claude-sonnet-4-6": _STANDARD_RES,
       "claude-haiku-4-5": _STANDARD_RES,
       # ...add a model only after confirming its caps in the docs; unknown -> raise
   }
   ```

3. **Pre-resize the image client-side with the exact same formula** (including the 28-grid rounding) before sending it, so the stated size equals the model input and the server does no further downscaling **and no padding**. The model class does this in `_encode_image()`.

4. **Keep the cap table + resize rule in ONE place — the model file.** The `claude` branch of `get_resized_img_shape()` in [`medvision_utils.py`](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py) does **not** re-implement the rule; it imports and calls `anthropic_resized_hw()` from `claude.py` (lazy, function-local import so the SFT path never loads the model layer). So `SUPPORTED_MODEL_CAPS` and the resize formula live only in the model file — the prompt-side size and the API-sent image can never drift. Tested directly in [`unit-test/claude-image-resize/test_claude_resize.py`](https://github.com/YongchengYAO/MedVision/blob/master/unit-test/claude-image-resize/test_claude_resize.py) and verified against the live API with [`unit-test/claude-image-resize/check_claude_count_tokens.py`](https://github.com/YongchengYAO/MedVision/blob/master/unit-test/claude-image-resize/check_claude_count_tokens.py) (both import the function directly; run them in an env with the eval deps).

> [!TIP]
> The cap table is the **single documented source of truth** and lives in the model file under `lmms_eval/models/`. This keeps the same image-processing strategy applied consistently to every supported model code (including OpenRouter ids like `anthropic/claude-opus-4.8`, which are normalized to the bare Anthropic form for lookup).

### Providers, auth, and thinking

`claude.py` supports two providers via a `provider` model-arg:

- `anthropic` (default): direct Anthropic API, key from `ANTHROPIC_API_KEY`, Anthropic Messages format, `thinking={"type":"adaptive"}`.
- `openrouter`: OpenAI-compatible endpoint (`https://openrouter.ai/api/v1`), key from `OPENROUTER_API_KEY`, model ids like `anthropic/claude-fable-5`, reasoning via `extra_body={"reasoning":{"enabled":True}}`.

Notes that bit us in practice:

- **Sanitize API keys** — pod/k8s-injected secrets can carry a trailing newline, which is an illegal HTTP header value. `claude.py` calls `.strip()`; the eval `.sh` scripts also `tr -d '\n'`.
- **Adaptive thinking only** — on Fable 5 / Opus 4.8 / 4.7, `budget_tokens` and sampling params (`temperature`/`top_p`/`top_k`) are removed (400 if sent), and an explicit `thinking:{"type":"disabled"}` 400s on Fable 5 (omit the param to disable instead).
- **Pin `transformers`** — the API model itself doesn't need `transformers`, but the lmms_eval framework imports it. Leaving it unpinned pulls a version incompatible with the pinned `huggingface_hub==0.36.0` (`ImportError: cannot import name 'is_offline_mode'`). The `claude` extras pin `transformers==4.57.1` (the version validated with hf_hub 0.36.0).

### Files to create (for completeness)

| File | Purpose |
|---|---|
| `lmms_eval/models/<model>.py` | model class + `@register_model` + `SUPPORTED_MODEL_CAPS` + pre-resize |
| `lmms_eval/models/__init__.py` | add to `AVAILABLE_MODELS` |
| `lmms_eval/tasks/medvision/medvision_utils.py` | `_process_img_<model>()` + branch in `get_resized_img_shape()` |
| `medvision_lmms_eval/pyproject.toml` | `[project.optional-dependencies]` extras (SDK + any framework pins) |
| `src/medvision_bm/benchmark/eval__<model>.py` | eval entry script (mirrors `eval__gemini.py`) |
| `script/benchmark-{detect,TL,AD}/eval__<model>__*.sh` | one shell script per task family |
| `unit-test/<model>-image-resize/test_<model>_resize.py` | resize-formula + loud-failure tests |

## Reference

- [New models guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/guides/model_guide.md) from `EvolvingLMMs-Lab/lmms-eval `