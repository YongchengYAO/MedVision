# Model Catalog — evaluation wiring and launcher defaults

This page answers "how is this model wired into the evaluation, and what did the repository actually run it with?".
For VRAM/GPU counts, foundation pins (torch / vLLM / transformers / huggingface_hub) and released checkpoints, use the
root roster `../../../references/model-roster.md`. To add a model that is not here, see
`../../extending-models-and-tasks/SKILL.md`.

The same data in machine-readable form is `scripts/model_catalog.json`; `scripts/make_eval_launcher.py --list-models`
prints the key list and `--dry-run` resolves every value for one model x task.

## Wiring: entry point → `lmms_eval` key → dependency extra → requirements

| Generator key | Launcher stem (`script/benchmark-*/eval__<stem>__<task>.sh`) | Entry point `python -m medvision_bm.benchmark.…` | `--model` key | conda env | `--lmms_eval_opt_deps` | requirements file | install method |
|---|---|---|---|---|---|---|---|
| `qwen25vl` | `Qwen-2.5-VL` | `eval__qwen2_5_vl` | `vllm_qwen25vl` | `eval-qwen25vl` | `qwen2_5_vl` | `requirements_eval_qwen25vl.txt` | 1 |
| `qwen3vl` | `Qwen-3-VL-32B-Thinking` | `eval__qwen3_vl` | `vllm_qwen3vl` | `eval-qwen3vl` | `qwen3_vl` | `requirements_eval_qwen3vl.txt` | 2 |
| `internvl3` | `InternVL3-38B` | `eval__intern_vl3` | `vllm_internvl3` | `eval-internvl3` | – | `requirements_eval_internvl3.txt` | 1 |
| `gemma3` | `Gemma-3-27B-it` | `eval__gemma3` | `vllm_gemma3` | `eval-gemma3` | – | `requirements_eval_gemma3.txt` | 1 |
| `gemma4` | `Gemma-4-31B-it` | `eval__gemma4` | `vllm_gemma4` | `eval-gemma4` | – | `requirements_eval_gemma4.txt` | 2 |
| `llama32vision` | `Llama-3.2-Vision` | `eval__llama3_2_vision` | `vllm_llama_3_2_vision` | `eval-llama3-vision` | – | `requirements_eval_llama3_vision.txt` | 1 |
| `llavaonevision` | `LLaVA-OneVision` | `eval__llava_onevision` | `vllm_llava_onevision` | `eval-llava-onevision` | – | `requirements_eval_llava_onevision.txt` | 1 |
| `glm46v` / `glm46v-flash` | `GLM-4.6V` / `GLM-4.6V-Flash` | `eval__glm4v` | `vllm_glm4v` | `eval-glm4v` | `glm4v` | `requirements_eval_glm4v.txt` | 1 |
| `minimax-m3` / `minimax-m3-int4` | `MiniMax-M3` / `MiniMax-M3-INT4` | `eval__minimax_m3` | `vllm_minimax_m3` | `eval-minimax-m3` / `-int4` | `minimax_m3` | – / `requirements_eval_minimax-m3-int4.txt` | two-pass |
| `medvision-v0` | `MedVision-V0-7B` | `eval__medvision-model-rft` | `vllm_qwen25vl` | `eval-medvision-v0` | `medvision_v0` | `requirements_eval_medvision-v0.txt` | 1 |
| `medgemma-4b` / `medgemma-27b` | `MedGemma-4B` / `MedGemma-27B` | `eval__medgemma` | `medgemma` | `eval-medgemma` | – | `requirements_eval_medgemma.txt` | 1 |
| `lingshu` | `Lingshu` | `eval__lingshu` | `lingshu` | `eval-lingshu` | `lingshu` | `requirements_eval_lingshu.txt` | 1 |
| `meddr` | `MedDr` | `eval__meddr` | `meddr` | `eval-meddr` | `meddr` | `requirements_eval_meddr.txt` | 1 |
| `huatuogpt-vision` | `HuatuoGPT-Vision-34B` | `eval__huatuogpt_vision` | `huatuogpt_vision` | `eval-huatuogpt-vision` | `huatuogpt_vision` | `requirements_eval_huatuogpt_vision.txt` | 2 |
| `llava-med` | `LLaVA_Med` | `eval__llava_med` | `llava_med` | `eval-llavamed` | `llava_med` | `requirements_eval_llavamed.txt` | 2 |
| `healthgpt-l14` (`-xl32`) | `HealthGPT-L14` (XL32 has no launcher) | `eval__healthgpt` | `healthgpt` | `eval-healthgpt` | – | `requirements_eval_healthgpt.txt` | 2 |
| `claude` | `Claude-Fable5` | `eval__claude` | `claude` | `eval-claude` | `claude` | `requirements_eval_claude.txt` | 1 |
| `openai-gpt55` / `-pro` | `GPT5.5` / `GPT5.5-Pro` | `eval__openai` | `openai` | `eval-openai` | `openai` | `requirements_eval_gpt.txt` (launcher currently leaves a TODO instead of pinning) | 1 |
| `gemini` | `Gemini-3.1-Pro` | `eval__gemini` | `gemini` | `eval-gemini` | `gemini` | `requirements_eval_gemini.txt` | 1 |
| `kimi` | `Kimi-K2.6` | `eval__kimi` | `kimi` | `eval-kimi` | `kimi` | `requirements_eval_kimi.txt` | 1 |
| `qwen25vl-tooluse` | (no public launcher) | `eval__qwen25vl_tooluse` | – (calls vLLM directly) | – | `qwen2_5_vl` | `requirements_eval_qwen25vl.txt` | 2 |

**Extras.** The vendored engine's `pyproject.toml` defines 19 optional-dependency groups:
`audio`, `metrics`, `meddr`, `lingshu`, `huatuogpt_vision`, `llava_med`, `qwen2_5_vl`, `medvision_v0`, `qwen3_vl`,
`minimax_m3`, `glm4v`, `gemini`, `claude`, `openai`, `kimi`, `reka`, `qwen`, `mmsearch`, `all`. Each one installs a
real pin set — `meddr` brings `bitsandbytes==0.45.2`, `lingshu` `transformers==4.52.1` + `qwen-vl-utils==0.0.11`,
and `huatuogpt_vision` `transformers==4.40.0` — so passing the extra materially changes the environment.

Install method 1 = the manual trio + `--skip_env_setup`; method 2 = the entry point's built-in setup;
two-pass = `--env_setup_only`, repair the environment, `--skip_env_setup` (see `launcher-anatomy.md`).

## What the repository launchers actually pass

`sample_limit` is 1000 for every open-weight row and 100 for every API row. Budget columns give the effective output
budget per task family (`max_new_tokens`, or `max_tokens` for the API rows).

| Launcher stem | `batch_size_per_gpu` | `gpu_memory_utilization` | detect | T/L | A/D | other launcher flags |
|---|---|---|---|---|---|---|
| Qwen-2.5-VL | 10 | 0.9 | 4096 | 4096 | 4096 | – |
| Qwen-3-VL-32B-Thinking | 2 | 0.95 | 4096 | 4096 | 4096 | `--lmmseval_module vllm_qwen3vl`, `--temperature 0.8 --top_p 0.95 --top_k 20`, `--stop_strings '</answer>'` |
| InternVL3-38B | 2 | 0.9 | 4096 | 4096 | 4096 | – |
| Gemma-3-27B-it | 4 | 0.9 | 4096 | 4096 | 4096 | – |
| Gemma-4-31B-it | 10 | 0.95 | 4096 | 4096 | 4096 | `--max_model_len 8192`, `--no-enable_thinking`, `--stop_strings '</answer>'` |
| Llama-3.2-Vision | 4 | 0.9 | **16000** | 4096 | 4096 | – |
| LLaVA-OneVision | 1 | 0.9 | 4096 | 4096 | 4096 | – |
| GLM-4.6V | 1 | 0.95 | 4096 | 4096 | **16000** | `--lmmseval_module vllm_glm4v`, `--temperature 0.8 --top_p 0.6 --top_k 2 --repetition_penalty`, `--stop_strings '</answer>'` |
| GLM-4.6V-Flash | 2 | 0.95 | 4096 | 4096 | **16000** | same as GLM-4.6V |
| MiniMax-M3 / -INT4 | 1 | 0.90 | **16384** | **16384** | **16384** | `--lmmseval_module vllm_minimax_m3`, `--vllm_version`, `--cpu_offload_gb 0`, `--temperature 1.0 --top_p 0.95 --top_k 40`, `--stop_strings '</answer>'` |
| MedVision-V0-7B | 10 | 0.9 | 4096 | 4096 | 4096 | `--reshape_image_hw 512x512`, `--use_system_prompt` |
| MedGemma-4B | 10 | – | 4096 | 4096 | 4096 | – |
| MedGemma-27B | 10 (detect) / 2 (TL, AD) | – | 4096 | **16000** | **16000** | TL/AD drop to 2: data-parallel keeps a full 54.9 GB replica per GPU, so KV for 10 concurrent 16K sequences will not fit 80 GB |
| Lingshu | 2 | – | 4096 | 4096 | 4096 | – |
| MedDr | 2 | – | 4096 | 4096 | 4096 | `--dir_third_party`; launcher exports `PYTHONPATH=<third_party>/MedDr` |
| HuatuoGPT-Vision-34B | 2 | – | 4096 | 4096 | 4096 | `--dir_third_party`, `--stop_strings '</answer>'`; launcher exports `PYTHONPATH=<third_party>/HuatuoGPT-Vision` |
| LLaVA_Med | 20 | – | 4096 | 4096 | 4096 | `--dir_third_party`, `--stop_strings` |
| HealthGPT-L14 | 10 | – | 4096 | 4096 | 4096 | `--model_choice HealthGPT-L14`, `--dir_third_party`; exports `PYTHONPATH=<third_party>/HealthGPT` |
| Claude-Fable5 | `--batch_size 1` | – | 16000 | 16000 | 16000 | `--api_provider anthropic`, `--anthropic_model_code claude-fable-5`, `--reshape_image_hw 512x512` |
| GPT5.5 / GPT5.5-Pro | `--batch_size 1` | – | **4096** | **4096** | **4096** | `--api_provider openrouter`, `--openai_model_code openai/gpt-5.5[-pro]`, `--reasoning_effort low`, `--reshape_image_hw 512x512` |
| Gemini-3.1-Pro | `--batch_size 1` | – | 16000 | 16000 | 16000 | `--api_provider`, `--google_model_code`, `--reshape_image_hw 512x512` |
| Kimi-K2.6 | `--batch_size 1` | – | 16000 | 16000 | 16000 | `--api_provider`, `--kimi_model_code`, `--reshape_image_hw 512x512` |

Every T/L launcher additionally exports `MedVision_ACK_RELEASE='1.4.0'`; no Detection or A/D launcher does. Every
launcher exports `MedVision_PLANNER_VERSION='1.0.0'`.

The GPT rows are the one place where an API budget is *below* the 16000 default: `max_tokens=4096` is shared with the
model's hidden reasoning tokens, so the visible answer competes with the chain of thought. Check the truncation rate
before trusting a GPT SuccessRate.

## Backend / parallelism at a glance

- **vLLM tensor-parallel** (`vllm_*` keys): `tensor_parallel_size` = number of visible GPUs; `batch_size_per_gpu ×
  GPUs` becomes `max_num_seqs`. `gpu_memory_utilization` is meaningful here only.
- **HF data-parallel** (`medgemma`, `lingshu`, `meddr`, `huatuogpt_vision`, `llava_med`, `healthgpt`): the driver wraps
  `lmms_eval` in `accelerate launch --num_processes=<GPUs>`; each process loads a **full replica**, and `lmms_eval`
  shards samples by rank. `batch_size` is cosmetic for four of the six (MedDr, HuatuoGPT-Vision,
  LLaVA-Med, HealthGPT — one sample per iteration); **MedGemma and Lingshu genuinely batch**
  (`medgemma.py:211`, `lingshu.py:172-205`), so raising it there lifts throughput and VRAM.
- **API** (`claude`, `openai`, `gemini`, `kimi`): no local inference; one request per sample with exponential backoff
  (10 tries, immediate give-up on HTTP 400).

## API providers, model codes and key variables

| Entry point | Providers | Direct model code | OpenRouter model code | Key variables |
|---|---|---|---|---|
| `eval__claude` | `anthropic`, `openrouter` | `claude-fable-5` | `anthropic/claude-fable-5` | `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` |
| `eval__openai` | `openai`, `openrouter` | `gpt-5.5`, `gpt-5.5-pro` | `openai/gpt-5.5[-pro]` | `OPENAI_API_KEY` / `OPENROUTER_API_KEY` |
| `eval__gemini` | `google`, `openrouter` | `gemini-3.1-pro-preview` | `google/gemini-3.1-pro-preview` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` / `OPENROUTER_API_KEY` |
| `eval__kimi` | `moonshot`, `openrouter` | `kimi-k2.6` | `moonshotai/kimi-k2.6` | `MOONSHOT_API_KEY` (+ `MOONSHOT_BASE_URL`) / `OPENROUTER_API_KEY` |

Every wrapper validates its model code against an explicit cap table and **raises on an unknown code** rather than
guessing an image-resize rule. Adding a sibling model means adding it to that table (see
`../../extending-models-and-tasks/SKILL.md`), not just changing the launcher string.

## Where the values live

- Entry-point defaults: the `parse_args()` of each `eval__*.py` (`cli-reference.md` tabulates them).
- Launcher values: the repository's `script/benchmark-{detect,TL,AD}/eval__*.sh` (this page tabulates them).
- Pins: `requirements/requirements_eval_*.txt` (`../../environment-setup/references/requirements-catalog.md`).
- Registered `lmms_eval` keys: `AVAILABLE_MODELS` in the vendored engine's `lmms_eval/models/__init__.py` — 20 active
  keys (5 more are commented out), one per wrapper, each of which must also have a branch in `get_resized_img_shape()`.
