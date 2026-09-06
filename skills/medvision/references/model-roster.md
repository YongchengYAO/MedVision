# Model Roster

## Purpose

Read this when a request names a VLM and you need to know whether MedVision
already supports it, which evaluation entry point and launcher family serve it,
which dependency stack it needs, how it is parallelised, and what hardware the
repository documents for it. Evaluation mechanics live in
`../sub-skills/benchmark-evaluation/SKILL.md`; adding a new model is covered by
`../sub-skills/extending-models-and-tasks/SKILL.md`.

## Roster (verified against `src/medvision_bm/benchmark/eval__*.py`, `lmms_eval/models/__init__.py`, `requirements/`, `dockerfile/`, `script/benchmark-*/`)

| Display name (launcher stem) | Eval entry point `python -m medvision_bm.benchmark.…` | `lmms_eval` model key | Backend | Opt-deps extra (`--lmms_eval_opt_deps`) | Requirements file | Docker tag |
| --- | --- | --- | --- | --- | --- | --- |
| MedVision-V0-7B (ours) | `eval__medvision-model-rft` | `vllm_qwen25vl` | vLLM, TP | `medvision_v0` / `qwen2_5_vl` | `requirements_eval_medvision-v0.txt` | `eval_medvision-v0` |
| Qwen-2.5-VL (7B / 32B) | `eval__qwen2_5_vl` | `vllm_qwen25vl` | vLLM, TP | `qwen2_5_vl` | `requirements_eval_qwen25vl.txt` (`_update1` = newer stack) | `eval_qwen25vl`, `eval_qwen25vl_update1` |
| Qwen2.5-VL tool-use (SFT variant) | `eval__qwen25vl_tooluse` | `vllm_qwen25vl_tooluse` | vLLM, TP | `qwen2_5_vl` | `requirements_eval_qwen25vl.txt` | – |
| Qwen-3-VL-32B-Thinking | `eval__qwen3_vl` | `vllm_qwen3vl` | vLLM, TP | `qwen3_vl` | `requirements_eval_qwen3vl.txt` | `eval_qwen3vl` |
| InternVL3-38B | `eval__intern_vl3` | `vllm_internvl3` | vLLM, TP | – | `requirements_eval_internvl3.txt` | `eval_internvl3` |
| Gemma-3-27B-it | `eval__gemma3` | `vllm_gemma3` | vLLM, TP | – | `requirements_eval_gemma3.txt` | `eval_gemma3` |
| Gemma-4-31B-it | `eval__gemma4` | `vllm_gemma4` | vLLM, TP | – | `requirements_eval_gemma4.txt` | `eval_gemma4` |
| Llama-3.2-Vision (11B) | `eval__llama3_2_vision` | `vllm_llama_3_2_vision` | vLLM, TP | – | `requirements_eval_llama3_vision.txt` | `eval_llama3vision` |
| LLaVA-OneVision (72B) | `eval__llava_onevision` | `vllm_llava_onevision` | vLLM, TP | – | `requirements_eval_llava_onevision.txt` | `eval_llavaonevision` |
| GLM-4.6V (107.7B MoE, ~12B active) / GLM-4.6V-Flash (10.3B dense) | `eval__glm4v` | `vllm_glm4v` | vLLM, TP | `glm4v` | `requirements_eval_glm4v.txt` | `eval_glm4v` |
| MiniMax-M3 (428B MoE) / MiniMax-M3-INT4 | `eval__minimax_m3` | `vllm_minimax_m3` | vLLM, TP | `minimax_m3` | `requirements_eval_minimax-m3-int4.txt` | `eval_minimax-m3-int4` |
| MedGemma (4B / 27B) | `eval__medgemma` | `medgemma` | HF pipeline, DP (`accelerate launch`) | – | `requirements_eval_medgemma.txt` | `eval_medgemma` |
| Lingshu (32B) | `eval__lingshu` | `lingshu` | HF, DP | `lingshu` | `requirements_eval_lingshu.txt` | `eval_lingshu` |
| MedDr (40B) | `eval__meddr` | `meddr` | HF, DP + DDP wrapper | `meddr` | `requirements_eval_meddr.txt` | `eval_meddr` |
| HuatuoGPT-Vision (34B) | `eval__huatuogpt_vision` | `huatuogpt_vision` | HF, DP | `huatuogpt_vision` | `requirements_eval_huatuogpt_vision.txt` | `eval_huatuogptvision` |
| LLaVA-Med (7B) | `eval__llava_med` | `llava_med` | HF, DP + DDP wrapper | `llava_med` | `requirements_eval_llavamed.txt` | `eval_llavamed` |
| HealthGPT-L14 (14B) | `eval__healthgpt` | `healthgpt` | HF, DP + DDP wrapper (third-party checkout + HLoRA weights) | – | `requirements_eval_healthgpt.txt` | `eval_healthgpt` |
| Claude (Fable 5 and other Anthropic codes) | `eval__claude` | `claude` | API (`anthropic` or `openrouter` provider) | `claude` | `requirements_eval_claude.txt` | `eval_claude` |
| GPT-5.5 / GPT-5.5-Pro | `eval__openai` | `openai` | API (OpenAI or OpenRouter) | `openai` | `requirements_eval_gpt.txt` | `eval_gpt` |
| Gemini-3.1-Pro | `eval__gemini` | `gemini` | API (Google GenAI or OpenRouter) | `gemini` | `requirements_eval_gemini.txt` | `eval_gemini` |
| Kimi-K2.6 | `eval__kimi` | `kimi` | API (Moonshot or OpenRouter) | `kimi` | `requirements_eval_kimi.txt` | `eval_kimi` |

Launchers exist for every row **except `Qwen2.5-VL tool-use (SFT variant)`** (whose launchers are kept
outside the tracked tree, under `local/`) in each of the three task folders
(`script/benchmark-{detect,TL,AD}/eval__<Display>__{detect,TL,AD}.sh`, 24 per
task family). The bundled launcher generator
`../sub-skills/benchmark-evaluation/scripts/make_eval_launcher.py` reproduces
them from `model_catalog.json`.

## Foundation pins per requirements file

| Requirements file | torch / torchvision | vllm | transformers | accelerate | huggingface_hub |
| --- | --- | --- | --- | --- | --- |
| `requirements_eval_qwen25vl.txt`, `requirements_eval_medvision-v0.txt` | 2.7.1 / 0.22.1 | 0.10.0 | 4.54.1 | 1.9.0 | 0.35.3 |
| `requirements_eval_qwen25vl_update1.txt` | 2.9.1 / 0.24.1 | 0.14.0 | 5.0.0rc2 | 1.9.0 | 1.3.3 |
| `requirements_eval_qwen3vl.txt` | 2.8.0 / 0.23.0 | 0.11.0 | 4.57.0 | 1.13.0 | 0.36.0 |
| `requirements_eval_internvl3.txt`, `requirements_eval_llava_onevision.txt` | 2.7.1 / 0.22.1 | 0.10.0 | 4.57.1 | 1.10.1 | 0.35.3 |
| `requirements_eval_gemma3.txt`, `requirements_eval_llama3_vision.txt` | 2.8.0 / 0.23.0 | 0.10.2 | 4.57.1 | 1.10.1 | 0.35.3 |
| `requirements_eval_gemma4.txt` | 2.10.0 / 0.25.0 | 0.19.0 | 5.10.2 | 1.13.0 | 1.18.0 |
| `requirements_eval_glm4v.txt` | 2.10.0 / 0.25.0 | 0.19.1 | 5.12.1 | 1.14.0 | 1.20.1 |
| `requirements_eval_minimax-m3-int4.txt` | 2.11.0 / 0.26.0 | (vLLM build with native `minimax_m3_vl`) | 5.12.1 | 1.14.0 | 1.20.1 |
| `requirements_eval_medgemma.txt` | 2.9.0 / 0.24.0 | – | 4.57.1 | 1.10.1 | 0.35.3 |
| `requirements_eval_lingshu.txt` | 2.6.0 / 0.21.0 | – | 4.52.1 | 1.10.1 | 0.35.3 |
| `requirements_eval_huatuogpt_vision.txt` | 2.6.0 / 0.21.0 | – | 4.40.0 | 1.10.1 | 0.35.3 |
| `requirements_eval_llavamed.txt`, `requirements_eval_meddr.txt` | 2.6.0 / 0.21.0 | – | 4.37.2 | 1.10.1 / 0.34.2 | 0.35.3 |
| `requirements_eval_healthgpt.txt` | 2.6.0 / 0.21.0 | – | (third-party pin) | 0.27.0 | 0.35.3 |
| `requirements_eval_{claude,gpt,gemini,kimi}.txt` | (CPU host is enough) | – | 4.57.1 | 1.14.0 | 0.36.0 |
| `requirements_sft_qwen25vl.txt` | 2.6.0 / 0.21.0 | – | 4.54.0 | 1.11.0 | 0.35.3 |
| `requirements_sft_medgemma.txt` | 2.6.0 / 0.21.0 | – | 4.54.0 | 1.9.0 | 0.36.0 |
| `requirements_sft_gemma4.txt`, `requirements_sft_qwen3.6vl.txt` | 2.6.0+cu124 / 0.21.0+cu124 | – | 5.5.0 | 1.14.0 | 1.22.0 |

Rule of thumb: transformers 4.x needs `huggingface_hub<1.0`; transformers 5.x
needs `huggingface_hub>=1.5`. Never mix stacks in one environment; the launchers
create one conda env per model (`ENV_NAME="eval-<model>"`).

## Parallelism (from the repository's parallelism summary)

- **Tensor parallel (vLLM)**: every `vllm_*` key. `tensor_parallel_size` equals the number of visible GPUs, so expose exactly the GPUs you want to shard across via `CUDA_VISIBLE_DEVICES`. `gpu_memory_utilization` is 0.9-0.95 across the launchers (0.9 x18, 0.90 x6, 0.95 x12; no launcher sets 0.99), and the wrapper defaults are 0.8-0.9.
- **Data parallel (HF)**: MedGemma, Lingshu, HuatuoGPT-Vision (per-process `device_map`), and LLaVA-Med, HealthGPT, MedDr (wrapped by `accelerator.prepare_model()`; unwrapped before `generate`). Launched with `accelerate launch --num_processes=<GPUs>`; each process holds a full model copy and `lmms_eval` shards the samples by rank.
- **API**: no local inference; concurrency and token budgets are set per request.

## Hardware notes (only what the repository documents)

- Paper evaluation rig: 4× H100 80 GB for all open-weight models; MedVision-V0 training on 4× H200 140 GB.
- **MiniMax-M3** (428B total / ~23B active MoE): all experts must be resident, so weight VRAM is ~856 GB in BF16, ~428 GB in FP8, ~214 GB in INT4/AWQ; INT4 fits at TP=4-8 on 80 GB cards, BF16 needs 8× H200. KV cache is negligible for MedVision's ~1.1K-token prompts.
- **GLM-4.6V** (107.7B MoE, ~12B active): ~215 GB BF16 weights → TP≥4 on 80 GB (matches the 4× H100 rig); TP=2 does not fit.
- **GLM-4.6V-Flash** (10.3B dense): ~21 GB BF16 weights → one ≥32 GB GPU, TP=1; exposing more GPUs wastes them.
- Everything else in the roster was run on the 4× H100 rig; per-GPU footprint is `weights / TP + KV + activations`. Smaller open-weight models (7B-14B) fit a single 80 GB GPU; the 27B-40B models are normally run with TP=2-4.

## Token budgets

The usual launcher values are `--max_new_tokens 4096` for local models and `--max_tokens 16000` for API models, and every wrapper honours the launcher value because no task YAML sets a budget. Several launcher families deviate deliberately, so read the launcher rather than assuming: the MiniMax-M3 pair passes 16384 on all three tasks, five other local launchers pass 16000, and the GPT launchers pass 4096, which is below the API default and is shared with hidden reasoning tokens. A missing budget in a new wrapper silently falls back to third-party defaults (the historical HuatuoGPT-Vision 512-token bug). Details: `../sub-skills/benchmark-evaluation/references/image-processing-and-token-budgets.md`.

## Perceived image size per model

Fixed perceived size: Gemma-3/MedGemma 896×896, MedDr 448×448, LLaVA-Med 336×336, HuatuoGPT-Vision 336×336, HealthGPT-L14 336×336. Input-dependent ("smart resize"): Qwen2.5-VL / Qwen3-VL / Lingshu (multiples of 28), Llama-3.2-Vision (560×560 tiles), LLaVA-OneVision (384×384 tiles), Gemma-4, GLM-4.6V, MiniMax-M3 (processor probes), InternVL3 (dynamic 448-px tiling — it stretches the image to `image_size × cols/rows`, so 448×448 only for square inputs; probed by `_process_img_internvl3`). API rules: Claude (28-grid pre-resize against a per-model cap table), Gemini, OpenAI, Kimi (provider resize rules re-implemented client-side). The dispatch lives in `get_resized_img_shape()`; see `../sub-skills/extending-models-and-tasks/references/image-size-dispatch.md`.

## Released MedVision checkpoints

- `YongchengYAO/MedVision-V0-7B`: Qwen2.5-VL-7B-Instruct → SFT (121K CoT samples) → GRPO RFT (sequential A/D → T/L → detection). Evaluate with `eval__medvision-model-rft` and the `eval__MedVision-V0-7B__*` launchers.
- SFT-only checkpoints per task (`YongchengYAO/MedVision__SFT-m__qwen25vl-{7b,32b}__{detect,TL,AD}`) are listed in the repository's SFT checkpoint document; see `../sub-skills/sft/SKILL.md`.
