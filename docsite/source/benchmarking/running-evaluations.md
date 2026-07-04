# Running evaluations

This page covers the first step of the benchmark pipeline: producing per-sample model outputs. Evaluation is the only step that needs a GPU (for open-weight models) or an API key (for hosted models); once outputs exist on disk, [parsing and summarizing](parsing-and-summarizing.md) run on CPU. For the bigger picture of the three-step flow, see the [benchmarking overview](overview.md).

Every model has a launcher shell script under `script/benchmark-AD/`, `script/benchmark-TL/`, or `script/benchmark-detect/` (one directory per task family). Each launcher wraps a single Python entry point, `python -m medvision_bm.benchmark.eval__<model>`. The launcher exists to prepare an isolated environment and pass the right flags; the entry point does the actual work.

## The launcher skeleton

Open-weight and API launchers share the same shape. Reading one top-to-bottom is the fastest way to understand what a run does.

1. **Create or reuse a per-model conda env.** Each model family gets its own environment (for example `eval-qwen25vl`, `eval-openai`, `eval-claude`) so that mutually incompatible `transformers` / `vllm` pins never collide. The launcher creates it on first run and reuses it afterwards.

2. **Build and install `medvision_bm` from source.** The launcher copies the packaging inputs (`pyproject.toml`, `src/`, and the other files the wheel needs) into a private temporary directory, builds the wheel there, and force-reinstalls it. Building in a private temp dir — rather than in the shared repo checkout — avoids an intermittent setuptools build failure on networked storage; a `flock` serializes the install into the shared conda env.

3. **Pin the dataset contract.** Two exports are set before anything touches the data:

   ```bash
   export MedVision_PLANNER_VERSION='1.0.0'
   export MedVision_ACK_RELEASE='1.1.1'
   ```

   `MedVision_PLANNER_VERSION` selects which released version of the dataset planner (i.e., annotations) the run evaluates. `MedVision_ACK_RELEASE` is a required acknowledgement whenever you pin the planner below the latest release. See [dataset concepts](../dataset/concepts.md) for what these control.

4. **Install runtime dependencies.** Three helper entry points do this:

   ```bash
   python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
   python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
   pip install -r "${benchmark_dir}/requirements/requirements_eval_qwen25vl.txt" --no-deps
   ```

   The first installs the `medvision_ds` dataset loader, the second installs the vendored `lmms-eval` harness with the model's optional-dependency group, and the third applies a frozen, per-model requirements file. The `--lmms_eval_opt_deps` value and the requirements filename change per model.

5. **Run the eval entry point.** The final command is `python -m medvision_bm.benchmark.eval__<model>` with the model's flags (below).

:::{note}
Each launcher offers two paths, Method 1 and Method 2. Method 1 installs requirements explicitly (as shown above) and then runs the entry point with `--skip_env_setup`, so the eval process trusts the environment you just built. Method 2 lets the entry point install everything itself. Method 1 is the more reproducible choice; Method 2 is convenient but exposes you to fresh upstream package versions.
:::

## Worked example: an open-weight vLLM model

`eval__qwen2_5_vl` is representative of the local-inference entry points. It launches the vendored `lmms-eval` harness against a vLLM backend, one task at a time, and records completion in the status file so a re-run resumes rather than repeats.

```bash
python -m medvision_bm.benchmark.eval__qwen2_5_vl \
    --skip_env_setup \
    --model_hf_id Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name Qwen2.5-VL-7B-Instruct \
    --results_dir "${benchmark_dir}/Results/MedVision-AD-CoT" \
    --data_dir "${benchmark_dir}/Data" \
    --tasks_list_json_path "${benchmark_dir}/tasks_list/tasks_MedVision-AD-CoT.json" \
    --task_status_json_path "${benchmark_dir}/completed_tasks/completed_tasks_MedVision-AD-CoT.json" \
    --batch_size_per_gpu 10 \
    --gpu_memory_utilization 0.9 \
    --sample_limit 1000
```

The core flags:

| Flag | Purpose |
| --- | --- |
| `--model_hf_id` | Hugging Face repo ID **or a local filesystem path** of the weights to load. |
| `--model_name` | Label used for the output subdirectory under `--results_dir` and for the status file. |
| `--data_dir` | Path to the local `Data/` tree (must match `MedVision_DATA_DIR`). |
| `--tasks_list_json_path` | JSON list of task names to iterate over, from `tasks_list/`. |
| `--task_status_json_path` | Resumable per-run status file under `completed_tasks/`; finished tasks are skipped. |
| `--results_dir` | Output root; per-sample JSONL lands in `<results_dir>/<model_name>/`. |
| `--batch_size_per_gpu` | Per-GPU batch size. |
| `--gpu_memory_utilization` | vLLM KV-cache memory fraction (e.g. `0.9`). |
| `--sample_limit` | Max samples evaluated per task. |

Optional knobs worth knowing:

- `--reshape_image_hw H,W` resizes every image before inference (useful for models or providers with a fixed input size).
- `--log-sys-prompt` records the system prompt, if any, in the per-sample JSONL.
- `--sample_indices` selects a subrange for partial runs; it overrides `--sample_limit` when set.
- `--lora_path` attaches a LoRA adapter; `--dtype` and `--max_new_tokens` override the defaults.

**Thinking / sampling variants.** `eval__qwen3_vl` adds sampling controls for the Qwen3-VL "Thinking" checkpoints, which are validated for sampling rather than greedy decoding:

```bash
python -m medvision_bm.benchmark.eval__qwen3_vl \
    --lmmseval_module vllm_qwen3vl \
    --model_hf_id Qwen/Qwen3-VL-32B-Thinking \
    --model_name Qwen3-VL-32B-Thinking \
    ... \
    --temperature 0.8 \
    --top_p 0.95 \
    --top_k 20 \
    --stop_strings '</answer>'
```

:::{warning}
For Thinking models, do not set `--temperature 0`: greedy decoding makes them collapse to an early end-of-sequence token while still inside the `<think>` block, so the final `<answer>` is never emitted. Use the sampling defaults instead (`--temperature 0.8 --top_p 0.95 --top_k 20`).

Generation stops on the model's end-of-sequence token by default. The old behaviour of forwarding lmms-eval's `\n\n` few-shot delimiter as a generation stop — which truncated the CoT before `<answer>` — was removed in v1.1.1; string stops now apply only when you pass `--stop_strings`. Passing `--stop_strings '</answer>'` is therefore **optional**: the launchers still include it as a clean, explicit terminator, but it is no longer required to prevent premature truncation.
:::

## Worked example: a hosted API model

API entry points skip the vLLM stack entirely and call the provider over HTTP. `eval__openai` handles both the OpenAI direct API and OpenRouter, selected with `--api_provider`:

```bash
python -m medvision_bm.benchmark.eval__openai \
    --skip_env_setup \
    --api_provider openrouter \
    --openai_model_code openai/gpt-5.5 \
    --model_name GPT-5.5 \
    --reasoning_effort low \
    --max_tokens 4096 \
    --results_dir "${benchmark_dir}/Results/MedVision-AD-CoT" \
    --data_dir "${benchmark_dir}/Data" \
    --tasks_list_json_path "${benchmark_dir}/tasks_list/tasks_MedVision-AD-CoT.json" \
    --task_status_json_path "${benchmark_dir}/completed_tasks/completed_tasks_MedVision-AD-CoT.json" \
    --batch_size 1 \
    --sample_limit 100
```

API-specific flags:

| Flag | Purpose |
| --- | --- |
| `--api_provider` | `openai` (direct) or `openrouter`. Governs which API key is required. |
| `--openai_model_code` | Provider model ID: e.g. `gpt-5.5` for OpenAI, `openai/gpt-5.5` for OpenRouter. |
| `--model_name` | Output-directory label, independent of the model code. |
| `--reasoning_effort` | `low` / `medium` / `high` for reasoning models; omitted entirely when unset, so the provider default applies. |
| `--max_tokens` | Default max output tokens per request; a per-task value from the task YAML takes precedence. |
| `--batch_size` | Concurrent requests. |

`--api_provider openai` reads `OPENAI_API_KEY`; `--api_provider openrouter` reads `OPENROUTER_API_KEY`. The entry point fails fast if the relevant key is empty.

:::{tip}
Pod-injected environment variables often carry a trailing newline that corrupts the HTTP `Authorization` header. The launchers strip it before running:

```bash
export "${api_key_var}"="$(printf '%s' "${!api_key_var}" | tr -d '\n')"
```

Do this for any API key you export by hand.
:::

**Anthropic.** `eval__claude` follows the same pattern with two differences: pass `--api_provider anthropic` and name the model with `--anthropic_model_code` (for example `claude-fable-5`); it reads `ANTHROPIC_API_KEY`. It also accepts `--thinking` / `--no-thinking` to toggle adaptive thinking.

:::{note}
**Image reshaping is not uniform across the API models.** The Claude, Gemini and Kimi launchers pin `--reshape_image_hw 512x512` on all three tasks, so the stated pixel size matches a fixed perceived resolution. The GPT launchers are the exception: they reshape to `512x512` only for the TL task and run at native resolution for AD and Detection. Open-weight launchers use native resolution (no reshape); only the released MedVision-V0 — trained at 512×512 — pins `--reshape_image_hw 512x512`.
:::

## Model-to-entry-point map

Pick the launcher directory by task (`benchmark-AD`, `benchmark-TL`, `benchmark-detect`); inside, each model's script calls the entry point below.

| Model | Entry point (`python -m medvision_bm.benchmark.<...>`) |
| --- | --- |
| Qwen2.5-VL | `eval__qwen2_5_vl` |
| Qwen3-VL | `eval__qwen3_vl` |
| GPT (e.g. GPT-5.5) | `eval__openai` |
| Claude | `eval__claude` |
| Gemini | `eval__gemini` |
| GLM-4.6V | `eval__glm4v` |
| Kimi | `eval__kimi` |
| MiniMax-M3 | `eval__minimax_m3` |
| Gemma-3 | `eval__gemma3` |
| Gemma-4 | `eval__gemma4` |
| MedGemma | `eval__medgemma` |
| InternVL3 | `eval__intern_vl3` |
| Llama-3.2-Vision | `eval__llama3_2_vision` |
| LLaVA-OneVision | `eval__llava_onevision` |
| LLaVA-Med | `eval__llava_med` |
| Lingshu | `eval__lingshu` |
| MedDr | `eval__meddr` |
| HuatuoGPT-Vision | `eval__huatuogpt_vision` |
| HealthGPT | `eval__healthgpt` |
| MedVision-V0 | `eval__medvision-model-rft` |

## Per-model example commands

Every command below is abbreviated to its **entry point and model-specific flags**. To each, add the shared flags shown in the worked examples above — `--data_dir`, `--tasks_list_json_path`, `--task_status_json_path`, `--results_dir`, and `--sample_limit` (plus `--skip_env_setup` under Method 1) — and pick the launcher directory for your task family (`benchmark-AD`, `benchmark-TL`, or `benchmark-detect`). Open-weight models take `--batch_size_per_gpu`; API models take `--batch_size` and require their provider's API key. Batch sizes shown are the launcher defaults for a single 80 GB GPU tier and are yours to tune.

### Open-weight models

:::{dropdown} Qwen2.5-VL — `eval__qwen2_5_vl`
The vLLM baseline the other wrappers extend; no unique flags. `--lmmseval_module` defaults to `vllm_qwen25vl` (use `vllm_qwen25vl_tooluse` for tool-use SFT variants).

```bash
python -m medvision_bm.benchmark.eval__qwen2_5_vl \
    --model_hf_id Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name Qwen2.5-VL-7B-Instruct \
    --batch_size_per_gpu 10 --gpu_memory_utilization 0.9
```
:::

:::{dropdown} Qwen3-VL (Thinking) — `eval__qwen3_vl`
Unique: sampling controls `--temperature` / `--top_p` / `--top_k` (do **not** set temperature 0). `--stop_strings` is optional (clean terminator).

```bash
python -m medvision_bm.benchmark.eval__qwen3_vl \
    --model_hf_id Qwen/Qwen3-VL-32B-Thinking \
    --model_name Qwen3-VL-32B-Thinking \
    --batch_size_per_gpu 2 --gpu_memory_utilization 0.95 \
    --temperature 0.8 --top_p 0.95 --top_k 20 \
    --stop_strings '</answer>'
```
:::

:::{dropdown} InternVL3 — `eval__intern_vl3`
No unique flags; default `--batch_size_per_gpu` is `1`.

```bash
python -m medvision_bm.benchmark.eval__intern_vl3 \
    --model_hf_id OpenGVLab/InternVL3-38B \
    --model_name InternVL3-38B \
    --batch_size_per_gpu 2 --gpu_memory_utilization 0.9
```
:::

:::{dropdown} Gemma-3 — `eval__gemma3`
No unique flags.

```bash
python -m medvision_bm.benchmark.eval__gemma3 \
    --model_hf_id google/gemma-3-27b-it \
    --model_name Gemma-3-27b-it \
    --batch_size_per_gpu 4 --gpu_memory_utilization 0.9
```
:::

:::{dropdown} Gemma-4 — `eval__gemma4`
Unique: `--enable_thinking` / `--no-enable_thinking` (launchers pass `--no-enable_thinking`), `--max_model_len` (caps the 256K context to fit KV cache), and `--min_new_tokens`.

```bash
python -m medvision_bm.benchmark.eval__gemma4 \
    --model_hf_id google/gemma-4-31B-it \
    --model_name gemma-4-31B-it \
    --batch_size_per_gpu 10 --gpu_memory_utilization 0.95 \
    --max_model_len 8192 --no-enable_thinking \
    --stop_strings '</answer>'
```
:::

:::{dropdown} Llama-3.2-Vision — `eval__llama3_2_vision`
No unique flags.

```bash
python -m medvision_bm.benchmark.eval__llama3_2_vision \
    --model_hf_id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --model_name Llama-3.2-11B-Vision-Instruct \
    --batch_size_per_gpu 4 --gpu_memory_utilization 0.9
```
:::

:::{dropdown} LLaVA-OneVision — `eval__llava_onevision`
Unique: `--max_model_len` (set e.g. `16384` when GPU memory is tight). Default `--batch_size_per_gpu` is `1`.

```bash
python -m medvision_bm.benchmark.eval__llava_onevision \
    --model_hf_id llava-hf/llava-onevision-qwen2-72b-ov-hf \
    --model_name LLaVA-OneVision \
    --batch_size_per_gpu 1 --gpu_memory_utilization 0.9 \
    --max_model_len 16384
```
:::

:::{dropdown} GLM-4.6V / GLM-4.6V-Flash — `eval__glm4v`
Unique: sampling controls plus `--repetition_penalty`. `--model_hf_id` selects the MoE (`zai-org/GLM-4.6V`) or dense (`zai-org/GLM-4.6V-Flash`) checkpoint.

```bash
python -m medvision_bm.benchmark.eval__glm4v \
    --lmmseval_module vllm_glm4v \
    --model_hf_id zai-org/GLM-4.6V \
    --model_name GLM-4.6V \
    --batch_size_per_gpu 1 --gpu_memory_utilization 0.95 \
    --temperature 0.8 --top_p 0.6 --top_k 2 --repetition_penalty 1.1 \
    --stop_strings '</answer>'
# GLM-4.6V-Flash: --model_hf_id zai-org/GLM-4.6V-Flash --model_name GLM-4.6V-Flash
```
:::

:::{dropdown} MedGemma — `eval__medgemma`
Runs through `accelerate` (not vLLM), so it does **not** accept `--gpu_memory_utilization`, `--dtype`, `--lora_path`, or `--stop_strings`.

```bash
python -m medvision_bm.benchmark.eval__medgemma \
    --model_hf_id google/medgemma-4b-it \
    --model_name MedGemma-4b-it \
    --batch_size_per_gpu 10
```
:::

:::{dropdown} Lingshu — `eval__lingshu`
Runs through `accelerate` with FlashAttention-2; no `--stop_strings` or vLLM flags.

```bash
python -m medvision_bm.benchmark.eval__lingshu \
    --model_hf_id lingshu-medical-mllm/Lingshu-32B \
    --model_name Lingshu-32b \
    --batch_size_per_gpu 2
```
:::

:::{dropdown} LLaVA-Med — `eval__llava_med`
Unique: `--dir_third_party` (clones the upstream repo at a pinned commit) and `--stop_strings`.

```bash
python -m medvision_bm.benchmark.eval__llava_med \
    --model_hf_id microsoft/llava-med-v1.5-mistral-7b \
    --model_name LLaVA-Med \
    --dir_third_party ./third_party \
    --batch_size_per_gpu 20 --stop_strings '</answer>'
```
:::

:::{dropdown} MedDr — `eval__meddr`
Unique: `--dir_third_party` (clones InternVL + MedDr at pinned commits).

```bash
python -m medvision_bm.benchmark.eval__meddr \
    --model_hf_id Sunanhe/MedDr_0401 \
    --model_name MedDr \
    --dir_third_party ./third_party \
    --batch_size_per_gpu 2
```
:::

:::{dropdown} HuatuoGPT-Vision — `eval__huatuogpt_vision`
Unique: `--dir_third_party` and `--stop_strings`. This is the only entry point with no `--max_new_tokens` flag.

```bash
python -m medvision_bm.benchmark.eval__huatuogpt_vision \
    --model_hf_id FreedomIntelligence/HuatuoGPT-Vision-34B \
    --model_name HuatuoGPT-Vision-34B \
    --dir_third_party ./third_party \
    --batch_size_per_gpu 2 --stop_strings '</answer>'
```
:::

:::{dropdown} HealthGPT — `eval__healthgpt`
Unique: `--model_choice` (`HealthGPT-L14` or `HealthGPT-XL32`) replaces `--model_hf_id` and selects the weights/base-model/dtype. Also `--dir_third_party`.

```bash
python -m medvision_bm.benchmark.eval__healthgpt \
    --model_choice HealthGPT-L14 \
    --model_name HealthGPT-L14 \
    --dir_third_party ./third_party \
    --batch_size_per_gpu 10
```
:::

:::{dropdown} MedVision-V0 — `eval__medvision-model-rft`
Unique: `--use_system_prompt` (injects the verl RFT training system prompt) and `--lmms_eval_module` (note the underscores). Pins `--reshape_image_hw 512x512`, the resolution it was trained at.

```bash
python -m medvision_bm.benchmark.eval__medvision-model-rft \
    --model_hf_id YongchengYAO/MedVision-V0-7B \
    --model_name MedVision-V0-7B \
    --batch_size_per_gpu 10 --gpu_memory_utilization 0.9 \
    --reshape_image_hw 512x512 --use_system_prompt
```
:::

### API models

:::{dropdown} Claude — `eval__claude`
Unique: `--anthropic_model_code`, `--thinking` / `--no-thinking`. `--api_provider` is `anthropic` (reads `ANTHROPIC_API_KEY`) or `openrouter` (reads `OPENROUTER_API_KEY`).

```bash
export ANTHROPIC_API_KEY=...
python -m medvision_bm.benchmark.eval__claude \
    --api_provider anthropic \
    --anthropic_model_code claude-fable-5 \
    --model_name Claude-Fable-5 \
    --batch_size 1 --sample_limit 100 \
    --reshape_image_hw 512x512
```
:::

:::{dropdown} OpenAI / GPT — `eval__openai`
Unique: `--openai_model_code`, `--reasoning_effort`. `--api_provider` is `openai` (reads `OPENAI_API_KEY`) or `openrouter` (reads `OPENROUTER_API_KEY`). GPT reshapes to 512×512 only for TL; AD/Detection run at native resolution.

```bash
export OPENROUTER_API_KEY=...
python -m medvision_bm.benchmark.eval__openai \
    --api_provider openrouter \
    --openai_model_code openai/gpt-5.5 \
    --model_name GPT-5.5 \
    --reasoning_effort low --max_tokens 4096 \
    --batch_size 1 --sample_limit 100
```
:::

:::{dropdown} Gemini — `eval__gemini`
Unique: `--google_model_code`; optional `--thinking_level`, `--media_resolution`, `--use_tool`, `--json_output`. `--api_provider` is `google` (reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`) or `openrouter`.

```bash
export GEMINI_API_KEY=...
python -m medvision_bm.benchmark.eval__gemini \
    --api_provider google \
    --google_model_code gemini-3.1-pro-preview \
    --model_name Gemini-3.1-Pro \
    --batch_size 1 --sample_limit 100 \
    --reshape_image_hw 512x512
```
:::

:::{dropdown} Kimi — `eval__kimi`
Unique: `--kimi_model_code`. `--api_provider` is `moonshot` (reads `MOONSHOT_API_KEY`; optional `MOONSHOT_BASE_URL`) or `openrouter`.

```bash
export MOONSHOT_API_KEY=...
python -m medvision_bm.benchmark.eval__kimi \
    --api_provider moonshot \
    --kimi_model_code kimi-k2.6 \
    --model_name Kimi-K2.6 \
    --batch_size 1 --sample_limit 100 \
    --reshape_image_hw 512x512
```
:::

## Debug and control flags

These three flags are shared across entry points and are meant for iterating without re-doing work:

- `--env_setup_only` runs the dependency-install steps and exits before any inference. Use it to warm an environment.
- `--skip_env_setup` skips installation and trusts the current environment. This is what Method 1 launchers pass after installing requirements themselves.
- `--skip_update_status` runs the tasks but does not write completion into the status file, so a run stays repeatable while you debug.

For the complete flag list of every entry point, see the [CLI reference](../reference/cli.md).
