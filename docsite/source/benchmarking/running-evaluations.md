# Running evaluations

This page covers the first step of the benchmark pipeline: producing per-sample model outputs. Evaluation is the only step that needs a GPU (for open-weight models) or an API key (for hosted models); once outputs exist on disk, [parsing and summarizing](parsing-and-summarizing.md) run on CPU. For the bigger picture of the three-step flow, see the [benchmarking overview](overview.md).

Every model has a launcher shell script under `script/benchmark-AD/`, `script/benchmark-TL/`, or `script/benchmark-detect/` (one directory per task family). Each launcher wraps a single Python entry point, `python -m medvision_bm.benchmark.eval__<model>`. The launcher exists to prepare an isolated environment and pass the right flags; the entry point does the actual work.

## The launcher skeleton

Open-weight and API launchers share the same shape. Reading one top-to-bottom is the fastest way to understand what a run does.

1. **Create or reuse a per-model conda env.** Each model family gets its own environment (for example `eval-qwen25vl`, `eval-openai`, `eval-claude`) so that mutually incompatible `transformers` / `vllm` pins never collide. The launcher creates it on first run and reuses it afterwards.

2. **Build and install `medvision_bm` from source.** The launcher tars `pyproject.toml`, `src/`, and friends into a private temp directory, builds a wheel there, and force-reinstalls it. Building off the shared filesystem avoids a setuptools caching failure on networked storage; a `flock` serializes the shared-env install.

3. **Pin the dataset contract.** Two exports are set before anything touches the data:

   ```bash
   export MedVision_PLANNER_VERSION='1.0.0'
   export MedVision_ACK_RELEASE='1.1.1'
   ```

   `MedVision_PLANNER_VERSION` selects which released version of the dataset planner (and therefore which sample set) the run evaluates. `MedVision_ACK_RELEASE` is a required acknowledgement whenever you pin the planner below the latest release. See [dataset concepts](../dataset/concepts.md) for what these control.

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
| `--model_hf_id` | Hugging Face repo ID of the weights to load. |
| `--model_name` | Label used for the output subdirectory under `--results_dir` and for the status file. |
| `--data_dir` | Path to the local `Data/` tree (must match `MedVision_DATA_DIR`). |
| `--tasks_list_json_path` | JSON list of task names to iterate over, from `tasks_list/`. |
| `--task_status_json_path` | Resumable per-run status file under `completed_tasks/`; finished tasks are skipped. |
| `--results_dir` | Output root; per-sample JSONL lands in `<results_dir>/<model_name>/`. |
| `--batch_size_per_gpu` | Per-GPU batch size; multiplied by the visible GPU count internally. |
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
For Thinking models, do not set `--temperature 0`. Passing an explicit `--stop_strings '</answer>'` is also needed: it gives a clean terminator and signals the wrapper to drop the harness's default `\n\n` stop, which would otherwise cut generation off mid-reasoning before the final answer is emitted.
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

**Anthropic.** `eval__claude` follows the same pattern with two differences: pass `--api_provider anthropic` and name the model with `--anthropic_model_code` (for example `claude-fable-5`); it reads `ANTHROPIC_API_KEY`. The Claude launcher also commonly sets `--reshape_image_hw 512x512`, since the provider counts image tokens by resolution.

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

## Debug and control flags

These three flags are shared across entry points and are meant for iterating without re-doing work:

- `--env_setup_only` runs the dependency-install steps and exits before any inference. Use it to warm an environment.
- `--skip_env_setup` skips installation and trusts the current environment. This is what Method 1 launchers pass after installing requirements themselves.
- `--skip_update_status` runs the tasks but does not write completion into the status file, so a run stays repeatable while you debug.

For the complete flag list of every entry point, see the [CLI reference](../reference/cli.md).
