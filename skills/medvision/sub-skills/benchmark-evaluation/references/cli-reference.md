# CLI Reference — `medvision_bm.benchmark.eval__*`

All 21 entry points are invoked as `python -m medvision_bm.benchmark.eval__<name> [flags]` and all 21 import and print
`--help` on a CPU-only host (verified with the package installed; no GPU or credentials needed for `--help`). Values
below are the **module defaults**, which are frequently *not* the values the repository launchers pass — the launcher
column of `model-catalog.md` has those.

## Shared flags (every `lmms_eval`-driven entry point)

> `eval__qwen25vl_tooluse` calls vLLM directly and is the exception: it has no `--sample_indices`,
> `--reshape_image_hw`, `--log-sys-prompt`, `--env_setup_only` or `--scaled_ps_low`/`--scaled_ps_high`.

| Flag | Default | Meaning |
|---|---|---|
| `--model_name` | per model | run label → `Results/<task_tag>/<model_name>/` and the key inside the task-status JSON |
| `--results_dir` | – | `Results/<task_tag>`; the driver appends `<model_name>` and passes it as `--output_path` |
| `--data_dir` | – | MedVision data directory; also the root for the HF/vLLM caches |
| `--tasks_list_json_path` | – | JSON whose top-level keys are the task names to run, in file order |
| `--task_status_json_path` | – | `completed_tasks/completed_tasks_<task_tag>.json` |
| `--sample_limit` | `1000` | samples per task → `lmms_eval --limit` |
| `--sample_indices` | `None` | `[start:stop]` or `[start,stop,step]`; **overrides** `--sample_limit` for selection |
| `--reshape_image_hw` | `None` | reshape the 2D slice at NIfTI load, before both `doc_to_visual` and `doc_to_text`; accepts `512x512`, `512,512`, `[512,512]` |
| `--log-sys-prompt` | off | also log the system prompt in the per-sample JSONL |
| `--skip_env_setup` | off | do not install anything (Method 1); vLLM entry points still call `setup_env_vllm` |
| `--env_setup_only` | off | run the built-in install order and exit before inference |
| `--skip_update_status` | off | never write the task-status JSON |
| `--scaled_ps_low` / `--scaled_ps_high` | `0.5` / `3.0` | pixel-size scaling range for the experimental `-scaledPS` task variants; exported as `MEDVISION_SCALED_PS_LOW/HIGH` |

Local (non-API) entry points additionally share (except where noted):

| Flag | Default | Meaning |
|---|---|---|
| `--model_hf_id` | per model | Hub id **or** local checkpoint directory |
| `--max_new_tokens` | `4096` | output budget (see `image-processing-and-token-budgets.md`) |
| `--batch_size_per_gpu` | 1-50 per model | multiplied by the visible-GPU count before reaching `lmms_eval` |

> `--stop_strings` is **not** defined on `eval__lingshu`, `eval__medgemma`, `eval__meddr`,
> `eval__healthgpt` or `eval__qwen25vl_tooluse`.
| `--stop_strings STRING [...]` | `None` | e.g. `'</answer>'`. The only source of string-level stops — no delimiter is auto-injected; default stopping is the model's EOS |

vLLM entry points also share `--lora_path` (`None`), `--dtype` (`auto`) and `--gpu_memory_utilization` (`0.99`).

## Per-entry-point differences

| Entry point | `lmms_eval` key (`--model`) | Backend | Extra flags (defaults) |
|---|---|---|---|
| `eval__qwen2_5_vl` | `vllm_qwen25vl` | vLLM TP | `--lmmseval_module vllm_qwen25vl` (switch to `vllm_qwen25vl_tooluse`) |
| `eval__qwen3_vl` | `vllm_qwen3vl` | vLLM TP | `--lmmseval_module`, `--temperature 0.8 --top_p 0.95 --top_k 20` |
| `eval__gemma3` | `vllm_gemma3` | vLLM TP | – |
| `eval__gemma4` | `vllm_gemma4` | vLLM TP | `--enable_thinking/--no-enable_thinking` (default on), `--min_new_tokens 0`, `--max_model_len 8192` |
| `eval__glm4v` | `vllm_glm4v` | vLLM TP | `--lmmseval_module`, `--temperature 0.8 --top_p 0.6 --top_k 2 --repetition_penalty 1.1` |
| `eval__intern_vl3` | `vllm_internvl3` | vLLM TP | – |
| `eval__llama3_2_vision` | `vllm_llama_3_2_vision` | vLLM TP | – |
| `eval__llava_onevision` | `vllm_llava_onevision` | vLLM TP | `--max_model_len` (`None`) |
| `eval__minimax_m3` | `vllm_minimax_m3` | vLLM TP | `--lmmseval_module`, `--temperature 1.0 --top_p 0.95 --top_k 40`, `--cpu_offload_gb 0`, `--vllm_version 0.11.0` |
| `eval__medvision-model-rft` | `vllm_qwen25vl` | vLLM TP | `--lmms_eval_module` (note the **different spelling**), `--use_system_prompt` |
| `eval__medgemma` | `medgemma` | HF DP (`accelerate launch`) | – |
| `eval__lingshu` | `lingshu` | HF DP | – |
| `eval__meddr` | `meddr` | HF DP + DDP | `--dir_third_party` |
| `eval__huatuogpt_vision` | `huatuogpt_vision` | HF DP | `--dir_third_party`, `--do_sample`, `--temperature` (both `None`) |
| `eval__llava_med` | `llava_med` | HF DP + DDP | `--dir_third_party` |
| `eval__healthgpt` | `healthgpt` | HF DP + DDP | `--dir_third_party`, `--model_choice {HealthGPT-L14,HealthGPT-XL32}`; **no** `--model_hf_id` |
| `eval__claude` | `claude` | API | `--api_provider {anthropic,openrouter}`, `--anthropic_model_code`, `--thinking/--no-thinking` (on), `--max_tokens 16000`, `--batch_size 1` |
| `eval__openai` | `openai` | API | `--api_provider {openai,openrouter}`, `--openai_model_code`, `--reasoning_effort` (`None`), `--max_tokens 16000`, `--batch_size 1` |
| `eval__gemini` | `gemini` | API | `--api_provider {google,openrouter}`, `--google_model_code gemini-3.1-pro-preview`, `--thinking_level {minimal,low,medium,high}`, `--thinkingBudget`, `--media_resolution {low,medium,high}`, `--use_tool/--no-use_tool`, `--json_output/--no-json_output`, `--max_tokens 16000`, `--batch_size 1` |
| `eval__kimi` | `kimi` | API | `--api_provider {moonshot,openrouter}`, `--kimi_model_code`, `--max_tokens 16000`, `--batch_size 1` |
| `eval__qwen25vl_tooluse` | – (does not call `lmms_eval`) | vLLM, in-process | `--max_tokens_phase1 512`, `--max_tokens_phase2 64`, `--batch_size 20`, `--gpu_memory_utilization 0.95`, `--sample_limit 100`; two-phase generate → `safe_exec_python` → generate |

Notes:

- `--max_tokens` on the API entry points is documented as a *default* per request: a per-task `max_new_tokens` from a
  task YAML would take precedence, but no MedVision task YAML sets one.
- `eval__openai --stop_strings` is documented as unsafe for gpt-5.x / o-series, which may reject the stop parameter.
- `eval__gemini`: `--thinking_level` is for the Gemini 3 series and `--thinkingBudget` for 2.5; `--media_resolution`
  defaults to `high` when unset (the SDK default returns a ~258-token thumbnail on 2.5); `--use_tool` and
  `--json_output` are `google`-provider only.

## The `lmms_eval` command the driver builds

```
python3 -m lmms_eval \
  --model <lmms_eval key> \
  --model_args <k=v,…> \
  --tasks <one task name> \
  --batch_size <batch_size_per_gpu * num_processes>   # API: --batch_size as given
  --limit <sample_limit> \
  --log_samples \
  --output_path <results_dir>/<model_name> \
  --verbosity=INFO                                    # DEBUG in the HF data-parallel and API entry points
  [--sample_indices '[…]'] [--log_sys_prompt]
```

HF data-parallel entry points prefix this with `python -m accelerate.commands.launch --num_processes=<GPUs>`.

`model_args` per backend:

| Backend | `model_args` keys |
|---|---|
| vLLM | `model_hf`, optional `lora_path`, `gpu_memory_utilization`, `tensor_parallel_size`, `max_num_seqs` (= the computed batch size), `hf_overrides` (Qwen2.5-VL: the resolved `vision_start_token_id`), `max_new_tokens`, `dtype`, optional `stop_strings`, optional `reshape_image_hw`, plus model-specific sampling / `max_model_len` / `cpu_offload_gb` |
| HF | `model_hf`, `max_new_tokens`, optional `reshape_image_hw`, `stop_strings`, model-specific keys |
| API | `model=<model code>`, `provider=<api_provider>`, `model_hf=<model code>`, model-specific (`thinking`, `reasoning_effort`, `thinking_level`, `media_resolution`, …), `max_tokens`, optional `reshape_image_hw`, `stop_strings` |

Two of those keys are read back by the *task* side, not only the model: `evaluator.py` parses `model_args`, and injects
`model_hf`, the `--model` key (as `model_name`) and a normalised `reshape_image_hw` into every task's
`lmms_eval_specific_kwargs`. That is how the prompt builder knows which perceived-size rule to apply without any task
YAML edit.

## Helper functions used by the drivers (`medvision_bm.utils`, `medvision_bm.benchmark.eval_utils`)

| Function | Behaviour |
|---|---|
| `load_tasks(json_file_path)` | returns the top-level keys of the task-list JSON, in file order, and prints the count |
| `load_tasks_status(tasks_status_file, model_name)` | `{task: True}` for that model; `{}` if the file or the model entry is missing; raises `ValueError` if the file exists but cannot be parsed |
| `update_task_status(json_path, model_name, task_name)` | sets `data[model_name][task_name] = True` and writes via `atomic_write_json` |
| `atomic_write_json(json_path, data, indent=4)` | temp file in the same directory → `fsync` → `os.replace`; a failed write leaves the original intact instead of truncating it |
| `set_cuda_num_processes()` | number of ids in `CUDA_VISIBLE_DEVICES`, else `torch.cuda.device_count()`; ≥1 when the variable is set |
| `parse_sample_indices(s)` | `"[0:10]"` → `range(0,10)`; `"0,10,2"` → `range(0,10,2)`; anything else raises `ValueError` |
| `setup_env_hf_medvision_ds(data_dir)` | called **before** the env-setup branch in every driver; sets the `medvision_ds` and Hugging Face environment variables for `data_dir` |
| `setup_env_vllm(data_dir)` | sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `XDG_CACHE_HOME=<data_dir>/.cache/vllm`; called on the `--skip_env_setup` path and by `install_vllm` |

## Environment variables read or written during a run

| Variable | Set by | Effect |
|---|---|---|
| `MedVision_PLANNER_VERSION` | you (launcher) | **required**; selects the annotation version. Leaderboard runs use `1.0.0` |
| `MedVision_ACK_RELEASE` | you (T/L launchers) | acknowledges pinning below the newest annotation release (`1.4.0`) |
| `MedVision_DATA_DIR`, `HF_HOME`, `HF_DATASETS_CACHE`, `MedVision_FORCE_INSTALL_CODE` | `setup_env_hf_medvision_ds` / `install_medvision_ds` | dataset package + HF cache wiring. `eval__glm4v` deliberately sets `MedVision_FORCE_INSTALL_CODE=false` after setup so the loader's run-time reinstall cannot re-pin `huggingface_hub` under a transformers-5 stack |
| `MEDVISION_RESP_CACHE` | you | `0` disables the per-sample resume cache (default enabled) |
| `MEDVISION_SCALED_PS_LOW` / `_HIGH` | the driver, from `--scaled_ps_low/high` | pixel-size scaling range for `-scaledPS` variants |
| `CUDA_VISIBLE_DEVICES` | you | determines `tensor_parallel_size` (vLLM) and `--num_processes` (HF) |
| `VLLM_WORKER_MULTIPROC_METHOD`, `XDG_CACHE_HOME` | `setup_env_vllm` | vLLM spawn method and cache location |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `MOONSHOT_API_KEY` (+ `MOONSHOT_BASE_URL`), `OPENROUTER_API_KEY` | you | API credentials; strip whitespace before exporting |
| `HF_TOKEN` | you | gated checkpoints/datasets; strip whitespace |
| `PYTHONPATH` | HealthGPT, MedDr, HuatuoGPT-Vision launchers | prepends `<third_party>/<pkg>` so `accelerate` subprocesses can import it |

## Output files

```
Results/<task_tag>/<model_name>/
  <YYYYMMDD_HHMMSS>_samples_<task>.jsonl   # per-sample records written by lmms_eval --log_samples
  <YYYYMMDD_HHMMSS>_results.json           # aggregate for that invocation
  response_cache/<task>_rank<N>.jsonl      # {"key": "<task>::<split>::<doc_id>::<sha256(prompt)[:16]>", "response": …}
completed_tasks/completed_tasks_<task_tag>.json   # {"<model_name>": {"<task>": true, …}, …}
```

The tool-use entry point writes `<task>_samples_0.jsonl` instead (it does not go through `lmms_eval`).
`scripts/check_results_tree.py` understands both shapes.
