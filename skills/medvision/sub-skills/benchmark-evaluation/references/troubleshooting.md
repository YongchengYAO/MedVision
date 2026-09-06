# Troubleshooting — running an evaluation

Symptom → likely cause → concrete fix → when to stop. Cross-cutting install and dataset failures are in
`../../../references/troubleshooting.md`; metric oddities in `../../results-parsing-and-metrics/references/troubleshooting.md`.

Before anything else, establish three facts: which interpreter (`which python`), which GPUs
(`echo $CUDA_VISIBLE_DEVICES`, `nvidia-smi`), and what is already on disk
(`python scripts/check_results_tree.py --results-dir <repo>/Results/<task_tag> --repo-root <repo> --show-tasks`).

## 1. The run does not start

| Symptom | Likely cause | Fix / check | Stop when |
|---|---|---|---|
| Loader aborts asking for an annotation version | `MedVision_PLANNER_VERSION` unset — the dataset loader hard-fails without it | `export MedVision_PLANNER_VERSION='1.0.0'` **before** the first `datasets` import (the launchers export it above the run block) | – |
| Loader refuses the pin / asks for an acknowledgement, on T/L only | pinning below the newest annotation release | also `export MedVision_ACK_RELEASE='1.4.0'`; every T/L launcher does, Detection and A/D launchers do not | – |
| `could not create '.../build/...': No such file or directory` while installing `medvision_bm` | setuptools `build_py` memo race on a shared network filesystem (CephFS-like) | build the wheel in a node-local temp dir and `flock` the install — the block in `launcher-anatomy.md`, or `make_eval_launcher.py --install-mode wheel`, or the bundled script in `../../environment-setup/SKILL.md` | – |
| `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'` in a worker or on the next task | transformers 4.x with `huggingface_hub>=1.0`, or transformers 5.x with `huggingface_hub==0.36.0`. Something re-pinned the hub mid-run — often the dataset loader's run-time reinstall of `medvision_ds` (it applies its own `huggingface_hub==0.36.0`, and pip output is swallowed) | reinstall the pin from the model's requirements file. For transformers-5 stacks set `MedVision_FORCE_INSTALL_CODE=false` after the installers have run (this is exactly what `eval__glm4v` does) so the loader cannot downgrade the hub | – |
| GLM-4.6V: `Unrecognized image processor` / vLLM refuses transformers 5 | `Glm46VImageProcessor` exists only in transformers ≥ 5.2.0, and vLLM 0.12.0 pins `transformers<5` | use the documented window: vLLM 0.19.x + transformers 5.12.1 (what `requirements_eval_glm4v.txt` pins) | – |
| MiniMax-M3: vLLM "unsupported architecture" | the pip vLLM does not register `minimax_m3_vl` (AWQ-INT4 needs a patched fork) | run the two-pass pattern: `--env_setup_only`, replace vLLM, `--skip_env_setup`; or point `--vllm_version` at a release that registers the architecture | no vLLM build registers it — stop |
| `AttributeError: ... TokenizersBackend ...` at vLLM `LLM()` init | transformers 5.x under a vLLM that expects 4.x | downgrade to the pin in the model's requirements file | – |
| Third-party model dies with `ModuleNotFoundError` only inside the `accelerate` subprocesses | the third-party package is not on `PYTHONPATH` in the child processes | export it before the run, e.g. `PYTHONPATH="<third_party>/HealthGPT:${PYTHONPATH:-}"` (the HealthGPT, MedDr and HuatuoGPT-Vision launchers do this) | – |
| `Cannot re-initialize CUDA in forked subprocess` | `lmms_eval.__main__` builds an `Accelerator()` before the model, touching CUDA in the parent | this is why `eval__glm4v` forces accelerate onto CPU and lets vLLM own the GPUs; reproduce that pattern rather than reordering the driver | – |

## 2. Out of memory

| Symptom | Likely cause | Fix / check | Stop when |
|---|---|---|---|
| `torch.OutOfMemoryError` at vLLM engine start | `gpu_memory_utilization` too high, or the weights do not fit at this `tensor_parallel_size` | lower `--gpu_memory_utilization` (0.99 → 0.9), then expose more GPUs (`CUDA_VISIBLE_DEVICES` **is** `tensor_parallel_size`); check the per-GPU footprint in `../../../references/model-roster.md` | `weights / TP` still exceeds one card at your maximum GPU count |
| OOM by a fraction of a GiB during vLLM's post-load profiling, with GiB "reserved but unallocated" | allocator fragmentation | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (inherited by the vLLM workers) | – |
| The engine dies, restarts, and fails the startup memory check in a loop | a dead vLLM engine leaks a few GiB, so a high utilisation target can no longer be met on retry | drop `gpu_memory_utilization` to ~0.90 and restart cleanly (this is why the MiniMax launchers use 0.90, not 0.95) | – |
| OOM mid-generation on a vLLM run | `max_num_seqs` (= `batch_size_per_gpu × GPUs`) too high for the KV budget, or `max_new_tokens` raised | lower `--batch_size_per_gpu`; keep `max_new_tokens` well under `max_model_len` minus prompt+image tokens | – |
| OOM on an HF data-parallel model | each rank holds a **full replica**; batch size does not shard | lower `--batch_size_per_gpu` (2 is the repository value for the 27-40 B models) or use fewer ranks; the model must fit **one** GPU | replica does not fit one GPU |
| Process killed with no traceback during model load | host RAM (cgroup `memory.max`, not what `free -g` reports); `cpu_offload_gb` is **per GPU worker**, so TP=4 multiplies it by 4; HuatuoGPT-Vision materialises N × 34 B on CPU when N ranks start together | keep `--cpu_offload_gb 0`; stagger or reduce ranks | container memory limit cannot be raised |

## 3. Nothing runs, or the wrong thing runs

| Symptom | Likely cause | Fix / check | Stop when |
|---|---|---|---|
| `Task <name> already completed. Skipping...` for a task you want to re-run | the entry exists in `completed_tasks/completed_tasks_<task_tag>.json` under that `model_name` | remove that one `{model: {task: true}}` entry, or run with `--skip_update_status` into a different `--results_dir`/`task_tag`; never delete the whole tracker for a shared tree | – |
| A re-run finishes instantly and produces identical responses | the per-sample response cache replayed them (`response_cache/<task>_rank<N>.jsonl`) | that is correct for a resume. To force regeneration delete the task's cache shards, or run the whole thing with `MEDVISION_RESP_CACHE=0` | – |
| A prompt/config edit did **not** invalidate the cache | the key hashes the rendered prompt only — a changed `max_new_tokens`, sampling parameter or wrapper fix does not change the prompt | delete the cache shards for the affected tasks before re-running | – |
| Two `*_samples_<task>.jsonl` files for one task | a re-run appended a second timestamped output; nothing is ever deleted | `check_results_tree.py` reports `duplicates=N`; move or delete the stale timestamped pair (`*_samples_<task>.jsonl` + its `*_results.json`) before parsing | – |
| Only a fraction of the expected tasks ran | one task returned non-zero: the driver prints `Warning: Task <name> failed (return code N)` and continues without marking it | read that task's log, fix the cause, re-run the same command (finished tasks are skipped) | – |
| Fewer samples than expected per task | `--sample_indices` overrides `--sample_limit`; or the task genuinely has fewer samples than the limit | drop `--sample_indices`; compare with the task-list JSON counts | – |
| A whole GPU set was claimed unexpectedly | `CUDA_VISIBLE_DEVICES` unset → `torch.cuda.device_count()` | always export it explicitly | – |

## 4. The model answers, but the results are bad

| Symptom | Likely cause | Fix / check | Stop when |
|---|---|---|---|
| Low SuccessRate; responses stop mid-sentence with no `</answer>` | output budget exhausted — the launcher value (`--max_new_tokens` / `--max_tokens`), not the module default, decides | raise the budget (see the documented overrides in `image-processing-and-token-budgets.md`), then **delete the cache shards** or the truncated responses replay | budget already near `max_model_len` minus prompt tokens |
| A reasoning model runs to the full budget without ever closing `</answer>` | nothing terminates it: the vendored engine injects no stop sequence and relies on the model's EOS token (`api/task.py:142-149`) | pass `--stop_strings '</answer>'` (what the launchers do for Qwen3-VL, Gemma-4, GLM-4.6V(-Flash) and MiniMax-M3); not for OpenAI reasoning models, which may reject a stop parameter | – |
| HuatuoGPT-Vision responses truncate at ~512 tokens | the upstream `HuatuoChatbot` hardcodes `max_new_tokens=512`; MedVision runs made before the 2026-08-08 fix (commit `09206a2`) inherited it silently | re-run with the fixed package (4096). Pre-fix and post-fix outputs are **not comparable**; clear the response cache before re-evaluating | – |
| Gemma-4 degenerates into repetition and ignores the tag format | native thinking mode is on | pass `--no-enable_thinking` (what the launcher does) | – |
| Every T/L or A/D measurement is off by a consistent factor, while Detection looks fine | the stated pixel size does not match the model's perceived resolution — Detection uses relative coordinates and never consults it | confirm the `--model` key has the right `get_resized_img_shape` branch and that `--reshape_image_hw` matches the run you are comparing against; see `image-processing-and-token-budgets.md` | – |
| Errors only on non-square slices, or coronal/sagittal OOD splits | the branch returns a single square constant where the processor letterboxes/pads (LLaVA-OneVision, Llama-3.2-Vision, the CLIP-336 trio) or tiles anisotropically (InternVL3) | the branch must return **both** the padded canvas and the pre-pad content; validate any new or changed branch on a non-square slice | – |
| Gemini measurements are hopeless on small structures | `media_resolution` unset → Gemini 2.5 returns a ~258-token thumbnail | the wrapper pins `high`; do not override it, and prefer the `google` provider (OpenRouter gives no control over this) | – |
| `ValueError: [Error] <name> is not recognised/supported.` | the `--model` key has no branch in `get_resized_img_shape` — a new or misspelled key, or an SFT alias that was never registered | use a registered key (20 exist in `AVAILABLE_MODELS`, 5 more commented out), or add both the registry entry and the resize branch: `../../extending-models-and-tasks/SKILL.md` | – |
| An API wrapper raises on the model code at start-up | unknown model codes are rejected on purpose so an unverified sibling cannot inherit another model's image caps | use a code from the wrapper's cap table, or add the new code with its verified caps | you cannot verify the new model's resize/token rules |

## 5. API failures

| Symptom | Likely cause | Fix / check | Stop when |
|---|---|---|---|
| HTTP 401 with a key you know is valid | the key carries a trailing newline (container/Kubernetes secrets usually do), which is illegal in an auth header | `export KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"` — the launchers strip the trailing newline (`tr -d '\n'`) before the run | – |
| HTTP 402 / "insufficient credits" although the balance looks sufficient | OpenRouter reserves the **full `max_tokens`** as a credit hold for the duration of each request; with `--max_tokens 16000` the hold can exceed the balance long before the tokens are spent | top up, lower `--max_tokens`, or switch to the direct provider (`--api_provider anthropic|openai|google|moonshot` with that vendor's key) | – |
| HTTP 400, retried once and abandoned | 400s are deterministic, so the backoff decorator gives up immediately instead of burning retries | read the request: an explicit `thinking: {"type": "disabled"}` is rejected by Claude Fable 5 (use `--no-thinking`, which omits the parameter entirely); OpenAI reasoning models may reject `--stop_strings`; a Gemini 3 model rejects `--thinkingBudget` and a 2.5 model rejects `--thinking_level` | – |
| A run is far more expensive than expected | `--sample_limit` above the 100-sample pilot convention, or a 16000-token budget on a verbose model | fix the limit before restarting; every API request costs money even when the response is later discarded | – |
| Wrong provider key requested | provider and model code must change together | `--api_provider openrouter` needs the vendor-prefixed code (`anthropic/claude-fable-5`, `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`, `moonshotai/kimi-k2.6`) **and** `OPENROUTER_API_KEY` | – |

## 6. When to stop and ask

- The model does not fit the GPUs available at the maximum tensor-parallel size you can expose.
- The host has no GPU at all: only `--help`, `--env_setup_only` and the bundled inspection scripts are meaningful.
- The fix requires changing a pinned version in an environment other runs depend on.
- Re-running would overwrite or invalidate published results — confirm the `task_tag`, the planner version and the
  roster with the user first.
- The step needs credentials, credits or a gated checkpoint you do not have.
