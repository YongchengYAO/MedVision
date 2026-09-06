---
name: benchmark-evaluation
description: "Runs, scripts, resumes and debugs step 1 of the MedVision benchmark: the 21 python -m medvision_bm.benchmark.eval__<model> entry points (vLLM tensor-parallel, HF data-parallel via accelerate launch, and API models), the shell-launcher anatomy (benchmark_dir/data_dir/model_hf_id/model_name/batch_size_per_gpu/gpu_memory_utilization/task_tag/result_dir/sample_limit/max_new_tokens, MedVision_PLANNER_VERSION and the T/L MedVision_ACK_RELEASE, the node-local wheel build, the load-bearing Method 1 install trio, --skip_env_setup / --env_setup_only / --skip_update_status), sample limits (1000 open-weight vs 100 API pilot), output token budgets, CUDA_VISIBLE_DEVICES to tensor_parallel_size, the Results/<task_tag>/<model_name>/ layout, the completed_tasks tracker, the crash-safe response_cache resume (MEDVISION_RESP_CACHE), API keys and OpenRouter providers, and the perceived-image-size / pixel-size invariant. Use when a user asks to evaluate a VLM on Detection, T/L or A/D, write or fix an eval launcher, resume an interrupted run, choose a batch size or token budget, or explain why an evaluation skipped, OOM'd, truncated or produced no answers."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision benchmark evaluation (step 1)

Use this sub-skill to produce `Results/<task_tag>/<model_name>/*.jsonl` — the raw per-sample model responses that steps 2-4
(parse, summarize, LLM-judge) consume. It covers the three task families: Detection (bounding box, relative `[0,1]`
coordinates), Tumor/Lesion size (T/L, mm) and Angle/Distance (A/D, degrees or mm).

**Open-weight evaluation requires CUDA GPUs** (vLLM or HF data-parallel; 1-8 x 80 GB depending on the model). API models
(Claude / GPT / Gemini / Kimi) require credentials and network but no GPU. Nothing here runs usefully on a CPU-only host
beyond `--help`, `--env_setup_only` and the bundled inspection scripts.

## Quick start (one model, one task)

```bash
# 0. one environment per model (see ../environment-setup/SKILL.md); then, in it:
export MedVision_PLANNER_VERSION='1.0.0'          # REQUIRED by the dataset loader
export MedVision_ACK_RELEASE='1.4.0'              # T/L tasks only, when pinning below the newest release
export CUDA_VISIBLE_DEVICES=0,1,2,3               # vLLM: tensor_parallel_size == number of ids listed

# 1. the load-bearing "Method 1" install trio (order matters; frozen pins applied LAST)
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
pip install -r <repo>/requirements/requirements_eval_qwen25vl.txt --no-deps

# 2. run (requires GPU)
python -m medvision_bm.benchmark.eval__qwen2_5_vl \
    --skip_env_setup \
    --model_hf_id Qwen/Qwen2.5-VL-7B-Instruct --model_name Qwen2.5-VL-7B-Instruct \
    --results_dir <repo>/Results/MedVision-detect-CoT --data_dir <data_dir> \
    --tasks_list_json_path <repo>/tasks_list/tasks_MedVision-detect-CoT.json \
    --task_status_json_path <repo>/completed_tasks/completed_tasks_MedVision-detect-CoT.json \
    --batch_size_per_gpu 10 --gpu_memory_utilization 0.9 --max_new_tokens 4096 --sample_limit 1000
```

Generate that whole launcher (including the conda block, the node-local wheel build and the commented alternative
install method) instead of typing it:

```bash
python scripts/make_eval_launcher.py --list-models
python scripts/make_eval_launcher.py --model qwen25vl --task detect --benchmark-dir <repo> --out run.sh
python scripts/make_eval_launcher.py --model claude --task AD --api-provider openrouter --dry-run
```

Then check what landed on disk and what is left to run:

```bash
python scripts/check_results_tree.py --results-dir <repo>/Results/MedVision-detect-CoT --repo-root <repo> --show-tasks
```

## Route by task

| Need | Read / run |
|---|---|
| Plan and execute a run end to end: first run, resume, partial re-run, pilot limits, multi-task sweeps, API runs | `references/workflows.md` |
| Understand or rewrite a launcher: every variable, the wheel-build block, Method 1 vs Method 2, the two-pass MiniMax pattern, API-key sanitising | `references/launcher-anatomy.md` |
| Exact flags of every `eval__<model>` entry point + the `lmms_eval` command it builds + the env vars it sets | `references/cli-reference.md` |
| Which entry point / `lmms_eval` key / extra / requirements file / launcher defaults a model uses | `references/model-catalog.md` (hardware and pins: `../../references/model-roster.md`) |
| Why the prompt's image size and pixel size must match what the model perceives, and how output token budgets resolve | `references/image-processing-and-token-budgets.md` |
| GPU count, `tensor_parallel_size`, `batch_size_per_gpu`, data-parallel vs tensor-parallel | `references/hardware-and-parallelism.md` |
| A crash, an OOM, an empty answer, a skipped task, a 400/401/402 | `references/troubleshooting.md` |
| Emit a launcher for any catalogued model x task | `python scripts/make_eval_launcher.py --help` |
| Inventory a `Results/<task_tag>/` tree: outputs, resume cache, `parsed/`, tracker mismatches | `python scripts/check_results_tree.py --help` |

## Facts to keep straight (verified against the package)

- **21 entry points, 24 launcher stems.** `medvision_bm.benchmark` ships 21 `eval__*.py` modules; the repository's
  `script/benchmark-{detect,TL,AD}/` folders hold 24 launchers each, because some modules serve several checkpoints
  (`eval__medgemma` → 4B and 27B, `eval__glm4v` → GLM-4.6V and -Flash, `eval__minimax_m3` → MXFP8 and INT4,
  `eval__openai` → GPT-5.5 and GPT-5.5-Pro). `eval__qwen25vl_tooluse` has no launcher, so 20
  launchered modules + 4 doubled = 24. All 21 modules import and print `--help` on a CPU-only host.
- **The driver is a loop, not an engine.** Each `main()` reads task names from the task-list JSON (`load_tasks`), skips
  any already marked in `completed_tasks/completed_tasks_<task_tag>.json` (`load_tasks_status`), builds and
  `subprocess.run`s one `python3 -m lmms_eval --model <key> --model_args ... --tasks <task> --batch_size ... --limit ...
  --log_samples --output_path <results_dir>/<model_name>` per task, and only on return code 0 calls
  `update_task_status` (atomic write). One failed task does not abort the others.
- **`batch_size` is `batch_size_per_gpu * num_processes`**, where `num_processes = set_cuda_num_processes()` = the number
  of ids in `CUDA_VISIBLE_DEVICES` (or `torch.cuda.device_count()` when unset). The same number becomes vLLM's
  `tensor_parallel_size` and `accelerate launch --num_processes` for the HF models.
- **Sample limits.** 1000 samples per subtask for open-weight models, 100 for the API pilot study. `--sample_indices
  "[start:stop]"` overrides `--sample_limit` for partial inference.
- **Token budgets.** Launchers pass `--max_new_tokens` (local, default 4096) or `--max_tokens` (API, default 16000).
  No MedVision task YAML sets `max_new_tokens`, so the launcher value decides in practice. Several launchers override:
  MiniMax-M3 uses 16384 everywhere, MedGemma-27B 16000 on T/L and A/D, GLM-4.6V(-Flash) 16000 on A/D,
  Llama-3.2-Vision 16000 on Detection, and the GPT launchers cap `max_tokens` at 4096 (shared with hidden reasoning).
- **Resume is automatic.** Every finished response is appended and fsynced to
  `Results/<task_tag>/<model_name>/response_cache/<task>_rank<N>.jsonl` as it is produced. The key is
  `task::split::doc_id::sha256(prompt)[:16]`, so an unchanged prompt resumes and an edited prompt/config invalidates
  itself. `MEDVISION_RESP_CACHE=0` disables the layer. Caching assumes greedy decoding.
- **Two different "done" markers.** The response cache resumes *within* a task; `completed_tasks_<task_tag>.json`
  skips a *whole* task. Deleting the cache forces regeneration; removing the tracker entry (or `--skip_update_status`)
  is what lets a completed task run again.
- **The pixel-size invariant.** T/L and A/D prompts state the image size and pixel size, so those numbers must describe
  the image *after the model's own internal resize*. `get_resized_img_shape(model_name, ...)` dispatches on the
  `lmms_eval` model key injected by `evaluator.py`; an unregistered key raises
  `[Error] <name> is not recognised/supported.` Detection prompts are relative and never call it.
- **`MedVision_PLANNER_VERSION` is part of the experiment identity.** The loader hard-fails without it; the leaderboard
  uses `1.0.0`; T/L launchers additionally export `MedVision_ACK_RELEASE='1.4.0'` because only T/L annotations changed
  across releases.

## Bundled scripts

- `scripts/make_eval_launcher.py` — emits a runnable bash launcher for one catalogued model x one task, reproducing the
  repository skeleton (conda env, node-local wheel build with `flock`, planner-version exports, the Method 1 install
  trio with the right `--lmms_eval_opt_deps` extra and requirements file, the eval command with exactly the flags that
  model's launcher passes, API-key sanitising, the commented alternative method). `--list-models`, `--dry-run`,
  `--install-mode {wheel,editable,skip}`, `--no-conda-env`, `--method {1,2,two-pass}`, and per-run overrides for
  paths, limits, budgets and providers. Generates only; never installs or runs anything.
- `scripts/model_catalog.json` — the machine-readable data behind the generator (entry point, `lmms_eval` key, backend,
  extra, requirements file, conda env, launcher defaults, per-task budget overrides, API providers and key variables,
  pins, repository launcher names). Read or extend it when adding a model.
- `scripts/check_results_tree.py` — read-only inventory of `Results/<task_tag>/`: per model the sample JSONLs and
  `*_results.json`, sample counts, `response_cache` shards, `parsed/` and `llm-parsed_<judge>/` folders, and the
  differences against the task list and the `completed_tasks` tracker (`--strict` exits 1 on gaps).

## Boundaries

- Installing packages, per-model pins, Docker images, `env_setup` internals: `../environment-setup/SKILL.md`.
- Task lists, dataset config names, downloads, annotation versions as *data*: `../dataset-and-tasks/SKILL.md`
  (task inventory: `../dataset-and-tasks/scripts/list_tasks.py`).
- Parsing responses and computing metrics (steps 2-3): `../results-parsing-and-metrics/SKILL.md`.
- The LLM-judge re-parse (step 4): `../llm-judge-parsing/SKILL.md`.
- Registering a new model key, its resize branch and its launchers: `../extending-models-and-tasks/SKILL.md`.
- Per-model VRAM, GPU counts and foundation pins: `../../references/model-roster.md`; vocabulary:
  `../../references/concepts-and-glossary.md`; cross-cutting failures: `../../references/troubleshooting.md`.

## Safe operating rules

1. **Never start an evaluation the user did not ask for.** A full open-weight run is hours of GPU time; an API run
   spends real money. Confirm model, task, `sample_limit` and `task_tag` first, and prefer `--env_setup_only` or a
   `--sample_indices "[0:2]"` smoke run to validate wiring.
2. **State the target interpreter and GPUs.** Each model has its own conda env and pin set; check `which python` and
   `CUDA_VISIBLE_DEVICES` before running, and do not `pip install` into a working eval env to satisfy an import.
3. **Treat `Results/`, `completed_tasks/` and `response_cache/` as data.** Do not hand-edit result JSONLs. Deleting a
   response cache silently costs a full re-run; deleting a tracker entry silently re-runs a task.
4. **Keep the run identity fixed across a report**: planner version, `sample_limit`, token budget and `reshape_image_hw`
   must match for every model whose numbers are compared. Put reduced-limit or exploratory runs in their own
   `task_tag` (the repository keeps such trees under names like `MedVision-TL-CoT-limit100`).
5. **Do not run the repository's private launcher directories.** Reproduce the public
   `script/benchmark-{detect,TL,AD}/eval__*.sh` recipe with `scripts/make_eval_launcher.py` instead.
6. **Sanitise credentials, never print them.** `export KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"`; a
   container-injected trailing newline produces an HTTP 401 with an otherwise valid key.
