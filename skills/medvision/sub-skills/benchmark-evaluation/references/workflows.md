# Evaluation Workflows

Every command below is step 1 of the four-step MedVision pipeline. Steps 2-4 are owned by
`../../results-parsing-and-metrics/SKILL.md` and `../../llm-judge-parsing/SKILL.md`.

Placeholders: `<repo>` = the MedVision checkout used as the working directory (`benchmark_dir` in the launchers),
`<data_dir>` = the dataset directory (conventionally `<repo>/Data`), `<task_tag>` = one of `MedVision-detect-CoT`,
`MedVision-TL-CoT`, `MedVision-AD-CoT`.

**OOD splits** use the same entry points with a different task list and results dir. The four shipped lists are
`tasks_list/OOD/tasks_MedVision-{TL,detect}-CoT-{plane,task}OOD.json`, driven by
`script/ablation/OOD/eval__MedVision-V0-7B__{TL,detect}__{plane,task}OOD.sh`, writing to
`Results/MedVision-{TL,detect}-CoT-{plane,task}OOD/` (the shipped `task_tag`s; the historical published trees carry an extra `-v2`). There is no A/D OOD split. The bundled
`scripts/make_eval_launcher.py` only knows the three in-distribution tags, so copy an OOD launcher rather than
generating one.

---

## 0. Decide the run identity before typing anything

Five values define the experiment and must stay constant across every model you intend to compare:

| Value | Where it goes | Benchmark convention |
|---|---|---|
| task family | `--tasks_list_json_path`, `task_tag` | Detection / T/L / A/D, `-CoT` task lists |
| annotation version | `MedVision_PLANNER_VERSION` (+ `MedVision_ACK_RELEASE` for T/L) | `1.0.0` for leaderboard-comparable numbers |
| samples per subtask | `--sample_limit` | 1000 open-weight, 100 API pilot |
| output budget | `--max_new_tokens` / `--max_tokens` | 4096 local, 16000 API, with documented per-model overrides |
| input reshape | `--reshape_image_hw` | unset for most open-weight models; `512x512` for the API models and MedVision-V0 |

`task_tag` is the only knob that keeps runs apart: it names both `Results/<task_tag>/` and
`completed_tasks/completed_tasks_<task_tag>.json`. Give an exploratory or reduced-limit run its own tag.

---

## 1. First run of an open-weight vLLM model (requires GPU)

```bash
conda activate eval-qwen25vl                      # one env per model; see ../../environment-setup/SKILL.md
cd <repo>

export MedVision_PLANNER_VERSION='1.0.0'
export CUDA_VISIBLE_DEVICES=0,1,2,3                # -> tensor_parallel_size=4, batch_size = 10*4

# Method 1 (the load-bearing order; frozen pins last, then run with --skip_env_setup)
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <repo>/Data
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
pip install -r <repo>/requirements/requirements_eval_qwen25vl.txt --no-deps

python -m medvision_bm.benchmark.eval__qwen2_5_vl \
    --skip_env_setup \
    --model_hf_id Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name Qwen2.5-VL-7B-Instruct \
    --results_dir <repo>/Results/MedVision-detect-CoT \
    --data_dir <repo>/Data \
    --tasks_list_json_path <repo>/tasks_list/tasks_MedVision-detect-CoT.json \
    --task_status_json_path <repo>/completed_tasks/completed_tasks_MedVision-detect-CoT.json \
    --batch_size_per_gpu 10 --gpu_memory_utilization 0.9 \
    --max_new_tokens 4096 --sample_limit 1000
```

What happens per task in the list: the driver checks the tracker, builds a `model_args` string
(`model_hf=…,gpu_memory_utilization=…,tensor_parallel_size=…,max_num_seqs=…,max_new_tokens=…,dtype=…`), runs
`python3 -m lmms_eval` as a subprocess, and marks the task complete only on return code 0.

Generate the equivalent script instead of typing it:

```bash
python scripts/make_eval_launcher.py --model qwen25vl --task detect \
    --benchmark-dir <repo> --cuda-visible-devices 0,1,2,3 --out run_qwen_detect.sh
```

## 2. First run of an HF data-parallel model (requires GPU)

Same shape, but the driver wraps `lmms_eval` in `python -m accelerate.commands.launch --num_processes=<GPUs>`, there is
no `--gpu_memory_utilization`, and each process holds a **full model replica** — so `batch_size_per_gpu` must be much
smaller (the repository uses 2 for MedGemma-27B, MedDr, Lingshu and HuatuoGPT-Vision).

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <repo>/Data
python -m medvision_bm.benchmark.install_vendored_lmms_eval          # medgemma has no opt-deps extra
pip install -r <repo>/requirements/requirements_eval_medgemma.txt --no-deps

export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.4.0'                 # T/L only
python -m medvision_bm.benchmark.eval__medgemma \
    --skip_env_setup --model_hf_id google/medgemma-27b-it --model_name MedGemma-27b-it \
    --results_dir <repo>/Results/MedVision-TL-CoT --data_dir <repo>/Data \
    --tasks_list_json_path <repo>/tasks_list/tasks_MedVision-TL-CoT.json \
    --task_status_json_path <repo>/completed_tasks/completed_tasks_MedVision-TL-CoT.json \
    --batch_size_per_gpu 2 --max_new_tokens 16000 --sample_limit 1000
```

Models that need a third-party checkout (`meddr`, `llava_med`, `huatuogpt_vision`, `healthgpt`) additionally take
`--dir_third_party <repo>/third_party`; the HealthGPT launcher also exports
`PYTHONPATH="<repo>/third_party/HealthGPT:${PYTHONPATH:-}"` so the `accelerate` subprocesses can import it, and selects
the architecture with `--model_choice {HealthGPT-L14,HealthGPT-XL32}` rather than `--model_hf_id`.

## 3. First run of an API model (pilot; no GPU, costs money)

```bash
export ANTHROPIC_API_KEY="$(printf '%s' "$ANTHROPIC_API_KEY" | tr -d '[:space:]')"   # newline -> HTTP 401
export MedVision_PLANNER_VERSION='1.0.0'

python -m medvision_bm.benchmark.install_medvision_ds --data_dir <repo>/Data
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps claude
pip install -r <repo>/requirements/requirements_eval_claude.txt --no-deps

python -m medvision_bm.benchmark.eval__claude \
    --skip_env_setup \
    --api_provider anthropic --anthropic_model_code claude-fable-5 \
    --model_name Claude-Fable-5 --max_tokens 16000 \
    --results_dir <repo>/Results/MedVision-AD-CoT --data_dir <repo>/Data \
    --tasks_list_json_path <repo>/tasks_list/tasks_MedVision-AD-CoT.json \
    --task_status_json_path <repo>/completed_tasks/completed_tasks_MedVision-AD-CoT.json \
    --batch_size 1 --sample_limit 100 --reshape_image_hw 512x512
```

API rules that matter:

- **`--sample_limit 100`** is the published pilot size for every proprietary model; going higher multiplies cost.
- **`--reshape_image_hw 512x512`** is applied by every API launcher, so the slice sent is a known size before the
  client-side fixed-point resize runs (see `image-processing-and-token-budgets.md`).
- **`--batch_size 1`** — the wrappers issue one request per sample with exponential backoff (up to 10 tries, giving up
  immediately on HTTP 400 because a 400 is deterministic).
- **Provider switch.** Each API entry point takes `--api_provider` and a provider-specific model code:
  `eval__claude` (`anthropic`|`openrouter`, `ANTHROPIC_API_KEY`|`OPENROUTER_API_KEY`), `eval__openai`
  (`openai`|`openrouter`), `eval__gemini` (`google`|`openrouter`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`), `eval__kimi`
  (`moonshot`|`openrouter`, `MOONSHOT_API_KEY`, base URL overridable with `MOONSHOT_BASE_URL`). Under OpenRouter the
  code carries a vendor prefix (`anthropic/claude-fable-5`, `openai/gpt-5.5`, `moonshotai/kimi-k2.6`).
- **OpenRouter reserves the full `max_tokens` as credit for the duration of each request.** With `--max_tokens 16000`
  and a thin balance the account can hit HTTP 402 long before the tokens are actually spent; either top up, lower
  `--max_tokens`, or use the direct provider.
- **Thinking/reasoning parameters are model-specific**: `--thinking/--no-thinking` (Claude adaptive thinking; the
  disabled form omits the parameter entirely because an explicit `"disabled"` is rejected with 400),
  `--reasoning_effort` (OpenAI), `--thinking_level` / `--thinkingBudget` / `--media_resolution` (Gemini 3 vs 2.5).

## 4. Resume an interrupted run

Re-run **exactly the same command**. Two independent mechanisms make that cheap:

1. `completed_tasks_<task_tag>.json` makes the driver print `Task <name> already completed. Skipping...` for every
   task that previously returned 0.
2. Inside the task that was interrupted, the per-sample response cache
   (`Results/<task_tag>/<model_name>/response_cache/<task>_rank<N>.jsonl`) returns every already-generated response, so
   only the in-flight sample is lost. All rank shards for the task are loaded, so the resume also works if the world
   size changed (e.g. you now expose 2 GPUs instead of 4).

Inspect the state first:

```bash
python scripts/check_results_tree.py --results-dir <repo>/Results/MedVision-detect-CoT \
    --repo-root <repo> --model <model_name> --show-tasks
```

Rows reading `NO OUTPUT (cache has N responses -> interrupted mid-task)` are exactly the tasks the resume will finish.

**Cache invalidation is automatic on a prompt change**: the key hashes the rendered prompt, so editing a prompt, a task
YAML, `reshape_image_hw` or anything else that changes the text produces a miss and regenerates. It is *not* keyed on
the token budget or sampling parameters — after changing `--max_new_tokens` (or fixing a wrapper's default) delete the
task's cache shards before re-running, or the old truncated responses come back.

## 5. Force a clean re-run

```bash
# one task, one model: drop the tracker entry and the cache shards for that task
python - <<'PY'
import json, pathlib
p = pathlib.Path("<repo>/completed_tasks/completed_tasks_MedVision-TL-CoT.json")
d = json.loads(p.read_text()); d["<model_name>"].pop("<task_name>", None)
p.write_text(json.dumps(d, indent=4))
PY
rm <repo>/Results/MedVision-TL-CoT/<model_name>/response_cache/<task_name>_rank*.jsonl

# whole run, no caching at all
MEDVISION_RESP_CACHE=0 python -m medvision_bm.benchmark.eval__<model> ... --skip_update_status
```

`--skip_update_status` runs the tasks but never writes the tracker, which is the right flag for debugging sweeps.
Old `<timestamp>_samples_<task>.jsonl` files are **not** deleted — a re-run adds a second file for the same task, which
`check_results_tree.py` reports as `duplicates=N` and which the parser will see twice. Move or delete the stale
timestamped pair (`*_samples_<task>.jsonl` and its `*_results.json`) before re-running if you want a single copy.

## 6. Smoke test the wiring without paying for a full run

```bash
# environment only (installs everything, then exits before inference)
python -m medvision_bm.benchmark.eval__<model> ... --env_setup_only

# two samples of every task in the list, into a throwaway task_tag
python -m medvision_bm.benchmark.eval__<model> ... \
    --results_dir <repo>/Results/smoke-detect \
    --task_status_json_path <repo>/completed_tasks/completed_tasks_smoke-detect.json \
    --sample_indices "[0:2]" --skip_update_status
```

`--sample_indices` accepts `[start:stop]` or `[start,stop,step]` and overrides `--sample_limit` for sample selection.
On a CPU-only host the furthest you can get is `--help` and `--env_setup_only`; anything that loads a model needs a GPU.

## 7. Sweep the three task families

The task family is a *pair* of values — the task list and the tag — so a sweep is three invocations of the same
command with three substitutions:

```bash
for t in detect TL AD; do
  tag="MedVision-${t}-CoT"
  extra=""; [ "$t" = TL ] && export MedVision_ACK_RELEASE='1.4.0'
  python -m medvision_bm.benchmark.eval__<model> --skip_env_setup \
      --model_hf_id <hf-id> --model_name <label> \
      --results_dir <repo>/Results/${tag} --data_dir <repo>/Data \
      --tasks_list_json_path <repo>/tasks_list/tasks_${tag}.json \
      --task_status_json_path <repo>/completed_tasks/completed_tasks_${tag}.json \
      --batch_size_per_gpu <n> --max_new_tokens <budget> --sample_limit 1000
done
```

Check the per-task budget overrides in `model-catalog.md` before reusing one budget for all three families — several
models were run at 16000/16384 on some families and 4096 on others.

## 8. Evaluate your own SFT / RFT checkpoint

- `--model_hf_id` accepts a **local checkpoint directory** as well as a Hub id; `--model_name` is the free-form label
  that becomes the `Results/<task_tag>/<model_name>/` folder.
- LoRA adapters: `--lora_path <path>` on all 11 entry points that expose it — every vLLM driver plus `eval__qwen25vl_tooluse` (`eval__qwen2_5_vl`, `eval__qwen3_vl`, `eval__gemma3`, `eval__intern_vl3`, `eval__llama3_2_vision`, `eval__llava_onevision`, `eval__qwen25vl_tooluse`,
  `eval__gemma4`, `eval__glm4v`, `eval__minimax_m3`, `eval__medvision-model-rft`).
- RFT checkpoints trained with the verl system prompt: use `eval__medvision-model-rft` with `--use_system_prompt` (the
  MedVision-V0 launchers pair it with `--reshape_image_hw 512x512`, matching the 512x512 training resolution).
- The `lmms_eval` key still has to be one `get_resized_img_shape()` understands. Pick the right family with
  `--lmmseval_module` (`--lmms_eval_module` on `eval__medvision-model-rft`) — e.g. `vllm_qwen25vl_tooluse` for a
  tool-use SFT model — or register a new key via `../../extending-models-and-tasks/SKILL.md`.

## 9. Hand off to step 2

A finished run leaves, per model:

```
Results/<task_tag>/<model_name>/
  <YYYYMMDD_HHMMSS>_samples_<task>.jsonl     # one JSON object per sample (doc, prompt, resps, filtered_resps, metrics)
  <YYYYMMDD_HHMMSS>_results.json             # lmms_eval aggregate for that invocation
  response_cache/<task>_rank<N>.jsonl        # resume cache (safe to keep; delete only to force regeneration)
```

Verify completeness, then parse and summarize:

```bash
python scripts/check_results_tree.py --results-dir <repo>/Results/MedVision-TL-CoT --repo-root <repo> --strict
python -m medvision_bm.benchmark.parse_outputs --task_type TL --task_dir Results/MedVision-TL-CoT -p 8
```

Details of those two steps: `../../results-parsing-and-metrics/SKILL.md`.
