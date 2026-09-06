# Launcher Anatomy

The repository ships one shell launcher per model per task family in `script/benchmark-detect/`,
`script/benchmark-TL/` and `script/benchmark-AD/` (24 stems each, named `eval__<Display>__{detect,TL,AD}.sh`). They are
all the same seven-block skeleton with per-model substitutions. `scripts/make_eval_launcher.py` reproduces that
skeleton from `scripts/model_catalog.json`; read this page when you need to write, review or repair one by hand.

```
1. conda env block            5. planner-version exports (+ token budget)
2. paths and configs          6. install method + eval command
3. API provider + key block   7. conda deactivate
4. medvision_bm install block
```

---

## Block 1 — conda environment

```bash
ENV_NAME="eval-qwen25vl"
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"
```

One env per model, named `eval-<model>`; Python 3.11 everywhere except four launchers — MedDr 3.9, LLaVA-Med 3.10,
MiniMax-M3 (MXFP8) 3.12 and MiniMax-M3-INT4 3.12, the last of which uses 3.12 to
match its patched vLLM fork's precompiled wheels. The trailing `conda remove -n $ENV_NAME --all -y` at the bottom of
every launcher is commented out on purpose — envs are expensive to rebuild.

## Block 2 — paths and configs (the part you edit)

| Variable | Meaning | Notes |
|---|---|---|
| `benchmark_dir` | working directory = the checkout | supplies `tasks_list/`, `requirements/`, `Results/`, `completed_tasks/` |
| `data_dir` | `${benchmark_dir}/Data` | passed as `--data_dir`; also where `medvision_ds` and the HF/vLLM caches live |
| `model_hf_id` | Hub id **or local checkpoint dir** | absent for API models and HealthGPT |
| `model_name` | free-form run label | becomes `Results/<task_tag>/<model_name>/` and the key in the task-status JSON |
| `batch_size_per_gpu` | per-GPU batch | multiplied by the visible-GPU count before reaching `lmms_eval` |
| `gpu_memory_utilization` | vLLM VRAM fraction | vLLM models only; 0.9-0.95 in the launchers |
| `task_tag` | `MedVision-{detect,TL,AD}-CoT` | the only thing separating one run from another |
| `result_dir` | `${benchmark_dir}/Results/${task_tag}` | passed as `--results_dir` |
| `tasks_list_json_path` | `${benchmark_dir}/tasks_list/tasks_${task_tag}.json` | top-level keys are the task names |
| `task_status_json_path` | `${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json` | per-model completion tracker |
| `sample_limit` | `1000` open-weight, `100` API pilot | `--limit` for `lmms_eval` |
| `max_new_tokens` / `max_tokens` | output budget | 4096 local / 16000 API by default, with per-model overrides |
| `reshape_image_hw` | e.g. `512x512` | API launchers and MedVision-V0; reshapes the slice at NIfTI load |
| `dir_third_party` | `${benchmark_dir}/third_party` | MedDr, LLaVA-Med, HuatuoGPT-Vision, HealthGPT |

Model-specific extras appear here too: sampling parameters (`temperature`/`top_p`/`top_k`, and `repetition_penalty` for
GLM-4.6V), `stop_string='</answer>'`, `max_model_len` (Gemma-4), `cpu_offload_gb` and `vllm_version` (MiniMax-M3),
`model_choice` (HealthGPT).

## Block 3 — API provider and key sanitising (API launchers only)

```bash
api_provider="anthropic"
anthropic_model_code="claude-fable-5"

if [ "${api_provider}" = "anthropic" ]; then
    api_key_var="ANTHROPIC_API_KEY"
else
    api_key_var="OPENROUTER_API_KEY"
fi
if [ -z "${!api_key_var:-}" ]; then
    echo "[Error] ${api_key_var} is not set." >&2
    exit 1
fi
export "${api_key_var}"="$(printf '%s' "${!api_key_var}" | tr -d '\n')"
```

The strip is load-bearing: secrets injected as container/Kubernetes environment variables usually carry a trailing
newline, which is illegal in an HTTP auth header and surfaces as a 401 with an otherwise valid key. Switching to
OpenRouter means changing three things together — `api_provider`, the model code (vendor-prefixed:
`anthropic/claude-fable-5`, `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`, `moonshotai/kimi-k2.6`) and the key
variable. Note that Gemini's `--media_resolution` and its code-execution / JSON-output options are `google`-only.

## Block 4 — the node-local wheel build (CephFS race)

```bash
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"
```

Why: setuptools' `build_py` memoises the directories it created in a process-global cache. On a shared network
filesystem (CephFS and similar) a build subdirectory can transiently disappear; the memo then refuses to recreate it
and a later file copy dies with `could not create '...': No such file or directory`. Building in a private node-local
temp directory is immune. Only the install into the shared env needs the `flock`, because several launchers may start
at once. This block is optional — it exists to refresh `medvision_bm` from the checkout. `make_eval_launcher.py`
reproduces it with `--install-mode wheel` (default) and offers `editable` (`pip install -e`) and `skip`.
`../../environment-setup/SKILL.md` bundles the same recipe as a standalone script.

## Block 5 — planner version and token budget

```bash
export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.4.0'     # T/L launchers only
max_new_tokens=4096                      # or max_tokens=16000 for API launchers
```

`MedVision_PLANNER_VERSION` is **required** — the dataset loader hard-fails without it. Pinning below the newest
annotation release additionally needs `MedVision_ACK_RELEASE` set to that newest release; only the T/L family's
annotations changed across `1.0.0` … `1.4.0`, which is why every T/L launcher carries the second export and no
Detection or A/D launcher does. HealthGPT's launcher also exports
`PYTHONPATH="${dir_third_party}/HealthGPT:${PYTHONPATH:-}"` here so the `accelerate` subprocesses can import the
third-party package.

## Block 6 — install method and the eval command

Every launcher contains both methods; one is live and the other is commented out.

**Method 1 (default, "more robust")** — install by hand, then run with `--skip_env_setup`:

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
pip install -r "${benchmark_dir}/requirements/requirements_eval_qwen25vl.txt" --no-deps

python -m medvision_bm.benchmark.eval__qwen2_5_vl --skip_env_setup <flags…>
```

These three lines are **load-bearing and ordered**: the vendored engine and the dataset package first (both re-pin
`huggingface_hub`), then the frozen requirements with `--no-deps` **last** so the pins win and nothing is re-resolved.
The `--lmms_eval_opt_deps` value is the model's extra in the vendored engine's `pyproject.toml`; models with no extra
call `install_vendored_lmms_eval` bare. The GPT launchers currently stop after the extra with a `TODO` instead of
pinning (a `requirements_eval_gpt.txt` does exist in the repository and can be used the same way).

**Method 2** — let the entry point install its own dependencies. Drop `--skip_env_setup` and the driver runs its
built-in order before the first task: `setup_env_hf_medvision_ds` → `ensure_hf_hub_installed` →
`install_vendored_lmms_eval` → `install_medvision_ds` → `install_torch_cu124` → `install_vllm` → a model-specific
transformers/accelerate reinstall. Simpler, but it resolves versions at run time and can pick up an incompatible new
release. The launchers that ship with Method 2 live are Qwen3-VL, Gemma-4, HuatuoGPT-Vision, LLaVA-Med and HealthGPT.

Three debugging flags are documented in the commented block of every launcher:

| Flag | Effect |
|---|---|
| `--env_setup_only` | run the built-in install order, print a notice, exit before any inference |
| `--skip_env_setup` | touch nothing in the environment (what Method 1 relies on); the vLLM entry points still call `setup_env_vllm` so `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `XDG_CACHE_HOME=<data_dir>/.cache/vllm` are set |
| `--skip_update_status` | run the tasks but never write `completed_tasks_<task_tag>.json` |

**Two-pass variant (MiniMax-M3).** When pip cannot supply a usable vLLM, the launcher defines the flags once in a bash
array and calls the entry point twice:

```bash
common_args=( --lmmseval_module vllm_minimax_m3 --model_hf_id "$model_hf_id" … --stop_strings "$stop_string" )
python -m medvision_bm.benchmark.eval__minimax_m3 "${common_args[@]}" --env_setup_only   # step 1: standard setup
#   step 2: replace vLLM (patched fork), realign NCCL/CUDA, set CUDA_HOME/LD_LIBRARY_PATH …
python -m medvision_bm.benchmark.eval__minimax_m3 "${common_args[@]}" --skip_env_setup   # step 3: run
```

Use the array so the two invocations cannot drift. `make_eval_launcher.py --method two-pass` emits this shape with an
empty step 2 for you to fill in.

## Block 7 — teardown

```bash
conda deactivate
# conda remove -n $ENV_NAME --all -y
```

---

## Per-launcher differences worth knowing

- **`sample_limit`**: 1000 for every open-weight launcher, 100 for all five API launchers.
- **`gpu_memory_utilization`**: absent for the HF data-parallel models; 0.9 (Qwen2.5-VL, Gemma-3, InternVL3,
  LLaVA-OneVision, Llama-3.2, MedVision-V0), 0.90 (MiniMax-M3, deliberately not 0.95 — a dead vLLM engine leaks a few
  GiB, and 0.95 then fails the startup memory check in a restart loop), 0.95 (Qwen3-VL, Gemma-4, GLM-4.6V).
- **`batch_size_per_gpu`**: 1-20, chosen per model; the vLLM value becomes `max_num_seqs` (concurrency), the HF value
  multiplies a *full replica* per GPU.
- **Sampling**: greedy by default. Qwen3-VL (`0.8/0.95/20`), GLM-4.6V (`0.8/0.6/2` plus a repetition penalty) and
  MiniMax-M3 (`1.0/0.95/40`) are reasoning models and set explicit `temperature/top_p/top_k` mirroring their model
  cards. Generation still uses a fixed internal seed.
- **`--stop_strings '</answer>'`**: required for the reasoning models. Nothing stops them otherwise — the vendored
  engine injects no stop sequence and relies on the model's EOS token (`api/task.py:142-149`), so without it a
  reasoning model runs to `--max_new_tokens`. An explicit stop string supplies a clean terminator.
- **`--no-enable_thinking` (Gemma-4)**: with native thinking on, the model ignores the `<think>`/`<answer>` format.
- **`--use_system_prompt` (MedVision-V0)**: injects the verl RFT training system prompt.
- **`--reshape_image_hw 512x512`**: all API launchers and MedVision-V0 (matching its 512x512 training resolution).
