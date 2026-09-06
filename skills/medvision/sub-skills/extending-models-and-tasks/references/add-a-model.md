# Adding a VLM to the MedVision benchmark

Applies to all three backends: **vLLM** (open weights, tensor parallel), **HF/transformers**
(open weights, data parallel via `accelerate launch`) and **API** (hosted model, no GPU).
API-specific rules are in `api-model-rules.md`; the perceived-size dispatch is in
`image-size-dispatch.md`.

One naming rule ties everything together: the **model key** is a lowercase snake_case string
used identically in four places — the `@register_model("<key>")` decorator, the module file name
`lmms_eval/models/<key>.py`, the `AVAILABLE_MODELS` dict key, and the `--model <key>` value the
eval entry point passes. `get_model()` resolves the registry value as
`lmms_eval.models.<key>.<ClassName>`, so a mismatch is an import error, not a warning.

## Checklist

"Required" = the benchmark cannot run the model without it. "Operational" = the run works but
is not reproducible or reviewable without it; the repository has all of them for every shipped
model.

| # | File | What to add | Why | How to verify |
| --- | --- | --- | --- | --- |
| 1 | `lmms_eval/models/<key>.py` | Model class decorated with `@register_model("<key>")`, implementing `generate_until(requests)`; `loglikelihood` / `generate_until_multi_round` may raise `NotImplementedError`. Constructor takes `model_hf` (or `model` for API), backend knobs, and an **explicit `max_new_tokens` / `max_tokens` default**. | Required. This is the inference wrapper lmms_eval instantiates from `--model_args`. | `python scripts/list_registered_models.py --expect <key>` reports the decorator and class; import needs the backend installed. |
| 2 | `lmms_eval/models/__init__.py` | `"<key>": "<ClassName>"` inside `AVAILABLE_MODELS`. | Required. `get_model()` raises `Model <key> not found in available models.` otherwise. | Same command; also `get_available_model_names()`. |
| 3 | `lmms_eval/tasks/medvision/medvision_utils.py` | A branch in `get_resized_img_shape()` for `<key>` (and for the SFT `model_family_name` alias, if any), plus a `_process_img_<key>()` probe when the model resizes dynamically. | Required for T/L and A/D. Without it every prompt build raises `ValueError: [Error] <key> is not recognised/supported.` | `scripts/list_registered_models.py` shows `branch=yes` and the strategy letter; then a 2-sample T/L run. |
| 4 | `medvision_lmms_eval/pyproject.toml` | `<key> = [ ... ]` under `[project.optional-dependencies]` — the runtime SDK plus every pin the stack needs (**always pin `transformers`**). | Required in practice: `install_vendored_lmms_eval --lmms_eval_opt_deps <key>` installs exactly this extra. | `python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps <key>` in a scratch env. |
| 5 | `src/medvision_bm/benchmark/eval__<family>.py` — the module is named after the model **family**, not the registry key (key `vllm_qwen25vl` -> `eval__qwen2_5_vl.py`); only 10 of the 20 keys coincide | Entry point: argparse, the fixed env-setup call order, the `load_tasks` loop with `load_tasks_status` / `update_task_status`, and the `--model_args` string containing `model_hf=<id>`. | Required. This is what launchers and users invoke; it is also where `model_hf` enters the pipeline. | `python -m medvision_bm.benchmark.eval__<key> --help` (needs `medvision_bm` installed, no GPU). |
| 6 | `script/benchmark-detect/`, `-TL/`, `-AD/` — one `eval__<Display>__<fam>.sh` each | Conda env, wheel build/install, `MedVision_PLANNER_VERSION` (+ `MedVision_ACK_RELEASE` on T/L), the token budget variable, the three-line "Method 1" install block, and the `python -m ... eval__<key>` call. | Operational. Encodes the load-bearing install order and the exact per-model resources. | `bash -n <script>`; then run with a tiny `sample_limit` on a GPU host. |
| 7 | `requirements/requirements_eval_<key>.txt` | A full `pip freeze` of the working environment. | Operational. Launchers install it with `--no-deps`, so every transitive dependency must be present and pinned. | `pip install -r ... --no-deps` into a fresh env, then import the wrapper. |
| 8 | `unit-test/<key>-image-resize/test_<key>_resize.py` (API models only — claude/gemini/kimi/openai); local models are covered by `unit-test/perceived-size-resize/test_perceived_size_resize.py` | Assertions on the resize/caps rule (API) or on `get_resized_img_shape` against the real processor (local). | Operational, and the only cheap guard against silent measurement corruption. | Run the file directly; the repository's tests are plain `assert`s runnable with or without pytest. |
| 9 | `README.md` -> "Benchmarked Models" | A row/link in the right group (ours / general-purpose open-weight / medical open-weight / proprietary API). | Operational. The README table is the human-facing roster. | Visual review. |
| 10 | `dockerfile/Dockerfile.eval_<key>` | A conda env layer mirroring the launcher, built `FROM` the repository base image. | Optional. Only needed if the model should ship a container. | `docker build` (requires Docker; never push from an agent session). |
| 11 | `lmms_eval/tasks/medvision/lmms_eval_specific_kwargs.yaml` | A block keyed by the model key — **only** when the wrapper needs parameters beyond `model_name` / `model_hf`. | Optional. Almost never needed; see below. | Inspect the merged kwargs in a logged sample. |

### What you do *not* have to touch

**No per-model edit to any task YAML.** `model_name` and `model_hf` are injected at run time by
the evaluator, before any task builds a prompt:

```python
# lmms_eval/evaluator.py
_parsed_model_args   = simple_parse_args_string(cli_args.model_args)
_model_arg_model_hf  = _parsed_model_args.get("model_hf", None)     # --model_args model_hf=...
_model_name          = cli_args.model                                # --model
...
task.lmms_eval_specific_kwargs["model_hf"]   = _model_arg_model_hf
task.lmms_eval_specific_kwargs["model_name"] = _model_name
task.lmms_eval_specific_kwargs["reshape_image_hw"] = _reshape_image_hw   # when provided
```

`model_hf` is additionally parsed *early* (before the `TaskManager` is built) so task
construction can already see it. The chain is:

```
eval__<key>.sh   --model_hf_id <HF_ID>  --model_name <Display>
      |
      v
eval__<key>.py   --model <key>  --model_args "model_hf=<HF_ID>,..."
      |
      v
evaluator.py     injects model_name / model_hf into lmms_eval_specific_kwargs
      |
      v
get_resized_img_shape() / _process_img_*()
```

The one exception is site 11. `lmms_eval_specific_kwargs.yaml` (included by every base task YAML)
carries a per-model block only when the wrapper needs *extra* construction parameters. The single
shipped example is HealthGPT, which needs its base/vision checkpoints and HLoRA settings:

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

The file's own header warns: do not rename the model-field keys (they are read by
`medvision_utils.py`) and do not remove the empty `dataset:` field.

## The output-token budget rule

Every wrapper must set an **explicit** output-token budget. Budgets resolve in this order:

1. task YAML `generation_kwargs.max_new_tokens` — no MedVision task YAML sets it, so this channel
   is inert today, but wrappers should still honour it;
2. `model_args` `max_new_tokens` (or `max_tokens` for API models), injected by the launcher — this
   is the channel that decides in practice;
3. the wrapper's constructor default.

If a wrapper sets none of these, a third-party library default takes over. That is exactly what
happened once: an upstream chat class hard-coded `max_new_tokens=512` and every run before the
fix silently generated under a 512-token cap. Repository defaults today: **4096** for local
models, **16000** for API models (thinking/reasoning tokens share that budget), with a few
launchers overriding (16384 for one large reasoning model; 4096 for the OpenAI launchers).
Verbose CoT models can exhaust 4096 — check the share of responses ending exactly at the cap
before trusting a success rate.

## Wrapper anatomy (local, vLLM)

The shipped vLLM wrappers share one skeleton:

- `__init__(model_hf, lora_path=None, tensor_parallel_size=1, gpu_memory_utilization=0.8,
  batch_size=1, max_frame_num=32, max_new_tokens=4096, threads=16, trust_remote_code=True,
  chat_template=None, stop_strings=None, system_prompt=None, **kwargs)`.
- `**kwargs` values that look like JSON objects (`hf_overrides={...}`) are `json.loads`-ed, so
  model-specific vLLM arguments need no new parameter.
- `kwargs.pop("reshape_image_hw", None)` — MedVision-only kwargs must not reach the vLLM
  constructor.
- Optional LoRA: read `r` from `adapter_config.json`, set `enable_lora` / `max_lora_rank`, and
  pass a `LoRARequest` at chat time.
- `generate_until` resolves the per-sample response cache first (greedy decoding only; sampled
  decoding is never cached because identical arguments would collide on one key), batches the
  remainder by `batch_size_per_gpu`, sets `max_new_tokens` / `temperature=0` / `top_p=0.95`
  defaults, encodes visuals to base64 PNG in a thread pool, and calls `client.chat(...)`.
- **`until` from the task config is deliberately not forwarded as a decoding stop.** lmms-eval
  defaults it to the few-shot delimiter `"\n\n"`, which would truncate multi-paragraph CoT after
  the first blank line. String stops are applied only when `--stop_strings` is passed explicitly.
- Tensor parallelism comes from `tensor_parallel_size`, which the eval entry point sets to the
  number of visible GPUs (`set_cuda_num_processes()`), not from anything in the wrapper.

HF wrappers instead expose `rank` / `world_size` / `batch_size` properties from an
`accelerate.Accelerator`, are launched with `python -m accelerate.commands.launch
--num_processes=<GPUs> -m lmms_eval ...`, and read the per-request budget as
`gen_kwargs.get("max_new_tokens", self.max_new_tokens)`.

## Entry-point anatomy (`eval__<key>.py`)

Shared flags across all entry points: `--model_name` (required; names the results sub-directory
and the completed-tasks key), `--tasks_list_json_path`, `--results_dir`, `--task_status_json_path`,
`--data_dir`, `--sample_limit`, `--sample_indices`, `--log-sys-prompt`,
`--reshape_image_hw`, `--skip_env_setup`, `--skip_update_status`, `--env_setup_only`,
`--scaled_ps_low` / `--scaled_ps_high` (for `-scaledPS` task variants). Local entry points add
`--model_hf_id`, `--batch_size_per_gpu`, `--max_new_tokens` and (vLLM) `--gpu_memory_utilization`;
API entry points add `--api_provider`, a provider model-code flag and `--max_tokens`.

The env-setup block is explicitly order-sensitive and must be copied as-is:

```python
# NOTE: DO NOT change the order of these calls
setup_env_hf_medvision_ds(data_dir)
if not args.skip_env_setup:
    ensure_hf_hub_installed(hf_hub_version="0.36.0")   # version varies per model stack
    install_vendored_lmms_eval(proj_dependency="<key>")
    install_medvision_ds(data_dir)
    # local models only: install_torch_cu124(); install_vllm(data_dir, version="...")
    # then re-install the transformers/accelerate pins LAST so they win resolution
    if args.env_setup_only:
        return
else:
    setup_env_vllm(data_dir)      # vLLM entry points only
```

API entry points additionally fail fast when no provider key is set, using a
`{provider: [env var names]}` table, and pass `model_hf=<model code>` so the task layer can look
up the provider caps.

## Launcher anatomy

Each launcher is one file per model per task family (`detect`, `TL`, `AD`) and contains:

1. conda env creation/activation (`eval-<key>`, python 3.11);
2. `benchmark_dir` / `data_dir` / `model_hf_id` / `model_name` and the per-model resource knobs;
3. `task_tag`, `result_dir`, `tasks_list_json_path`, `task_status_json_path`, `sample_limit`
   (1000 for open weights, 100 for the API pilot);
4. the wheel-build-on-local-disk install block (a shared network filesystem can make setuptools'
   cached build directory vanish mid-build; a `flock` guards the shared-env install);
5. `export MedVision_PLANNER_VERSION=...` (T/L launchers also `export MedVision_ACK_RELEASE=...`);
6. the explicit token budget variable;
7. the **"Method 1"** block — three load-bearing lines in this order:
   `install_medvision_ds` -> `install_vendored_lmms_eval --lmms_eval_opt_deps <key>` ->
   `pip install -r requirements/requirements_eval_<key>.txt --no-deps` — followed by the
   `python -m medvision_bm.benchmark.eval__<key> --skip_env_setup ...` call;
8. a commented-out **"Method 2"** block that runs the same entry point *without* `--skip_env_setup`.

> **Edit both blocks.** Method 2 is commented out but is the documented fallback, and the
> repository keeps the two in sync. A flag added to Method 1 only will silently disappear the
> moment someone switches. API launchers additionally sanitise the key with
> `export KEY="$(printf '%s' "${!KEY}" | tr -d '\n')"`.

`../../benchmark-evaluation/scripts/make_eval_launcher.py` generates these for models already in
the catalog; `scripts/scaffold_new_model.py` writes the three launcher stubs for a new one.

## Suggested order of work

1. `python scripts/scaffold_new_model.py --key <key> --class-name <Class> --kind vllm|hf|api
   --hf-id <id> --out-dir <scratch>` — produces every file above as a TODO-marked skeleton.
2. Fill in the wrapper, then copy the two `patches/` snippets into `models/__init__.py` and
   `medvision_utils.py` by hand (they are fragments, not appliable diffs).
3. `python scripts/list_registered_models.py --expect <key>` until it exits 0.
4. Add the `pyproject.toml` extra, build the environment, freeze it into
   `requirements/requirements_eval_<key>.txt`.
5. Run the resize test, then **two samples of one T/L task** and read the logged prompt: the
   stated image size and pixel size must match the model's real perceived canvas. Use a
   non-square slice — square inputs hide per-axis errors.
6. Only then run a full task family; add the README row and (optionally) the Dockerfile.

Cross-links: environment mechanics and pins -> `../../environment-setup/SKILL.md`; running and
resuming evaluations -> `../../benchmark-evaluation/SKILL.md`; the existing roster ->
`../../../references/model-roster.md`.
