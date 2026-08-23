# Command-line reference

`medvision_bm` ships almost no importable Python API — the top-level package
exposes only `__version__`. Everything you actually run is an `argparse` entry
point invoked as a module:

```bash
python -m medvision_bm.<subpackage>.<module> [flags]
```

This page is the exhaustive flag reference for every user-facing command,
grouped by workflow stage. Defaults shown are the argparse defaults baked into
the source; a blank default means the flag is optional with no default (usually
`None`) or is required.

:::{note}
Several commands read configuration from environment variables
(`MedVision_DATA_DIR`, `MedVision_PLANNER_VERSION`, `MedVision_ACK_RELEASE`,
`HF_TOKEN`, and friends). Those are documented on the
[Installation](../getting-started/installation.md) page, not repeated here.
:::

Conventions used below:

- Boolean flags marked **store_true** are switches — pass the flag with no value.
- Flags typed **str2bool** (SFT only) take an explicit truthy/falsy value, e.g.
  `--process_img true`.
- Where a command accepts `-p`, it is the short alias for `--processes`.

---

## Benchmarking (evaluation entry points)

Every model has its own `eval__<model>` driver under
`medvision_bm.benchmark`, but they share one loop: set up the environment,
read a task list, and for each task shell out to the vendored `lmms_eval`
runner, writing per-sample JSONL into `<results_dir>/<model_name>/`. Completed
tasks are recorded in the status JSON so a re-run resumes where it stopped.

The flag surface splits into two shapes — a **vLLM / local-weights** shape
(represented by `eval__qwen2_5_vl`) and an **API** shape (represented by
`eval__openai` and `eval__claude`). Both are documented in full below; the
remaining drivers reuse one of these shapes plus a few model-specific extras.

### `eval__qwen2_5_vl` (representative vLLM driver)

Run local-weight vision-language models through vLLM.

```bash
python -m medvision_bm.benchmark.eval__qwen2_5_vl \
  --model_name Qwen2.5-VL-7B-Instruct \
  --model_hf_id Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks_list_json_path tasks_list/TL.json \
  --data_dir "$MedVision_DATA_DIR" \
  --results_dir Results/TL \
  --task_status_json_path completed_tasks/TL_qwen25vl.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--lmmseval_module` | `vllm_qwen25vl` | Name of the `lmms_eval` model wrapper to load (e.g. a `*_tooluse` variant for tool-use SFT checkpoints). |
| `--model_hf_id` | `Qwen/Qwen2.5-VL-7B-Instruct` | Hugging Face model ID of the base weights. |
| `--lora_path` | | Hugging Face path to a LoRA adapter; leave unset for full weights. |
| `--model_name` | `Qwen2.5-VL-7B-Instruct` | Display name; becomes the results subfolder and status-file key. |
| `--dtype` | `auto` | Weight dtype passed to vLLM (`auto`, `float16`, `bfloat16`, `float32`). |
| `--reshape_image_hw` | | Force-resize every image to `H,W` before inference; unset keeps native size. |
| `--max_new_tokens` | `4096` | Max tokens generated per sample. |
| `--stop_strings` | | One or more stop sequences (e.g. `--stop_strings '</answer>'`); generation halts at the first match. |
| `--batch_size_per_gpu` | `20` | Per-GPU batch size; multiplied by the detected GPU count for the effective batch. |
| `--gpu_memory_utilization` | `0.99` | vLLM GPU memory fraction. |
| `--tasks_list_json_path` | | Path to the task-list JSON to evaluate. |
| `--results_dir` | | Root directory for per-sample outputs. |
| `--task_status_json_path` | | Path to the resumable completed-tasks JSON. |
| `--data_dir` | | MedVision data directory. |
| `--sample_limit` | `1000` | Max samples evaluated per task. |
| `--sample_indices` | | Evaluate a subset by index; accepts `[start:stop]` or `[start,stop,step]`. Overrides `--sample_limit` when set. |
| `--log-sys-prompt` | off | **store_true** — also record the system prompt in each JSONL row. |
| `--skip_env_setup` | off | **store_true** — skip package/env installation (debug only). |
| `--skip_update_status` | off | **store_true** — do not mark tasks complete (debug only). |
| `--env_setup_only` | off | **store_true** — install the environment and exit without evaluating. |
| `--scaled_ps_low` | `0.5` | Lower bound of the pixel-size scaling range for `-scaledPS` task variants (exported as `MEDVISION_SCALED_PS_LOW`). |
| `--scaled_ps_high` | `3.0` | Upper bound of the pixel-size scaling range for `-scaledPS` task variants (exported as `MEDVISION_SCALED_PS_HIGH`). |

### `eval__openai` (OpenAI / OpenRouter API driver)

Evaluate hosted OpenAI models directly or via OpenRouter.

```bash
export OPENAI_API_KEY=sk-...
python -m medvision_bm.benchmark.eval__openai \
  --openai_model_code gpt-5.5 \
  --model_name gpt-5.5 \
  --tasks_list_json_path tasks_list/Detection.json \
  --data_dir "$MedVision_DATA_DIR" \
  --results_dir Results/Detection \
  --task_status_json_path completed_tasks/Detection_openai.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--openai_model_code` | **required** | Provider model ID (e.g. `gpt-5.5` for `openai`, or `openai/gpt-5.5` for `openrouter`). |
| `--api_provider` | `openai` | Provider: `openai` (direct) or `openrouter`. Selects which API key env var is required (`OPENAI_API_KEY` / `OPENROUTER_API_KEY`). |
| `--model_name` | **required** | Display name; results subfolder and status-file key. |
| `--reasoning_effort` | | Reasoning effort for reasoning models (`low`/`medium`/`high`). Omitted from the request when unset. |
| `--max_tokens` | `16000` | Default max output tokens; a per-task `max_new_tokens` in the task YAML wins. |
| `--reshape_image_hw` | | Force-resize every image to `H,W` before sending. |
| `--batch_size` | `1` | Requests batched together. |
| `--tasks_list_json_path` | **required** | Path to the task-list JSON. |
| `--results_dir` | **required** | Root directory for per-sample outputs. |
| `--task_status_json_path` | **required** | Path to the resumable completed-tasks JSON. |
| `--data_dir` | **required** | MedVision data directory. |
| `--sample_limit` | `1000` | Max samples per task. |
| `--sample_indices` | | Index subset selector (`[start:stop]` or `[start,stop,step]`); overrides `--sample_limit`. |
| `--log-sys-prompt` | off | **store_true** — record the system prompt in each JSONL row. |
| `--stop_strings` | | Stop sequences. Note: OpenAI reasoning models (gpt-5.x / o-series) may reject the `stop` parameter — leave unset for those. |
| `--skip_env_setup` | off | **store_true** — skip environment setup (debug only). |
| `--skip_update_status` | off | **store_true** — do not mark tasks complete (debug only). |
| `--env_setup_only` | off | **store_true** — install the environment and exit. |
| `--scaled_ps_low` | `0.5` | Lower bound of the `-scaledPS` scaling range. |
| `--scaled_ps_high` | `3.0` | Upper bound of the `-scaledPS` scaling range. |

### `eval__claude` (Anthropic / OpenRouter API driver)

Same shape as `eval__openai`, with Claude-specific thinking control instead of
reasoning effort.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m medvision_bm.benchmark.eval__claude \
  --anthropic_model_code claude-fable-5 \
  --model_name claude-fable-5 \
  --tasks_list_json_path tasks_list/AD.json \
  --data_dir "$MedVision_DATA_DIR" \
  --results_dir Results/AD \
  --task_status_json_path completed_tasks/AD_claude.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--anthropic_model_code` | **required** | Provider model ID (e.g. `claude-fable-5` for `anthropic`, or `anthropic/claude-opus-4.8` for `openrouter`). |
| `--api_provider` | `anthropic` | Provider: `anthropic` (direct) or `openrouter`. Selects the required key (`ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY`). |
| `--model_name` | **required** | Display name; results subfolder and status-file key. |
| `--thinking` / `--no-thinking` | on | Toggle adaptive thinking. `--no-thinking` omits the thinking parameter entirely (an explicit "disabled" is rejected by some models). |
| `--max_tokens` | `16000` | Default max output tokens; a per-task `max_new_tokens` wins. |
| `--reshape_image_hw` | | Force-resize every image to `H,W` before sending. |
| `--batch_size` | `1` | Requests batched together. |
| `--tasks_list_json_path` | **required** | Path to the task-list JSON. |
| `--results_dir` | **required** | Root directory for outputs. |
| `--task_status_json_path` | **required** | Path to the completed-tasks JSON. |
| `--data_dir` | **required** | MedVision data directory. |
| `--sample_limit` | `1000` | Max samples per task. |
| `--sample_indices` | | Index subset selector; overrides `--sample_limit`. |
| `--log-sys-prompt` | off | **store_true** — record the system prompt in each JSONL row. |
| `--stop_strings` | | Stop sequences. |
| `--skip_env_setup` | off | **store_true** — skip environment setup (debug only). |
| `--skip_update_status` | off | **store_true** — do not mark tasks complete (debug only). |
| `--env_setup_only` | off | **store_true** — install the environment and exit. |
| `--scaled_ps_low` | `0.5` | Lower bound of the `-scaledPS` scaling range. |
| `--scaled_ps_high` | `3.0` | Upper bound of the `-scaledPS` scaling range. |

### Other evaluation drivers

The following modules follow the same command shape — the vLLM shape for
local-weight models, the API shape for hosted ones — with model-specific extras
(different `--lmmseval_module` default, tailored install pins, or provider
options). Consult `--help` on the specific driver for its exact surface.

vLLM / local-weight drivers:

```text
eval__qwen3_vl        eval__gemma3          eval__gemma4
eval__glm4v           eval__healthgpt       eval__huatuogpt_vision
eval__intern_vl3      eval__kimi            eval__lingshu
eval__llama3_2_vision eval__llava_med       eval__llava_onevision
eval__meddr           eval__medgemma        eval__minimax_m3
eval__medvision-model-rft
```

API drivers:

```text
eval__openai   eval__claude   eval__gemini
```

:::{tip}
`eval__gemini` extends the API shape with Google-specific controls —
`--google_model_code`, `--media_resolution`, `--thinking_level`,
`--thinkingBudget`, `--use_tool`, and `--json_output` — on top of the common
flags shown above.
:::

See [Running evaluations](../benchmarking/running-evaluations.md) for the
end-to-end workflow and the `script/benchmark-*/eval__*.sh` launchers that wrap
these drivers.

---

## Dataset & environment

### `download_datasets`

Fetch MedVision datasets named either by a configuration CSV or by a task-list
JSON. Exactly one of `--configs_csv` / `--tasks_json` is required.

```bash
python -m medvision_bm.benchmark.download_datasets \
  --data_dir "$MedVision_DATA_DIR" \
  --tasks_json tasks_list/TL.json --split test
```

| Flag | Default | Description |
| --- | --- | --- |
| `--data_dir` | **required** | Directory where datasets and dataset code are stored. |
| `--force_download_data` | off | **store_true** — re-download even if the data is already present. |
| `--configs_csv` | | CSV whose first column lists dataset configurations to fetch. |
| `--tasks_json` | | Task-list JSON whose keys name the datasets to fetch. |
| `--split` | `test` | Split to download: `train` or `test`. |

### `env_setup`

Provision a full evaluation environment: vendored `lmms_eval`, the dataset
package, the CUDA toolkit, vLLM, and any extra requirements file.

```bash
python -m medvision_bm.benchmark.env_setup \
  -r requirements.txt --data_dir "$MedVision_DATA_DIR" \
  --cuda_version 12.4 --vllm_version 0.10.0
```

| Flag | Default | Description |
| --- | --- | --- |
| `-r`, `--requirement` | **required** | Path to a `requirements.txt` to install after the core packages. |
| `--data_dir` | **required** | Directory for datasets and dataset code. |
| `--lmms_eval_opt_deps` | | Optional-dependency group to install with `lmms_eval` (e.g. a model extra). |
| `--cuda_version` | `12.4` | CUDA toolkit version to install. |
| `--vllm_version` | `0.10.0` | vLLM version to install. |

### `install_medvision_ds`

Install only the MedVision dataset package (`medvision_ds`) into the current
environment.

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "$MedVision_DATA_DIR"
```

| Flag | Default | Description |
| --- | --- | --- |
| `--data_dir` | **required** | Directory for datasets and dataset code (created if missing). |

### `install_vendored_lmms_eval`

Install only the vendored `lmms_eval` package, optionally with an extras group.

```bash
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
```

| Flag | Default | Description |
| --- | --- | --- |
| `--lmms_eval_opt_deps` | | Optional-dependency group to install alongside `lmms_eval`. |

---

## Parsing & summarizing

These run after evaluation, on the JSONL files under a results directory. Step 2
(`parse_outputs`) extracts predictions and computes per-sample metrics; step 3
(`summarize_*`) aggregates them into per-model, per-anatomy summaries. Point
each command at a `--task_dir` (all model subfolders) or a single `--model_dir`.

### `parse_outputs`

Extract predictions from raw JSONL and write parsed files plus updated
per-file summaries. Requires `--task_type` and one of `--task_dir` / `--model_dir`.

```bash
python -m medvision_bm.benchmark.parse_outputs \
  --task_type TL --task_dir Results/TL -p 16
```

| Flag | Default | Description |
| --- | --- | --- |
| `--task_type` | **required** | Task family: `AD`, `TL`, or `Detection` (validated). |
| `--task_dir` | | Task directory whose model subfolders are all processed. |
| `--model_dir` | | Single model directory to process. |
| `--limit` | | Cap on samples processed per JSONL file. |
| `--skip_existing` | off | **store_true** — skip files that already have parsed output. |
| `-p`, `--processes` | `None` | Worker processes; `None` runs single-process. |
| `--rm_old` | off | **store_true** — delete the existing `parsed/` folder before running. |

### `summarize_AD_task`

Aggregate Angle/Distance metrics (MAE, MRE, nMAE, SuccessRate) per model, grouped
by anatomy (FeTA-Distance, Ceph-Angle, Ceph-Distance) and by metric type across
datasets (Distance, Angle).

```bash
python -m medvision_bm.benchmark.summarize_AD_task --task_dir Results/AD -p 16
```

| Flag | Default | Description |
| --- | --- | --- |
| `--task_dir` | | Task directory containing model result folders. |
| `--model_dir` | | Single model directory to summarize. |
| `--limit` | `None` | Cap on samples per JSONL file; unset processes all. |
| `--skip_model_wo_parsed_files` | off | **store_true** — skip models with no `parsed/` folder. Only valid with `--task_dir`. |
| `-p`, `--processes` | `None` | Worker processes for metric calculation. |

:::{note}
`summarize_*` requires either `--task_dir` or `--model_dir`, and
`--skip_model_wo_parsed_files` may only be combined with `--task_dir`.
:::

### `summarize_TL_task`

Aggregate Tumour/Lesion-size metrics (MAE, MRE, nMAE, SuccessRate, in mm). Shares
the AD flags and adds removed-sample filtering.

```bash
python -m medvision_bm.benchmark.summarize_TL_task --task_dir Results/TL -p 16
```

| Flag | Default | Description |
| --- | --- | --- |
| `--task_dir` | | Task directory containing model result folders. |
| `--model_dir` | | Single model directory to summarize. |
| `--limit` | `None` | Cap on samples per JSONL file; unset processes all. |
| `--skip_model_wo_parsed_files` | off | **store_true** — skip models without a `parsed/` folder (with `--task_dir`). |
| `-p`, `--processes` | `None` | Worker processes for metric calculation. |
| `--removed_samples_dir` | `None` | Root of per-dataset removed-samples JSONs (e.g. `.../Data/Datasets`); listed samples are excluded and output filenames gain a `_filtered` suffix. |
| `--removed_samples_filename` | `multi_cluster_samples_v1.0.0_to_v1.1.0.json` | Filename of the removed-samples JSON inside each dataset subdirectory. |

### `summarize_detection_task`

Aggregate Detection metrics (IoU, Precision, Recall, F1, SuccessRate) per model,
grouped by anatomy.

```bash
python -m medvision_bm.benchmark.summarize_detection_task --task_dir Results/Detection -p 16
```

| Flag | Default | Description |
| --- | --- | --- |
| `--task_dir` | | Task directory containing model result folders. |
| `--model_dir` | | Single model directory to summarize. |
| `--limit` | `None` | Cap on samples per JSONL file; unset processes all. |
| `--skip_model_wo_parsed_files` | off | **store_true** — skip models without a `parsed/` folder (with `--task_dir`). |
| `-p`, `--processes` | `None` | Worker processes. |

See [Parsing and summarizing](../benchmarking/parsing-and-summarizing.md) for
how these fit together and where the output files land.

---

## Fine-tuning (SFT)

Supervised fine-tuning drivers live under `medvision_bm.sft` as
`train__<variant>__<family>` modules (e.g. `train__SFT-CoT__qwen2_5_vl`,
`train__fullFT-CoT__qwen2_5_vl`). They all parse the same argument surface
(`parse_validate_args_multiTask`), so the flags below apply to every driver.

```bash
python -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
  --model_family_name qwen2_5_vl \
  --base_model_hf Qwen/Qwen2.5-VL-7B-Instruct \
  --data_dir "$MedVision_DATA_DIR" \
  --tasks_list_json_path_TL tasks_list/TL.json \
  --run_name my-sft-run
```

At least one of `--tasks_list_json_path_AD`, `--tasks_list_json_path_detect`, or
`--tasks_list_json_path_TL` must be provided.

**Model & run identity**

| Flag | Default | Description |
| --- | --- | --- |
| `--run_name` | | Human-readable name for the run. |
| `--model_family_name` | **required** | Family key selecting the shared image processor / collate logic; validated against supported models. |
| `--base_model_hf` | **required** | Hugging Face ID of the base model. |
| `--lora_checkpoint_dir` | | Local path for the LoRA checkpoint. |
| `--merged_model_hf` | | Hugging Face repo ID for the merged model. |
| `--merged_model_dir` | | Local path for the merged model. |

**Weights & Biases logging**

| Flag | Default | Description |
| --- | --- | --- |
| `--wandb_resume` | `allow` | Resume mode (`allow`, `must`, `never`). |
| `--wandb_dir` | | Directory for wandb logs. |
| `--wandb_project` | | Project name. |
| `--wandb_run_name` | | Run name. |
| `--wandb_run_id` | | Run ID for resuming. |

**Data & preprocessing**

| Flag | Default | Description |
| --- | --- | --- |
| `--data_dir` | **required** | Dataset folder. |
| `--tasks_list_json_path_AD` | | Task-list JSON for the Angle/Distance task. |
| `--tasks_list_json_path_detect` | | Task-list JSON for the Detection task. |
| `--tasks_list_json_path_TL` | | Task-list JSON for the Tumour/Lesion-size task. |
| `--process_img` | `false` | Process images during dataset formatting. |
| `--process_dataset_only` | `false` | Prepare the dataset and exit without training. |
| `--skip_process_dataset` | `false` | Skip formatting and load a prepared dataset from disk. |
| `--prepared_ds_dir` | | Path to a prepared dataset to load from disk. |
| `--save_processed_img_to_disk` | `false` | Save processed images as PNGs during formatting. |
| `--new_shape_hw` | | Resize target as two ints `H W` (e.g. `--new_shape_hw 1080 1920`). |
| `--ds_download_mode` | `reuse_dataset_if_exists` | `reuse_dataset_if_exists`, `reuse_cache_if_exists`, or `force_redownload`. |

**Training loop**

| Flag | Default | Description |
| --- | --- | --- |
| `--epoch` | `1` | Number of training epochs. |
| `--save_steps` | `1000` | Steps between checkpoints. |
| `--eval_steps` | `50` | Steps between evaluations. |
| `--logging_steps` | `50` | Steps between log lines. |
| `--save_total_limit` | `10` | Max checkpoints kept. |
| `--per_device_train_batch_size` | `20` | Train batch size per device. |
| `--per_device_eval_batch_size` | `20` | Eval batch size per device. |
| `--gradient_accumulation_steps` | `2` | Accumulation steps before an update. |
| `--use_flash_attention_2` | `true` | Enable FlashAttention-2. |
| `--gradient_checkpointing` | `false` | Trade compute for memory via checkpointing. |
| `--dataloader_pin_memory` | `true` | Pin host memory for faster transfer. |
| `--resume_from_checkpoint` | `false` | Resume from the latest checkpoint. |

**Dataloader workers**

| Flag | Default | Description |
| --- | --- | --- |
| `--num_workers_concat_datasets` | `4` | Workers for concatenating per-task datasets (≤ number of tasks). |
| `--num_workers_format_dataset` | `32` | Workers for dataset formatting. |
| `--dataloader_num_workers` | `8` | Workers for data loading. |

**Sample limits**

| Flag | Default | Description |
| --- | --- | --- |
| `--train_sample_limit_per_task` | `-1` | Per-task train cap; `-1` = no limit. |
| `--val_sample_limit_per_task` | `100` | Per-task validation cap. |
| `--train_sample_limit_task_AD` | `-1` | Train cap for the AD task. |
| `--val_sample_limit_task_AD` | `-1` | Validation cap for the AD task. |
| `--train_sample_limit_task_Detection` | `-1` | Train cap for the Detection task. |
| `--val_sample_limit_task_Detection` | `-1` | Validation cap for the Detection task. |
| `--train_sample_limit_task_TL` | `-1` | Train cap for the TL task. |
| `--val_sample_limit_task_TL` | `-1` | Validation cap for the TL task. |
| `--train_sample_limit` | `-1` | Total train cap across tasks; `-1` = no limit. |
| `--val_sample_limit` | `100` | Total validation cap across tasks. |

**LoRA / merge control**

| Flag | Default | Description |
| --- | --- | --- |
| `--push_LoRA` | `false` | Push the LoRA checkpoint to the Hub after each save. |
| `--push_merged_model` | `false` | Push the merged model to the Hub after merging. |
| `--merge_model` | `false` | Merge LoRA into the base model after training. |
| `--merge_only` | `false` | Merge and push only — no training. |

**Temperature sampler (multi-task)**

| Flag | Default | Description |
| --- | --- | --- |
| `--enable_temperature_sampler` | `false` | Use temperature-weighted random sampling across tasks. |
| `--temperature_sampler_T` | `3.0` | Temperature `T`; `p(task) ∝ count^(1/T)`. `T=1` is proportional, larger flattens. |
| `--temperature_sampler_task_column` | `__task_name` | Prepared-dataset column holding task labels. |
| `--temperature_sampler_num_samples` | `-1` | Samples drawn per epoch; `≤0` uses `len(train_dataset)`. |

See [Supervised fine-tuning](../fine-tuning/sft.md) for the full workflow.

---

## RFT (reinforcement fine-tuning)

### `build_parquet_ds`

Build the train/validation Parquet dataset consumed by the
[verl](https://github.com/YongchengYAO/verl/tree/medvision-rl) trainer. Output is
written under `<data_dir>/verl_datasets/<model_family_name>/…`.

```bash
python -m medvision_bm.rft.verl.build_parquet_ds \
  --model_family_name qwen2_5_vl \
  --model_hf Qwen/Qwen2.5-VL-7B-Instruct \
  --data_dir "$MedVision_DATA_DIR" \
  --tasks_list_json_path_TL tasks_list/TL.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--model_family_name` | **required** | Family key identifying the shared image processor. |
| `--model_hf` | **required** | Hugging Face ID whose image processor drives preprocessing. |
| `--data_dir` | **required** | Dataset folder (also the Parquet output root). |
| `--prepared_ds_dir` | | Path to a prepared dataset to load from disk. |
| `--ds_download_mode` | `reuse_dataset_if_exists` | `reuse_dataset_if_exists`, `reuse_cache_if_exists`, or `force_redownload`. |
| `--new_shape_hw` | | Resize target as two ints `H W`; unset keeps native size. |
| `--without_cot_instruction` | off | **store_true** — omit CoT instructions from prompts (deprecated; kept for parity with SFT-CoT). |
| `--tasks_list_json_path_AD` | | Task-list JSON for the Angle/Distance task. |
| `--tasks_list_json_path_detect` | | Task-list JSON for the Detection task. |
| `--tasks_list_json_path_TL` | | Task-list JSON for the Tumour/Lesion-size task. |
| `--num_workers_concat_datasets` | `4` | Workers for concatenating per-task datasets. |
| `--num_workers_format_dataset` | `32` | Workers for dataset formatting. |
| `--dataloader_num_workers` | `8` | Workers for data loading. |
| `--train_sample_limit_per_task` | `-1` | Per-task train cap; `-1` = no limit. |
| `--val_sample_limit_per_task` | `100` | Per-task validation cap. |
| `--train_sample_limit_task_AD` | `-1` | Train cap for the AD task. |
| `--val_sample_limit_task_AD` | `-1` | Validation cap for the AD task. |
| `--train_sample_limit_task_Detection` | `-1` | Train cap for the Detection task. |
| `--val_sample_limit_task_Detection` | `-1` | Validation cap for the Detection task. |
| `--train_sample_limit_task_TL` | `-1` | Train cap for the TL task. |
| `--val_sample_limit_task_TL` | `-1` | Validation cap for the TL task. |
| `--train_sample_limit` | `-1` | Total train cap; `-1` = no limit. |
| `--val_sample_limit` | `-1` | Total validation cap; `-1` = no limit. |

See [Reinforcement fine-tuning](../fine-tuning/rft.md) for how the Parquet
dataset feeds the trainer.

---

## Utilities

Both utilities turn a configuration CSV (`ConfigurationsList_*.csv`) into a JSON
artifact, sharing an identical filter/flag surface. Streaming the dataset is the
default; `--no-count` skips loading entirely for a fast naming-only pass.

### `configs_to_tasks`

Emit a task-list JSON (task name → sample count) from a config CSV.

```bash
python -m medvision_bm.utils.configs_to_tasks \
  --data_dir "$MedVision_DATA_DIR" \
  --configs_csv ConfigurationsList_TL.csv \
  --out tasks_list/TL.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--data_dir` | **required** | MedVision data/code directory. |
| `--configs_csv` | **required** | Path to the `ConfigurationsList_*.csv`. |
| `--out` | **required** | Output task-list JSON path. |
| `--families` | `BoxSize,MaskSize,TumorLesionSize,BiometricsFromLandmarks` | Comma-separated task families to include. |
| `--planes` | `Axial,Coronal,Sagittal` | Comma-separated imaging planes to include. |
| `--split` | `test` | Split to include and count: `train`, `test`, or `all`. |
| `--cot` | off | **store_true** — append `-CoT` to task names. |
| `--limit` | | Cap the number of configs (for quick tests). |
| `--no-count` | off | **store_true** — skip dataset loading; write `0` counts. |
| `--no-streaming` | off | **store_true** — count via `len(load_dataset(...))` instead of streaming (materializes the Arrow cache). |

### `configs_to_pixel_sizes`

Emit per-task raw pixel-size (in-plane spacing) distributions plus an
isotropic/anisotropic rollup, writing a second `__summary` sibling file.

```bash
python -m medvision_bm.utils.configs_to_pixel_sizes \
  --data_dir "$MedVision_DATA_DIR" \
  --configs_csv ConfigurationsList_TL.csv \
  --out pixel_sizes_TL.json
```

| Flag | Default | Description |
| --- | --- | --- |
| `--data_dir` | **required** | MedVision data/code directory. |
| `--configs_csv` | **required** | Path to the `ConfigurationsList_*.csv`. |
| `--out` | **required** | Output pixel-size JSON path (`<out>__summary.json` is also written). |
| `--families` | `BoxSize,MaskSize,TumorLesionSize,BiometricsFromLandmarks` | Comma-separated task families to include. |
| `--planes` | `Axial,Coronal,Sagittal` | Comma-separated imaging planes to include. |
| `--split` | `test` | Split to include and count: `train`, `test`, or `all`. |
| `--cot` | off | **store_true** — append `-CoT` to task names. |
| `--limit` | | Cap the number of configs (for quick tests). |
| `--no-count` | off | **store_true** — skip dataset loading; write empty distributions. |
| `--no-streaming` | off | **store_true** — count via `load_dataset(...)` instead of streaming. |
