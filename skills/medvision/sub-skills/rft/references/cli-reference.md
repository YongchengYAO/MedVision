# CLI reference: parquet builders, checkpoint helpers, RFT-model evaluation

All entry points were run with `--help` on a CPU-only host (exit 0); defaults below are the argparse defaults.
Nothing here needs a GPU except the actual `eval__medvision-model-rft` inference.

## 1. `python -m medvision_bm.rft.verl.build_parquet_ds` (23 flags)

Builds `train_verl.parquet` + `validation_verl.parquet` in one pass (whole split in RAM). Required flags are marked.

| Flag | Default | Meaning / notes |
| --- | --- | --- |
| `--model_family_name` (req.) | -- | key of `get_resized_img_shape`; decides how the perceived image size / pixel size in T/L and A/D prompts is computed. Local families: `qwen25vl`, `qwen3vl`, `gemma3`, `gemma4`, `medgemma`, `lingshu`, `llama_3_2_vision`, `llava_onevision`, `internvl3`, `minimax_m3`, `glm4v`, `meddr`, `llava_med`, `huatuogpt_vision`, `healthgpt` (the `vllm_*` benchmark keys are accepted too). Also the output sub-directory name |
| `--model_hf` (req.) | -- | HF id or local dir whose **image processor** is loaded to probe the resize (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) |
| `--data_dir` (req.) | -- | MedVision data dir; output goes to `<data_dir>/verl_datasets/...`. **Does not set `MedVision_DATA_DIR`** -- export it yourself |
| `--prepared_ds_dir` | None | parsed but **unused** by every builder (verified from source) |
| `--ds_download_mode` | `reuse_dataset_if_exists` | or `reuse_cache_if_exists`, `force_redownload` (HF `datasets` download mode) |
| `--new_shape_hw H W` | None (original size) | resize before embedding; the paper uses `512 512` |
| `--without_cot_instruction` | off | lite prompts (`SYSTEM_PROMPT_LITE`), lite `extra_info`; marked deprecated in source (train/SFT distribution shift) |
| `--tasks_list_json_path_AD` | None | A/D task list (SFT namespace: `<dataset>_BiometricsFromLandmarks_...` keys; loader appends `_Train`) |
| `--tasks_list_json_path_detect` | None | Detection task list (`<dataset>_BoxSize_...`) |
| `--tasks_list_json_path_TL` | None | T/L task list (`<dataset>_TumorLesionSize_...`) -- at least one of the three is required |
| `--num_workers_concat_datasets` | 4 | parallel per-task loading; should be <= number of tasks |
| `--num_workers_format_dataset` | 32 | `.map` workers (clamped to cgroup CPUs); the main RAM knob (x 50 buffered images each) |
| `--dataloader_num_workers` | 8 | parsed but **unused** by the builders |
| `--train_sample_limit_per_task` | -1 | fallback per-task train cap when the task-specific one is <= 0 |
| `--val_sample_limit_per_task` | 100 | fallback per-task validation size |
| `--train_sample_limit_task_AD` / `_Detection` / `_TL` | -1 | task-specific train caps (`-1` = whole pool after the val carve-out) |
| `--val_sample_limit_task_AD` / `_Detection` / `_TL` | -1 (-> 100 fallback) | task-specific validation sizes; the loader asserts the resolved value is > 0 |
| `--train_sample_limit` | -1 | **global** cap after concatenating tasks; `> size` samples **with replacement**; else seeded shuffle + head |
| `--val_sample_limit` | -1 | global validation cap, same semantics |

`parse_sample_limits` rules (shared with SFT): a literal `0` raises `ValueError` ("ambiguous: use -1 ... to skip a task,
omit its `--tasks_list_json_path_*`"); a task whose JSON is absent gets limit 0 and is skipped. Shuffling and
with-replacement sampling use `SEED` from `medvision_bm.utils.configs`.

Console output to look for: `Prepared Verl parquet dataset directory: ...` then `Saving <split> split to ...`.

## 2. `python -m medvision_bm.rft.verl.build_parquet_ds__checkpointed` (24 flags)

Same flags as §1 plus:

| Flag | Default | Notes |
| --- | --- | --- |
| `--shard_size` | 50000 | training rows per shard; each shard is formatted -> cleaned -> written to `shards/train_shard_NNNN.parquet` -> freed |
| `--num_workers_format_dataset` | **64** | (differs from §1) |

Behaviour differences: raw rows of every task are loaded first **without images** (per-task train limit applied on
raw rows), tagged with `_task_tag`, concatenated, shuffled / globally capped, then formatted shard by shard;
`checkpoint.json` records finished shards so a killed job resumes (orphan shard files found on disk are adopted);
the validation split is formatted unsharded; finally shards are stream-merged with `pyarrow.parquet.ParquetWriter`
into `train_verl.parquet` (one shard in RAM at a time). Raises `ValueError("No task JSON paths provided ...")` when no
task list is given and `FileNotFoundError` if a shard is missing at merge time. Source memory budget at
`shard_size=50000`, 64 workers: ~2.4 GB map buffers + ~5 GB shard, 10-15 GB peak per shard.

## 3. `build_parquet_ds_with_testset` / `build_parquet_ds_with_testset__checkpointed` (29 / 30 flags)

Same as §1 / §2 (`__checkpointed` also has `--shard_size`, format-worker default 64) plus a **test split** written to
`test_verl.parquet` from the MedVision `_Test` configs via `load_split_limit_dataset_tr_val_ts`:

| Extra flag | Default | Notes |
| --- | --- | --- |
| `--test_sample_limit_task_AD` / `_Detection` / `_TL` | -1 | per-task test caps |
| `--test_sample_limit` | -1 | global test cap (with-replacement semantics as above) |
| `--train_sample_limit_per_subset` | -1 | cap per HF dataset config **before** merging (train pool) |
| `--test_sample_limit_per_subset` | -1 | same for the test pool |

The RFT recipes never read `test_verl.parquet` (source comment: "prepared for debugging and future flexibility").

## 4. `python -m medvision_bm.rft.verl.patch_layer_name`

Cleans **PEFT/LoRA wrappers** out of a verl-exported checkpoint: every safetensors key starting with
`base_model.model.` is renamed without that prefix (in place, multiprocess, `min(cpu_count, 32)` workers) and
`model.safetensors.index.json` is patched accordingly; files without the prefix are skipped.

| Flag | Notes |
| --- | --- |
| `--model_dir` | directory with `*.safetensors` (asserted to exist) |
| `--push_to_hub` | upload the cleaned folder with `HfApi.upload_folder` (needs `--repo_id` and credentials) |
| `--repo_id` | target Hub repo |

Only relevant if you trained with LoRA in verl; the paper's full-parameter recipes do not need it.

## 5. `python -m medvision_bm.benchmark.eval__medvision-model-rft` (23 flags) -- requires GPU + vLLM

Benchmarks a Qwen2.5-VL-family RFT checkpoint through the vendored `lmms_eval` `vllm_qwen25vl` wrapper, one task at a
time, with resume via the completed-tasks JSON.

| Flag | Default | Notes |
| --- | --- | --- |
| `--lmms_eval_module` | `vllm_qwen25vl` | lmms_eval model module (e.g. `vllm_qwen3vl`) |
| `--model_hf_id` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF id **or local merged checkpoint dir** |
| `--lora_path` | None | optional LoRA adapter |
| `--model_name` | `Qwen2.5-VL-7B-Instruct` | results sub-dir + completed-tasks key |
| `--dtype` | `auto` | vLLM dtype |
| `--reshape_image_hw` | None | e.g. `512x512` -- must match the training resize |
| `--max_new_tokens` | 4096 | |
| `--stop_strings ...` | None | e.g. `</answer>` |
| `--use_system_prompt` | off | injects `rft_prompts.SYSTEM_PROMPT` as the first message -- **required** for verl-GRPO models |
| `--batch_size_per_gpu` | 20 | total batch = this x visible GPUs (`max_num_seqs`) |
| `--gpu_memory_utilization` | 0.99 | vLLM |
| `--tasks_list_json_path` | -- | eval task list (`-CoT` namespace, e.g. `tasks_MedVision-detect-CoT.json`) |
| `--results_dir`, `--task_status_json_path`, `--data_dir` | -- | Results root, completed-tasks tracker, MedVision data dir |
| `--sample_limit` | 1000 | per task; `--sample_indices [start:stop]` or `[start,stop,step]` overrides it |
| `--log-sys-prompt` | off | store the system prompt in per-sample JSONL |
| `--skip_env_setup` / `--env_setup_only` / `--skip_update_status` | off | env-install control / debugging |
| `--scaled_ps_low` / `--scaled_ps_high` | 0.5 / 3.0 | pixel-size scaling range for `-scaledPS` task variants |

Without `--skip_env_setup` the script installs (in order): `huggingface_hub==0.35.3`, vendored lmms_eval with the
`qwen2_5_vl` extra, `medvision_ds`, torch cu124, `vllm==0.10.0`, then `transformers==4.54.1` + `accelerate==1.9.0`.
It sets tensor parallel = number of visible CUDA devices and passes `hf_overrides={"vision_start_token_id": ...}`
resolved from the model config/tokenizer (fallback 151652). The repository's MedVision-V0 launchers use the manual
path instead: `install_medvision_ds --data_dir`, `install_vendored_lmms_eval --lmms_eval_opt_deps medvision_v0`
(extra = `transformers==4.54.1`, `accelerate==1.9.0`, `decord`, `qwen_vl_utils`), `pip install -r
requirements_eval_medvision-v0.txt --no-deps` (pins `vllm==0.10.0`, `torch==2.7.1`), then the eval with
`--skip_env_setup`. Details of the eval/parse/summarize pipeline: `../../benchmark-evaluation/SKILL.md`,
`../../results-parsing-and-metrics/SKILL.md`.

## 6. Bundled scripts

```
bash   scripts/build_parquet_ds.sh --help      # wrapper with explicit paths, small defaults, --checkpointed, --dry-run
python scripts/inspect_parquet_ds.py --help    # schema / counts / first row of a parquet file or dataset dir
```

`build_parquet_ds.sh` exports `MedVision_DATA_DIR=<data-dir>`, refuses to run when `MedVision_PLANNER_VERSION` is unset
(warns only under `--dry-run`), defaults the global `--train-limit` / `--val-limit` to the sum of the per-task limits, and
prints the expected output directory before running.
