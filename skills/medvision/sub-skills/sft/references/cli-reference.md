# MedVision SFT — CLI and environment reference

All ten entry points share one argparse definition (`parse_args_multiTask` in `medvision_bm.sft.sft_utils`,
validated by `parse_validate_args_multiTask`). Verified by diffing `--help` output across
`train__SFT-CoT__qwen2_5_vl`, `train__SFT-CoT__medgemma`, `train__fullFT-CoT__gemma4`,
`train__fullFT-CoT__qwen3vl`, `train__SFT__qwen2_5_vl` and `train__qwen25vl_AD_TL_tooluse`: **the flag sets are
identical**. What differs is which flags the entry point actually consumes (last column of the matrix below).

Boolean flags take a value (`--gradient_checkpointing true`), parsed by `str2bool`, which accepts
`yes/true/t/y/1` and `no/false/f/n/0` case-insensitively and raises `argparse.ArgumentTypeError` otherwise.

---

## 1. Entry-point matrix

| Module (`python -m medvision_bm.sft.<module>`) | FT mode | Family key | Collate fn | Trainer | Notes |
| --- | --- | --- | --- | --- | --- |
| `train__SFT-CoT__qwen2_5_vl` | QLoRA | `qwen25vl` | `make_collate_fn_Qwen25VL` | `prepare_trainer` | CoT targets |
| `train__SFT-CoT__qwen3vl` | QLoRA | `qwen3vl` | `make_collate_fn_Qwen25VL` | `prepare_trainer` | Qwen3-VL reuses the ChatML collate |
| `train__SFT-CoT__gemma4` | QLoRA | `gemma4` | `make_collate_fn_Gemma4` | `prepare_trainer` | Gemma-4 turn markers |
| `train__SFT-CoT__medgemma` | QLoRA | `medgemma` | `make_collate_fn_MedGemma` | `prepare_trainer` | Gemma-3 lineage |
| `train__fullFT-CoT__qwen2_5_vl` | full | `qwen25vl` | `make_collate_fn_Qwen25VL` | `prepare_trainer_fullFT` | FSDP; no merge step |
| `train__fullFT-CoT__qwen3vl` | full | `qwen3vl` | `make_collate_fn_Qwen25VL` | `prepare_trainer_fullFT` | FSDP |
| `train__fullFT-CoT__gemma4` | full | `gemma4` | `make_collate_fn_Gemma4` | `prepare_trainer_fullFT` | FSDP |
| `train__fullFT-CoT__medgemma` | full | `medgemma` | `make_collate_fn_MedGemma` | `prepare_trainer_fullFT` | FSDP |
| `train__SFT__qwen2_5_vl` | QLoRA | `qwen25vl` | `make_collate_fn_Qwen25VL` | `prepare_trainer` | **non-CoT** targets; dataset root `SFT_datasets` |
| `train__qwen25vl_AD_TL_tooluse` | full | `qwen25vl` | `make_collate_fn_Qwen25VL_tooluse` | `prepare_trainer_fullFT` | A/D + T/L only; 5-turn tool-call targets |

Mode-specific argument handling:

- Full-FT and tool-use entry points **rename** `--lora_checkpoint_dir` to `checkpoint_dir` internally
  (`args_dict["checkpoint_dir"] = args_dict.pop("lora_checkpoint_dir")`), so the same flag is used everywhere.
- They **ignore** `--merge_model`, `--merge_only`, `--merged_model_hf`, `--merged_model_dir` (no LoRA to merge),
  and reuse `--push_LoRA` as the trainer's `push_model`.
- `--model_family_name` is validated by `check_model_supported`, which compares against
  `lmms_eval.models.get_available_model_names()` extended with the `vllm_`-stripped names. Unknown keys raise
  `ValueError` listing the supported set, before any data is read.
- `parse_validate_args_multiTask` raises `AssertionError` if **all three** `--tasks_list_json_path_*` are absent,
  and exports `WANDB_RESUME` / `WANDB_DIR` / `WANDB_PROJECT` / `WANDB_NAME` / `WANDB_RUN_ID` from the wandb flags
  (creating `wandb_dir` if needed).

---

## 2. All flags

### Model and output

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--model_family_name` | str | **required** | Image-processor family key; must be in the `lmms_eval` registry (`vllm_` prefix optional). |
| `--base_model_hf` | str | **required** | Hub id or local folder of the base model (also the processor source). |
| `--run_name` | str | None | Run identifier; passed to `SFTConfig.run_name` and W&B. |
| `--lora_checkpoint_dir` | str | None | Adapter output dir (LoRA) / checkpoint output dir (full-FT, tool-use). |
| `--merged_model_hf` | str | None | Hub repo id for the merged model (LoRA only). |
| `--merged_model_dir` | str | None | Local dir to save the merged model (LoRA only). |

### Data

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--data_dir` | str | **required** | Dataset cache; sets `MedVision_DATA_DIR` and the HF env vars, and roots the default `prepared_ds_dir`. |
| `--tasks_list_json_path_AD` | str | None | A/D task list (`tag_ds="BiometricsFromLandmarks"`). |
| `--tasks_list_json_path_detect` | str | None | Detection task list (`tag_ds="BoxSize"`; `_BoxCoordinate_` names auto-renamed). |
| `--tasks_list_json_path_TL` | str | None | T/L task list (`tag_ds="TumorLesionSize"`). |
| `--process_img` | bool | `False` | Embed decoded images in the Arrow dataset (`processed_images`). Not recommended — huge cache. |
| `--process_dataset_only` | bool | `False` | Stop after preparing and saving the dataset (phase A). |
| `--skip_process_dataset` | bool | `False` | Skip formatting/saving; load the prepared dataset from disk (phase B). Without `--prepared_ds_dir` the load+split stage still runs on rank 0 to resolve the default name. |
| `--prepared_ds_dir` | str | None | Explicit prepared-dataset directory, taken as-is; None derives the default (true-size) name. With `--skip_process_dataset true` it also skips the load+split stage. Phase A prints the directory; the launchers pass it here. |
| `--save_processed_img_to_disk` | bool | `False` | Write one PNG per slice and store the path in `image_file_png` (recommended). |
| `--new_shape_hw` | 2 ints | None | Explicit `(height, width)` resize during preparation; also changes the pixel size printed in the prompt. |
| `--ds_download_mode` | str | `reuse_dataset_if_exists` | Also `reuse_cache_if_exists`, `force_redownload`. |

### Schedule

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--epoch` | int | `1` | Recipes use 10 (LoRA) / 3 (full-FT). |
| `--save_steps` | int | `1000` | Recipes use 100. |
| `--eval_steps` | int | `50` | Recipes use 100. |
| `--logging_steps` | int | `50` | Recipes use 20-50. |
| `--save_total_limit` | int | `10` | Recipes use 3 for 31B full-FT (checkpoints ~190 GB each). |

### Compute

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--per_device_train_batch_size` | int | `20` | `effective = this * grad_accum * num_gpus`. |
| `--per_device_eval_batch_size` | int | `20` | |
| `--gradient_accumulation_steps` | int | `2` | |
| `--use_flash_attention_2` | bool | `True` | `false` falls back to **eager**, not SDPA; use `MEDVISION_SFT_ATTN=sdpa` for SDPA. |
| `--gradient_checkpointing` | bool | `False` | Required at 7B+ for full-FT; recipes set `true` everywhere. |
| `--dataloader_pin_memory` | bool | `True` | |
| `--dataloader_num_workers` | int | `8` | Trainer DataLoader workers (persistent workers are disabled). |
| `--num_workers_concat_datasets` | int | `4` | Task-loading processes; clamped to `min(cgroup CPUs, #tasks)` and forced to 1 when a dataset was just downloaded. |
| `--num_workers_format_dataset` | int | `32` | Formatting-map processes; clamped to the cgroup CPU count. |

### Sample limits

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--train_sample_limit_per_task` | int | `-1` | Fallback per-task train cap when the task-specific flag is unset/<=0. |
| `--val_sample_limit_per_task` | int | `100` | Fallback per-task validation target. |
| `--train_sample_limit_task_AD` / `_Detection` / `_TL` | int | `-1` | Per-task train caps (`-1` = full pool). |
| `--val_sample_limit_task_AD` / `_Detection` / `_TL` | int | `-1` | Per-task validation targets (`<=0` falls back to `--val_sample_limit_per_task`, default 100). |
| `--train_sample_limit` | int | `-1` | **Global** cap after concatenation. Larger than the pool => sampling with replacement. |
| `--val_sample_limit` | int | `100` | Global validation cap (same semantics). The repository recipes pass `200` (older `v1.0.0` recipes `500`); pass `-1` only if you want to keep every per-task validation row. |

A value of exactly `0` on any limit raises `ValueError` from `_get_sample_limit` (ambiguous). Semantics are in
`data-preparation.md`; `scripts/check_sample_limits.py` resolves them through the real function.

### Merge, push and resume

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--push_LoRA` | bool | `False` | Push adapters after each save (`SFTConfig.push_to_hub`, private repo). Full-FT reuses it for the trained model. |
| `--push_merged_model` | bool | `False` | Push the merged model; requires `--merged_model_hf`. |
| `--merge_model` | bool | `False` | Merge after training (LoRA only). |
| `--merge_only` | bool | `False` | Skip training entirely; merge the adapter saved **directly at** `--lora_checkpoint_dir` (written by `trainer.save_model()` at the end of training) — not the newest `checkpoint-*` subdirectory — and optionally push. |
| `--resume_from_checkpoint` | bool | `False` | Resume from the newest checkpoint in the checkpoint dir. |

### Multi-task sampling

| Flag | Type | Default | Notes |
| --- | --- | --- | --- |
| `--enable_temperature_sampler` | bool | `False` | Training-only; ignored during `--process_dataset_only true`. |
| `--temperature_sampler_T` | float | `3.0` | `p(task) ~ count^(1/T)`; must be > 0. Recipes use 5. |
| `--temperature_sampler_task_column` | str | `__task_name` | Column added during preparation; missing column raises. |
| `--temperature_sampler_num_samples` | int | `-1` | Draws per epoch; `<= 0` keeps `len(train_dataset)`. |

### Weights & Biases

| Flag | Type | Default |
| --- | --- | --- |
| `--wandb_resume` | str | `allow` (also `must`, `never`) |
| `--wandb_dir` | str | None (created if given) |
| `--wandb_project` | str | None |
| `--wandb_run_name` | str | None |
| `--wandb_run_id` | str | None — must be unique within the project |

---

## 3. `accelerate launch` flags used by the recipes

```
CUDA_VISIBLE_DEVICES=<gpu list> accelerate launch \
    --num_processes=<#gpus> --main_process_port=<port> [--mixed_precision=bf16] \
    [FSDP flags] -m medvision_bm.sft.<module> <trainer flags>
```

| Flag | When | Why |
| --- | --- | --- |
| `--num_processes=N` | always | Must equal the number of visible GPUs. |
| `--main_process_port=P` | always | Distinct per concurrent run. |
| `--mixed_precision=bf16` | LoRA, and the 140 GB full-FT recipe | Gives fp32 master weights under FSDP. **Omit** it for the pure-bf16 80 GB recipe — `SFTConfig(bf16=False)` cannot override the `ACCELERATE_MIXED_PRECISION` env var this flag injects. |
| `--use_fsdp` | full-FT, tool-use | Enables FSDP; also makes `prepare_trainer_fullFT` skip `device_map` (see `training-configuration.md`). |
| `--fsdp_sharding_strategy FULL_SHARD` | full-FT | Shards params, grads and optimizer state. |
| `--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP` | full-FT | Wrap per decoder layer. |
| `--fsdp_transformer_layer_cls_to_wrap <Class>` | full-FT | Must match the installed transformers' class name for that checkpoint. |
| `--fsdp_state_dict_type FULL_STATE_DICT` | full-FT | Saves a consolidated checkpoint (all-gathered on rank 0). |
| `--fsdp_offload_params false` | full-FT | CPU offload needs ~700 GB host RAM at 31B; check the **cgroup** limit, not `free`. |
| `--fsdp_cpu_ram_efficient_loading true` | full-FT | Rank 0 loads on CPU, other ranks on meta. |
| `--fsdp_sync_module_states true` | full-FT | Broadcasts rank-0 weights at wrap time. |

No accelerate YAML config file is required or shipped: the recipes pass every FSDP setting on the command line.

---

## 4. `MEDVISION_SFT_*` environment knobs

Read directly by `medvision_bm.sft.sft_utils` / the collate modules. All default to off.

| Variable | Read by | Effect |
| --- | --- | --- |
| `MEDVISION_SFT_ATTN` | both trainers | Overrides the attention implementation (e.g. `sdpa`) regardless of `--use_flash_attention_2`. Needed for architectures whose FA2 path is unvalidated on their transformers pin. |
| `MEDVISION_SFT_COMPLETION_ONLY=1` | Gemma-4 and MedGemma collates | Adds completion-only (assistant-turn) loss masking on top of pad/image masking. This is what the `__cmplLoss` launcher variants set. Qwen collates always mask; this flag has no effect there. |
| `MEDVISION_SFT_OPTIM` | both trainers | `SFTConfig.optim` override. Default `adamw_torch_fused`. Recipes use `paged_adamw_8bit` (Gemma-family QLoRA) and `adamw_bnb_8bit` (80 GB full-FT). |
| `MEDVISION_SFT_SAVE_ONLY_MODEL=1` | full-FT | `save_only_model=True`: weights only, no optimizer/scheduler. **Required** with bnb 8-bit optimizers under FSDP (quantized state cannot be gathered by `FULL_STATE_DICT`). Resume then restarts the optimizer fresh. |
| `MEDVISION_SFT_LR` | full-FT | `learning_rate` override. Default `2e-5`; pure-bf16 recipes raise it to `4e-5` so AdamW updates stay above the bf16 rounding floor. |
| `MEDVISION_SFT_PURE_BF16=1` | full-FT | Sets `SFTConfig(bf16=False)`: no fp32 master weights, no persistent bf16 `_mp_shard`. Must be combined with **omitting** `--mixed_precision` on the launch. |
| `MEDVISION_SFT_USE_LIGER=1` | full-FT | `use_liger_kernel=True` — fused linear cross-entropy, which avoids materializing seq x vocab logits (matters for 262k-vocab Gemma models). Requires `pip install liger-kernel` (>=0.5.4 for Gemma 3). |
| `MEDVISION_SFT_BF16_GRADS=1` | full-FT | Sets the FSDP plugin's `MixedPrecision(keep_low_precision_grads=True)`. **Incompatible with accelerate bf16 MP** (torch's `.grad` setter requires matching dtypes); use `MEDVISION_SFT_PURE_BF16=1` instead. Warns and no-ops without an FSDP plugin. |
| `MEDVISION_SFT_SYNC_EACH_BATCH=1` | full-FT | Replaces `accelerator.no_sync` with a null context so gradients reduce-scatter every micro-batch and accumulate **sharded**. Without it FSDP accumulates full unsharded grads during accumulation — the root cause of step-0 OOMs at 27-31B. |
| `MEDVISION_SFT_MEMPROBE=1` | full-FT | Adds a callback printing per-rank allocated/reserved/device-used after the FSDP wrap and after step 1, plus the live forward module, gradient-checkpointing state, FSDP mixed-precision policy and optimizer class/state dtypes. |
| `MEDVISION_SFT_MEMSNAPSHOT=1` | full-FT | Records the CUDA allocator history and dumps `oom_memsnap_rank<N>.pickle` into the checkpoint dir on `torch.OutOfMemoryError`. Small steady-state overhead. |

Other environment variables that matter:

| Variable | Purpose |
| --- | --- |
| `MedVision_PLANNER_VERSION` | Annotation version; the dataset loader hard-fails without it. |
| `MedVision_ACK_RELEASE` | Required in addition when pinning below the newest dataset release. |
| `MedVision_DATA_DIR` | Set from `--data_dir` by `setup_env_hf_medvision_ds`; `load_split_limit_dataset` asserts on it. |
| `HF_TOKEN` | Gated models/datasets. Strip whitespace — a trailing newline yields 401. |
| `WANDB_MODE=offline`, `WANDB_DEBUG`, `WANDB_CORE_DEBUG` | Logging debug switches. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces fragmentation in the long full-FT runs. |
| `CUDA_HOME` | DeepSpeed's import-time compatibility check needs it even when DeepSpeed is unused. |
| `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE` | Left at defaults (0); only touched when debugging interconnect issues. |

---

## 5. `python -m medvision_bm.sft.env_setup`

| Flag | Required | Meaning |
| --- | --- | --- |
| `--data_dir` | yes | Data directory; `medvision_ds` is installed against it. |
| `-r`, `--requirement` | no | Install from a frozen requirements file instead of the individual package set. |
| `--lmms_eval_opt_deps` | no | Optional-dependency group for the vendored `lmms_eval` (e.g. `qwen2_5_vl`, `qwen3_vl`). |

Without `--requirement` it runs, in order: vendored `lmms_eval` -> `install_basic_packages()` ->
`install_flash_attention_torch_and_deps_py311_v2()` -> `transformers==4.54.0` -> `install_medvision_ds(data_dir)`.
With `--requirement` it runs: vendored `lmms_eval` -> the requirements file -> `install_medvision_ds(data_dir)`.
The frozen SFT requirement sets are `requirements_sft_{qwen25vl,medgemma,gemma4,qwen3.6vl}.txt` — key pins:
torch 2.6.0 / torchvision 0.21.0, `flash-attn` 2.7.3 (cu12/torch2.6/cp311 wheel), `trl==0.19.1`,
`datasets==3.6.0`, `wandb==0.21.4`, `protobuf==6.33.0`; transformers 4.54.0 for qwen25vl and medgemma versus
5.5.0 for gemma4 and qwen3.6vl; `huggingface_hub` 0.35.3 (qwen25vl) / 0.36.0 (medgemma) versus 1.22.0
(gemma4, qwen3.6vl), and
`liger_kernel==0.8.0` only in the medgemma set.

## 6. `python -m medvision_bm.utils.push_hf_model` (reference only)

`--repo_id <user>/<name>` (required), `--folder_path <dir>` (required), `--message <commit message>` (optional).
Creates the repo if missing and uploads the folder with `HfApi.upload_folder`. Needs Hub credentials; never run
it without explicit consent.
