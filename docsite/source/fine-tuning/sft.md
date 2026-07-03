# Supervised fine-tuning (SFT)

SFT teaches a vision-language model to produce MedVision's structured measurements from a medical image plus an instruction. The reference recipes fine-tune **Qwen2.5-VL-7B-Instruct** with chain-of-thought (CoT) targets: the response reasons through the measurement (landmark coordinates, pixel geometry) before emitting the final value, so the model learns the *procedure* rather than memorising numbers.

All training runs through `python -m medvision_bm.sft.<entry-point>` argparse drivers. You rarely call them by hand — the shell scripts under `script/sft/` wire up the environment, the two-phase pipeline, and the full flag list for you.

:::{note}
This page assumes the package and data are already in place. See [Installation](../getting-started/installation.md) for the environment and the `MedVision_*` variables, and [Dataset loading](../dataset/loading.md) for how task-list JSONs resolve to samples.
:::

## The three recipes

`script/sft/` ships three ready-to-run scripts, all training the same 121K-sample multi-task mix (110K Detection + 5.5K Angle/Distance + 5.5K Tumour/Lesion):

| Script | Method | Resolution | Launcher |
|---|---|---|---|
| `train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh` | LoRA adapters | native / dynamic | DDP |
| `train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh` | LoRA adapters | 512×512 | DDP |
| `train__fullSFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh` | full-parameter | 512×512 | FSDP `FULL_SHARD` |

The LoRA scripts train adapters on a frozen backbone and launch with plain DistributedDataParallel. The full-parameter script updates every weight; at 7B that does not fit in DDP on 80 GB GPUs (weights + gradients + FP32 AdamW state ≈ 84 GB/GPU before activations), so it shards optimizer state, gradients, and parameters across GPUs with FSDP.

The `__512x512` variants add `--new_shape_hw 512 512`, which resizes each slice during dataset preparation and re-derives the physical pixel size for that resolution. Because measurement tasks depend on knowing the real millimetre-per-pixel scale, the prompt's pixel size always matches the resolution the model actually perceives — the 512×512 LoRA recipe is the one behind the released MedVision-V0 checkpoints.

To run one, set the paths and identifiers at the top of the script (`benchmark_dir`, `data_dir`, `base_model_hf`, `run_name`, W&B fields, and the batch/GPU settings) and execute it from the repo root:

```bash
bash script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh
```

Each script first provisions a dedicated conda env (`sft-qwen25vl`), builds `medvision_bm` into a wheel, and installs the model-specific extras:

```bash
python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --lmms_eval_opt_deps qwen2_5_vl
```

The scripts pin the planner version and acknowledge the release (required whenever you pin below `latest`):

```bash
export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.1.1'
```

## Two-phase pipeline: prepare on CPU, train on GPU

Building the prepared dataset for 121K samples — slicing NIfTI volumes, normalising, formatting CoT targets, caching PNGs — is CPU-bound and slow enough to trip distributed-training timeouts if done inside the training job. So every script invokes the *same* entry module twice.

**Phase 1 — dataset preparation (CPU, single process).** Runs with `--process_dataset_only true`, which downloads and formats every sample and writes the prepared dataset to disk. `--save_processed_img_to_disk true` also emits processed slices as PNGs so training loads them directly instead of re-slicing volumes:

```bash
python -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --process_dataset_only true \
    --skip_process_dataset false \
    --save_processed_img_to_disk true \
    --data_dir ${data_dir} \
    --model_family_name qwen25vl \
    --base_model_hf Qwen/Qwen2.5-VL-7B-Instruct \
    --new_shape_hw 512 512 \
    ...   # task lists + sample limits (see below)
```

The prepared dataset lands in `--prepared_ds_dir`, defaulting to a path derived from the per-task limits, e.g. `<data_dir>/tmp_prepared_ds_AD5500_D110000_TL5500_all121000`.

**Phase 2 — training (GPU, distributed).** Launched under `accelerate` with `--skip_process_dataset true` so it loads the cached dataset instead of rebuilding it.

LoRA uses a plain DDP launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=29502 --mixed_precision=bf16 \
    -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --skip_process_dataset true \
    --process_dataset_only false \
    ...
```

Full-parameter training swaps in the `train__fullFT-CoT__qwen2_5_vl` module and adds FSDP flags to shard the model. Note the transformer layer class to wrap is model-specific (`Qwen2_5_VLDecoderLayer` for Qwen2.5-VL):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=29502 --mixed_precision=bf16 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap Qwen2_5_VLDecoderLayer \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params false \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_sync_module_states true \
    -m medvision_bm.sft.train__fullFT-CoT__qwen2_5_vl \
    --skip_process_dataset true \
    ...
```

:::{tip}
Because phase 2 only reads the cache, re-running with `--skip_process_dataset true` skips preparation entirely. Combined with `--resume_from_checkpoint true`, an interrupted run under the same `run_name` simply picks up from its last checkpoint.
:::

## Multi-task inputs and sample limits

Tasks enter training as task-list JSONs, one flag per task; supply at least one, or several for joint multi-task training:

```bash
--tasks_list_json_path_AD     tasks_list/tasks_MedVision-AD__train_SFT.json
--tasks_list_json_path_detect tasks_list/tasks_MedVision-detect__train_SFT.json
--tasks_list_json_path_TL     tasks_list/tasks_MedVision-TL__train_SFT.json
```

Global caps `--train_sample_limit` and `--val_sample_limit` are always required. On top of them you pick one of two balancing strategies:

- **Balanced** — `--train_sample_limit_per_task` / `--val_sample_limit_per_task` spread the budget roughly evenly across the three tasks.
- **Per-task** (the shipped setting) — `--train_sample_limit_task_AD`, `--train_sample_limit_task_Detection`, `--train_sample_limit_task_TL` (and their `--val_...` counterparts) set exact counts, e.g. 5.5K / 110K / 5.5K.

If a limit exceeds the available samples, the pool is drawn with replacement rather than truncated.

## Key hyperparameters

These are the knobs the scripts expose most often; they map straight to `SFTConfig`/`TrainingArguments`:

| Flag | Role | Recipe defaults |
|---|---|---|
| `--epoch` | training epochs | `10` (LoRA), `3` (full-FT) |
| `--per_device_train_batch_size` / `--per_device_eval_batch_size` | per-GPU batch | LoRA `4`, full-FT `8` |
| `--gradient_accumulation_steps` | accumulation; effective batch = per-device × accum × #GPUs | `8` |
| `--gradient_checkpointing` | trade compute for memory | `true` (required for full-FT at 7B) |
| `--use_flash_attention_2` | FlashAttention-2 kernels | `true` |
| `--new_shape_hw <H> <W>` | resize + rescale pixel size in prep | `512 512` for the 512 recipes |
| `--save_steps` / `--eval_steps` / `--logging_steps` | checkpoint / eval / log cadence | `100 / 100 / 50` |
| `--save_total_limit` | max retained checkpoints | `10` |
| `--resume_from_checkpoint` | resume the same `run_name` | `true` |

The model family is chosen with `--model_family_name` (e.g. `qwen25vl`) plus `--base_model_hf` (a Hub ID or local path). The family name is validated at startup against the registered model list — both `vllm_qwen25vl` and the bare `qwen25vl` are accepted — so a typo fails fast instead of mid-run.

## Temperature-based multi-task sampling

With 110K detection samples against 5.5K each for A/D and T/L, uniform sampling lets detection swamp every batch. Turning on the temperature sampler re-weights how often each task is drawn:

```bash
--enable_temperature_sampler true \
--temperature_sampler_T 5
```

Internally this swaps the trainer for a `TemperatureSamplerSFTTrainer` subclass whose train sampler is a `WeightedRandomSampler` (with replacement, seeded from the project `SEED`). Per-task probability is `count^(1/T)`, normalised, and each sample's weight is that task probability divided by the task's count. `T = 1` reproduces count-proportional sampling; larger `T` flattens the distribution so the minority tasks are oversampled — the scripts use `T = 5`. It only reshapes training batches and has no effect during phase-1 preparation. With a single task present, it transparently falls back to the standard sampler.

## Merging and pushing (LoRA only)

The LoRA drivers can merge the trained adapter back into the base model and push either artifact to the Hub:

| Flag | Effect |
|---|---|
| `--merge_model true` | after training, merge the final adapter into the base weights |
| `--merged_model_dir` / `--merged_model_hf` | local output dir and Hub repo name for the merged model |
| `--push_merged_model true` | upload the merged model to the Hub |
| `--push_LoRA true` | upload the LoRA adapter after each save |
| `--merge_only true` | skip training; merge and push the last existing checkpoint |

The full-parameter driver writes complete model checkpoints directly, so it has no merge/push options (its `--lora_checkpoint_dir` argument is reinterpreted internally as the plain checkpoint directory).

:::{warning}
Merging a LoRA adapter into the base weights can slightly degrade measurement accuracy versus serving base + adapter. Keep the unmerged adapter around if you care about the last decimal.
:::

## Entry points and other model families

Two CoT drivers ship today, both targeting Qwen2.5-VL:

- `medvision_bm.sft.train__SFT-CoT__qwen2_5_vl` — LoRA
- `medvision_bm.sft.train__fullFT-CoT__qwen2_5_vl` — full-parameter

They share the preparation, sampler, and trainer plumbing in `medvision_bm.sft.sft_utils` (`prepare_dataset`, `prepare_trainer`, `prepare_trainer_fullFT`). Extending the same recipe to another supported family — for example `gemma4`, `medgemma`, or `qwen3vl` — follows the identical two-recipe pattern: a `train__SFT-CoT__<family>` / `train__fullFT-CoT__<family>` module reusing these helpers, the matching `--model_family_name`, the family's decoder-layer class in `--fsdp_transformer_layer_cls_to_wrap`, and the right `--lmms_eval_opt_deps` for `env_setup`. See [Add a model](../extending/add-a-model.md) for that walkthrough.

## See also

- [Reinforcement fine-tuning (RFT)](rft.md) — GRPO-based training on the same tasks.
- [CLI reference](../reference/cli.md) — the full flag list for each entry point.
- [API reference](../reference/api/index.md) — `sft_utils` functions (`prepare_dataset`, `prepare_trainer`, `prepare_trainer_fullFT`).
