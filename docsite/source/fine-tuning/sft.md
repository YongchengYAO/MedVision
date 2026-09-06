# Supervised fine-tuning (SFT)

SFT teaches a vision-language model to produce MedVision's structured measurements from a medical image plus an instruction. The reference recipes fine-tune **Qwen2.5-VL-7B-Instruct** with chain-of-thought (CoT) targets: the response reasons through the measurement (landmark coordinates, pixel geometry) before emitting the final value, so the model learns the *procedure* rather than memorising numbers.

All training runs through `python -m medvision_bm.sft.<entry-point>` argparse drivers. You rarely call them by hand — the shell scripts under `script/sft/` wire up the environment, the two-phase pipeline, and the full flag list for you.

:::{note}
This page assumes the package and data are already in place. See [Installation](../getting-started/installation.md) for the environment and the `MedVision_*` variables, and [Dataset loading](../dataset/loading.md) for how task-list JSONs resolve to samples.
:::

## The three recipes

`script/sft/` ships three ready-to-run **Qwen2.5-VL-7B reference recipes** (of 21 launchers in total), all training the same 121K-sample multi-task mix (110K Detection + 5.5K Angle/Distance + 5.5K Tumour/Lesion):

| Script | Method | Resolution | Launcher |
|---|---|---|---|
| `train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh` | LoRA adapters | native / dynamic | DDP |
| `train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh` | LoRA adapters | 512×512 | DDP |
| `train__fullSFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh` | full-parameter | 512×512 | FSDP `FULL_SHARD` |

The LoRA scripts are **QLoRA**: the base model is loaded in 4-bit NF4 with double quantization and frozen, and adapters (`r=16`, `alpha=32`, `dropout=0.05`, `target_modules="all-linear"`, plus `modules_to_save=["lm_head", "embed_tokens"]`) train on top, launched with plain DistributedDataParallel. The LoRA learning rate is `2e-4`; full-parameter runs use `2e-5`, overridable with the `MEDVISION_SFT_LR` environment variable. The full-parameter script updates every weight; at 7B that does not fit in DDP on 80 GB GPUs (weights + gradients + FP32 AdamW state ≈ 84 GB/GPU before activations), so it shards optimizer state, gradients, and parameters across GPUs with FSDP.

The `__512x512` variants add `--new_shape_hw 512 512`, which resizes each slice during dataset preparation and re-derives the physical pixel size for that resolution. Because measurement tasks depend on knowing the real millimetre-per-pixel scale, the prompt's pixel size always matches the resolution the model actually perceives — the 512×512 full SFT recipe is the one behind the released MedVision-V0 checkpoints.

:::{note}
MedVision-V0 is produced by **two-stage post-training**: this full-parameter 512×512 SFT, followed by reinforcement fine-tuning (GRPO). See [Reinforcement fine-tuning](rft.md).
:::

Beyond these 7B reference recipes, `script/sft/` carries the same layout for larger families — MedGemma-27B, Gemma-4-31B, Qwen3.5/Qwen3.6-27B — whose memory-recipe variants are covered in *Scaling full-parameter SFT to 27B and beyond* below.

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

The prepared dataset lands in `--prepared_ds_dir`. Left unset, it defaults to
`<data_dir>/SFT-CoT_datasets/<model_family_name>/ds__AD<n>_D<n>_TL<n>_all<n><suffix>` — for the run above,
`<data_dir>/SFT-CoT_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-wh-512x512`.
Each `<n>` is the requested cap when that limit is set, otherwise the true post-split row count, and
`<suffix>` is `__resized-wh-<W>x<H>` with `--new_shape_hw` or `__original` without it.

**Phase 2 — training (GPU, distributed).** Launched under `accelerate` with `--skip_process_dataset true` so it loads the cached dataset instead of rebuilding it. Phase 2 must be told **where** that cache is: the prep run prints `Prepared dataset saved at '<dir>'`, and the shipped launchers tee that output to a log, `sed` the path out of it, and pass it back as `--prepared_ds_dir`. Omit it and training re-runs the whole load-and-split stage.

LoRA uses a plain DDP launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 --main_process_port=29502 --mixed_precision=bf16 \
    -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --skip_process_dataset true \
    --process_dataset_only false \
    --prepared_ds_dir ${prepared_ds_dir} \
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
    --prepared_ds_dir ${prepared_ds_dir} \
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

Global caps `--train_sample_limit` and `--val_sample_limit` are both **optional**: unset or `-1` means no total train cap, and `--val_sample_limit` defaults to 100. Only `0` is rejected as ambiguous. (What *is* required is at least one `--tasks_list_json_path_*`.) On top of them you pick one of two balancing strategies:

- **Balanced** — `--train_sample_limit_per_task` / `--val_sample_limit_per_task` spread the budget roughly evenly across the three tasks.
- **Per-task** (the shipped setting) — `--train_sample_limit_task_AD`, `--train_sample_limit_task_Detection`, `--train_sample_limit_task_TL` (and their `--val_...` counterparts) set exact counts, e.g. 5.5K / 110K / 5.5K.

This no-op rule applies to the **per-task** limits only: if one exceeds the samples available for its task, the pool is capped at what is available and never oversampled. The **global** caps behave differently — a `--train_sample_limit` / `--val_sample_limit` larger than the combined pool bootstrap-resamples **with replacement** (seeded from `SEED`), so it can duplicate rows to reach the requested count. The optional temperature sampler (`--enable_temperature_sampler`) is a third, independent mechanism that rebalances the multi-task mix by task frequency.

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
| `--save_steps` / `--eval_steps` / `--logging_steps` | checkpoint / eval / log cadence | `100 / 100 / 50` (the two Qwen2.5-VL-7B LoRA reference recipes); `100 / 100 / 20` (the other 19 launchers). The other rows in this table are the three Qwen2.5-VL-7B reference recipes' values — other families differ (e.g. `--use_flash_attention_2 false` in the six Gemma-4-31B and six MedGemma-27B launchers) |
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

## Loss masking (completion-only)

Which tokens count toward the training loss is decided per model family in the collate functions:

- **Qwen collates** (shared by the Qwen2.5-VL and Qwen3-VL drivers) are **completion-only by default**: only each assistant response and its closing turn marker stay in the loss; padding, image tokens, and the entire user prompt are set to the `-100` ignore index.
- **Gemma-family collates** (MedGemma, Gemma 4) mask only padding and image tokens by default, so the user prompt *is* part of the language-modeling loss — the same objective as Google's [official MedGemma fine-tuning notebook](https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb), whose collator masks exactly those tokens.

Setting `MEDVISION_SFT_COMPLETION_ONLY=1` switches the Gemma-family collates to the same completion-only objective (the Qwen collates already mask, and ignore the flag). It applies to LoRA and full-parameter training alike, and raises at the first batch — rather than silently mis-masking — if a checkpoint's chat template lacks the expected Gemma turn markers.

:::{warning}
`train/loss` is **not comparable across this flag**: with masking on, the loss averages over only the response tokens (roughly 15 % of the sequence — all of them the hard answer tokens) instead of being diluted by near-identical prompt boilerplate, so the reported value jumps up. That is the flag working, not a regression.

And because it is an environment variable, a `MEDVISION_SFT_COMPLETION_ONLY=1` left exported in the shell silently turns the next *baseline* Gemma run into a completion-only run. Launch the `__cmplLoss` script variants (which export this flag) in a fresh shell, or `unset` the variable before a baseline run; a sudden `train/loss` jump is the tell.
:::

For MedVision's long-CoT targets the expected downstream-accuracy effect is roughly neutral; the motivation is objective consistency with the Qwen family rather than an accuracy gain. The evidence review lives in the repository at `docs/literature-review__loss-masking-in-SFT.md`.

## Scaling full-parameter SFT to 27B and beyond

`script/sft/` ships full-parameter CoT recipes for MedGemma-27B, Gemma-4-31B, and Qwen3.5/Qwen3.6-27B in two memory recipes per family:

| Script variant | Recipe | Checkpoints | Hardware |
|---|---|---|---|
| `train__fullSFT-CoT__<Model>__...__512x512.sh` | anti-OOM: pure bf16 + 8-bit AdamW | weights-only (~54 GB at 27B) | 4× 80 GB |
| `...__4xGPU-140G-fp32master.sh` | fp32 master weights + fused fp32 AdamW | fully resumable (~160 GB at 27B) | 4× 140 GB-class |

`__cmplLoss` variants of either recipe additionally export the loss-masking flag above.

The standard AMP setup (bf16 compute, fp32 master weights, fused fp32 AdamW) carries ~121.5 GB of fixed per-GPU state at 27B across 4 ranks — far beyond 80 GB cards. The **anti-OOM recipe** trains bf16-native instead: `MEDVISION_SFT_PURE_BF16=1` removes the fp32 masters (the launch also omits `--mixed_precision=bf16`), `MEDVISION_SFT_OPTIM=adamw_bnb_8bit` shrinks optimizer state 8×, and the fixed cost drops to ~40.5 GB. The learning rate is raised to `4e-5` so AdamW updates clear bf16's ~0.4 % rounding resolution.

:::{warning}
The anti-OOM recipe is **not fully resumable**. Its 8-bit optimizer state cannot be gathered by FSDP's `FULL_STATE_DICT`, so checkpoints are weights-only (`MEDVISION_SFT_SAVE_ONLY_MODEL=1`): any restart — preemption, pod loss, a deliberate stop — discards the optimizer moments and LR-schedule position and warm-restarts from the last saved weights.
:::

**Prefer the fp32-master recipe whenever 140 GB-class GPUs are available**; reserve the anti-OOM recipe for 80 GB pods. It keeps standard AMP numerics and fully resumable checkpoints, and fits 4 ranks — validated at 27B (Qwen3.6: post-FSDP-wrap ≈ 38 GiB/rank). At 31B its worst-case fixed cost (~139.5 GB/rank) sits at the edge of the budget: treat Gemma-4-31B on 4 GPUs as unvalidated and watch the memory probes on the first run.

The knobs are environment variables exported by the launcher scripts (not argparse flags). Every knob defaults to the legacy behavior, so the 7B pipelines are untouched:

| Env var | Default | Effect when set |
|---|---|---|
| `MEDVISION_SFT_PURE_BF16=1` | off | disable AMP (`bf16=False`): no fp32 masters, no bf16 `_mp_shard`; the launch must also omit `--mixed_precision=bf16` |
| `MEDVISION_SFT_OPTIM` | `adamw_torch_fused` | any `SFTConfig.optim` value; the anti-OOM recipe uses `adamw_bnb_8bit`, whose state lives inside the torch allocator (unlike `paged_adamw_8bit`, whose UVM pages the allocator cannot evict) |
| `MEDVISION_SFT_SAVE_ONLY_MODEL=1` | off | weights-only checkpoints; required with 8-bit optimizers under FSDP `FULL_STATE_DICT` |
| `MEDVISION_SFT_LR` | `2e-5` | full-FT learning-rate override |
| `MEDVISION_SFT_SYNC_EACH_BATCH=1` | off | neutralise `no_sync` during gradient accumulation so gradients reduce-scatter every micro-batch and accumulate *sharded*; without it FSDP accumulates full unsharded gradients and OOMs in the first backward at 27B+ |
| `MEDVISION_SFT_ATTN` | follows `--use_flash_attention_2` | attention-implementation override (e.g. `sdpa` for families whose FlashAttention wheel is unvalidated) |
| `MEDVISION_SFT_USE_LIGER=1` | off | Liger kernels; the fused cross-entropy removes the vocab-sized logits spike (requires `pip install liger-kernel`) |
| `MEDVISION_SFT_MEMPROBE=1` | off | log per-rank memory after FSDP wrap and after step 1 |
| `MEDVISION_SFT_MEMSNAPSHOT=1` | off | on OOM, dump a per-rank CUDA allocator snapshot into the checkpoint dir |

Resuming a full-parameter run is FSDP-aware: the entry points detect the last checkpoint before building the trainer and load its weights through the same sharded `from_pretrained` path as a fresh start, skipping the Trainer's own checkpoint loader (which would all-gather the full unsharded model on every rank — an OOM at 27B+). This happens automatically with `--resume_from_checkpoint true`, and a checkpoint moves freely between pods with different GPU memory as long as the effective batch (world size × accumulation × per-device batch) stays the same.

:::{tip}
`MEDVISION_SFT_MEMPROBE=1` costs two log lines. On any new model-size/GPU combination, check the post-wrap figure first: expect roughly `params × 2 bytes / world_size` for pure bf16, or that plus the fp32 masters under AMP — a full-model-sized figure means FSDP sharding did not engage.
:::

Background for these choices is collected in the repository at `docs/literature-review__anti-OOM-fullFT-techniques.md`.

## Merging and pushing (LoRA only)

The LoRA drivers can merge the trained adapter back into the base model and push either artifact to the Hub:

| Flag | Effect |
|---|---|
| `--merge_model true` | after training, merge the final adapter into the base weights |
| `--merged_model_dir` / `--merged_model_hf` | local output dir and Hub repo name for the merged model |
| `--push_merged_model true` | upload the merged model to the Hub |
| `--push_LoRA true` | upload the LoRA adapter after each save |
| `--merge_only true` | skip training; merge and push the last existing checkpoint |

The full-parameter driver writes complete model checkpoints directly, so the **merge** options above (`--merge_model`, `--merged_model_*`, `--merge_only`) do not apply to it — but it does reuse `--push_LoRA` as the push-to-Hub switch for the full checkpoint. Its `--lora_checkpoint_dir` argument is reinterpreted internally as the plain checkpoint directory.

:::{warning}
Merging a LoRA adapter into the base weights can slightly degrade measurement accuracy versus serving base + adapter. Keep the unmerged adapter around if you care about the last decimal.
:::

## Entry points and other model families

CoT drivers ship for four families, each with a LoRA and a full-parameter module under `medvision_bm.sft`
(two further drivers sit outside this table: `train__SFT__qwen2_5_vl`, the non-CoT variant, and
`train__qwen25vl_AD_TL_tooluse`, the tool-use/function-calling driver whose prompts live in `sft_prompts_tooluse`):

| Family (`--model_family_name`) | LoRA driver | Full-parameter driver |
|---|---|---|
| `qwen25vl` | `train__SFT-CoT__qwen2_5_vl` | `train__fullFT-CoT__qwen2_5_vl` |
| `qwen3vl` (Qwen3-VL / Qwen3.5 / Qwen3.6) | `train__SFT-CoT__qwen3vl` | `train__fullFT-CoT__qwen3vl` |
| `gemma4` | `train__SFT-CoT__gemma4` | `train__fullFT-CoT__gemma4` |
| `medgemma` | `train__SFT-CoT__medgemma` | `train__fullFT-CoT__medgemma` |

They share the preparation, sampler, and trainer plumbing in `medvision_bm.sft.sft_utils` (`prepare_dataset`, `prepare_trainer`, `prepare_trainer_fullFT`) and differ mainly in their collate function and chat template. Extending the same recipe to a new family follows the identical two-recipe pattern: a `train__SFT-CoT__<family>` / `train__fullFT-CoT__<family>` module reusing these helpers, the matching `--model_family_name`, the family's decoder-layer class in `--fsdp_transformer_layer_cls_to_wrap`, and the right `--lmms_eval_opt_deps` for `env_setup`. See [Add a model](../extending/add-a-model.md) for that walkthrough.

## See also

- [Reinforcement fine-tuning (RFT)](rft.md) — GRPO-based training on the same tasks.
- [CLI reference](../reference/cli.md) — the full flag list for each entry point.
- [API reference](../reference/api/index.md) — `sft_utils` functions (`prepare_dataset`, `prepare_trainer`, `prepare_trainer_fullFT`).
