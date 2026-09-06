# MedVision SFT — launcher catalogue

The repository ships one shell recipe per (model, FT mode, hardware profile, loss variant) under `script/sft/`.
They are **provenance, not runnable instructions** here: each creates a conda environment, force-installs
packages and hard-codes an absolute working directory. Use `scripts/sft_launcher_template.sh` (which reproduces
their structure with the same variable names) or the plain `python -m` commands in `workflows.md`.

## Naming scheme

```
train__{SFT|fullSFT}-CoT__{model}__D110k-AD5.5k-TL5.5k[__512x512][__4xGPU-140G-fp32master][__cmplLoss].sh
```

| Token | Meaning |
| --- | --- |
| `SFT-CoT` | QLoRA adapter training (`train__SFT-CoT__<family>` entry point, plain DDP). |
| `fullSFT-CoT` | Full-parameter training (`train__fullFT-CoT__<family>` entry point, FSDP FULL_SHARD). |
| `D110k-AD5.5k-TL5.5k` | The data mix: 110 000 detection + 5 500 A/D + 5 500 T/L per-task train caps, global cap 121 000. |
| `__512x512` | `--new_shape_hw 512 512`. Absent = native resolution. |
| `__4xGPU-140G-fp32master` | The 140 GB-class full-FT recipe: `--mixed_precision=bf16` (fp32 master weights) and fully resumable checkpoints. Absent on a full-FT script means the anti-OOM pure-bf16 80 GB recipe. |
| `__cmplLoss` | `MEDVISION_SFT_COMPLETION_ONLY=1` — completion-only loss masking for the Gemma families. |

## Catalogue (21 top-level launchers)

Shared by all of them: `CUDA_VISIBLE_DEVICES=0,1,2,3` with `--num_processes=4`; the same three SFT task lists;
`train_sample_limit=121000`, `val_sample_limit=200`, per-task `5500 / 110000 / 5500` train and `45 / 105 / 50`
validation; `resume_from_checkpoint=true`; `gradient_checkpointing=true`; `enable_temperature_sampler=true`
with `temperature_sampler_T=5`; `MedVision_PLANNER_VERSION=1.0.0` with `MedVision_ACK_RELEASE=1.1.1`;
`save_steps=100`, `eval_steps=100`. Every launcher runs phase A through `tee` into
`${lora_checkpoint_dir}/prepare_dataset.log`, reads the reported prepared-dataset directory back from that log,
aborts before the GPU launch if it is missing, and passes it to phase B as `--prepared_ds_dir` (blocks 14-18
below).

### QLoRA (`train__SFT-CoT__*`)

| Launcher | Family key | Base model | Resize | epoch | bs x accum | Env knobs |
| --- | --- | --- | --- | --- | --- | --- |
| `train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh` | `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | **native** | 10 | 4 x 8 | none; `env_setup --lmms_eval_opt_deps qwen2_5_vl` |
| `train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh` | `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | 512x512 | 10 | 4 x 8 | none (the MedVision-V0 SFT recipe) |
| `train__SFT-CoT__Qwen3.5-27B__D110k-AD5.5k-TL5.5k__512x512.sh` | `qwen3vl` | `Qwen/Qwen3.5-27B` | 512x512 | 10 | 2 x 16 | `MEDVISION_SFT_ATTN=sdpa`; transformers re-pinned to 5.5.0; `--lmms_eval_opt_deps qwen3_vl` |
| `train__SFT-CoT__Qwen3.6-27B__D110k-AD5.5k-TL5.5k__512x512.sh` | `qwen3vl` | `Qwen/Qwen3.6-27B` | 512x512 | 10 | 2 x 16 | same as Qwen3.5 |
| `train__SFT-CoT__Gemma-4-31B-it__D110k-AD5.5k-TL5.5k__512x512.sh` | `gemma4` | `google/gemma-4-31B-it` | 512x512 | 10 | 1 x 32 | `MEDVISION_SFT_ATTN=sdpa`, `MEDVISION_SFT_OPTIM=paged_adamw_8bit`; transformers 5.5.0 |
| `..._Gemma-4-31B-it__..._512x512__cmplLoss.sh` | `gemma4` | same | 512x512 | 10 | 1 x 32 | above + `MEDVISION_SFT_COMPLETION_ONLY=1` |
| `train__SFT-CoT__MedGemma-27B-it__D110k-AD5.5k-TL5.5k__512x512.sh` | `medgemma` | `google/medgemma-27b-it` | 512x512 | 10 | 1 x 32 | `MEDVISION_SFT_OPTIM=paged_adamw_8bit`; transformers 4.54.0 |
| `..._MedGemma-27B-it__..._512x512__cmplLoss.sh` | `medgemma` | same | 512x512 | 10 | 1 x 32 | above + `MEDVISION_SFT_COMPLETION_ONLY=1` |

QLoRA launchers pass **no** FSDP flags: the 4-bit base model is replicated per rank (DDP).

### Full parameter (`train__fullSFT-CoT__*`, entry point `train__fullFT-CoT__*`)

All use `epoch=3`; `save_total_limit=3` in the 27-31B recipes and `10` in the 7B recipe; `--use_fsdp --fsdp_sharding_strategy FULL_SHARD
--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_state_dict_type FULL_STATE_DICT
--fsdp_offload_params false --fsdp_cpu_ram_efficient_loading true --fsdp_sync_module_states true`,
plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `save_total_limit=3` and
`MEDVISION_SFT_SYNC_EACH_BATCH=1` are the **27-31B** recipes only — the 7B full-SFT recipe sets
`save_total_limit=10` and exports no `MEDVISION_SFT_*` variable.

| Launcher | Family | FSDP layer class | Profile | bs x accum | Distinguishing knobs |
| --- | --- | --- | --- | --- | --- |
| `train__fullSFT-CoT__Qwen2.5VL7B__..._512x512.sh` | `qwen25vl` | `Qwen2_5_VLDecoderLayer` | 4 GPU | 8 x 8 | baseline 7B full-FT: `--mixed_precision=bf16`, no `MEDVISION_SFT_*` knobs |
| `train__fullSFT-CoT__Qwen3.5-27B__..._512x512.sh` | `qwen3vl` | `Qwen3_5DecoderLayer` | 4x80 GB | 1 x 64 | pure-bf16 anti-OOM: `PURE_BF16=1`, `LR=4e-5`, `OPTIM=adamw_bnb_8bit`, `SAVE_ONLY_MODEL=1`, `ATTN=sdpa`, `MEMPROBE`, `MEMSNAPSHOT` |
| `train__fullSFT-CoT__Qwen3.5-27B__..._512x512__4xGPU-140G-fp32master.sh` | `qwen3vl` | `Qwen3_5DecoderLayer` | 4x140 GB | 1 x 64 | `--mixed_precision=bf16`, `ATTN=sdpa`, `MEMPROBE`; resumable checkpoints |
| `train__fullSFT-CoT__Qwen3.6-27B__..._512x512.sh` | `qwen3vl` | `Qwen3_5DecoderLayer` | 4x80 GB | 1 x 64 | as Qwen3.5 pure-bf16 |
| `train__fullSFT-CoT__Qwen3.6-27B__..._512x512__4xGPU-140G-fp32master.sh` | `qwen3vl` | `Qwen3_5DecoderLayer` | 4x140 GB | 1 x 64 | as Qwen3.5 fp32-master |
| `train__fullSFT-CoT__Gemma-4-31B-it__..._512x512.sh` | `gemma4` | `Gemma4TextDecoderLayer` | 4x80 GB | 1 x 64 | pure-bf16 set + `ATTN=sdpa`; liger deliberately NOT enabled (unverified on Gemma 4) |
| `train__fullSFT-CoT__Gemma-4-31B-it__..._512x512__cmplLoss.sh` | `gemma4` | `Gemma4TextDecoderLayer` | 4x80 GB | 1 x 64 | above + `COMPLETION_ONLY=1` |
| `train__fullSFT-CoT__Gemma-4-31B-it__..._512x512__4xGPU-140G-fp32master.sh` | `gemma4` | `Gemma4TextDecoderLayer` | 4x140 GB | 1 x 64 | `--mixed_precision=bf16`, `ATTN=sdpa`, `MEMPROBE` |
| `..._Gemma-4-31B-it__..._4xGPU-140G-fp32master__cmplLoss.sh` | `gemma4` | `Gemma4TextDecoderLayer` | 4x140 GB | 1 x 64 | above + `COMPLETION_ONLY=1` |
| `train__fullSFT-CoT__MedGemma-27B-it__..._512x512.sh` | `medgemma` | `Gemma3DecoderLayer` | 4x80 GB | 1 x 64 | pure-bf16 set + `USE_LIGER=1` (installs `liger-kernel>=0.5.4` with `--no-deps`) |
| `..._MedGemma-27B-it__..._512x512__cmplLoss.sh` | `medgemma` | `Gemma3DecoderLayer` | 4x80 GB | 1 x 64 | above + `COMPLETION_ONLY=1` |
| `..._MedGemma-27B-it__..._4xGPU-140G-fp32master.sh` | `medgemma` | `Gemma3DecoderLayer` | 4x140 GB | 1 x 64 | `--mixed_precision=bf16`, `MEMPROBE` |
| `..._MedGemma-27B-it__..._4xGPU-140G-fp32master__cmplLoss.sh` | `medgemma` | `Gemma3DecoderLayer` | 4x140 GB | 1 x 64 | above + `COMPLETION_ONLY=1` |

`script/sft/v1.0.0/medvision-sft/` holds the recipes behind the six released checkpoints
(`MedVision__SFT-m__qwen25vl-{7b,32b}__{detect,TL,AD}`). The four scripts directly under
`script/sft/v1.0.0/` are other legacy single-task/multi-task runs
(`train__SFT-CoT-AD__Qwen2.5-VL.sh`, `train__SFT-CoT-TL__Qwen2.5-VL.sh`, `train__SFT__Qwen2.5-VL.sh`,
`train__SFT__MedGemma.sh`), and `script/sft/dev_medvision-sft/` holds development scripts. Neither is a current
recipe; prefer the top-level ones. No launcher exists for the tool-use entry point in the public tree.

## Anatomy of a launcher (the blocks the template reproduces)

1. **Conda env + CUDA toolkit.** `ENV_NAME="sft-<family>"`, created once, `conda install -c nvidia
   cuda-toolkit=12.4`. Both Qwen2.5-VL QLoRA recipes also carry `conda config --set solver classic` as a workaround
   for a broken conda solver.
2. **`HF_TOKEN` sanitising.** `export HF_TOKEN="$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')"` — pod-injected
   secrets carry a trailing newline that corrupts the Authorization header (401 on gated models).
3. **Annotation version.** `export MedVision_PLANNER_VERSION='1.0.0'` and `MedVision_ACK_RELEASE='1.1.1'`.
4. **Paths.** `benchmark_dir`, `train_sft_dir="${benchmark_dir}/SFT"`, `data_dir="${benchmark_dir}/Data"`.
5. **Task lists.** The three `tasks_list_json_path_{AD,detect,TL}` paths, annotated with the pool sizes.
6. **Model block.** `model_family_name`, `base_model_hf`, `run_name`, `lora_checkpoint_dir` (with a trailing
   `${run_name}` subfolder so pushed LoRA repos get distinct names), `merged_model_hf`, `merged_model_dir`.
   Full-FT scripts add `transformers_version` and `fsdp_layer_cls`, and note that `--lora_checkpoint_dir` is
   remapped to `checkpoint_dir` internally.
7. **Training block.** `epoch`, `save_steps`, `eval_steps`, `logging_steps`, `save_total_limit`,
   `use_flash_attention_2`, `num_workers_concat_datasets`, `num_workers_format_dataset`,
   `dataloader_num_workers`, `dataloader_pin_memory`.
8. **Sample-limit block** with the standing warning: unset limits mean the full dataset, unset per-task
   validation limits fall back to 100, and a limit of `0` is rejected — drop a task by commenting out its task
   list instead.
9. **Resource block.** `gradient_checkpointing`, `per_device_train_batch_size`, `per_device_eval_batch_size`,
   `gradient_accumulation_steps`, annotated with `effective_batch_size = bs * accum * num_gpus`.
10. **Merge/push block.** `push_LoRA`, `push_merged_model`, `merge_model`, `merge_only`.
11. **wandb block.** `wandb_resume`, `wandb_dir`, `wandb_project`, `wandb_run_name`, `wandb_run_id`
    (unique per project; reuse it to continue an existing chart).
12. **Node-local wheel build.** The package is built into a temporary directory on node-local disk and installed
    under `flock`, because `setuptools`' `build_py` caches created directories in a process-global memo and a
    shared network filesystem can make a build subdirectory transiently vanish, after which the cache refuses to
    recreate it and a later file copy dies with `could not create '...': No such file or directory`.
13. **`env_setup` + `protobuf` pin.** `python -m medvision_bm.sft.env_setup --data_dir <data_dir>
    [--lmms_eval_opt_deps ...]` followed by `python -m pip install "protobuf==6.33.0"`, because `env_setup`
    leaves a protobuf that `wandb>=0.21`'s generated stubs reject (`cannot import name 'Imports' from
    wandb.proto`), which breaks the `trl.SFTTrainer` import at train time. Transformers-5.x families re-pin
    `transformers==<version>` here too, since `env_setup` force-installs 4.54.0 last.
14. **Dataset-processing config block.** `skip_process_dataset`, `save_processed_img_to_disk`,
    an optional `prepared_ds_dir` override (forwarded to phase A when set), the temperature-sampler settings
    (noted as training-only), then `mkdir -p "${lora_checkpoint_dir}"` and
    `prep_log="${lora_checkpoint_dir}/prepare_dataset.log"`.
15. **Phase A**: `python -m medvision_bm.sft.train__... --process_dataset_only true
    ${prepared_ds_dir:+--prepared_ds_dir ${prepared_ds_dir}} ... --new_shape_hw 512 512 2>&1 | tee "${prep_log}"`.
    Under the scripts' `set -euo pipefail` a failed phase A aborts here.
16. **Prepared-directory capture**:
    `prepared_ds_dir="$(sed -n "s/.*Prepared dataset saved at '\([^']*\)'.*/\1/p" "${prep_log}" | tail -n 1)"`,
    followed by `exit 1` when the value is empty or not a directory, so a failed or stale preparation never
    reaches the GPU launch.
17. **Self-heal probe** (the twelve transformers-5.x scripts — Gemma-4 and Qwen3.5/3.6, QLoRA and full-FT alike; the QLoRA variant also imports `peft`): after phase A, `python -c "import transformers; from trl import
    SFTTrainer"`; on failure, a surgical `pip install --upgrade "transformers==<pin>" huggingface_hub
    "protobuf==6.33.0"`. This exists because phase A reinstalls `medvision_ds`, whose `huggingface_hub==0.36.0`
    pin drags hub below the transformers-5.x floor on **every** preparation run. A blanket
    `--force-reinstall transformers` is explicitly avoided: re-resolving its dependency tree can pull an `fsspec`
    newer than `datasets`' cap, which the next preparation run then downgrades mid-process.
18. **Phase B**: `CUDA_VISIBLE_DEVICES=... accelerate launch --num_processes=4 --main_process_port=<port>
    [--mixed_precision=bf16] [FSDP flags] -m medvision_bm.sft.train__... --skip_process_dataset true
    --prepared_ds_dir ${prepared_ds_dir} --process_dataset_only false ... --enable_temperature_sampler ...
    --new_shape_hw 512 512`. With the directory given, rank 0 loads it directly instead of re-running the
    load+split stage to recompute the name.
19. `conda deactivate` (the env removal line is left commented out).
