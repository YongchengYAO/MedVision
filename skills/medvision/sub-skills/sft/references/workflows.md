# MedVision SFT — end-to-end workflows

Placeholders: `<benchmark_dir>` (working root), `<data_dir>` (dataset cache, usually `<benchmark_dir>/Data`),
`<sft_dir>` (checkpoints/logs, usually `<benchmark_dir>/SFT`), `<tasks_dir>` (folder with the task-list JSONs),
`<run_name>` (run identifier).

---

## 0. Decide the run before touching anything

| Question | Consequence |
| --- | --- |
| Which model family? | Picks the entry point AND the prepared dataset — a dataset is not portable between families. |
| LoRA or full parameters? | LoRA = `train__SFT-CoT__*` + plain DDP; full = `train__fullFT-CoT__*` + FSDP + `--use_fsdp` flags. |
| Which resize? | `--new_shape_hw 512 512` is the published recipe; omitting it trains at native resolution and changes the prompt's pixel size. |
| Which tasks? | Each task list you pass adds a task; omit a list to drop the task entirely. |
| Which limits? | Per-task caps + a global cap; see `data-preparation.md`. Run `scripts/check_sample_limits.py` first. |
| Completion-only loss? | Qwen collates always mask non-assistant turns; Gemma/MedGemma need `MEDVISION_SFT_COMPLETION_ONLY=1`. |

---

## 1. Environment

Install the SFT stack for the family (see `../../environment-setup/SKILL.md` for the pins and install order).
The repository's own recipes call:

```
python -m medvision_bm.sft.env_setup --data_dir <data_dir> [--lmms_eval_opt_deps qwen2_5_vl|qwen3_vl]
python -m pip install "protobuf==6.33.0"
```

`sft.env_setup` installs the vendored `lmms_eval`, the basic package set (`datasets==3.6.0`, `numpy==1.26.4`,
`protobuf==3.20`, `wandb==0.21.4`, `trl==0.19.1`, `huggingface_hub==0.36.0`, plus bitsandbytes / peft / nibabel /
scipy / Pillow / accelerate), a flash-attention + torch bundle, **`transformers==4.54.0` last**, and finally
`medvision_ds`. Two consequences you must handle:

- The `protobuf==3.20` it leaves is incompatible with `wandb>=0.21`'s generated stubs, which breaks the
  `trl.SFTTrainer` import at train time. Re-pin `protobuf==6.33.0` afterwards (the frozen SFT requirements use it).
- Families that need transformers 5.x (Gemma 4, Qwen3-VL) must **re-pin transformers after `env_setup`**, because
  `env_setup` force-installs 4.54.0 at the end.

`env_setup` mutates the active environment. Do not run it on a user's shared environment without asking; the
bundled launcher template only runs it when `RUN_ENV_SETUP=1`.

Log in once if you intend to log or push: `wandb login`, `hf auth login`. Sanitise a pod-injected token first —
a trailing newline in `HF_TOKEN` corrupts the Authorization header and yields 401 on gated models:

```
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '[:space:]')"
```

Set the annotation version before either phase:

```
export MedVision_PLANNER_VERSION=1.0.0     # the published recipe
export MedVision_ACK_RELEASE=<release>     # only needed when pinning below the newest release
```

---

## 2. Phase A — build the prepared dataset (CPU)

```
python -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --process_dataset_only true \
    --skip_process_dataset false \
    --save_processed_img_to_disk true \
    --model_family_name qwen25vl \
    --base_model_hf Qwen/Qwen2.5-VL-7B-Instruct \
    --data_dir <data_dir> \
    --tasks_list_json_path_AD     <tasks_dir>/tasks_MedVision-AD__train_SFT.json \
    --tasks_list_json_path_detect <tasks_dir>/tasks_MedVision-detect__train_SFT.json \
    --tasks_list_json_path_TL     <tasks_dir>/tasks_MedVision-TL__train_SFT.json \
    --train_sample_limit_task_AD 5500 --val_sample_limit_task_AD 45 \
    --train_sample_limit_task_Detection 110000 --val_sample_limit_task_Detection 105 \
    --train_sample_limit_task_TL 5500 --val_sample_limit_task_TL 50 \
    --train_sample_limit 121000 --val_sample_limit 200 \
    --num_workers_concat_datasets 4 --num_workers_format_dataset 32 \
    --new_shape_hw 512 512
```

Or, equivalently, `bash scripts/sft_launcher_template.sh --phase A ...`.

What it does, in order: loads and splits every requested task, resolves the prepared-dataset directory name
from the **true** split sizes, formats each task into chat `messages` with CoT targets (writing one PNG per slice
when `--save_processed_img_to_disk true`), tags rows with `__task_name`, concatenates, applies the global limits,
and `save_to_disk`s a `DatasetDict{train, validation}`. It prints
`Data processing completed. Prepared dataset saved at '<dir>'.` — **pass that path to phase B as
`--prepared_ds_dir`.** The repository launchers do this for you: phase A runs as
`python -m ... 2>&1 | tee "${lora_checkpoint_dir}/prepare_dataset.log"`, the directory is read back with
`sed -n "s/.*Prepared dataset saved at '\([^']*\)'.*/\1/p"`, the script exits before the GPU launch if the line
is missing or the directory does not exist, and phase B gets `--prepared_ds_dir ${prepared_ds_dir}`.
`scripts/sft_launcher_template.sh` reproduces the same hand-off.

Why it is a separate phase: preparation is CPU- and IO-bound and can run for hours, while the other ranks sit in
a barrier waiting for rank 0 — long enough to blow the distributed process-group timeout (30 minutes by default;
only the tool-use entry point actually raises it, see `troubleshooting.md` §6).

Notes:
- Only rank 0 prepares; other ranks wait at a barrier and then receive the resolved directory by broadcast.
- `--skip_process_dataset true` **without** `--prepared_ds_dir` still performs the load+split stage (every
  config is loaded and the volume-grouped split recomputed on rank 0) so the **default** directory name resolves
  identically; it only skips formatting and saving. That is minutes on the published task lists and far longer
  on full-data lists, and it repeats on every restart.
- `--prepared_ds_dir <dir>` bypasses the naming convention entirely (taken as-is on every rank) and, together
  with `--skip_process_dataset true`, skips the load+split stage too. This is what phase B should receive.

**Cost:** the detection task list covers whole source datasets; expect large downloads and one PNG per training
slice under `tmp_prepared_png/` next to each NIfTI. Start with small `--train_sample_limit_task_*` values.

---

## 3. Verify the prepared dataset

```
python scripts/inspect_prepared_dataset.py --prepared-ds-dir <prepared_ds_dir> --check-images 50
```

Confirm: both splits exist; `messages` has 2 turns (user with one image + prompt, assistant with the CoT target)
— 5 turns for tool-use; `__task_name` is present with the per-task counts you expect (the temperature sampler
needs it); `image_file_png` is present when you asked for the PNG cache and the files exist.

---

## 4. Phase B — train (GPU)

LoRA (DDP across the visible GPUs):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --main_process_port=29502 --mixed_precision=bf16 \
    -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl \
    --skip_process_dataset true --process_dataset_only false \
    --prepared_ds_dir <dir printed by phase A> \
    <the same model / data / limit / resize flags as phase A> \
    --run_name <run_name> --lora_checkpoint_dir <sft_dir>/<run_name>/checkpoints/<run_name> \
    --epoch 10 --save_steps 100 --eval_steps 100 --logging_steps 50 --save_total_limit 10 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 \
    --gradient_checkpointing true --use_flash_attention_2 true --dataloader_pin_memory true \
    --enable_temperature_sampler true --temperature_sampler_T 5 \
    --resume_from_checkpoint true --merge_model true --merged_model_hf <repo> \
    --wandb_project <project> --wandb_run_name <run_name> --wandb_run_id <unique id> \
    --new_shape_hw 512 512
```

Full parameters (FSDP FULL_SHARD): same command with the `train__fullFT-CoT__*` module plus

```
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap <DecoderLayerClass> \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_offload_params false \
    --fsdp_cpu_ram_efficient_loading true \
    --fsdp_sync_module_states true
```

`<DecoderLayerClass>` must match the class name in the **installed** transformers for that checkpoint
(`Gemma4TextDecoderLayer`, `Gemma3DecoderLayer`, `Qwen3_5DecoderLayer`, ...). Verify once with
`python -c "from transformers import AutoModelForImageTextToText as M; m=M.from_pretrained('<base_model_hf>');
print(sorted({type(x).__name__ for x in m.modules() if 'DecoderLayer' in type(x).__name__}))"`.

`--num_processes` must equal the number of GPUs in `CUDA_VISIBLE_DEVICES`; the effective batch size is
`per_device_train_batch_size * gradient_accumulation_steps * num_gpus`. Give every concurrent run a distinct
`--main_process_port`.

Memory knobs for 27-31B full-FT live in `training-configuration.md`; the two published recipes are
"140 GB fp32-master" (resumable) and "80 GB pure bf16" (weights-only checkpoints).

Useful before a long full-FT run:

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"   # DeepSpeed import check
export MEDVISION_SFT_MEMPROBE=1                                        # 2 memory lines per rank
```

---

## 5. Monitor

- Weights & Biases: `wandb_project` / `wandb_run_name` / `wandb_run_id` (`WANDB_*` env vars are set for you by
  `parse_validate_args_multiTask`). `wandb_resume=allow` continues an existing `wandb_run_id`; use a fresh id for
  a fresh chart. Offline debugging: `export WANDB_MODE=offline`.
- With `MEDVISION_SFT_MEMPROBE=1` the log carries `[MEMPROBE] train_begin(post-FSDP-wrap)` and
  `[MEMPROBE] after_step_1` lines with allocated / reserved / device-used per rank, whether gradient
  checkpointing engaged, the live FSDP mixed-precision policy, and the optimizer class and state dtypes.
- On resume the log prints the `[Resume]` block with world size, dataset size, steps per epoch and the
  recomputed `max_steps`.

---

## 6. Resume an interrupted run

Set `--resume_from_checkpoint true` and re-issue the **same** phase-B command (with
`--skip_process_dataset true` so the prepared dataset is reused, and `--prepared_ds_dir` so rank 0 does not
re-run the load+split stage on every restart). The entry point calls
`transformers.trainer_utils.get_last_checkpoint` on the checkpoint directory:

- No checkpoint -> prints "No valid checkpoint found ... Starting training from scratch" and trains normally.
- A checkpoint -> `train_resume_from_checkpoint` recomputes `max_steps` from the **current** dataset size, world
  size, batch size, accumulation and `--epoch`, broadcasts it to all ranks, restores `trainer_state.json`, and
  calls `trainer.train(resume_from_checkpoint=...)`.

Because `max_steps` is recomputed, changing the dataset size, GPU count or batch geometry between runs changes
the horizon. If the recomputed horizon is already satisfied, the run is marked finished and exits — raise
`--epoch` to continue.

Full-FT resume differs: the checkpoint is detected **before** the trainer is built and its weights are loaded
through `from_pretrained` (the FSDP-aware path), then `Trainer._load_from_checkpoint` is skipped. Do not try to
bolt a plain `--resume_from_checkpoint` onto a hand-rolled full-FT trainer; see `troubleshooting.md`.

---

## 7. Merge and push (LoRA only)

`--merge_model true` merges after training; `--merge_only true` skips training entirely and merges the adapter
saved directly at `--lora_checkpoint_dir` (not the newest `checkpoint-*` subdirectory). Both run on the main process only and are **CPU-only**: the base model is loaded in fp32 on CPU (so
the sub-bf16 LoRA delta is representable), the adapter is merged with `safe_merge=True` (raises on NaN/inf), the
processor is taken from the adapter directory, and the result is saved to `--merged_model_dir` and/or pushed to
`--merged_model_hf` in 2 GB shards as a **private** repo.

`--push_LoRA true` pushes the adapter after every save (the full-FT entry points reuse this flag to push the
trained model). `--push_merged_model true` requires `--merged_model_hf`; otherwise `merge_models` raises.

For a checkpoint that already exists on disk, `medvision_bm.utils.push_hf_model` (reference only — it uploads a
folder to the Hub with `HfApi.upload_folder`) takes `--repo_id`, `--folder_path` and optional `--message`.

Released MedVision SFT checkpoints are published under the `MedVision-SFT-Models` collection on the Hub
(`YongchengYAO/MedVision__SFT-m__qwen25vl-{7b,32b}__{detect,TL,AD}`), i.e. single-task Qwen2.5-VL SFT runs.

---

## 8. Hand off to evaluation

A merged LoRA model or a full-FT checkpoint is evaluated like any other local model — see
`../../benchmark-evaluation/SKILL.md`. Two things must match the training setup: the image resize
(`512x512` if you trained at 512x512) and, for models trained with a system prompt, its re-injection.
The RFT stage that continues from an SFT checkpoint is in `../../rft/SKILL.md`.

---

## 9. Tool-use variant

`python -m medvision_bm.sft.train__qwen25vl_AD_TL_tooluse` trains Qwen2.5-VL to emit a `<tool_call>` instead of
doing the arithmetic itself. Differences from the CoT entry points:

- **A/D and T/L only.** Detection is not in its task table; a `--tasks_list_json_path_detect` value is ignored
  for formatting (`scripts/sft_launcher_template.sh --mode tooluse` clears it for you).
- Targets are 5-turn conversations: system, user (image + prompt), assistant (`<think>` steps 1-2 then a
  `<tool_call>` carrying Python code), tool (the executed result), assistant (`<answer>`).
- The chat template is applied with `tools=[TOOL_DEF]` (`execute_python`), and loss masking is per-turn, so only
  assistant turns train.
- It uses the **full-parameter** trainer, so launch it with the `--use_fsdp` flags.
- The prepared-dataset directory gets a `-tooluse` suffix on the resize token.
- Tool results are produced by `medvision_bm.utils.tool_execution.safe_exec_python`: an AST allowlist that
  permits numeric/string constants, plain assignments, arithmetic, f-strings, tuples, `import math`, calls to
  `print/round/abs/min/max/len/sum`, and `math.<func>`. Attributes starting with `_`, subscripts, other imports
  and every other call target are rejected before execution; the code then runs with a restricted `__builtins__`.
  Failures return an `ERROR: ...` string instead of raising.

Only the public CLI above is documented here; the tool-use launcher scripts are not part of the public tree.

---

## 10. Where to stop and ask

- **No GPU** -> phase A, `scripts/check_sample_limits.py`, `scripts/inspect_prepared_dataset.py` and
  `DRY_RUN=1` previews are all you can do. Say so instead of launching phase B.
- **No disk budget** -> phase A writes PNGs and Arrow shards; full-FT checkpoints are ~190 GB each at 31B with
  fp32-master saves. Ask before consuming.
- **No Hub credentials / consent** -> never set `--push_LoRA` or `--push_merged_model`.
- **Environment not pinned** -> a mismatched `transformers` / `flash-attn` / `protobuf` fails at import or at
  step 0. Fix the environment first (`../../environment-setup/SKILL.md`), do not "just try".
