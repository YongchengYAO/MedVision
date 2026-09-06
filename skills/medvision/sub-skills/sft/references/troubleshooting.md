# MedVision SFT — troubleshooting

Symptom -> cause -> fix -> how to verify. Failures that are not SFT-specific (install, dataset package,
result trees) are in `../../../references/troubleshooting.md`.

---

## 1. Dataset preparation (phase A)

### `[Error] <flag>=0 is ambiguous: use -1 (or leave it unset) for no limit; ...`
`_get_sample_limit` rejects an explicit `0` on any sample-limit flag.
**Fix:** use `-1` (or omit the flag) for "no limit"; to drop a task, omit its `--tasks_list_json_path_*`.
Note the shell idiom `${var:--1}` in the recipes: an unset variable becomes `-1`, never `0`.
**Verify:** `python scripts/check_sample_limits.py ...` exits 2 and prints the same message.

### `AssertionError: At least one of --tasks_list_json_path_AD, --tasks_list_json_path_detect, or --tasks_list_json_path_TL must be provided.`
All three task-list flags are missing (or the shell expanded them to empty).
**Fix:** pass at least one. Check for typos and for an empty variable in a launcher.

### `AssertionError: limit_val_sample must be greater than 0`
A task is used but its resolved validation target is `<= 0`. This happens when both
`--val_sample_limit_task_<X>` and `--val_sample_limit_per_task` are negative.
**Fix:** leave them unset (fallback 100) or give a positive value.

### `AssertionError: MedVision_DATA_DIR environment variable must be set`
`setup_env_hf_medvision_ds` runs on the main process from `--data_dir`; this appears when the loader is called
outside the entry point, or when `--data_dir` points somewhere without `.downloaded_datasets.json`.
**Fix:** pass a real `--data_dir` that has been used for at least one dataset download.

### The loader hard-fails on the planner version
`medvision_ds` requires `MedVision_PLANNER_VERSION`; pinning below the newest release additionally requires
`MedVision_ACK_RELEASE`.
**Fix:** `export MedVision_PLANNER_VERSION=1.0.0` (and `MedVision_ACK_RELEASE=<release>` when pinning back)
before **both** phases. Keep it constant for the life of a study — annotation versions change the T/L sample set.
Details in `../../dataset-and-tasks/SKILL.md`.

### `RuntimeError: Some tasks failed to load` / per-task timeout
Each task load has a 120 s timeout and any failure aborts the batch. Usual causes: a task name absent from the
HF dataset, a partially downloaded cache, or a network problem.
**Fix:** read the per-task error list printed above the exception. If a name is wrong, fix the task list (a
detection list written with `_BoxCoordinate_` is auto-renamed to `_BoxSize_`; other namespace mistakes are not).
If a download raced, re-run — the loader forces single-threaded loading once it sees an undownloaded dataset.

### The process is killed during Arrow generation / `⚠️ High cgroup memory`
Detection plans are large and parallel task loading multiplies peak memory. On a container the ceiling is the
**cgroup** memory limit, not what `free` reports — an OOM kill can arrive with no traceback.
**Fix:** lower `--num_workers_concat_datasets` (2 is safe), lower `--num_workers_format_dataset`, and reduce
`--train_sample_limit_task_Detection`. The loader already warns above 80% cgroup memory.
**Verify:** re-run and watch for the "Using N workers for dataset loading (available CPUs: M)" line and the
memory warnings.

### `IndexError` from `save_to_disk`
`num_proc` exceeded a split's row count. The entry points already clamp
`num_proc = max(1, min(cgroup CPUs, smallest split size))`, so this only appears on hand-rolled calls with a
tiny smoke split.
**Fix:** clamp `num_proc` the same way.

### `ValueError: [Error] <name> is not recognised/supported.` from `get_resized_img_shape`
The `model_family_name` passed `check_model_supported` (it is in the `lmms_eval` registry) but has no branch in
the image-size dispatch used by the A/D and T/L prompt builders.
**Fix:** use one of the families with a branch, or add one — see `../../extending-models-and-tasks/SKILL.md`.
Detection-only runs are unaffected (detection prompts carry no pixel size).

### `[Error] Model '<name>' is not supported. Supported models are: [...]`
`check_model_supported` rejected `--model_family_name` before any data was touched.
**Fix:** use a registry key (the `vllm_` prefix may be omitted): `qwen25vl`, `qwen3vl`, `gemma4`, `medgemma`, ...
See `../../../references/model-roster.md`.

---

## 2. Prepared dataset / phase boundary

### Phase B cannot find the prepared dataset
`--skip_process_dataset true` makes phase B load `prepared_ds_dir` from disk; if the default name it derives
differs from the one phase A wrote, the load fails.
**Cause:** the two phases were given different limits, a different `--new_shape_hw`, a different
`--model_family_name`, or a different entry point (the non-CoT one uses `SFT_datasets`, tool-use appends
`-tooluse`). The name encodes all of them.
**Fix:** pass the directory phase A reported to phase B as `--prepared_ds_dir <dir>` (the repository launchers
and `scripts/sft_launcher_template.sh` do this automatically), or make the flag sets identical. The phase-A log
line `Prepared dataset saved at '<dir>'` is the authoritative path.
**Verify:** `python scripts/inspect_prepared_dataset.py --prepared-ds-dir <dir>`.

### Phase B spends minutes in "Loading dataset ..." before any training step
`--skip_process_dataset true` was given **without** `--prepared_ds_dir`, so rank 0 re-runs stage 1 (every
config loaded, the volume-grouped split recomputed) just to derive the default directory name, while the other
ranks wait at the barrier — and it repeats on every resume.
**Fix:** pass `--prepared_ds_dir <dir reported by phase A>`; with it, phase B loads nothing but the prepared
directory.
**Verify:** the phase-B log shows `[Info] Using user-specified prepared dataset directory: <dir>` and no
`[Info] Starting dataset preparation from ...` lines.

### `[Error] Could not resolve the prepared dataset directory from <checkpoint_dir>/prepare_dataset.log; aborting before the training launch.`
A repository launcher (or the bundled template) could not find the `Prepared dataset saved at '<dir>'` line in
the tee'd phase-A output, or the directory it names does not exist: phase A failed, was interrupted, or reported
a path from a different mount.
**Fix:** read `prepare_dataset.log` for the real error and re-run phase A with `--skip_process_dataset false`.
The abort is deliberate — it stops the GPU launch from starting on a missing or stale dataset.

### `--skip_process_dataset true` on a first run silently does nothing useful
Without `--prepared_ds_dir`, stage 1 (load + split) still runs so the default directory name resolves, but stage 2
(format + save) is skipped — then the load fails, or, worse, an older directory with the same name is picked up.
With `--prepared_ds_dir`, nothing is loaded and `load_from_disk` fails on the missing directory.
**Fix:** the first run of a configuration needs `--skip_process_dataset false`.

### Training uses stale or wrong images
`prepared_ds_dir` was reused across model families or resizes. The prompts contain the perceived image size and
the adjusted pixel size for whichever family prepared them; nothing detects the mismatch.
**Fix:** rebuild per family and per resize; keep the default naming so the identity is visible in the path.

### `Skipping example due to image loading error` warnings, or `All examples in this batch failed to process`
The collate could not open `image_file_png` (PNG cache deleted or `data_dir` moved) and, for
`image_file`, could not read the NIfTI either.
**Fix:** re-run phase A with `--save_processed_img_to_disk true` to regenerate the PNGs, or restore the data
directory. **Verify:** `python scripts/inspect_prepared_dataset.py --prepared-ds-dir <dir> --check-images 200`
(exit 1 when files are missing).

### The temperature sampler refuses to start
`ValueError: Temperature sampler requires column '__task_name' in train dataset.`
**Cause:** the prepared dataset predates task tagging, or a custom `--temperature_sampler_task_column`.
**Fix:** regenerate the dataset, or disable `--enable_temperature_sampler`.
With only one task present it logs a notice and falls back to standard sampling — that is not an error.

---

## 3. Environment and imports

### `cannot import name 'Imports' from wandb.proto ...` (usually while importing `trl.SFTTrainer`)
`sft.env_setup` installs `protobuf==3.20`, which `wandb>=0.21`'s generated stubs cannot use.
**Fix:** `python -m pip install "protobuf==6.33.0"` after every `env_setup` run (this matches the frozen SFT
requirement files). **Verify:** `python -c "import transformers; from trl import SFTTrainer"`.

### `ImportError` on `transformers` right after a successful phase A
Phase A reinstalls `medvision_ds`, whose exact pin `huggingface_hub==0.36.0` drags the hub below the floor
required by transformers 5.x — on **every** preparation run.
**Fix (surgical, as the recipes do):**
```
python -c "import transformers; from trl import SFTTrainer" || \
  python -m pip install --upgrade "transformers==<your pin>" huggingface_hub "protobuf==6.33.0"
```
Do **not** `pip install --force-reinstall transformers`: re-resolving its dependency tree can pull an `fsspec`
newer than `datasets`' cap, which the next preparation run downgrades mid-process (observed crash:
`ModuleNotFoundError: fsspec.implementations.chained`).
**Verify:** the probe command above exits 0 before you spend GPU time.

### Gemma 4 / Qwen3-VL fail to load the processor or model
`env_setup` force-installs `transformers==4.54.0` **last**; those families need 5.x.
**Fix:** re-pin transformers after `env_setup` (`5.5.0` in the recipes; the frozen `requirements_sft_gemma4.txt`
and `requirements_sft_qwen3.6vl.txt` also pin 5.5.0). Expect to disable FA2 as well: the shipped
`flash-attn` 2.7.3 wheel is built for torch 2.6, so the recipes set `use_flash_attention_2=false` **and**
`MEDVISION_SFT_ATTN=sdpa` (false alone would fall back to eager, which materialises O(seq^2) attention).

### `NameError` / `ImportError` mentioning DeepSpeed at import time
DeepSpeed runs a CUDA compatibility check on import even when it is not the backend.
**Fix:** `export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}"`.

### `ValueError: GPU does not support bfloat16, please use a GPU that supports bfloat16.`
Both trainers require compute capability >= 8. On a CPU-only host, `torch.cuda.get_device_capability()` raises
before this check.
**Fix:** run phase B on an Ampere-or-newer GPU. On CPU, only phase A and the bundled inspection scripts work.

---

## 4. CUDA OOM

Work through these in order; the first three are cheap.

1. `--gradient_checkpointing true` (required for full-FT at 7B+).
2. Lower `--per_device_train_batch_size` and raise `--gradient_accumulation_steps` to keep the effective batch
   size (`bs * accum * num_gpus`) constant.
3. Lower the resolution: `--new_shape_hw 512 512` instead of native (this changes the dataset, so phase A must
   be re-run and the prepared directory name changes).
4. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. Full-FT only, in order of impact:
   - `MEDVISION_SFT_SYNC_EACH_BATCH=1`. **This is the fix for an OOM in the very first backward pass.** HF
     `Trainer` wraps non-final micro-steps in `accelerator.no_sync()`, under which FSDP accumulates *full
     unsharded* gradients on every rank; they do not shrink with world size, so more GPUs does not help.
   - Move to the pure-bf16 recipe on 80 GB cards: omit `--mixed_precision` on the launch **and** set
     `MEDVISION_SFT_PURE_BF16=1` (plus `MEDVISION_SFT_LR=4e-5`, `MEDVISION_SFT_OPTIM=adamw_bnb_8bit`,
     `MEDVISION_SFT_SAVE_ONLY_MODEL=1`). `SFTConfig(bf16=False)` alone cannot win: the launch flag injects
     `ACCELERATE_MIXED_PRECISION`, which takes precedence.
   - `MEDVISION_SFT_USE_LIGER=1` when the vocabulary is large (fused linear cross-entropy avoids materialising
     seq x vocab logits); needs `pip install liger-kernel` and a supported architecture, otherwise transformers
     warns and silently runs the stock loss.
   - Use `adamw_bnb_8bit`, not `paged_adamw_8bit`: paged state lives in CUDA unified-memory pages that are
     resident on the device but outside the torch allocator, which cannot evict them, so the next micro-batch
     OOMs anyway.
   - `MEDVISION_SFT_BF16_GRADS=1` is **not** a fix under accelerate mixed precision — it raises
     "attempting to assign a gradient with dtype 'c10::BFloat16' to a tensor with dtype 'float'". Use pure bf16.
   - Do **not** enable `--fsdp_offload_params` on a memory-capped container: ~700 GB host RAM at 31B, and the
     ceiling is the cgroup limit.
6. Diagnose rather than guess: `MEDVISION_SFT_MEMPROBE=1` prints per-rank allocated/reserved/device-used after
   the FSDP wrap and after step 1, plus whether gradient checkpointing engaged, the FSDP mixed-precision policy,
   and the optimizer class and state dtypes (`uint8` = 8-bit optimizer actually engaged).
   `MEDVISION_SFT_MEMSNAPSHOT=1` dumps `oom_memsnap_rank<N>.pickle` into the checkpoint dir on OOM.
   **Sanity check:** post-wrap allocated should be about `params / world_size`. A full-model figure means FSDP
   did not shard; roughly double the bf16 shard means the fp32-master upcast is still active.
7. Note that a large slice of device memory (tens of GB) can sit outside your rank's allocator (sibling
   contexts, NCCL, VMM). The MEMPROBE `device_used - reserved` gap quantifies it, and it is what shrinks the
   usable ceiling below the nominal card size.

---

## 5. FSDP resume

### OOM while loading a checkpoint (not during training)
Under FSDP, `Trainer._load_from_checkpoint` routes a sharded weights-only checkpoint into transformers'
`load_sharded_checkpoint`, whose `model.state_dict()` all-gathers the **full unsharded model on every rank**.
**Fix:** the shipped full-FT and tool-use entry points already avoid this: they detect the checkpoint *before*
building the trainer, pass it as `prepare_trainer_fullFT(model_weights_from=...)` so weights arrive through the
FSDP-aware `from_pretrained` path, and then call
`train_resume_from_checkpoint(..., weights_preloaded=True)`, which replaces `_load_from_checkpoint` with a
no-op. Never bypass that ordering in a custom script.
**Verify:** the log shows `[Resume] Loading model weights from checkpoint: <dir>` followed by
`[Resume] Skipping Trainer._load_from_checkpoint: ...`.

### Resume exits immediately with "Training already satisfies (or exceeds) the new reduced horizon"
`recompute_total_max_steps` derived a `max_steps` that the checkpoint's `global_step` already reaches — usually
because the dataset shrank, the GPU count grew, or the batch geometry changed.
**Fix:** raise `--epoch`, or restore the original geometry.

### `RuntimeError: [Resume] Failed to read trainer_state.json ...`
The checkpoint directory is incomplete (interrupted save) or unreadable.
**Fix:** point `--lora_checkpoint_dir` at a directory whose newest `checkpoint-*` has a valid
`trainer_state.json`, or delete the broken one so `get_last_checkpoint` selects the previous checkpoint.

### `RuntimeError` in `_convert_all_state_info` when saving with an 8-bit optimizer
FSDP `FULL_STATE_DICT` tries to gather optimizer state shaped like each flat parameter, but bnb 8-bit state is
quantized (uint8 + per-block absmax) -> flat-numel mismatch.
**Fix:** `export MEDVISION_SFT_SAVE_ONLY_MODEL=1`. Accept that resume then starts with a fresh optimizer and LR
schedule.

---

## 6. Save hangs and NCCL timeouts

**Symptom:** the run stops at a save step; ranks eventually die with an NCCL collective timeout (the PyTorch
default process-group timeout is 30 minutes).

**Cause:** a `FULL_STATE_DICT` save at 27-31B all-gathers weights (and, without `save_only_model`, the fp32
optimizer state) to rank 0 host RAM and writes ~190 GB to storage. On slow shared storage that outlasts the
timeout, and the other ranks time out waiting in the collective.

**Fixes:**
- Widen the process-group timeout. Every entry point builds an
  `InitProcessGroupKwargs(timeout=timedelta(hours=1))`, but that object is **not consumed** by HF `Trainer`,
  which creates its own `Accelerator` internally — so on the CoT entry points the effective timeout is still
  the 30-minute PyTorch default. Only `train__qwen25vl_AD_TL_tooluse` makes it stick, by assigning
  `torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT = timedelta(hours=3)` at import time before
  `init_process_group` runs. Copy that pattern (or set the equivalent NCCL/PyTorch timeout in the environment)
  when saves on your storage take longer than 30 minutes.
- `MEDVISION_SFT_SAVE_ONLY_MODEL=1` roughly halves what is gathered and written.
- Raise `--save_steps` and lower `--save_total_limit` so saves are rarer and old ones are pruned.
- Save to fast local storage rather than a shared network filesystem where possible.
- Probe before committing: run once with `--save_steps 1` and confirm a save **and** a resume complete.

---

## 7. Loss and masking

### Loss is NaN from the first step on a Gemma-family model
The wrong masker was applied: on a Gemma tokenizer `<|im_start|>` resolves to `unk_token_id`, so
`mask_non_assistant_turns` never matches an assistant header and labels every token `-100`.
**Fix:** Gemma-lineage checkpoints must use `make_collate_fn_Gemma4` / `make_collate_fn_MedGemma`
(i.e. the matching `train__*__gemma4` / `__medgemma` entry point). The Gemma masker refuses rather than guesses:
`ValueError: No Gemma turn markers in this tokenizer's vocabulary (tried ...)` or `RuntimeError: Completion-only
masking left no tokens in the loss`.

### Training loss looks suspiciously low / the model does not learn the answer format (Gemma families)
Without `MEDVISION_SFT_COMPLETION_ONLY=1`, the Gemma collates mask only padding and image tokens, so the loss is
computed over the **prompt** as well. MedVision prompts are long (task description, format requirement, CoT
instruction, image and pixel-size arithmetic), so most of the signal is next-token prediction of your own
prompt. Qwen collates always apply completion-only masking, so this is a Gemma-only pitfall.
**Fix:** `export MEDVISION_SFT_COMPLETION_ONLY=1` (the `__cmplLoss` variants). Treat it as a different
experiment: give the run its own `run_name`, `wandb_run_id` and merged-model name.
**Verify:** the fraction of non-masked tokens drops sharply; the assistant turns and their closing end-of-turn
token remain in the loss.

### A decoded batch shows two BOS tokens at the start (Gemma lineage)
The collates call `processor.apply_chat_template(..., tokenize=False)` and then pass the resulting **string**
back through the processor, so a chat template that already emits BOS can be joined with the tokenizer's own
`add_special_tokens` BOS. It is a property of the checkpoint's template, not of MedVision's code, and the
codebase applies no workaround. Do **not** patch the chat template to remove it: `sft_prompts.py` and the
templates are shared with benchmark evaluation, so changing them silently changes eval behaviour too. If you
need to confirm what the model actually sees, decode one collated batch
(`processor.tokenizer.decode(batch["input_ids"][0])`) rather than reasoning about it.

---

## 8. Weights & Biases

### A resumed run creates a second chart, or refuses to resume
`wandb_run_id` must be unique per project and stable across resumes; `wandb_resume` is `allow` / `must` /
`never`. Two runs sharing an id in the same project collide.
**Fix:** reuse the exact `--wandb_run_id` (with `--wandb_resume allow`) to continue; choose a new id for a new
experiment; use `--wandb_resume must` to fail loudly when the id does not already exist.
Note `parse_validate_args_multiTask` exports these as `WANDB_RESUME` / `WANDB_RUN_ID` / `WANDB_NAME` /
`WANDB_PROJECT` / `WANDB_DIR` — an existing `WANDB_*` value in the shell is overwritten by the flags.
**Offline debugging:** `export WANDB_MODE=offline` (and `WANDB_DEBUG=true`, `WANDB_CORE_DEBUG=true`).

---

## 9. Merge and push

### `--merge_only true` still tries to prepare data
It should not: `merge_only` skips the whole training block including dataset preparation. If you see loading,
`merge_only` did not parse as true — it is a `str2bool` flag and needs an explicit value
(`--merge_only true`, not a bare `--merge_only`).

### `ValueError: [Error] merged_model_hf must be specified when push_to_hub is True.`
`--push_merged_model true` without `--merged_model_hf`.
**Fix:** supply the target repo id, or set `--push_merged_model false` and only save locally with
`--merged_model_dir`.

### Merge is killed / swaps
`merge_models` loads the base model in **fp32 on CPU** (so the sub-bf16 LoRA delta is representable). At 27-31B
that is a very large host allocation, and on a container the ceiling is the cgroup memory limit.
**Fix:** run the merge on a host with enough RAM, or skip merging and evaluate the adapter directly.

### `safe_merge=True` raises on NaN/inf
The adapter contains non-finite weights — usually a diverged run.
**Fix:** merge an earlier checkpoint, and investigate the loss curve (LR too high, or a masking bug).

### HTTP 401 when pushing (or when downloading a gated base model)
A pod-injected `HF_TOKEN` carries a trailing newline that corrupts the Authorization header.
**Fix:** `export HF_TOKEN="$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')"` before the run, and confirm
`hf auth login` (or a valid token) has write scope for the target namespace. All pushes here create **private**
repos (`hub_private_repo=True`, `private=True`).

### Full-FT run "ignored" the merge flags
Correct: the `train__fullFT-CoT__*` and tool-use entry points have no merge step and do not read
`--merge_model` / `--merge_only` / `--merged_model_*`. They save complete checkpoints; `--push_LoRA` is reused
as the trainer's `push_model`.

---

## 10. When to stop and ask

- No GPU on the host -> phase B, and therefore any real training, is impossible. Report that instead of trying.
- The environment does not satisfy the family's pins -> fix it via `../../environment-setup/SKILL.md` first; a
  mismatched `transformers` / `flash-attn` / `protobuf` will fail at import or at step 0.
- The fix requires a bigger machine (140 GB-class GPUs, more host RAM for a merge, faster storage for saves) ->
  say so with the number, do not burn hours on knob-twiddling.
- The fix requires Hub credentials or would push a model -> ask first.
- Rebuilding the prepared dataset means re-downloading whole source datasets and rewriting the PNG cache ->
  quote the cost before starting.
