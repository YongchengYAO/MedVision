# MedVision SFT — trainer configuration

Both trainers are built in `medvision_bm.sft.sft_utils` and return a `trl.SFTTrainer` (or a temperature-sampling
subclass). Both raise `ValueError("GPU does not support bfloat16, ...")` when
`torch.cuda.get_device_capability()[0] < 8`, so neither can be constructed on a CPU-only host.

---

## 1. LoRA / QLoRA — `prepare_trainer`

Used by the four `train__SFT-CoT__*` entry points and `train__SFT__qwen2_5_vl`.

**Model loading**
- `AutoModelForImageTextToText.from_pretrained(base_model_hf, ...)` in `torch.bfloat16`, `trust_remote_code=True`,
  `device_map={"": PartialState().process_index}` (one full copy per rank — plain DDP, not sharded).
- 4-bit NF4 quantization: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16, bnb_4bit_quant_storage=bfloat16)` — i.e. **QLoRA**.
- `attn_implementation = MEDVISION_SFT_ATTN or ("flash_attention_2" if use_flash_attention_2 else "eager")`.
- `model.config.use_cache = False` (and `text_config.use_cache = False`): training never generates, and under
  activation-checkpoint recompute a live cache is appended twice.
- Processor from `base_model_hf` with `tokenizer.padding_side = "right"`.

**Adapter** — `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
target_modules="all-linear", task_type="CAUSAL_LM", modules_to_save=["lm_head", "embed_tokens"])`.
`modules_to_save` means the LM head and embeddings are trained in full precision — that is why the Gemma-family
27-31B QLoRA recipes need an 8-bit optimizer (`MEDVISION_SFT_OPTIM=paged_adamw_8bit`).

**`SFTConfig`** — `learning_rate=2e-4`, `lr_scheduler_type="linear"`, `warmup_ratio=0.03`, `max_grad_norm=0.3`
(QLoRA-paper settings), `bf16=True`, `optim=MEDVISION_SFT_OPTIM or "adamw_torch_fused"`,
`seed=data_seed=SEED`, `save_strategy="steps"`, `eval_strategy="steps"`,
`gradient_checkpointing_kwargs={"use_reentrant": False}`, `dataset_kwargs={"skip_prepare_dataset": True}`
(the dataset is already formatted), `remove_unused_columns=False`, `label_names=["labels"]`,
`dataloader_persistent_workers=False`, `report_to="wandb"`, `push_to_hub=push_LoRA`, `hub_private_repo=True`.
`output_dir = lora_checkpoint_dir`.

---

## 2. Full parameters — `prepare_trainer_fullFT`

Used by the four `train__fullFT-CoT__*` entry points and `train__qwen25vl_AD_TL_tooluse`.

**Model loading**
- Same `AutoModelForImageTextToText` in bf16, **no quantization, no PEFT** — every parameter trains.
- `device_map` is set **only when FSDP is off** (`ACCELERATE_USE_FSDP != "true"`). Under FSDP a `device_map`
  would dispatch the whole model onto each GPU before the wrap, leaving every rank holding unsharded weights
  (~77 GiB/GPU at 27-31B) and OOMing in backward. Without it, transformers' FSDP-aware loading
  (`fsdp_cpu_ram_efficient_loading` + `fsdp_sync_module_states`) materialises rank 0 on CPU and the rest on meta,
  and FSDP shards onto the GPUs at wrap time.
- `model_weights_from` (the resume path) replaces `base_model_hf` in `from_pretrained` so resumed weights come in
  through that same FSDP-aware path. The processor still comes from `base_model_hf`.

**`SFTConfig`** — `learning_rate = MEDVISION_SFT_LR or 2e-5`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.03`,
`max_grad_norm=1.0`, `bf16 = (MEDVISION_SFT_PURE_BF16 != "1")`,
`optim = MEDVISION_SFT_OPTIM or "adamw_torch_fused"`, `save_only_model = (MEDVISION_SFT_SAVE_ONLY_MODEL == "1")`,
`use_liger_kernel = (MEDVISION_SFT_USE_LIGER == "1")`, `gradient_checkpointing` default **True**, plus the same
`skip_prepare_dataset` / `remove_unused_columns` / `label_names` / wandb settings. `output_dir = checkpoint_dir`.

**Post-construction patches** (all opt-in, see `cli-reference.md` §4): `MEDVISION_SFT_BF16_GRADS` (FSDP
`MixedPrecision(keep_low_precision_grads=True)`), `MEDVISION_SFT_SYNC_EACH_BATCH` (replaces
`trainer.accelerator.no_sync` with a null context), `MEDVISION_SFT_MEMSNAPSHOT` (wraps `trainer.train` to dump a
CUDA allocator snapshot on OOM), and the `MEDVISION_SFT_MEMPROBE` callback.

---

## 3. FSDP topology and the two memory recipes

Launch flags (see `cli-reference.md` §3): `--use_fsdp --fsdp_sharding_strategy FULL_SHARD
--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap <Class>
--fsdp_state_dict_type FULL_STATE_DICT --fsdp_offload_params false --fsdp_cpu_ram_efficient_loading true
--fsdp_sync_module_states true`.

`<Class>` per family, as used by the recipes: `Gemma4TextDecoderLayer` (Gemma 4), `Gemma3DecoderLayer`
(MedGemma), `Qwen3_5DecoderLayer` (Qwen3.5/3.6). Verify against the installed transformers before a long run.

### Recipe A — 140 GB-class GPUs, fp32 master weights, fully resumable

```
--mixed_precision=bf16               # on the accelerate launch
export MEDVISION_SFT_SYNC_EACH_BATCH=1
export MEDVISION_SFT_MEMPROBE=1
# deliberately NO OPTIM / SAVE_ONLY_MODEL / PURE_BF16 / LR override
```

Default fused fp32 AdamW, `FULL_STATE_DICT` saves that include optimizer state, so
`--resume_from_checkpoint true` restores the full training state. Cost: each checkpoint is roughly bf16 weights
plus optimizer state (~190 GB at 31B) and the save all-gathers the fp32 optimizer state to rank-0 host RAM —
run a 1-save + 1-resume probe (`--save_steps 1`) before committing to a multi-day run, and keep
`--save_total_limit` small (3 in the recipes).

### Recipe B — 80 GB-class GPUs, pure bf16, weights-only checkpoints

```
# NO --mixed_precision flag on the accelerate launch
export MEDVISION_SFT_PURE_BF16=1        # SFTConfig(bf16=False): no fp32 masters, no bf16 _mp_shard
export MEDVISION_SFT_LR=4e-5            # keep AdamW updates above the bf16 rounding floor
export MEDVISION_SFT_OPTIM=adamw_bnb_8bit
export MEDVISION_SFT_SAVE_ONLY_MODEL=1  # required: bnb 8-bit state cannot be gathered by FULL_STATE_DICT
export MEDVISION_SFT_SYNC_EACH_BATCH=1
export MEDVISION_SFT_MEMPROBE=1
export MEDVISION_SFT_MEMSNAPSHOT=1
# MedGemma additionally: pip install --no-deps "liger-kernel>=0.5.4"; export MEDVISION_SFT_USE_LIGER=1
```

Trade-off: checkpoints hold weights only, so `--resume_from_checkpoint` continues from the saved weights with a
**fresh optimizer and LR schedule**.

Facts that drive both recipes (all recorded in the recipes' own comments):

- `MEDVISION_SFT_SYNC_EACH_BATCH=1` is not optional at 27-31B. HF `Trainer` wraps non-final micro-steps in
  `accelerator.no_sync()`, under which FSDP accumulates **full unsharded** gradients on every rank; those do not
  shrink with world size and OOM partway into the *first* backward. Syncing each micro-batch reduce-scatters
  immediately so gradients accumulate sharded — numerically neutral, negligible cost on NVLink.
- `paged_adamw_8bit` leaves its state as CUDA unified-memory pages that are resident on the device but outside
  the torch allocator, which cannot evict them; `adamw_bnb_8bit` keeps the same state inside the torch pool.
  This is why the full-FT recipes use `adamw_bnb_8bit` and only the QLoRA Gemma recipes use `paged_adamw_8bit`.
- `MEDVISION_SFT_BF16_GRADS=1` hard-fails under accelerate bf16 mixed precision (torch's `.grad` setter needs
  matching dtypes); pure bf16 makes it unnecessary.
- Do not enable `--fsdp_offload_params` on a memory-capped container: at 31B it wants ~700 GB host RAM, and the
  ceiling is the **cgroup** limit, not what `free` reports.
- With `MEDVISION_SFT_MEMPROBE=1`, post-wrap allocated should be roughly `params / world_size`. A full-model
  figure means sharding did not engage; a figure ~2x the bf16 shard means the fp32-master upcast is active.

Published GPU profiles: QLoRA runs use 4 GPUs (80 GB class); full-FT of 27-31B needs either 4x 140 GB with
recipe A or 4x 80 GB with recipe B; the 7B full-FT run fits 4 GPUs at `per_device_train_batch_size=8`.

---

## 4. Loss masking

Every collate function builds `labels = input_ids.clone()` and then masks with `-100`.

| Family | Collate | Always masked | Completion-only masking |
| --- | --- | --- | --- |
| Qwen2.5-VL / Qwen3-VL | `make_collate_fn_Qwen25VL` | pad token, image token | **always on** (`mask_non_assistant_turns`) |
| Qwen2.5-VL tool-use | `make_collate_fn_Qwen25VL_tooluse` | pad token, image token | **always on**; chat template applied with `tools=[TOOL_DEF]` |
| MedGemma (Gemma 3) | `make_collate_fn_MedGemma` | pad token, `boi_token`/`eoi_token`/`image_token` | only when `MEDVISION_SFT_COMPLETION_ONLY=1` |
| Gemma 4 | `make_collate_fn_Gemma4` | pad token, `boi_token`/`eoi_token`/`image_token` | only when `MEDVISION_SFT_COMPLETION_ONLY=1` |

`mask_non_assistant_turns` (ChatML) and `mask_non_assistant_turns_gemma` share `_mask_turns`: it scans for an
assistant/model turn header (start-of-turn token immediately followed by the role token), masks the header, the
role token and the role newline, keeps the response content **and the closing end-of-turn token** in the loss,
and masks everything else (system, user, tool turns). It only ever writes `-100`, so pad/image masks written
earlier survive.

Family-specific details:

- ChatML markers: `<|im_start|>` / `<|im_end|>`, role token `assistant`.
- Gemma markers are probed from `_GEMMA_TURN_MARKERS`: `<start_of_turn>`/`<end_of_turn>` (Gemma 3, MedGemma) or
  `<|turn>`/`<turn|>` (Gemma 4), role token `model`. `_resolve_special_token_id` treats a result equal to
  `unk_token_id` as *absent*, because `convert_tokens_to_ids` returns unk (id 3 on Gemma) rather than None for
  out-of-vocabulary tokens — that is exactly how the wrong generation's markers would look valid.
- The two maskers must not be swapped: on a Gemma tokenizer `<|im_start|>` resolves to unk, no header ever
  matches, every label becomes `-100` and the loss silently goes NaN. `mask_non_assistant_turns_gemma` therefore
  raises `ValueError` when no marker pair resolves and `RuntimeError` when masking leaves nothing in the loss.
- Gemma pad masking runs before the turn scan safely, because `pad_token_id` differs from the end-of-turn id.
- Both collates skip (with a warning) examples whose image fails to load, but only
  `make_collate_fn_Gemma4` raises `RuntimeError` when a whole batch fails; in
  `make_collate_fn_MedGemma` an all-failed batch surfaces as a processor error instead.

**Consequence for Gemma-family runs without `MEDVISION_SFT_COMPLETION_ONLY=1`:** the loss is computed over the
prompt tokens too (only pad and image tokens are excluded). With MedVision's long pixel-size arithmetic prompts
that is a large fraction of the sequence, diluting the signal — which is precisely why the `__cmplLoss` launcher
variants exist. Turning it on changes the objective, so treat a `cmplLoss` run as a different experiment
(the launchers give it a distinct `run_name`, `merged_model_hf` and `wandb_run_id`).

---

## 5. Temperature-based multi-task sampling

`--enable_temperature_sampler true` routes trainer construction through `_build_temperature_sampler_trainer`
(shared by both trainers):

1. Requires `--temperature_sampler_T > 0` and the task column (default `__task_name`) in the train split;
   otherwise `ValueError` ("Regenerate prepared dataset with task labels or disable temperature sampler").
2. Counts rows per task label. With a **single** task it logs a notice and falls back to a plain `SFTTrainer`.
3. `task_probs = counts^(1/T)`, normalised. `T=1` is proportional to the raw counts; larger `T` flattens the
   distribution (the recipes use `T=5`).
4. Per-sample weight `= p(task) / count(task)`, so a `torch.utils.data.WeightedRandomSampler` reproduces exactly
   those task-level proportions.
5. `_get_train_sampler` is overridden to return that sampler with `replacement=True` (needed so minority-task
   rows can be drawn more often than their cardinality in one epoch) and
   `torch.Generator().manual_seed(SEED)`.
6. `num_samples` per epoch = `--temperature_sampler_num_samples` when `> 0`, otherwise `len(train_dataset)` —
   so by default the epoch length is unchanged and only the task composition shifts.

It logs the task counts, the resulting per-task probabilities and `num_samples`. The flag is **training-only**;
it is ignored during `--process_dataset_only true`.

---

## 6. Resume — `train_resume_from_checkpoint` and `recompute_total_max_steps`

`recompute_total_max_steps(trainer)`: world size from `PartialState` (falling back to `args.world_size` then
`WORLD_SIZE`), `effective_bsz = per_device_train_batch_size * world_size * gradient_accumulation_steps`,
`steps_per_epoch = ceil(len(train_dataset) / effective_bsz)` (floor when `dataloader_drop_last`),
`new_max_steps = steps_per_epoch * epochs`. Computed on rank 0 and **broadcast**, so all ranks share one horizon
(a mismatch would hang the collectives).

`train_resume_from_checkpoint(trainer, last_checkpoint, weights_preloaded=False)`:

1. Asserts a checkpoint was found, recomputes `max_steps`.
2. Rank 0 reads `trainer_state.json` for `global_step` and the recorded `max_steps`; if
   `new_max_steps <= recorded_max and global_step >= new_max_steps`, the run is marked **finished**. The decision
   is broadcast. A failure to read the file raises `RuntimeError` rather than silently restarting.
3. Applies `max_steps` and `is_finished` on every rank, then calls
   `trainer.train(resume_from_checkpoint=last_checkpoint)`.
4. `weights_preloaded=True` (full-FT and tool-use) replaces `trainer._load_from_checkpoint` with a no-op logger,
   because the weights already came in through `from_pretrained`; letting the Trainer re-load a sharded
   weights-only checkpoint under FSDP goes through `load_sharded_checkpoint`, whose `model.state_dict()`
   all-gathers the full unsharded model **on every rank** and OOMs at 27B+.

LoRA entry points build the trainer first and then look for a checkpoint; full-FT and tool-use entry points look
for the checkpoint **first** and pass it as `model_weights_from`.

Practical consequences: changing the dataset size, GPU count, batch size, accumulation or `--epoch` between runs
changes the recomputed horizon. If it is already satisfied the run exits immediately — raise `--epoch`.

---

## 7. Merge — `merge_models`

Called on the main process only, when `--merge_model` or `--merge_only` is set, **after** the trainer is deleted
and `torch.cuda.empty_cache()` has run. It is CPU-only:

1. `AutoModelForImageTextToText.from_pretrained(base_model_hf, low_cpu_mem_usage=True,
   torch_dtype=torch.float32, device_map="cpu")` — fp32 so the sub-bf16 LoRA delta is representable.
2. `PeftModel.from_pretrained(model, lora_checkpoint_dir)`, cast to fp32,
   `merge_and_unload(safe_merge=True)` (raises on NaN/inf).
3. Processor from the **adapter** directory.
4. `save_pretrained(merged_model_dir, safe_serialization=True, max_shard_size="2GB")` when a dir is given.
5. `push_to_hub(merged_model_hf, private=True, max_shard_size="2GB")` plus the processor when
   `--push_merged_model true`; raises `ValueError` if `merged_model_hf` is None.

`--merge_only true` skips the entire training block (no dataset preparation, no trainer), so it is the cheap way
to merge or re-push a finished run — but it needs the base model downloadable and enough host RAM for an fp32
copy of it. The full-FT and tool-use entry points have no merge step at all.
