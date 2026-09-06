# MedVision SFT — dataset construction

Everything here happens on the **main process only**, during phase A
(`--process_dataset_only true`) or the first part of a combined run. Other ranks wait at a barrier and then
receive the resolved `prepared_ds_dir` by broadcast, so every rank loads the same directory.

---

## 1. Pipeline

```
per task:  load_split_limit_dataset(...)            # load *_Train configs, concat, volume-grouped split, per-task train cap
           -> format_clean_dataset(...)             #   format_dataset(map) -> clean_dataset(remove_columns)
           -> add_column("__task_name", <label>)    # tag for the temperature sampler
concat all tasks -> global train/val limits (shuffle | select | bootstrap) -> save_to_disk(prepared_ds_dir)
```

`prepare_dataset` is the single-call convenience wrapper (`load_split_limit_dataset` + `format_clean_dataset`);
the entry points call the two stages separately so they can read the **true** split sizes before naming the
output directory.

Stage 1 (load + split) runs whenever the default name has to be resolved (no `--prepared_ds_dir`) or formatting
will follow (`--skip_process_dataset false`). With `--prepared_ds_dir` **and** `--skip_process_dataset true` —
the training launch as the repository launchers issue it — nothing is loaded and the directory is taken as-is.

---

## 2. Loading and splitting — `load_split_limit_dataset`

Signature: `(tasks_list_json_path, limit_train_sample, limit_val_sample, num_workers_concat_datasets=4,
tag_ds=None, download_mode="reuse_dataset_if_exists")`.

1. **Read the task list.** The JSON's keys are task names. When `tag_ds == "BoxSize"`, keys containing
   `_BoxCoordinate_` are renamed to `_BoxSize_` first — detection lists written in the eval namespace
   (e.g. the dataset-info AllSlices lists) therefore work unchanged. `tag_ds` per task:
   A/D `BiometricsFromLandmarks`, Detection `BoxSize`, T/L `TumorLesionSize`.
2. **Load each task's training split** from the `YongchengYAO/MedVision` HF dataset, config `<task>_Train`,
   in a `ProcessPoolExecutor` with `min(num_workers_concat_datasets, cgroup CPUs, #tasks)` workers and a
   120 s per-task timeout. Dataset name = the part of the task name before `_<tag_ds>`.
   If any needed dataset is missing from `<data_dir>/.downloaded_datasets.json`, worker count drops to **1**
   to avoid cache conflicts during the download.
   After each task completes, the cgroup memory usage is checked and a warning printed above 80%.
   Any failed task raises `RuntimeError` after all futures are collected.
3. **Concatenate in the JSON order**, not the completion order, so the seeded shuffle/split downstream is
   reproducible run to run.
4. **Split** with `group_train_test_split(group_column="image_file", test_size=limit_val_sample, seed=SEED,
   stratify_column="dataset_name")` — see below.
5. **Apply the per-task train cap**: only when `0 < limit_train_sample < len(train)` does it
   `shuffle(seed=SEED).select(range(limit_train_sample))`. A cap larger than the pool is a **no-op** here
   (no upsampling at the per-task level).

Assertions: `limit_val_sample > 0`, `limit_train_sample != 0`, `tag_ds is not None`, `MedVision_DATA_DIR` set.

`SEED` comes from `medvision_bm.utils.configs` — never hard-code a seed.

### `group_train_test_split`

Prevents 3D-volume leakage: every 2D slice of one volume (`image_file`) lands in the same split.

1. Group row indices by `image_file`.
2. Shuffle the volumes with `numpy.random.default_rng(seed)`. With `stratify_column="dataset_name"`, volumes are
   first bucketed per dataset, shuffled inside each bucket, then **round-robin interleaved** one volume per
   dataset per round, so every dataset contributes a validation volume before any dataset gets a second one.
3. Greedily append whole volumes to validation until the cumulative **sample** count reaches or exceeds
   `test_size`. Because volumes are added whole, the validation split can end up slightly **larger** than the
   target — this is expected.
4. Everything else becomes train; both index lists are shuffled.

`test_size` is an absolute sample count when `>= 1` (that is how the entry points use it) or a fraction when a
float `< 1.0`.

---

## 3. Sample-limit semantics — `parse_sample_limits` / `_get_sample_limit`

Resolution order per task (A/D, Detection, T/L):

1. `--train_sample_limit_task_<X>`; if `<= 0`, fall back to `--train_sample_limit_per_task` (default `-1`).
2. `--val_sample_limit_task_<X>`; if `<= 0`, fall back to `--val_sample_limit_per_task` (default `100`).
3. If that task's `--tasks_list_json_path_*` is **None**, both limits are forced to `0` — the task is not used.

`--val_sample_limit` (the global one) is passed through `_get_sample_limit` too, purely so a `0` is rejected at
the same choke point; the entry points read it back from kwargs.

Rules to remember:

- **Unset or `-1` = the full pool.** This applies to every train limit including the global one.
- **`0` is rejected** with
  `[Error] <flag>=0 is ambiguous: use -1 (or leave it unset) for no limit; to skip a task, omit its
  --tasks_list_json_path_* instead.` Zeroing a limit is never the way to drop a task.
- **Per-task train caps apply after the validation carve-out** and only when the cap is smaller than the
  remaining pool.
- **Validation carve-out is per task and volume-grouped**, with a fallback target of 100 rows per task.
- **`--train_sample_limit` is a GLOBAL cap applied after concatenating all tasks:**
  - `limit > concatenated size` -> `np.random.seed(SEED)` then `np.random.choice(size, limit, replace=True)`:
    a **bootstrap with duplicates**. This is deliberate (it is how a small task can be oversampled to a target
    count) but it means the "121000" in the published recipe is a *draw count*, not a distinct-sample count when
    the pools are smaller.
  - `0 < limit <= concatenated size` -> `shuffle(seed=SEED).select(range(limit))`. If the limit is **less than
    the sum of the per-task limits, the excess is silently truncated** and the task mix you configured is not
    what you get. Keep the global limit equal to the sum of the per-task limits unless you mean to truncate.
  - `-1` -> shuffle only.
  - `--val_sample_limit` behaves identically on the concatenated validation split.

Use `scripts/check_sample_limits.py` to resolve a flag combination through the real function; pass
`--pool_AD/--pool_Detection/--pool_TL` to simulate the outcome against known pool sizes.

---

## 4. Formatting — `format_dataset` / `format_clean_dataset` / `clean_dataset`

`format_dataset(dataset, mapping_func, mapping_func_args, num_workers_format_dataset, writer_batch_size=1000)`:

- Worker count is `min(num_workers_format_dataset, get_cgroup_limited_cpus())`.
- Before mapping, rows are **sorted by `image_file`** so each worker's contiguous shard hits the one-slot
  per-process NIfTI cache (`_NIFTI_CACHE`) and each `.nii.gz` is decompressed once per volume instead of once per
  slice; afterwards the original row order is restored with the inverse permutation, because the seeded shuffles
  downstream permute positions and must see the same input order every run.
- Applied to each split of a `DatasetDict` independently.

`format_clean_dataset` calls it with `writer_batch_size=50` and `fn_kwargs =
{model_name: model_family_name, model_hf: base_model_hf, process_img, save_processed_img_to_disk, new_shape_hw}`,
then prunes columns.

`clean_dataset(dataset, keys_to_keep)` uses `remove_columns` — a **schema-only** operation, no data pass. Kept
columns: `messages`, `labels`*, `image_file`, `slice_dim`, `slice_idx`, plus `processed_images` when
`--process_img` and `image_file_png` when `--save_processed_img_to_disk`. The entry points then add
`__task_name`. *`labels` is in the whitelist but never actually present — no source config or
`_format_data_*` mapper creates it, and `remove_columns` cannot; the collator builds `batch["labels"]`
at train time.

`safe_concatenate_datasets` (with `safe_concat_align_top_keys` / `safe_concat_align_dict_keys`) exists in
`sft_utils` to align mismatched schemas before concatenation; the current entry points concatenate the already
schema-pruned per-task splits directly with `datasets.concatenate_datasets`.

### cgroup-aware helpers

Both worker counts go through `get_cgroup_limited_cpus()`, which reads `cpu.cfs_quota_us`/`cpu.cfs_period_us`
(cgroup v1) or `cpu.max` (v2) and falls back to `os.cpu_count()`. `get_cgroup_memory_percent()` returns
`(used_GiB, limit_GiB, percent)` from `memory.current`/`memory.max` (v2), the v1 equivalents, or psutil. On a
container, `psutil` and `free` report the **host**, so these cgroup readings are the only trustworthy signal —
this is why the loader can warn about memory pressure that `free` would not show.

`save_to_disk` uses `num_proc = max(1, min(cgroup CPUs, smallest split's row count))`, because `datasets` raises
`IndexError` when `num_proc` exceeds a split's row count (hits tiny smoke-test splits).

---

## 5. Image handling and the pixel-size invariant

`_doc_to_visual` / `_load_resize_nifti_2d` extract the 2D slice `slice_idx` along `slice_dim` from the NIfTI at
`image_file`, apply CT windowing or general normalisation, and optionally resize to `new_shape_hw`.

Two storage options:

- `img_proccessor_nii2png_save2disk` (**recommended**, `--save_processed_img_to_disk true`): writes
  `<volume>_dim<d>_slice<i>_resized-wh-<W>x<H>.png` (or `..._original-wh-<W>x<H>.png`) into a
  `tmp_prepared_png/` folder **next to the source NIfTI**, and stores the path in `image_file_png`.
  With an explicit `new_shape_hw` the output filename is known without touching the NIfTI, so an existing PNG is
  reused — it is verified to fully decode first, because a killed run could have left a truncated file. Writes
  are atomic (`.tmp<pid>` then `os.replace`), so a surviving file is always complete.
- `img_proccessor_nii2png_save2dataset` (`--process_img true`): embeds the PNG bytes in the Arrow dataset.
  Not recommended — the cache becomes enormous.

The collate functions load images in the priority `processed_images` > `image_file_png` > `image_file`
(decoding the NIfTI on the fly), so a prepared dataset still works if the PNG cache is deleted, just slower.

**The pixel-size invariant.** For A/D and T/L, the prompt states the image size *and* the pixel size the model
will perceive. The chain is:

1. `_load_resize_nifti_2d` returns the voxel spacing of the (possibly `new_shape_hw`-resized) slice.
2. `get_resized_img_shape(model_name, img_2d, {"model_hf": ...})` — the shared function in the vendored
   `lmms_eval` task layer — returns `(perceived_canvas_hw, content_hw)` for that model family. It is a hardcoded
   dispatch on the family key: the SFT `model_family_name` must be one of the strings it branches on (the same
   keys as the benchmark's `AVAILABLE_MODELS`, with the `vllm_` prefix optional). An unknown key raises
   `ValueError: [Error] <name> is not recognised/supported.`
3. The pixel size is divided by the per-axis content resize ratio, and both numbers are written into the prompt.

Consequence: **a prepared dataset is valid for exactly one `model_family_name` and one `new_shape_hw`.** Reusing
one for a different family trains on wrong physical scales with no error. Adding a new family means adding a
branch to `get_resized_img_shape` — see `../../extending-models-and-tasks/SKILL.md`.

Detection prompts contain no pixel size (coordinates are relative), so `_format_data_DetectionTask*` ignores
`model_name`; the argument is kept only for a uniform mapping signature.

---

## 6. Targets

Non-CoT (`_format_data_*Task`): the assistant turn is the bare ground-truth string (a number, a pair, or four
relative box coordinates).

CoT (`_format_data_*Task_CoT`, used by every `-CoT` entry point): the prompt builder returns
`(question, values_dict)` and the target builder fills a template from `medvision_bm.sft.sft_prompts` with
`values_dict`, producing an assistant turn shaped as

```
<think> <step-1-reasoning> ... </step-1-reasoning> <step-1-answer> ... </step-1-answer> ... </think>
<answer> ... </answer>
```

`values_dict` carries the **intermediate ground truth**, which is what makes the chain-of-thought supervision
real rather than free-form: landmark relative coordinates, the stated image width/height, the adjusted pixel
width/height, the vector components and the derived angle/length for A/D; the ellipse major/minor endpoints for
T/L; the box corner coordinates for detection.

Templates and instructions live in `sft_prompts.py`: `COT_INSTRUCT_{DISTANCE,ANGLE,TL,TL_NORM,DETECTION}`,
`COT_TEMPLATE_{DISTANCE,ANGLE,TL,TL_NORM,DETECTION}`, and the format prompts
`FORMAT_PROMPT_{AD,TL,DETECTION}_REASONING` (plus the non-CoT `FORMAT_PROMPT_*` and `GENERAL_FORMAT_PROMPT` /
`SYSTEM_PROMPT_LITE`). **These strings are shared with benchmark evaluation** — editing one changes both the
training targets and how responses are parsed at eval time.

Tool-use targets (`_build_tooluse_messages_AD` / `_TL` in `sft_utils`, templates in `sft_prompts_tooluse.py`) are
5 turns — system, user, assistant (`<think>` steps 1-2 + `<tool_call>` with Python code), tool (the output of
`safe_exec_python` on that code), assistant (`<answer>`) — with `TOOL_DEF` describing the `execute_python` tool.

The final assistant message is always a plain text content block; the user message is always
`[{"type": "image"}, {"type": "text", "text": <prompt>}]`, i.e. exactly one image per sample.

---

## 7. The prepared dataset on disk

Default directory:

```
<data_dir>/SFT-CoT_datasets/<model_family_name>/ds__AD<a>_D<d>_TL<t>_all<n><suffix>
```

- `<a>/<d>/<t>` = the requested per-task train cap when it is `> 0`, otherwise the **true** split size measured
  after loading (that is why the name can only be resolved after stage 1). `<n>` = the requested global cap, or
  the sum of the true sizes.
- `<suffix>` = `__resized-wh-<W>x<H>` when `--new_shape_hw H W` was given, else `__original`.
- The non-CoT entry point uses `SFT_datasets` instead of `SFT-CoT_datasets`; the tool-use entry point appends
  `-tooluse` to the suffix (`__resized-wh-512x512-tooluse`).
- `--prepared_ds_dir` overrides the whole thing and is taken as-is on every rank; combined with
  `--skip_process_dataset true` it also skips stage 1.

Contents: a `datasets.DatasetDict` written with `save_to_disk`, splits `train` and `validation`, columns
`messages`, `image_file`, `slice_dim`, `slice_idx`, `__task_name`, plus `image_file_png` and/or
`processed_images` as configured.

Phase A prints `Data processing completed. Prepared dataset saved at '<dir>'.` **Pass that directory to phase B
as `--prepared_ds_dir`.** The repository launchers do it automatically: phase A is piped through `tee` into
`<checkpoint_dir>/prepare_dataset.log`, the line is extracted with `sed`, the script aborts before the GPU
launch if no directory was reported, and phase B receives `--prepared_ds_dir`. Only a phase B launched
**without** `--prepared_ds_dir` has to re-derive the name: then the two phases must have identical
limit/resize/family flags, and rank 0 pays for a full load+split pass (every config loaded, the volume-grouped
split recomputed) before training starts — on every restart.

Inspect any prepared dataset with `scripts/inspect_prepared_dataset.py --prepared-ds-dir <dir>`
(add `--check-images 50` to verify the PNG cache still exists).
