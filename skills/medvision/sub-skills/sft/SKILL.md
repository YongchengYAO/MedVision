---
name: sft
description: "Runs supervised fine-tuning of a vision-language model on MedVision tasks: builds the prepared chain-of-thought SFT dataset (load_split_limit_dataset, volume-grouped train/validation split, per-task and global sample limits, PNG slice cache, prepared_ds_dir naming), then trains through the ten medvision_bm.sft.train__* entry points — QLoRA CoT and full-parameter FSDP CoT for qwen25vl, qwen3vl, gemma4 and medgemma, plus the non-CoT and tool-use variants — using the two-phase launcher pattern (--process_dataset_only, then accelerate launch --skip_process_dataset with the prepared_ds_dir captured from phase A's log). Covers every CLI flag and default, the MEDVISION_SFT_* memory knobs, temperature-based multi-task sampling, per-family loss masking and completion-only loss, resuming, merging LoRA adapters, pushing to the Hub, and diagnosing CUDA OOM, FSDP resume OOM, NCCL save hangs, protobuf/wandb and huggingface_hub dependency drift. Use for SFT, LoRA, QLoRA, full finetuning, FSDP, accelerate launch, prepared dataset, sample limits, loss masking, merge, or resume in MedVision."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision SFT: dataset construction, LoRA and full-parameter training

Use this sub-skill for the **supervised fine-tuning** stage of MedVision: turning MedVision tasks into a prepared
chain-of-thought training set, training a VLM on it (adapter or full parameters), then resuming, merging, pushing
and debugging that run. Terminology: Detection / Tumor-Lesion size (T/L) / Angle-Distance (A/D); `medvision_bm`
(this package), `medvision_ds` (dataset package), `lmms_eval` (vendored fork, source of the model-family registry).

## Route here for

- The ten entry points `python -m medvision_bm.sft.train__*`: QLoRA CoT (`train__SFT-CoT__{qwen2_5_vl,gemma4,
  medgemma,qwen3vl}`), full-parameter CoT (`train__fullFT-CoT__{qwen2_5_vl,gemma4,medgemma,qwen3vl}`), non-CoT
  (`train__SFT__qwen2_5_vl`) and tool-use (`train__qwen25vl_AD_TL_tooluse`) — all sharing one 55-flag parser.
- The two-phase pattern: phase A `--process_dataset_only true` on CPU, phase B `accelerate launch ...
  --skip_process_dataset true` on GPUs, and why the phases are separated.
- Sample-limit semantics (`parse_sample_limits`): unset/`-1` = full pool, `0` rejected, per-task vs global caps,
  the validation carve-out grouped by 3D volume, and the bootstrap-with-replacement case.
- Dataset construction internals: `load_split_limit_dataset`, `format_dataset`, the `_format_data_*_CoT` builders,
  the PNG slice cache, `prepared_ds_dir` naming, and the `model_family_name` -> image-size -> pixel-size chain.
- Trainer configuration: the LoRA/QLoRA recipe, the FSDP full-FT recipe and its `MEDVISION_SFT_*` memory knobs,
  the temperature sampler, per-family loss masking, `merge_models`, and resume behaviour.

## Do not use for

- Installing `medvision_bm` / `medvision_ds`, the frozen requirements files, pins and install order ->
  `../environment-setup/SKILL.md`.
- Task lists, dataset configs, download modes and `MedVision_PLANNER_VERSION` -> `../dataset-and-tasks/SKILL.md`.
- Evaluating the fine-tuned checkpoint on the benchmark -> `../benchmark-evaluation/SKILL.md`.
- GRPO / verl parquet building and the RFT stage that follows SFT -> `../rft/SKILL.md`.
- Adding a new model family (including the image-size dispatch a new `model_family_name` needs) ->
  `../extending-models-and-tasks/SKILL.md`.
- Vocabulary and directory layout -> `../../references/concepts-and-glossary.md`; failures that are not
  SFT-specific -> `../../references/troubleshooting.md`; model names and hardware -> `../../references/model-roster.md`.

## Prerequisites

1. `medvision_bm` importable **with the SFT stack for the chosen family** (torch, accelerate, trl, peft,
   transformers, bitsandbytes, and flash-attn where used); `medvision_ds` installed under `<data_dir>/src`.
2. `export MedVision_PLANNER_VERSION=<version>` (`1.0.0` is the published recipe). Pinning below the newest
   release also needs `MedVision_ACK_RELEASE`. `MedVision_DATA_DIR` is set for you by the entry point from
   `--data_dir`, and `<data_dir>/.downloaded_datasets.json` must exist.
3. The SFT-namespace task lists `tasks_MedVision-{AD,detect,TL}__train_SFT.json`.
4. Phase A is CPU-only but downloads datasets and writes PNGs. Phase B **requires GPUs** (bf16-capable,
   compute capability >= 8): 4x 80 GB for QLoRA, 4x 140 GB (H200 class) for 27-31B full-FT.

## Fast paths

**Check what your sample limits will actually do (CPU, no data):**

```
python scripts/check_sample_limits.py \
    --tasks_list_json_path_AD a.json --tasks_list_json_path_detect d.json --tasks_list_json_path_TL t.json \
    --train_sample_limit 121000 --train_sample_limit_task_Detection 110000 \
    --train_sample_limit_task_AD 5500 --train_sample_limit_task_TL 5500 --pool_TL 5456
```

**Preview a complete two-phase run without touching a GPU:**

```
DRY_RUN=1 bash scripts/sft_launcher_template.sh --benchmark-dir <benchmark_dir> \
    --family qwen25vl --base-model Qwen/Qwen2.5-VL-7B-Instruct --run-name <run_name> --shape "512 512"
```

**Phase A only (CPU, hours; writes the prepared dataset), then inspect it:**

```
bash scripts/sft_launcher_template.sh --phase A --benchmark-dir <benchmark_dir> --family qwen25vl ...
python scripts/inspect_prepared_dataset.py \
    --prepared-ds-dir <data_dir>/SFT-CoT_datasets/qwen25vl/ds__AD5500_D110000_TL5500_all121000__resized-wh-512x512
```

**Phase B (GPU):** `bash scripts/sft_launcher_template.sh --phase B --gpus 0,1,2,3 ...`, adding
`--mode fullft --fsdp-layer-cls <DecoderLayerClass>` for full-parameter training and
`SFT_ENV_KNOBS="MEDVISION_SFT_SYNC_EACH_BATCH=1 MEDVISION_SFT_MEMPROBE=1"` for the FSDP memory recipe.

**Merge a finished LoRA run without retraining:** re-run the phase-B command with `--merge_only true`
(and `--push_merged_model true --merged_model_hf <repo>` to publish it).

## References and scripts

- Read `references/workflows.md` for the end-to-end sequence: prerequisites, phase A, verification, phase B,
  monitoring, resuming, merging, pushing, hand-off to evaluation, and where to stop and ask.
- Read `references/cli-reference.md` for all 55 flags with types and defaults, the entry-point matrix, the
  `accelerate launch` flags, every `MEDVISION_SFT_*` environment knob, and `sft.env_setup`.
- Read `references/data-preparation.md` for how the prepared dataset is built: loading, the volume-grouped split,
  sample-limit semantics, formatting, CoT targets, the PNG cache, columns, and directory naming.
- Read `references/training-configuration.md` for the LoRA and full-FT trainer settings, FSDP topology, the
  memory recipes, the temperature sampler, loss masking per family, resume and merge.
- Read `references/launcher-catalog.md` to map every launcher name in the repository's `script/sft/` to its
  family, FT mode, resolution, GPU profile and variant flags.
- Read `references/troubleshooting.md` for OOM, FSDP resume, save hangs, dependency drift, masking, wandb,
  merge and push failures.
- Run `scripts/check_sample_limits.py --help` to resolve limit flags through the real `parse_sample_limits`.
- Run `scripts/inspect_prepared_dataset.py --help` to read back a prepared dataset (splits, per-task counts,
  one formatted example, optional image-file existence check).
- Run `scripts/sft_launcher_template.sh --help` for the parameterised two-phase launcher (`DRY_RUN=1` prints
  the commands; it never installs anything unless `RUN_ENV_SETUP=1`).

## Key invariants

1. **The prepared dataset belongs to one model family and one resize.** Preparation bakes the perceived image
   size and the adjusted pixel size into the prompt text via `model_family_name`. Reusing a `qwen25vl` dataset
   for Gemma (or a 512x512 dataset for a native-resolution run) silently trains on wrong numbers.
2. **The default `prepared_ds_dir` name is the dataset identity**:
   `<data_dir>/SFT-CoT_datasets/<model_family_name>/ds__AD<a>_D<d>_TL<t>_all<n>__resized-wh-<W>x<H>`
   (`__original` when unresized; `SFT_datasets` for the non-CoT entry point; `-tooluse` appended for tool-use).
   `<a>/<d>/<t>/<n>` are the requested caps, or the **true** split sizes for unset limits — so the name is only
   known after phase A's load+split stage. Phase A prints it (`Prepared dataset saved at '<dir>'`); hand it to
   phase B as `--prepared_ds_dir`, as the repository launchers and `scripts/sft_launcher_template.sh` do. A
   phase B launched without it re-runs load+split on rank 0 just to recompute the name and then needs the same
   limit, resize and family flags as phase A.
3. **A sample limit of `0` is rejected.** Drop a task by omitting its `--tasks_list_json_path_*`, not by zeroing
   a limit. Unset or `-1` means the full pool.
4. **`--train_sample_limit` is a global cap applied after concatenation.** Smaller than the sum of the per-task
   limits it truncates silently; larger than the concatenated pool it samples **with replacement**.
5. **Validation is carved out first, grouped by `image_file`** (the source 3D volume) and stratified by
   `dataset_name`, so no slice of a validation volume can reach the training split.
6. `--model_family_name` must pass `check_model_supported`, i.e. appear in the vendored `lmms_eval` registry
   (the `vllm_` prefix may be omitted). An unknown key raises before any data is touched.
7. Phase B rebuilds nothing: it needs `--skip_process_dataset true` **and** a prepared dataset on disk. With
   `--prepared_ds_dir` it also skips the load+split stage entirely and loads that directory as-is.

## Safe operating rules

- Never launch phase B on a user's cluster unless asked: a 121K-sample multi-task run is GPU-days, and
  full-FT checkpoints are ~190 GB each at 31B. State the cost first.
- Phase A downloads whole source datasets and writes a PNG per slice next to each NIfTI; check free space.
- Do not run the repository's own `script/sft/*.sh` recipes as-is: they create conda environments, force-install
  packages and pin absolute paths. Use `scripts/sft_launcher_template.sh` or the plain `python -m` commands.
- Do not `pip install` into a user's environment without warning; the SFT stacks are version-locked
  (see `../environment-setup/SKILL.md`) and one wrong `transformers` breaks the run.
- Never push to the Hugging Face Hub (`--push_LoRA`, `--push_merged_model`) without explicit consent; both
  create **private** repos under the logged-in account.
