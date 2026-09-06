---
name: rft
description: "Builds verl-ready parquet datasets from MedVision tasks (medvision_bm.rft.verl.build_parquet_ds, the checkpointed/sharded and with-testset variants, sample-limit semantics, the model_family_name image-processor constraint, the prompt/reward parquet schema), explains the GRPO reinforcement fine-tuning recipe of MedVision-V0 that runs in the external verl fork (branch medvision-rl: format/process/answer rewards with exp(-error), CIoU for detection, temperature-scaled task mixing, epoch-level curriculum, sequential and multi-task recipes, DATASET_ROOT/BASE_MODEL_PATH variables), and evaluates the trained model with eval__medvision-model-rft. Use for RFT, RL fine-tuning, GRPO, verl, reward design, parquet dataset building, or reproducing MedVision-V0 training."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision RFT: verl parquet builders, GRPO recipe, RFT-model evaluation

Use this sub-skill when a task involves the **reinforcement fine-tuning (RFT)** stage of MedVision: turning MedVision
tasks into verl parquet datasets, understanding or reproducing the GRPO recipe behind `MedVision-V0-7B`, or
benchmarking a verl-trained checkpoint. Terminology: Detection / Tumor-Lesion size (T/L) / Angle-Distance (A/D);
`medvision_bm` (this package), `medvision_ds` (dataset package), the verl fork branch `medvision-rl` (external).

## Route here for

- `python -m medvision_bm.rft.verl.build_parquet_ds` and its variants (`build_parquet_ds__checkpointed` with
  `--shard_size`, `build_parquet_ds_with_testset[__checkpointed]`), their 23-30 flags, sample-limit rules, output
  layout `<data_dir>/verl_datasets/<model_family_name>/ds__AD<a>_D<d>_TL<t>_all<total>__resized-hw-<H>x<W>/`.
- The parquet row contract verl consumes (`prompt`, `images`, `ground_truth`, `data_source`, `ability`,
  `reward_model`, `extra_info`) and the RFT system prompts (`rft_prompts.SYSTEM_PROMPT[_LITE]`).
- Why a dataset is valid only for models sharing the image processor of `--model_family_name` / `--model_hf`.
- Rewards (`r = r_format + r_process * r_answer`, additive ablation, `exp(-error)`, CIoU), temperature-scaled task
  mixing (T=8), epoch-level curriculum, the five fork recipes and the MedVision-V0 configuration (sequential
  A/D -> T/L -> detection, 4x H200, GRPO hyper-parameters).
- `python -m medvision_bm.benchmark.eval__medvision-model-rft` (`--use_system_prompt`, `--reshape_image_hw 512x512`,
  `--lmms_eval_opt_deps medvision_v0` install path) and `medvision_bm.rft.verl.patch_layer_name` (LoRA wrapper cleanup).

## Do not use for

- SFT data construction internals (`load_split_limit_dataset`, `format_dataset`, PNG cache, samplers) and SFT
  launchers -> `../sft/SKILL.md` (the RFT builders reuse those helpers through `medvision_bm.sft.sft_utils`).
- Task lists, config naming, dataset download and `MedVision_PLANNER_VERSION` -> `../dataset-and-tasks/SKILL.md`.
- Installing `medvision_bm` / `medvision_ds`, pins, install order, wheel builds -> `../environment-setup/SKILL.md`.
- General benchmark evaluation, results layout and resume -> `../benchmark-evaluation/SKILL.md`; metrics and
  `parse_outputs` / `summarize_*` -> `../results-parsing-and-metrics/SKILL.md`.
- Terms and file layout -> `../../references/concepts-and-glossary.md`; cross-cutting failures ->
  `../../references/troubleshooting.md`.

## Prerequisites (CPU is enough for data building)

1. `medvision_bm` importable **with the target family's SFT extras** (the builders import `medvision_bm.sft.sft_utils`
   and load `--model_hf`'s image processor via transformers); `medvision_ds` installed in `<data_dir>/src`.
2. `export MedVision_DATA_DIR=<data_dir>` and `export MedVision_PLANNER_VERSION=<version>` (`1.0.0` = paper data).
3. SFT-namespace task lists `tasks_MedVision-{AD,TL,detect}__train_SFT.json`.
4. GPU (4x H200 for the paper) and the fork's environment only for training; GPU + vLLM only for evaluation.

## Fast paths

**Build a small dataset and check it (CPU):**

```
MedVision_PLANNER_VERSION=1.0.0 bash scripts/build_parquet_ds.sh --data-dir <data_dir> \
    --tasks-tl <tasks_dir>/tasks_MedVision-TL__train_SFT.json --train-limit-tl 200 --val-limit-tl 20 --dry-run
python scripts/inspect_parquet_ds.py --path <data_dir>/verl_datasets/qwen25vl/ds__AD0_D0_TL200_all200__resized-hw-512x512
```

**Build the paper's 1M detection set without OOM:** add `--tasks-detect ... --train-limit-detect 1000000
--val-limit-detect 500 --checkpointed --shard-size 50000 --workers-format 64` (32 GB box: `--workers-format 16
--shard-size 20000`); rerun the same command to resume from `checkpoint.json`.

**Train (GPU, external):** in the verl fork, `DATASET_ROOT=<parquet dir> BASE_MODEL_PATH=<SFT ckpt> bash
examples/grpo_trainer/train__rft-sequential__1-AD.sh` (then `2-TL`, `3-detection`; `DRY_RUN=1` previews). Never
pass a Hub id directly to verl; use `BASE_MODEL_HF` so the recipe downloads it first.

**Evaluate (GPU):** `python -m medvision_bm.benchmark.eval__medvision-model-rft --model_hf_id <merged model>
--reshape_image_hw 512x512 --use_system_prompt ...` with the `-CoT` eval task lists, then parse / summarise.

## References and scripts

- Read `references/workflows.md` for the end-to-end sequence (prerequisites, smoke build, the five paper-scale
  builds, checkpointed/resume rules, RAM and disk budgets, training hand-off, evaluation commands, where to stop).
- Read `references/cli-reference.md` for every flag and default of the four builders, `patch_layer_name`, and
  `eval__medvision-model-rft`, plus the `parse_sample_limits` rules and the unused flags.
- Read `references/parquet-schema.md` when you need the exact Arrow schema, per-task `data_source` / `ability` /
  `ground_truth` / `extra_info` values, the system-prompt text, image encoding, or the mixed-task union behaviour.
- Read `references/rft-recipes.md` for rewards, task mixing, curriculum options, the recipe table with environment
  variables, the shared GRPO configuration, the merge step and the transformers<5 pin (reference only; external fork).
- Read `references/troubleshooting.md` when a build OOMs or fails on env vars / limits / family keys, when a parquet is
  reused for the wrong model family, when fork recipes reject variables or rewards look wrong, or when the pinned eval
  stack cannot read a merged checkpoint.
- Run `scripts/build_parquet_ds.sh --help` / `--dry-run` to build with explicit paths and small defaults (wraps the
  normal and checkpointed builders; no env creation).
- Run `scripts/inspect_parquet_ds.py --path <file or dir>` to print schema, row counts, per-task counts, the first
  row and the image encoding of any built dataset (exit 2 if a verl column is missing).

## Key invariants

1. The output directory name **is** the dataset identity: limits, resize and CoT flag are encoded in it and the fork
   recipes expect the exact names (e.g. `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512`).
2. T/L and A/D prompts embed the image size and pixel size **as perceived by `model_family_name`'s processor**;
   detection prompts do not. Rebuild for every new family; never share a `qwen25vl` parquet with Gemma/InternVL models.
3. Keep `--train_sample_limit` equal to the sum of per-task limits; a smaller value truncates silently, a larger one
   samples with replacement (seeded by `SEED`).
4. A sample limit of `0` is rejected; skip a task by omitting its task list. Validation defaults to 100 per task.
5. Keep the CoT instruction in RFT prompts (default); `--without_cot_instruction` removes the landmark ground truth the
   process reward needs and is marked deprecated in source.
6. The training system prompt must be re-injected at evaluation (`--use_system_prompt`) with the same resize.

## Safe operating rules

- Building downloads datasets on first use and can consume tens to hundreds of GB; preview with `--dry-run`, start
  with the small defaults, and never write into a user's `Data/` without asking.
- Training and evaluation require GPUs; document commands, do not launch them on CPU hosts. Never run private
  launcher directories; use the bundled scripts or the public `python -m` entry points.
- Do not `pip install` into a user's environment without pointing at the pins in `../environment-setup/SKILL.md`
  (the eval path pins `transformers==4.54.1`, `vllm==0.10.0`).
- Treat the verl fork as external: quote its documented options, do not invent reward or curriculum settings.
