---
name: medvision-pipeline
description: How to run the MedVision benchmark (eval → parse_outputs → summarize, per task), download datasets, set up the environment, and where the SFT launchers live. Use when running or scripting any benchmark, dataset, environment, or SFT step.
---

## Benchmark Pipeline (4 steps)

### Step 1: Run model evaluation
```bash
# Each eval script runs a specific model on MedVision tasks
python -m medvision_bm.benchmark.eval__<model_name> [args]
# Or use shell scripts in script/benchmark-AD/, script/benchmark-TL/, script/benchmark-detect/
```

### Step 2: Parse outputs
```bash
python -m medvision_bm.benchmark.parse_outputs \
  --task_type [AD|TL|Detection] \
  --task_dir <path> \
  --model_dir <path> \
  [--limit N] [--skip_existing] [-p N_PROCESSES] [--rm_old]
```

### Step 3: Summarize results
```bash
python -m medvision_bm.benchmark.summarize_AD_task --task_dir <path> --model_dir <path>
python -m medvision_bm.benchmark.summarize_TL_task --task_dir <path> --model_dir <path>
python -m medvision_bm.benchmark.summarize_detection_task --task_dir <path> --model_dir <path>
```

### Step 4: LLM-judge re-parsing (this is what the reported metrics come from)

Step 2's strict parser accepts an answer **only** inside `<answer></answer>`, so a model that measured
correctly but formatted differently is scored as a failure. This pass re-reads every response with a
second model (reader `gemma-4-31b`) and re-scores with the *same* metric code, writing a format-robust
twin of each report under `llm-parsed_<reader>/`. **Published MedVision numbers are computed from these
records**, so stopping at step 3 gives metrics that will not match the leaderboard.

```bash
# single entry point; it re-roots itself to the repo, so any cwd works
bash script/llm-parsing/run_llm_parsing.sh              # every step, in order
bash script/llm-parsing/run_llm_parsing.sh --list       # show what would run, only
bash script/llm-parsing/run_llm_parsing.sh --from full  # that step onward
bash script/llm-parsing/run_llm_parsing.sh --fresh      # FULL re-judge (destructive, prompts)
```

Then re-run step 3 against the recovered records:

```bash
python -m medvision_bm.benchmark.summarize_detection_task \
  --task_dir <path> --model_dir <path> --parsed_dirname llm-parsed_gemma-4-31b
```

This pipeline ships in the **repository checkout** under `script/llm-parsing/` — it is not part of the
installed `medvision_bm` package — and needs its own judge environment (see `script/llm-parsing/README.md`).

## SFT Training
Use scripts in folder: script/sft/

## Dataset Download

```bash
export MedVision_PLANNER_VERSION=1.0.0   # required: the loader hard-fails without it
python -m medvision_bm.benchmark.download_datasets \
  --data_dir <path> \
  --tasks_json <path_to_tasks_json>
```

## Environment Setup

```bash
python -m medvision_bm.benchmark.env_setup
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <path>
```
