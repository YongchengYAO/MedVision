# Quickstart

This walkthrough runs the full MedVision benchmark pipeline end to end on a tiny
slice of the **Angle/Distance** task using Qwen2.5-VL-7B. It is deliberately
small — a smoke test to confirm your environment, data, and outputs line up
before you commit to a real sweep.

The flow is always the same three stages: **evaluate → parse → summarize**.

## 1. Install and configure the environment

Install the package (see [Installation](installation.md) for the complete
setup), then point MedVision at your data and pin the dataset planner version:

```bash
export MedVision_DATA_DIR=/path/to/MedVision/Data
export MedVision_PLANNER_VERSION=1.1.1
```

`MedVision_DATA_DIR` tells the loader where the datasets live, and
`MedVision_PLANNER_VERSION` locks the sample-selection logic to a specific
dataset release so results are reproducible.

:::{note}
`MedVision_ACK_RELEASE=1.1.1` is only required when you pin
`MedVision_PLANNER_VERSION` *below* the latest release. Pinning the latest, as
above, needs no acknowledgement variable.
:::

## 2. Evaluate on 10 Angle/Distance samples

Run the Qwen2.5-VL driver against the Angle/Distance CoT task list, capped at 10
samples per task:

```bash
python -m medvision_bm.benchmark.eval__qwen2_5_vl \
    --model_hf_id Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name Qwen2.5-VL-7B-Instruct \
    --data_dir "$MedVision_DATA_DIR" \
    --tasks_list_json_path tasks_list/tasks_MedVision-AD-CoT.json \
    --task_status_json_path completed_tasks/completed_tasks_MedVision-AD-CoT.json \
    --results_dir Results/MedVision-AD-CoT \
    --sample_limit 10
```

This writes one **per-sample JSONL** file per task under
`Results/MedVision-AD-CoT/Qwen2.5-VL-7B-Instruct/`, containing each model
response alongside its ground truth. Completed tasks are recorded in the
`--task_status_json_path` file, so an interrupted run resumes where it left off.

:::{note}
The first invocation provisions the model's inference environment (PyTorch,
vLLM, the vendored lmms-eval, plus the dataset install), which can take a while.
A GPU is required.
:::

## 3. Parse the raw outputs into metrics

Turn the raw JSONL responses into structured, scored records:

```bash
python -m medvision_bm.benchmark.parse_outputs \
    --task_type AD \
    --task_dir Results/MedVision-AD-CoT \
    -p 8
```

This extracts the predicted angles and distances from each response and computes
per-sample **parsed metrics** (MAE and MRE, in degrees and mm), written to a
`parsed/` folder next to the JSONL files. `-p 8` runs 8 worker processes.

## 4. Summarize into per-anatomy tables

Aggregate the parsed metrics across all samples:

```bash
python -m medvision_bm.benchmark.summarize_AD_task \
    --task_dir Results/MedVision-AD-CoT \
    -p 8
```

This produces a **per-anatomy summary** (CSV and JSON) reporting mean MAE and MRE
for the model. Add `--skip_model_wo_parsed_files` to ignore any model folders
that were never parsed.

:::{tip}
This is a 10-sample smoke run, not a benchmark result. For the full task suite,
larger sample limits, and the other models and tasks, see
[Running evaluations](../benchmarking/running-evaluations.md) and the
[Benchmark overview](../benchmarking/overview.md).
:::
