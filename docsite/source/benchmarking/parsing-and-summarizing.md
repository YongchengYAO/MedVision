# Parsing and summarizing results

Running a model over the benchmark (see [Running evaluations](running-evaluations.md)) only produces
raw per-sample predictions. Turning those into scores is two more steps:

1. **Parse** — read each model's raw JSONL, pull the numeric answer out of the model's free-text
   response, and score every sample. This writes a `parsed/` copy of the results.
2. **Summarize** — aggregate the parsed per-sample scores into per-anatomy and per-model tables.

Both steps run offline on CPU and are safe to re-run; nothing here touches the model or the GPU.
For what the metrics mean (IoU, MAE, MRE, nMAE, SuccessRate, …) see the [Benchmark overview](overview.md).

## Where the files live

Evaluation writes one folder per model under a task tag:

```text
Results/<task_tag>/<model>/
    <task_id>_samples_....jsonl     # raw per-sample predictions (one JSON object per line)
    <task_id>_results.json          # lmms-eval run summary for that task
```

Parsing adds a sibling `parsed/` folder; summarizing then drops aggregate files into it and a
human-readable table at the task level:

```text
Results/<task_tag>/<model>/parsed/
    <task_id>_samples_....jsonl       # same samples, now with per-sample metrics attached
    <task_id>_results.json            # run summary updated with avgMAE / avgMRE / avgIoU / SuccessRate
    summary_metrics_<task>_Task.json  # aggregated metrics per anatomy label (written by summarize)
    summary_values_<task>_Task.json   # raw targets + predictions per group (written by summarize)
Results/<task_tag>/
    summary_<task>_task.txt           # formatted per-model tables printed to console + saved here
```

## Step 1 — Parse raw outputs

```bash
python -m medvision_bm.benchmark.parse_outputs \
    --task_type Detection \
    --task_dir Results/MedVision-detect-v2 \
    -p 32
```

Point `--task_dir` at a task folder and every model subfolder under it is processed in turn. (You can
instead target one model with `--model_dir <path>`; exactly one of the two is required.)

For each raw `*.jsonl` the parser sorts samples by `doc_id`, extracts the last numeric values from
within the `<answer></answer>` tags of the model's response (1 value for A/D, 2 for T/L, 4 box
coordinates for Detection), scores each sample,
and writes the annotated copy to `<model>/parsed/`. It also refreshes that model's `<task_id>_results.json`
with the aggregate `avgMAE`, `avgMRE`, `avgIoU`, `SuccessRate`, and the binned `MRE<0.1 … MRE<1.0` fractions.

| Flag | Meaning |
| --- | --- |
| `--task_type {AD,TL,Detection}` | Required. Selects the scoring logic and the expected number of predicted values. |
| `--task_dir <dir>` | Task folder; loops over every model subfolder inside it. |
| `--model_dir <dir>` | Alternative to `--task_dir`: parse a single model folder. |
| `-p`, `--processes <N>` | Worker processes (one JSONL file per worker). Omit for single-process. |
| `--limit <N>` | Score only the first `N` samples per file — handy for a quick smoke test. |
| `--skip_existing` | Leave files that already have a parsed counterpart untouched. |
| `--rm_old` | Delete the model's existing `parsed/` folder before re-parsing (clean rebuild). |

:::{tip}
`--skip_existing` resumes an interrupted parse cheaply; `--rm_old` forces a full re-parse when you have
changed the scoring code or re-run the evaluation. Do not combine them — they pull in opposite directions.
:::

## Step 2 — Summarize into tables

Each task type has its own summarizer. They read the `parsed/` folders, group samples by anatomy, and
emit `summary_metrics_<task>_Task.json` / `summary_values_<task>_Task.json` per model plus a task-level
`summary_<task>_task.txt`.

```bash
# Angle / Distance
python -m medvision_bm.benchmark.summarize_AD_task \
    --task_dir Results/MedVision-AD-v2-CoT -p 32 --skip_model_wo_parsed_files

# Tumour / Lesion size  (note the removed-samples caveat below)
python -m medvision_bm.benchmark.summarize_TL_task \
    --task_dir Results/MedVision-TL-v2-CoT -p 32 --skip_model_wo_parsed_files \
    --removed_samples_dir Data/Datasets

# Detection
python -m medvision_bm.benchmark.summarize_detection_task \
    --task_dir Results/MedVision-detect-v2 -p 32 --skip_model_wo_parsed_files
```

Shared flags across all three:

| Flag | Meaning |
| --- | --- |
| `--task_dir <dir>` / `--model_dir <dir>` | Aggregate every model under a task tag, or a single model. One is required. |
| `-p`, `--processes <N>` | Parallelize the per-label metric computation. |
| `--limit <N>` | Match a limited parse run; also suffixes output filenames with `_limit<N>`. |
| `--skip_model_wo_parsed_files` | Ignore model folders that were never parsed. Valid only with `--task_dir`. |

**What each summarizer groups on:**

- **A/D** groups by `dataset_metricType_metricKey` labels and also reports rolled-up group averages
  (FeTA-Distance, Ceph-Angle, Ceph-Distance). Samples whose ground-truth value is below the near-zero
  threshold (`0.1`) are dropped so relative error stays meaningful.
- **T/L** groups by anatomy label crossed with imaging modality and slice plane, weighting per-label
  averages by sample count.
- **Detection** groups by anatomy/modality/slice, then additionally splits regions into an **anatomy**
  bucket vs a **tumour/lesion (T/L)** bucket. Regions tagged miscellaneous/others and regions with
  fewer than the minimum group size (50 samples) are excluded from the grouped means, and a
  `random_detection` baseline folder, if present, is skipped.

### T/L only: excluding multi-cluster samples

The published T/L benchmark is scored against v1.0.0 annotations, but dataset release v1.1.0 identified a
set of samples whose target spanned multiple disconnected clusters. To reproduce the reported numbers
you must drop those samples at summarize time:

```bash
--removed_samples_dir Data/Datasets
```

With this set, the summarizer looks for `multi_cluster_samples_v1.0.0_to_v1.1.0.json` inside each
dataset folder under that root (override the filename with `--removed_samples_filename`) and skips any
matching sample. The resulting output files gain a `_filtered` suffix so they never overwrite an
unfiltered run. This flag exists only on `summarize_TL_task`.

:::{warning}
Omitting `--removed_samples_dir` for T/L silently includes the multi-cluster samples and yields numbers
that do not line up with the [paper](https://arxiv.org/abs/2511.18676).
:::
