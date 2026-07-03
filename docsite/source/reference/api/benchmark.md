# `benchmark` pipeline

The scoring backend that turns per-sample model outputs into aggregated metrics.
These modules are normally invoked as command-line entry points (see
{doc}`../cli`); the functions below are the reusable pieces they are built from.

## `eval_utils`

```{eval-rst}
.. automodule:: medvision_bm.benchmark.eval_utils
   :members:
```

## `parse_outputs`

Reads the raw per-sample JSONL written during evaluation and computes per-sample
metrics. `main()` is the CLI entry point; `--task_type` selects the scoring logic.

```{eval-rst}
.. automodule:: medvision_bm.benchmark.parse_outputs
   :members: main
```

## `summarize_AD_task`

Aggregates parsed Angle/Distance results per anatomy.

```{eval-rst}
.. automodule:: medvision_bm.benchmark.summarize_AD_task
   :members: cal_metrics_AD_task, main
```

## `summarize_TL_task`

Aggregates parsed Tumour/Lesion-size results per anatomy.

```{eval-rst}
.. automodule:: medvision_bm.benchmark.summarize_TL_task
   :members: cal_metrics_TL_task, main
```

## `summarize_detection_task`

Aggregates parsed detection results per anatomy.

```{eval-rst}
.. automodule:: medvision_bm.benchmark.summarize_detection_task
   :members: main
```
