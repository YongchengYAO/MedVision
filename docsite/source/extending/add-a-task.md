# Adding a new task

A "task" in MedVision is one evaluation configuration: [a dataset split, the prompt
that gets shown to the model, and the metrics computed on its answer]{.mv-accent}. Tasks are
defined declaratively as YAML files (the benchmark harness is a vendored fork of
`lmms-eval`) that point at Python hook functions. Adding a task therefore means
writing a couple of YAML files and, when the behaviour is genuinely new, a few
hook functions.

:::{seealso}
Full code-level guide in the repository: [`docs/New-Tasks-Guide.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/New-Tasks-Guide.md).
:::

Before starting, it helps to understand what a sample actually contains — see
[Dataset concepts](../dataset/concepts.md). If you are wiring up a new *model*
rather than a new task, go to [Adding a new model](add-a-model.md) instead.

:::{important}
A new task YAML is not run by anything until its `task:` name is registered in a roster JSON
under `tasks_list/` — `tasks_MedVision-{detect,TL,AD}-CoT.json`, the `__train_SFT.json` twins, or
a file under `tasks_list/OOD/`. The eval drivers only run the names listed there, via
`--tasks_list_json_path`. Do this last, after the YAML loads cleanly.
:::

## Folder layout

Every dataset gets its own folder under the harness task tree:

```text
src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/
├── <dataset>/
│   ├── utils.py                                   # per-dataset entry points
│   ├── <dataset>_<task_type>_base-CoT.yaml        # base config (shared knobs)
│   └── <dataset>_<task_type>_<num>_<slice>-CoT.yaml   # concrete task instances
└── medvision/
    ├── medvision_utils.py                         # all shared hook logic
    ├── lmms_eval_specific_kwargs.yaml             # shared prompt kwargs
    └── metadata.yaml
```

Two YAML layers keep things DRY:

- **Base config** (`<dataset>_<task_type>_base-CoT.yaml`) holds everything shared
  across variants of the same task type: `dataset_path` (the HF ID, normally
  `YongchengYAO/MedVision`), `output_type: generate_until`, the four hook
  bindings, and the `metric_list`. It pulls in the shared kwargs and metadata via
  `include:`.
- **Task instance** (`<dataset>_<task_type>_<num>_<slice>-CoT.yaml`) is tiny — it
  `include:`s the base config and sets just `task:` (the unique task ID) and
  `dataset_name:` (the concrete HF subset/split to load).

The per-dataset `utils.py` is a thin shim. The preprocessing modules it binds come from the
companion **`medvision_ds`** package (`medvision_ds.datasets.<Dataset>` — `preprocess_detection`,
`preprocess_biometry`, `preprocess_segmentation`), not from this repository's task tree. It imports the real implementations
from `medvision_utils.py` and, for the factory-style prompt builders, binds the
dataset's own preprocessing module. YAML `!function utils.<name>` references
resolve against that file.

## The four hook functions

Each base YAML binds four hooks (plus the metric aggregators). All live in
`medvision_utils.py`:

- **`doc_to_visual`** — loads the requested NIfTI slice for a sample and returns
  it as a PIL image (the visual input).
- **`doc_to_text`** — builds the text prompt. These are produced by *factory*
  functions named `create_doc_to_text_<TaskType>(...)` that take the dataset's
  preprocessing module and return the actual closure. Each task type has a family
  of factories, for example: `create_doc_to_text_BoxCoordinate_CoT` 
  for detection; `create_doc_to_text_TumorLesionSize_CoT` 
  for size.
- **`doc_to_target`** — returns the ground-truth **value(s)** for the sample, not a formatted
  string: a list of four relative box coordinates for detection, a `[major, minor]` pair for
  size, and a scalar for angle/distance and area (`doc_to_target_BoxCoordinate`,
  `doc_to_target_TumorLesionSize`, `doc_to_target_BiometricsFromLandmarks`,
  `doc_to_target_MaskSize`).
- **`process_results`** — parses the model's raw generation and computes the
  per-sample metrics (`process_results_BoxCoordinate`,
  `process_results_TumorLesionSize`, `process_results_BiometricsFromLandmarks`,
  `process_results_MaskSize`).

The `metric_list` then names aggregators that reduce the per-sample values across
the split, e.g. `aggregate_results_MAE`, `aggregate_results_MRE`,
`aggregate_results_avgMAE`, `aggregate_results_avgMRE`,
`aggregate_results_SuccessRate`, and `aggregate_results_NMAE`. Pick the ones that
match the task: detection **and** Tumour/Lesion size use the `avgMAE` / `avgMRE` /
`SuccessRate` family, while Angle/Distance (`BiometricsFromLandmarks`) uses `MAE` / `MRE` /
`SuccessRate`. `nMAE` (aggregator `aggregate_results_NMAE`) is declared in both the TL and the
AD base configs.

## Task-type labels and naming

The `<task_type>` token in every filename must be one of the fixed labels — the
harness and the downstream parse/summarize steps key off it:

| Label | Task |
| --- | --- |
| `BoxCoordinate` | Detection (bounding boxes) |
| `TumorLesionSize` | Tumour/lesion size estimation (mm) |
| `BiometricsFromLandmarks` | Angle / distance measurement |
| `MaskSize` | Area estimation (new / preview) |

Follow the two-tier naming exactly:

- base config → `<dataset>_<task_type>_base-CoT.yaml`
- task instance → `<dataset>_<task_type>_<num>_<slice>-CoT.yaml` (for example
  `BraTS24_BoxCoordinate_Task01_Axial-CoT.yaml`)

:::{note}
A task instance's `dataset_name:` can read differently from its task-type label —
for example a `BoxCoordinate` task may load a `..._BoxSize_..._Test` subset. That
naming bridge is expected; keep the *file's* `<task_type>` label canonical.
:::

## Where the shared logic lives

Put new hook implementations in `medvision_utils.py` (the single source of truth
for all datasets), then import them from the new dataset's `utils.py`. Reuse an
existing hook whenever the semantics match — most new tasks are a new dataset
folder plus a handful of YAML files and no new Python at all.

