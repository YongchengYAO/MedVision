# Adding a task (and regenerating task YAMLs)

A MedVision "task" is one unit of evaluation: one dataset, one annotation family, one task id,
one imaging plane, one prompt variant. It is defined by a **pair** of YAML files plus a
per-dataset `utils.py`, and it only ever *runs* if its name appears in a task-list JSON.

## File layout

```
lmms_eval/tasks/
  medvision/
    lmms_eval_specific_kwargs.yaml     # shared; per-model extra kwargs + default pre/post prompt
    metadata.yaml                      # shared; `metadata: [{version: ...}]`
    medvision_utils.py                 # ALL shared prompt / target / metric / image-size code
  <Dataset>/
    utils.py                           # thin adapter: imports from medvision_utils + medvision_ds
    <Dataset>_<TaskType>_base[-CoT].yaml          # base ("template") config, NO `task:` key
    <Dataset>_<TaskType>_<TaskID>_<Plane>[-CoT].yaml   # one per subtask
  _task_utils/                         # upstream lmms-eval helpers (file/video/metric utilities)
```

`lmms_eval/tasks/__init__.py` builds the index by **walking the whole tree**: any `*.yaml` with a
`task:` key is registered as a task, any `group:` key as a group, and each entry of a `tag:` key
becomes a callable tag. A file without `task:` or `group:` (a base config) is never registered —
that is what makes `_base` templates invisible to `--tasks`. `__pycache__` and
`.ipynb_checkpoints` are the only skipped directories, so nothing else needs registering: **the
directory is the registry.**

## Base YAML anatomy

```yaml
include:
  - ../medvision/lmms_eval_specific_kwargs.yaml
  - ../medvision/metadata.yaml
tag: MedVision-BoxCoordinate,BraTS24
dataset_path: YongchengYAO/MedVision
dataset_kwargs:
  token: True
  trust_remote_code: True
  # download_mode: force_redownload
test_split: test
fewshot_split: test
num_fewshot: 0
output_type: generate_until
doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text_BoxCoordinate_CoT
doc_to_target: !function utils.doc_to_target_BoxCoordinate
process_results: !function utils.process_results_BoxCoordinate
metric_list:
- metric: avgMAE
  aggregation: !function utils.aggregate_results_avgMAE
  higher_is_better: false
- metric: avgMRE
  aggregation: !function utils.aggregate_results_avgMRE
  higher_is_better: false
- metric: SuccessRate
  aggregation: !function utils.aggregate_results_SuccessRate
  higher_is_better: true
```

| key | meaning |
| --- | --- |
| `include` | shared fragments merged in. Paths are relative **to the file's own directory**; the two `../medvision/...` entries are required (per-model extra kwargs + the metadata version). Order between them does not matter — both orders appear in the shipped files. |
| `tag` | a single scalar of the form `MedVision-<TaskType>,<Dataset>` (e.g. `MedVision-BoxCoordinate,BraTS24`). Despite the comma, `lmms_eval` registers the **whole string as one tag** — `tasks/__init__.py:462-464` wraps a `str` in a one-element list and nothing splits on `,` — so `--tasks MedVision-TumorLesionSize` and `--tasks BraTS24` match nothing and raise `ValueError: Tasks not found`. Pass the exact composite string, or a wildcard such as `--tasks 'MedVision-TumorLesionSize*'`. MedVision itself drives tasks one name at a time from `tasks_list/*.json`. |
| `dataset_path` | the HF dataset id (`YongchengYAO/MedVision`). Overriding it with a bare string in a *task* YAML replaces the repo id instead of selecting a config — a common typo, see `troubleshooting.md`. |
| `dataset_kwargs` | `token: True` (gated dataset), `trust_remote_code: True` (the loader script). |
| `test_split` / `fewshot_split` / `num_fewshot` | MedVision is zero-shot on the `test` split. |
| `output_type` | always `generate_until`. |
| the four hooks | `doc_to_visual` (image), `doc_to_text` (prompt), `doc_to_target` (ground-truth string), `process_results` (per-sample metrics). `!function utils.<name>` resolves against the **sibling** `utils.py`. |
| `metric_list` | one entry per metric: `metric`, `aggregation: !function utils.aggregate_results_*`, `higher_is_better`. |

Metric sets differ by family: detection and T/L use `avgMAE`, `avgMRE`, `SuccessRate` (T/L adds
`nMAE`); A/D uses `MAE`, `MRE`, `SuccessRate`, `nMAE` (scalar, not averaged over coordinates).

## Per-task YAML anatomy

Three lines, nothing else:

```yaml
include: BraTS24_BoxCoordinate_base-CoT.yaml
task: BraTS24_BoxCoordinate_Task01_Axial-CoT
dataset_name: BraTS24_BoxSize_Task01_Axial_Test
```

- `include` — the base template in the same directory (a bare file name, no `../`).
- `task` — the registered task name; this is the string that goes into `tasks_list/*.json` and
  becomes the results sub-directory. **No trailing whitespace** (YAML strips it, so a stray space
  produces a silent duplicate of another task).
- `dataset_name` — the **dataset config** name on the Hub. The task name and the config name are
  deliberately different namespaces: detection tasks are named `BoxCoordinate` but load
  `..._BoxSize_...` configs.

## Naming conventions

- base: `<Dataset>_<TaskType>_base[-<Variant>].yaml`
- task: `<Dataset>_<TaskType>[_<Sub>]_<TaskID>_<Plane>[-<Variant>].yaml`, with `task:` equal to
  the file stem.

Fixed `<TaskType>` labels:

| `<TaskType>` | task family |
| --- | --- |
| `BoxCoordinate` | Detection (bounding box) |
| `TumorLesionSize` | Tumor/Lesion size (T/L) |
| `BiometricsFromLandmarks` | Angle/Distance (A/D); an extra `Angle` / `Distance` segment splits the two |
| `MaskSize` | area estimation (new/preview) |

`<TaskID>` is `TaskNN`; `<Plane>` is `Axial` / `Coronal` / `Sagittal`. Variants seen in the tree:
`-CoT` (chain-of-thought, the benchmark default), `-VP` (visual prompt), `-VP-woMedImg`,
`-CoT-woInstruct`, `-CoT-scaledPS`, and no suffix (plain). Each variant has its **own base
template** with a different `doc_to_text`, so a variant task YAML must include the matching base
— including the wrong base is a silent prompt swap that nothing detects.

## The `create_doc_to_text_*` factory pattern

`medvision_utils.py` holds every prompt builder, but a prompt needs the dataset's own benchmark
plan (labels map, image description, per-task metadata). That plan is supplied by the dataset
package, so the shared builders are **factories** that take the dataset's `preprocess_*` module
and return the concrete hook. The per-dataset `utils.py` is only the wiring:

```python
from lmms_eval.tasks.medvision.medvision_utils import (
    aggregate_results_avgMAE, aggregate_results_avgMRE, aggregate_results_SuccessRate,
    aggregate_results_NMAE,
    create_doc_to_text_BoxCoordinate, create_doc_to_text_BoxCoordinate_CoT,
    create_doc_to_text_TumorLesionSize, create_doc_to_text_TumorLesionSize_CoT,
    create_doc_to_text_MaskSize,
    doc_to_target_BoxCoordinate, doc_to_target_TumorLesionSize, doc_to_target_MaskSize,
    doc_to_visual,
    process_results_BoxCoordinate, process_results_TumorLesionSize, process_results_MaskSize,
)
from medvision_ds.datasets.BraTS24 import (
    preprocess_biometry, preprocess_detection, preprocess_segmentation,
)

doc_to_text_BoxCoordinate      = create_doc_to_text_BoxCoordinate(preprocess_detection)
doc_to_text_BoxCoordinate_CoT  = create_doc_to_text_BoxCoordinate_CoT(preprocess_detection)
doc_to_text_TumorLesionSize    = create_doc_to_text_TumorLesionSize(preprocess_biometry)
doc_to_text_MaskSize           = create_doc_to_text_MaskSize(preprocess_segmentation)
```

Mapping from annotation family to `preprocess_*` module: detection -> `preprocess_detection`;
T/L **and** A/D -> `preprocess_biometry` (both are stored as biometry); MaskSize ->
`preprocess_segmentation`. A/D also has factories for the target and the result processor
(`create_doc_to_target_BiometricsFromLandmarks_scaledPS`,
`create_process_results_BiometricsFromLandmarks_scaledPS`) because the scaled-pixel-size variant
rescales the ground truth too.

`doc_to_visual`, `doc_to_target_*` and `process_results_*` (non-scaledPS) are plain functions —
import them directly, no factory.

Inside the builders, the prompt gets the perceived image size and adjusted pixel size from
`get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)` — see
`image-size-dispatch.md`. That is why a new *model* needs no task edit, and a new *task* needs no
model edit.

## Copyable minimal pair

`lmms_eval/tasks/MyDataset/MyDataset_TumorLesionSize_base-CoT.yaml`:

```yaml
include:
  - ../medvision/lmms_eval_specific_kwargs.yaml
  - ../medvision/metadata.yaml
tag: MedVision-TumorLesionSize,MyDataset
dataset_path: YongchengYAO/MedVision
dataset_kwargs:
  token: True
  trust_remote_code: True
test_split: test
fewshot_split: test
num_fewshot: 0
output_type: generate_until
doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.doc_to_text_TumorLesionSize_CoT
doc_to_target: !function utils.doc_to_target_TumorLesionSize
process_results: !function utils.process_results_TumorLesionSize
metric_list:
- metric: avgMAE
  aggregation: !function utils.aggregate_results_avgMAE
  higher_is_better: false
- metric: avgMRE
  aggregation: !function utils.aggregate_results_avgMRE
  higher_is_better: false
- metric: SuccessRate
  aggregation: !function utils.aggregate_results_SuccessRate
  higher_is_better: true
- metric: nMAE
  aggregation: !function utils.aggregate_results_NMAE
  higher_is_better: false
```

`lmms_eval/tasks/MyDataset/MyDataset_TumorLesionSize_Task01_Axial-CoT.yaml`:

```yaml
include: MyDataset_TumorLesionSize_base-CoT.yaml
task: MyDataset_TumorLesionSize_Task01_Axial-CoT
dataset_name: MyDataset_TumorLesionSize_Task01_Axial_Test
```

`lmms_eval/tasks/MyDataset/utils.py`:

```python
from lmms_eval.tasks.medvision.medvision_utils import (
    aggregate_results_avgMAE,
    aggregate_results_avgMRE,
    aggregate_results_NMAE,
    aggregate_results_SuccessRate,
    create_doc_to_text_TumorLesionSize_CoT,
    doc_to_target_TumorLesionSize,
    doc_to_visual,
    process_results_TumorLesionSize,
)
from medvision_ds.datasets.MyDataset import preprocess_biometry

doc_to_text_TumorLesionSize_CoT = create_doc_to_text_TumorLesionSize_CoT(preprocess_biometry)
```

Then register it — without this the task never runs:

```json
{
  "MyDataset_TumorLesionSize_Task01_Axial-CoT": 1000
}
```

in `tasks_list/tasks_MedVision-TL-CoT.json`. The pipeline reads only the **top-level keys**
(`medvision_bm.utils.utils.load_tasks`); the counts are informational.

## Adding a task on a *new* dataset

The dataset side (`medvision_ds`) must exist first: `utils.py` imports
`medvision_ds.datasets.<Dataset>`, and the config name in `dataset_name` must be published in the
dataset repository. Dataset config naming, annotation versions and `MedVision_PLANNER_VERSION`
are owned by `../../dataset-and-tasks/SKILL.md`.

## Regenerating task YAMLs and task lists (reference only)

The repository generates lists rather than editing them by hand. These need the dataset source
tree and network access, so treat them as reference:

- `python -m medvision_bm.utils.configs_to_tasks --data_dir <dir> --configs_csv <csv> --out <json>
  [--families BoxSize,TumorLesionSize,BiometricsFromLandmarks,MaskSize] [--planes Axial,Coronal,Sagittal]
  [--split train|test|all] [--cot] [--limit N] [--no-count] [--no-streaming]`
  converts a dataset-config CSV into a task-list JSON. `--cot` appends `-CoT` to the task names;
  `--no-count` skips dataset loading and writes `0` counts (fast naming-only run); counting
  otherwise streams every config.
- The repository's `script/misc/convert_configs_to_tasks_v*.sh` wrappers call it once per
  family/plane/split with `MedVision_PLANNER_VERSION` exported and the CSVs from
  `dataset-info/dataset-configs/<version>/ConfigurationsList_All.csv` as input.
- `script/misc/regen_all_tasks.py --version <planner version> [--data_dir ...] [--dataset_path ...]
  [--cache ...] [--out_dir ...] [--no-count]` regenerates the per-version task inventories. It
  reads the *loader's own* `_ANNOTATION_INDEX` and `_PAUSED_ANNOTATIONS` tables rather than a
  hand-maintained CSV, and caches counts by `(config, resolved annotation version)` — run the
  versions oldest-first so later ones reuse the cache.
- Only `regen_all_tasks_v1.2.0-v1.4.0.sh` exports `PYTHONPATH=<repo>/src`; the
  `convert_configs_to_tasks_v*.sh` wrappers do **not**, so export it yourself before running the CSV
  converter against an edited checkout. It matters because a non-editable `medvision_bm` in
  `site-packages` silently shadows a checkout, so a freshly edited helper would otherwise be
  imported as its installed copy.

**`tasks_list/*.json` is the sole authority for which YAMLs are used.** The tree ships far more
task YAMLs than any list references (variant suffixes, planes and datasets that are not part of
the published benchmark). `scripts/list_task_yamls.py --tasks-json <lists...>` prints exactly
which are referenced, and `--unused` shows the rest.

## Validation checklist for a new task

1. `python scripts/list_task_yamls.py --dataset <Dataset>` — exit 0; the base appears under
   `[base]`, the task under `[task]` with the right `dataset_name` and inherited `tag`.
2. `python scripts/list_task_yamls.py --tasks-json <your list>` — the new name shows `USED`, and
   no "has no task YAML" problem is reported.
3. `python -c "import lmms_eval.tasks.<Dataset>.utils"` inside an evaluation environment — catches
   a missing `medvision_ds` dataset module or a misspelled factory import.
4. Two-sample run of the new task with any cheap model; read the logged prompt and target.
