# Process accuracy and equation accuracy

Two CoT-level analyses that open up a single benchmark number into the reasoning that produced it. Both re-read existing
result records and write beside them; both are CPU-only and take seconds to minutes per model.

|  | Process accuracy | Equation accuracy |
|---|---|---|
| Question | Which reasoning step went wrong? | Did the model compute the equation it wrote? |
| Compared against | **ground truth** (landmarks + measurements) | the **model's own** stated answer |
| Needs ground truth / dataset files | yes | no |
| Scripts | `scripts/analyze_process_accuracy_TL.py`, `..._AD.py` | `scripts/analyze_equation_accuracy_TL.py`, `..._AD.py` |
| Output suffix | `_proc_acc` | `_eq_acc` |

Both read `resps` - the **raw model response** - not `filtered_resps`. They are therefore independent of which parser
produced the folder: pointing them at `llm-parsed_<judge>/*.jsonl` gives the same per-sample numbers as `parsed/`,
because the judge pass only renames the prediction key and leaves `doc` and `resps` untouched. What can differ is the
sample set.

---

## 1. Process accuracy - what each step measures

### T/L, 4 steps (`analyze_process_accuracy_TL.py`)

| Step | Model output | Ground truth | Metric |
|---|---|---|---|
| 1 | major-axis endpoints P1, P2 (relative coordinates) | GT P1, P2 from the landmark file | normalised L2 distance, `sqrt(dx^2+dy^2)/sqrt(2)`, endpoint pairing taken as `min(d1, d2)` so a swapped pair is not punished |
| 2 | minor-axis endpoints P3, P4 | GT P3, P4 | same |
| 3 | major-axis length (mm) | `biometric_profile.metric_value_major_axis[0]` | MRE, plus nMAE against the physical slice diagonal |
| 4 | minor-axis length (mm) | `metric_value_minor_axis[0]` | MRE, plus nMAE |

Per-sample record keys: `step{1,2}_pred`, `step{1,2}_normL2`, `step{3,4}_pred`, `step{3,4}_MRE`, `step{3,4}_nMAE`.
A tool-use fallback recovers steps 3/4 when the response answered through a single `<tool_call>` instead of the step
tags.

### A/D, 3 steps (`analyze_process_accuracy_AD.py`)

| Step | Distance samples | Angle samples | Metric |
|---|---|---|---|
| 1 | landmark 1 | line-1 endpoints | normalised L2 |
| 2 | landmark 2 | line-2 endpoints | normalised L2 |
| 3 | the distance (mm) | the angle (deg) | MRE; nMAE for distances only |

Ground truth is resolved through the dataset's benchmark plan
(`medvision_ds.datasets.<pkg>.preprocess_biometry.benchmark_plan`, selected by
`medvision_bm.utils.configs.DATASETS_NAME2PACKAGE`) to learn which landmark keys the metric uses, then the coordinates
are read from `doc["landmark_file"]`. **`medvision_ds` is therefore mandatory for A/D process accuracy**; without it
every sample fails with `benchmark_plan error: ...`.

### Near-zero A/D ground truths

`AD_NEAR_ZERO_GT_THRESHOLD = 0.1` (imported from `medvision_bm.utils.configs`) is applied at aggregation time in
`analyze_process_accuracy_AD.py`: a sample whose GT scalar is below the threshold is counted in `n_ignored` and its
step-3 MRE/nMAE are excluded from the averages. Steps 1 and 2 (coordinate errors) are *not* excluded. `n_samples`,
`n_valid` and `n_ignored` are all reported per label, so the exclusion is always visible in the summary JSON.
The same constant is applied in `analyze_equation_accuracy_AD.py` (see below) and by `summarize_AD_task`
(`../../results-parsing-and-metrics/SKILL.md`).

### Aggregation and outputs

Per-sample records are grouped by label:

- T/L: `"<renamed label> @ <MR|CT|US|XR|PET> (<S|C|A>)"`, resolved through the benchmark plan plus
  `medvision_bm.utils.configs.label_map_rename`.
- A/D: `"<dataset>_<metric_type>_<metric_key>"` - built from record fields only.

| File | Where | Content |
|---|---|---|
| `<stem>_proc_acc[_filtered].jsonl` | beside each input JSONL | one record per sample, all step fields |
| `summary_proc_acc_{TL,AD}_metrics[_filtered].json` | in the parsed folder | per label: `step*_avg_*`, `n_samples`, `success_rate`, plus `n_valid` and `n_ignored` (A/D only) — for T/L process accuracy read `success_rate` |
| `summary_proc_acc_{TL,AD}_model[_filtered].txt` | in the model folder | weighted average + per-label table |
| `summary_proc_acc_{TL,AD}_task[_filtered].txt` | in the task folder (`--task_dir` only) | the same, one block per model |

`success_rate` counts samples where every step produced a value.

### scaledPS inputs

A file whose **name** contains `scaledPS` is handled specially: the model was prompted with a scaled pixel size, so its
lengths are in scaled mm and the eval `target` field holds the scaled ground truth, while `biometric_profile` still
holds the unscaled value. The analyzers use the eval `target` for those files and rebuild a matching scaled physical
diagonal for the nMAE denominator via `_compute_physical_diagonal` from the vendored eval utilities. If that import
fails the script prints one loud `[error] cannot import _compute_physical_diagonal ...` message and every scaledPS nMAE
becomes NaN while the MREs stay correct. Non-scaledPS files are unaffected.

---

## 2. Equation accuracy - arithmetic independent of ground truth

For each step the analyzer extracts the equation the model wrote inside `<step-k-reasoning>`, converts it to a Python
expression, evaluates it with a restricted AST evaluator (numeric operators plus a whitelist of `math` functions -
`sqrt`, `acos`, `atan2`, ... - nothing else), and compares that value with the model's own `<step-k-answer>`:

```
equation_MRE = |model_answer - python_eval| / (|python_eval| + 1e-15)
```

A high `equation_MRE` means the model wrote a formula and then did not evaluate it correctly. It says nothing about
whether the formula or its inputs were right - that is what process accuracy is for.

Notation handling verified by the repository's own unit tests: `|...|` becomes `abs(...)`; `^` becomes `**` with bare
(possibly negative) bases parenthesised so `-24.0^2` is `(-24.0)**2`; `arccos`/`arctan2`/`sqrt` map to the `math`
equivalents; angle expressions are wrapped in `math.degrees(...)`. When several `sqrt(` calls appear, the **last** one
is taken.

| Task | Steps evaluated | Notes |
|---|---|---|
| T/L | step 3 (major axis), step 4 (minor axis) | tool-use fallback: a single `<tool_call>` whose stdout and `<answer>` both yield >= 2 numbers fills steps 3 and 4 |
| A/D | step 3 only (the distance or the angle) | `metric_type` decides distance vs angle; `wrap_degrees` is applied for angles |

### What happens when there is no equation

Nothing is fabricated and nothing raises. For a step with no parseable formula the record gets
`step{k}_raw_expr = None`, `step{k}_python_eval = None`, `step{k}_equation_MRE = None`; the sample is not counted in
that step's mean and shows up as `fail=N` on the per-file console line and as a gap between `n_samples` and
`n_valid_3` / `n_valid_4` in the summary JSON. A verified example from a three-row fixture (one clean sample, one with a
deliberate arithmetic slip, one answering in prose with no formula):

```text
[MedVision_TumorLesionSize_samples_KiTS23_axial.jsonl]
  Total: 3  (tl=3, task_type_fail=0, parse_fail=0, success_rate=66.7% 2/3)
  Step3 equation MRE (major axis): mean=0.1000 +- sd=0.1000 (n=2, fail=1)
  Step4 equation MRE (minor axis): mean=0.0000 +- sd=0.0000 (n=2, fail=1)
```

Related failure labels on the same line: `task_type_fail` (T/L only - `doc["taskType"]` contains neither "Tumor" nor
"Lesion", i.e. a non-T/L file landed in the folder) and `parse_fail`. If an expression uses a function outside the
whitelist the record carries `step{k}_eval_error` (e.g. `Disallowed function: 'log'`) and its MRE stays `None`.

### Near-zero A/D ground truths

In `analyze_equation_accuracy_AD.py` the same `AD_NEAR_ZERO_GT_THRESHOLD = 0.1` is applied to the **Python-evaluated
value** (`step3_python_eval`): a sample below the threshold goes to `n_ignored` instead of the mean, because dividing by
a near-zero denominator makes the ratio meaningless. `n_samples`, `n_valid` and `n_ignored` are all reported.

### Outputs

| File | Where | Content |
|---|---|---|
| `<stem>_eq_acc[_filtered].jsonl` | beside each input JSONL | `step{k}_model_answer`, `step{k}_raw_expr`, `step{k}_python_eval`, `step{k}_equation_MRE`, optional `step{k}_eval_error` |
| `summary_eq_acc_{TL,AD}_metrics[_filtered].json` | in the parsed folder | per label: `step*_avg_equation_MRE`, `n_samples`, `n_valid*` (`n_ignored` for A/D) |
| `summary_eq_acc_{TL,AD}_model[_filtered].txt` | in the model folder | weighted average + per-label table |
| `summary_eq_acc_{TL,AD}_task[_filtered].txt` | in the task folder (`--task_dir` only) | the same, one block per model |

---

## 3. Prerequisites at a glance

| Analyzer | `medvision_bm` | `medvision_ds` | dataset files on disk |
|---|---|---|---|
| `analyze_process_accuracy_AD.py` | required | **required** (benchmark plans resolve the GT landmarks) | landmark JSONs at the recorded `doc["landmark_file"]` paths |
| `analyze_process_accuracy_TL.py` | required | needed only for the per-label aggregation | landmark JSONs at the recorded paths |
| `analyze_equation_accuracy_TL.py` | required | needed only for the per-label aggregation | none |
| `analyze_equation_accuracy_AD.py` | required | not needed | none |

"Needed only for the per-label aggregation" means: without it the per-sample `_proc_acc`/`_eq_acc` JSONL is complete but
every label lookup fails and the summary JSON comes out empty (the lookup swallows the exception). Use `--jsonl` mode
when you only want per-sample records - it skips aggregation entirely.

The landmark files are read from the **absolute paths recorded in the records at evaluation time**. If the results were
produced on another machine, or the dataset tree moved, process accuracy fails per sample with the underlying
`[Errno 2] No such file or directory: ...` inside a `gt_extraction_failed` / `error` field; equation accuracy is
unaffected.

---

## 4. Sample-set flags

`--removed_samples_dir` / `--removed_samples_filename` exist on the **T/L** analyzers only and mirror
`summarize_TL_task`: they drop the multi-cluster slices and add a `_filtered` marker to every output name. The exclusion
key is `(image_file relative to the dataset folder, slice_dim, slice_idx, doc["taskID"])`. The A/D analyzers have no
such flag because there is no mask and therefore no multi-cluster slice.

Use the same choice (filtered or not) across every model in one comparison, and the same choice the benchmark summary
used, or the process/equation numbers will not line up with the headline metrics.
