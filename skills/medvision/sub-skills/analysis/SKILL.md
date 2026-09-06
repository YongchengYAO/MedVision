---
name: analysis
description: "Runs MedVision's post-hoc analyses over already-parsed benchmark results, on CPU and without re-running any model: Clinical Decision Agreement (measurements pushed through published clinical cutoff tables into categories, scored with Cohen's / quadratic-weighted kappa and a bootstrap that resamples whole imaging volumes), step-wise CoT process accuracy (T/L 4 steps, A/D 3 steps, against ground truth), equation accuracy (extract the equation the model wrote, evaluate it in Python, compare with the model's own answer), and detection metrics stratified by box-to-image ratio against a random-box baseline. Use when a user asks whether a measurement error would change a clinical decision, which reasoning step failed, whether the arithmetic or the formula was wrong, how accuracy depends on target size, what kappa / weighted kappa / CDA / process accuracy / equation accuracy / boxImgRatio numbers mean, or how to score an llm-parsed_<judge> folder instead of parsed/."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision post-hoc analyses

Four analyses that re-read benchmark records already on disk and say something the headline metric cannot. All of them
are **CPU-only, network-free and never re-run inference**; they take seconds to minutes per model.

They consume the output of benchmark step 2: `Results/<task tag>/<model>/parsed/*.jsonl` (regex parser) or
`.../llm-parsed_<judge>/*.jsonl` (LLM-judge re-parse). Producing those folders belongs to
`../results-parsing-and-metrics/SKILL.md` and `../llm-judge-parsing/SKILL.md`.

**Every analysis writes beside its inputs** - per-sample JSONL next to the source file, per-model summaries inside the
parsed folder, task-level reports in the task directory. Copy the tree first if the originals must stay untouched, and
use `--dry-run` on the two shell wrappers before a first run against a real results tree.

## The four analyses

| Analysis | Answers | Compared against | Needs | Entry point |
|---|---|---|---|---|
| **Clinical Decision Agreement (CDA)** | would this measurement error change the clinical decision? | ground truth, after both sides pass through the same published cutoff table | `numpy` only (`PyYAML` for a config); **not** `medvision_bm` | `scripts/run_cda.sh` |
| **Process accuracy** | which CoT step went wrong? T/L 4 steps (major/minor endpoints by normalised L2, then axis lengths by MRE); A/D 3 steps (two landmark coordinates, then the scalar) | ground truth landmarks + measurements | `medvision_bm`, `medvision_ds`, the dataset landmark files at their recorded paths | `scripts/analyze_process_accuracy_TL.py`, `scripts/analyze_process_accuracy_AD.py` |
| **Equation accuracy** | did the model compute the equation it wrote? | the model's **own** answer - no ground truth involved | `medvision_bm` (T/L aggregation also wants `medvision_ds`) | `scripts/analyze_equation_accuracy_TL.py`, `scripts/analyze_equation_accuracy_AD.py` |
| **Detection x target size** | is it worse on small targets, and how much of the score is box size? | a random-box baseline, per 5% box-to-image-ratio bin | `medvision_bm`, `medvision_ds`, matplotlib/pandas/PyYAML | `scripts/detection_target_size.sh` |

A fifth, the **scaledPS ablation**, is reference-only here: its evaluation half is GPU work owned by
`../benchmark-evaluation/SKILL.md`, and its analysis half is just process accuracy pointed at the scaledPS results
(`references/workflows.md` section 5).

## Quick start

```bash
# 0. sanity-check the environment
python ../../scripts/check_medvision_env.py

# 1. CDA: categorise, bootstrap over volumes, render one Markdown leaderboard  (--dry-run first!)
bash scripts/run_cda.sh \
    --ad-task-dir ${benchmark_dir}/Results/<AD task tag> --ad-config scripts/cda/config-AD-CoT.yaml \
    --tl-task-dir ${benchmark_dir}/Results/<TL task tag> --tl-config scripts/cda/config-TL-CoT.yaml \
    --removed-samples-dir <data_dir>/Datasets --repo-root ${benchmark_dir} --out ${benchmark_dir}/CDA_REPORT.md

# 2. step-wise CoT accuracy against ground truth
python scripts/analyze_process_accuracy_TL.py --task_dir ${benchmark_dir}/Results/<TL task tag>
python scripts/analyze_process_accuracy_AD.py --task_dir ${benchmark_dir}/Results/<AD task tag>

# 3. arithmetic correctness, ground truth not needed
python scripts/analyze_equation_accuracy_TL.py --task_dir ${benchmark_dir}/Results/<TL task tag>

# 4. detection metrics per box-to-image ratio + random baseline + figure
bash scripts/detection_target_size.sh --task-dir ${benchmark_dir}/Results/<Detection task tag> \
    --config scripts/config-detect-boxImgRatio.yaml --out-dir ${benchmark_dir}/Figures/boxImgRatio \
    --skip-model-wo-parsed-files -p 8
```

Edit `scripts/cda/config-AD-CoT.yaml`, `scripts/cda/config-TL-CoT.yaml` and `scripts/config-detect-boxImgRatio.yaml` first: they are templates
listing the repository's paper roster, and results folder names are run-specific.

## References

| Read this when | File |
|---|---|
| You need purpose -> inputs -> command -> outputs -> interpretation for any of the analyses, including the judge-folder and scaledPS variants | `references/workflows.md` |
| You need the CDA method: cutoff tables and their sources, kappa variants, the volume-level bootstrap, `cda_config.py` schema, config YAML rules, every output file, the report's structure | `references/cda.md` |
| You need the step definitions, metric formulas, near-zero-GT filtering, scaledPS handling, and what "no equation found" does to the counts | `references/process-and-equation-accuracy.md` |
| You need the 5% bin table, the random-box baseline, the two analyzer modules, the two figures and how their configs differ | `references/detection-target-size.md` |
| You need the exact `--help` of every bundled analyzer and of the `medvision_bm` box-size modules | `references/cli-reference.md` |
| A run errors, a count changed, a report came out empty, or a number looks wrong | `references/troubleshooting.md` |

## Bundled scripts

| File | What it does |
|---|---|
| `scripts/run_cda.sh` | Runs the three CDA steps in order (agreement -> uncertainty -> report) for one or both task directories, resolves the output marker from the parsed source, refuses a non-existent removed-samples root, and supports `--dry-run`. |
| `scripts/cda/summarize_CDA_task.py` | Step 1: per-sample categorisation and per-proxy agreement; writes the per-model metrics/values JSONs and the task-level `.txt` leaderboards. |
| `scripts/cda/cda_uncertainty.py` | Step 2: clustered bootstrap 95% CIs and one-sided p-values for kappa > 0, resampling whole imaging volumes. Must run after step 1. |
| `scripts/cda/build_CDA_report.py` | Step 3: renders one Markdown leaderboard from what steps 1 and 2 persisted. Recomputes nothing and emits no timestamp. |
| `scripts/cda/cda_config.py` | The clinical cutoff tables, the parsed-source prefix->field map, the output filenames and `CDA_SEED = 1024`. Edit here to change a cutoff. |
| `scripts/cda/cda_stats.py` | Categorisation, both kappas, config loading, model-dir resolution, removed-samples filtering. numpy only. |
| `scripts/cda/config-AD-CoT.yaml`, `config-TL-CoT.yaml` | `model_display_name` templates - one per task, because a model's folder can differ between tasks. |
| `scripts/analyze_process_accuracy_TL.py`, `scripts/analyze_process_accuracy_AD.py` | Step-wise CoT accuracy against ground truth; `--task_dir` / `--model_dir` / `--jsonl`, T/L also takes the removed-samples flags. |
| `scripts/analyze_equation_accuracy_TL.py`, `scripts/analyze_equation_accuracy_AD.py` | Extract the model's equation, evaluate it with a restricted numeric AST evaluator, and report MRE against the model's own answer. |
| `scripts/detection_target_size.sh` | Runs the box-ratio metrics (+ random baseline) then the metric-vs-ratio figure, rewriting the config keys to `<folder>/<parsed_dirname>`; `--dry-run`, `--skip-viz`, `--repo-root`. |
| `scripts/config-detect-boxImgRatio.yaml` | `model_display_name` template for that figure, including the `random_detection` baseline row. |

Every *runnable* script (the two wrappers and the five analyzer/report entry points) takes explicit paths, runs from any working directory, and prints usage with `--help`. `cda/cda_config.py` and `cda/cda_stats.py` are importable modules with no CLI.

## Facts worth keeping straight

- **The parsed source and the prediction field travel together.** `parsed/` carries `filtered_resps`,
  `llm-parsed*/` carries `LLM_filtered_resps`. CDA matches the folder by prefix and derives the field from it, so a
  wrong-source run fails instead of quietly reporting `n_parsed = 0`.
- **Process and equation accuracy read `resps`** (the raw response), not the prediction field, so they give identical
  per-sample numbers on `parsed/` and on `llm-parsed_<judge>/`. They also hardcode `parsed/` under
  `--task_dir`/`--model_dir`; use `--jsonl` for any other folder.
- **CDA covers three proxies only**: SNA and SNB on Ceph-Biometrics-400, and the AJCC renal T category on KiTS23 and
  KiPA22. Any other dataset yields no proxy and simply contributes no rows.
- **Angles are folded into [0, 90] degrees** by the benchmark's own target definition, so an SNA above 90 deg reflects
  back below it and can cross a band edge. Carry this caveat with any SNA/SNB number.
- **The bootstrap resamples volumes, not slices** (`doc.image_file` is the cluster id), because one tumour contributes
  many annotated slices. `doc["taskID"]` - that spelling - is the task id in the removed-samples key.
- **Kappa is not comparable across proxies**; compare models within a proxy. Check `Nparsed`/`n_valid` before quoting
  any mean: each analysis excludes samples for its own reasons (`AD_NEAR_ZERO_GT_THRESHOLD = 0.1`, unparseable
  predictions, missing equations, removed multi-cluster slices).
- **Output markers are load-bearing**: `_<source>` then `_filtered` then `_canonical` then `_limit<N>`, so runs of
  different scope never overwrite each other. Filtered and unfiltered numbers are not interchangeable.

## Boundaries

- Producing `parsed/` and the headline metrics (SR, MAE, MRE, nMAE, IoU/F1, denominators, grouping, `MINIMUM_GROUP_SIZE`):
  `../results-parsing-and-metrics/SKILL.md`. This sub-skill only consumes its output.
- Producing `llm-parsed_<judge>/` (judge environment, roster, driver steps): `../llm-judge-parsing/SKILL.md`.
- Running any evaluation, including the scaledPS variant: `../benchmark-evaluation/SKILL.md` (requires GPU or API keys).
- Figure entry points beyond the two box-size plots: `../../references/visualization-catalog.md`.
- Task names, dataset configs, results-tree layout: `../dataset-and-tasks/SKILL.md` and
  `../../references/concepts-and-glossary.md`.
- Installing `medvision_bm` / `medvision_ds` and the version pins: `../environment-setup/SKILL.md`.
- Adding a new clinical proxy, a new dataset mapping or a new task: `../extending-models-and-tasks/SKILL.md`.
- Cross-cutting failures (environment, data, GPU): `../../references/troubleshooting.md`.

## Safe operating rules

- Never re-run inference to answer an analysis question; if the question needs new model outputs it is an evaluation
  task, not an analysis task.
- Treat `Results/` and the dataset tree as data: read them, write only the documented analysis outputs, and never
  hand-edit a JSONL.
- Changing a cutoff, a boundary direction, an `ordinal` flag or the resampling unit in `cda_config.py` changes published
  numbers - re-run the whole CDA pipeline and say what changed.
- Fix one parsed source, one sample-set choice and one model roster for a whole comparison, and state them in any
  write-up.
- Do not `pip install` into an evaluation environment to satisfy an import; run the analyses in their own environment
  instead.
