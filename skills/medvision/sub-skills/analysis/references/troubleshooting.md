# Troubleshooting: post-hoc analyses

Symptom -> cause -> fix. Everything here is CPU-only, so "needs a GPU" appears only where the *upstream* step does.
Cross-cutting environment and data problems live in `../../../references/troubleshooting.md`.

## Inputs and parsed sources

| Symptom / error fragment | Likely cause | Fix / validation | When to stop |
|---|---|---|---|
| `AssertionError: Parsed files directory does not exist: <model>/parsed`, or `[skip] no parsed/ dir: <model>` | the model was never parsed, or the folder under `--task_dir` is not a model folder (a backup, a figure dir, `random_detection/`) | run benchmark step 2 first (`../../results-parsing-and-metrics/SKILL.md`); for the detection/CDA analyzers add `--skip_model_wo_parsed_files`; restrict with `--model_dir` | never blocks |
| CDA/detection report is complete but **every model shows `n_parsed = 0`** (or the curve is flat at zero) | `--parsed_dirname` names a folder whose row schema does not match the field being read. CDA prevents this by pairing prefix->field, so this is the shape of the bug if you ever bypass it | never set the prediction field independently of the folder; use `--parsed-dirname` on the wrapper and let `cda_config.parsed_source_field` pick the field | never blocks |
| `ValueError: Unknown parsed source '<name>': it starts with none of llm-parsed, parsed.` | `--parsed_dirname` matches no registered prefix (typo, or a genuinely new parser family) | pass the real folder name; the accepted prefixes are `parsed` and `llm-parsed*` (any judge/limit suffix is fine) | needs a code change only for a genuinely new parser family |
| `FileNotFoundError: No CDA metrics found under '<dirname>/' in either <ad dir> or <tl dir>` from `build_CDA_report.py` | the source folder does not exist for these models, or the agreement step was never run with that same `--parsed_dirname` | re-run all three CDA steps with one `--parsed_dirname` (the wrapper does that for you); check `ls <model>/<dirname>` | never blocks |
| Process/equation accuracy silently analysed `parsed/` when you meant the judge folder | those two analyzers **hardcode** `parsed/` in `--task_dir`/`--model_dir` mode; only `--jsonl` can point elsewhere | `--jsonl "<model>/llm-parsed_<judge>/*.jsonl"`; note this skips the per-label aggregation | never blocks |
| Analysing an `llm-parsed_<judge>/` folder gives *identical* process/equation numbers to `parsed/` and you expected a difference | expected: those two analyzers read `resps` (the raw response), which the judge pass copies through unchanged - only the prediction key is renamed (`filtered_resps` -> `LLM_filtered_resps`) | if you wanted the judge's effect, look at CDA or the detection box-size analyzers, which do consume the prediction field | never blocks |
| `KeyError` / empty results when a folder mixes judge records and regex records | one folder must carry one schema | keep one schema per folder; re-run the judge pass for the whole folder or remove the stray files | never blocks |

## Model configs (CDA and the detection figure)

| Symptom / error fragment | Likely cause | Fix / validation | When to stop |
|---|---|---|---|
| `FileNotFoundError` naming a model folder from `config-{AD,TL}-CoT.yaml` | that folder is not in the task directory: a stale config, a run-specific suffix (`_bugfix-<sha>`, a token-budget suffix), or the **A/D config passed against the T/L directory** | fix the folder names, or pass the config matching the directory. This hard failure is deliberate - it is what catches a config/task mix-up. Drop `--config_yaml` to analyse every subfolder instead | never blocks |
| CDA numbers look stale though the config validated | a listed folder exists but is superseded by a newer run of the same model - no filesystem check can catch this | keep the CDA configs in step with whichever config your project treats as the canonical run list; check folder modification times | never blocks |
| A model is missing from the CDA Markdown report but no error was raised | it appears in the **Not reported** block: the model has no metrics for the selected parsed source | run the agreement step for that model with the same `--parsed_dirname`, or accept the partial roster and say so | never blocks |
| A model line is missing from `metrics_boxImgRatio-dotline.pdf` | its config key does not match a folder, or that folder has no `summary_metrics_per_boxImgRatio_detect_Task.json` for the chosen source | check the legend against the folders; re-run step 1 for that model with the same `--parsed-dirname` | never blocks |
| `viz_detection_sampleSize_per_label_x_boxSize` raises about missing CSVs | it consumes `summary_metrics_boxImgRatio_x_{label,fineLabel}_detect_Task.csv`, which only `analyze_detection_task_boxsize` writes (not the `..._vs_random` variant the wrapper runs) | run `python -m medvision_bm.benchmark.analyze_detection_task_boxsize --task_dir ... --parsed_dirname ...` first | never blocks |

## Ground truth, plans and dataset files (process accuracy)

| Symptom / error fragment | Likely cause | Fix / validation | When to stop |
|---|---|---|---|
| Every A/D sample records `gt_extraction_failed: benchmark_plan error: ...` or `Dataset '<name>' not found in DATASETS_NAME2PACKAGE.` | `medvision_ds` is not installed, lacks that dataset module, or `medvision_bm.utils.configs.DATASETS_NAME2PACKAGE` has no entry for the dataset name parsed from the filename | install it with `python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>` (see `../../environment-setup/SKILL.md`); a missing mapping entry is a maintainer change (`../../extending-models-and-tasks/SKILL.md`) | first install needs network |
| `_proc_acc.jsonl` is fine but `summary_proc_acc_*_metrics.json` is **empty** | the per-label lookup needs `medvision_ds` and swallows its exception, so every label resolves to `None` and nothing aggregates | install `medvision_ds`, or use `--jsonl` when you only want per-sample records | first install needs network |
| Per-sample `error: "[Errno 2] No such file or directory: '<...>.json'"` and `gt_fail=N` on every file | the landmark JSONs are not at the **absolute paths recorded during evaluation** - results copied from another machine, or the dataset tree moved | mount/download the dataset tree at the recorded location, or re-run evaluation where the data lives. Equation accuracy is unaffected and can be run instead | needs the dataset (large) |
| `error: "gt_extraction_failed: biometric_profile parse error: ..."` | the record has no `biometric_profile` (a non-T/L or non-A/D file landed in the folder, or a hand-made fixture) | keep one task type per folder; check the file name matches `*_samples_<dataset>_*` for the expected task | never blocks |
| `error: "unexpected taskType: ''"` on every row of an equation-accuracy T/L run | `doc["taskType"]` contains neither "Tumor" nor "Lesion" - the file is not a T/L result | point the T/L analyzer at T/L files only; use the A/D analyzer for A/D files | never blocks |
| `[error] cannot import _compute_physical_diagonal ...; scaledPS nMAE will be NaN` | a scaledPS file needs the vendored eval utilities (torch / transformers / nibabel / `medvision_ds`) to rebuild the scaled physical diagonal | run the analysis in the full eval environment, or accept NaN nMAE - the MREs are still correct. Non-scaledPS files are unaffected | never blocks |

## Sample counts that do not match

| Symptom / error fragment | Likely cause | Fix / validation | When to stop |
|---|---|---|---|
| A/D step-3 averages are computed over fewer samples than `n_samples` | `AD_NEAR_ZERO_GT_THRESHOLD = 0.1` excluded near-zero ground truths (process accuracy) or near-zero Python-evaluated values (equation accuracy); the count is reported as `n_ignored` | expected - a relative error against a near-zero denominator is meaningless. Quote `n_valid`, not `n_samples`. The same constant is applied by `summarize_AD_task` | never blocks |
| CDA scores more T/L samples than the benchmark's own T/L summary | the benchmark's canonical T/L report excludes multi-cluster slices; CDA does not by default | add `--removed-samples-dir <data_dir>/Datasets`; every T/L output then gains a `_filtered` marker. Pick one convention and use it everywhere | never blocks |
| `_filtered`-named files hold **unfiltered** numbers | the removed-samples root did not exist, so nothing was excluded while the marker was still applied | the bundled `scripts/run_cda.sh` refuses a non-existent `--removed-samples-dir` up front; when calling the scripts directly, verify `ls <dir>/<dataset>/multi_cluster_samples_v1.0.0_to_v1.1.0.json` | never blocks |
| The A/D task produced a full set of `_filtered` files identical to the unfiltered ones | the exclusion list marks slices whose *mask* has several connected components; A/D measures landmarks, so it is a no-op there | do not pass the removed-samples flags to A/D. The CDA analyzer does accept them on any task dir and will stamp `_filtered` on the output names, but A/D datasets ship no removed-samples JSON so nothing is excluded — which is why the wrapper never passes them for A/D. The process/equation A/D analyzers do not accept them | never blocks |
| `--limit N` results do not line up between models | `--limit` is per JSONL file and interacts with filtering | use the same `--limit` (or none) for every model in one comparison; task-level CDA outputs carry a `_limit<N>` marker so a debug run cannot overwrite a full one | never blocks |
| An anatomy group is missing from a *summary* report but present in a box-ratio bin | `MINIMUM_GROUP_SIZE = 50` drops small groups from `summarize_detection_task` / `summarize_TL_task` averages; it is **not** applied to box-ratio bins | expected. Read `num_samples` per bin before quoting a thin bin; the group-average rules live in `../../results-parsing-and-metrics/references/metrics.md` | never blocks |
| A box-ratio bin has a wild F1 | very few samples in that bin | check `num_samples` in `summary_metrics_per_boxImgRatio_detect_Task.json`; report the bin count next to the value | never blocks |

## Equation extraction

| Symptom / error fragment | Likely cause | Fix / validation | When to stop |
|---|---|---|---|
| `fail=N` on the console line, `n_valid_3`/`n_valid_4` far below `n_samples` | for those samples no equation was found: `step{k}_raw_expr = None`, `step{k}_python_eval = None`, `step{k}_equation_MRE = None`. They are **excluded from the mean, not scored 0** | expected for models that answer without writing a formula. Always quote `n_valid` beside the mean - a model that writes formulas rarely can post an excellent `equation_MRE` on a handful of samples | never blocks |
| `step{k}_eval_error: "Disallowed function: '<name>'"` | the expression used a function outside the evaluator's numeric whitelist (`sqrt`, `acos`, `atan2`, ...) | expected and deliberate: the evaluator is restricted on purpose. Those rows are excluded from the mean | needs a code change to widen the whitelist |
| `step{k}_eval_error` about parsing / unbalanced parentheses | the model wrote a malformed formula | expected; the row is excluded | never blocks |
| A model's `equation_MRE` is suspiciously perfect | it may have answered through a tool call rather than a written formula; the T/L analyzer has a `<tool_call>` fallback that pairs the tool's stdout with `<answer>` | inspect a few `_eq_acc.jsonl` records - `step{k}_raw_expr` is `None` when the fallback supplied the numbers | never blocks |

## Runtime and environment

| Symptom / error fragment | Likely cause | Fix / validation | When to stop |
|---|---|---|---|
| `[analyze_*] cannot import medvision_bm (No module named 'medvision_bm').` | the bundled analyzers need the benchmark package | `pip install medvision-bm`, or put a checkout's `src/` on `PYTHONPATH`; see `../../environment-setup/SKILL.md` for the pins before installing into an eval environment | first install needs network |
| `error: medvision_bm is not importable by '<python>'` from `scripts/detection_target_size.sh` | same, for the interpreter passed with `--python` | pass a suitable `--python`, or `--repo-root <checkout>` to prepend `<checkout>/src` | first install needs network |
| `ModuleNotFoundError: No module named 'yaml'` from a CDA script | `PyYAML` is imported lazily and only for `--config_yaml` | install PyYAML, or drop the config to analyse every subfolder | never blocks |
| CDA fails on an import of `medvision_bm` | it should not: `scripts/cda/` is self-contained (numpy only, PyYAML lazily). An import there means the folder was modified | keep CDA free of `medvision_bm` and `scikit-learn` - that independence is what lets it run in a bare environment | never blocks |
| `-p N` is slower than serial, or workers die without a traceback | too many processes for the machine; each worker loads a whole detection JSONL into memory and those files are large | keep `-p` at or below the core count and well within RAM; run models one at a time with `--model_dir` | never blocks |
| `PicklingError` / `AttributeError: Can't pickle local object` when driving the detection analyzers from a notebook with `processes > 1` | multiprocessing needs worker functions importable by name; interactive redefinitions and monkeypatches break that | run them as `python -m medvision_bm.benchmark....` from a shell, or drop `-p`. Only the detection analyzers use multiprocessing; the CDA and process/equation scripts are single-process | never blocks |
| CDA confidence intervals differ between two machines for the same data and seed | directory listing order feeds cluster order, which the seeded RNG indexes into; the modules sort every listing to prevent this | ensure no local edit replaced `cda_stats.sorted_glob` / `get_subfolders` with bare `glob`/`scandir`. Point estimates are order-independent, so a matching kappa with a differing CI is the signature | never blocks |
| Re-running an analysis overwrote results you wanted to keep | every analysis writes beside its inputs, and only the documented markers (`_filtered`, `_canonical`, `_limit<N>`, `_<source>`) separate runs | copy the tree, or change `--parsed_dirname` / `--output_suffix` / `--out` before re-running. Use `--dry-run` on the wrappers first | never blocks |

## When to stop and hand back

- The upstream `parsed/` folders do not exist and cannot be produced without a GPU evaluation run.
- The dataset tree needed by process accuracy is not available and cannot be downloaded (large, needs network and an
  HF token).
- A task has **no cutoff table**: CDA covers only SNA/SNB on Ceph-Biometrics-400 and the renal T category on
  KiTS23/KiPA22. Every other dataset simply yields no proxy and contributes no rows - which reads as an empty report,
  not an error. Adding a proxy means adding a published cutoff table plus its boundary direction to `cda_config.py`,
  which changes published numbers; that is a maintainer decision, not a fix (`../../extending-models-and-tasks/SKILL.md`).
- The question needs new model outputs (a different prompt, a different scale factor, another model): that is
  evaluation, not analysis - `../../benchmark-evaluation/SKILL.md`, and it needs a GPU or API credentials.
