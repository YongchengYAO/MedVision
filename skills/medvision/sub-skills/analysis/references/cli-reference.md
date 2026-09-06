# CLI reference - post-hoc analyses

Every block below is the verbatim `--help` output of a bundled script (run from `scripts/`) or of an installed
`medvision_bm` module. Nothing here is paraphrased; if a flag is not listed, it does not exist.

Contents: [process accuracy](#process-accuracy) - [equation accuracy](#equation-accuracy) -
[CDA](#clinical-decision-agreement-cda) - [detection x target size](#detection-x-target-size).

Conventions shared by all of them:

- `--task_dir` = the results directory of one task (`Results/<task tag>`), whose immediate subdirectories are model
  folders. `--model_dir` = a single model folder. The two are mutually exclusive in the wrappers.
- The process/equation analyzers hardcode `parsed/` under `--task_dir`/`--model_dir`; use `--jsonl` to point them at any
  other folder (for example `llm-parsed_<judge>/`). The CDA and detection analyzers take `--parsed_dirname` instead.
- All of them write **next to their inputs** (per-sample JSONL beside the source file, per-model summaries inside the
  parsed folder, task-level reports in the task directory). Copy a tree first if you need the originals untouched.

---

## Process accuracy

### `scripts/analyze_process_accuracy_TL.py`

```text
usage: analyze_process_accuracy_TL.py [-h] [--task_dir TASK_DIR]
                                      [--model_dir MODEL_DIR]
                                      [--jsonl JSONL [JSONL ...]]
                                      [--output_suffix OUTPUT_SUFFIX]
                                      [--removed_samples_dir REMOVED_SAMPLES_DIR]
                                      [--removed_samples_filename REMOVED_SAMPLES_FILENAME]

Analyze intermediate-step accuracy for T/L task JSONL files.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Task results directory whose immediate subdirectories
                        are model folders. Each model folder must contain a
                        'parsed/' subfolder with JSONL files.
  --model_dir MODEL_DIR
                        Single model directory containing a 'parsed/'
                        subfolder with JSONL files.
  --jsonl JSONL [JSONL ...]
                        One or more explicit JSONL file paths (or glob
                        patterns) to analyze
  --output_suffix OUTPUT_SUFFIX
                        Suffix appended before .jsonl in the output filename
                        (default: _proc_acc)
  --removed_samples_dir REMOVED_SAMPLES_DIR
                        Root directory containing per-dataset removed_samples
                        JSON files (e.g. .../Data/Datasets). When provided,
                        samples listed in those files are excluded and output
                        filenames get a '_filtered' suffix.
  --removed_samples_filename REMOVED_SAMPLES_FILENAME
                        Filename of the removed-samples JSON within each
                        dataset subdirectory.
```

At least one of `--task_dir` / `--model_dir` / `--jsonl` is required (`parser.error` otherwise). Outputs (verified in
source): `<stem>_proc_acc[_filtered].jsonl` per input file, `summary_proc_acc_TL_metrics[_filtered].json` in the parsed
folder, `summary_proc_acc_TL_model[_filtered].txt` in the model folder, and
`summary_proc_acc_TL_task[_filtered].txt` in the task folder (`--task_dir` only).

### `scripts/analyze_process_accuracy_AD.py`

```text
usage: analyze_process_accuracy_AD.py [-h] [--task_dir TASK_DIR]
                                      [--model_dir MODEL_DIR]
                                      [--jsonl JSONL [JSONL ...]]
                                      [--output_suffix OUTPUT_SUFFIX]

Analyze intermediate-step accuracy for A/D task JSONL files.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Task results directory whose immediate subdirectories
                        are model folders. Each model folder must contain a
                        'parsed/' subfolder with JSONL files.
  --model_dir MODEL_DIR
                        Single model directory containing a 'parsed/'
                        subfolder with JSONL files.
  --jsonl JSONL [JSONL ...]
                        One or more explicit JSONL file paths (or glob
                        patterns) to analyze
  --output_suffix OUTPUT_SUFFIX
                        Suffix appended before .jsonl in the output filename
                        (default: _proc_acc)
```

No removed-samples flags: the multi-cluster exclusion list is a mask concept and A/D has no masks. Outputs:
`<stem>_proc_acc.jsonl`, `summary_proc_acc_AD_metrics.json`, `summary_proc_acc_AD_model.txt`,
`summary_proc_acc_AD_task.txt`.

---

## Equation accuracy

### `scripts/analyze_equation_accuracy_TL.py`

```text
usage: analyze_equation_accuracy_TL.py [-h] [--task_dir TASK_DIR]
                                       [--model_dir MODEL_DIR]
                                       [--jsonl JSONL [JSONL ...]]
                                       [--output_suffix OUTPUT_SUFFIX]
                                       [--removed_samples_dir REMOVED_SAMPLES_DIR]
                                       [--removed_samples_filename REMOVED_SAMPLES_FILENAME]

Analyze equation computing accuracy for T/L task JSONL files.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Task results directory; each immediate subdir is a
                        model folder with parsed/ subdir.
  --model_dir MODEL_DIR
                        Single model directory containing a parsed/ subfolder
                        with JSONL files.
  --jsonl JSONL [JSONL ...]
                        One or more explicit JSONL file paths (or glob
                        patterns) to analyze.
  --output_suffix OUTPUT_SUFFIX
                        Suffix appended before .jsonl in the output filename
                        (default: _eq_acc).
  --removed_samples_dir REMOVED_SAMPLES_DIR
                        Root directory containing per-dataset removed_samples
                        JSON files (e.g. .../Data/Datasets). When provided,
                        samples listed in those files are excluded and output
                        filenames get a '_filtered' suffix.
  --removed_samples_filename REMOVED_SAMPLES_FILENAME
                        Filename of the removed-samples JSON within each
                        dataset subdirectory.
```

Outputs: `<stem>_eq_acc[_filtered].jsonl`, `summary_eq_acc_TL_metrics[_filtered].json`,
`summary_eq_acc_TL_model[_filtered].txt`, `summary_eq_acc_TL_task[_filtered].txt`.

### `scripts/analyze_equation_accuracy_AD.py`

```text
usage: analyze_equation_accuracy_AD.py [-h] [--task_dir TASK_DIR]
                                       [--model_dir MODEL_DIR]
                                       [--jsonl JSONL [JSONL ...]]
                                       [--output_suffix OUTPUT_SUFFIX]

Analyze equation computing accuracy for A/D task JSONL files.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Task results directory; each immediate subdir is a
                        model folder with parsed/ subdir.
  --model_dir MODEL_DIR
                        Single model directory containing a parsed/ subfolder
                        with JSONL files.
  --jsonl JSONL [JSONL ...]
                        One or more explicit JSONL file paths (or glob
                        patterns) to analyze.
  --output_suffix OUTPUT_SUFFIX
                        Suffix appended before .jsonl in the output filename
                        (default: _eq_acc).
```

Outputs: `<stem>_eq_acc.jsonl`, `summary_eq_acc_AD_metrics.json`, `summary_eq_acc_AD_model.txt`,
`summary_eq_acc_AD_task.txt`.

---

## Clinical Decision Agreement (CDA)

### `scripts/run_cda.sh` (wrapper: runs the three steps in order)

```text
Usage: run_cda.sh [--ad-task-dir DIR] [--tl-task-dir DIR] [options]

At least one of --ad-task-dir / --tl-task-dir is required. The Markdown report
(build_CDA_report.py) needs BOTH; with only one task the runner performs the
agreement + uncertainty steps for that task and skips the report.

Task directories / configs
  --ad-task-dir DIR        A/D results directory (angle proxies: SNA, SNB on Ceph-Biometrics-400)
  --tl-task-dir DIR        T/L results directory (renal AJCC T-category proxy on KiTS23, KiPA22)
  --ad-config YAML         model_display_name config for the A/D dir (template: cda/config-AD-CoT.yaml)
  --tl-config YAML         model_display_name config for the T/L dir (template: cda/config-TL-CoT.yaml)
                           Without a config every subfolder of the task dir is analysed and the
                           task-level report has no "_canonical" marker. A config-listed folder
                           that is missing on disk is a hard error.

Parsed source
  --parsed-dirname NAME    Folder inside each model dir to read: "parsed" (regex parser, default) or
                           any "llm-parsed*" folder (LLM-judge re-parse). The prefix selects the row
                           field holding the prediction (filtered_resps vs LLM_filtered_resps).
                           Task-level outputs gain a marker such as "_llm-parsed-gemma-4-31b".

Sample set (T/L only)
  --removed-samples-dir DIR         <data_dir>/Datasets; drops multi-cluster T/L slices so CDA scores the
                                    same sample set as summarize_TL_task; T/L outputs gain "_filtered".
                                    Never applied to the A/D task (landmarks have no mask clusters).
  --removed-samples-filename NAME   default: multi_cluster_samples_v1.0.0_to_v1.1.0.json

Uncertainty
  --n-boot N               bootstrap resamples (default 4000)
  --seed N                 resampling seed (default: CDA_SEED = 1024 from cda_config.py)

Output / misc
  --out FILE               report path (default: ./CDA_REPORT<source-marker>.md)
  --repo-root DIR          directory against which the report shortens paths (default: cwd)
  --python EXE             interpreter (default: python)
  --dry-run                print the commands and exit 0
  -h, --help               this text
```

### `scripts/cda/summarize_CDA_task.py` (step 1: categorise + agreement)

```text
usage: summarize_CDA_task.py [-h] [--task_dir TASK_DIR]
                             [--model_dir MODEL_DIR]
                             [--parsed_dirname PARSED_DIRNAME] [--limit LIMIT]
                             [--skip_model_wo_parsed_files]
                             [--config_yaml CONFIG_YAML]
                             [--removed_samples_dir REMOVED_SAMPLES_DIR]
                             [--removed_samples_filename REMOVED_SAMPLES_FILENAME]

Summarize Clinical Decision Agreement (CDA) proxy metrics.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Path to the task directory containing model result
                        folders.
  --model_dir MODEL_DIR
                        Path to a specific model directory containing a
                        parsed-source folder.
  --parsed_dirname PARSED_DIRNAME
                        Which parsed-results folder inside each model
                        directory to read: 'parsed' (regex parser), or any
                        'llm-parsed*' folder written by an LLM-judge re-parse
                        (e.g. 'llm-parsed_gemma-4-31b'). Matched by prefix,
                        since the judge writes one folder per judge model. The
                        prefix also selects the row field holding the
                        prediction, so a source and its field cannot be mixed
                        up. Per-model outputs are written back into the folder
                        read; task-level reports gain a source marker (e.g.
                        '_llm-parsed-gemma-4-31b').
  --limit LIMIT         Limit the number of samples processed per JSONL file
                        (default: all).
  --skip_model_wo_parsed_files
                        Skip model directories without the selected
                        --parsed_dirname folder. Only valid with --task_dir.
  --config_yaml CONFIG_YAML
                        CDA config listing the models to report: config-AD-
                        CoT.yaml for an A/D task dir, config-TL-CoT.yaml for a
                        T/L one. When given, the cross-model report is
                        restricted to those models, labelled, and ordered as
                        in the config, with per-proxy leaderboards.
  --removed_samples_dir REMOVED_SAMPLES_DIR
                        Root directory of per-dataset removed-samples JSONs
                        (e.g. Data/Datasets). Excludes the multi-cluster T/L
                        slices the benchmark's own summarize_TL_task.py
                        excludes, so CDA scores the same sample set. Output
                        filenames gain a '_filtered' marker.
  --removed_samples_filename REMOVED_SAMPLES_FILENAME
                        Filename of the removed-samples JSON within each
                        dataset subdirectory. Matches summarize_TL_task.py's
                        default.
```

### `scripts/cda/cda_uncertainty.py` (step 2: clustered bootstrap)

```text
usage: cda_uncertainty.py [-h] --task_dir TASK_DIR [--config_yaml CONFIG_YAML]
                          [--filtered] [--parsed_dirname PARSED_DIRNAME]
                          [--n_boot N_BOOT] [--seed SEED]

Clustered bootstrap CIs and p-values for CDA agreement statistics.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Results/<experiment> directory.
  --config_yaml CONFIG_YAML
                        CDA config listing the models to analyze: config-AD-
                        CoT.yaml for an A/D task dir, config-TL-CoT.yaml for a
                        T/L one.
  --filtered            Read the '_filtered' inputs written by a
                        --removed_samples_dir run of the two analysis scripts,
                        and write a '_filtered' output. This script does no
                        filtering itself; it only needs to know which files to
                        read.
  --parsed_dirname PARSED_DIRNAME
                        Which parsed-results folder to read the per-sample
                        categorisations from. Must match the --parsed_dirname
                        the analysis scripts ran with; the output filename
                        gains a matching source marker.
  --n_boot N_BOOT       Bootstrap resamples.
  --seed SEED           Resampling seed.
```

`--task_dir` is required and there is no `--model_dir`: uncertainty always works over a whole task directory.

### `scripts/cda/build_CDA_report.py` (step 3: render)

```text
usage: build_CDA_report.py [-h] --ad_task_dir AD_TASK_DIR --tl_task_dir
                           TL_TASK_DIR --ad_config_yaml AD_CONFIG_YAML
                           --tl_config_yaml TL_CONFIG_YAML
                           [--parsed_dirname PARSED_DIRNAME] [--filtered]
                           --out OUT [--repo_root REPO_ROOT]

Render the final CDA leaderboard report as Markdown.

options:
  -h, --help            show this help message and exit
  --ad_task_dir AD_TASK_DIR
                        A/D results directory.
  --tl_task_dir TL_TASK_DIR
                        T/L results directory.
  --ad_config_yaml AD_CONFIG_YAML
                        Config for the A/D dir.
  --tl_config_yaml TL_CONFIG_YAML
                        Config for the T/L dir.
  --parsed_dirname PARSED_DIRNAME
                        Which parsed-results folder the analysis wrote into.
                        Must match the --parsed_dirname the analysis scripts
                        ran with. Recorded in the report's provenance table;
                        pass a distinct --out per source.
  --filtered            Read the '_filtered' artifacts where a task publishes
                        them. A task that publishes none (A/D) falls back to
                        unfiltered, and the report header records which set
                        each task contributed.
  --out OUT             Markdown file to write.
  --repo_root REPO_ROOT
                        Directory against which paths in the provenance table
                        are shortened (default: current working directory).
                        Paths outside it are printed as given.
```

`--repo_root` is specific to this bundled copy (the repository version hardcodes its own checkout root).

---

## Detection x target size

### `scripts/detection_target_size.sh` (wrapper: metrics + figure)

```text
Usage: detection_target_size.sh (--task-dir DIR | --model-dir DIR) [options]

Input
  --task-dir DIR              Detection task dir; every model subfolder is analysed and a random_detection/
                              baseline is generated from the first model that has --parsed-dirname.
  --model-dir DIR             One model dir only (no random baseline; the plot then covers that model
                              plus any random_detection/ sibling already present).
  --parsed-dirname NAME       Parsed-records subfolder inside each model dir (default: parsed). Use e.g.
                              llm-parsed_gemma-4-31b for LLM-judge re-parsed records; outputs go into that
                              folder so published summaries are never overwritten.
  --skip-model-wo-parsed-files  Skip models lacking the parsed subfolder (--task-dir only). Recommended
                              with an llm-parsed source: a missing folder is fatal otherwise.
  --limit N                   Samples per JSONL (debug).
  -p, --processes N           Worker processes for metric aggregation.

Plot
  --config YAML               model_display_name map (default: config-detect-boxImgRatio.yaml next to this
                              script; edit folder names to match your Results tree).
  --out-dir DIR               Figure directory (default: ./Figures).
  --save-as-png               Also write a PNG (the wrapper passes --save_as_pdf too, because
                              --save_as_png alone would replace the PDF rather than add to it).
  --skip-viz                  Run step 1 only.

Environment
  --python EXE                Interpreter (default: python).
  --repo-root DIR             Prepend DIR/src to PYTHONPATH (use when medvision_bm is not pip-installed).
  --dry-run                   Print the commands and exit 0.
  -h, --help                  This text.
```

The repository's own launcher defaults `--parsed_dirname` to the judge folder; this wrapper defaults to `parsed`,
so state the source explicitly whenever it matters.

### `python -m medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random`

```text
usage: analyze_detection_task_boxsize_vs_random.py [-h] [--task_dir TASK_DIR]
                                                   [--model_dir MODEL_DIR]
                                                   [--ref_model_dir REF_MODEL_DIR]
                                                   [--out_dir OUT_DIR]
                                                   [--limit LIMIT]
                                                   [--parsed_dirname PARSED_DIRNAME]
                                                   [--skip_model_wo_parsed_files]
                                                   [--processes PROCESSES]

Analyze detection task performance by bounding box size and generate a random
detection baseline. Use --task_dir to process all models and generate the
random baseline, or --ref_model_dir/--out_dir for the random baseline alone
(same interface as simulate_random_detection.py).

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Path to the task directory containing model result
                        folders.
  --model_dir MODEL_DIR
                        Path to a specific model directory containing JSONL
                        files.
  --ref_model_dir REF_MODEL_DIR
                        Path to a model's parsed/ folder used as GT source for
                        the random baseline. Requires --out_dir. Generates
                        random_detection/ output only.
  --out_dir OUT_DIR     Output directory for random baseline (used with
                        --ref_model_dir).
  --limit LIMIT         Limit the number of samples to process per JSONL file.
                        If not set, processes all samples.
  --parsed_dirname PARSED_DIRNAME
                        Name of the parsed-results subfolder to read inside
                        each model directory (e.g. 'parsed' for the regex
                        parser, 'llm-parsed_gemma-4-31b' for the LLM-judge re-
                        parse). Ignored in --ref_model_dir mode, which takes
                        the folder path directly.
  --skip_model_wo_parsed_files
                        Skip model directories that don't have a parsed-
                        results folder. Only valid with --task_dir.
  --processes PROCESSES, -p PROCESSES
                        Number of worker processes for metric calculation.
```

### `python -m medvision_bm.benchmark.analyze_detection_task_boxsize`

```text
usage: analyze_detection_task_boxsize.py [-h] [--task_dir TASK_DIR]
                                         [--model_dir MODEL_DIR]
                                         [--parsed_dirname PARSED_DIRNAME]
                                         [--limit LIMIT]
                                         [--skip_model_wo_parsed_files]
                                         [--processes PROCESSES]

Analyze detection task performance grouped by bounding box size relative to
image size. Reads BoxCoordinate JSONL files from model_dir/<parsed_dirname>/.
Outputs summary_metrics_per_boxImgRatio_detect_Task.json,
summary_values_per_boxImgRatio_detect_Task.json,
summary_metrics_boxImgRatio_x_fineLabel_detect_Task.csv and
summary_metrics_boxImgRatio_x_label_detect_Task.csv into that same folder.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Path to the task directory containing model result
                        folders.
  --model_dir MODEL_DIR
                        Path to a specific model directory containing a
                        parsed-records subfolder.
  --parsed_dirname PARSED_DIRNAME
                        Per-model subdirectory to read parsed records from.
                        Default 'parsed' (the published pipeline). Use e.g.
                        'llm-parsed_gemma-4-31b' to analyze LLM-judge-parsed
                        records; outputs are written into that same folder, so
                        the published summaries are never overwritten.
  --limit LIMIT         Limit the number of samples to process per JSONL file.
                        If not set, processes all samples.
  --skip_model_wo_parsed_files
                        Skip model directories that don't have the parsed-
                        records folder. Only valid with --task_dir.
  --processes PROCESSES, -p PROCESSES
                        Number of worker processes for metric calculation.
```

This is the variant to run when you want the **per-label x box-size CSVs** (the `..._vs_random` variant writes only the
two ratio JSONs plus the random baseline).

### `python -m medvision_bm.benchmark.viz_detection_performance_per_boxImgRatio`

```text
usage: viz_detection_performance_per_boxImgRatio.py [-h] [--config CONFIG]
                                                    --in_dir IN_DIR --out_dir
                                                    OUT_DIR [--save_as_png]
                                                    [--save_as_pdf]

Plot detection metrics vs box-to-image ratio for multiple models.

options:
  -h, --help         show this help message and exit
  --config CONFIG    Path to the YAML config file (default: config-detect-
                     boxImgRatio.yaml next to this script)
  --in_dir IN_DIR    Directory containing model subfolders with metrics JSON
                     files
  --out_dir OUT_DIR  Directory to save the output figure
  --save_as_png      Save figures as PNG.
  --save_as_pdf      Save figures as PDF.
```

Writes `metrics_boxImgRatio-dotline.pdf` into `--out_dir`. `scripts/detection_target_size.sh` rewrites every config key
except `random_detection` to `<folder>/<parsed_dirname>` before calling it, so the plot reads the JSON from whichever
parsed source was analysed.

### `python -m medvision_bm.benchmark.viz_detection_sampleSize_per_label_x_boxSize`

```text
usage: viz_detection_sampleSize_per_label_x_boxSize.py [-h] --config CONFIG
                                                       --in_dir IN_DIR
                                                       [--parsed_dirname PARSED_DIRNAME]
                                                       --out_dir OUT_DIR
                                                       [--label_level | --anatomy_level]
                                                       [--save_as_png]
                                                       [--save_as_pdf]

Plot metrics and sample size distribution per label and box size for multiple
models.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file (model_display_name mapping)
  --in_dir IN_DIR       Directory containing model subfolders (each with a
                        {parsed_dirname}/ subdirectory)
  --parsed_dirname PARSED_DIRNAME
                        Per-model subdirectory to read the per-label CSVs
                        from, e.g. llm-parsed_gemma-4-31b. Non-default sources
                        suffix the output figure name with __{parsed_dirname}.
                        Default: parsed.
  --out_dir OUT_DIR     Directory to save the output figure
  --label_level         Read fine-grained label CSV (default). Outputs
                        fig_detection__metrics-boxSize__labelLevel.pdf
  --anatomy_level       Read anatomy-grouped label CSV. Outputs
                        fig_detection__metrics-boxSize__anatomyLevel.pdf
  --save_as_png         Save figures as PNG.
  --save_as_pdf         Save figures as PDF.
```

This one consumes the **CSVs** written by `analyze_detection_task_boxsize` (not the `..._vs_random` variant), so run
that module first. Unlike the ratio plot it takes the config keys as plain folder names plus `--parsed_dirname`.
