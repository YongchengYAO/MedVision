# CLI reference (help text captured from the installed package)

All commands are `python -m medvision_bm.benchmark.<module> ...`. Help text below is verbatim output of `--help`.

## parse_outputs

```text
usage: parse_outputs.py [-h] --task_type TASK_TYPE [--task_dir TASK_DIR]
                        [--model_dir MODEL_DIR] [--limit LIMIT]
                        [--skip_existing] [--processes PROCESSES] [--rm_old]

Parse benchmark output JSONL files and update summaries.

options:
  -h, --help            show this help message and exit
  --task_type TASK_TYPE
                        Type of the task to process: ['AD', 'TL',
                        'Detection'].
  --task_dir TASK_DIR   Path to the benchmark result directory for a specific
                        task where model results directory is located.
  --model_dir MODEL_DIR
                        Path to the model results directory containing JSONL
                        files.
  --limit LIMIT         Limit the number of samples to process per JSONL file.
  --skip_existing       Skip processing files that already have parsed
                        outputs.
  --processes PROCESSES, -p PROCESSES
                        Number of worker processes to use for processing JSONL
                        files. If None, uses single process.
  --rm_old              Remove the old parsed directory before processing.
```

Notes (from source): `--task_type` is validated against `['AD', 'TL', 'Detection']` (exact spelling, `Detection` capitalised).
One of `--task_dir`/`--model_dir` is required (`ValueError` otherwise). Records are sorted by `doc_id` before scoring, so
`--limit N` keeps the N lowest `doc_id`s. Worker output is silenced when `-p > 1`.

## summarize_AD_task

```text
usage: summarize_AD_task.py [-h] [--task_dir TASK_DIR] [--model_dir MODEL_DIR]
                            [--limit LIMIT] [--parsed_dirname PARSED_DIRNAME]
                            [--resps_key RESPS_KEY]
                            [--skip_model_wo_parsed_files]
                            [--models MODELS [MODELS ...]]
                            [--processes PROCESSES]

Process model folders and generate summary metrics.

options:
  -h, --help            show this help message and exit
  --task_dir TASK_DIR   Path to the task directory containing model result
                        folders.
  --model_dir MODEL_DIR
                        Path to a specific model directory containing JSONL
                        files.
  --limit LIMIT         Limit the number of samples to process per JSONL file.
                        If not set, processes all samples.
  --parsed_dirname PARSED_DIRNAME
                        Per-model subdirectory to read parsed records from.
                        Default 'parsed' (the published pipeline). Use e.g.
                        'parsed-llm-limit100' to summarize LLM-judge-parsed
                        records; task-level reports are then written with a
                        '__<parsed_dirname>' qualifier so published reports
                        are never overwritten.
  --resps_key RESPS_KEY
                        Record key holding the parsed prediction. Default
                        'filtered_resps' (the published pipeline). Pass
                        'LLM_filtered_resps' when reading an llm-parsed*/
                        directory: those records have the strict key REMOVED,
                        so forgetting this flag aborts rather than silently
                        reporting on nothing.
  --skip_model_wo_parsed_files
                        Skip model directories that don't have a 'parsed'
                        folder. Only valid with --task_dir.
  --models MODELS [MODELS ...]
                        Restrict to these model-directory basenames (the
                        roster). Default: every directory under --task_dir. A
                        results tree holds far more model directories than any
                        one study reports on -- superseded bugfix variants,
                        training checkpoints, baselines -- so an unfiltered
                        run reports on models the study never included, and
                        one malformed record in any of them aborts the whole
                        run.
  --processes PROCESSES, -p PROCESSES
                        Number of worker processes for metric calculation.
```

## summarize_TL_task

Same options as `summarize_AD_task` plus:

```text
  --removed_samples_dir REMOVED_SAMPLES_DIR
                        Root directory containing per-dataset removed_samples
                        JSON files (e.g. .../Data/Datasets). When provided,
                        samples listed in those files are excluded from metric
                        computation and output filenames get a '_filtered'
                        suffix.
  --removed_samples_filename REMOVED_SAMPLES_FILENAME
                        Filename of the removed-samples JSON within each
                        dataset subdirectory.
```

Default `--removed_samples_filename`: `multi_cluster_samples_v1.0.0_to_v1.1.0.json`. A dataset folder without that file is
simply not filtered (no error).

## summarize_detection_task

Same options as `summarize_AD_task` (the `-p` help reads "Number of worker processes to use for parsing JSONL files. If None,
uses single process."). No removed-samples option. Sub-folder `random_detection` is excluded automatically in task_dir mode.

## Common validation rules (all three summarizers)

- `parser.error` when neither `--task_dir` nor `--model_dir` is given, or when `--skip_model_wo_parsed_files` is used
  without `--task_dir`.
- `--parsed_dirname` names a sub-folder of each model folder; the current judge pipeline writes `llm-parsed_<judge>` (the
  `parsed-llm-limit100` string in the help text is an older example name).
- `--resps_key` must exist in the first readable record of the first JSONL, else `SystemExit` with a `[FATAL]` message listing the
  keys present.

## remove_duplicate_samples

```text
usage: remove_duplicate_samples.py [-h] --dir DIR --out_dir OUT_DIR

Deduplicate JSONL files by doc_id (keep first occurrence).

options:
  -h, --help         show this help message and exit
  --dir DIR          Working folder with subfolders containing JSONL files.
  --out_dir OUT_DIR  Output directory (preserves subfolder structure).
```

Iterates the immediate sub-folders of `--dir`, copies `*.json` unchanged, rewrites `*.jsonl` keeping the first record per
`doc_id`, prints `(<original> -> <kept>, -N dup|clean)` per file. If `--dir` is not a directory it prints an error and returns
exit code 0 (check the message).

## Detection box-size helpers (flags only)

| Module | Required | Optional |
|---|---|---|
| `analyze_detection_task_boxsize` | `--task_dir` or `--model_dir` | `--parsed_dirname` (default `parsed`), `--limit`, `--skip_model_wo_parsed_files`, `-p/--processes` |
| `analyze_detection_task_boxsize_vs_random` | one of `--task_dir`, `--model_dir`, `--ref_model_dir` (+ `--out_dir`) | `--limit`, `--parsed_dirname`, `--skip_model_wo_parsed_files`, `-p/--processes` |
| `viz_detection_performance_per_boxImgRatio` | `--in_dir`, `--out_dir` | `--config` (YAML with `model_display_name:`; default looks for `config-detect-boxImgRatio.yaml` next to the module, which the installed package does not ship, so pass your own), `--save_as_png`, `--save_as_pdf` (default PDF) |
| `viz_detection_sampleSize_per_label_x_boxSize` | `--config`, `--in_dir`, `--out_dir` | `--parsed_dirname`, `--label_level` (default) or `--anatomy_level`, `--save_as_png`, `--save_as_pdf` |

The YAML config maps model folder names to display names (`model_display_name: {"<folder>": "<label>"}`); the order sets the legend.

## Bundled scripts

- `scripts/parse_and_summarize.sh --help`: wrapper running step 2 then step 3 (`--task-type`, `--task-dir|--model-dir`, `-p`,
  `--limit`, `--skip-existing`, `--rm-old`, `--skip-parse`, `--parsed-dirname`, `--resps-key`, `--models`,
  `--skip-model-wo-parsed-files`, `--removed-samples-dir`, `--removed-samples-filename`, `--python`, `--dry-run`).
- `scripts/metrics_demo.py [--json]`: prints and asserts the metric semantics on a synthetic fixture.
- `scripts/inspect_summary.py --path <file|dir> [--sort-by KEY] [--top N] [--json]`: read-only viewer.
