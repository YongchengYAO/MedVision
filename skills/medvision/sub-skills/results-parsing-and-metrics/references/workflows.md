# Workflows: from raw eval JSONL to metrics and summaries

## Where these steps sit in the benchmark pipeline

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Evaluate (GPU or API) | `python -m medvision_bm.benchmark.eval__<model>` (see `../../benchmark-evaluation/SKILL.md`) | dataset | `Results/<task_tag>/<model_name>/<ts>_samples_<config>.jsonl` + `<ts>_results.json` |
| 2. Parse | `python -m medvision_bm.benchmark.parse_outputs --task_type {AD,TL,Detection}` | the raw `*.jsonl` of step 1 | `<model_name>/parsed/*.jsonl` (records + per-sample metrics), `<model_name>/parsed/<ts>_results.json` |
| 3. Summarize | `python -m medvision_bm.benchmark.summarize_{AD,TL,detection}_task` | `<model_name>/parsed/*.jsonl` (+ benchmark plans from `medvision_ds`) | `parsed/summary_metrics_*.json`, `parsed/summary_values_*.json`, task-level `summary_<task>_task.txt` |
| 4. LLM-judge re-parse (GPU) | driver in `../../llm-judge-parsing/SKILL.md` | raw `*.jsonl` | `<model_name>/llm-parsed_<judge>/*.jsonl`; then step 3 is re-run with `--parsed_dirname llm-parsed_<judge> --resps_key LLM_filtered_resps` |

`<task_tag>` is the task-list name used at eval time (for example `MedVision-TL-CoT`, `MedVision-AD-CoT`, `MedVision-detect-CoT`); `<ts>` is the
eval timestamp `YYYYMMDD_HHMMSS`; `<config>` is the dataset config name such as `BraTS24_TumorLesionSize_Task04_Axial-CoT`.
Each raw JSONL holds one record per sample with `doc`, `target`, `resps`; see `output-files.md`.

Steps 2 and 3 are CPU-only, need no network, and take seconds to minutes. `parse_outputs`, `summarize_TL_task` and `summarize_AD_task` import the vendored eval utilities
(`medvision_utils`), pulling in `torch`, `transformers` and the sibling package `medvision_ds` at module level;
only `summarize_detection_task` is light (`numpy` plus `medvision_bm.utils`). Install what the stage you run needs (CPU builds suffice; expect 20-60 s of import time per process). `medvision_ds` also supplies
the benchmark plans that give label names and modalities in step 3; install it with
`python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>` (details in `../../environment-setup/SKILL.md`). nMAE
additionally needs the NIfTI images at the paths recorded in each record's `doc.image_file` (only the header is read).

## Step 2: parse_outputs

```bash
# all models of a task folder, 16 worker processes
python -m medvision_bm.benchmark.parse_outputs --task_type TL --task_dir Results/MedVision-TL -p 16
# one model, keep files already parsed
python -m medvision_bm.benchmark.parse_outputs --task_type Detection --model_dir Results/MedVision-detect/<model_name> --skip_existing -p 16
# start from scratch (deletes <model_name>/parsed first)
python -m medvision_bm.benchmark.parse_outputs --task_type AD --task_dir Results/MedVision-AD --rm_old
```

What it does per raw JSONL (`_process_jsonl_file`, verified from source):

1. Loads every record, sorts by `doc_id`.
2. Extracts the response text (`resps[0][0]`, or `resps[0][0][0]` when nested) and runs
   `extract_last_k_nums_within_answer_tag(text, k)` with k = 1 (AD), 2 (TL), 4 (Detection). The result (a comma-joined string
   or `""`) becomes `filtered_resps = [<string>]`.
3. Scores the record with `cal_metrics(record, task_type)` and stores `avgMAE`, `SuccessRate`, plus `avgMRE` (AD/TL) or
   `avgIoU`, `F1`, `Precision`, `Recall` (Detection). For TL/AD it also stores `nMAE` unless the raw record already carries one
   (eval-time value is passed through untouched). Detection records get `box_img_ratio` (box area / image area) and, when
   missing, `doc.image_size_2d`.
4. Writes the augmented records to `<model_name>/parsed/<same file name>` and rewrites the task's `<ts>_results.json` into
   `parsed/` with `avgMAE,none`, `avgMRE,none`, `avgIoU,none`, `SuccessRate,none`, `MRE<0.1` ... `MRE<1.0` (`"N/A"` for Detection).

Flags: `--task_dir` loops over every sub-folder (each treated as a model directory); `--model_dir` handles one folder;
`--limit N` keeps the first N records (by `doc_id`) of each file and truncates the parsed file to them; `--skip_existing` skips files
whose parsed twin exists; `--rm_old` deletes `parsed/` before starting; `-p N` parses N JSONL files concurrently (per model folder).
The raw JSONL must sit next to its `<ts>_results.json`, otherwise `ValueError: Results file not found for task ...`.

## Step 3: summarize per task type

### Angle/Distance

```bash
python -m medvision_bm.benchmark.summarize_AD_task --task_dir Results/MedVision-AD -p 8 --skip_model_wo_parsed_files
```

Groups records by `"<dataset>_<metric_type>_<metric_key>"` (for example `Ceph-Biometrics-400_angle_SNA`, `FeTA24_distance_L-1-2`);
The two dataset name patterns are **hard-coded** in `find_and_group_jsonl_files`, so a JSONL from any other dataset or task is silently ignored. Ceph files are summarized one file per key, all `FeTA24_BiometricsFromLandmarks_Task01*` files (three planes) are combined.
Samples with ground truth `< AD_NEAR_ZERO_GT_THRESHOLD` (0.1) are dropped before counting. Writes
`parsed/summary_metrics_AD_Task.json`, `parsed/summary_values_AD_Task.json`, and (task_dir mode) `summary_AD_task.txt` with
group rows FeTA-Distance, Ceph-Angle, Ceph-Distance, Distance, Angle (sample-weighted).

### Tumor/Lesion size

```bash
python -m medvision_bm.benchmark.summarize_TL_task --task_dir Results/MedVision-TL -p 8 \
    --removed_samples_dir <data_dir>/Datasets --skip_model_wo_parsed_files
```

Groups by `"<renamed label> @ <modality> (<plane>)"` (for example `kidney tumor @ CT (A)`), where the label comes from the
dataset's biometry benchmark plan (`target_label` -> `labels_map`) and `label_map_rename`. `--removed_samples_dir` points at the
dataset root that holds `<dataset>/multi_cluster_samples_v1.0.0_to_v1.1.0.json` (default `--removed_samples_filename`); listed
slices (v1.0.0 cases whose target had several disconnected clusters, removed in v1.1.0) are skipped and every output name gets
a `_filtered` suffix (`summary_metrics_TL_Task_filtered.json`, `summary_TL_task_filtered.txt`). Use it whenever the eval ran on
annotation v1.0.0 and you want numbers comparable with later versions; omit it for v1.1.0+ runs.

### Detection

```bash
python -m medvision_bm.benchmark.summarize_detection_task --task_dir Results/MedVision-detect -p 8 --skip_model_wo_parsed_files
```

Groups by `"<anatomy group> @ <modality> (<plane>)"` via `label_map_regroup` (for example `Kidney Tumor/Lesion @ CT (A)`), writes
`summary_metrics_detect_Task.json`, `summary_values_detect_Task.json`, then splits regions into `anatomy` vs `T/L`
(`TUMOR_LESION_GROUP_KEYS`, `EXCLUDED_KEYS`, `MINIMUM_GROUP_SIZE`) into `summary_metrics_anatomy_vs_lesion_detect_Task.json`.
In task_dir mode it also writes `summary_metrics_all_models_detect_Task.json` and `summary_detection_task.txt` at the task level
and silently skips a sub-folder named `random_detection`.

### Flags shared by the three summarizers

- `--task_dir` / `--model_dir`: exactly one. Model-dir mode writes only the per-model JSON files (no cross-model report).
- `--models NAME [NAME ...]` (task_dir mode): restrict to these sub-folder names; results trees typically hold many more folders
  (superseded variants, checkpoints, baselines) than the roster you report on, and one malformed folder aborts the run.
- `--skip_model_wo_parsed_files` (task_dir mode only): in **TL and AD** the report step otherwise raises
  `FileNotFoundError` on the first model folder without the metrics JSON. `summarize_detection_task`
  guards the read with `os.path.exists`, so it silently omits those models from the report instead.
- `--limit N`: first N lines of each parsed JSONL; output names get `_limit<N>` (`summary_metrics_TL_Task_limit100.json`,
  `summary_TL_task_limit100.txt`). Use it to compare full runs with a 100-sample pilot.
- `--parsed_dirname NAME` and `--resps_key KEY`: see the next section.
- `-p N`: detection parallelizes over JSONL files; TL/AD parallelize over label groups.

## Re-summarizing LLM-judge output

The judge pipeline writes `<model_name>/llm-parsed_<judge>/*.jsonl` whose records carry `LLM_filtered_resps` and NOT
`filtered_resps`. Summarize them with the same modules; nothing needs re-parsing:

```bash
python -m medvision_bm.benchmark.summarize_TL_task --task_dir Results/MedVision-TL \
    --parsed_dirname llm-parsed_gemma-4-31b --resps_key LLM_filtered_resps \
    --removed_samples_dir <data_dir>/Datasets --skip_model_wo_parsed_files --models <model_a> <model_b>
```

- Forgetting `--resps_key` aborts immediately with `[FATAL] ... has no 'filtered_resps' key` (`assert_resps_key`).
- Per-model outputs land inside `llm-parsed_<judge>/`; TL names also carry `__llm-parsed_<judge>`. Task-level reports become
  `summary_TL_task_filtered__llm-parsed_gemma-4-31b.txt`, `summary_AD_task__llm-parsed_<judge>.txt`,
  `summary_detection_task__llm-parsed_<judge>.txt`, `summary_metrics_all_models_detect_Task__llm-parsed_<judge>.json`, so published
  reports are never overwritten.
- Diff the two reports: the SuccessRate gap is how much apparent failure was formatting, and MAE/MRE move because more samples
  become parseable. Judge internals live in `../../llm-judge-parsing/SKILL.md`.

## Duplicated samples

Multi-rank evals can emit the same `doc_id` twice in a raw JSONL (each duplicate is scored and counted). Detect with
`python scripts/inspect_summary.py --path <model_name>` (column `dup_ids`), then rebuild a clean tree and parse that:

```bash
python -m medvision_bm.benchmark.remove_duplicate_samples --dir Results/MedVision-detect --out_dir Results/MedVision-detect-deduped
python -m medvision_bm.benchmark.parse_outputs --task_type Detection --task_dir Results/MedVision-detect-deduped
```

`--dir` is a folder of model sub-folders; `.json` files are copied, `.jsonl` files keep the first record per `doc_id`.

## Detection box-size analyses (pointer)

`python -m medvision_bm.benchmark.analyze_detection_task_boxsize` (per-sample CSVs and `summary_metrics_per_boxImgRatio_detect_Task.json`
in the parsed folder), `analyze_detection_task_boxsize_vs_random` (adds a `random_detection` baseline with
`RANDOM_BOX_SIMULATIONS = 100` boxes per sample; `--ref_model_dir <parsed dir> --out_dir <dir>` for the baseline alone),
`viz_detection_performance_per_boxImgRatio` and `viz_detection_sampleSize_per_label_x_boxSize` (figures from a YAML
`model_display_name:` mapping). Run them after step 3; both analyzers accept `--parsed_dirname`. Interpretation and the full
workflow belong to `../../analysis/SKILL.md`; figure conventions to `../../../references/visualization-catalog.md`.

## Bundled wrapper

`scripts/parse_and_summarize.sh --task-type TL --task-dir Results/MedVision-TL -p 8 --removed-samples-dir <data_dir>/Datasets`
runs step 2 then step 3 with matching flags; `--dry-run` prints the two commands; `--parsed-dirname llm-parsed_<judge>` skips the
parse step automatically. Afterwards `scripts/inspect_summary.py --path <model_name>/parsed` shows what was written.

## Validation checklist after a run

1. `parsed/` holds one `.jsonl` and one `_results.json` per raw task file (same names).
2. `SuccessRate` in `summary_metrics_*` equals the fraction of records whose `filtered_resps` has k numbers; `MRE<1.0` equals `SuccessRate`.
3. Group sample counts match the task list limits (1000 per config for open-weight runs, 100 for API pilots); TL `_filtered`
   counts are lower by the removed slices.
4. Detection: `Acc@IoU>=0.50` equals `IoU>0.5`; mean `IoU` is at most `SuccessRate`.
5. nMAE columns are finite; all-NaN nMAE means the NIfTI files were not reachable at parse time (see `troubleshooting.md`).
