# Output files, names and JSON shapes

## Directory layout after steps 1-4

```text
Results/<task_tag>/                                   # e.g. MedVision-TL-CoT, MedVision-AD-CoT, MedVision-detect-CoT
  summary_TL_task[_filtered][_limit<N>][__<parsed_dirname>].txt        # step 3, TL (task_dir mode)
  summary_AD_task[_limit<N>][__<parsed_dirname>].txt                   # step 3, AD
  summary_detection_task[_limit<N>][__<parsed_dirname>].txt            # step 3, Detection
  summary_metrics_all_models_detect_Task[_limit<N>][__<parsed_dirname>].json
  judge-queue_*.jsonl, judge-out_*.jsonl, summary_judge_task__*.txt    # step 4 (judge pipeline)
  <model_name>/
    <ts>_samples_<config>.jsonl          # step 1 raw records (one per sample)
    <ts>_results.json                    # step 1 eval-harness results for that config
    response_cache/*_rank*.jsonl         # step 1 resume cache (not read by steps 2-3)
    parsed/
      <ts>_samples_<config>.jsonl        # step 2: raw record + filtered_resps + per-sample metrics
      <ts>_results.json                  # step 2: results block with recomputed metrics
      summary_metrics_<T>_Task[...].json # step 3 per-model metrics per group
      summary_values_<T>_Task[...].json  # step 3 per-model targets/responses per group
      summary_metrics_anatomy_vs_lesion_detect_Task[_limit<N>].json   # Detection only
      summary_*_per_boxImgRatio_*, summary_metrics_boxImgRatio_x_*.csv, summary_metrics_per_sample*.csv  # box-size analyses
    llm-parsed_<judge>/                  # step 4: same file names, records carry LLM_filtered_resps
      summary_metrics_TL_Task[_filtered]__llm-parsed_<judge>.json      # TL adds the qualifier even inside the folder
      summary_metrics_AD_Task.json / summary_metrics_detect_Task.json  # AD/Detection do not (folder already disambiguates)
      summary_metrics_judge_Task.json
```

`<ts>` = `YYYYMMDD_HHMMSS`; `<config>` = dataset config (`<Dataset>_<TaskType>_<TaskNN>_<Plane>[-CoT][-scaledPS]`). `parse_outputs`
derives the results-file name from the text before `_samples_` and the summarizers derive the dataset name from the text between
`samples_` and the next `_`, so raw file names must not be renamed.

Exact summary file names (constants in `medvision_bm.utils.configs`):

| Constant | Value |
|---|---|
| `SUMMARY_FILENAME_TL_METRICS` / `_VALUES` | `summary_metrics_TL_Task.json` / `summary_values_TL_Task.json` |
| `SUMMARY_FILENAME_AD_METRICS` / `_VALUES` | `summary_metrics_AD_Task.json` / `summary_values_AD_Task.json` |
| `SUMMARY_FILENAME_DETECT_METRICS` / `_VALUES` | `summary_metrics_detect_Task.json` / `summary_values_detect_Task.json` |
| `SUMMARY_FILENAME_GROUPED_ANATOMY_VS_TUMOR_LESION_DETECT_METRICS` | `summary_metrics_anatomy_vs_lesion_detect_Task.json` |
| `SUMMARY_FILENAME_ALL_MODELS_DETECT_METRICS` | `summary_metrics_all_models_detect_Task.json` |
| `SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS` / `_VALUES` | `summary_metrics_per_boxImgRatio_detect_Task.json` / `summary_values_per_boxImgRatio_detect_Task.json` |
| `SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_METRICS` | `summary_metrics_per_sample_detect_Task.csv` |
| `SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS` | `summary_metrics_boxImgRatio_x_label_detect_Task.csv` |
| `SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_METRICS` | `summary_metrics_per_sample_fineLabel_detect_Task.csv` |
| `SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_MEAN_METRICS` | `summary_metrics_boxImgRatio_x_fineLabel_detect_Task.csv` |

Suffix rules: `_filtered` (TL, when `--removed_samples_dir` is given) -> `_limit<N>` (when `--limit`) -> `__<parsed_dirname>`
(TL per-model files and all task-level files when `--parsed_dirname != parsed`). Example:
`summary_metrics_TL_Task_filtered_limit100__llm-parsed_gemma-4-31b-limit100.json`.

## Parsed record (one JSONL line)

Keys added by the eval harness: `doc_id`, `doc`, `target` (string), `arguments`, `resps` (`[[<response text>]]`), `filtered_resps`,
`doc_hash`, `prompt_hash`, `target_hash`, `input`. Keys added or rewritten by `parse_outputs`: `filtered_resps` (`["<k numbers>"]` or
`[""]`), `avgMAE`, `SuccessRate`, then per task:

| Task | Extra keys | `doc` keys |
|---|---|---|
| T/L | `avgMRE`, `nMAE` | `dataset_name, taskID, taskType, image_file, landmark_file, mask_file, slice_dim, slice_idx, label, image_size_2d, pixel_size, image_size_3d, voxel_size, biometric_profile{metric_type, metric_map_name, metric_key_major_axis, metric_value_major_axis, metric_key_minor_axis, metric_value_minor_axis, metric_unit}` |
| A/D | `avgMRE`, `nMAE` (eval-time records may also carry `MAE`, `MRE`) | as T/L minus `mask_file`/`label`; `biometric_profile{metric_type, metric_map_name, metric_key, metric_value, metric_unit, slice_dim}` |
| Detection | `avgMRE` (eval-time, NaN), `avgIoU`, `F1`, `Precision`, `Recall`, `box_img_ratio` | `dataset_name, taskID, taskType, image_file, mask_file, slice_dim, slice_idx, label, image_size_2d, pixel_size, image_size_3d, voxel_size, bounding_boxes{min_coords, max_coords, center_coords, dimensions, sizes}` |

Metric entry shapes: `avgMAE = {"MAE": float|NaN, "success": bool}`, `avgMRE = {"MRE": ..., "success": ...}`,
`SuccessRate = {"success": bool}`, `nMAE = {"NMAE": float|NaN, "success": bool}`, `avgIoU = {"IoU": float}`, `F1 = {"F1": float}`,
`Precision = {"Precision": float}`, `Recall = {"Recall": float}`. NaN is serialized as `NaN` (Python `json` accepts it on read).
scaledPS records add `pixel_size_scale = {"s_h", "s_w", "mode"}`. Judge records replace `filtered_resps` by `LLM_filtered_resps` and add
`LLM_judge_answer_mode`, `LLM_judge_SR`, `LLM_judge`, `LLM_judge_steps`.

Example (T/L, abbreviated):

```json
{"doc_id": 0, "doc": {"dataset_name": "MSD", "taskID": "06", "slice_dim": 2, "slice_idx": 31, "label": 1,
 "image_size_2d": [512, 512], "biometric_profile": {"metric_value_major_axis": [12.4140625], "metric_value_minor_axis": [8.34375], "metric_unit": ["mm"]}},
 "target": "[12.4140625, 8.34375]", "resps": [["<think>...</think><answer>104.40, 55.68</answer>"]],
 "filtered_resps": ["104.40,55.68"],
 "avgMAE": {"MAE": 69.66, "success": true}, "avgMRE": {"MRE": 6.54, "success": true},
 "SuccessRate": {"success": true}, "nMAE": {"NMAE": 0.14, "success": true}}
```

## `parsed/<ts>_results.json`

The eval harness file (`results`, `configs`, `versions`, `n-samples`, `model_name`, `date`, ...) with `results[<config>]` rewritten:

```json
{"alias": "MSD_TumorLesionSize_Task06_Axial-CoT",
 "avgMAE,none": "69.7462800786465", "avgMRE,none": "2.7056583293921928", "avgIoU,none": "nan",
 "SuccessRate,none": 0.8977272727272727,
 "MRE<0.1": 0.0227, "MRE<0.2": 0.0454, "...": "...", "MRE<1.0": 0.2443}
```

Averages are stored as strings; Detection gets `"avgMRE,none": "nan"` and `"MRE<k": "N/A"`; A/D and T/L get `"avgIoU,none": "nan"`.
Keys written by the harness itself (`nMAE,none`, `*_stderr,none`) are left as they were.

## Per-model summary JSON

`summary_metrics_<T>_Task*.json`: `{<group key>: {<metric>: value, ...}}`

- T/L and A/D metrics: `avgMAE, avgMRE, SuccessRate, avgNMAE, num_samples, MRE<0.1 ... MRE<1.0`.
- Detection metrics: `avgMAE, IoU, F1, Precision, Recall, SuccessRate, num_samples, MAE<0.1 ... MAE<1.0, IoU>0.5 ... IoU>0.9,
  F1>0.5 ... , Precision>0.5 ..., Recall>0.5 ..., Acc@IoU>=0.50 ... Acc@IoU>=0.95, Acc@IoU[0.50:0.95]`.

`summary_values_<T>_Task*.json`:

- T/L: `{<group>: {"targets": [str], "responses": [str], "doc_metas": [{image_file, slice_dim, slice_idx, image_size_2d, scale_mode, nmae_precomputed, taskID, label, pixel_size_scale}]}}`
- Detection: `{<group>: {"targets": [str], "responses": [str]}}`
- A/D: a flat list `[{"label": "<dataset>_<type>_<key>", "targets": "<str>", "responses": ["<str>"], "doc_meta": {..., "metric_type": ...}}]`
  (written before the near-zero filter, so it may hold more items than `num_samples` sums to).

`summary_metrics_anatomy_vs_lesion_detect_Task*.json`:

```json
{"anatomy": {"mean_metrics": {"avgMAE": ..., "IoU": ..., "F1": ..., "...": ..., "total_samples": 13421, "num_regions": 18},
             "regions": ["Kidney @ CT (A)", "..."], "detailed_data": {"Kidney @ CT (A)": {<group metrics>}, "...": {}}},
 "T/L": {"mean_metrics": {...}, "regions": [...], "detailed_data": {...}}}
```

## Task-level files

- `summary_metrics_all_models_detect_Task*.json`: `{<model_name>: {"anatomy": {Recall, Precision, F1, IoU, SuccessRate, "IoU>0.5", "F1>0.5",
  AccIoU_50, AccIoU_75, AccIoU_mean, total_samples, num_regions}, "T/L": {...}}}`.
- `summary_detection_task*.txt`: per model, one `ANATOMY` and one `T/L` line with the same quantities.
- `summary_TL_task*.txt`: per model a `Weighted Average MAE/MRE/SR/nMAE (Total Samples)` line, weighted `MRE<0.1/0.2/0.3`, then a
  label table `Label | MAE | MRE | SR | nMAE | MRE<0.1 | MRE<0.2 | MRE<0.3 | Samples` sorted by sample count.
- `summary_AD_task*.txt`: same header, then `Group averages` (FeTA-Distance, Ceph-Angle, Ceph-Distance, Distance, Angle) and the
  per-label table `Label | MAE | MRE | nMAE | SR | MRE<0.1 | MRE<0.2 | MRE<0.3 | Samples`.

## Box-size analysis outputs (Detection)

`summary_metrics_per_boxImgRatio_detect_Task.json` / `summary_values_per_boxImgRatio_detect_Task.json`: keyed by the 19 box-to-image
ratio bins; `summary_metrics_per_sample_detect_Task.csv` and `summary_metrics_boxImgRatio_x_label_detect_Task.csv` (anatomy-level),
`summary_metrics_per_sample_fineLabel_detect_Task.csv` and `summary_metrics_boxImgRatio_x_fineLabel_detect_Task.csv` (fine labels).
A `random_detection/` sub-folder produced by `analyze_detection_task_boxsize_vs_random` holds the random-box baseline and is skipped
by `summarize_detection_task`.
