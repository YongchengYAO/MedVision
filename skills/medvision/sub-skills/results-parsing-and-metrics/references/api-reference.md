# Python API reference (signatures verified with `inspect.signature` on the installed package)

Import root: `medvision_bm.utils.parse_utils` unless stated otherwise. All functions are pure Python/NumPy; none needs a GPU.

## Extraction and conversion

| Signature | Behaviour |
|---|---|
| `extract_last_k_nums_within_answer_tag(text, k)` | Numbers inside the first `<answer>...</answer>`; last k joined by `,`; `""` if no tag or fewer than k numbers. Thousands separators stripped. |
| `extract_last_k_nums(text, k)` | Same regex over the WHOLE text (legacy; not used by the pipeline). |
| `convert_numpy_to_python(obj)` | Recursively turns `np.float32` -> `float`, `np.ndarray` -> list; used before `json.dump`. |
| `get_subfolders(task_dir, models=None)` | Immediate sub-directories (paths); `models` = iterable of basenames to keep. |
| `assert_resps_key(parsed_files_dir, jsonl_files, resps_key)` | Probes the first readable record; `SystemExit` with a `[FATAL]` message if the key is absent. |
| `load_nifti_2d(img_path, slice_dim, slice_idx)` | `(pixel_size, image_2d)`; pixel size = the two in-plane header zooms. |

## Scoring

| Signature | Returns |
|---|---|
| `cal_metrics(results, task_type)` | `results = {"filtered_resps": [str], "target": str}`; `task_type in {"AD","TL","Detection"}`. AD/TL -> `{"avgMAE": {"MAE","success"}, "avgMRE": {"MRE","success"}, "SuccessRate": {"success"}}`; Detection delegates to the next row. `ValueError` for other task types. |
| `cal_metrics_detection_task(results)` | `{"avgMAE": {"MAE","success"}, "avgIoU": {"IoU"}, "F1": {"F1"}, "Precision": {"Precision"}, "Recall": {"Recall"}, "SuccessRate": {"success"}}`; failure -> MAE NaN, overlaps 0. |
| `cal_IoU(pred, target)` | float in [0, 1]; corners sorted; `ValueError` unless both have exactly 4 numbers. |
| `cal_F1(pred, target)` | Dice `2I/(A_p + A_t)`; NaN when both areas are 0. |
| `cal_Precision(pred, target)` | `I / A_p`; NaN when `A_p = 0`. |
| `cal_Recall(pred, target)` | `I / A_t`; `ValueError` when `A_t <= 0`. |

Summarizer-side scorers (import from the summarizer modules):

| Signature | Notes |
|---|---|
| `medvision_bm.benchmark.summarize_TL_task.cal_metrics_TL_task(results)` | As `cal_metrics(..., "TL")` plus `"nMAE": {"NMAE","success"}`; `results` may carry `doc_meta` (`image_file, slice_dim, slice_idx, image_size_2d, scale_mode, nmae_precomputed, taskID, label, pixel_size_scale`). |
| `medvision_bm.benchmark.summarize_AD_task.cal_metrics_AD_task(results)` | Same for A/D; nMAE only when `doc_meta["metric_type"] == "distance"`. |
| `summarize_TL_task.process_label_group_TL(parent_class, data)` / `summarize_AD_task.process_label_group(label, data)` | `data = {"targets": [str], "responses": [str], "doc_metas": [...]}` -> `(key, metrics dict)`; the A/D version applies `AD_NEAR_ZERO_GT_THRESHOLD`. |
| `summarize_detection_task.calculate_summary_metrics_per_anatomy_detection_task(grouped_data)` | `grouped_data` from `group_by_anatomy_modality_slice` -> `{group: metrics}` including `IoU>k` and `Acc@IoU` keys. |
| `summarize_detection_task.acc_iou_key(threshold)`; constants `COCO_IOU_THRESHOLDS`, `ACC_IOU_MEAN_KEY` | `acc_iou_key(0.5) == "Acc@IoU>=0.50"`; grid `[0.5, 0.55, ..., 0.95]`; mean key `"Acc@IoU[0.50:0.95]"`. |
| `summarize_detection_task.group_anatomy_vs_tumor_lesion(model_path, limit=None)` | Reads the per-model metrics JSON in `model_path` and writes the anatomy-vs-T/L file next to it. |

## Grouping

| Signature | Input tuples | Output |
|---|---|---|
| `group_by_anatomy_modality_slice(data)` | `(imgModality, label_name, target, filtered_resps, _, slice_dim)` | `{"<regroup> @ <mod> (<plane>)": {"targets": [...], "responses": [...]}}`; `ValueError` if label not in `label_map_regroup` or `slice_dim` not in 0..2 |
| `group_by_label_modality_slice(data)` | same | keyed by `label_map_rename` |
| `group_by_boxImgRatio(data)` | `(_, target, filtered_resps, _, box_img_ratio, image_size_2d)` | 5 %-wide bins from `"Box/Image < 5%"` to `"90% <= Box/Image"`, each with `targets`, `responses`, `image_size_2d` |

## Benchmark-plan lookups (need `medvision_ds`)

| Signature | Returns |
|---|---|
| `get_labelsMap_imgModality_from_seg_benchmark_plan(dataset_name, task_id)` | `(labels_map, image_modality)` from `medvision_ds.datasets.<pkg>.preprocess_segmentation.benchmark_plan["tasks"][task_id-1]`; `ValueError` on import/lookup failure |
| `get_labelsMap_imgModality_from_biometry_benchmark_plan(dataset_name, task_id)` | same from `preprocess_biometry`; `{}` when the dataset is unknown to `DATASETS_NAME2PACKAGE` |
| `get_targetLabel_imgModality_from_biometry_benchmark_plan(dataset_name, task_id)` | `(target_label, image_modality)` |

`DATASETS_NAME2PACKAGE` (in `medvision_bm.utils.configs`) maps dataset names to importable package names
(`"AbdomenAtlas1.0Mini" -> "AbdomenAtlas__1_0__Mini"`, `"PI-CAI" -> "PICAI"`, ...). Importing one dataset's plan takes several seconds
the first time.

## Constants (`medvision_bm.utils.configs`)

`SEED = 1024`; `AD_NEAR_ZERO_GT_THRESHOLD = 0.1`; `MINIMUM_GROUP_SIZE = 50`; `EXCLUDED_KEYS = ["miscellaneous", "others"]`;
`TUMOR_LESION_GROUP_KEYS = ["tumor", "lesion", "metastatic"]`; `RANDOM_BOX_SIMULATIONS = 100`; `label_map_regroup` (266 labels ->
anatomy group), `label_map_rename` (265 labels -> canonical name); the `SUMMARY_FILENAME_*` constants listed in `output-files.md`.
Always import these by name instead of re-typing them.

## Eval-time functions (`medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils`)

Verified from source. The module imports `torch`, `transformers`, `PIL`, `scipy` and `medvision_ds` at import time, so it (and every
`medvision_bm.benchmark.parse_outputs` / `summarize_*` run, which import it) needs those packages installed; CPU-only builds are
fine and no GPU is used.

| Signature | Notes |
|---|---|
| `parser_last_k_nums(text, k)` | Delegates to `extract_last_k_nums_within_answer_tag`. |
| `doc_to_target_BoxCoordinate(doc, lmms_eval_specific_kwargs=None)` | Relative `[x0, y0, x1, y1]`, lower-left origin; honours `reshape_image_hw`. |
| `doc_to_target_TumorLesionSize(doc)` / `doc_to_target_BiometricsFromLandmarks(doc)` | `[major, minor]` / `metric_value`. |
| `process_results_BoxCoordinate(doc, results)` | `results = [response text]`; returns `avgMAE`, `avgMRE`, `SuccessRate` (no IoU at eval time). |
| `process_results_TumorLesionSize(doc, results)` | adds `nMAE`. |
| `process_results_BiometricsFromLandmarks(doc, results)` | returns `MAE {"AE"}`, `MRE {"RE"}`, `SuccessRate`, `nMAE`; near-zero GT -> all failures with `near_zero_gt: True`. |
| `aggregate_results_avgMAE(results)`, `aggregate_results_avgMRE(results)`, `aggregate_results_MAE(results)`, `aggregate_results_MRE(results)`, `aggregate_results_NMAE(results)` | mean over entries with `success == True`; NaN if none. |
| `aggregate_results_SuccessRate(results)` | successes / entries, skipping `near_zero_gt` entries; NaN if none. |
| `_compute_physical_diagonal(doc, scale_mode=None, *, explicit_scale)` | `sqrt((H*px_h*s_h)^2 + (W*px_w*s_w)^2)`; `explicit_scale = {"s_h","s_w"}` or `None`; `scale_mode in {None,"uniform","anisotropic"}`. Reads the NIfTI header (cached). |
| `_get_pixel_size_scale_factor(doc, mode)` | `"uniform"` -> float, `"anisotropic"` -> `(S_h, S_w)`; range from `MEDVISION_SCALED_PS_LOW/HIGH` (defaults 0.5 / 3.0). |

## Minimal programmatic use

```python
from medvision_bm.utils.parse_utils import cal_metrics, extract_last_k_nums_within_answer_tag

text = "<think>...</think><answer>12.0 mm x 8.0 mm</answer>"
record = {"filtered_resps": [extract_last_k_nums_within_answer_tag(text, 2)], "target": "[12.4, 8.3]"}
print(cal_metrics(record, "TL"))
# {'avgMAE': {'MAE': 0.35, 'success': True}, 'avgMRE': {'MRE': 0.034, 'success': True}, 'SuccessRate': {'success': True}}
```

`scripts/metrics_demo.py` runs this end to end for all three task types and asserts the failure semantics.
