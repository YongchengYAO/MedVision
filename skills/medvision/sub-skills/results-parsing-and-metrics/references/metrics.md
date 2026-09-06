# Metric definitions (verified against `medvision_bm.utils.parse_utils`, the three summarizers and `medvision_utils`)

## 1. Answer extraction (what counts as an answer)

`extract_last_k_nums_within_answer_tag(text, k)`:

1. Take the content of the FIRST `<answer> ... </answer>` block (`re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)`).
   No block -> `""` -> the sample is a failure, even if the correct numbers appear elsewhere (`\boxed{}`, `**Answer:**`, prose).
2. Find all numbers in that block with `[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?`; thousands separators are removed
   (`8,000.25` -> `8000.25`).
3. Return the LAST k numbers joined by `,`; fewer than k numbers -> `""` (failure). Extra numbers earlier in the block are ignored,
   so `<answer>12.5 mm x 8.2 mm (slice 42)</answer>` with k = 2 yields `8.2,42`: a wrong answer, not a failure.

| Task | k | `target` string (parsed with `ast.literal_eval`) | Units |
|---|---|---|---|
| Angle/Distance (A/D) | 1 | `"27.5"` | mm for `metric_type == "distance"`, degrees for `"angle"` |
| Tumor/Lesion size (T/L) | 2 | `"[major_axis, minor_axis]"` | mm (dataset `metric_unit`) |
| Detection | 4 | `"[x_min, y_min, x_max, y_max]"` | relative coordinates in [0, 1] |

Eval-time scoring (`process_results_*` in `medvision_utils`) uses `parser_last_k_nums`, which delegates to the same function, so
`parse_outputs` reproduces the eval-time numbers. Raw JSONLs written before that alignment may carry an eval-time `nMAE` computed
from the whole response; `parse_outputs` passes a stored `nMAE` through untouched, and the summarizers neutralize it by gating
nMAE on the strict parse succeeding.

Detection box format (`doc_to_target_BoxCoordinate`): `[coor0_w, coor0_h, coor1_w, coor1_h]` = lower-left corner then upper-right
corner, origin at the LOWER-LEFT of the displayed image, each value divided by image width/height. The planner's array-index boxes
(`bounding_boxes.min_coords/max_coords`, top-left origin) are converted with `img_h = H - planner_h`. The overlap functions sort
each corner pair, so a prediction given as `[x_max, x_min, ...]` is not penalized for ordering.

## 2. Per-sample metrics

`cal_metrics(results, task_type)` with `results = {"filtered_resps": [<string>], "target": <string>}`:

- Prediction string -> split on `,` -> `float32` values. Length must equal k; anything else (empty string, wrong count,
  non-numeric) is a **failure**: `success = False`.
- **A/D and T/L**: `MAE = mean(|pred - gt|)` (mean over the 2 axes for T/L; n = 1 for A/D), `MRE = mean(|pred - gt| / (gt + 1e-15))`.
  Failure -> `MAE = NaN`, `MRE = NaN`. Returned dict: `{"avgMAE": {"MAE", "success"}, "avgMRE": {"MRE", "success"}, "SuccessRate": {"success"}}`.
- **Detection** (`cal_metrics_detection_task`, the single implementation used by both `parse_outputs` and the summarizer):
  `MAE` over the 4 relative coordinates, `IoU`, `F1`, `Precision`, `Recall` from the box geometry. Failure -> `MAE = NaN` but
  **`IoU = F1 = Precision = Recall = 0`**. Returned dict adds `"avgIoU": {"IoU"}, "F1": {"F1"}, "Precision": {"Precision"}, "Recall": {"Recall"}`.
- **nMAE** (`cal_metrics_TL_task` / `cal_metrics_AD_task` in the summarizers, `process_results_*` at eval time):
  `nMAE = MAE / diagonal`, `diagonal = sqrt((H * px_h * s_h)^2 + (W * px_w * s_w)^2)` with `H, W = doc.image_size_2d`, pixel sizes from the
  NIfTI header (mm) and `s_h = s_w = 1` for regular tasks. T/L: every successful sample. A/D: only `metric_type == "distance"`;
  angles get `{"NMAE": NaN, "success": False}`. Failed parses never get an nMAE. Precedence in the summarizers: stored record `nMAE`
  -> recompute from stored `pixel_size_scale` -> hash-derived scale (scaledPS only) -> NaN if the NIfTI file cannot be read.

Box geometry (`cal_IoU`, `cal_F1`, `cal_Precision`, `cal_Recall`): intersection `I` of the two axis-aligned boxes;
`IoU = I / (A_pred + A_gt - I)`, `F1 = 2I / (A_pred + A_gt)` (Dice; NaN if both areas are 0), `Precision = I / A_pred` (NaN if `A_pred = 0`),
`Recall = I / A_gt`. Degenerate (zero-area) boxes are caught by the no-overlap test first and score **0.0** for all three — the NaN / `ValueError` branches below them in `parse_utils.py` are dead code. No overlap -> 0.
Values are clamped to `<= 1`.

## 3. Aggregation in the summarizers (per group, then reported)

Let `N` = `num_samples` = records in the group (after the A/D near-zero filter and the T/L removed-samples filter).

| Key | Definition | Denominator | Failures |
|---|---|---|---|
| `SuccessRate` | successful parses / N | N | counted as misses |
| `avgMAE`, `avgMRE` | mean of finite per-sample MAE / MRE | successes only | excluded (NaN) |
| `avgNMAE` | mean of nMAE with `success = True` | successes with a diagonal | excluded |
| `MRE<0.1` ... `MRE<1.0` (A/D, T/L) | count of finite MRE in cumulative 0.1-wide buckets / N; bucket index `min(int(MRE*10), 9)`, so the last bucket is `[0.9, inf)` and **`MRE<1.0` equals `SuccessRate`** | N | misses |
| `MAE<0.1` ... `MAE<1.0` (Detection) | same construction on the relative-coordinate MAE | N | misses |
| `IoU`, `F1`, `Precision`, `Recall` (Detection) | mean of finite values; failures are 0 and finite, so this is effectively the mean over ALL N samples | N (minus rare NaN from zero-area boxes) | counted as 0 |
| `IoU>0.5` ... `IoU>0.9`, `F1>k`, `Precision>k`, `Recall>k` | count(value >= k) / N (the key says `>`, the code uses `>=`) | N | misses |
| `Acc@IoU>=0.50` ... `Acc@IoU>=0.95` | count(IoU >= tau) / N on the COCO grid `{0.50, 0.55, ..., 0.95}`; `Acc@IoU>=0.50 == IoU>0.5` | N | misses |
| `Acc@IoU[0.50:0.95]` | mean of the ten `Acc@IoU>=tau` values | N | misses |

Consequences worth stating to users:

- Detection mean IoU is bounded above by `SuccessRate` in the sense that `IoU_reported = IoU_over_successes * SuccessRate`; a model
  that answers 60 % of the time with perfect boxes reports `IoU = 0.6`.
- `IoU>0.5` is a fraction of samples, not a mean; it is routinely far below the mean IoU when many boxes overlap partially
  (`IoU = 0.62` with `IoU>0.5 = 0.41` is consistent, not a bug: 41 % of ALL samples reached 0.5, while the successful boxes
  average well above it).
- `parse_outputs`' own `<ts>_results.json` uses the same rules: `avgMAE,none`/`avgMRE,none` over successes, `avgIoU,none` over finite
  values (failures = 0), `SuccessRate,none` and `MRE<k` over the total count. **One difference in the buckets:** the summarizers
  clamp the bucket index (`min(int(value * 10), 9)`), so their last bucket is `[0.9, inf)` and `MRE<1.0 == SuccessRate`.
  `parse_outputs` does not clamp, so a sample with `MRE >= 1.0` falls in no bucket and its `MRE<1.0` is strictly the fraction
  below 1.0 — lower than `SuccessRate`. The identity holds only in `summary_metrics_*`.
- Cross-group and cross-model report rows (`summary_*_task.txt`, `summary_metrics_anatomy_vs_lesion_detect_Task.json`,
  `summary_metrics_all_models_detect_Task.json`) are **sample-weighted (micro) averages** of the group values; they skip NaN
  group values and, for detection, regions with fewer than `MINIMUM_GROUP_SIZE = 50` samples or whose name contains one of
  `EXCLUDED_KEYS = ["miscellaneous", "others"]`. A region is `T/L` when its name contains any of
  `TUMOR_LESION_GROUP_KEYS = ["tumor", "lesion", "metastatic"]`, else `anatomy`. The T/L summarizer prints these three constants
  but applies none of them (it keeps every label group).

## 4. Grouping keys

| Task | Grouping function | Key | Label source |
|---|---|---|---|
| Detection | `group_by_anatomy_modality_slice` | `"<label_map_regroup[label]> @ <MR|CT|US|XR|PET> (<S|C|A>)"` | `doc.label` -> `labels_map` of the dataset's segmentation benchmark plan |
| T/L | `group_by_label_modality_slice` | `"<label_map_rename[label]> @ <modality> (<plane>)"` | `target_label` -> `labels_map` of the biometry benchmark plan |
| A/D | inline in `summarize_AD_task` | `"<dataset>_<metric_type>_<metric_key>"` | `doc.biometric_profile` |
| Box size (analysis) | `group_by_boxImgRatio` | `"Box/Image < 5%"`, `"5% <= Box/Image < 10%"`, ... `"90% <= Box/Image"` | `box_img_ratio` written by `parse_outputs` |

Modality codes: `MRI -> MR`, `ultrasound -> US`, `X-ray -> XR`; `slice_dim 0/1/2 -> S/C/A` (sagittal/coronal/axial). A label absent from
`label_map_regroup` / `label_map_rename` raises `ValueError` (new datasets need entries in `medvision_bm.utils.configs`).

## 5. Sample filters

- **A/D near-zero ground truth**: `AD_NEAR_ZERO_GT_THRESHOLD = 0.1`. The summarizer's `process_label_group` skips samples whose GT
  is below it BEFORE counting (they are absent from `num_samples`). Eval time flags them (`near_zero_gt: True`) and
  `aggregate_results_SuccessRate` skips them. `parse_outputs` / `cal_metrics` do NOT apply the threshold, so
  `parsed/<ts>_results.json` still includes them. Reason: `MRE = |pred - gt| / gt` explodes as gt -> 0.
- **T/L removed samples** (`--removed_samples_dir`): key `(image_file relative to the dataset folder, slice_dim as int, slice_idx, task_ID)`
  from `<dataset>/multi_cluster_samples_v1.0.0_to_v1.1.0.json` (entries: `task_ID`, `image_file`, `case_ID`, `split`, `slice_dim` in
  `x|y|z`, `slice_idx`, `n_total_clusters`). Matching records are skipped but still count toward `--limit`.
- **`--limit N`**: first N records of each file (parse: lowest `doc_id`s; summarize: first N lines of the parsed file).
- Files whose names contain `_proc_acc` or `_eq_acc` (process/equation-accuracy analyses) are ignored by the summarizers.

## 6. Units and value ranges

| Quantity | Unit / range |
|---|---|
| T/L `MAE` | mm (average of the two axis errors) |
| A/D `MAE` | mm (distance) or degrees (angle); a label group never mixes the two |
| `MRE` | unitless ratio; can exceed 1 by a lot (a 300 mm answer for 12 mm gives 24) |
| `nMAE` | unitless (mm / mm); typical good values are 0.01 to 0.1 |
| Detection `MAE` | relative-coordinate units (0 to 1 for in-range boxes; pixel answers give values > 1) |
| `IoU`, `F1`, `Precision`, `Recall`, `SuccessRate`, `*<k`, `*>k`, `Acc@IoU*` | fractions in [0, 1] |

## 7. Pixel-size-scaled variant (scaledPS) - reference only

Task lists `tasks_MedVision-TL-CoT-scaledPS.json` / `tasks_MedVision-AD-CoT-scaledPS.json` (config names end in `-CoT-scaledPS`) use
`create_doc_to_text_*_scaledPS` prompts: the pixel size shown in the prompt is multiplied by a deterministic per-sample factor
(`_get_pixel_size_scale_factor`: BLAKE2b hash of `image_file|slice_dim|slice_idx|taskID|label`, uniform in
`[MEDVISION_SCALED_PS_LOW, MEDVISION_SCALED_PS_HIGH]`, defaults 0.5 and 3.0), while the image is unchanged. T/L scales both axes by
one `S` (`uniform`); A/D draws `(S_h, S_w)` (`anisotropic`) and recomputes the distance/angle from landmarks in the scaled physical
space (angles change only when `S_h != S_w`). The ground truth is rescaled accordingly, so a model that reasons from the stated
pixel size keeps its score while one that ignores it is penalized in proportion to `S`. `process_results_*_scaledPS` stores
`pixel_size_scale = {"s_h", "s_w", "mode"}` in the record, and nMAE uses the scaled diagonal. `parse_outputs` and the summarizers
detect the variant from `scaledPS` in the file name (`uniform` for TL, `anisotropic` for AD); records without a stored scale need
the same `MEDVISION_SCALED_PS_LOW/HIGH` values as the eval run. Running the variant is an eval-step concern
(`../../benchmark-evaluation/SKILL.md`).
