# Release v1.1.0

## Summary

- **TL sample filtering fix.** The old filter (`len(biometric_profile) > 1`) only counted clusters that survived the pixel-size threshold, so sub-threshold clusters remained invisible to it — yet still visible in the image, making the measurement task ambiguous. The filter is corrected to use the raw connected-component count (`n_total_clusters > 1`) captured before any threshold is applied. New annotation files (`v1.1.0`) are generated to store this count; see [TL Task: Corrected Sample Filtering](#tl-task-corrected-sample-filtering) for details.


- **Backward compatibility via version control.** Because the corrected filter requires new annotation files, version control is introduced to let users load either the old (`v1.0.0`) or new (`v1.1.0`) annotations independently of the codebase version. Set `MedVision_PLANNER_VERSION` to select which annotation files are loaded; when loading v1.0.0 files the data loader automatically falls back to the original filter. See [Version Update](#version-update) for details.

  ```bash
  export MedVision_PLANNER_VERSION=1.0.0  # load original annotations ⚠️
  export MedVision_PLANNER_VERSION=1.1.0  # load updated annotations (default)
  ```

- **Cluster size threshold reduced from 200 to 20 pixels.** The minimum cluster size for ellipse fitting is lowered, so smaller single-cluster slices that were previously discarded now produce valid annotation entries and are included as training samples. See [TL Task: Reduced Cluster Size Threshold](#tl-task-reduced-cluster-size-threshold) for details.


- **Image normalization for visualization.** A complete normalization module (`src/medvision_ds/utils/image_normalization.py`) is added to the codebase, enabling correct image display in landmark visualization figures. CT images use HU window-based normalization selected by anatomy group; all other modalities use percentile min-max normalization. See [Image Normalization](#image-normalization) for details.

- **Versioned landmark and figure folders.** The data generation pipeline now appends `-v{version}` to `landmark_folder` and `landmark_figure_folder` paths (e.g. `Landmarks-Label1-v1.1.0/`), making every generated file traceable to the codebase version that produced it. This applies to the `preprocess_biometry.py` scripts. See [Annotation Data Generation](#annotation-data-generation) for details.

- **Version-aware download state tracker.** The `.downloaded_datasets.json` cache file now stores version strings instead of booleans, tracking codebase and annotation data versions independently. Users upgrading from v1.0.0 automatically receive the new annotation files; users who set `MedVision_PLANNER_VERSION=1.0.0` are not forced to re-download. Legacy boolean entries are treated as v1.0.0. See [Download State Tracker](#download-state-tracker) for details.

---


## TL Task: Corrected Sample Filtering

**Affected task:** Tumor-Lesion-Size (TL)

### Problem

The previous filter in `MedVision.py` used `len(case["biometric_profile"]) > 1` to reject slices with multiple target regions. `biometric_profile` only contains clusters that survived the pixel-size threshold, so sub-threshold clusters were invisible to the filter — yet they remained visible in the image presented to the model, making the measurement task ambiguous.

### Fix

**`src/medvision_ds/utils/benchmark_planner.py`**

- `__fit_ellipses`: the total number of connected components (`n_total_clusters`) is now captured from `scipy.ndimage.label` before any size threshold is applied, and returned alongside the landmark list.
- `_extract_ellipse_landmarks`: `n_total_clusters` is stored in each per-slice entry of the landmark JSON file.
- `_get_biometrics_batch`: `n_total_clusters` is propagated from the landmark JSON into the benchmark plan slice profile.
- `MedVision_BenchmarkPlannerBiometry_fromSeg.flatten_slice_profiles_2d`: `n_total_clusters` is included in every flattened sample dict.

**`MedVision.py`** (TL data loading)

- Filter changed from `len(case["biometric_profile"]) > 1` to `case["n_total_clusters"] > 1`.
- Slices where the raw binary mask contains more than one connected component are now rejected, regardless of whether those clusters survived the size threshold.
- **Backward compatible:** when `MedVision_PLANNER_VERSION=1.0.0` is set, the loader reads v1.0.0 JSON files which do not contain `n_total_clusters`. In this case the filter automatically falls back to the original `len(biometric_profile) > 1` behaviour, so no error is raised and results match the original release.

### Bug Fix

**`src/medvision_ds/utils/benchmark_planner.py`** (`_find_scaled_bounding_boxes_2D`)

- Fixed a `TypeError: 'numpy.bool' object cannot be interpreted as an integer` raised by `scipy.ndimage.find_objects`. The boolean cluster mask is now cast to `np.int32` before being passed to `find_objects`.

---


## Version Update

- `src/medvision_ds/__version__.py`: bumped `__version__` from `1.0.0` to `1.1.0`.
- `MedVision.py` (`MedVisionConfig.__init__`): update version from `1.0.0` to `1.1.0`. Noth that this version string must be hardcoded.
- `MedVision.py` (`_split_generators`): `MedVision_PLANNER_VERSION` is **required**. If the variable is not set, an `EnvironmentError` is raised at dataset load time with a message explaining both annotation versions. This is a **breaking change** for v1.0.0 users — existing scripts must export the variable before loading the dataset.

  The variable controls which benchmark plan JSON file is loaded, independent of the installed codebase version. The special value `latest` resolves to the current codebase version at runtime.

  | `MedVision_PLANNER_VERSION` | Benchmark plan file loaded |
  |-----------------------------|---------------------------|
  | `1.1.0` or `latest`         | `benchmark_plan_*_v1.1.0.json.gz` |
  | `1.0.0`                     | `benchmark_plan_*_v1.0.0.json.gz` |
  | not set                     | `EnvironmentError` raised |

  Recommended migration for existing v1.0.0 users — add one of the following before loading:

  ```bash
  export MedVision_PLANNER_VERSION=latest   # always use the newest annotations
  export MedVision_PLANNER_VERSION=1.0.0    # keep original v1.0.0 annotations ⚠️
  ```

---


## TL Task: Reduced Cluster Size Threshold

**Affected files:** all six TL-task `preprocess_biometry.py` scripts

| Dataset | File |
|---------|------|
| BraTS24 | `src/medvision_ds/datasets/BraTS24/preprocess_biometry.py` |
| autoPET_III | `src/medvision_ds/datasets/autoPET_III/preprocess_biometry.py` |
| KiTS23 | `src/medvision_ds/datasets/KiTS23/preprocess_biometry.py` |
| KiPA22 | `src/medvision_ds/datasets/KiPA22/preprocess_biometry.py` |
| HNTSMRG24 | `src/medvision_ds/datasets/HNTSMRG24/preprocess_biometry.py` |
| MSD | `src/medvision_ds/datasets/MSD/preprocess_biometry.py` |

`CLUSTER_SIZE_THRESHOLD` reduced from `200` to `20` pixels. The threshold determines the minimum cluster size for which an ellipse is fitted and a biometric profile entry is created. The lower threshold allows smaller but geometrically valid clusters to be included as samples.

---


## Image Normalization

NOTE: This update is marginal if the PNG files are not used in your pipeline.

PNG files are created for quality control and visual inspection. We display the normalized images (HU-based image intensity normalization, check [this issue](https://github.com/YongchengYAO/MedVision/issues/7)) in the updated figures. 

**New file:** `src/medvision_ds/utils/image_normalization.py`

Previously, landmark visualization figures displayed raw pixel values without normalization, resulting in poor contrast (near-black images for CT, arbitrary intensity ranges for MRI). This release adds a complete normalization mechanism to the `medvision_ds` codebase, ported from `medvision_bm/sft/sft_utils.py` and `medvision_bm/utils/configs.py`.

### Normalization logic

For CT images, the label name is looked up in `LABEL_MAP_REGROUP` to determine the anatomy group, which is then used to select a preset HU window from `CT_HU_WINDOWS_WL`. The image is clipped to the window and mapped to [0, 255]. Two exceptions fall back to percentile normalization:
- The label is not found in `LABEL_MAP_REGROUP` (mapped to "Others").
- The dataset/task appears in `TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION` (e.g. contrast CT in KiPA22, where HU windows are unreliable).

For all non-CT modalities (MRI, PET, etc.), percentile (0.5–99.5) min-max normalization is applied.

### Contents of `image_normalization.py`

| Symbol | Description |
|--------|-------------|
| `normalize_img` | Main dispatcher — routes to HU-window or percentile normalization |
| `normalize_ct_img` | HU window clip → [0, 255] |
| `normalize_general_img` | Percentile (0.5–99.5) min-max → [0, 255] |
| `CT_HU_WINDOWS_WL` | Per-anatomy HU window presets (width, level) |
| `LABEL_MAP_REGROUP` | Maps fine-grained label names to anatomy groups |
| `TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION` | Tasks that override CT normalization with percentile fallback |

### Integration with `benchmark_planner.py`

`__plot_img_ellipse_landmarks` now calls `normalize_img` before `plt.imshow`, receiving `image_modality`, `label_name`, `dataset_name`, `taskID`, and `taskType` extracted from `task_info`. Visualization is enabled in `scripts/regenerate_tl_biometry.py`.

---


## New T/L Annotation Data Generation

### Versioned landmark and figure folders

**`src/medvision_ds/utils/benchmark_planner.py`** (`MedVision_BenchmarkPlannerBiometry_fromSeg.process_each_task`)

The planner now appends `-v{self.version}` to `landmark_folder` and `landmark_figure_folder` in each task before processing begins. This applies automatically to every caller — the original `preprocess_biometry.py` scripts and the regeneration wrapper — without any per-dataset configuration.

Example output layout for BraTS24:
```
BraTS24/
├── benchmark_plan_biometry_v1.1.0.json.gz
└── BraTS24-GLI/
    ├── Landmarks-Label1-v1.1.0/      ← landmark JSON files
    └── Landmarks-Label1-fig-v1.1.0/  ← visualization figures
```

Old v1.0.0 folders (`Landmarks-Label1/`, `Landmarks-Label1-fig/`) are untouched. Both versions coexist in the same dataset directory.

### Regeneration script: `scripts/regenerate_tl_annotation_v1.1.0.py`

This script regenerates TL-task annotation files for a single dataset in a non-destructive way. It separates raw image/mask data (folder specified by `--data_dir`) from annotation outputs (set by `--output_dir`).

**What it produces:**

| Output | Location |
|--------|----------|
| New benchmark plan | `{output_dir}/{dataset_name}_regen/benchmark_plan_biometry_v1.1.0.json.gz` |
| Landmark JSON files | `{output_dir}/{dataset_name}_regen/{subdir}/Landmarks-{label}-v1.1.0/` |
| Visualization figures | `{output_dir}/{dataset_name}_regen/{subdir}/Landmarks-{label}-fig-v1.1.0/` |
| Removed samples report | `{output_dir}/{dataset_name}_regen/removed_samples_report.json` |

The `removed_samples_report.json` lists every slice that passed the old filter (`len(biometric_profile) ≤ 1`) but is rejected by the new filter (`n_total_clusters > 1`), with `image_file`, `slice_dim`, `slice_idx`, and `n_total_clusters` for each entry. This lets you identify which samples were ambiguous in v1.0.0.

**Usage:**

```bash
python scripts/regenerate_tl_annotation_v1.1.0.py \
    --dataset_name BraTS24 \
    --data_dir   /path/to/raw/image/folder \
    --output_dir /path/to/new/annotations/folder
```

Run once per dataset. Supported datasets and the `--dataset_name` value to use:

| Dataset | `--dataset_name` |
|---------|-----------------|
| BraTS24 | `BraTS24` |
| autoPET-III | `autoPET-III` |
| KiTS23 | `KiTS23` |
| KiPA22 | `KiPA22` |
| HNTSMRG24 | `HNTSMRG24` |
| MSD | `MSD` |

After regeneration, merge the `{dataset_name}_regen/` subdirectories and `benchmark_plan_biometry_v1.1.0.json.gz` into the published `{dataset_name}/` directory. The versioned folder names ensure no old files are overwritten.

---

## Summary of the T/L Annotations Update

### How the comparison was generated

The numbers below were produced by running `scripts/run_compare_planners.sh`, which iterates over all six datasets and calls `scripts/compare_planners_v1.0.0_to_v1.1.0.py` for each one. The script reads both `benchmark_plan_biometry_v1.0.0.json.gz` and `benchmark_plan_biometry_v1.1.0.json.gz` from the dataset folder, indexes every 2D slice entry by the key `(task_ID, image_file, slice_dim, slice_idx)`, and classifies changes into the three categories described below. Per-dataset JSON reports are written alongside the annotation files.

### Three categories of change

**Removed (key absent)** — slices present in v1.0.0 annotations but absent from v1.1.0 annotations.

These slices were dropped during v1.1.0 annotation regeneration by the `all_within` bounding-box quality check in `__fit_ellipses` (`benchmark_planner.py`). The check requires all four ellipse landmark points (P1–P4, the major/minor axis endpoints) to lie in the buffer zone between the cluster's 0.9× shrunk and 1.1× enlarged bounding boxes. Slices that fail have their landmark points inside the shrunk bounding box, meaning the fitted ellipse is disproportionately small relative to the cluster extent and would produce unreliable measurements. These slices were present in v1.0.0 because the v1.0.0 annotations were generated before this quality filter was in place.

**Multi-cluster in v1** — slices present in both versions whose v1.1.0 annotation carries `n_total_clusters > 1`.

These slices have more than one connected component in the raw binary mask (counted before any size threshold). They are retained in the v1.1.0 annotation file but will be rejected at data-loading time by the corrected TL filter in `MedVision.py` (`case["n_total_clusters"] > 1`). This group is a subset of "Common"; the per-dataset JSON report lists each affected slice together with its `n_total_clusters` value.

**Added (v1 \ v0)** — slices present in v1.1.0 annotations but absent from v1.0.0 annotations.

These are entirely new entries produced by the cluster size threshold reduction from 200 to 20 pixels. Slices containing small but geometrically valid clusters (20–199 pixels) that previously fell below the threshold now produce successful ellipse fits and enter the annotation.

### Per-dataset numbers

```
========== autoPET-III ==========
v1.0.0 total samples : 742
v1.1.0 total samples : 3773
Common               : 491
Removed (key absent) : 251  → autoPET-III/removed_samples_v1.0.0_to_v1.1.0.json
Multi-cluster in v1  : 287  → autoPET-III/multi_cluster_samples_v1.0.0_to_v1.1.0.json
Added   (v1 \ v0)    : 3282  → autoPET-III/added_samples_v1.0.0_to_v1.1.0.json

========== BraTS24 ==========
v1.0.0 total samples : 11082
v1.1.0 total samples : 30248
Common               : 8289
Removed (key absent) : 2793  → BraTS24/removed_samples_v1.0.0_to_v1.1.0.json
Multi-cluster in v1  : 2380  → BraTS24/multi_cluster_samples_v1.0.0_to_v1.1.0.json
Added   (v1 \ v0)    : 21959  → BraTS24/added_samples_v1.0.0_to_v1.1.0.json

========== HNTSMRG24 ==========
v1.0.0 total samples : 1404
v1.1.0 total samples : 4422
Common               : 1117
Removed (key absent) : 287  → HNTSMRG24/removed_samples_v1.0.0_to_v1.1.0.json
Multi-cluster in v1  : 322  → HNTSMRG24/multi_cluster_samples_v1.0.0_to_v1.1.0.json
Added   (v1 \ v0)    : 3305  → HNTSMRG24/added_samples_v1.0.0_to_v1.1.0.json

========== KiPA22 ==========
v1.0.0 total samples : 3095
v1.1.0 total samples : 3142
Common               : 3095
Removed (key absent) : 0  → KiPA22/removed_samples_v1.0.0_to_v1.1.0.json
Multi-cluster in v1  : 55  → KiPA22/multi_cluster_samples_v1.0.0_to_v1.1.0.json
Added   (v1 \ v0)    : 47  → KiPA22/added_samples_v1.0.0_to_v1.1.0.json

========== KiTS23 ==========
v1.0.0 total samples : 8512
v1.1.0 total samples : 14082
Common               : 7462
Removed (key absent) : 1050  → KiTS23/removed_samples_v1.0.0_to_v1.1.0.json
Multi-cluster in v1  : 893  → KiTS23/multi_cluster_samples_v1.0.0_to_v1.1.0.json
Added   (v1 \ v0)    : 6620  → KiTS23/added_samples_v1.0.0_to_v1.1.0.json

========== MSD ==========
v1.0.0 total samples : 7570
v1.1.0 total samples : 16232
Common               : 6060
Removed (key absent) : 1510  → MSD/removed_samples_v1.0.0_to_v1.1.0.json
Multi-cluster in v1  : 2407  → MSD/multi_cluster_samples_v1.0.0_to_v1.1.0.json
Added   (v1 \ v0)    : 10172  → MSD/added_samples_v1.0.0_to_v1.1.0.json
```

---


## Download State Tracker

**`MedVision.py`** (`_split_generators`)

The `.downloaded_datasets.json` cache file previously stored boolean `True` for every completed download. Once set, the entry never expired, so users who upgraded from v1.0.0 would never receive new annotation files.

### Changes

The tracker now stores **version strings** for all three tracked keys:

| Key | Value stored | Re-download trigger |
|-----|-------------|-------------------|
| `medvision_ds` | codebase version (e.g. `"1.1.0"`) | stored value `!=` `self.config.version` |
| `medvision_ds_installed` | codebase version | stored value `!=` `self.config.version` |
| `dataset_{name}` | requested annotation version | stored value `!=` `MedVision_PLANNER_VERSION` |

Annotation data tracking is independent of codebase tracking: `MedVision_PLANNER_VERSION` determines which annotation version to check against, defaulting to the current codebase version.

### Backward compatibility with legacy boolean entries

Users upgrading from v1.0.0 have a checker file that looks like this:

```json
{
  "medvision_ds": true,
  "medvision_ds_installed": true,
  "dataset_BraTS24": true
}
```

The new code handles these legacy boolean `True` values via `_version_tuple()`, which converts any non-version value (including booleans) to `(1, 0, 0)` — the tuple representation of v1.0.0. This reflects the fact that all data downloaded under the old mechanism was v1.0.0.

**Scenario: user upgrades to v1.1.0 codebase and sets `MedVision_PLANNER_VERSION=1.0.0`**

- Codebase check: `True != "1.1.0"` → re-downloads and installs v1.1.0 code. ✓
- Data check: `_version_tuple(True)` → `(1, 0, 0)`; `_version_tuple("1.0.0")` → `(1, 0, 0)`; equal → **no re-download**. ✓

The old data already on disk is used as-is. No conflict, no unnecessary re-download.

**Scenario: user upgrades to v1.1.0 and sets `MedVision_PLANNER_VERSION=1.1.0`**

- Data check: `_version_tuple(True)` → `(1, 0, 0)`; `_version_tuple("1.1.0")` → `(1, 1, 0)`; not equal → **re-downloads** the new annotation zip. ✓

---


