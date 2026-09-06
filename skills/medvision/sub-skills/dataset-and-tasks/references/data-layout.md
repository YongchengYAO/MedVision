# On-Disk Layout and Benchmark Plans

## `Data/` tree after downloads

```text
<data_dir>/
  .downloaded_datasets.json          # tracker (see downloading.md); *.lock files alongside
  .cache/huggingface/{datasets,...}  # HF cache when HF_HOME/HF_DATASETS_CACHE point here
  src/                               # medvision_ds source snapshot (pip-installed from here)
  SFT-CoT_datasets/<family>/ds__AD<n>_D<n>_TL<n>_all<n><suffix>/   # prepared SFT datasets
  raw_parquet/ , verl_datasets/       # RFT parquet inputs and verl-format outputs
  Datasets/
    <dataset>/
      benchmark_plan_segmentation_v1.0.0.json.gz
      benchmark_plan_detection_v1.0.0.json.gz
      benchmark_plan_biometry_v1.0.0.json.gz  (+ _v1.1.0, _v1.1.1, _v1.4.0 for T/L datasets)
      Images/  *.nii.gz              # 3D volumes, RAS+ (some datasets: Images-<modality>/)
      Images/tmp_prepared_png/       # PNG cache when save_processed_img_to_disk is used
      Masks/   *.nii.gz
      Landmarks[-LabelN][-vX.Y.Z]/   *.json.gz  # biometry landmarks, versioned since 1.1.0
      Landmarks[-LabelN]-fig[-vX.Y.Z]/          # QC figures (opt-in)
      <sub-dataset>/Images*,Masks,Landmarks*    # multi-part datasets: AMOS22-CT/-MRI, BCV15-*, BraTS24-GLI/-MEN-RT/-MET/-PED, HNTSMRG24-*, MSD-<Task>, TopCoW24-CT/-MR
      *.json                          # optional per-release diagnostics (added_/removed_/changed_measurements_*.json)
```

`image_file`, `mask_file`, `landmark_file` in loaded samples are absolute paths rooted at `<data_dir>/Datasets/<dataset>/`, which is why two data roots get separate Arrow caches.

## Plan files

Name: `benchmark_plan_<kind>_v<X.Y.Z>.json.gz`, `kind` in `segmentation | detection | biometry`. Family to kind: `MaskSize -> segmentation`, `BoxSize -> detection`, `TumorLesionSize` and `BiometricsFromLandmarks -> biometry` (both biometry families write the same filename, which is why the loader refuses a dataset registered with both). Sizes vary by orders of magnitude: segmentation plans run ~0.2 MB to ~134 MB (whole-body CT — AbdomenAtlas1.0Mini 134 MB, TotalSegmentator 127 MB); detection plans store every box on every slice and reach hundreds of MB compressed for whole-body CT datasets; biometry plans ~0.005 MB to ~50 MB (the v1.4.0 T/L regeneration made BraTS24 50 MB and MSD 21 MB).

Schema (gzip JSON), verified on shipped plans:

```text
{
  "dataset_info": {dataset, dataset_website, dataset_data[], license[], paper[]},
  "tasks_number": N,
  "tasks": [
    {
      "task_ID": "01", "task_type": "segmentation|detection|biometry",
      "image_modality": "CT", "image_description": "...",
      "image_folder", "mask_folder", "image_prefix", "image_suffix", "mask_prefix", "mask_suffix",
      "labels_map": {"1": "renal vein", ...},
      # biometry only:
      "landmark_folder", "landmark_figure_folder", "landmark_prefix", "landmark_suffix",
      "landmarks_map", "lines_map", "angles_map", "biometrics_map", "target_label",
      "cluster_size_threshold", "min_major_axis_mm",
      "train_cases_number", "train_cases": [...], "test_cases_number", "test_cases": [...]
    }
  ]
}
case = {
  "case_ID", "image_file", "mask_file", ["landmark_file"],
  "image_file_info": {"voxel_size"[3], "affine"[4x4], "orientation", "array_size"[3]},
  "mask_file_info": {...},
  "slice_profiles_x": [...], "slice_profiles_y": [...], "slice_profiles_z": [...], ["profile_3D"]
}
slice entry (segmentation) = {"slice_idx", "slice_profile": [{"label", "pixel_count", "ROI_area"}]}
slice entry (detection)    = {"slice_idx", "slice_profile": [{"label", "bboxes": [{"min_coords","max_coords","center_coords","dimensions","sizes"}]}]}
slice entry (biometry, fromSeg/T-L) = {"slice_idx", "n_total_clusters", "slice_profile": [[{"metric_type","metric_map_name","metric_key","metric_value","metric_unit","slice_dim"}, ...]]}
slice entry (biometry, landmark family: AFIDs, Ceph-Biometrics-400, FeTA24, PDDCA, VerSe)
                           = {"slice_idx", "slice_profile": [{"metric_type","metric_map_name","metric_key","metric_value","metric_unit","slice_dim"}, ...]}   # no n_total_clusters; slice_profile is a FLAT list
```

Axis convention: volumes are RAS+, `array_size = [X, Y, Z]`; `slice_profiles_x` = Sagittal (`slice_dim 0`), `_y` = Coronal (1), `_z` = Axial (2). The 2D `(H, W)` of a slice is `array_size` with that axis removed, in array order.

Note that `labels_map` baked into a plan can be a stale snapshot (e.g. `"tumor"` vs `"kidney tumor"`); the maintainer summarizer reads the live map from `medvision_ds` when it can.

## `medvision_bm.utils.plan_utils` API (offline, no HF, no nibabel)

| Name | Behaviour |
| --- | --- |
| `AXIS_TO_PLANE = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}`, `PLANE_TO_AXIS` | axis/plane mapping |
| `FAMILY_TO_PLAN_TYPE` | `boxsize`/`masksize -> segmentation` (deliberately not the huge detection plan for image sizes), `tumorlesionsize`/`biometricsfromlandmarks -> biometry` |
| `find_plan_files(dataset_dir, plan_type)` | sorted glob of `benchmark_plan_<type>_v*.json.gz` |
| `plan_version_of(path)` | version tuple parsed from the filename |
| `resolve_plan_path(dataset_dir, plan_type, version=None)` | newest plan at or **below** `version` (the loader's ceiling rule), `None` if the family is absent or nothing was published at or before `version`; `version=None` = newest |
| `dataset_exists_at(dataset_dir, version=None)` | any plan of any kind resolves at `version` |
| `load_benchmark_plan(dataset_dir, plan_type, version=None)` | cached (`lru_cache(maxsize=2)`), returns the dict or `None`; warns once to stderr `[plan_utils] <dataset>: <type> plan v<ver> not found; using <file> instead` when the resolved file is not an exact version match. Treat the dict as read-only. |
| `split_cases(task, split)` | `train_cases`, `test_cases`, or both for `"all"` |
| `slice_2d_size(array_size, axis)` | `(H, W)` after dropping the sliced axis |
| `slice_entries(case, axis)` | `case["slice_profiles_<axis>"]` or `[]` |
| `anatomy_group(fine_label)` | coarse group via `configs.label_map_regroup`, `"UNMAPPED"` if unknown |

Why the ceiling rule replaced "exact match, else highest available": Ceph and FeTA ship biometry `v1.0.0` only, so a pinned `1.1.1` summary must still include them (ceiling does); but eight datasets first published at `1.2.0` and "highest available" leaked them into `1.0.0`/`1.1.0`/`1.1.1` summaries. With the ceiling, a dataset added later contributes nothing to an older version's summary, so pinning an older `--plan_version` reproduces that release's summary even when `Datasets/` also holds newer datasets. Across the 22 pre-1.2.0 datasets x 4 pins x 3 kinds the two rules agree on every case.

`scripts/inspect_benchmark_plan.py` exercises `find_plan_files`, `plan_version_of`, `resolve_plan_path`, `dataset_exists_at`, `load_benchmark_plan`, `split_cases`, `slice_entries` and `slice_2d_size` on one dataset directory.
