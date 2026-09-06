# MedVision Dataset Concepts

All facts below were verified against the dataset loader script (`MedVision.py`, release `1.4.0`) and the `medvision_bm` sources.

## Vocabulary

| Term | Meaning |
| --- | --- |
| `MedVision` | The Hugging Face dataset `YongchengYAO/MedVision`: MedVision's own annotations (`Datasets/<dataset>.zip`), the dataset codebase (`src/` = `medvision_ds`), the loader script `MedVision.py`, and per-version config lists under `info/`. Raw images are **not** redistributed there; the loader downloads them from each source through `medvision_ds` per-dataset download scripts. |
| `dataset` | One public source dataset: `BraTS24`, `MSD`, `OAIZIB-CM`, `KiPA22`, ... (22 at v1.0.0-v1.1.1, 30 at v1.2.x, 31 at v1.3.0+). Dataset names never contain `_`, which is why config names can be split on `_`. |
| `data-config` | A named, ready-to-load subset passed as `name=` to `load_dataset`. |
| `task` (benchmark) | One evaluation (or SFT data) unit; each task name maps to one task YAML in the vendored `lmms_eval` and to exactly one config. See `task-lists.md`. |
| `benchmark plan` | `benchmark_plan_{segmentation,detection,biometry}_v<X.Y.Z>.json.gz` inside `<data_dir>/Datasets/<dataset>/`: the annotation file the loader flattens into samples. See `data-layout.md`. |

## Config naming

```text
{dataset}_{annotation-type}_{task-ID}_{slice}_{split}
```

| Field | Values | Notes |
| --- | --- | --- |
| `annotation-type` | `BoxSize` | detection (bounding boxes); loader `taskType="Box-Size"`, plan kind `detection` |
| | `TumorLesionSize` | tumor/lesion major/minor axis in mm from an ellipse fit; `taskType="Tumor-Lesion-Size"`, plan kind `biometry` (family `fromSeg`) |
| | `BiometricsFromLandmarks` | angle (degrees) / distance (mm) from landmarks; `taskType` one of `Biometrics-From-Landmarks`, `-Distance`, `-Angle`; plan kind `biometry` (family `landmark`) |
| | `MaskSize` | mask area; `taskType="Mask-Size"`, plan kind `segmentation` |
| `task-ID` | `Task01`, `Task02`, ... | **local** to the dataset (`Task01` of BraTS24 has nothing to do with `Task01` of MSD); defined by the dataset's `preprocess_*.py` in `medvision_ds` |
| `slice` | `Sagittal`, `Coronal`, `Axial` | slicing plane of the RAS+ volume: axis x/y/z; `slice_dim` 0/1/2 |
| `split` | `Train`, `Test` | subject-level split (70/30) |

Angle/distance configs of `Ceph-Biometrics-400` carry an extra metric token: `Ceph-Biometrics-400_BiometricsFromLandmarks_Distance_Task01_Sagittal_Test` and `..._Angle_Task01_Sagittal_Test`; `FeTA24_BiometricsFromLandmarks_Task01_{Sagittal,Coronal,Axial}_Test` has none (its config yields both metric types, split downstream by `metric_type`).

Examples: `OAIZIB-CM_BoxSize_Task01_Axial_Test`, `BraTS24_TumorLesionSize_Task01_Axial_Train`, `KiPA22_TumorLesionSize_Task01_Axial_Test`.

The complete list per release is shipped as CSV (no header, one config per line): `ConfigurationsList_{All,Test,Train}.csv` for `v1.0.0-v1.1.1` (818 configs), `v1.2.0` (920), `v1.3.0` (998), `v1.4.0` (1002). The `split` argument of `load_dataset` must match the split baked into the config name (`"test"` for `*_Test`, `"train"` for `*_Train`).

## Fields returned per config

The loader defines one feature dict per annotation type; the dict yielded by `_generate_examples()` matches it. Common to all: `dataset_name`, `taskID`, `taskType`, `image_file` (absolute path to the 3D NIfTI under `<data_dir>/Datasets/...`), `slice_dim` (uint8), `slice_idx` (uint16), `image_size_2d` ([H, W] uint16), `pixel_size` ([h, w] mm, float16), `image_size_3d` (uint16 x3), `voxel_size` (float16 x3).

**MaskSize** adds `mask_file`, `label` (uint16), `pixel_count` (uint32), `ROI_area` (float16).

**BoxSize** adds `mask_file`, `label`, and `bounding_boxes`: a sequence of `{min_coords[2], max_coords[2], center_coords[2], dimensions[2]}` (uint16) and `sizes[2]` (float16, mm). With single-instance filtering the sequence has exactly one box.

**BiometricsFromLandmarks** adds `landmark_file` and one `biometric_profile` dict: `metric_type` (`"distance"`|`"angle"`), `metric_map_name`, `metric_key`, `metric_value` (float16), `metric_unit`, `slice_dim`.

**TumorLesionSize** adds `landmark_file`, `mask_file`, `label` (the task's `target_label`), and `biometric_profile` as a sequence of `{metric_type, metric_map_name, metric_key_major_axis, metric_value_major_axis, metric_key_minor_axis, metric_value_minor_axis, metric_unit}`.

Numeric fields are stored as `float16`/`uint16`; convert before arithmetic that needs more precision.

## Single-instance vs multi-instance samples

A benchmark sample is a *(2D slice, target)* pair; several instances of one target on one slice still count as one sample. By default the loader applies per-sample filters ("single-instance"); `MedVision_DISABLE_SAMPLE_FILTERING=true` (value compared case-insensitively to `"true"`) bypasses them ("multi-instance", a superset). The distance/angle split by `metric_type` is always kept.

| Task type | Dropped when filtering is on |
| --- | --- |
| Box-Size | the slice has more than one box for the target (`len(bounding_boxes) > 1`), **or** the single box is `< 10` px on either side |
| Tumor-Lesion-Size | `n_total_clusters > 1` (annotation v1.1.0+), else `len(biometric_profile) > 1` (v1.0.0 fallback) |
| Mask-Size | `pixel_count < 200` |
| Biometrics-From-Landmarks(-Distance/-Angle) | never dropped |

Multi-instance samples are not for leaderboard comparison and MedVision-V0 was not trained for them. Enabling the flag also changes the HF builder cache fingerprint (`disable_sample_filtering=True` is added to the config id), so filtered and unfiltered builds never share an Arrow cache.

## Annotation versions

`MedVision_PLANNER_VERSION` selects the newest annotation you are willing to load; each `(dataset, plan kind)` then resolves to the newest version it published **at or below** that ceiling. Accepted values: `latest` (= release `1.4.0`) or a published three-part version. Malformed (`v1.1.1`, `1.2`) or unpublished (`1.1.5`) values are refused.

| Version | Change (from the loader's own notes and release documents) |
| --- | --- |
| `1.0.0` | original T/L filtering, cluster threshold 200 px. **Leaderboard annotations.** 22 datasets. |
| `1.1.0` | corrected T/L filtering (`n_total_clusters`), cluster threshold 20 px; version control introduced (`MedVision_PLANNER_VERSION`); versioned `Landmarks-*-v<ver>` folders; tracker stores versions. |
| `1.1.1` | fixes transposed in-plane voxel spacing in the T/L ellipse fit; ~22 % fewer T/L samples on anisotropic (sagittal/coronal) slices, axial essentially unchanged; used for plane-OOD ablations. |
| `1.2.0` | adds 8 datasets (AFIDs, PDDCA, VerSe, PI-CAI, MAMA-MIA, DEEP-PSMA, LNQ2023, LIDC-IDRI; catalogue 818 -> 920 shipped configs (the release note's 820 -> 950 is the pre-recut count)); per-dataset ceiling resolution; per-dataset `MedVision_ACK_RELEASE`; cache keyed on the *resolved* version and the data root. |
| `1.2.1` | MAMA-MIA and PI-CAI annotations corrected to RAS+; their `1.2.0` annotations are **withdrawn** (a `1.2.0` pin errors for them). |
| `1.3.0` | adds MSWAL (484 abdominal CT cases, 42 configs). |
| `1.4.0` | regenerates T/L annotations of all 12 tumour/lesion datasets: mm size floor `max(2.0 mm, 2 x coarser in-plane spacing)` replaces the pixel-count floor, containment gate removed, four ellipse-fit guards; ~20x more T/L samples; **train/test split moved on six datasets** (HNTSMRG24, KiPA22, KiTS23, MSD, autoPET-III, BraTS24); QC figures moved to separate `_fig.zip` archives. |

Invariants worth stating to users:

- Only **Tumor-Lesion-Size** annotations differ between versions. Detection plans have never been regenerated; A/D plans are `1.0.0` only (Ceph-Biometrics-400, FeTA24). A release changes a dataset only if it republished that dataset's plan.
- Biometry families: `landmark` = AFIDs, Ceph-Biometrics-400, FeTA24, PDDCA, VerSe; `fromSeg` (T/L) = autoPET-III, BraTS24, DEEP-PSMA, HNTSMRG24, KiPA22, KiTS23, LIDC-IDRI, LNQ2023, MAMA-MIA, MSD, MSWAL, PI-CAI.
- **Paused** (refused at every pin, even from a warm cache): AFIDs, PDDCA, VerSe biometry `1.2.0`. Hence A/D covers only Ceph-Biometrics-400 and FeTA24 in every `all_tasks__ds_v*` catalogue.
- **Withdrawn**: MAMA-MIA and PI-CAI `1.2.0`; use `1.2.1` or later.
- Pinning below a dataset's newest annotation requires `MedVision_ACK_RELEASE` set to either that dataset's newest version (shown in the error) or the release version `1.4.0` (blanket, the only value that works for a catalogue sweep).
- The HF builder cache id is suffixed with `planner_version=<resolved version>-<8-hex hash of the absolute data root>`, so switching pins or data roots builds separate Arrow caches instead of colliding (`NonMatchingSplitsSizesError` was the pre-1.1.1 symptom).
- New studies: `latest`. Reproducing published numbers: `1.0.0` + ACK, and never compare a v1.4.0 test-split metric with a pre-1.4.0 one on the six re-split datasets.
