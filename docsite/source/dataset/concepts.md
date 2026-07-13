# Dataset concepts

MedVision is a benchmark built for *quantitative* medical image analysis: instead of asking a model to name a finding, it asks the model to measure one, and it scores that measurement against a physically grounded ground truth. This page explains the ideas you need before you load a single sample — where the data comes from, how subsets are named, what the annotations mean, and why every target is expressed in real-world units.

![How MedVision turns 3D volumes and headers into 2D slices with physical-unit annotations](../_static/medvision-dataset-flow.svg)

## What MedVision holds

At the data level, MedVision consolidates **22 public medical imaging datasets** — including collections such as BraTS24, MSD, and OAIZIB-CM — into a single, uniformly structured resource of roughly **30.8 million image–annotation pairs**. The imaging spans five modalities: X-ray (XR), CT, MRI, ultrasound (US), and PET, across many anatomical regions.

Source images are kept as **3D volumes reoriented to RAS+** (a canonical right-anterior-superior axis convention), which makes plane definitions consistent across datasets that were originally stored with different orientations. Because most vision-language models consume 2D images, MedVision does not ship pre-cut slices. Instead the loader **slices volumes to 2D on the fly** along any of the three anatomical planes — axial, coronal, or sagittal — at load time. This keeps the on-disk footprint tied to the volumes themselves (the full dataset is around 1 TB) rather than to an exploded set of PNGs.

:::{note}
MedVision distributes only the annotations. The Hugging Face loader script fetches and preprocesses the raw imaging for you into `MedVision_DATA_DIR`. The end-to-end mechanics are covered in [Loading data](loading.md).
:::

## Datasets vs. data configs

Two vocabulary terms do a lot of work throughout the codebase:

- A **dataset** is one of the 22 upstream sources, referenced by its short name (`BraTS24`, `MSD`, `OAIZIB-CM`, …).
- A **data config** is a named, ready-to-load subset of MedVision. You pass a config name to select exactly which slices and annotations you get.

Config names follow a fixed five-part convention:

```text
{dataset}_{annotation-type}_{task-ID}_{slice}_{split}
```

| Field | Values | Meaning |
| --- | --- | --- |
| `dataset` | e.g. `OAIZIB-CM`, `BraTS24` | which upstream source |
| `annotation-type` | `BoxSize`, `TumorLesionSize`, `BiometricsFromLandmarks`, `MaskSize` | what kind of target (see below) |
| `task-ID` | `Task01`, `Task02`, … | a **local** task index within that dataset, not a global MedVision ID |
| `slice` | `Axial`, `Coronal`, `Sagittal` | slicing plane |
| `split` | `Train`, `Test` | subject-level split |

A couple of concrete config names:

```text
OAIZIB-CM_BoxSize_Task01_Axial_Test
BraTS24_TumorLesionSize_Task01_Axial_Train
```

The `task-ID` is per-dataset because a single source can define several image–mask targets; those tasks are declared in the dataset-construction code (`medvision_ds/datasets/<dataset>/preprocess_*.py`), so `Task01` for one dataset is unrelated to `Task01` for another.

## The four annotation types

The `annotation-type` field selects what the model is asked to produce and, correspondingly, which fields each returned sample carries:

- **`BoxSize`** — bounding-box detection. Each sample lists boxes with pixel-space `min_coords` / `max_coords` / `center_coords` / `dimensions`, plus per-box physical `sizes`. This is the annotation behind the Detection task (metrics: IoU, Precision, Recall, F1, SuccessRate).
- **`TumorLesionSize`** — the physical extent of a tumour or lesion, reported as major- and minor-axis measurements **in millimetres**. This backs the Tumour/Lesion size task (metrics: MAE, MRE, nMAE, SuccessRate).
- **`BiometricsFromLandmarks`** — clinical biometrics computed from anatomical landmarks: **angles in degrees and distances in millimetres**. This backs the Angle/Distance task (metrics: MAE, MRE), with a `biometric_profile` carrying the metric type, value, and unit.
- **`MaskSize`** — segmentation-mask area, exposed via a `ROI_area` field alongside pixel and voxel geometry.

Every sample, regardless of type, also carries the geometry needed to interpret it: `image_size_2d`, `pixel_size` (per-axis, 2D), `image_size_3d`, `voxel_size` (per-axis, 3D), and the slice locator (`slice_dim`, `slice_idx`).

## Multi-instance vs single-instance annotations

A benchmark sample is a *(2D slice, target)* pair, counted **per target, not per instance**: several boxes or clusters of the same target on one slice still count as a single annotation. What differs between the two annotation sets is whether the per-sample quality/size filters are applied:

- **Multi-instance** (unfiltered) — every target carrying ≥ 1 annotation is kept, however many instances it has on the slice and whatever their size.
- **Single-instance** (filtered) — a target is kept only when it is a single, large-enough instance. The per-task drop rules are:

| Benchmark task | Single-instance drops the sample when… |
|---|---|
| **Box** — detection | the slice has **more than one** box for the target (`len(boxes) > 1`), **or** a box is **< 10 px** on any side |
| **T/L** — tumor / lesion size | the target has **more than one** cluster on the slice (`n_clusters > 1`; `len(biometric_profile) > 1` on the v1.0.0 fallback) |
| **A/D** — biometrics (angle / distance) | *never dropped* — every angle and distance sample is kept (the loader only splits them by `metric_type`) |

Because filtering only ever removes samples, every single-instance target is also a multi-instance one: **single-instance ⊆ multi-instance**. The default loader returns the single-instance set; to load the unfiltered set, see [Loading unfiltered (multi-instance) samples](loading.md#loading-unfiltered-multi-instance-samples). Per-version counts for both sets are tabulated in [Dataset versions & statistics](statistics.md#benchmark-annotations-by-version).

:::{warning}
Single-instance (filtered) is the set to use for leaderboard comparison. The multi-instance set is not — MedVision-V0's SFT/RFT training is not optimized for multi-instance detection and measurement.
:::

## Why the targets are physical

The defining property of MedVision is that ground-truth targets are **real-world physical quantities** — millimetres and degrees — not pixel counts. They are derived from the voxel spacing stored in each image's header: a bounding box measured as 40 pixels wide means something only once you multiply by the millimetres-per-pixel of that particular scan. Because the annotations bake in this spacing, a size or distance target is comparable across scanners, resolutions, and datasets, which is exactly what makes the benchmark *quantitative* rather than categorical.

:::{warning}
**Once an image is resized, its pixel size must be updated to match.** The pixel-to-millimetre mapping stated in the prompt is only correct at the resolution the model actually sees, so whenever a model's preprocessing resizes an input, the pixel size has to be rescaled by the same factor — that way `image size × pixel size` still equals the true physical extent, and the model can do the pixel→mm arithmetic itself. A non-square resize scales height and width by different factors, so the update is applied **per axis**. Getting this right is essential for fair scoring; the per-model details are covered when you [add a model](../extending/add-a-model.md).
:::

## Where to go next

- [Loading data](loading.md) — install the loader, set `MedVision_DATA_DIR` and the version env vars, and pull a config with `load_dataset()`.
- [Add a model](../extending/add-a-model.md) — how the pixel-size recomputation is wired into a model's image-processing path.
