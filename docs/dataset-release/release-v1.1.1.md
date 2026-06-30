# Release v1.1.1

**v1.1.1 is a Tumor-Lesion-Size (TL) biometry bugfix.** If you use the TL task on volumes with **anisotropic in-plane pixel spacing** (most commonly sagittal/coronal reslices), your accepted samples and reported axis lengths change — across the six TL datasets the accepted sample set drops **~22%** (range −13.7% to −43.1%), with axis-length corrections up to **~72 mm**. **Isotropic in-plane data (e.g. KiPA22) is essentially unchanged** (sub-millimetre only). All non-TL tasks are untouched. To use the corrected annotations:

```bash
export MedVision_PLANNER_VERSION=1.1.1   # or 'latest', which now resolves to 1.1.1
```

## Changes since v1.1.0

Essential commits from the v1.1.0 release to v1.1.1:

- **TL ellipse-fit bugfix + corrected annotations** (`ac72b3e`, `7414b3c`) — the transposed-spacing (A1) and continuous-axis major/minor (A2) fixes, regenerated v1.1.1 annotations for all 6 TL datasets, with the train/test split re-aligned to v1.0.0. Detailed below.
- **Non-TL fallback to v1.0.0** (`01e5122`) — tasks other than Tumor-Lesion-Size load the v1.0.0 plan when a newer one is absent, so a version bump never breaks them.
- **Planner correctness + split reproducibility** (`dbb2028`) — fixes to planner template calls, mask dtype, converter failures, and deterministic train/test splitting.
- **Per-version builder-cache isolation** (`0092594`, `0c4028c`) — `MedVision_PLANNER_VERSION` enters the HF builder-cache fingerprint (avoids `NonMatchingSplitsSizesError` on a version switch); raw data is not redownloaded when the cached version already satisfies the request.
- **License corrected to CC-BY-4.0** (`64c2631`) — previously CC-BY-NC-4.0.
- **Ceph landmark mirror off-by-one** (`5c6abef`) — minor; no effect on released annotations.

## Summary

- **~22% fewer TL samples on anisotropic data, with corrected axis lengths.** Across the 6 TL datasets, **18,041 of 71,884** v1.1.0 sample-slices are dropped and **14,880** mislabelled `major<minor` cases are eliminated; per-dataset change ranges −13.7% (BraTS24) to −43.1% (KiTS23). KiPA22 (isotropic) is unchanged. See [Per-dataset impact](#per-dataset-impact).
- **`L-1-2 ≥ L-3-4` is now guaranteed** — major/minor axes are labelled and measured from the continuous real-space ellipse.
- **Root cause: transposed in-plane voxel spacing** — the TL ellipse fit paired each contour coordinate with the *wrong* axis's spacing (an OpenCV-vs-NumPy axis-order mismatch). See [How it was fixed](#how-it-was-fixed).
- **Version selection:** `MedVision_PLANNER_VERSION` is still required; it now accepts `1.1.1`, and `latest` resolves to `1.1.1`. v1.0.0 / v1.1.0 remain selectable. See [Selecting the version](#selecting-the-annotation-version).
- **Scope: only the TL task changed** — every other task resolves to identical v1.0.0 annotations. See [Scope](#scope-and-backward-compatibility).

## Who is affected

- **Anisotropic in-plane slices** — slices whose two in-plane voxel spacings differ (notably sagittal/coronal reslices). These get corrected ellipse geometry and account for the dropped samples and the large length deltas.
- **Isotropic in-plane data** — essentially unchanged. KiPA22 keeps every sample (0 removed/added, 0 cluster mismatches); only sub-millimetre corrections appear (worst 0.78 mm).

## Per-dataset impact

v1.1.0 → v1.1.1, computed from the released plans (reproducible via `scripts/compare_planners_v1.1.0_to_v1.1.1.py`):

| Dataset | v1.1.0 | v1.1.1 | Δ% | Removed | Added | Changed clusters | Worst Δ (mm) | ≥1 mm | major<minor fixed | Cluster mismatches |
|---------|-------:|-------:|----:|--------:|------:|-----------------:|-------------:|-----:|------------------:|-------------------:|
| KiPA22 | 3142 | 3142 | 0.0% | 0 | 0 | 3142 | 0.7761 | 0 | 5 | 0 |
| HNTSMRG24 | 4422 | 3161 | −28.5% | 1508 | 247 | 2923 | 68.1654 | 1374 | 2688 | 18 |
| autoPET-III | 3773 | 3025 | −19.8% | 1312 | 564 | 2468 | 72.5825 | 1555 | 1121 | 34 |
| KiTS23 | 14082 | 8015 | −43.1% | 6700 | 633 | 7355 | 66.2202 | 2755 | 4996 | 44 |
| MSD | 16232 | 12604 | −22.4% | 3997 | 369 | 12442 | 56.9634 | 1682 | 2734 | 79 |
| BraTS24 | 30233 | 26086 | −13.7% | 4524 | 377 | 25796 | 26.6941 | 1615 | 3336 | 10 |
| **Total** | **71884** | **56033** | **−22.1%** | **18041** | **2190** | **54126** | **72.58 (max)** | **8981** | **14880** | **185** |

A "sample" is one 2D slice entry; "Removed/Added" are slices present in only one version. "major<minor fixed" counts all v1.1.0 clusters with `L-1-2 < L-3-4` (eliminated in v1.1.1). "Cluster mismatches" are shared slices whose valid-cluster count differs.

## Selecting the annotation version

`MedVision_PLANNER_VERSION` is **required** — loading raises `EnvironmentError` if unset (no silent default). Set one of:

```bash
export MedVision_PLANNER_VERSION=latest   # newest annotations — now v1.1.1
export MedVision_PLANNER_VERSION=1.1.1    # corrected TL ellipse fit
export MedVision_PLANNER_VERSION=1.1.0    # previous TL annotations
export MedVision_PLANNER_VERSION=1.0.0    # original annotations
```

| `MedVision_PLANNER_VERSION` | Benchmark plan loaded |
|-----------------------------|-----------------------|
| `1.1.1` / `latest`          | `benchmark_plan_*_v1.1.1.json.gz` |
| `1.1.0`                     | `benchmark_plan_*_v1.1.0.json.gz` |
| `1.0.0`                     | `benchmark_plan_*_v1.0.0.json.gz` |
| not set                     | `EnvironmentError` |

When unset, the error leads with: *"any slice with anisotropic in-plane pixel size (e.g. sagittal or coronal reslices) should use the v1.1.1+ annotation."* Upgrading the `medvision_ds` package never silently changes annotations — the version is always chosen here.

### Using an older version requires acknowledgement

To stop outdated annotations from being used unknowingly, selecting any version **older than the latest** — for **any task** — is a hard error unless you explicitly acknowledge it:

```bash
export MedVision_PLANNER_VERSION=1.1.0      # an older annotation version
export MedVision_ACK_RELEASE=1.1.1          # confirms you have read this note
```

The acknowledgement value is the **latest version** (`1.1.1`). Every release bumps the dataset version, so old acknowledgements stop working and you are prompted to read the new release note; `latest` / `1.1.1` (and newer) never require it. Pinning an older version is a valid, supported choice — the **substantive** data difference in this release is confined to the Tumor-Lesion-Size task (every other task resolves to identical v1.0.0 annotations), so for those tasks the acknowledgement is purely a "you have seen the release" confirmation.

## Scope and backward compatibility

**Only the `Tumor-Lesion-Size` task changed.** For any other task, if a `1.1.x` plan is absent the loader transparently falls back to the v1.0.0 file — so `Mask-Size`, `Box-Size`, and `Biometrics-From-Landmarks`/`-Distance`/`-Angle` resolve to **identical v1.0.0 annotations regardless of the version selected**. v1.0.0 and v1.1.0 plans are untouched and remain selectable for exact reproducibility (`MedVision_PLANNER_VERSION=1.0.0` reproduces pre-fix results).

All three TL releases also share v1.0.0's exact image-level train/test split (see [Split alignment](#split-alignment-to-v100)); annotation values are byte-identical across that realignment — only the partition changed.

## How it was fixed

All versions fit the TL ellipse in **real (mm) space** — v1.1.1 is *not* a pixel-space-vs-real-space change. It corrects two things in `__fit_ellipses` (`src/medvision_ds/utils/benchmark_planner.py`).

### Transposed in-plane spacing (root cause)

On slices with unequal in-plane spacing, the fitted ellipse — both its lengths and its major/minor labels — was distorted because the two pixel spacings were swapped. Isotropic slices were unaffected.

*Details.* OpenCV returns contour points as `(x=column=dim1, y=row=dim0)`, but the spacing array is `(spacing_dim0, spacing_dim1)`, so the old conversion multiplied each coordinate by the *other* axis's spacing:

```python
# v1.0.0 / v1.1.0 — column scaled by row-spacing and vice versa
contour_real = contours[0].squeeze() * pixel_sizes
```

v1.1.1 pairs each coordinate with its own axis spacing (and applies the same per-coordinate map to the center and the four endpoints):

```python
pixel_size_x = pixel_sizes[1]   # x = column = dim1
pixel_size_y = pixel_sizes[0]   # y = row    = dim0
contour_real = np.column_stack(
    (contour[:, 0] * pixel_size_x, contour[:, 1] * pixel_size_y)
)
```

### Major/minor from the continuous axes

On near-circular lesions the labelled major axis (`L-1-2`) could report shorter than the minor (`L-3-4`); v1.1.1 guarantees `L-1-2 ≥ L-3-4`.

*Details.* The labels and reported lengths now come from the continuous real-space ellipse axes and are stored as `measurements`, instead of being re-derived from the rounded integer landmark indices:

```python
major_axis_mm = max(axes_real[0], axes_real[1])
minor_axis_mm = min(axes_real[0], axes_real[1])
landmark_dict["measurements"] = {"L-1-2": float(major_axis_mm), "L-3-4": float(minor_axis_mm)}
```

`_cal_distance` returns the stored `measurements` value when present; non-ellipse tasks (e.g. Ceph) carry no `measurements` and are unchanged.

### Why the sample set shrank — the filter did *not* change

The acceptance filter was **not** tightened. v1.1.1 simply accepts fewer samples because the corrected ellipses fail the *same, unchanged* filter more often on anisotropic slices.

*Details.* The per-cluster `all_within` gate (the four axis endpoints must lie in the ring between the 0.9× shrunk and 1.1× enlarged lesion bounding box) is byte-identical to v1.1.0. Because it is evaluated per cluster, on multi-cluster anisotropic slices the corrected endpoints also change which clusters pass — the source of the "Cluster mismatches" column.

## For maintainers

### Version bump

`src/medvision_ds/__version__.py` → `1.1.1`; `MedVision.py` config default `version="1.1.1"` (what `latest` resolves to), folded into the HF builder cache fingerprint via `create_config_id` so switching versions uses a distinct cache folder.

### Regenerating v1.1.1

`scripts/regenerate_tl_annotation_v1.1.1.py` regenerates the corrected TL annotations, leaving v1.0.0/v1.1.0 untouched (versioned folder names). It pins the train/test split to the released v1.1.0 membership (`split_override`) so v1.1.1 is a pure ellipse-value correction, emits lineage + multi-cluster sidecars in the released `*_v1.0.0_to_v1.1.0.json` structure, and relativizes the emitted plan's paths to the released form. Run once per dataset (`--dataset_name` is the dataset's own name: BraTS24, autoPET-III, KiTS23, KiPA22, HNTSMRG24, MSD), then merge each `{dataset}_regen/` into the published `{dataset}/`:

```bash
python scripts/regenerate_tl_annotation_v1.1.1.py \
    --dataset_name BraTS24 \
    --data_dir   /path/to/MedVision/Data/Datasets \
    --output_dir /path/to/MedVision/Data/Datasets
```

### Split alignment to v1.0.0

`scripts/align_tl_split_to_v1.0.0.py` re-aligns the released v1.1.0 and v1.1.1 plans to v1.0.0's exact image-level split (v1.1.0 had reshuffled ~41% of cases after the post-v1.0.0 `sorted()`-glob change). It only relabels which `train_cases`/`test_cases` a case sits in — annotation values are byte-identical — backs up every touched file to `{dataset}/dev/bak/`, and verifies `v1.0.0 == v1.1.0 == v1.1.1` membership and order across all six datasets.


