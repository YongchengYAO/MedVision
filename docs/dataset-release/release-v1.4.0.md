# Release v1.4.0

**v1.4.0 regenerates the Tumor-Lesion-Size (TL) annotations of all 12 tumour/lesion datasets.** Clusters are now selected by a physical size floor in millimetres instead of a raw pixel count, a gate that discarded rotated ellipses is removed, and the ellipse fit itself is guarded against degenerate results. Published TL landmarks grow from **75,840 to 3,801,540 (50x)**. Everything else — every other task, and every previously published annotation file — is unchanged.

```bash
export MedVision_PLANNER_VERSION=1.4.0   # or 'latest', which now resolves to 1.4.0
```

## Summary

| Change | Why | Action for users |
| --- | --- | --- |
| Millimetre size floor replaces pixel-count floor | a pixel count is not a physical size across planes | expect far more samples |
| `all_within` containment gate removed | it silently discarded rotated ellipses | — |
| Four guards on the ellipse fit | degenerate fits produced impossible measurements | — |
| Natural train/test split on six datasets | their old splits were force-aligned to v1.0.0 | re-derive any cached split |
| `latest` resolves to 1.4.0 | — | older pins need `MedVision_ACK_RELEASE=1.4.0` |

**Scope.** The 12 `fromSeg` datasets: autoPET-III, BraTS24, DEEP-PSMA, HNTSMRG24, KiPA22, KiTS23, LIDC-IDRI, LNQ2023, MAMA-MIA, MSD, MSWAL, PI-CAI (38 biometry tasks, 216 TL configs). The five landmark-family biometry datasets and all segmentation/detection plans are untouched. The release is **additive**: every prior plan and `Landmarks-*` folder stays published, and existing pins resolve to the same bytes as before.

## What changed

**1. A cluster is measured when its fitted major axis clears a physical floor:**

```
max(2.0 mm, 2 x coarser in-plane spacing of the plane being measured)
```

The old rule was a raw pixel count (20 px; 10 px for LIDC-IDRI). A pixel count is not a physical size: sagittal and coronal slices are reconstructed across the slice axis, so on a 0.977/0.977/3.0 mm case a 20 px floor cut at ~4.9 mm axially but ~8.6 mm sagittally — a 3.07x area difference behind one number. The new floor is a *resolution* floor, not a clinical one: per-lesion thresholds like RECIST 10 mm are left to downstream consumers, and the floor still differs across planes because plane resolution does.

**2. The `all_within` gate is removed.** It required the fitted ellipse to sit inside a box around the cluster, which is a shape/rotation test, not a size test — a tilted ellipse protrudes from an axis-aligned box no matter how well it fits (a 6.7 cc liver lesion at 131 degrees produced zero landmarks). This removal, not the lower floor, drives most of the growth: on KiPA22 the median size barely moved (34.3 -> 33.5 mm) while the maximum *grew* (85.3 -> 106.8 mm) — a lowered floor cannot produce larger lesions.

**3. Four guards bracket `cv2.fitEllipse`.** Without the pixel-count pre-filter, the fit now sees tiny and degenerate clusters. cv2 solves a least-squares conic in float32, which on such contours goes near-singular and fails in both directions — one axis collapses toward zero, or the other explodes (one 5 px cluster measured 152 km). The guards:

| Guard | Rejects |
| --- | --- |
| contour under 5 points | what cv2 itself refuses by raising an error |
| non-finite conic | unbounded fits (previously an `OverflowError` crash) |
| minor axis under one voxel | collapsed fits — nothing is thinner than the voxel it is sampled on |
| major axis over 1.5x the cluster's own bbox diagonal | runaway fits; 1.5x calibrated on 536 synthetic lesion shapes (max 1.289x) |

The logged guards rejected **95,800 fits** across the corpus (11,035 non-finite + 84,765 runaway; the two silent guards are on top of that). Do not raise the contour floor above 5: a point count is not a conditioning test — a 5 px *line* returns 8 contour points and still degenerates, while some compact 5 px blocks trace exactly 5 points and are removed for no benefit. The sub-voxel and bbox-diagonal guards are what actually catch degeneracy, and they catch opposite directions of it.

## Per-dataset impact

Landmarks across all three planes, previous published version -> v1.4.0:

| Dataset | From | Landmarks | Factor |
| --- | --- | ---: | ---: |
| DEEP-PSMA | 1.2.0 | 753 -> 156,439 | 208x |
| LNQ2023 | 1.2.0 | 238 -> 37,898 | 159x |
| autoPET-III | 1.1.1 | 3,118 -> 488,375 | 157x |
| LIDC-IDRI | 1.2.0 | 515 -> 59,179 | 115x |
| MAMA-MIA | 1.2.1 | 5,071 -> 369,419 | 73x |
| PI-CAI | 1.2.1 | 409 -> 28,324 | 69x |
| BraTS24 | 1.1.1 | 26,198 -> 1,645,550 | 63x |
| MSD | 1.1.1 | 12,914 -> 712,247 | 55x |
| HNTSMRG24 | 1.1.1 | 3,188 -> 69,093 | 22x |
| MSWAL | 1.3.0 | 12,260 -> 145,136 | 12x |
| KiTS23 | 1.1.1 | 8,034 -> 76,016 | 9.5x |
| KiPA22 | 1.1.1 | 3,142 -> 13,864 | 4.4x |
| **Total** | | **75,840 -> 3,801,540** | **50x** |

The ordering is itself evidence: the biggest gains are on the datasets with the smallest, most irregular lesions — exactly where a pixel-count floor and a containment test bite hardest. The loader still discards multi-cluster slices, so published sample counts grow by less than the raw counts above.

## Annotation statistics and recall

`scripts/tl-stats/` derives the corpus statistics for this release; the full report is `doc/tl-annotation-stats-v1.4.0.md`, and the tables, per-cluster parquets and figures ship in `scripts/tl-stats/tl-stats-v1.4.0/`.

**Recall.** The segmentation masks contain **6,951,667** connected components across the 12 datasets; **3,801,540 (0.547)** carry a measurement. The denominator is a direct per-slice recount of the masks with no size filter — not the biometry plan's own `n_total_clusters`, which only exists on slices that yielded a landmark and therefore overestimates recall (0.571 corpus-wide, up to 0.848 vs 0.538 on PI-CAI). The recount was verified against the plans on 2,018,996 shared slices with **zero** disagreements. Recall by cluster size in the slice, attributed exactly per cluster via the landmark files' `ROI_pixels_count`:

| Cluster size | Clusters | Measured | Recall |
| --- | ---: | ---: | ---: |
| 1–2 px | 1,872,149 | 0 | 0.000 |
| 3–5 px | 831,751 | 34,347 | 0.041 |
| 6–10 px | 665,002 | 324,716 | 0.488 |
| 11–20 px | 649,587 | 551,337 | 0.849 |
| 21–50 px | 756,110 | 723,779 | 0.957 |
| 51–100 px | 490,959 | 485,418 | 0.989 |
| 101–500 px | 1,003,733 | 999,961 | 0.996 |
| 501–1000 px | 346,583 | 346,242 | 0.999 |
| >1000 px | 335,793 | 335,740 | 1.000 |

The missing 45% is speckle, not lesions: 95.5% of unmeasured clusters are 10 px or smaller, and a quarter of all mask components are 1–2 px specks that cannot yield a contour at all. Above 21 px recall is ≥0.96, above 101 px ≥0.996 — the floor and the fit guards remove sub-resolution fragments, so read 0.547 as a fraction of *components*, not of findable lesions.

**Other summary metrics.** Per dataset, recall spans **0.996 (KiPA22) to 0.336 (MSD)** — MSD's floor is its BrainTumour cohort, whose isotropic masks are speckle-dominated. On thick-slice datasets the axial plane is near-complete while the reconstructed planes are not (PI-CAI 0.999 vs 0.516): reformatting cuts thin slivers that land in the collapsing size bins — the same anisotropy the millimetre floor addresses, now visible as geometry rather than threshold. After the loader's multi-cluster drop, the **published yield** is 0.139 of mask components corpus-wide (0.939 KiPA22, 0.044 DEEP-PSMA). The **median major axis** per label spans 7.7 mm (BraTS24-GLI non-enhancing core) to 43.3 mm (KiTS23 kidney tumour); measurements run from the 2.0 mm floor to 540.1 mm, and the extreme tail is real — the 540 mm fit is 0.93x its own component's bbox diagonal on a whole-body PET coronal reformat, with only 291 of 3.8M measurements above 200 mm. Across all 3.8M measurements, **zero** rows label the major axis shorter than the minor.

## Train/test split

Case counts per split are unchanged, but **which** cases land on each side changes for six datasets: HNTSMRG24 (47%), KiPA22 (43%), KiTS23 (43%), MSD (42%), autoPET-III (41%), BraTS24 (41%). These are exactly the datasets whose earlier splits were force-aligned to v1.0.0; v1.4.0 is their first natural seeded split (seed 1024, ratio 0.7). The other six datasets move 0%. Do not compare a v1.4.0 test-set metric against a pre-1.4.0 one on the six affected datasets.

## Selecting the annotation version

```bash
export MedVision_PLANNER_VERSION=1.4.0    # or 'latest'
```

To stay on an older annotation, acknowledge this release:

```bash
export MedVision_PLANNER_VERSION=1.0.0
export MedVision_ACK_RELEASE=1.4.0
```

Only the 12 TL datasets prompt; Angle-Distance and detection workloads need no ACK.

## Reproducing

```bash
python scripts/gen-annotations/build_dataset.py \
  --data_dir <DATA> --dataset <NAME> --steps biometry \
  --new-annotation-version --cluster_filter_method major_axis
```

v1.4.0 was produced with **OpenCV 4.12.0** and **numpy 2.2.6**. `cv2.fitEllipse` *is* the measurement, so a different OpenCV may return marginally different values. The legacy rule remains available as `--cluster_filter_method pixel_count`.
