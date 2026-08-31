# Release v1.4.0

**Last edit:** 2026-08-29

**v1.4.0 regenerates the Tumor-Lesion-Size (TL) annotations of all 12 tumour/lesion datasets.** Clusters are selected by a physical size floor in millimetres rather than by a raw pixel count, the containment gate that discarded rotated ellipses is removed, and the ellipse fit is guarded against degenerate results. Published TL landmarks increase from **75,840 to 3,801,540 (50x)**. All other tasks and all previously published annotation files are unchanged.

```bash
export MedVision_PLANNER_VERSION=1.4.0   # or 'latest', which now resolves to 1.4.0
```

## Summary

| Change | Rationale | Action for users |
| --- | --- | --- |
| Millimetre size floor replaces pixel-count floor | a pixel count is not a physical size across planes | expect substantially more samples |
| `all_within` containment gate removed | it discarded rotated ellipses | none |
| Four guards added to the ellipse fit | degenerate fits produced impossible measurements | none |
| Natural train/test split on six datasets | their previous splits were force-aligned to v1.0.0 | re-derive any cached split |
| `latest` resolves to 1.4.0 | new default annotation version | older pins require `MedVision_ACK_RELEASE=1.4.0` |

**Scope.** The 12 `fromSeg` datasets (autoPET-III, BraTS24, DEEP-PSMA, HNTSMRG24, KiPA22, KiTS23, LIDC-IDRI, LNQ2023, MAMA-MIA, MSD, MSWAL, PI-CAI), comprising 38 biometry tasks and 228 TL configs (38 tasks x 3 planes x 2 splits). The five landmark-family biometry datasets and all segmentation and detection plans are unchanged. The release is **additive**: every prior plan and `Landmarks-*` folder remains published, and existing pins resolve to identical bytes.

## Annotation procedure changes

**1. Physical size floor.** A cluster is measured when its fitted major axis clears

```
max(2.0 mm, 2 x coarser in-plane spacing of the plane being measured)
```

The previous rule was a raw pixel count (20 px; 10 px for LIDC-IDRI). A pixel count is not a physical size: sagittal and coronal slices are reconstructed across the slice axis, so on a case of 0.977/0.977/3.0 mm spacing a 20 px floor cut at ~4.9 mm axially but ~8.6 mm sagittally, a 3.07x difference in area behind a single number. The new floor is a *resolution* floor, not a clinical one; per-lesion thresholds such as RECIST 10 mm are left to downstream consumers, and the floor still varies across planes because plane resolution does.

**2. Removal of the `all_within` gate.** The gate required the fitted ellipse to lie inside a box around the cluster, which tests shape and rotation rather than size: a tilted ellipse protrudes from an axis-aligned box regardless of fit quality, and a 6.7 cc liver lesion at 131 degrees yielded zero landmarks. This removal, not the lower floor, accounts for most of the growth: on KiPA22 the median size moved only from 34.3 mm to 33.5 mm while the maximum *increased* from 85.3 mm to 106.8 mm, which a lowered floor cannot do.

**3. Four guards on `cv2.fitEllipse`.** Without the pixel-count pre-filter, the fit receives tiny and degenerate clusters. cv2 solves a least-squares conic in float32, which becomes near-singular on such contours and fails in both directions: one axis collapses toward zero, or the other diverges (one 5 px cluster measured 152 km).

| Guard | Rejects |
| --- | --- |
| contour under 5 points | inputs that cv2 itself refuses by raising an error |
| non-finite conic | unbounded fits (previously an `OverflowError` crash) |
| minor axis under one voxel | collapsed fits; nothing is thinner than the voxel it is sampled on |
| major axis over 1.5x the cluster bbox diagonal | runaway fits; 1.5x calibrated on 536 synthetic lesion shapes (maximum 1.289x) |

The logged guards rejected **95,800 fits** corpus-wide (11,035 non-finite, 84,765 runaway), in addition to the two silent guards. The contour floor should not be raised above 5 points, because a point count is not a conditioning test: a 5 px *line* returns 8 contour points and still degenerates, whereas some compact 5 px blocks trace exactly 5 points and would be removed without benefit. Degeneracy is caught by the sub-voxel and bbox-diagonal guards, which act in opposite directions.

## Per-dataset impact

Landmarks across all three planes.

| Dataset | Previous version | Previous | v1.4.0 | Factor |
| --- | --- | ---: | ---: | ---: |
| DEEP-PSMA | 1.2.0 | 753 | 156,439 | 208x |
| LNQ2023 | 1.2.0 | 238 | 37,898 | 159x |
| autoPET-III | 1.1.1 | 3,118 | 488,375 | 157x |
| LIDC-IDRI | 1.2.0 | 515 | 59,179 | 115x |
| MAMA-MIA | 1.2.1 | 5,071 | 369,419 | 73x |
| PI-CAI | 1.2.1 | 409 | 28,324 | 69x |
| BraTS24 | 1.1.1 | 26,198 | 1,645,550 | 63x |
| MSD | 1.1.1 | 12,914 | 712,247 | 55x |
| HNTSMRG24 | 1.1.1 | 3,188 | 69,093 | 22x |
| MSWAL | 1.3.0 | 12,260 | 145,136 | 12x |
| KiTS23 | 1.1.1 | 8,034 | 76,016 | 9.5x |
| KiPA22 | 1.1.1 | 3,142 | 13,864 | 4.4x |
| **Total** | | **75,840** | **3,801,540** | **50x** |

The largest gains occur on the datasets with the smallest and most irregular lesions, where a pixel-count floor and a containment test are most restrictive. By default the loader publishes only single-cluster slices — a Tumor-Lesion-Size sample is skipped when its slice's mask carries more than one cluster (`n_total_clusters > 1`) — so published sample counts grow by less than the counts above. To load the multi-instance annotations, set `MedVision_DISABLE_SAMPLE_FILTERING=true`: the per-sample filters are bypassed, every slice in the plan is returned, and a sample's `biometric_profile` then carries one measurement block per measured cluster on its slice.

## Annotation statistics

`scripts/tl-stats/` derives the corpus statistics for this release. The full report is `doc/tl-annotation-stats-v1.4.0.md`; tables, per-cluster parquets and figures ship in `scripts/tl-stats/tl-stats-v1.4.0/`.

## Recall

A *cluster* is a connected component of the binarised target-label mask on one slice; recall is measured clusters divided by all clusters, reported for **all 12 datasets**.

The numerator is the count of published measurements. The denominator is the count of *all* clusters, obtained by re-opening every published NIfTI mask, binarising it, slicing it along each plane, and counting connected components. How each quantity is computed — the mask recount, the per-cluster attribution, and the three size stratifications — is documented in `scripts/tl-stats/README.md`.

| Dataset | Clusters | Measured | Recall | axial | coronal+sagittal |
| --- | ---: | ---: | ---: | ---: | ---: |
| KiPA22 | 13,913 | 13,864 | 0.996 | 0.998 | 0.995 |
| MSWAL | 168,281 | 145,136 | 0.862 | 0.967 | 0.833 |
| KiTS23 | 92,403 | 76,016 | 0.823 | 0.938 | 0.803 |
| HNTSMRG24 | 85,017 | 69,093 | 0.813 | 0.991 | 0.778 |
| BraTS24 | 2,482,105 | 1,645,550 | 0.663 | **0.717** | **0.641** |
| DEEP-PSMA | 242,469 | 156,439 | **0.645** | 0.678 | 0.628 |
| LIDC-IDRI | 94,360 | 59,179 | 0.627 | **0.902** | **0.572** |
| autoPET-III | 801,169 | 488,375 | 0.610 | 0.627 | 0.600 |
| LNQ2023 | 67,336 | 37,898 | 0.563 | **0.980** | **0.512** |
| PI-CAI | 52,658 | 28,324 | 0.538 | **0.999** | **0.516** |
| MAMA-MIA | 731,369 | 369,419 | 0.505 | **0.529** | **0.498** |
| MSD | 2,120,587 | 712,247 | **0.336** | **0.313** | **0.346** |
| **All 12** | **6,951,667** | **3,801,540** | **0.547** | | |

### Recall by cluster size in pixels

Recall by cluster size, attributed exactly per cluster via the published landmark files (`unresolved_attribution_slices = 0`, and the diagnostic `ambiguous = 0` in every bin of every dataset; the attribution mechanism is documented in `scripts/tl-stats/README.md`). Each recall table reads the same counts three ways: the bin in isolation, the cumulative pool of clusters at or below a threshold (≤ t), and the complementary pool above it (> t); the two pools at one threshold partition the corpus.

| Cluster size | Clusters | Measured | Recall | Size ≤ t | Clusters | Measured | Recall | Size > t | Clusters | Measured | Recall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1–2 px | 1,872,149 | 0 | 0.000 | ≤2 px | 1,872,149 | 0 | 0.000 | >2 px | 5,079,518 | 3,801,540 | 0.748 |
| 3–5 px | 831,751 | 34,347 | 0.041 | **≤5 px** | **2,703,900** | **34,347** | 0.013 | >5 px | 4,247,767 | 3,767,193 | 0.887 |
| 6–10 px | 665,002 | 324,716 | 0.488 | **≤10 px** | **3,368,902** | **359,063** | 0.107 | **>10 px** | 3,582,765 | 3,442,477 | **0.961** |
| 11–20 px | 649,587 | 551,337 | 0.849 | ≤20 px | 4,018,489 | 910,400 | 0.227 | **>20 px** | 2,933,178 | 2,891,140 | **0.986** |
| 21–50 px | 756,110 | 723,779 | 0.957 | ≤50 px | 4,774,599 | 1,634,179 | 0.342 | >50 px | 2,177,068 | 2,167,361 | 0.996 |
| 51–100 px | 490,959 | 485,418 | 0.989 | ≤100 px | 5,265,558 | 2,119,597 | 0.403 | >100 px | 1,686,109 | 1,681,943 | 0.998 |
| 101–500 px | 1,003,733 | 999,961 | 0.996 | ≤500 px | 6,269,291 | 3,119,558 | 0.498 | >500 px | 682,376 | 681,982 | 0.999 |
| 501–1000 px | 346,583 | 346,242 | 0.999 | ≤1000 px | 6,615,874 | 3,465,800 | 0.524 | >1000 px | 335,793 | 335,740 | 1.000 |
| >1000 px | 335,793 | 335,740 | 1.000 | **All** | **6,951,667** | **3,801,540** | **0.547** |  |  |  |  |

#### Recall is 0.961 above 10 px and 0.986 above 20 px

Clusters of 1–2 px, over a quarter of the corpus, cannot yield a 5-point contour, so their recall is 0 by construction rather than by threshold. Pooled over every cluster above a threshold (> t block), **recall is 0.961 above 10 px and 0.986 above 20 px**: the floor and the fit guards remove sub-resolution fragments, not measurable lesions. The corpus value of 0.547 is thus a fraction of components, not of findable lesions.

#### 84.7% of unmeasured clusters are ≤5 px and 95.5% are ≤10 px

The 45% unmeasured clusters consist of lesions too small to be clinically measurable: **95.5% of the 3,150,127 unmeasured clusters are ≤10 px and 84.7% are ≤5 px**. Both shares come from the `≤ t` block of the table above, converted from cluster counts to *unmeasured* counts by subtracting the measured column from the clusters column in the same row:

```
unmeasured, all sizes = 6,951,667 - 3,801,540 = 3,150,127
unmeasured ≤10 px     = 3,368,902 -   359,063 = 3,009,839    ->  3,009,839 / 3,150,127 = 0.955
unmeasured ≤5 px      = 2,703,900 -    34,347 = 2,669,553    ->  2,669,553 / 3,150,127 = 0.847
```

### Recall by major axis length

`tl_recall_by_length.csv` restratifies the same clusters by physical extent instead of pixel count: per-cluster length is the **maximum Feret diameter of the cluster's pixel centres in millimetres**, which exists for every cluster including the unmeasured ones (definition, provenance and caveats in `scripts/tl-stats/README.md`). Its margins reproduce the corpus values exactly (6,951,667 clusters, 3,801,540 measured, recall 0.547).

| Feret length | Clusters | Measured | Recall | Length ≤ t | Clusters | Measured | Recall | Length > t | Clusters | Measured | Recall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| ≤2 mm | 2,232,427 | **1,952** | 0.001 | ≤2 mm | 2,232,427 | 1,952 | 0.001 | >2 mm | 4,719,240 | 3,799,588 | 0.805 |
| 2–5 mm | 974,881 | 409,412 | 0.420 | **≤5 mm** | **3,207,308** | **411,364** | 0.128 | >5 mm | 3,744,359 | 3,390,176 | 0.905 |
| 5–10 mm | 955,724 | 694,683 | 0.727 | **≤10 mm** | **4,163,032** | **1,106,047** | 0.266 | **>10 mm** | 2,788,635 | 2,695,493 | **0.967** |
| 10–20 mm | 1,055,235 | 976,105 | 0.925 | ≤20 mm | 5,218,267 | 2,082,152 | 0.399 | **>20 mm** | **1,733,400** | **1,719,388** | **0.992** |
| 20–50 mm | 1,263,241 | 1,250,268 | 0.990 | ≤50 mm | 6,481,508 | 3,332,420 | 0.514 | >50 mm | 470,159 | 469,120 | 0.998 |
| 50–100 mm | 413,816 | 412,849 | 0.998 | ≤100 mm | 6,895,324 | 3,745,269 | 0.543 | >100 mm | 56,343 | 56,271 | 0.999 |
| >100 mm | 56,343 | 56,271 | 0.999 | **All** | **6,951,667** | **3,801,540** | **0.547** |  |  |  |  |

#### Recall is 0.967 above 10 mm and 0.992 above 20 mm

This is the clinical-units counterpart of the pixel table and the direct check of the millimetre floor. The `> t` block is the one a user of the dataset sees, because selecting lesions above a size floor pools every cluster above it rather than reading a single bin: **recall is 0.967 for clusters longer than 10 mm and 0.992 for clusters longer than 20 mm.** Per dataset, above 10 mm none falls below 0.872 and above 20 mm none falls below 0.972:

| Dataset | Overall recall | >10 mm clusters | Measured | Recall | >20 mm clusters | Measured | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KiPA22 | 0.996 | 13,001 | 13,001 | 1.000 | 11,297 | 11,297 | 1.000 |
| LIDC-IDRI | 0.627 | 24,276 | 23,995 | 0.988 | 8,039 | 8,036 | 1.000 |
| HNTSMRG24 | 0.813 | 59,526 | 58,880 | 0.989 | 36,959 | 36,936 | 0.999 |
| MSWAL | 0.862 | 108,916 | 108,142 | 0.993 | 73,392 | 73,341 | 0.999 |
| MAMA-MIA | 0.505 | 236,773 | 233,701 | 0.987 | 155,306 | 154,901 | 0.997 |
| BraTS24 | 0.663 | 1,093,280 | 1,085,850 | 0.993 | 705,471 | 702,719 | 0.996 |
| DEEP-PSMA | **0.645** | 149,854 | 143,434 | 0.957 | 84,086 | 83,675 | **0.995** |
| LNQ2023 | 0.563 | 34,617 | 32,102 | 0.927 | 17,718 | 17,542 | 0.990 |
| KiTS23 | 0.823 | 78,164 | 74,352 | 0.951 | 66,580 | 65,835 | 0.989 |
| MSD | **0.336** | 469,198 | 452,981 | 0.965 | 307,790 | 303,468 | **0.986** |
| autoPET-III | 0.610 | 492,026 | 443,762 | 0.902 | 254,062 | 249,289 | 0.981 |
| PI-CAI | 0.538 | 29,004 | 25,293 | **0.872** | 12,700 | 12,349 | **0.972** |
| **All 12** | **0.547** | **2,788,635** | **2,695,493** | **0.967** | **1,733,400** | **1,719,388** | **0.992** |

**A low overall recall reflects a dataset's share of tiny fragments, not the quality of its annotations.** MSD is last overall at **0.336** yet reads **0.986** above 20 mm, and DEEP-PSMA climbs from **0.645** to **0.995**. What spreads the overall column is how much of each corpus consists of sub-resolution debris — a property of the source segmentations and of the voxel grid, not of this release's measurement rules. Above the size floors that spread nearly closes: every dataset reads at least 0.872 above 10 mm and at least 0.972 above 20 mm. Two practical consequences: a low overall recall is not evidence that a dataset is poorly annotated, and comparisons between datasets should be made on the >10 mm and >20 mm columns.

#### 88.8% of unmeasured clusters are ≤5 mm and 97.0% are ≤10 mm

The size profile of the unmeasured remainder says the same thing. Of the 3,150,127 unmeasured clusters, **88.8% are ≤5 mm, 97.0% are ≤10 mm, and 14,012 (0.44%) are longer than 20 mm**, each share obtained from the length table exactly as for the pixel table — clusters minus measured, in the row at that threshold, over the corpus-wide unmeasured total:

```
unmeasured, all lengths = 6,951,667 - 3,801,540 = 3,150,127
unmeasured ≤5 mm        = 3,207,308 -   411,364 = 2,795,944  ->  2,795,944 / 3,150,127 = 0.888
unmeasured ≤10 mm       = 4,163,032 - 1,106,047 = 3,056,985  ->  3,056,985 / 3,150,127 = 0.970
unmeasured >20 mm       = 1,733,400 - 1,719,388 =    14,012  ->     14,012 / 3,150,127 = 0.0044
```

The gradient between 2 and 20 mm reflects the resolution-adjusted floor rather than leakage: v1.4.0 measures a cluster when its *fitted* major axis clears `max(2 mm, 2 × the coarser in-plane spacing)`, so the 2 mm term binds only on sub-millimetre grids; a reformat of 3 mm spacing, for instance, retains nothing under 6 mm.

Two technical caveats of this stratification — the Feret proxy slightly understates the fitted axis that the floor actually tests, which is why the ≤2 mm bin contains 1,952 measured clusters, and measured-status ties at equal pixel counts can swap a tied pair between neighbouring length bins without affecting any count or recall — are detailed in `scripts/tl-stats/README.md`.

### Recall by cluster area

`tl_recall_by_area.csv` restratifies the same exactly attributed clusters a third way, by **area in square millimetres**: each cluster's pixel count multiplied by the in-plane voxel area of its own case (conversion details in `scripts/tl-stats/README.md`). Margins reproduce the corpus values exactly (6,951,667 clusters, 3,801,540 measured, recall 0.547).

| Cluster area | Clusters | Measured | Recall | Area ≤ t | Clusters | Measured | Recall | Area > t | Clusters | Measured | Recall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| ≤2 mm² | 1,684,011 | 64 | 0.000 | ≤2 mm² | 1,684,011 | 64 | 0.000 | >2 mm² | 5,267,656 | 3,801,476 | 0.722 |
| 2–5 mm² | 696,041 | 40,093 | 0.058 | ≤5 mm² | 2,380,052 | 40,157 | 0.017 | >5 mm² | 4,571,615 | 3,761,383 | 0.823 |
| 5–10 mm² | 526,878 | 237,702 | 0.451 | **≤10 mm²** | **2,906,930** | **277,859** | 0.096 | >10 mm² | 4,044,737 | 3,523,681 | 0.871 |
| 10–20 mm² | 545,631 | 371,569 | 0.681 | **≤20 mm²** | **3,452,561** | **649,428** | 0.188 | >20 mm² | 3,499,106 | 3,152,112 | 0.901 |
| 20–50 mm² | 725,574 | 530,074 | 0.731 | ≤50 mm² | 4,178,135 | 1,179,502 | 0.282 | >50 mm² | 2,773,532 | 2,622,038 | 0.945 |
| 50–100 mm² | 582,689 | 467,469 | 0.802 | ≤100 mm² | 4,760,824 | 1,646,971 | 0.346 | **>100 mm²** | 2,190,843 | 2,154,569 | **0.983** |
| 100–500 mm² | 1,347,005 | 1,311,216 | 0.973 | ≤500 mm² | 6,107,829 | 2,958,187 | 0.484 | **>500 mm²** | 843,838 | 843,353 | **0.999** |
| 500–1000 mm² | 428,510 | 428,120 | 0.999 | ≤1000 mm² | 6,536,339 | 3,386,307 | 0.518 | >1000 mm² | 415,328 | 415,233 | 1.000 |
| >1000 mm² | 415,328 | 415,233 | 1.000 | **All** | **6,951,667** | **3,801,540** | **0.547** |  |  |  |  |

#### Recall is 0.983 above 100 mm² and 0.999 above 500 mm²

Read from the `> t` block, **recall is 0.983 above 100 mm² and 0.999 above 500 mm²**.

#### 83.5% of unmeasured clusters are ≤10 mm² and 89.0% are ≤20 mm²

The unmeasured remainder concentrates at the bottom of this ladder as it does on the other two: **83.5% of the 3,150,127 unmeasured clusters are ≤10 mm² and 89.0% are ≤20 mm²**:

```
unmeasured ≤10 mm² = 2,906,930 - 277,859 = 2,629,071  ->  2,629,071 / 3,150,127 = 0.835
unmeasured ≤20 mm² = 3,452,561 - 649,428 = 2,803,133  ->  2,803,133 / 3,150,127 = 0.890
```

### The axial vs sagittal/coronal gap tracks anisotropy

**On thick-slice datasets recall is near-complete on axial slices but much lower on sagittal and coronal ones, and the gap comes from the voxel grid, not from the measurement rule.** PI-CAI reads 0.999 axial against 0.516 sagittal/coronal, LNQ2023 0.980 against 0.512, LIDC-IDRI 0.902 against 0.572.

The mechanism is geometric. A thick-slice volume samples a lesion densely within an axial slice but sparsely across slices, so a sagittal or coronal slice through the same lesion shows a strip only as wide as the number of axial slices the lesion spans — often one or two pixels — and those strips fall in exactly the size range where recall collapses, at any threshold. The millimetre floor keeps the *criterion* fair across planes: on a sagittal or coronal plane whose coarser in-plane side is the slice thickness, it demands proportionally more pixels. What it cannot change is the *object* — sagittal and coronal slices genuinely contain thinner cross-sections. The residual gap measures sampled geometry, which is why the corpus is reported per plane.

On near-isotropic volumes the gap nearly vanishes (MAMA-MIA 0.529 against 0.498; BraTS24 0.717 against 0.641).

## Train/test split

Case assignment changes for six datasets: HNTSMRG24 (43%), KiPA22 (43%), KiTS23 (43%), MSD (42%), BraTS24 (41%), autoPET-III (41%). These are the datasets whose earlier splits were force-aligned to v1.0.0; v1.4.0 is their first natural seeded split (seed 1024, ratio 0.7). The other six move 0%. Each percentage is the share of cases shared with that dataset's previous biometry plan whose train/test membership differs, measured against the previous version named in the per-dataset impact table above.

Split *sizes* are unchanged everywhere except BraTS24, which gains one case: `BraTS-MET-00232-000` is present in the v1.4.0 plan and absent from v1.1.1, taking BraTS24 from 2,121/912 to 2,122/912 train/test. That case also postdates BraTS24's v1.0.0 detection plan, which `doc/tl-annotation-stats-v1.4.0.md` records as the one coverage gap of the detection-sourced denominator.

Test-set metrics from v1.4.0 should not be compared against pre-1.4.0 metrics on the six affected datasets.

## Selecting the annotation version

```bash
export MedVision_PLANNER_VERSION=1.4.0    # or 'latest'
```

To stay on an older annotation, acknowledge this release:

```bash
export MedVision_PLANNER_VERSION=1.0.0
export MedVision_ACK_RELEASE=1.4.0
```

Only the 12 TL datasets prompt for acknowledgement; Angle-Distance and detection workloads do not require it.

## Reproducing

```bash
python scripts/gen-annotations/build_dataset.py \
  --data_dir <DATA> --dataset <NAME> --steps biometry \
  --new-annotation-version --cluster_filter_method major_axis
```

v1.4.0 was produced with **OpenCV 4.12.0** and **numpy 2.2.6**. `cv2.fitEllipse` is itself the measurement, so a different OpenCV version may return marginally different values. The legacy rule remains available as `--cluster_filter_method pixel_count`.
