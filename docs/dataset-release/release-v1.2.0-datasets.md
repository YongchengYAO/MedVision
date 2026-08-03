# Release v1.2.0 — the 8 new datasets

Companion to [`doc/release-v1.2.0.md`](release-v1.2.0.md), which covers the loader and
annotation-versioning changes. **This note covers the data**: what was added, where it came from,
what was included or filtered and why, and how it is distributed.

```bash
export MedVision_DATA_DIR=/path/to/data   # required — the loader raises without it
export MedVision_PLANNER_VERSION=latest   # resolves to 1.2.0
```

Nothing here changes an existing dataset. Every annotation published before v1.2.0 loads byte for
byte as it did at v1.1.1.

**These 8 datasets require a pin of `1.2.0` or `latest`.** Their annotations did not exist at any
earlier version, so requesting one at `MedVision_PLANNER_VERSION=1.1.1` or below raises an error
naming the dataset and the versions that do exist — see
[Pinning a version older than v1.2.0](release-v1.2.0.md#pinning-a-version-older-than-v120).

## At a glance

**+3,609 subjects · +3,709 volumes · +130 configs** (catalogue 820 → 950).

| Dataset | Anatomy / Modality | Cases | Tasks | Configs | Licence |
| --- | --- | ---: | --- | ---: | --- |
| **AFIDs** | brain / T1w MRI | 72 | Landmarks | 4 | CC BY 4.0 |
| **PDDCA** | head & neck / CT | 48 | Mask, Box, Landmarks | 16 | public domain, CC BY 3.0 |
| **VerSe** | spine / CT | 325 | Mask, Box, Landmarks | 14 | CC BY-SA 4.0 |
| **PI-CAI** | prostate / bpMRI (T2W) | 425 | Mask, Box, T/L | 18 | CC BY-NC 4.0 |
| **MAMA-MIA** | breast / DCE-MRI | 1,506 | Mask, Box, T/L | 18 | CC BY-NC 4.0 |
| **DEEP-PSMA** | whole body / PSMA + FDG PET | 100 ×2 tracers | Mask, Box, T/L | 28 | CC BY-NC 4.0 |
| **LNQ2023** | mediastinum / CT | 120 | Mask, Box, T/L | 14 | CC BY 4.0 |
| **LIDC-IDRI** | lung / CT | 1,013 | Mask, Box, T/L | 18 | CC BY 3.0 |

*Mask = Mask-Size, Box = Box-Size, T/L = Tumor-Lesion-Size, Landmarks = Biometrics-From-Landmarks.*

**What this widens:**

- **PET becomes a first-class modality.** DEEP-PSMA adds PSMA and FDG tracers with SUV volumes —
  the first PSMA data in MedVision.
- **Spine, head-and-neck OARs, and brain fiducials are new anatomies.** VerSe covers C1–L6 per
  vertebra; PDDCA covers 9 organs at risk; AFIDs covers 32 standardized brain fiducials.
- **Biometrics-From-Landmarks triples.** It was the rarest task family; AFIDs, PDDCA and VerSe add
  355 landmark cases across 3 anatomies.
- **Oncology breadth.** Prostate (PI-CAI), breast (MAMA-MIA), lung nodule (LIDC-IDRI) and
  mediastinal node (LNQ2023) lesions join the existing T/L datasets.

## The datasets

### AFIDs — 72 cases, T1w brain MRI, landmarks only

[Anatomical Fiducials](https://github.com/afids/afids-data), OpenNeuro `ds004470` (32 SNSX,
7T MP2RAGE) + `ds004471` (40 LHSCPD, 1.5T). The landmark coordinates are **CC BY 4.0** (per the
afids-data `LICENSE.md`); the accompanying imaging is CC0. CC BY 4.0 governs the combined product.

32 expert fiducials per case (anterior/posterior commissure, mammillary bodies, corpus callosum
genu and splenium, …). Parsed from `.fcsv` (`# CoordinateSystem = 0`, i.e. RAS world-mm) and
converted to 0-based voxel indices in the RAS+ volume. No segmentation masks exist, so AFIDs
publishes the **biometry task only** — 4 configs.

### PDDCA — 48 cases, head-and-neck CT

[PDDCA v1.4.1](http://www.imagenglab.com/newsite/pddca/), derived from the TCIA Head-Neck
Cetuximab collection. Public domain / CC BY 3.0.

9 organ-at-risk labels merged from per-structure NRRDs into one multi-label mask: mandible,
brainstem, both parotids, both submandibular glands, both optic nerves, optic chiasm.

Two properties worth knowing:

- **Structure availability is ragged.** 6 of the 9 structures appear in all 48 cases; the
  submandibular glands and mandible in fewer (36–41). Mask building skips missing structures
  rather than asserting — asserting would have rejected 8 otherwise-valid cases.
- **Only 33 cases ship landmarks upstream.** The biometry task therefore uses a 33-case subset
  (`Images-landmark/`); segmentation and detection use all 48.

> **LPS fix.** PDDCA's NRRDs declare `space: left-posterior-superior`. Copying that direction
> matrix verbatim into a (RAS-by-definition) NIfTI affine mirrors the volume left-right and
> anterior-posterior and makes the RAS+ reorientation a silent no-op. The affine is corrected
> before reorientation. Evidence: the chin landmark sits **0.0 mm** from the mandible mask after
> the fix versus **181.9 mm** before, and all 48 cases now have `_R` structures right of `_L`.

### VerSe — 325 scans, spine CT

[VerSe'19 + VerSe'20](https://github.com/anjany/verse). **CC BY-SA 4.0** — note the ShareAlike
obligation propagates to derived annotations.

Per-vertebra masks for C1–L6 plus T13 (26 labels), and lumbar centroid landmarks
(L1–L5) for the 250 scans whose field of view contains all five.

- **Centroid convention.** The challenge `*_ctd.json` values are **voxel indices in the native
  orientation**, not world-mm. Measured across all 374 scans / 4,522 centroids: read as voxel
  indices, 98.8% land inside the correct vertebra label with 0 out of bounds; read as world-mm,
  **94.7% fall outside the volume entirely**. The converter maps native ijk → world → RAS+ index.
- **QFORM geometry.** VerSe files carry `sform_code=0, qform_code=1`, so geometry is read through
  nibabel's `.affine` (which resolves QFORM automatically). Reading SFORM directly returns zeros.
- **Field of view varies** from cervical-only to whole-body, which is why the lumbar biometry
  subset (`Images-lumbar/`, 250) is smaller than the full set (325).

### PI-CAI — 425 cases, prostate T2-weighted MRI

[PI-CAI](https://pi-cai.grand-challenge.org/); imaging from
[Zenodo](https://zenodo.org/records/6624726), labels from
[`picai_labels`](https://github.com/DIAGNijmegen/picai_labels). CC BY-NC 4.0.

Clinically significant prostate cancer (csPCa) lesions. `picai_labels` publishes human-expert
delineations for **all 1500** cases, split across two disjoint folders — and MedVision uses both:

| Folder | Cases | Note |
| --- | ---: | --- |
| `human_expert/resampled/` | 1295 | original expert annotations, resampled onto the axial T2W grid |
| `human_expert/Pooch25/` | 205 | added 2025-07-01 by [Pooch et al., 2025](https://doi.org/10.1101/2025.05.13.25327456) for the positives that previously carried only an AI mask — **all 205 are positive** |
| **Kept (non-empty mask)** | **425** | 220 from `resampled/` + all 205 from `Pooch25/` |

Cases whose expert mask is all-zero are not redistributed: with no delineated lesion there is
nothing for the Tumor-Lesion-Size task to measure.

> An earlier draft of this note claimed the 205 carried "only AI-derived masks". That was true
> until 2025-07-01 and is now wrong — reading `resampled/` alone silently discarded 205
> expert-annotated positives, nearly halving the usable data.

**T2W only.** PI-CAI is biparametric (T2W + ADC + HBV). Per the upstream README the original
annotations were drawn at T2W, ADC *or* DWI/HBV resolution depending on the annotator, so the
T2W-resampled delineations are the ones with an exact image/mask correspondence. Diffusion
sequences are acquired far coarser (~2 mm in-plane) than these T2W scans (0.23–0.56 mm), so a
mask drawn on ADC and resampled up would carry ~2 mm of boundary quantisation into a millimetre
measurement — on lesions often under 10 mm. Only **2 of 425** masks needed resampling onto the
T2W grid; the rest matched exactly.

Masks encode the **ISUP grade** as the voxel value (`{2,3,4,5}`, with no label 1) and are
binarized to `{0,1}`.

### MAMA-MIA — 1,506 cases, breast DCE-MRI

[MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA) via
[Synapse syn60868042](https://www.synapse.org/Synapse:syn60868042). CC BY-NC 4.0. Four cohorts
(DUKE, ISPY1, ISPY2, NACT), each case with an expert primary-tumour mask.

**One DCE phase per case.** Each case ships a pre-contrast volume (`_0000`) plus several
post-contrast phases; the expert mask is drawn on the **first post-contrast** (`_0001`), so that
is the volume published. The convention was confirmed against the official
`MAMA-MIA/src/preprocessing.py::read_mri_phase_from_patient_id`.

### DEEP-PSMA — 100 cases × 2 tracers, PET

[DEEP-PSMA](https://deep-psma.grand-challenge.org/) via
[Zenodo](https://zenodo.org/records/15281784). CC BY-NC 4.0.

Total tumour burden (TTB) on **PSMA** and **FDG** PET. The two tracers are kept in separate
image/mask folders (`Images-PSMA`, `Images-FDG`, …) as **two task IDs**, so the subject-level
train/test split cannot place the same patient's two scans on opposite sides.

**PET only — no CT.** TTB is defined by SUV thresholding on the PET and delivered on the PET grid
(e.g. `192×192×335` at `2.87 × 2.87 × 3.27` mm). A PET/CT's CT component is acquired near 1 mm for
attenuation correction; using it as the image would require resampling the mask onto a ~3× finer
grid — inventing lesion boundary detail that was never annotated, and changing the physical
measurements the benchmark scores.

### LNQ2023 — 120 cases, mediastinal lymph nodes, chest CT

[LNQ2023](https://lnq2023.grand-challenge.org/), redistributed from the **TCIA** release
[MEDIASTINAL-LYMPH-NODE-SEG](https://www.cancerimagingarchive.net/collection/mediastinal-lymph-node-seg/)
(DOI 10.7937/QVAZ-JA09, **CC BY 4.0**) — deliberately *not* the Zenodo challenge copy, which is
CC BY-NC-ND and forbids derivative works.

MedVision's first DICOM + DICOM-SEG pipeline. CT series and SEG objects are paired via
`ReferencedSeriesSequence` rather than by filename order — verified correct on all 513 series of
the collection, before the completeness filter below reduces the shipped set to 120.

**Only exhaustively annotated cases are kept: 513 → 120.** Each SEG series in the TCIA release
declares its own completeness in the DICOM `SeriesDescription` tag — `Fully Annotated` (120) or
`Partially Annotated` (393). The partially annotated set is the challenge's *training* split,
where only a subset of the visible nodes was contoured. Measured over the masks themselves:

| `SeriesDescription` | cases | nodes/case (mean) | median | max | cases with exactly 1 node |
| --- | ---: | ---: | ---: | ---: | --- |
| `Fully Annotated` | 120 | **9.00** | 8 | 42 | 1 / 120 (1%) |
| `Partially Annotated` | 393 | **1.46** | 1 | 6 | 249 / 393 (63%) |

A **6.2×** gap: 63% of partially annotated cases carry exactly one segmented node against a
median of 8 for the fully annotated ones, so most true nodes there are unlabelled. **Unlabelled
is not negative** — a model that correctly detects such a node is scored as a false positive, and
a size measurement on it has no reference. Those cases cannot serve as benchmark ground truth, so
the downloader skips any SEG series not marked `Fully Annotated`.

### LIDC-IDRI — 1,013 scans, lung nodules, chest CT

[LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) (TCIA), CC BY 3.0. The
largest addition in this release.

MedVision's first multi-reader XML-contour pipeline. Masks are **consensus** binary nodule masks
built from four radiologists' contours via `pylidc` at the 50% consensus level. 880 of the 1,013
scans contain at least one nodule; the rest carry an all-zero mask. 8 patients contributed two CT
series, each getting a unique case ID.

Two exclusions apply:

- **CT only.** The 237 DX and 53 CR series in the same collection are projection radiographs, not
  volumes, and carry no nodule contours. 1,018 CT series remain.
- **Duplicate-z series dropped (1,018 → 1,013).** `LIDC-IDRI-0085`, `-0146`, `-0418`, `-0572` and
  `-0979` each contain two or more DICOM slices at the same `ImagePositionPatient` z, so the
  series has no single well-defined volume. This matters here because the image and the mask are
  built by *different* libraries — SimpleITK reconstructs the volume from an ordered file list,
  while `pylidc`'s `consensus()` returns **array indices** into its own view of that volume. The
  two reconstructions must agree index-for-index or a nodule contour lands on the wrong slice,
  producing a correctly-shaped mask over the wrong anatomy: measured against pylidc's own slice
  selection, a mismatched choice differs by up to **1,386 HU**, i.e. a completely different
  structure rather than resampling noise. Reproducing pylidc's internal tie-break is possible but
  makes correctness depend on an undocumented implementation detail of a third-party library, so
  these five series are excluded instead.

## Distribution

**In plain English.** The heavy files — images and masks — live in a separate mirror repository
per dataset. The small files — the annotations that say what to measure — stay in the main
MedVision repository. The loader fetches from both and assembles one folder. Splitting them this
way is what lets an annotation correction ship without moving a single gigabyte of imaging.

**Technically.** Image and mask volumes are mirrored on the Hugging Face Hub so users skip the
from-source pipeline (VerSe alone is a 51 GB fetch plus hours of planning). **Annotations stay in
[`YongchengYAO/MedVision`](https://huggingface.co/datasets/YongchengYAO/MedVision)** and are the
only thing the `_v{X}` version in `benchmark_plan_*_v{X}.json.gz` tracks — which is why the
annotation version of a dataset can advance while its mirror commit stays pinned.

| Dataset | Image/mask mirror | Size | Shards | Pinned commit |
| --- | --- | ---: | ---: | --- |
| AFIDs | `YongchengYAO/AFIDs-Lite` | 1.3 GB | 1 | `c6b5568bd8a1` |
| PDDCA | `YongchengYAO/PDDCA-Lite` | 1.7 GB | 1 | `dd814c9679d6` |
| LNQ2023 | `YongchengYAO/LNQ2023-Lite` | 3.2 GB | 1 | `f7c7ef4f5ac1` |
| VerSe | `YongchengYAO/VerSe-Lite` | 45.5 GB | 5 | `d521b23100ea` |
| PI-CAI | `YongchengYAO/PI-CAI-Lite` | 3.3 GB | 1 | `c381f77130fa` |
| MAMA-MIA | `YongchengYAO/MAMA-MIA-Lite` | 17.7 GB | 2 | `989c74c2f1c4` |
| DEEP-PSMA | `YongchengYAO/DEEP-PSMA-Lite` | 3.2 GB | 1 | `f89fc6abd847` |
| LIDC-IDRI | `YongchengYAO/LIDC-IDRI-Lite` | 77.0 GB | 8 | `1488897f4df6` |

Volumes are stored as **uncompressed (`ZIP_STORED`) shards of ≤10 GiB** — `.nii.gz` is already
deflated, so re-compressing costs hours for no gain, and sharding is required because the Hub
caps a single LFS file at 50 GB.

Each `download_fast.py` pins its mirror by **commit SHA**, not by branch, so a later push to a
mirror cannot silently change what a given MedVision version resolves to. 

### The `-Lite` suffix

**Every mirror carries `-Lite`**, because every one is a *derived* redistribution rather than a
copy of its source: all volumes are format-converted (NRRD / DICOM / `.mha` → `nii.gz`) and
reoriented to RAS+, and masks are normalised onto the image grid. The suffix is a provenance
marker, not a quality one — it says "reproduce MedVision from this, but cite the original
release".

Six mirrors additionally **exclude** part of their source. Every exclusion has a stated reason:

| Mirror | What is not mirrored |
| --- | --- |
| `VerSe-Lite` | 30 `sub-gl*` scans (CC BY-NC-ND — derivatives forbidden) and 19 duplicate `_split-verse<NNN>` series (would leak a subject across the train/test split). 374 → 344 redistributable → **325**. |
| `PI-CAI-Lite` | The ADC and HBV sequences, and cases whose expert mask is empty. Both expert folders (`resampled/` 1295 + `Pooch25/` 205) are used → **425** positives of 1500. |
| `MAMA-MIA-Lite` | DCE phases other than the annotated first post-contrast. |
| `DEEP-PSMA-Lite` | The companion CT and `totseg_24` volumes. |
| `LIDC-IDRI-Lite` | The 237 DX and 53 CR projection-radiograph series, plus 5 CT series with duplicate-z slices (1018 → **1013**). |
| `LNQ2023-Lite` | The 393 `Partially Annotated` cases — only the 120 `Fully Annotated` ones are kept (513 → **120**). |

`AFIDs-Lite` and `PDDCA-Lite` carry every case of their source — they are `-Lite` purely by
virtue of the preprocessing above.

### Subset folders are rebuilt, not mirrored

**In plain English.** Only some cases carry landmarks, so the landmark task needs its own image
folder containing exactly those cases and no others. Those folders hold copies of volumes that
are already in `Images/`, so mirroring them would upload and download the same data twice. They
are rebuilt locally instead, from the list of cases the annotations name.

**Technically.** `VerSe/Images-lumbar/` and `PDDCA/Images-landmark/` are strict subsets of
`Images/` that exist because the biometry planner requires its `image_folder` to be exactly 1:1
with `Landmarks/` (it raises `FileNotFoundError` on a case with no landmark file). Rather than
mirror ~38 GB of duplicate volumes, `download_fast.py` rebuilds them by hardlinking the cases
named in `Landmarks/`.

This depends on an ordering guarantee in the loader: step 3.1 extracts the annotation archive
**before** step 3.2 runs the downloader, so `Landmarks/` is already on disk when
`download_fast.py` reads it. Step 3.1 now also holds a per-dataset lock, so two configs of one
dataset prepared concurrently cannot race each other through that extraction.

## Loading

**In plain English.** Two environment variables must be set before the loader is imported: where
to put the data, and which annotation version you are willing to load. Neither has a default —
the loader refuses to guess, because guessing either one wrong is silent rather than loud.

**Technically.** `MedVision_DATA_DIR` is checked at module import and raises `ValueError` if
unset; `MedVision_PLANNER_VERSION` raises `EnvironmentError` during `_split_generators` if unset.
Both must therefore be in the environment before `load_dataset` runs.

```python
import os
os.environ["MedVision_DATA_DIR"] = "/path/to/data"
os.environ["MedVision_PLANNER_VERSION"] = "latest"   # or "1.2.0"

from datasets import load_dataset

ds = load_dataset(
    "YongchengYAO/MedVision",
    "VerSe_BiometricsFromLandmarks_Task01_Sagittal_Test",
    trust_remote_code=True,
)
```

Config names follow the existing grammar
`<Dataset>_<TaskType>_Task<NN>_<Plane>_<Split>`. DEEP-PSMA uses `Task01` for PSMA and `Task02`
for FDG. Full lists: `info/v1.2.0/ConfigurationsList_{All,Train,Test}.csv`.

`datasets==3.6.0` is required — `datasets>=4` removed loading-script support entirely.

Every one of these datasets resolves to annotation version `1.2.0`, since that is the only
version they publish. Loading them alongside pre-existing datasets at `latest` works and needs no
acknowledgement: `MedVision_ACK_RELEASE` is required only when your pin is *older* than a
dataset's newest annotation, which no pin of `latest` ever is. See
[How annotation versions are resolved](release-v1.2.0.md#how-annotation-versions-are-resolved).

## Provenance and verification

The shipped corpus is **28,060 files / 194.4 GiB** across the 8 datasets. Every mirror was
verified against the from-source pipeline output:

- **Round-trip verification, all 8 datasets.** Each mirror was re-downloaded into a fresh
  directory exactly as `MedVision.py` step 3 does it — annotation zip first, then the package's
  own `download_fast.py` chosen by the same first-match-wins rule the loader uses — and checked
  against the pipeline output for **per-folder file counts**, **SHA-256 byte-identity** on
  sampled volumes, and (for VerSe and PDDCA) that the rebuilt `Images-*` subset is exactly 1:1
  with `Landmarks/`. **8/8 PASS.**
- **Exhaustive hash comparison** — an earlier full sweep SHA-256'd every file on both sides in
  both directions (29,230 files / 208.9 GiB at that point, before the LNQ2023, PI-CAI and
  LIDC-IDRI rebuilds): **zero content mismatches, zero one-sided files.**
- **Independent from-source re-run** — AFIDs and PDDCA re-downloaded from the original upstream
  (OpenNeuro S3, imagenglab.com) with the Hub bypassed entirely; `Images/`, `Masks/` and
  `Images-landmark/` byte-identical to both the pipeline output and the mirror. This is the only
  non-circular evidence: the mirrors are built *from* the pipeline output, so comparing the two
  proves the pack/upload/download cycle is lossless, not that the two code paths agree.
- **Splits** are subject-level 70/30 with `random_seed=1024`, `sorted()` before shuffle. Verified
  after every rebuild — e.g. LIDC-IDRI's 1013 cases split 709/304 = exactly 0.700 across all
  three plans.
- **Plan validation** — 22 plan files, 10,893 task-case entries: all splits within 66–74%,
  sampled `image_file`/`mask_file`/`landmark_file` references resolve on disk, and each dataset's
  `dataset_info` is byte-identical across its plans (required by `compile_dataset_info.py`).

Two reproducibility notes for anyone regenerating from source:

- `Landmarks/*.json.gz` regenerated locally differ from the published bytes **in the gzip MTIME
  header field only** (RFC 1952 offsets 4–7); decompressed content is identical.
- `Landmarks-*fig*/` PNGs are matplotlib-version dependent. The published figures were rendered
  with **matplotlib 3.11.1**; with that version, re-rendering reproduces them byte for byte.

## Citation and licence obligations

MedVision redistributes derived annotations and preprocessed volumes; the original licences
continue to govern. In particular:

- **VerSe is CC BY-SA 4.0** — ShareAlike propagates to anything derived from it.
- **PI-CAI, MAMA-MIA and DEEP-PSMA are CC BY-NC 4.0** — non-commercial use only.
- **AFIDs is CC BY 4.0** (landmarks; its imaging is CC0); PDDCA is public domain / CC BY 3.0;
  LNQ2023 CC BY 4.0; LIDC-IDRI CC BY 3.0.

Cite the original dataset publications, not only MedVision. Each mirror's dataset card lists the
source papers. Three datasets need more than one citation:

- **VerSe** — all three of its papers (Löffler 2020, Liebl 2021, Sekuboyina 2021).
- **PI-CAI** — the challenge dataset *and*
  [Pooch et al., 2025](https://doi.org/10.1101/2025.05.13.25327456), whose expert annotations
  supply 205 of the 425 shipped cases.
- **MAMA-MIA** — the *Scientific Data* paper plus
  [arXiv:2603.01250](https://arxiv.org/abs/2603.01250).

MedVision is for research and education. It is not a medical device and must not be used for
clinical decision-making.

## See also

- [`doc/release-v1.2.0.md`](release-v1.2.0.md) — loader changes, annotation-version resolution,
  the acknowledgement gate, and the stale-cache fix
- [`doc/design-annotation-version-resolution.md`](design-annotation-version-resolution.md) —
  why the version-resolution mechanism has the shape it has, and what was rejected
- [`doc/file-structure.md`](file-structure.md) — dataset directory layout
