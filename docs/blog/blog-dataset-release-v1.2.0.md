# MedVision v1.2.0: 8 new datasets, 30 in total, and no change to previously published annotations

MedVision v1.2.0 extends the catalogue from 22 to 30 datasets and from 820 to 950 configs. The release adds prostate, breast, lung-nodule and mediastinal-node oncology, the spine, head-and-neck organs at risk, brain fiducials, and PSMA/FDG PET. **No annotation published before this release was regenerated**: every `(pin, dataset, task)` combination that resolved at v1.1.1 returns byte-identical rows.

```bash
export MedVision_DATA_DIR=/path/to/data
export MedVision_PLANNER_VERSION=latest   # resolves to 1.2.0
```

---

## Recommended reading

The remainder of this article assumes the terminology and structural conventions established by the following material.

- [MedVision v1.1.1 release article](https://huggingface.co/blog/YongchengYAO/medvision-ds-v1-1-1): the preceding release note, describing the state of the catalogue that v1.2.0 extends.
- [Dataset and data-configuration concepts](https://medvision.readthedocs.io/en/latest/dataset/concepts.html): the reference documentation for tasks, planes, splits, config naming, and the annotation-versioning model referred to throughout.
- [Dataset Explorer](https://medvision-vlm.github.io/explorer.html): an interactive view of the released catalogue by dataset, anatomy, modality and task.

## Motivation

MedVision benchmarks the *quantitative* axis of medical image understanding: bounding boxes, tumour/lesion sizes in millimetres, and angles and distances derived from landmarks. The value of that axis is bounded by the anatomy it covers. At v1.1.1 the catalogue was unevenly distributed: abdominal CT and brain MRI were densely represented, while several clinically routine measurement settings were absent entirely.

v1.2.0 addresses the most conspicuous of those gaps.

## The 8 new datasets

| Dataset | Anatomy / Modality | Cases | Tasks | Configs | Licence |
| --- | --- | ---: | --- | ---: | --- |
| **AFIDs** | brain / T1w MRI | 72 | Landmarks | 4 | CC BY 4.0 |
| **PDDCA** | head & neck / CT | 48 | Mask, Box, Landmarks | 16 | public domain, CC BY 3.0 |
| **VerSe** | spine / CT | 325 | Mask, Box, Landmarks | 14 | CC BY-SA 4.0 |
| **PI-CAI** | prostate / bpMRI (T2W) | 425 | Mask, Box, T/L | 18 | CC BY-NC 4.0 |
| **MAMA-MIA** | breast / DCE-MRI | 1,506 | Mask, Box, T/L | 18 | CC BY-NC 4.0 |
| **DEEP-PSMA** | whole body / PSMA + FDG PET | 100 × 2 tracers | Mask, Box, T/L | 28 | CC BY-NC 4.0 |
| **LNQ2023** | mediastinum / CT | 120 | Mask, Box, T/L | 14 | CC BY 4.0 |
| **LIDC-IDRI** | lung / CT | 1,013 | Mask, Box, T/L | 18 | CC BY 3.0 |

*Mask = Mask-Size · Box = Box-Size · T/L = Tumor-Lesion-Size · Landmarks = Biometrics-From-Landmarks.*

**Coverage gains:**

- **PET ceases to be a single-dataset modality.** DEEP-PSMA contributes PSMA and FDG tracers with SUV volumes, the first PSMA data in MedVision, taking PET from 1,038 to 1,238 volumes.
- **Two new anatomy groups are introduced**: *Breast Tumor/Lesion* and *Prostate Tumor/Lesion*, bringing the anatomy taxonomy from 36 to 38 groups over 255 labels.
- **Landmark-derived measurement more than doubles its sources.** Geometric biometry, meaning angles and distances computed from human-placed landmarks, previously came from two datasets (Ceph-Biometrics-400, FeTA24). It now comes from five: AFIDs (32 brain fiducials, 72 cases), PDDCA (33 cases), and VerSe (lumbar centroids, 250 cases) add 355 landmark cases across three new anatomies. **Angle measurement in particular goes from one dataset to two**, with VerSe joining Ceph-Biometrics-400 at a mean lumbar angle of 15.9° against cephalometry's 45.3°.
- **Oncology breadth.** Prostate (PI-CAI), breast (MAMA-MIA), lung nodule (LIDC-IDRI) and mediastinal node (LNQ2023) lesions join the existing tumour/lesion datasets, covering four organ systems that previously had no quantitative representation.
- **The spine is covered in full.** VerSe covers C1–L6 per vertebra (26 labels), contributing 3.3M annotated 2D slices.

## Corpus statistics

All figures below are from the shipped summary of the released corpus.

| | v1.1.1 | v1.2.0 | Δ |
| --- | ---: | ---: | ---: |
| Datasets | 22 | **30** | +8 |
| Configs | 820 | **950** | +130 |
| Subjects | 18,086 | **21,695** | +3,609 |
| 3D volumes | 29,031 | **32,740** | +3,709 |
| 2D slices | 11,237,090 | **11,867,840** | +630,750 |
| Annotations (single-instance) | 24,279,534 | **24,738,696** | +459,162 |
| Annotations (multi-instance) | 45,338,754 | **46,666,781** | +1,328,027 |

By task family, single-instance:

| Task | v1.1.1 | v1.2.0 | Δ |
| --- | ---: | ---: | ---: |
| Box-Size | 24,236,327 | **24,689,147** | +1.9% |
| Tumor-Lesion-Size | 35,282 | **39,560** | **+12.1%** |
| Biometrics-From-Landmarks | 7,925 | **9,989** | **+26.0%** |

By modality, in 3D volumes:

| Modality | v1.1.1 | v1.2.0 | Δ |
| --- | ---: | ---: | ---: |
| MRI | 15,613 | **17,616** | +2,003 |
| CT | 10,980 | **12,486** | +1,506 |
| PET | 1,038 | **1,238** | **+19.3%** |
| Ultrasound | 1,000 | 1,000 | — |
| X-Ray | 400 | 400 | — |

The headline count moves little for Box-Size, which was already at 24M, whereas the two *measurement* tasks, the most demanding component of the benchmark and consistently the scarcest, grow by 12% and 26%.

## Backward compatibility

A dataset release ordinarily requires re-validation of everything downstream. That is not the case here:

- **No pre-existing dataset was regenerated.** Every annotation file published before v1.2.0 is unmodified on disk.
- Where a pin matched a file exactly, it still resolves to that file. Where it fell back to `1.0.0`, it still falls back to `1.0.0`.
- No environment variable, config name, feature schema or split name changed.
- Pinning `MedVision_PLANNER_VERSION=1.1.1` continues to work exactly as before. The 8 new datasets simply cannot be loaded at that pin, since their annotations did not yet exist; a request for one now raises a named error rather than failing deep inside the loader.

Verified across all 950 configs × every accepted pin.

## Loader corrections

v1.2.0 is also a correctness release. Full detail is given in [`doc/release-v1.2.0.md`](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/release-v1.2.0.md); the principal changes are summarised below.

**1. Cached data could become stale, and on one occasion did.** The HuggingFace builder-cache key was derived from the version *requested*, not from the annotation file that version resolved to. Requesting a version does not pin down the data: the same request can point at different files at different times. When v1.1.1 re-aligned the train/test split for six datasets, the measurement values stayed byte-identical, so a stale cache appeared entirely normal and only the partition differed.

A user is affected only if **all four** conditions hold: a `Tumor-Lesion-Size` config was loaded, of BraTS24 / HNTSMRG24 / KiPA22 / KiTS23 / MSD / autoPET-III, cached before v1.1.1 shipped, and that cache has been reused since. In that case, refresh once:

```python
import os
from datasets import load_dataset

os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "True"   # refresh the annotation file
ds = load_dataset(
    "YongchengYAO/MedVision",
    name="<your affected config>",
    trust_remote_code=True,
    split="test",
    download_mode="force_redownload",                  # rebuild the Arrow cache
)
```

From v1.2.0 the cache key is the annotation version that **actually loads**, so this class of silent staleness cannot recur. As a standing policy, **a published annotation file is never rewritten in place; corrections always receive a new version number.**

**2. Version resolution is now per dataset.** `MedVision_PLANNER_VERSION` is read as a *ceiling* rather than an exact match, requesting the newest annotations that existed at or before the given point, resolved dataset by dataset. The change was necessary: once the release became 1.2.0, the previous rule would have searched for a `1.2.0` Tumor-Lesion-Size annotation for every dataset, and the 162 pre-existing TL configs would all have failed. In consequence, `MedVision_ACK_RELEASE` is now required only when the dataset being requested has moved past the given pin. Since v1.2.0 changed no existing annotation, users pinned at `1.1.1` are never prompted.

**3. Two data roots no longer share one cache.** Rows contain absolute paths built from `MedVision_DATA_DIR`, but the root was not part of the cache key, so pointing at a second root could return the first root's paths. The canonicalised root is now part of the fingerprint. Users who have ever used two data roots with a bare `load_dataset` and no per-root `HF_DATASETS_CACHE` should clear those caches once. Users of [`medvision_bm`](https://github.com/YongchengYAO/MedVision) were never exposed, as `setup_env_hf_medvision_ds()` already moved the cache with the data root.

Alongside these, four download-reliability defects are fixed: a failed image download recorded as a finished install, a crash when two configs of one dataset were prepared concurrently, a broken relative data root, and roughly 27 GiB of needless re-downloading. None require action.

## Availability

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

Config names follow `<Dataset>_<TaskType>_Task<NN>_<Plane>_<Split>`. Full lists ship in the [MedVision repository](https://huggingface.co/datasets/YongchengYAO/MedVision) under `info/v1.2.0/` (950 configs) and `info/v1.0.0-v1.1.1/` (820 configs).

`datasets==3.6.0` is required, as `datasets>=4` removed loading-script support entirely.

## Distribution

Images and masks are mirrored per dataset as `-Lite` repositories on the Hub, 194.4 GiB across 28,060 files for these 8 datasets, while the annotations remain in the main [MedVision repository](https://huggingface.co/datasets/YongchengYAO/MedVision). That separation is what allows an annotation correction to ship without moving a single gigabyte of imaging. Each mirror is pinned by **commit SHA** rather than by branch, so a later push cannot silently change what a given MedVision version resolves to.

The `-Lite` suffix is a provenance marker, not a quality one: every mirror is a *derived* redistribution, format-converted to `nii.gz`, reoriented to RAS+, with masks normalised onto the image grid. Six mirrors additionally exclude part of their source for a stated reason (licence terms that forbid derivatives, duplicate series that would leak a subject across the train/test split, projection radiographs, or cases whose annotation is known to be incomplete).

Each of the 8 was round-trip verified against the from-source pipeline through per-folder file counts, SHA-256 byte-identity on sampled volumes, and 1:1 correspondence for rebuilt landmark subsets. 8/8 pass. AFIDs and PDDCA were additionally re-run from the original upstream with the Hub bypassed entirely.

## Source-dataset obligations and citation

MedVision redistributes derived annotations and preprocessed volumes; the original licences continue to govern. Two obligations in this release warrant explicit attention:

- **VerSe is CC BY-SA 4.0**, so ShareAlike propagates to anything derived from it.
- **PI-CAI, MAMA-MIA and DEEP-PSMA are CC BY-NC 4.0**, permitting non-commercial use only.

The original dataset publications must be cited, not only MedVision. Three require more than one citation: **VerSe** (Löffler 2020, Liebl 2021, Sekuboyina 2021), **PI-CAI** (the challenge dataset *and* Pooch et al. 2025, whose expert annotations supply 205 of the 425 shipped cases), and **MAMA-MIA** (the *Scientific Data* paper plus arXiv:2603.01250).

MedVision is intended for research and education. **It is not a medical device and must not be used for clinical decision-making.**

## License: CC-BY-4.0

MedVision is released under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Users are permitted to utilize, adapt, and build upon this dataset for both academic and commercial purposes, provided that appropriate credit is given. MedVision is a meta-dataset built upon various publicly available source datasets. While the annotations provided by MedVision are covered by the CC-BY 4.0 license, any downstream application must continue to comply with the specific usage terms and licensing requirements stipulated by the curators of the original raw imaging data. It is the responsibility of the user to ensure that their application of this data aligns with the license agreements of all constituent source datasets.

## Acknowledgement

This work was supported by the United Kingdom Research and Innovation (grant EP/S02431X/1), UKRI Centre for Doctoral Training in Biomedical AI at the University of Edinburgh, School of Informatics.

## Links

| | |
| --- | --- |
| 🩻 Dataset | [huggingface.co/datasets/YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision) |
| 🌏 Project | [medvision-vlm.github.io](https://medvision-vlm.github.io) |
| 🔎 Data Explorer | [medvision-vlm.github.io/explorer.html](https://medvision-vlm.github.io/explorer.html) |
| 📖 Paper | [arXiv:2511.18676](https://arxiv.org/abs/2511.18676) |

Full release notes: [`doc/release-v1.2.0.md`](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/release-v1.2.0.md) (loader and versioning) and [`doc/release-v1.2.0-datasets.md`](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/release-v1.2.0-datasets.md) (the 8 datasets in detail, covering provenance, filtering decisions, and the preprocessing defects identified during curation).
