# Release v1.3.0

**v1.3.0 adds MSWAL: 484 abdominal CT cases with 42 MedVision configurations.** The release provides segmentation and detection plans for gallstones and kidney stones, plus biometry plans for five tumour and lesion labels. It also fixes CT-window selection for several newly introduced label names, so their tumour/lesion figures render in the intended soft-tissue window.

```bash
export MedVision_PLANNER_VERSION=latest   # resolves to 1.3.0
```

## Summary

| | Change | In one line | Action |
| --- | --- | --- | --- |
| **Major** | | | |
| 1 | [MSWAL dataset](#mswal-dataset) | 484 abdominal CT cases, 42 configurations, and five biometry tasks | update to `1.3.0` or `latest` to use it |
| 2 | [Reproducible download paths](#download-paths) | build from the upstream source or use the pinned preprocessed mirror | none |
| **Minor** | | | |
| 3 | [CT figure normalization](#ct-figure-normalization) | new cancer, cyst, and stone labels use the intended CT windows | none |
| 4 | [Catalogue and validation updates](#catalogue-and-validation) | configuration catalogue grows from 950 to 992 entries | none |

## MSWAL dataset

MSWAL contributes **484 abdominal CT cases**. The upstream `dataset.json` names a 210-case test split, but those cases were not uploaded; MedVision therefore plans a reproducible split of the available cohort using seed `1024` and a `0.7` training ratio:

| Split | Cases |
| --- | ---: |
| Train | 338 |
| Test | 146 |
| Total | 484 |

The dataset adds 42 configurations:

| Family | Configurations | Scope |
| --- | ---: | --- |
| Mask-Size | 6 | segmentation size benchmarks |
| Box-Size | 6 | detection size benchmarks |
| Tumor-Lesion-Size | 30 | five labels across three anatomical planes |
| **Total** | **42** | |

The five biometry labels are liver tumour, kidney tumour, pancreatic cancer, liver cyst, and kidney cyst (labels 3–7). Gallstone and kidney stone (labels 1–2) are included for segmentation and detection only.

## Download paths

MSWAL has two supported preparation routes:

| Route | Source | Pinned revision | Notes |
| --- | --- | --- | --- |
| Raw build | `zhaodongwu/MSWAL` | `62c286b0` | `download_raw.py` copies headers, converts images to `uint16`, and reorients to RAS+ before planning |
| Fast download | `YongchengYAO/MSWAL-Lite` | `39fb50b6` | `download_fast.py` retrieves the prepared images and masks |

The raw route uses the 484 uploaded `imagesTr` cases and applies the split above. Reorienting to RAS+ during download ensures image arrays and generated annotations share the same coordinate frame.

## CT figure normalization

`LABEL_MAP_REGROUP` now recognizes `pancreatic cancer`, `liver cyst`, `gallstone`, and `kidney stone`. Therefore, we can use the intended soft-tissue Hounsfield-unit window for image normalization.

## Catalogue and validation

The release registers MSWAL in `MedVision.py`, including its annotation index, biometry family, package mapping, and version notes. The package version and release frontier are now `1.3.0`.

The published catalogue now contains **992 configurations**, up from 950. Validator expectations were updated to 75 annotation-resolution pairs across 31 datasets. The release was checked with:

- `test_annotation_resolution`: 440 / 440 checks passed
- `test_tl_ack_gate`: 16 / 16 checks passed

## For maintainers

The MSWAL build recipe is registered in `scripts/gen-annotations/dataset_specs.py`; use the raw route when rebuilding the dataset from source. The package is exposed through `medvision_ds.datasets`, and the preprocessed archive, benchmark plans, landmarks, and regenerated figures are published in `Datasets/MSWAL.zip`.

## See also

- `doc/changelog.md` — release history
- `doc/file-structure.md` — repository layout
- `scripts/gen-annotations/README.md` — rebuilding dataset annotations
