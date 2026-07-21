# Getting Started with the MedVision Dataset

> [!WARNING]
> `MedVision` relies on a remote data-loading script. `trust_remote_code` is no longer supported in `datasets>=4.0.0`, so pin **`datasets==3.6.0`**. Since dataset **v1.1.0**, the environment variable `MedVision_PLANNER_VERSION` is **required** — loading fails loudly if it is unset.

This is a 5-minute tour of the [MedVision dataset](https://huggingface.co/datasets/YongchengYAO/MedVision): what it is, how its configs are named, and how to install and load one. Deeper reference lives in the [documentation](https://medvision.readthedocs.io) — we point you there rather than inlining everything.

## 🩻 What is MedVision?

**MedVision** is a large-scale, multi-anatomy, multi-modality dataset for **quantitative medical image analysis**. It standardizes **22 public datasets** (`BraTS24`, `MSD`, `OAIZIB-CM`, …) — covering XR, CT, MRI, US, and PET — into one unified structure:

| | |
|---|---|
| 3D images | **29,031** |
| 2D slices | **11.2M** |
| Single-instance annotations | **24.3M** |
| Multi-instance annotations | **45.3M** |
| On-disk footprint (full copy) | **~1 TB** |

Images are stored as 3D volumes in RAS+ orientation, and annotations carry **real-world units** (mm and degrees) derived from the physical spacing in the image headers. That is what makes the benchmark *quantitative* rather than categorical: a box measured as 40 pixels wide means something only once you multiply by the millimetres-per-pixel of that scan.

A benchmark sample is a *(2D slice, target)* pair, counted **per target, not per instance**. The two annotation totals differ only in whether per-sample quality/size filters are applied — the filtered **single-instance** set (24.3M) is the default and the one to use for leaderboard comparison. See [Dataset versions & statistics](https://medvision.readthedocs.io/en/latest/dataset/statistics.html) for the full breakdown.

**Key features:** automatic download and processing of 3D volumes; dynamic 2D slicing along any anatomical plane (axial, coronal, sagittal); quantitative annotations (bounding boxes, mask size, tumor/lesion size, angle/distance); and a dedicated construction codebase, `medvision_ds`.

## 🧭 Data Configs: Naming What You Want

Two terms do a lot of work. A **dataset** is one of the 22 upstream sources. A **data config** is a named, ready-to-load subset — pass a config name to `load_dataset()` to select exactly which slices and annotations you get. Config names follow a fixed five-part convention:

```text
{dataset}_{annotation-type}_{task-ID}_{slice}_{split}
```

| Field | Values | Meaning |
|---|---|---|
| `dataset` | `OAIZIB-CM`, `BraTS24`, … | which upstream source |
| `annotation-type` | `BoxSize`, `TumorLesionSize`, `BiometricsFromLandmarks`, `MaskSize` | what kind of target |
| `task-ID` | `Task01`, `Task02`, … | a **local** task index within that dataset |
| `slice` | `Axial`, `Coronal`, `Sagittal` | slicing plane |
| `split` | `Train`, `Test` | subject-level split (70/30) |

Two concrete names:

```text
OAIZIB-CM_BoxSize_Task01_Axial_Test
BraTS24_TumorLesionSize_Task01_Axial_Train
```

The annotation type decides what the model must produce (and which fields each sample carries): `BoxSize` for bounding-box detection, `TumorLesionSize` for major/minor axis lengths in mm, `BiometricsFromLandmarks` for angles (degrees) and distances (mm) from landmarks, and `MaskSize` for segmentation-mask area. Every sample also carries the geometry needed to interpret it (`pixel_size`, `voxel_size`, slice locators, and more).

📚 Full reference: [Dataset concepts](https://medvision.readthedocs.io/en/latest/dataset/concepts.html) · browse configs interactively in the [Data Explorer](https://medvision-vlm.github.io/explorer.html).

## 📦 Installation

To just load the data, `datasets` is all you need — pinned to `3.6.0`:

```bash
pip install datasets==3.6.0
```

To use the benchmark tooling too (batch-download CLI, evaluation, SFT/RFT), install the `medvision_bm` package (Python ≥ 3.9); it already declares the `datasets==3.6.0` pin:

```bash
pip install medvision_bm
```

Note that the data-loading code (`medvision_ds`) is a **separate** package — installing `medvision_bm` does not pull it in. Install it explicitly, pointing at the folder where datasets should live:

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
```

Clone the repo if you plan to run evaluations or fine-tuning (the launcher scripts and task lists ship in the repository, not the package).

📚 Full reference: [Installation](https://medvision.readthedocs.io/en/latest/getting-started/installation.html)

## 🚀 Loading Your First Config

Set `MedVision_DATA_DIR` (where raw data, caches, and Arrow files live — a full copy is ~1 TB) and `MedVision_PLANNER_VERSION` **before** calling `load_dataset()`, and pass `trust_remote_code=True`:

```python
import os
from datasets import load_dataset

os.environ["MedVision_DATA_DIR"] = "/path/to/Data"
os.environ["MedVision_PLANNER_VERSION"] = "latest"   # required since v1.1.0

ds = load_dataset(
    "YongchengYAO/MedVision",
    name="OAIZIB-CM_BoxSize_Task01_Axial_Test",
    trust_remote_code=True,
    split="test",  # "test" for *_Test configs, "train" for *_Train configs
)
```

The `split` argument must match the split baked into the config name.

`MedVision_PLANNER_VERSION` selects which versioned annotation release builds the samples: `latest` (currently `1.1.1`), or an exact version like `1.1.0` / `1.0.0`. Different versions can change the exact sample set, so keep it fixed for reproducibility — **published leaderboard numbers use `1.0.0`**; for new studies we recommend `latest`. Pinning below the latest also requires `MedVision_ACK_RELEASE` set to the current latest.

📚 Full reference: [Loading data](https://medvision.readthedocs.io/en/latest/dataset/loading.html)

## 📥 Batch Downloads & Restricted Datasets

Downloading and building is slow, so for many configs use the CLI rather than scripting `load_dataset()` calls by hand. Ready-made lists (`ConfigurationsList_All.csv`, `_Test.csv`, `_Train.csv`) ship in the repo:

```bash
python -m medvision_bm.benchmark.download_datasets \
  --configs_csv dataset-info/dataset-configs/ConfigurationsList_Test.csv \
  --data_dir <data-folder>
```

Three source datasets (**FeTA24**, **SKM-TEA**, **ToothFairy2**) do not allow redistribution, so MedVision cannot mirror them. You apply for access from the owners yourself, then point MedVision at your own copy via environment variables. The other 19 datasets need no credentials. See the [restricted datasets overview](https://huggingface.co/datasets/YongchengYAO/MedVision#datasets) for the step-by-step guides.

## 🔗 Links

- 📊 Dataset: <https://huggingface.co/datasets/YongchengYAO/MedVision>
- 📖 Documentation: <https://medvision.readthedocs.io>
  - [Dataset concepts](https://medvision.readthedocs.io/en/latest/dataset/concepts.html) · [Loading data](https://medvision.readthedocs.io/en/latest/dataset/loading.html) · [Installation](https://medvision.readthedocs.io/en/latest/getting-started/installation.html)
- 🔎 Interactive Data Explorer: <https://medvision-vlm.github.io/explorer.html>
- 💻 Code: <https://github.com/YongchengYAO/MedVision>
