# Downloading and Loading MedVision Data

## Prerequisites

- `datasets==3.6.0` (`trust_remote_code` was removed in `datasets>=4.0.0`); `medvision_bm` pins it.
- `medvision_ds` reachable for the loader: installed into `<data_dir>` by `mvbm install mvds -d <data_dir>` / `python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>` (see `../../environment-setup/SKILL.md`). The loader itself also downloads `src/` and pip-installs it on a cold build when `MedVision_FORCE_INSTALL_CODE` is true (its default).
- Environment set **before** `import datasets`: `MedVision_DATA_DIR` (the loader raises at import otherwise) and `MedVision_PLANNER_VERSION`.

## `load_dataset` pattern

```python
import os
from datasets import load_dataset

os.environ["MedVision_DATA_DIR"] = "<data_dir>"
os.environ["MedVision_PLANNER_VERSION"] = "latest"   # or "1.0.0" (+ MedVision_ACK_RELEASE=1.4.0)

ds = load_dataset(
    "YongchengYAO/MedVision",
    name="OAIZIB-CM_BoxSize_Task01_Axial_Test",
    trust_remote_code=True,
    split="test",          # must match the _Test / _Train suffix of the config
)
```

First run: downloads the annotations zip, the `medvision_ds` code, the raw volumes (reoriented to RAS+), then builds Arrow files under `<data_dir>/.cache/huggingface/datasets` when `HF_DATASETS_CACHE` is pointed there (the `medvision_bm` CLIs do this through `setup_env_hf_medvision_ds`; plain `load_dataset` uses the default HF cache unless you set `HF_DATASETS_CACHE`). Second run: reuses the cache.

## Batch CLI

```bash
python -m medvision_bm.benchmark.download_datasets --data_dir <data_dir> --tasks_json <tasks.json> [--split test]
python -m medvision_bm.benchmark.download_datasets --data_dir <data_dir> --configs_csv <ConfigurationsList_Test.csv> [--split test]
```

| Flag | Meaning |
| --- | --- |
| `--data_dir` (required) | data root; sets `MedVision_DATA_DIR`, `HF_HOME=<data_dir>/.cache/huggingface`, `HF_DATASETS_CACHE=<data_dir>/.cache/huggingface/datasets`, and always `MedVision_FORCE_INSTALL_CODE=true` |
| `--tasks_json` | task list; keys pass through `tasks_to_configs` (appends split, `BoxCoordinate -> BoxSize`; does **not** strip `-CoT`, so use an SFT-style list or `scripts/download_datasets.sh`) |
| `--configs_csv` | CSV, first column = config name, no header, empty rows skipped; exactly one of the two sources |
| `--split {train,test}` | default `test`; must agree with the split baked into each config name |
| `--force_download_data` | sets `MedVision_FORCE_DOWNLOAD_DATA=true` (debug only: every config of the same dataset re-downloads the raw data) |

The CLI calls `load_dataset(..., streaming=False)` once per config. **Any split or config downloads the entire source dataset** (both splits, every plane); `--split` only chooses which Arrow build is materialised. Downloading `_Test` configs is enough to make `_Train` builds local-only later.

`scripts/download_datasets.sh` wraps this with `--dry-run`, an early check of `MedVision_PLANNER_VERSION`, token whitespace trimming, and `-CoT` stripping.

## Download modes and forced refresh

| `download_mode` | Raw data | Arrow dataset |
| --- | --- | --- |
| `reuse_dataset_if_exists` (default) | reuse | reuse |
| `reuse_cache_if_exists` | reuse | fresh |
| `force_redownload` with `MedVision_FORCE_DOWNLOAD_DATA` unset/`False` | reuse | fresh (script re-runs) |
| `force_redownload` with `MedVision_FORCE_DOWNLOAD_DATA=True` | fresh | fresh |

Three levers, from the repository README:

- **[1]** `download_mode="force_redownload"`: ignore cached Arrow files and re-run `MedVision.py`.
- **[2]** `MedVision_FORCE_DOWNLOAD_DATA=True`: force re-download of images, masks, landmarks and annotation plans (also re-fetches QC figures).
- **[3]** delete the dataset's `dataset_<name>` entry from `<data_dir>/.downloaded_datasets.json`: the next script run treats the dataset as never installed.

To update raw data you **must** combine [1] with [2] or [3]: with a valid Arrow cache, `load_dataset` returns rows without running the script, so [2]/[3] are never consulted. Update fields/Arrow only: [1] alone.

Version switches do not need any of this: since **1.2.0** each *resolved* version (and each data root) has its own cache id — 1.1.1 keyed only on the *requested* version, so a `Tumor-Lesion-Size` cache built before 1.2.0 can hold stale rows, and a downgrade reuses on-disk data (annotation zips are cumulative).

## Loader decision on a cold build (`_split_generators`)

1. `_normalize_requested(MedVision_PLANNER_VERSION)` -> error if unset/invalid/unknown.
2. Resolve `(dataset, plan kind)` against the published index: not published at the pin -> error; withdrawn -> error; paused -> error.
3. `MedVision_ACK_RELEASE` check when the pin is below this pair's newest version.
4. Code: download `src/*` and `pip install .` unless tracker has `medvision_ds` / `medvision_ds_installed` equal to the release and `MedVision_FORCE_INSTALL_CODE` is false.
5. Data: `_download_needed(force, tracker_entry, local, target)` = force **or** tracker entry missing **or** no plan on disk resolves for the pin **or** on-disk version `<` index target. Then: 3.1 `Datasets/<name>.zip` (annotations) extracted; 3.2 the dataset's download script (first of `download_debug.py`, `download.py`, `download_fast.py`, `download_raw.py` found) fetches and converts raw data; 3.3 RAS+ reorientation; 3.4 tracker entry written last.
6. QC figures (opt-in) after step 3.

## Environment variables the loader reads

| Variable | Default | Effect |
| --- | --- | --- |
| `MedVision_DATA_DIR` | (required) | data root; absolute path is folded into the cache id |
| `MedVision_PLANNER_VERSION` | (required) | annotation ceiling; `latest` = `1.4.0` |
| `MedVision_ACK_RELEASE` | unset | required when the pin is below a loaded dataset's newest annotation; `1.4.0` acknowledges the whole release |
| `MedVision_FORCE_DOWNLOAD_DATA` | `False` | lever [2]; also forces QC-figure re-fetch |
| `MedVision_FORCE_INSTALL_CODE` | `True` | re-download and reinstall `medvision_ds` on every cold build; set `false` after a first successful install to save minutes per build |
| `MedVision_DISABLE_SAMPLE_FILTERING` | `False` | multi-instance samples; separate cache id |
| `MedVision_DOWNLOAD_QC_FIGURES` | `False` | fetch per-slice QC figures (below) |
| `HF_HOME`, `HF_DATASETS_CACHE` | HF defaults | set by `setup_env_hf` in `medvision_bm` to `<data_dir>/.cache/huggingface[/datasets]` |
| `HF_TOKEN` | unset | needed for private mirrors and gated repos; trim trailing newlines |
| `SYNAPSE_TOKEN` | unset | FeTA24 raw download (`syn25649833`) |
| `MedVision_SKMTEA_HF_ID`, `MedVision_ToothFairy2_HF_ID` | private mirrors | your own HF dataset repo holding `SKM-TEA-nii.zip` / `ToothFairy2.zip` |

## Tracker file `<data_dir>/.downloaded_datasets.json`

Flat JSON, guarded by `.downloaded_datasets.json.lock`:

- `dataset_<name>`: `true` (legacy installs) or the annotation version string; **presence** marks a completed install (written after images are on disk), the version decision itself is made from the plan files present.
- `medvision_ds`, `medvision_ds_installed`: release version of the downloaded/installed codebase.
- `qc_figures_<name>`: biometry version the figures belong to, or `true` for datasets without figures.

The SFT/parquet loaders read this file to drop to a single worker when a dataset was never downloaded, and assert that it exists.

## Restricted and gated datasets

| Dataset | Access | Variable |
| --- | --- | --- |
| FeTA24 | apply on Synapse; the loader downloads with `synapseclient` | `SYNAPSE_TOKEN` (error: `SYNAPSE_TOKEN environment variable not set`) |
| SKM-TEA | apply to the owners, preprocess, upload to your **private** HF dataset repo | `MedVision_SKMTEA_HF_ID` + `HF_TOKEN`; the default id is a private mirror that returns 401 anonymously |
| ToothFairy2 | same as SKM-TEA | `MedVision_ToothFairy2_HF_ID` + `HF_TOKEN` |
| AbdomenAtlas1.0Mini | HF-gated official repo; accept the terms | `HF_TOKEN` and `hf auth login --token $HF_TOKEN --add-to-git-credential` |

Every other dataset needs no credentials.

## QC figures

Per-slice review PNGs (nothing in the loader or any task reads them). Since v1.4.0 they ship as `Datasets/<dataset>_fig.zip` or `<dataset>_fig.partNN.zip` (independent zips, any order) and are fetched only with `MedVision_DOWNLOAD_QC_FIGURES=True`. Size: ~298 GB of PNG against ~3 GB of annotations. Behaviour: checked on **every** load (from `_info()`), so the flag on an existing install fetches figures without re-triggering the data download; figures land at their pre-1.4.0 paths (`Landmarks-Label2-fig-v1.4.0/` etc.); the tracker records the biometry version, so a release that regenerates a dataset's biometry re-fetches its figures and untouched datasets are not re-pulled; a fetch failure only warns. About half the datasets publish no figures; `MedVision_FORCE_DOWNLOAD_DATA=True` retries.

## Disk and time

A full copy is ~1 TB before QC figures. Large detection datasets (AbdomenAtlas1.0Mini, TotalSegmentator, AbdomenCT-1K, AMOS22) dominate. The parquet/SFT loaders warn above 80 % RAM; parallel Arrow generation of large detection configs can exhaust container memory, so start with one worker.
