# Loading data

MedVision ships as a Hugging Face dataset with a custom loading script. Each *config* corresponds to one task on one source dataset (and one split), and asking for a config triggers download and building of the underlying Arrow files on first use. This page covers the two ways to get data onto disk — direct `load_dataset()` calls and the batch `download_datasets` CLI — plus the environment variables that control versioning and re-downloading.

For what the configs, tasks, and annotation types actually mean, see [Dataset concepts](concepts.md). Once data is present, wire it into a benchmark run via [Running evaluations](../benchmarking/running-evaluations.md).

## Loading a single config

The dataset builder reads its target directory from the `MedVision_DATA_DIR` environment variable, so set that **before** calling `load_dataset()`. Because the loading script runs remote code, pass `trust_remote_code=True`.

```python
import os
from datasets import load_dataset

# Where raw data, caches, and built Arrow files live (a full copy is ~1TB).
os.environ["MedVision_DATA_DIR"] = "/path/to/Data"
os.environ["MedVision_PLANNER_VERSION"] = "latest"

config = "OAIZIB-CM_BoxSize_Task01_Axial_Test"

ds = load_dataset(
    "YongchengYAO/MedVision",
    name=config,
    trust_remote_code=True,
    split="test",  # "test" for *_Test configs, "train" for *_Train configs
)
```

Config names encode the source dataset, annotation type, sub-task, plane, and split (for example `OAIZIB-CM_BoxSize_Task01_Axial_Test`). The `split` argument you pass must match the split baked into the config name.

:::{warning}
Pin `datasets==3.6.0`. The `trust_remote_code` mechanism that MedVision relies on to run its custom builder was removed in `datasets>=4.0.0`, so newer versions cannot load the dataset. This pin is already declared in `medvision_bm`'s dependencies.
:::

:::{note}
Requesting any single config pulls the raw imaging data for that source dataset in full — both the train and test halves — because the builder fetches the underlying archives before slicing. Selecting a `_Test` config does not mean only test-set bytes are downloaded.
:::

## Planner version and the acknowledgement gate

`MedVision_PLANNER_VERSION` selects which release of the annotation logic (the "planner") builds the samples. It is **required** from dataset v1.1.0 onward — loading fails loudly if it is unset. Accepted values:

| Value | Resolves to |
|-------|-------------|
| `latest` | the newest release (currently `1.1.1`) |
| a pinned version — `1.1.1`, `1.1.0`, or `1.0.0` | that exact annotation release |

Different planner versions can change the exact set and framing of samples, so keep this value fixed across a benchmark to stay reproducible.

:::{warning}
Pinning **below** the latest version additionally requires `MedVision_ACK_RELEASE`. Set it to the current latest (`1.1.1`) to acknowledge you have read the release note; without it, loading legacy data is blocked.

```bash
export MedVision_PLANNER_VERSION='1.1.0'
export MedVision_ACK_RELEASE='1.1.1'
```
:::

## Loading unfiltered (multi-instance) samples

By default the loader returns the **single-instance** (filtered) set — the one used for leaderboard comparison. To load the **multi-instance** (unfiltered) set instead, set:

```bash
export MedVision_DISABLE_SAMPLE_FILTERING=true   # default: off
```

This bypasses the per-sample quality/size filters and returns every planner sample (see [Multi-instance vs single-instance annotations](concepts.md#multi-instance-vs-single-instance-annotations) for what those filters drop). Per-version counts for both sets are in [Dataset versions & statistics](statistics.md#benchmark-annotations-by-version).

:::{warning}
Do not use multi-instance annotations to compare models on the leaderboard: the current MedVision-V0 SFT/RFT training is not optimized for multi-instance detection and measurement tasks.
:::

## Batch download: the `download_datasets` CLI

To fetch many datasets ahead of time (data downloading and building is slow), use the CLI instead of scripting `load_dataset()` calls by hand:

```bash
# From a task-list JSON (keys are task names):
python -m medvision_bm.benchmark.download_datasets \
  --tasks_json <task-list.json> \
  --data_dir <data-folder>

# ...or from a configs CSV (config names in the first column):
python -m medvision_bm.benchmark.download_datasets \
  --configs_csv dataset-info/dataset-configs/v1.0.0-v1.1.1/ConfigurationsList_Test.csv \
  --data_dir <data-folder>
```

Arguments:

- `--data_dir` — **required**; the folder that becomes `MedVision_DATA_DIR` (datasets and the fetched dataset source code land here).
- `--tasks_json` — path to a task-list JSON; its top-level keys are read as task names (the same format used under `tasks_list/`).
- `--configs_csv` — path to a CSV whose first column lists config names. Ready-made lists ship in [`dataset-info/dataset-configs/v1.0.0-v1.1.1/`](https://github.com/YongchengYAO/MedVision/tree/master/dataset-info/dataset-configs/v1.0.0-v1.1.1): `ConfigurationsList_All.csv`, `ConfigurationsList_Test.csv`, and `ConfigurationsList_Train.csv`.
- `--split` — `test` (default) or `train`; controls which split of each task/config is requested.
- `--force_download_data` — store-true flag that forces re-download of the raw imaging data.

Provide **exactly one** of `--tasks_json` or `--configs_csv` — supplying neither or both is an error. When you pass tasks, each task name is expanded to a config by appending `_Test`/`_Train`; as part of that expansion `BoxCoordinate` in a task name is rewritten to `BoxSize` to match the dataset's config naming.

:::{warning}
`--force_download_data` is a debugging aid. Because several tasks/configs can share one source dataset, it will re-download the same raw archives repeatedly. Leave it off for normal use.
:::

## Reuse, rebuild, and re-download

Loading has two independent caches, and the controls for each are separate:

- **Built Arrow files** — the cached, ready-to-serve dataset. On a second call with the same config, Hugging Face serves these directly and does **not** run the builder script. Pass `download_mode="force_redownload"` to ignore the cache and re-run the script.
- **Raw imaging data** — the source images, masks, and landmarks. Whether these are re-fetched is decided by:
  - `MedVision_FORCE_DOWNLOAD_DATA` — set to `True` to force re-downloading raw data.
  - `.downloaded_datasets.json` — a tracker file that records which datasets have been fetched. Deleting a dataset's entry causes its raw data to be re-downloaded next time the builder runs.

The catch is that both raw-data controls are only consulted **while the builder runs**. If a valid Arrow cache exists, the script is skipped and neither the env var nor the tracker is checked. So the rule is:

- To rebuild only the built fields: `download_mode="force_redownload"`.
- To refresh raw data: `download_mode="force_redownload"` **and** either `MedVision_FORCE_DOWNLOAD_DATA=True` or an edited `.downloaded_datasets.json`.

```python
import os
from datasets import load_dataset

os.environ["MedVision_DATA_DIR"] = "/path/to/Data"
os.environ["MedVision_PLANNER_VERSION"] = "latest"
os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "True"

ds = load_dataset(
    "YongchengYAO/MedVision",
    name="OAIZIB-CM_BoxSize_Task01_Axial_Test",
    trust_remote_code=True,
    split="test",
    download_mode="force_redownload",  # required, or the raw-data flags are never read
)
```

Note that the CLI's `--force_download_data` maps to `MedVision_FORCE_DOWNLOAD_DATA`; it forces the builder to run and re-fetch raw data for every requested item.

## Restricted source datasets

Three source datasets **do not allow redistribution**, so MedVision cannot ship or mirror them. You apply for access from the data owner yourself, then point MedVision at your own copy.

**FeTA24** only needs a token — it is hosted on Synapse behind a registration agreement:

```bash
export SYNAPSE_TOKEN=<your-synapse-token>
```

**SKM-TEA and ToothFairy2** need one extra step. Once access is granted, you download and process the raw data yourself, upload the *preprocessed* data to **your own private Hugging Face dataset repo**, and set the env var to that repo:

| Dataset | Apply for access at | Env var |
|---|---|---|
| SKM-TEA | <https://aimi.stanford.edu/datasets/skm-tea-knee-mri> | `MedVision_SKMTEA_HF_ID` |
| ToothFairy2 | <https://ditto.ing.unimore.it/toothfairy2/> | `MedVision_ToothFairy2_HF_ID` |

`HF_TOKEN` is what lets the loader read that private repo, so authenticate as well:

```bash
export HF_TOKEN=<your-hf-token>
hf auth login --token $HF_TOKEN --add-to-git-credential

export MedVision_SKMTEA_HF_ID=<your-user>/<your-private-repo>
export MedVision_ToothFairy2_HF_ID=<your-user>/<your-private-repo>
```

Step-by-step preparation guides live on the dataset card: [restricted datasets overview](https://huggingface.co/datasets/YongchengYAO/MedVision#datasets), [prepare SKM-TEA](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/dataset_skm-tea.md), [prepare ToothFairy2](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/dataset_toothfairy2.md).

Without these, requesting a config from a restricted dataset will fail at download time. The other 19 datasets need no credentials.
