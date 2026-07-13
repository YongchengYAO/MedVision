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

## Batch download: the `download_datasets` CLI

To fetch many datasets ahead of time (data downloading and building is slow), use the CLI instead of scripting `load_dataset()` calls by hand:

```bash
# From a task-list JSON (keys are task names):
python -m medvision_bm.benchmark.download_datasets \
  --tasks_json <task-list.json> \
  --data_dir <data-folder>

# ...or from a configs CSV (config names in the first column):
python -m medvision_bm.benchmark.download_datasets \
  --configs_csv dataset-info/dataset-configs/ConfigurationsList_Test.csv \
  --data_dir <data-folder>
```

Arguments:

- `--data_dir` — **required**; the folder that becomes `MedVision_DATA_DIR` (datasets and the fetched dataset source code land here).
- `--tasks_json` — path to a task-list JSON; its top-level keys are read as task names (the same format used under `tasks_list/`).
- `--configs_csv` — path to a CSV whose first column lists config names. Ready-made lists ship in [`dataset-info/dataset-configs/`](https://github.com/YongchengYAO/MedVision/tree/master/dataset-info/dataset-configs): `ConfigurationsList_All.csv`, `ConfigurationsList_Test.csv`, and `ConfigurationsList_Train.csv`.
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

## Gated source datasets

A few source datasets are access-restricted and need credentials before they will download. Set the relevant tokens in your environment first:

- **FeTA24** — hosted on Synapse behind a data-use agreement; provide `SYNAPSE_TOKEN`.
- **SKM-TEA** — Stanford AIMI knee-MRI released under a data-use agreement, so MedVision serves it from a *gated* Hugging Face mirror rather than fetching it freely; provide `MedVision_SKMTEA_HF_ID` (a mirror repo you have been granted access to) together with `HF_TOKEN`.
- **ToothFairy2** — a gated Hugging Face repo; provide `MedVision_ToothFairy2_HF_ID` and `HF_TOKEN`.

Without these, requesting a config from a gated dataset will fail at download time. See the dataset card for the exact per-dataset access steps.
