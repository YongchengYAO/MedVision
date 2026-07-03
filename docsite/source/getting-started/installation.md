# Installation

This page walks through getting `medvision_bm` onto your machine, pulling in the dataset codebase, and setting the environment variables that the loader and benchmark drivers read at runtime.

There is no user-facing Python API to speak of: the top-level package exposes only `__version__`, and everything you actually run is a `python -m medvision_bm.<...>` command backed by `argparse`. Pick the install path that matches how you intend to use the project.

## Two ways to install the package

### Full clone (benchmarking, SFT, RFT)

Choose this if you plan to run evaluations or fine-tuning. Those workflows depend on files that ship in the repository but are *not* part of the Python package — the launcher scripts under `script/`, the task-list JSONs under `tasks_list/`, and the `Results/` layout that outputs are written into. Cloning gives you that folder structure and installs the package from it in one shot:

```bash
git clone https://github.com/YongchengYAO/MedVision.git MedVision
cd MedVision
pip install .
pip show medvision_bm   # confirm the install
```

Treat the cloned `MedVision/` directory as your working directory from here on. Datasets land in `Data/`, per-model outputs in `Results/<task_tag>/<model>/`, checkpoints in `SFT/<run_name>/`, and resumable status trackers in `completed_tasks/`.

### Import-only (use the modules in your own code)

Choose this if you only want to import from the package — for example pulling in a parsing helper — and do not need the repo's scripts or folder layout. Install straight from GitHub:

```bash
pip install "git+https://github.com/YongchengYAO/MedVision.git"
pip show medvision_bm
```

:::{note}
The GitHub install gives you the importable package only. If you later decide to benchmark or fine-tune, switch to the full clone so the `script/` and `tasks_list/` directories are present.
:::

## Docker

Prebuilt images bundle the CUDA and Python dependencies for specific models, which saves you from resolving them by hand. Browse the available tags on [Docker Hub](https://hub.docker.com/r/vincentycyao/medvision/tags), then pull the one you want:

```bash
docker pull vincentycyao/medvision:<tag>
```

Start a container with GPU access and a bind mount so your work survives container removal. The mount maps a host folder onto the working directory inside the container:

```bash
# Replace </path/to/working/folder> and <tag>
docker run -it --rm \
    --gpus all \
    -v </path/to/working/folder>:/root/Documents/MedVision \
    vincentycyao/medvision:<tag> \
    bash
```

Once inside, clone the repo into the mounted path, activate the image's Conda environment, install the package, and pull in the dataset codebase:

```bash
# In the container
git clone https://github.com/YongchengYAO/MedVision.git /root/Documents/MedVision
cd /root/Documents/MedVision

conda env list          # find the bundled env
conda activate <env-name>

pip install .
pip show medvision_bm

python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
pip show medvision_ds
```

## Dataset codebase and environment setup

`medvision_bm` (the benchmark/training code) is separate from `medvision_ds` (the data-loading code that fetches raw images and builds the dataset). Installing the package does not pull in `medvision_ds`, so install it explicitly. The one required flag is `--data_dir`, the folder where datasets and the loader source will live:

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
```

If you want the full dependency stack in one command — the vendored `lmms_eval`, `medvision_ds`, a CUDA toolkit, vLLM, and any extra requirements file — use `env_setup` instead. It requires both a requirements file (`-r`/`--requirement`) and `--data_dir`:

```bash
python -m medvision_bm.benchmark.env_setup \
    --requirement requirements.txt \
    --data_dir ./Data
```

`env_setup` also accepts `--cuda_version` (default `12.4`), `--vllm_version` (default `0.10.0`), and `--lmms_eval_opt_deps` for optional `lmms_eval` extras. In practice you rarely call it directly — the per-model launchers under `script/benchmark-*/` invoke it for you with the right arguments.

:::{tip}
The benchmark and SFT/RFT launcher scripts run the dataset install and environment setup as their first steps, so if you go through those scripts you usually do not need to run these commands by hand.
:::

## Environment variables

The dataset loader and the benchmark drivers read their configuration from the environment. Set the required ones before running anything; the rest have sensible defaults.

| Variable | Required? | Purpose | Values |
| :-- | :-- | :-- | :-- |
| `MedVision_DATA_DIR` | Yes | Root folder for datasets and the working tree; the full dataset is roughly 1 TB. | Absolute or relative path |
| `MedVision_PLANNER_VERSION` | Yes | Selects which annotation version the loader builds. | `latest` (= `1.1.1`), `1.1.1`, `1.1.0`, `1.0.0` |
| `MedVision_ACK_RELEASE` | Only when pinning below latest | Acknowledges you have read the release note before loading legacy annotations. | `1.1.1` |
| `MedVision_FORCE_INSTALL_CODE` | No | Reinstall `medvision_ds` to the latest on load (keeps you notified of new releases). | `True` (default) / `False` |
| `MedVision_FORCE_DOWNLOAD_DATA` | No | Force re-download of raw images and annotations. | `False` (default) / `True` |
| `HF_TOKEN` | Restricted data only | Hugging Face access token for gated source datasets. | Token string |
| `SYNAPSE_TOKEN` | Restricted data only | Synapse access token for source datasets hosted there. | Token string |
| `MedVision_SKMTEA_HF_ID` | Restricted data only | Hugging Face repo ID granting SKM-TEA access. | `<user>/<repo>` |
| `MedVision_ToothFairy2_HF_ID` | Restricted data only | Hugging Face repo ID granting ToothFairy2 access. | `<user>/<repo>` |

:::{warning}
`MedVision_ACK_RELEASE=1.1.1` is required **only** when you pin `MedVision_PLANNER_VERSION` to a version below the latest (`1.1.0` or `1.0.0`). It confirms you have read the release note explaining what changed — most importantly the corrected tumour/lesion ellipse fit in `1.1.1`. When you run with `latest` (or `1.1.1`) you do not need to set it.
:::

A few source datasets (FeTA24, SKM-TEA, ToothFairy2) sit behind access gates. Only set the restricted tokens above if a subtask you are running touches one of those datasets; everything else downloads without credentials.

## Next steps

With the package installed, `medvision_ds` in place, and the environment variables exported, head to the [Quickstart](quickstart.md) to load a sample and run your first evaluation.
