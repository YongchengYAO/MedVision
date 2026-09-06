# Installation

This page walks through getting `medvision_bm` onto your machine, pulling in the dataset codebase, and setting the environment variables that the loader and benchmark drivers read at runtime.

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

Choose this if you only want to import from the package — for example pulling in a parsing helper — and do not need the repo's scripts or folder layout. Install from PyPI:

```bash
pip install medvision-bm
# or the latest development version from GitHub:
pip install "git+https://github.com/YongchengYAO/MedVision.git"
pip show medvision_bm
```

:::{note}
The PyPI (or GitHub) install gives you the importable package only. If you later decide to benchmark or fine-tune, switch to the full clone so the `script/` and `tasks_list/` directories are present.
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

`medvision_bm` (the benchmark/training code) is separate from `medvision_ds` (the data-loading code that downloads the imaging data and builds the dataset). Installing the package does not pull in `medvision_ds`, so install it explicitly. The one required flag is `--data_dir`, the folder where datasets and the loader source will live:

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
# or the console script that `pip install` puts on PATH:
mvbm install mvds -d ./Data
```

:::{note}
This installs the base `medvision_ds`, which is everything benchmarking and training need — the loader downloads each dataset's preprocessed release from Hugging Face.

Rebuilding a dataset from its *original* sources is a separate path. Those are the `download_raw.py` scripts under `medvision_ds/datasets/`. Only the two DICOM-based ones — LIDC-IDRI and LNQ2023 — need the `raw` extra:

```bash
pip install "medvision_ds[raw]"
```

It adds `pydicom`, `pydicom-seg` and `pylidc`. They are not in the base install because `pydicom-seg` requires `numpy<2`, which cannot be satisfied alongside the numpy 2.x stack the evaluation environments pin. Running a raw script without the extra fails immediately with a message naming this command.
:::

If you want the full dependency stack in one command — the vendored `lmms_eval`, `medvision_ds`, a CUDA toolkit, vLLM, and any extra requirements file — use `env_setup` instead. It requires both a requirements file (`-r`/`--requirement`) and `--data_dir`:

```bash
python -m medvision_bm.benchmark.env_setup \
    --requirement requirements.txt \
    --data_dir ./Data
```

`env_setup` also accepts `--cuda_version` (default `12.4`), `--vllm_version` (default `0.10.0`), and `--lmms_eval_opt_deps` for optional `lmms_eval` extras. It is what the published Docker images run at build time (`dockerfile/Dockerfile.eval_*`).

The launchers under `script/benchmark-*/` do **not** call `env_setup`. They run the equivalent steps by hand and then pass `--skip_env_setup` to the eval driver — this order is load-bearing, so keep it:

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
pip install -r requirements/requirements_eval_qwen25vl.txt --no-deps
```

`install_vendored_lmms_eval` installs the `lmms_eval` fork shipped inside `medvision_bm`; `--lmms_eval_opt_deps` selects the model's optional-dependency group.

:::{tip}
The benchmark and SFT/RFT launcher scripts run the dataset install and environment setup as their first steps, so if you go through those scripts you usually do not need to run these commands by hand.
:::

## Environment variables

The dataset loader and the benchmark drivers read their configuration from the environment. Set the required ones before running anything; the rest have sensible defaults.

| Variable | Required? | Purpose | Values |
| :-- | :-- | :-- | :-- |
| `MedVision_DATA_DIR` | Yes | Root folder for datasets and the working tree; the full dataset is roughly 1 TB. | Absolute or relative path |
| `MedVision_PLANNER_VERSION` | Yes | Selects which annotation version the loader builds. | `latest` (= `1.4.0`), or any published version `1.0.0`–`1.4.0` |
| `MedVision_ACK_RELEASE` | Only when pinning below latest | Acknowledges you have read the release note before loading legacy annotations. | `1.4.0` |
| `MedVision_FORCE_INSTALL_CODE` | No | Reinstall `medvision_ds` to the latest on load (keeps you notified of new releases). | `True` (default) / `False` |
| `MedVision_FORCE_DOWNLOAD_DATA` | No | Force re-download of raw images and annotations. | `False` (default) / `True` |
| `MedVision_DOWNLOAD_QC_FIGURES` | No | Also fetch the per-slice QC figures, which ship in their own archives and are not needed to load any config. Never re-triggers the image/landmark/planner download; figures already on disk are skipped, and re-fetched only when the dataset's biometry annotations are regenerated. | `False` (default) / `True` |
| `MEDVISION_RESP_CACHE` | No | Per-sample response cache that makes an interrupted evaluation resumable; set to `0` to disable and reproduce the plain no-cache behaviour. | enabled by default / `0` = off |
| `HF_TOKEN` | Restricted data only | Hugging Face access token, so the loader can read your own private repo holding restricted data. | Token string |
| `SYNAPSE_TOKEN` | Restricted data only | Synapse authentication token for Synapse-hosted source datasets (e.g. FeTA24). | Token string |
| `MedVision_SKMTEA_HF_ID` | Restricted data only | Your own private Hugging Face repo holding the SKM-TEA data you prepared. | `<user>/<repo>` |
| `MedVision_ToothFairy2_HF_ID` | Restricted data only | Your own private Hugging Face repo holding the ToothFairy2 data you prepared. | `<user>/<repo>` |

:::{warning}
`MedVision_ACK_RELEASE=1.4.0` is required **only** when you pin `MedVision_PLANNER_VERSION` below the newest version published *for the dataset you are loading*. It confirms you have read the release note explaining what changed — most importantly the regenerated tumour/lesion measurements in `1.4.0`. When you run with `latest` (or `1.4.0`) you do not need to set it, and because `1.4.0` changed only the T/L annotations, Angle/Distance and detection workloads load older pins without it.
:::

Three source datasets (FeTA24, SKM-TEA, ToothFairy2) cannot be redistributed, so you apply for access from the data owner and — for SKM-TEA and ToothFairy2 — host the data you prepared in your own private Hugging Face repo. Only set the restricted variables above if a subtask you are running touches one of those datasets; everything else downloads without credentials. See [Restricted source datasets](../dataset/loading.md#restricted-source-datasets) for the full procedure.

## Next steps

With the package installed, `medvision_ds` in place, and the environment variables exported, head to the [Quickstart](quickstart.md) to load a sample and run your first evaluation.
