# Setup

Everything here is a one-time, machine-mutating procedure that **requires** conda, a CUDA 12.4 GPU with `nvcc`
(detectron2 is compiled from source), network access to GitHub and HuggingFace, and a MedVision data directory.
Run it only when the user asked for the ablation and the hardware is confirmed. The CPU-safe pre-flight is
`scripts/check_biomedparse_env.py` (bundled).

## What `setup.sh` does (in order)

`${ABLATION_DIR}` is your copy of the repository's `script/ablation/biomedparse/` folder; the script resolves it
from its own location and derives `REPO_ROOT = ${ABLATION_DIR}/../../..`.

1. Sources `${ABLATION_DIR}/scripts/_env.local.sh` if present; exports
   `MedVision_DATA_DIR=${MedVision_DATA_DIR:-${REPO_ROOT}/Data}`; `ENV_NAME=${ENV_NAME:-biomedparse}`.
2. **Upstream clone (pinned):** `git clone https://github.com/microsoft/BiomedParse.git
   ${ABLATION_DIR}/third_party/BiomedParse` (skipped when `.git` exists), then
   `git checkout e02096c03af0d79c6994ffc2d60a49eeb0361e1f` and prints the short SHA.
3. **Conda env:** `conda create -n ${ENV_NAME} python=3.11 -y` (if missing), `conda activate ${ENV_NAME}`.
4. `pip install -r ${ABLATION_DIR}/requirements.txt` (pins below; extra index
   `https://download.pytorch.org/whl/cu124`).
5. **detectron2** (not on PyPI): if `import detectron2` fails,
   `pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"` - needs `nvcc`,
   ~10-20 min.
6. **medvision_ds:** `PYTHONPATH=${REPO_ROOT}/src python -m medvision_bm.benchmark.install_medvision_ds --data_dir
   ${MedVision_DATA_DIR}` (verified flag: `--data_dir`, required).
7. **Re-pin:** `pip install -r requirements.txt` again - the dataset-package installer only fills in dependencies
   that are missing or outside `medvision_ds`'s declared ranges, but its build step still runs
   `pip install --upgrade build`, which can lift `packaging` past the 23.0 that lightning 2.3 accepts.
   `huggingface-hub==0.36.0` sits inside the declared `>=0.35.3,<2.0` and survives.
8. `pip check || true` - residual conflicts are expected (see `troubleshooting.md`).

Afterwards: `conda activate biomedparse`. The pretrained weights are **not** downloaded by `setup.sh`; every
launcher calls `ensure_pretrained_ckpt` (below) the first time it needs them.

## `requirements.txt` pins (embedded in `scripts/check_biomedparse_env.py`)

| Package | Pin | Notes |
|---|---|---|
| numpy | 1.26.4 | keeps NumPy < 2 (ABI) |
| packaging | 23.0 | re-pinned after medvision_ds |
| setuptools | 65.6.3 | |
| ninja | 1.11.1.1 | detectron2 build |
| torch / torchvision / torchaudio | 2.6.0+cu124 / 0.21.0+cu124 / 2.6.0+cu124 | CUDA 12.4 wheels |
| pandas | 2.2.2 | |
| scikit-learn | 1.4.2 | |
| hydra-core | 1.3.2 | model configs are instantiated with Hydra |
| lightning | 2.3.0 | fine-tuning (Track B) |
| marshmallow | 3.23.2 | |
| timm | 0.9.16 | |
| deepspeed | 0.14.2 | upstream requirement (not used by the launchers) |
| transformers | 4.40.0 | requires huggingface-hub < 1.0 |
| open-clip-torch | 2.26.1 | |
| sentencepiece | 0.2.0 | |
| kornia | 0.7.3 | |
| python-dotenv | 1.0.1 | |
| huggingface-hub | 0.36.0 | `<1.0` for transformers 4.40 / tokenizers 0.19; `>=0.35.3` for medvision_ds |
| datasets | 3.6.0 | HF loading of `*_Test` / `*_Train` configs |
| unpinned | opencv-python-headless, nibabel, Pillow, tqdm, pydicom, SimpleITK, pynrrd, scipy, scikit-image, matplotlib, safetensors, accelerate, psutil | |

`medvision_bm` itself is **not** installed into the env: `src/_paths.py::add_medvision_to_path()` prepends
`${REPO_ROOT}/src` (repository source) and `${REPO_ROOT}/Data/src` (where `install_medvision_ds` places
`medvision_ds`) to `sys.path`. Consequently the ablation folder must live inside a MedVision checkout, and the data
directory used by `install_medvision_ds` must be `${REPO_ROOT}/Data` **or** `medvision_ds` must otherwise be
importable (for example installed in the env). Note that `_paths.py` uses the literal `Data/src` path, not
`MedVision_DATA_DIR`.

## Environment knobs

Every launcher has its settings at the top of the file; the ones below can be given as environment variables
(exported by `scripts/_env.sh`, mirrored in the bundled `scripts/env_template.sh`).

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `TASK` | all task launchers | `detect` | `detect` or `tl`; must be set **before** sourcing `_env.sh` |
| `GPU` | `eval/2_inference.sh`, `finetune/3_inference.sh`, smoke tests | `0` (Track B smoke: `1`) | `CUDA_VISIBLE_DEVICES` for the single-GPU inference (`--gpu`) |
| `CUDA_VISIBLE_DEVICES` | `finetune/2_finetune.sh` | `0,1` | GPUs for DDP training; `N_GPUS=2` in the launcher must match the count |
| `CHECKPOINT` | `finetune/3_inference.sh` | `${ABLATION_DIR}/models/finetuned-detect/last.ckpt` | fine-tuned checkpoint to evaluate |
| `ENV_NAME` | all | `biomedparse` | conda env name |
| `BIOMEDPARSE_DIR` | all | `${ABLATION_DIR}/third_party/BiomedParse` | upstream checkout (set it to reuse an existing clone) |
| `MedVision_DATA_DIR` | all | `${REPO_ROOT}/Data` | the folder holding `Datasets/`, `src/` and `.downloaded_datasets.json` |
| `MedVision_PLANNER_VERSION` | prepare / eval | `1.0.0` | annotation version used in the paper |
| `MedVision_ACK_RELEASE` | prepare / eval, **T/L only** | `1.4.0` | acknowledges the newest T/L annotation release when pinning an older one; must equal the latest release - bump when the dataset is re-released |
| `DATASET`, `TASKS`, `LIMIT_PER_SUBTASK`, `TRAIN_LIMIT`, `VAL_LIMIT`, `EPOCHS` | smoke tests only | see `tracks.md` | scope of the smoke run |

Machine-specific values (`export MedVision_DATA_DIR=/data/MedVision`, `export BIOMEDPARSE_DIR=...`) belong in
`${ABLATION_DIR}/scripts/_env.local.sh`, which is git-ignored and sourced by `setup.sh` and every launcher.

### What `_env.sh` does when sourced

1. Resolves `ABLATION_DIR`, `REPO_ROOT`; sources `_env.local.sh`.
2. Exports `BIOMEDPARSE_DIR`, `MedVision_DATA_DIR`, `MedVision_PLANNER_VERSION`, and `MedVision_ACK_RELEASE`
   when `TASK=tl`.
3. `eval "$(conda shell.bash hook)"; conda activate ${ENV_NAME}` - on failure prints
   `conda env '<name>' not found - run setup.sh first` and exits 1.
4. Defines `PRETRAINED_CKPT=${ABLATION_DIR}/models/biomedparse_v2.ckpt` and
   `ensure_pretrained_ckpt()`: if the file is missing/empty, runs
   `huggingface-cli download microsoft/BiomedParse biomedparse_v2.ckpt --local-dir ${ABLATION_DIR}/models`.

The bundled `scripts/env_template.sh` reproduces this with placeholders and adds `DRY_RUN=1` (print knobs, skip
conda and downloads). Usage:

```bash
# print resolved knobs without side effects
ABLATION_DIR=/path/to/MedVision/script/ablation/biomedparse TASK=tl DRY_RUN=1 bash env_template.sh
# inside your own launcher (activates conda, defines ensure_pretrained_ckpt)
TASK=detect
export ABLATION_DIR=/path/to/MedVision/script/ablation/biomedparse
source /path/to/env_template.sh
ensure_pretrained_ckpt
python "${ABLATION_DIR}/src/run_inference.py" --checkpoint "${PRETRAINED_CKPT}" ...
```

## Pretrained weights

- File: `biomedparse_v2.ckpt`, ~4.2 GB, HF repo `microsoft/BiomedParse`.
- Downloaded on demand by `ensure_pretrained_ckpt` into `${ABLATION_DIR}/models/`. `run_inference.py` also falls
  back to `hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v2.ckpt")` (HF cache, not
  `models/`) when `--checkpoint` is omitted - the launchers always pass `--checkpoint`.
- A gated/unauthenticated download fails with a 401/403 - log in with `huggingface-cli login` or export
  `HF_TOKEN` (strip trailing newlines if it comes from a secret file).

## Data directory prerequisites

The prepare and eval steps read local NIfTI files through the `image_file` / `mask_file` columns of the HF rows,
so the datasets referenced by the task JSONs must already be downloaded into `MedVision_DATA_DIR` with the
MedVision downloader (see `../../dataset-and-tasks/SKILL.md`). `eval_detect.py` / `eval_tl.py` reload **all** rows of
every `*_Test` config (no limit) to build the ground-truth lookup, so the same network/cache access is needed at
evaluation time.

## Pre-flight check (CPU-safe)

```bash
python check_biomedparse_env.py --help
python check_biomedparse_env.py --python /path/to/envs/biomedparse/bin/python \
    --ablation-dir /path/to/MedVision/script/ablation/biomedparse
python check_biomedparse_env.py --ablation-dir ... --json     # machine-readable
```

It reports: installed vs pinned versions (torch/CUDA build, detectron2, lightning, transformers, huggingface-hub,
hydra, numpy, cv2, nibabel, medvision_ds, medvision_bm, plus the optional pins), whether
`third_party/BiomedParse` is at the pinned commit (`git rev-parse HEAD`), `models/biomedparse_v2.ckpt` presence
and size, the data directory markers, `torch.cuda.is_available()`, and which env knobs are set. Exit code 1 means a
required import is missing; version mismatches are warnings.
