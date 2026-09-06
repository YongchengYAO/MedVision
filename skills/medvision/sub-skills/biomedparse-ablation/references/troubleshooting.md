# Troubleshooting - BiomedParse ablation

Triage order: (1) run `scripts/check_biomedparse_env.py --python <env python> --ablation-dir <dir>`;
(2) confirm the upstream clone is at `e02096c` and `models/biomedparse_v2.ckpt` is ~4.2 GB; (3) confirm the data
directory and `MedVision_PLANNER_VERSION` / `MedVision_ACK_RELEASE`; (4) reproduce with a smoke test scope before
touching the shipped result folders. Cross-cutting MedVision symptoms live in `../../../references/troubleshooting.md`.

## Setup and environment

| Symptom / error fragment | Likely cause | Fix / validation | Stop when |
|---|---|---|---|
| `pip install ... detectron2` fails: `nvcc: command not found`, `CUDA_HOME` unset, `error: command '/usr/local/cuda/bin/nvcc' failed`, or a CUDA/torch version mismatch (`The detected CUDA version (X) mismatches the version that was used to compile PyTorch (12.4)`) | detectron2 is compiled from source and needs a CUDA 12.4 toolkit matching `torch==2.6.0+cu124` | install/activate a CUDA 12.4 toolkit (`nvcc --version`), export `CUDA_HOME`, keep `--no-build-isolation` so the build sees the pinned torch/ninja; expect 10-20 min | no `nvcc` on the host - needs a GPU node with the toolkit |
| `pip check` after `setup.sh` lists `opencv-python`/`numpy`, `gdrive`/`setuptools`, `build`/`packaging` conflicts | metadata conflicts from the dataset package's own dependencies | harmless for this pipeline (`cv2` and the rest import with the pinned versions); `setup.sh` runs `pip check \|\| true` on purpose | - |
| After `install_medvision_ds`, `lightning 2.3.0 requires packaging<...`, `pip check` conflicts on `opencv-python`, or a pin you set has moved | the dataset-package install can still move pins, but only in two places: its wheel-build step runs `pip install --upgrade build`, which can lift `packaging` past 23.0, and the plain wheel install fills in deps that are *missing or outside* `medvision_ds`'s declared ranges (it declares `opencv-python` while this env pins `opencv-python-headless`, and an exact `datasets==3.6.0`). `huggingface-hub==0.36.0` is inside its declared `>=0.35.3,<2.0` and is left alone, so a hub-related `ImportError` here points at something else in the env, not at this installer. | re-run `pip install -r requirements.txt` (that is why `setup.sh` runs it twice); verify with `check_biomedparse_env.py` (hub must be 0.36.0, packaging 23.0, datasets 3.6.0) | - |
| `conda env 'biomedparse' not found - run setup.sh first` | `_env.sh` could not `conda activate ${ENV_NAME}` | run `setup.sh`, or set `ENV_NAME` to the env you created; check `conda env list` | conda itself is unavailable |
| `Upstream BiomedParse not found at ... Run setup.sh, or point BIOMEDPARSE_DIR at an existing checkout.` | `third_party/BiomedParse` is git-ignored and absent in a fresh clone | run `setup.sh` (clones + checks out `e02096c`) or export `BIOMEDPARSE_DIR=/path/to/BiomedParse` | no network to GitHub |
| upstream checkout is at a different commit | someone pulled/updated the clone | `git -C <BIOMEDPARSE_DIR> checkout e02096c03af0d79c6994ffc2d60a49eeb0361e1f`; the study relies on this commit's `src/`, `configs/model/`, `utils.py`, `inference.py` | - |
| `ModuleNotFoundError: No module named 'medvision_bm'` or `'medvision_ds'` from `src/*.py` | `_paths.add_medvision_to_path()` only adds `<repo>/src` and `<repo>/Data/src`; the ablation folder is outside a checkout, or `medvision_ds` was installed to a different data dir | keep the folder at `<repo>/script/ablation/biomedparse`; install `medvision_ds` with `--data_dir <repo>/Data` or install it into the env; see `../../environment-setup/SKILL.md` | - |
| `finetune.py --help` fails with `No module named 'lightning'` (or `src.datasets...`) | the script imports Lightning and the upstream package at module level | expected outside the `biomedparse` env; read the flag table in `cli-reference.md` instead | - |
| `check_biomedparse_env.py` reports `numpy 2.x` or `transformers 4.5x` | a foreign interpreter (not the `biomedparse` env) was checked | pass `--python /path/to/envs/biomedparse/bin/python` | - |

## Weights and data

| Symptom / error fragment | Likely cause | Fix / validation | Stop when |
|---|---|---|---|
| `huggingface-cli download microsoft/BiomedParse biomedparse_v2.ckpt` fails with 401/403, `GatedRepoError`, or times out | not logged in / no token, or no network | `huggingface-cli login` or export `HF_TOKEN` (strip a trailing newline: `HF_TOKEN=$(tr -d '\n' < token_file)`), retry; the file must end up at `${ABLATION_DIR}/models/biomedparse_v2.ckpt` (~4.2 GB) | no network / credentials - ask the user |
| `pretrained checkpoint missing` from a smoke test, or `ensure_pretrained_ckpt` loops | partial download left a 0-byte or truncated file | delete `models/biomedparse_v2.ckpt` and re-run; check size with `check_biomedparse_env.py` | - |
| T/L prepare/eval aborts inside `medvision_ds` asking to acknowledge the newest release (planner version 1.0.0 pinned) | `MedVision_ACK_RELEASE` not set - `_env.sh` sets it only when `TASK=tl` **before** sourcing | `export TASK=tl` before `source _env.sh`, or `export MedVision_ACK_RELEASE=1.4.0` (must equal the latest release; bump when the dataset is re-released) - see `../../dataset-and-tasks/SKILL.md` | - |
| prepare step: `Dataset <name> not found in DATASETS_NAME2PACKAGE` / `Label <id> not found in labels_map` | task JSON names a dataset/label unknown to the installed `medvision_ds` version | check the `medvision_ds` version and the task JSON; update the data package | - |
| prepare step is slow / hits HF rate limits; `[Warning] Attempt k failed ... retrying` | `_load_single_dataset` retries 5 times with back-off | let it retry; reuse the HF cache; reduce `-p` | persistent network failure |
| eval: `Warning: <base> not found in lookup, skipping` | the NPZ was built from a different annotation version or task JSON than the evaluator reloads | use the same `MedVision_PLANNER_VERSION` / task JSON for prepare and eval; regenerate NPZ | - |
| eval: `Warning: label '<name>' not in label_map_regroup, skipping` | a new label is missing from `medvision_bm.utils.configs.label_map_regroup` | add the label to the regroup map (maintainer change) or accept the exclusion | - |

## Running the tracks

| Symptom / error fragment | Likely cause | Fix / validation | Stop when |
|---|---|---|---|
| `CUDA out of memory` during inference | `--slice_batch_size 4` too large for the GPU | lower `SLICE_BATCH_SIZE` (launcher) / `--slice_batch_size`; it is the only VRAM knob; `--skip_existing` resumes where it stopped | GPU too small even at 1 |
| Inference uses one GPU although `--gpu 0,1` | the upstream model has no DP/DDP path | run separate tasks on separate GPUs (`TASK=detect GPU=0 ...` and `TASK=tl GPU=1 ...`) | - |
| `Using device: cpu` printed by `run_inference.py` | `torch.cuda.is_available()` is False (CPU-only torch or wrong `CUDA_VISIBLE_DEVICES`) | check `nvidia-smi`, `check_biomedparse_env.py` (torch build must be `+cu124`); CPU inference is impractically slow | no GPU |
| Fine-tuning hangs at start or crashes with a device-count mismatch | `N_GPUS=2` in `2_finetune.sh` does not match `CUDA_VISIBLE_DEVICES` | edit `N_GPUS` to the number of visible GPUs (uses `torchrun --nproc_per_node`), or set `CUDA_VISIBLE_DEVICES` to two GPUs | fewer than 2 GPUs and no time to edit the launcher |
| Fine-tuning loss `nan` / error mentioning `edge_loss` | `--edge_coeff 0` | keep `edge_coeff > 0` (upstream loss references `edge_loss` unconditionally; `edge_queries=4` is forced) | - |
| Fine-tuned results differ from the paper | B3 defaults to `last.ckpt`; the paper uses the best-validation `epoch=03` (val_loss 0.446) | `CHECKPOINT=${ABLATION_DIR}/models/finetuned-detect/biomedparse_medvision_epoch=03_val_loss=0.4460.ckpt` for **both** tasks, after clearing `results/<task>/finetuned/seg_masks/` (otherwise `--skip_existing` keeps the old masks) | the checkpoint is not on disk (re-run B2 - hours on 2 GPUs) |
| Switching checkpoints changed nothing | `--skip_existing` reused the existing masks | delete or move `results/<task>/<model>/seg_masks/` first | - |
| Smoke test metrics look terrible | expected: tiny pools (`LIMIT_PER_SUBTASK`), 32-sample 1-epoch model | use smoke tests only for code-path validation; never report their numbers | - |
| Track A smoke ran 100 samples per subtask although the header says 10 | the script's default is `LIMIT_PER_SUBTASK=${LIMIT_PER_SUBTASK:-100}` | pass `LIMIT_PER_SUBTASK=10` explicitly | - |
| Track B smoke fails with `CUDA error: invalid device ordinal` | its default is `GPU=1` | `GPU=0 bash scripts/finetune/smoke_test.sh` | - |
| `SMOKE TEST FAILED - <task>: N masks for M npz samples` | inference skipped/failed some samples (load errors, OOM) | re-run the smoke (masks are reused); inspect the inference log for `WARNING: skipping` | - |

## Re-running one dataset and result folders

| Symptom / error fragment | Likely cause | Fix / validation | Stop when |
|---|---|---|---|
| `run_inference.py --filter_dataset A,B` or `eval_tl.py --filter_dataset A,B` processes nothing | these two accept a **single** name (prefix `<name>__`); only the `prepare_*` scripts accept comma lists | run once per dataset | - |
| Need to re-score one dataset for Detection | `eval_detect.py` has no `--filter_dataset` | re-run inference with `--filter_dataset <name>` (masks replaced), then the full `eval_detect.py` - it rescans `seg_masks/` and rewrites every file | - |
| After `eval_tl.py --filter_dataset`, `summary_tl_task.txt` / `summary_metrics_tl_Task.json` unchanged and `eval_biomedparse_medvision_tl_group_summary.csv` shows only that dataset | by design the filtered run merges only the txt lists, `..._tl_results.csv` and figures; the group-summary CSV is rewritten from the subset and the JSON/TXT are skipped | run the full `eval_tl.py` (no filter) afterwards to refresh the aggregates | - |
| Old T/L figures for a dataset remain in another `MRE0k/` bucket | the MRE changed, so the bucket changed | `eval_tl.py --filter_dataset` deletes stale `<name>__*.png` in all buckets before regenerating; for a full run, clear `figures/tl/<model>/` manually | - |
| Fresh clone has no `results/`, `figures/`, `data/`, `models/`, `third_party/` | all are git-ignored | they are produced by `setup.sh` and the launchers; the shipped numbers are in the paper/leaderboard, not in git | - |
| `_env.local.sh` missing after cloning to a new machine | git-ignored, machine-specific | recreate it with `export MedVision_DATA_DIR=...`, `export BIOMEDPARSE_DIR=...` | - |
| Detection `avgMAE` is `None` for a region | every sample in that region failed (no mask) | expected; `IoU`/`F1`/... are 0 for those samples and `SuccessRate` shows the failure share (see `../../results-parsing-and-metrics/SKILL.md` for the same convention in the benchmark) | - |
