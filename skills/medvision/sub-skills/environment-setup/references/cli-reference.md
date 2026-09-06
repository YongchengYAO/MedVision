# Environment-Setup CLI and Python API Reference

Every command below is a public entry point of the installed `medvision_bm` package. Help texts were captured verbatim from `medvision_bm` 1.2.0 on Python 3.11 (`--help` exits 0 for all of them). The commands are thin wrappers over `medvision_bm.utils.install_utils`; the Python API section gives the exact signatures for scripting the same steps.

All of these commands **mutate the active Python environment** (pip installs, and for `env_setup` a conda install) and most **download from Hugging Face**. Read `installation.md` before running them, and never run them inside an environment whose pins you have not checked with `../scripts/check_env_pins.py`.

## `mvbm` (console script)

```text
usage: mvbm [-h] {install} ...

Shortcut commands for the MedVision benchmark.

positional arguments:
  {install}
    install   Install MedVision components.

options:
  -h, --help  show this help message and exit
```

```text
usage: mvbm install [-h] {mvds} ...

positional arguments:
  {mvds}
    mvds      Install the medvision_ds dataset codebase (alias of `python -m
              medvision_bm.benchmark.install_medvision_ds`).

options:
  -h, --help  show this help message and exit
```

```text
usage: mvbm install mvds [-h] -d DATA_DIR

options:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Directory to store downloaded datasets and source
                        code.
```

`mvbm install mvds -d <data_dir>` creates `<data_dir>` and calls `install_medvision_ds(<data_dir>)`. The heavy imports (torch, datasets) are deferred, so `mvbm --help` works in a minimal environment.

## `python -m medvision_bm.benchmark.install_medvision_ds`

```text
usage: install_medvision_ds.py [-h] --data_dir DATA_DIR

Install Python packages from a requirements file.

options:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Directory to store downloaded datasets and source code.
```

(The description string is inherited boilerplate; the command installs the `medvision_ds` dataset package, not a requirements file.) Runs: `os.makedirs(data_dir)` then `install_medvision_ds(data_dir)`.

## `python -m medvision_bm.benchmark.install_vendored_lmms_eval`

```text
usage: install_vendored_lmms_eval.py [-h]
                                     [--lmms_eval_opt_deps LMMS_EVAL_OPT_DEPS]

Install the vendored lmms_eval package.

options:
  -h, --help            show this help message and exit
  --lmms_eval_opt_deps LMMS_EVAL_OPT_DEPS
                        Optional dependencies for lmms_eval installation.
```

Runs `install_vendored_lmms_eval(proj_dependency=<opt_deps>)` (or without the extra when the flag is omitted). Valid extras are the `[project.optional-dependencies]` keys of the vendored `lmms_eval` pyproject; see the table in `installation.md`.

## `python -m medvision_bm.benchmark.env_setup`

```text
usage: env_setup.py [-h] -r REQUIREMENT --data_dir DATA_DIR
                    [--lmms_eval_opt_deps LMMS_EVAL_OPT_DEPS]
                    [--cuda_version CUDA_VERSION]
                    [--vllm_version VLLM_VERSION] [--skip_cuda_toolkit]
                    [--skip_vllm]

Install Python packages from a requirements file.

options:
  -h, --help            show this help message and exit
  -r REQUIREMENT, --requirement REQUIREMENT
                        Path to the requirements.txt file.
  --data_dir DATA_DIR   Directory to store downloaded datasets and source
                        code.
  --lmms_eval_opt_deps LMMS_EVAL_OPT_DEPS
                        Optional dependencies for lmms_eval installation.
  --cuda_version CUDA_VERSION
                        CUDA toolkit version to install (default: 12.4).
  --vllm_version VLLM_VERSION
                        vLLM version to install (default: 0.10.0).
  --skip_cuda_toolkit   Skip installing the CUDA toolkit (for API-only models
                        with no local GPU inference).
  --skip_vllm           Skip installing vLLM (for models that don't use it).
```

Execution order (verified from source):

1. `install_vendored_lmms_eval(proj_dependency=<opt_deps>)` — editable install of the vendored `lmms_eval` (with the extra when given).
2. `install_medvision_ds(<data_dir>)` — download `src/*` of the HF dataset repo, build the wheel, two-step install, set env vars.
3. unless `--skip_cuda_toolkit`: `install_cuda_toolkit(version=<cuda_version>)` → `conda install -c nvidia cuda-toolkit=<ver> -y` then `setup_env_cuda()` (needs `conda` on PATH and an active conda env).
4. unless `--skip_vllm`: `install_vllm(data_dir, version=<vllm_version>)` → `pip install blobfile`, `pip install vllm==<ver>`, then `setup_env_vllm(data_dir)`.
5. `run_pip_install(<requirement>)` → `python -m pip install --upgrade --force-reinstall --no-deps -r <requirement>` (the requirements file is applied **last**, so its pins win).

API-only models (Claude, Gemini, GPT, Kimi) are installed with `--skip_cuda_toolkit --skip_vllm`.

## `python -m medvision_bm.sft.env_setup`

```text
usage: env_setup.py [-h] --data_dir DATA_DIR [-r REQUIREMENT]
                    [--lmms_eval_opt_deps LMMS_EVAL_OPT_DEPS]

Install all dependencies for SFT on MedVision datasets.

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Data directory path
  -r REQUIREMENT, --requirement REQUIREMENT
                        Path to the requirements.txt file.
  --lmms_eval_opt_deps LMMS_EVAL_OPT_DEPS
                        Optional dependencies for lmms_eval installation.
```

Two code paths (verified from source):

- **With `-r`**: `install_vendored_lmms_eval(...)` → `run_pip_install(<requirement>)` (force-reinstall, no-deps) → `install_medvision_ds(<data_dir>)`. Note that `medvision_ds` is installed **after** the requirements here, so its plain-install step may add missing dependencies (but the two-step installer leaves an already-satisfied `huggingface_hub` alone).
- **Without `-r`**: `install_vendored_lmms_eval(...)` → `install_basic_packages()` (`pip install --upgrade pip`; `datasets==3.6.0 numpy==1.26.4 protobuf==3.20 wandb==0.21.4 trl==0.19.1 huggingface_hub==0.36.0`; then `-U bitsandbytes peft hf_xet tensorboard nibabel scipy Pillow accelerate`) → `install_flash_attention_torch_and_deps_py311_v2()` (torch/torchvision/torchaudio 2.6.0+cu124 from the cu124 index with `--force-reinstall`, the `flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311` wheel, `numpy==1.26.4`, `protobuf==3.20`) → `install_transformers("4.54.0")` → `install_medvision_ds(<data_dir>)`. This path assumes **Python 3.11** (the flash-attn wheel is cp311) and a CUDA 12.4 host.

The SFT launchers follow either path with `python -m pip install "protobuf==6.33.0"` because the `protobuf==3.20` left behind breaks `wandb>=0.21` (`cannot import name 'Imports' from wandb.proto...`).

## Python API (`medvision_bm.utils.install_utils`, re-exported by `medvision_bm.utils`)

Signatures verified with `inspect.signature`:

| Function | Signature | What it does |
| --- | --- | --- |
| `install_medvision_ds` | `(data_dir, local_dir=None)` | `local_dir=None`: `snapshot_download(repo_id="YongchengYAO/MedVision", repo_type="dataset", allow_patterns="src/*", local_dir=data_dir)`; else use `<local_dir>/src`. Then under `flock -w 600 <src>/.build.lock` (fallback: no lock): `rm -rf build/ dist/ medvision_ds.egg-info`, `rm -f wheels/*.whl`, `pip install --upgrade build`, `python -m build --wheel --outdir wheels/ <src>`, `pip install --no-cache-dir <wheel>` (fills missing deps only), `pip install --no-cache-dir --force-reinstall --no-deps <wheel>` (refreshes code). Finally `setup_env_hf_medvision_ds(data_dir)`. |
| `install_vendored_lmms_eval` | `(editable_install=True, proj_dependency=None)` | Locates `medvision_bm/medvision_lmms_eval` via `importlib.resources.files`, then `flock -w 600 <dir>/.build.lock` + `python -m pip install --no-cache-dir --force-reinstall -e .[<extra>]`. Editable is required so the task YAMLs are discovered. Non-editable without an extra builds a wheel into `<dir>/wheels/`; non-editable with an extra installs from source. |
| `install_lmms_eval` | `(benchmark_dir, lmms_eval_folder, editable_install=False, proj_dependency=None)` | Same installer for a non-vendored `lmms_eval` checkout at `<benchmark_dir>/<lmms_eval_folder>`. |
| `setup_env_hf_medvision_ds` | `(data_dir, force_install_code=True, force_download_data=False)` | `setup_env_medvision_ds(...)` then `setup_env_hf(data_dir)`. Called unconditionally at the top of every `eval__<model>.main()`. |
| `setup_env_medvision_ds` | `(data_dir, force_install_code=True, force_download_data=False)` | `os.makedirs(abspath(data_dir))`; `MedVision_DATA_DIR=<abs data_dir>`; sets `MedVision_FORCE_INSTALL_CODE=true` / `MedVision_FORCE_DOWNLOAD_DATA=true` only when the flags are `True` (never unsets them). |
| `setup_env_hf` | `(data_dir)` | `HF_HOME=<abs data_dir>/.cache/huggingface`, `HF_DATASETS_CACHE=<abs data_dir>/.cache/huggingface/datasets`. |
| `ensure_hf_hub_installed` | `(hf_hub_version='0.35.3')` | `pip install huggingface_hub[cli]==<ver>` only if `from huggingface_hub import snapshot_download` fails (no-op when any hub is importable). |
| `run_pip_install` | `(requirements_path)` | `python -m pip install --upgrade --force-reinstall --no-deps -r <file>` with `PIP_DISABLE_PIP_VERSION_CHECK=1`; raises `FileNotFoundError` for a missing file. |
| `install_cuda_toolkit` | `(version='12.4')` | `conda install -c nvidia cuda-toolkit=<ver> -y` then `setup_env_cuda()`. |
| `setup_env_cuda` | `()` | `CUDA_HOME=$CONDA_PREFIX`; prepends `$CUDA_HOME/bin` to `PATH` and `$CUDA_HOME/lib64`, `$CUDA_HOME/lib` to `LD_LIBRARY_PATH`. |
| `install_torch_cu124` | `()` | `pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall`, then `setup_env_cuda()`. |
| `install_torch_cu121` | `()` | Same with `torch==2.5.0+cu121 / torchvision==0.20.0+cu121 / torchaudio==2.5.0+cu121`. |
| `install_vllm` | `(data_dir, version='0.10.0')` | `pip install blobfile`; `pip install vllm==<ver>`; `setup_env_vllm(data_dir)`. Raises `RuntimeError` on failure. |
| `setup_env_vllm` | `(data_dir)` | `VLLM_WORKER_MULTIPROC_METHOD=spawn`; `XDG_CACHE_HOME=<abs data_dir>/.cache/vllm`. |
| `install_flash_attention_torch_and_deps_py311_v2` | `()` | `install_torch_cu124()`; `pip install <flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311 wheel URL>`; `numpy==1.26.4`; `protobuf==3.20`. `py39`/`py310` variants and the non-`_v2` variants (which also `conda install` cuda-toolkit 12.4.0, cuda-nvcc, cudnn and the `nvidia-cuda-*-cu12==12.4.*` wheels) exist under the same naming pattern. |
| `pip_install_medvision_ds` | `()` | `pip install "git+https://huggingface.co/datasets/YongchengYAO/MedVision.git#subdirectory=src"` (prints, does not raise). |
| `pip_install_medvision_bm` | `()` | `pip install "git+https://github.com/YongchengYAO/MedVision.git"` (prints, does not raise). |

Example of scripting the dataset-package install from an existing local source tree (no download):

```python
from medvision_bm.utils import install_medvision_ds
install_medvision_ds("<data_dir>", local_dir="<dir-containing-src>")  # installs <dir-containing-src>/src
```

## Related flags on the eval wrappers

Every `python -m medvision_bm.benchmark.eval__<model>` wrapper accepts `--skip_env_setup` (skips the guarded install block — but `eval__healthgpt` still runs `install_healthgpt_dependencies_post` and `install_flash_attention_torch_and_deps_py311_v2()` outside it, force-reinstalling torch 2.6.0+cu124; still sets the env vars and, for vLLM models, `setup_env_vllm`), `--env_setup_only` (run the installs, then exit before evaluating) and `--skip_update_status` — except `eval__qwen25vl_tooluse`, which has no `--env_setup_only`. Their semantics and the per-model install order are described in `installation.md`; running evaluations is covered by `../../benchmark-evaluation/SKILL.md`.
