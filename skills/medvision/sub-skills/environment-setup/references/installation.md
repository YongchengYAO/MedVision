# Installation Workflows

This reference walks through installing the MedVision stack on a fresh machine in the order the repository itself uses: the benchmark package `medvision_bm`, the dataset package `medvision_ds`, the vendored evaluation engine `lmms_eval`, and finally a per-model pinned stack (torch / vLLM / transformers). Exact help texts and Python signatures are in `cli-reference.md`; per-model pins in `requirements-catalog.md`; variables in `environment-variables.md`; failures in `troubleshooting.md`.

Conventions: `<repo>` = a clone of the MedVision GitHub repository; `<data_dir>` = the MedVision data directory (datasets, `src/`, caches; the repository uses `<repo>/Data`); `<model>` = a requirements key from `requirements-catalog.md`.

## 0. Prerequisites and hardware

- Python `>=3.9` (package metadata). The PyPI publish workflow, most launchers and most Dockerfiles use **Python 3.11**, and the SFT flash-attn wheel is cp311-only — so use 3.11 **unless** the model is one of three families that deliberately pin another interpreter: **MedDr 3.9**, **LLaVA-Med 3.10**, and **MiniMax-M3 / MiniMax-M3-INT4 3.12** (12 launchers in total, plus the matching `Dockerfile.eval_*`). Their frozen requirements pin cp39/cp310 flash-attn wheels or a py3.12-only stack, so installing them into a 3.11 environment fails with "not a supported wheel on this platform".
- Linux x86-64. GPU evaluation and training assume NVIDIA GPUs with a **CUDA 12.4** userland (`install_cuda_toolkit(version="12.4")`, `install_torch_cu124`). Per-model VRAM/GPU-count expectations: `../../../references/model-roster.md`.
- CPU-only hosts can install `medvision_bm` (the default `torch==2.6.0` wheel), run `mvbm`, `parse_outputs`, the `summarize_*` scripts and dataset inspection; they cannot run open-weight evaluation, SFT/RFT or the LLM judge.
- Tools used by the installers: `pip`, `flock` (util-linux), `tar`, `git`; `conda` only for `env_setup`'s CUDA-toolkit step and the launcher-style per-model environments.
- Network access to PyPI, GitHub and Hugging Face; `HF_TOKEN` for gated/private sources (see `environment-variables.md`).

## 1. Install `medvision_bm` (three ways)

Pinned base dependencies (pyproject): `datasets==3.6.0`, `huggingface_hub==0.36.0`, `torch==2.6.0`, `torchvision==0.21.0`, `accelerate==1.9.0`, `psutil==7.2.2`, plus unpinned `scipy`, `nibabel`, `matplotlib`. The package ships the vendored `medvision_lmms_eval/**` tree as package data; the console script is `mvbm`. There is no packaged SFT config directory — training configuration is passed as CLI flags and accelerate/FSDP config files (see the `sft` sub-skill).

| Path | Command | When |
| --- | --- | --- |
| PyPI (stable) | `pip install medvision-bm` | you only `import medvision_bm` in your own code and do not need `script/`, `tasks_list/`, `requirements/` |
| Local checkout | `git clone https://github.com/YongchengYAO/MedVision.git <repo> && cd <repo> && pip install .` | full pipeline (benchmark, SFT, RFT); the launchers rely on the repository layout |
| Local editable | `cd <repo> && pip install -e . --no-deps` | you edit `src/medvision_bm/` and want edits picked up without reinstalling; `--no-deps` keeps an already-pinned stack untouched (add deps separately if the env is fresh). PEP 660 editable installs need setuptools>=64 in the build environment — keep build isolation on (do **not** pass `--no-build-isolation` with an old setuptools) |
| Nightly (GitHub master) | `pip install "git+https://github.com/YongchengYAO/MedVision.git"` | latest commit without cloning |

Releases are published to PyPI from GitHub releases (non-prerelease) by a workflow that builds sdist+wheel with Python 3.11; the version is `medvision_bm.__version__` (1.2.0 at generation time).

Verify (all three checks, every time):

```bash
pip show medvision_bm                       # Version + Location; an editable install also prints "Editable project location"
python -c "import medvision_bm; print(medvision_bm.__version__, medvision_bm.__file__)"
mvbm --help
```

**Editable-install trap.** `pip install -e .` and a later plain `pip install .` (or a wheel install from the launcher wheel-build block) do not coexist: the plain install replaces the editable `.pth` link with a copy under `site-packages/medvision_bm/`, and from then on edits under `<repo>/src/` silently do nothing. Diagnose with `medvision_bm.__file__` — it must point into `<repo>/src/medvision_bm/`, not into `site-packages`. Fix: `pip install -e . --no-deps` again, or for a one-off run force the source tree first: `PYTHONPATH=<repo>/src python -m medvision_bm...`. `../scripts/check_env_pins.py` prints both `medvision_bm.__file__` and `medvision_ds.__file__` and flags site-packages locations. The same trap applies to `medvision_ds` (its installer always produces a non-editable copy).

## 2. Install the dataset package `medvision_ds`

`medvision_ds` is not on PyPI; its source lives in the `src/` folder of the Hugging Face dataset repo `YongchengYAO/MedVision`. Three equivalent entry points call the same function:

```bash
mvbm install mvds -d <data_dir>
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>
python -c "from medvision_bm.utils import install_medvision_ds; install_medvision_ds('<data_dir>')"
```

What `install_medvision_ds(data_dir, local_dir=None)` does, in order:

1. `os.makedirs(abspath(data_dir))`, then `snapshot_download(repo_id="YongchengYAO/MedVision", repo_type="dataset", allow_patterns="src/*", local_dir=<data_dir>)` → the source tree lands in `<data_dir>/src` (with `local_dir=<dir>` it uses `<dir>/src` and downloads nothing).
2. Under `flock -w 600 <data_dir>/src/.build.lock` (falls back to no lock with a warning if `flock` fails): remove `build/`, `dist/`, `medvision_ds.egg-info/`, `wheels/*.whl`; `pip install --upgrade build`; `python -m build --wheel --outdir <data_dir>/src/wheels <data_dir>/src`.
3. **Two-step install**: `pip install --no-cache-dir <wheel>` (only fills in dependencies that are *missing or outside* the declared ranges — an installed `huggingface_hub` inside `medvision_ds`'s declared `>=0.35.3,<2.0` is left alone, while a hub **below** the 0.35.3 floor is resolved up to the newest in-range release, i.e. 1.x), then `pip install --no-cache-dir --force-reinstall --no-deps <wheel>` (refreshes the package code without re-resolving anything). A single bare `--force-reinstall` would re-resolve every declared dependency to the newest in-range version (observed: `huggingface_hub` 0.36.0 → 1.29.0, which transformers 4.x rejects at import) — the frozen per-model requirements must stay in charge.
4. `setup_env_hf_medvision_ds(data_dir)` → exports `MedVision_DATA_DIR=<abs data_dir>`, `MedVision_FORCE_INSTALL_CODE=true`, `HF_HOME=<abs data_dir>/.cache/huggingface`, `HF_DATASETS_CACHE=<abs data_dir>/.cache/huggingface/datasets` (in the current process only — export them yourself for later shells).

`medvision_ds` dependencies worth knowing (its pyproject): `huggingface_hub>=0.35.3,<2.0` (deliberately a floor, because the dataset loader reinstalls the package inside live processes and no single version serves transformers 4.x and 5.x), `datasets==3.6.0`, `opencv-python` (non-headless: needs `libGL.so.1`), `gdrive` (pins `setuptools~=59.6.0`), `gdown`, `synapseclient`, `SimpleITK`, `scikit-image`, `pynrrd`, `rarfile`, `py7zr`, `pandas`. Optional extra `medvision_ds[raw]` (`pydicom<3`, `pydicom-seg`, `pylidc`, `setuptools<81`) is only for rebuilding datasets from original sources.

Alternative without the wheel build: `pip install "git+https://huggingface.co/datasets/YongchengYAO/MedVision.git#subdirectory=src"` (this is what `pip_install_medvision_ds()` runs); it does **not** set the environment variables and does not populate `<data_dir>/src`.

Verify:

```bash
pip show medvision_ds
python -c "import medvision_ds; print(medvision_ds.__version__, medvision_ds.__file__)"   # 1.4.0 at generation time
python -c "import huggingface_hub, transformers; print(huggingface_hub.__version__, transformers.__version__)"  # hub<1.0 with tf 4.x, >=1.5 with tf 5.x
```

Re-pin after the install when needed: `pip install "setuptools>=60,<81"` (undo the `gdrive`-induced downgrade; `<81` keeps `pkg_resources` for `pylidc`), and — if your model stack is transformers 4.x and hub moved — `pip install "huggingface_hub==0.36.0"` (or the exact version in the model's requirements file).

**The loader reinstalls `medvision_ds` too.** Independently of the command above, `load_dataset("YongchengYAO/MedVision", ...)` runs the dataset script `MedVision.py`, which — when `MedVision_FORCE_INSTALL_CODE` is true (the default, and what `setup_env_hf_medvision_ds` sets) or when `.downloaded_datasets.json` does not record the current release as installed — re-downloads `src/*` and runs `pip install .` in `<data_dir>/src` under `<data_dir>/src/.build.lock`, then `importlib.invalidate_caches()`. That in-tree `pip install .` fills missing dependencies like the first step above, so keep your pins applied *after* it or set `MedVision_FORCE_INSTALL_CODE=false` after an explicit install (see `environment-variables.md`).

## 3. Install the vendored evaluation engine `lmms_eval`

```bash
python -m medvision_bm.benchmark.install_vendored_lmms_eval                          # no extra
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen2_5_vl
```

`install_vendored_lmms_eval(editable_install=True, proj_dependency=None)` locates the `medvision_lmms_eval` folder **inside the installed `medvision_bm` package** and runs `python -m pip install --no-cache-dir --force-reinstall -e .[<extra>]` under `flock -w 600 <folder>/.build.lock`. Editable mode is mandatory: the MedVision task YAMLs are only discovered from the source tree. Consequences:

- The engine imports as `lmms_eval` and provides `python -m lmms_eval` / `lmms-eval`, which the eval wrappers spawn as a subprocess.
- Because it is editable *inside* `site-packages/medvision_bm/medvision_lmms_eval` (or inside `<repo>/src/medvision_bm/...` for an editable `medvision_bm`), **reinstalling `medvision_bm` invalidates it** — rerun this step after every `pip install .`/wheel install of `medvision_bm`. The eval wrappers do this automatically unless `--skip_env_setup` is passed; "Method 1" launchers run it explicitly.
- Passing an extra installs its pins (`transformers==4.54.1` for `qwen2_5_vl`, `transformers==5.12.1` for `glm4v`, ...; full table in `requirements-catalog.md`). Choose the extra matching your model; omit it for models without one (gemma3/gemma4/healthgpt/internvl3/llama3_vision/llava_onevision/medgemma).
- Base dependencies are broad (`torch>=2.1.0,<2.8`, `torchvision>=0.16.0,<0.23`, `datasets==3.6.0`, unpinned `transformers`, `deepspeed`, `bitsandbytes`, `opencv-python-headless`, ...). With `--force-reinstall` pip re-resolves them, so run this **before** applying the frozen requirements file, never after.

Verify: `python -c "import lmms_eval, pathlib; print(pathlib.Path(lmms_eval.__file__).parent)"` and `python -m lmms_eval --help` (needs torch importable).

## 4. Per-model stacks: automatic vs manual

### 4a. Order of operations inside every `eval__<model>.py` (the automatic path)

Every eval wrapper except `eval__qwen25vl_tooluse` has a `main()` block marked `DO NOT change the order of these calls`; the tool-use wrapper installs only vLLM + transformers/accelerate, so the engine and dataset package must already be present. For `eval__qwen2_5_vl` it is:

```python
setup_env_hf_medvision_ds(data_dir)                       # always: env vars only
if not args.skip_env_setup:
    ensure_hf_hub_installed(hf_hub_version="0.35.3")      # no-op if any huggingface_hub imports
    install_vendored_lmms_eval(proj_dependency="qwen2_5_vl")
    install_medvision_ds(data_dir)
    install_torch_cu124()                                 # torch 2.6.0+cu124 (--force-reinstall)
    install_vllm(data_dir, version="0.10.0")              # pulls vLLM's own torch (2.7.1) + transformers
    install_transformers_accelerate_for_qwen25vl(transformers_version="4.54.1", accelerate_version="1.9.0")
    if args.env_setup_only: return
else:
    setup_env_vllm(data_dir)                              # VLLM_WORKER_MULTIPROC_METHOD=spawn, XDG_CACHE_HOME
```

The order is load-bearing: `lmms_eval` first (broad resolver), then `medvision_ds` (two-step, leaves hub alone), then torch, then vLLM (which replaces torch with its ABI-matched build), then the **model-specific transformers/accelerate reinstall last** so it overrides whatever vLLM resolved (Gemma 4 and GLM-4.6V install transformers>=5.5 / 5.12.1 *after* vLLM 0.19.x precisely to override vLLM's `transformers<5` pin). API wrappers (`eval__claude`, `eval__gemini`, `eval__openai`, `eval__kimi`) stop after `install_medvision_ds`; HF-inference wrappers skip vLLM; `eval__medgemma` installs no torch either, but `eval__healthgpt`, `eval__huatuogpt_vision`, `eval__lingshu`, `eval__llava_med` and `eval__meddr` each call an `install_flash_attention_torch_and_deps_py3{9,10,11}_v2` helper that force-reinstalls torch 2.6.0+cu124 with a matching flash-attn wheel. Per-wrapper values: `requirements-catalog.md`.

Flags: `--env_setup_only` runs the block and exits (use it to build an environment once); `--skip_env_setup` skips the guarded install block and still exports the env vars and vLLM settings (note `eval__healthgpt` installs outside that block, so it is not fully skipped) (use it after a manual install; prints a warning). Running with `--skip_env_setup` **and** no manual install leaves nothing set up.

### 4b. Method 1 — frozen requirements (recommended, what the launchers do)

```bash
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps <extra>   # omit the flag if the model has no extra
pip install -r <repo>/requirements/requirements_eval_<model>.txt --no-deps
python -m medvision_bm.benchmark.eval__<module> --skip_env_setup ...                       # requires GPU for open-weight models
```

All three install lines are load-bearing and ordered: (1) also wipes stale `build/`/`dist/` in `<data_dir>/src` so the loader's later in-tree `pip install .` does not fail with `[Errno 17] File exists: 'build/bdist.linux-x86_64/wheel/...'`; (2) restores the editable engine after any `medvision_bm` reinstall; (3) applied last with `--no-deps` so the frozen torch/torchvision/vllm/transformers/hub pins win. Then confirm with `../scripts/check_env_pins.py --requirements <repo>/requirements/requirements_eval_<model>.txt` (exit 0 = clean).

### 4c. Method 2 — `env_setup` (automatic, simpler, more fragile)

```bash
python -m medvision_bm.benchmark.env_setup -r <repo>/requirements/requirements_eval_<model>.txt \
    --lmms_eval_opt_deps <extra> --data_dir <data_dir> [--cuda_version 12.4] [--vllm_version 0.10.0]
# API-only models:
python -m medvision_bm.benchmark.env_setup -r <repo>/requirements/requirements_eval_claude.txt \
    --lmms_eval_opt_deps claude --skip_cuda_toolkit --skip_vllm --data_dir <data_dir>
```

Order: vendored `lmms_eval` → `medvision_ds` → `conda install -c nvidia cuda-toolkit=<ver>` (unless skipped; needs conda) → `pip install blobfile vllm==<ver>` (unless skipped) → `pip install --upgrade --force-reinstall --no-deps -r <file>`. Pass the vLLM version that matches the requirements file (e.g. `--vllm_version 0.11.0` for `qwen3vl`, `0.19.0` for `gemma4`); the default `0.10.0` is right only for the torch-2.7.1 rows. Then run the wrapper with `--skip_env_setup`. 17 of the 23 model Dockerfiles are built with exactly this command; the five flash-attn images (`healthgpt`, `huatuogptvision`, `lingshu`, `llavamed`, `meddr`) and `minimax-m3-int4` use Method 1 (`pip install -r requirements/... --no-deps`) instead.

### 4d. SFT environments

```bash
python -m medvision_bm.sft.env_setup --data_dir <data_dir> --lmms_eval_opt_deps qwen2_5_vl                                   # no -r: installs a fixed cp311 stack (torch 2.6.0+cu124, flash-attn 2.7.3, transformers 4.54.0, protobuf 3.20)
python -m medvision_bm.sft.env_setup --data_dir <data_dir> -r <repo>/requirements/requirements_sft_<model>.txt --lmms_eval_opt_deps qwen2_5_vl
python -m pip install "protobuf==6.33.0"   # the SFT launchers add this: env_setup leaves protobuf 3.20, which breaks wandb>=0.21 / trl import
```
> **Locally-built packages are not pinned in these files.** `medvision_bm`, the vendored
> `medvision_lmms_eval` and `medvision_ds` are installed by the launcher and by
> `medvision_bm.sft.env_setup` (`install_vendored_lmms_eval`, `install_medvision_ds`), and LLaVA-Med is
> cloned into `--dir_third_party` by its eval driver — so the requirements files carry a comment where
> each used to sit rather than a path pin. Install those four the launcher's way, not with `-r`.

If the target stack is transformers 4.5x, re-check `huggingface_hub` afterwards (`check_env_pins.py --model sft_<model>`) and re-pin `huggingface_hub==0.36.0` if it was lifted; for the transformers-5 stacks (gemma4, qwen3.6vl) the hub must instead stay `>=1.5` — never copy the 0.36.0 pin there. SFT/RFT-specific recipes: `../../sft/SKILL.md`, `../../rft/SKILL.md`.

### 4e. Conda conventions from the launchers

One environment per model: `conda create -n eval-<model> python==3.11 -y` (SFT: `sft-<model>`, plus `conda install -c nvidia cuda-toolkit=12.4 -y`). The 5 RFT launchers force the classic solver for deterministic non-interactive installs — `conda config --set solver classic` and `conda --version | grep -qF "26.1.1" || conda install -y conda=26.1.1` — after seeing libmamba solver failures. Docker images create the env with `conda create --override-channels -n <env> python=3.11 pip -y -c conda-forge`.

## 5. Refreshing `medvision_bm` from a checkout: wheel build on local disk

Every launcher reinstalls `medvision_bm` from the checkout before running, using a wheel built on **node-local disk** rather than in the shared tree:

```bash
bash ../scripts/build_local_wheel.sh --repo <repo>                 # build in mktemp, copy to <repo>/.wheelhouse, flock-install with --no-deps
bash ../scripts/build_local_wheel.sh --repo <repo> --no-install    # just build; prints the wheel path
```

Why: setuptools' `build_py` caches created directories in a process-global memo; on CephFS-like file systems a build subdirectory can transiently vanish (async delete/recreate lag, or a concurrent writer), after which the cache refuses to recreate it and the copy dies with `could not create '<build>/lib/medvision_bm/...': No such file or directory`. The launcher block therefore `tar`-copies `pyproject.toml MANIFEST.in LICENSE src` into `mktemp -d`, runs `python -m pip wheel <tmp> -w <tmp>/wh --no-deps`, copies the wheel to `<repo>/.wheelhouse/`, and installs it with `flock <repo>/.medvision_build.lock python -m pip install --force-reinstall <wheel>`. The launchers' install line has **no `--no-deps`**, so it re-installs `medvision_bm`'s `torch==2.6.0`/`torchvision==0.21.0`/`huggingface_hub==0.36.0` and downgrades a working vLLM stack — which is exactly why the three Method-1 lines follow it. The bundled script defaults to `--no-deps` (pass `--with-deps` to reproduce the launcher behaviour) and prints where `medvision_bm` imports from afterwards. After any reinstall of `medvision_bm`, rerun `install_vendored_lmms_eval` (step 3).

## 6. Docker

Images: `vincentycyao/medvision:base` (Ubuntu 20.04, Miniconda under `/opt/miniconda`, `libgl1` and friends, `git-lfs`, the repository cloned with `GIT_LFS_SKIP_SMUDGE=1`) and one tag per model built `FROM` it: `eval_claude`, `eval_gemini`, `eval_gemma3`, `eval_gemma4`, `eval_glm4v`, `eval_gpt`, `eval_healthgpt`, `eval_huatuogptvision`, `eval_internvl3`, `eval_kimi`, `eval_lingshu`, `eval_llama3vision`, `eval_llavamed`, `eval_llavaonevision`, `eval_meddr`, `eval_medgemma`, `eval_medvision-v0`, `eval_minimax-m3-int4`, `eval_qwen25vl`, `eval_qwen25vl_update1`, `eval_qwen3vl`, `sft_medgemma`, `sft_qwen25vl` (each `dockerfile/Dockerfile.<tag>` maps 1:1 to the tag). A model image creates conda env `eval-<key>`/`sft-<key>`, installs `cuda-toolkit=12.4` (GPU images), `pip install`s the cloned repository, then runs `python -m medvision_bm.benchmark.env_setup -r requirements/requirements_eval_<key>.txt [--lmms_eval_opt_deps <extra>] --data_dir <container data dir>` (SFT images: `medvision_bm.sft.env_setup -r ...`); API images add `--skip_cuda_toolkit --skip_vllm` and then replace CUDA torch with `torch==2.6.0+cpu`. The container's default command activates the env.

```bash
docker pull vincentycyao/medvision:<tag>
docker run -it --rm --gpus all -v </path/to/working/folder>:<container workdir> vincentycyao/medvision:<tag> bash
# inside the container
git clone https://github.com/YongchengYAO/MedVision.git <container workdir> && cd <container workdir>
conda env list && conda activate <env-name>            # eval-<key> or sft-<key>
pip install . && pip show medvision_bm                  # latest medvision_bm
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data && pip show medvision_ds
```

With a model image the launcher's `pip install -r requirements/... --no-deps` line can be skipped, but `install_medvision_ds` and `install_vendored_lmms_eval --lmms_eval_opt_deps <extra>` must always run (the `pip install .` you just did replaced the editable engine). The container's working folder doubles as the benchmark directory: imaging data, results and checkpoints are written under it. The repository's image build/push helper is not part of this skill (it pushes to Docker Hub).

## 7. Final verification checklist

```bash
mvbm --help && python -m medvision_bm.benchmark.env_setup --help >/dev/null && echo cli-ok
python -c "import medvision_bm, medvision_ds; print(medvision_bm.__file__, medvision_ds.__file__)"
python ../scripts/check_env_pins.py --requirements <repo>/requirements/requirements_eval_<model>.txt   # exit 0
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"                        # GPU hosts: True
python ../../../scripts/check_medvision_env.py --repo-root <repo>                                        # root-level summary incl. env vars and optional deps
```

Then export `MedVision_DATA_DIR`, `MedVision_PLANNER_VERSION` (and `MedVision_ACK_RELEASE` when pinning), sanitise `HF_TOKEN`, and hand over to `../../benchmark-evaluation/SKILL.md` or `../../dataset-and-tasks/SKILL.md`.
