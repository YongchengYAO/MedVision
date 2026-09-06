---
name: environment-setup
description: "Installs and repairs the MedVision stack: medvision_bm (PyPI, local checkout, editable, nightly), the medvision_ds dataset package via mvbm install mvds / install_medvision_ds, the vendored lmms_eval engine and its per-model extras, benchmark.env_setup and sft.env_setup, the 25 frozen requirements files, Docker tags, the load-bearing install order inside eval__<model>.py, the wheel-build-on-local-disk recipe, the MedVision_* / HF_* / MEDVISION_* environment variables, and the huggingface_hub / transformers / torch / vLLM pin traps."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision Environment Setup

Use this sub-skill when a task is about getting the MedVision benchmark/post-training code to **install, import and start** on a machine — or about diagnosing why it stopped importing after a dependency moved. It covers the two Python packages (`medvision_bm`, the benchmark/SFT/RFT code; `medvision_ds`, the dataset code shipped in the `src/` folder of the Hugging Face dataset `YongchengYAO/MedVision`), the vendored `lmms_eval` evaluation engine, the per-model pinned stacks, Docker images, and every environment variable the loader and the eval runtime read.

## Route Here For

- Installing `medvision_bm` (PyPI `pip install medvision-bm`, `pip install .` from a checkout, `pip install -e . --no-deps`, or `pip install "git+https://github.com/YongchengYAO/MedVision.git"`) and verifying `pip show medvision_bm` / `medvision_bm.__file__` / `mvbm --help`.
- Installing the dataset package with `mvbm install mvds -d <data_dir>` (alias of `python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>`), understanding its snapshot-download + wheel-build + **two-step** install, and the env vars it exports.
- Installing the vendored engine with `python -m medvision_bm.benchmark.install_vendored_lmms_eval [--lmms_eval_opt_deps <extra>]` and choosing the right extra (`qwen2_5_vl`, `medvision_v0`, `qwen3_vl`, `glm4v`, `minimax_m3`, `lingshu`, `meddr`, `huatuogpt_vision`, `llava_med`, `gemini`, `claude`, `openai`, `kimi`).
- Building a per-model environment with `python -m medvision_bm.benchmark.env_setup -r <requirements> --data_dir <data_dir> ...` or `python -m medvision_bm.sft.env_setup ...`, or manually ("Method 1": `install_medvision_ds` → `install_vendored_lmms_eval` → `pip install -r requirements_eval_<model>.txt --no-deps` → run with `--skip_env_setup`).
- Explaining or preserving the **load-bearing install order** inside every `eval__<model>.py` (`setup_env_hf_medvision_ds` → `ensure_hf_hub_installed` → `install_vendored_lmms_eval` → `install_medvision_ds` → `install_torch_cu124` → `install_vllm` → model-specific transformers/accelerate reinstall) and the `--skip_env_setup` / `--env_setup_only` flags.
- Refreshing `medvision_bm` from a checkout on shared storage (CephFS `could not create ...: No such file or directory`) with the wheel-build-on-local-disk recipe.
- Setting `MedVision_DATA_DIR`, `MedVision_PLANNER_VERSION` (required), `MedVision_ACK_RELEASE`, `MedVision_FORCE_INSTALL_CODE`, `MedVision_FORCE_DOWNLOAD_DATA`, `MedVision_DISABLE_SAMPLE_FILTERING`, `MedVision_DOWNLOAD_QC_FIGURES`, `HF_HOME`, `HF_DATASETS_CACHE`, `HF_TOKEN` (with newline sanitising), `MEDVISION_RESP_CACHE`.
- Pin conflicts: `huggingface_hub` 0.36.0 vs `>=1.5` (transformers 4.x vs 5.x, `cannot import name 'is_offline_mode'`), `datasets==3.6.0` (`trust_remote_code`), torch/torchvision vs vLLM ABI (`torchvision::nms does not exist`, `vllm/_C ... undefined symbol`), `gdrive` → `setuptools~=59.6.0`, `opencv-python` → `libGL.so.1`, `protobuf==3.20` → wandb `Imports`, conda solver, the editable install silently reverted by a plain `pip install .`.
- Docker: `vincentycyao/medvision:base` and the `eval_<model>` / `sft_<model>` tags, the `docker run --gpus all -v ...` recipe and in-container steps.

## Do Not Use For

- Running evaluations, launcher variables, sample limits, token budgets, results layout → `../benchmark-evaluation/SKILL.md`.
- Dataset downloads, task JSONs, config names, annotation versions as *data* (which samples change between 1.0.0 and 1.4.0) → `../dataset-and-tasks/SKILL.md`.
- SFT/RFT training recipes beyond `medvision_bm.sft.env_setup` → `../sft/SKILL.md`, `../rft/SKILL.md`.
- The LLM-judge environment (separate vLLM pin) → `../llm-judge-parsing/SKILL.md`; the BiomedParse ablation environment → `../biomedparse-ablation/SKILL.md`.
- Per-model hardware (VRAM, GPU count, parallelism) → `../../references/model-roster.md`; cross-cutting failures → `../../references/troubleshooting.md`.

## References and Scripts

- Read `references/installation.md` for the full workflows: the three `medvision_bm` install paths and the editable trap, what `install_medvision_ds()` does step by step, the vendored engine and its extras, `env_setup`/`sft.env_setup` order of operations, the per-model automatic order vs Method 1/Method 2, the wheel-build recipe, conda conventions and Docker.
- Read `references/cli-reference.md` for the verbatim `--help` of `mvbm`, `benchmark.env_setup`, `benchmark.install_medvision_ds`, `benchmark.install_vendored_lmms_eval`, `sft.env_setup`, and the exact signatures/behaviour of every `medvision_bm.utils.install_utils` function.
- Read `references/requirements-catalog.md` when choosing or auditing a pin set: all 21 eval + 4 SFT requirements files with their torch/torchvision/vllm/transformers/accelerate/huggingface_hub pins, the model-key → wrapper → extra → Docker-tag map, and the vendored extras table.
- Read `references/environment-variables.md` before setting or debugging any `MedVision_*`, `HF_*`, `MEDVISION_*` variable — defaults, who reads them, exact error banners, and the planner-version/acknowledgement rules.
- Read `references/troubleshooting.md` when an import, install or dataset load fails: symptom fragment → cause → fix → when to stop (GPU, credentials, network, shared env).
- Run `scripts/check_env_pins.py --requirements <file>` (or `--model <key> [--repo-root <repo>]`, `--python <interp>`, `--json`) to compare installed torch/torchvision/vllm/transformers/accelerate/huggingface_hub/datasets/medvision_bm/medvision_ds/flash-attn against a pin set; it prints both packages' `__file__`, hints at hub/transformers mismatches, never installs, and exits 1 on any mismatch.
- Run `scripts/build_local_wheel.sh --repo <repo> [--wheelhouse <dir>] [--no-install] [--with-deps]` to rebuild `medvision_bm` from a checkout on node-local disk and flock-install it (default `--no-deps`, so torch/hub pins stay untouched); the last stdout line is the wheel path.
- For a one-shot environment summary that also covers optional dependencies per workflow, run the root `../../scripts/check_medvision_env.py --repo-root <repo>`.

## Minimal Recipe (fresh machine, one model)

```bash
conda create -n eval-<model> python==3.11 -y && conda activate eval-<model>     # most launchers use 3.11 - MedDr 3.9, LLaVA-Med 3.10, MiniMax-M3(-INT4) 3.12
git clone https://github.com/YongchengYAO/MedVision.git <repo> && cd <repo> && pip install . && pip show medvision_bm
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>       # == mvbm install mvds -d <data_dir>
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps <extra>   # omit the flag if the model has none
pip install -r <repo>/requirements/requirements_eval_<model>.txt --no-deps          # frozen pins win; applied LAST
pip install "setuptools>=60,<81"                                                   # undo the gdrive-induced setuptools downgrade
python skills/medvision/sub-skills/environment-setup/scripts/check_env_pins.py --requirements <repo>/requirements/requirements_eval_<model>.txt
export MedVision_DATA_DIR=<data_dir> MedVision_PLANNER_VERSION=latest              # planner version is REQUIRED by the loader
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN="$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')"
```

Open-weight evaluation additionally needs CUDA 12.4 GPUs (the pins install cu124 torch and vLLM); API models (Claude/Gemini/GPT/Kimi) need `--skip_cuda_toolkit --skip_vllm` in `env_setup` and API keys instead. Hand over to `../benchmark-evaluation/SKILL.md` to run.

## Safe Operating Rules

1. Every installer here mutates the active environment and most download from Hugging Face; state the target interpreter (`which python`) and the pin row you are applying before running, and never "just upgrade" `transformers`, `huggingface_hub`, `torch` or `datasets` in a MedVision env — pick the exact version from `references/requirements-catalog.md`.
2. Keep the order: vendored `lmms_eval` → `medvision_ds` → torch → vLLM → model-specific transformers/accelerate → frozen requirements `--no-deps` last. After any reinstall of `medvision_bm`, rerun `install_vendored_lmms_eval` (the editable engine lives inside the package tree).
3. Treat `MedVision_PLANNER_VERSION` as part of the experiment identity: the loader hard-fails without it, a pin below a dataset's newest annotation needs `MedVision_ACK_RELEASE`, and for T/L tasks the version changes the sample set.
4. Prefer `--no-install` / `--env_setup_only` / `check_env_pins.py` dry checks before touching a shared conda prefix; GPU-only verification (vLLM init, flash-attn) must be flagged "requires GPU" on CPU hosts.
5. Never print or commit tokens; sanitise `HF_TOKEN` and API keys with `tr -d '[:space:]'` (container-injected secrets carry a trailing newline that vLLM rejects while `huggingface_hub` silently tolerates it).
6. Do not run the repository's private launcher scripts or its Docker build/push helper; reproduce their install blocks with the commands above and `scripts/build_local_wheel.sh`.
