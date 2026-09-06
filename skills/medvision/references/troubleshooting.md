# Cross-Cutting Troubleshooting

## Purpose

Read this first when a MedVision command fails for a reason that is not
specific to one workflow: package installation and version pins, the dataset
package and its environment variables, GPU/host limits, or result-tree hygiene.
Workflow-specific failures live in the nearest sub-skill
`references/troubleshooting.md` (linked at the end of each section). Run
`../scripts/check_medvision_env.py` before anything else; most rows below are
visible in its output.

## 1. Install and import

| Symptom | Likely cause | Fix / check | Stop when |
| --- | --- | --- | --- |
| `ModuleNotFoundError: No module named 'medvision_bm'` | package not installed in *this* interpreter (launchers create one conda env per model) | `pip install medvision-bm` (PyPI) or `pip install .` in a checkout; confirm with `python -c "import medvision_bm; print(medvision_bm.__file__)"` | – |
| Edits under `src/medvision_bm/` "do nothing" | a plain `pip install .` replaced an editable install with a site-packages copy | `pip install -e <repo> --no-deps` (needs build isolation, i.e. network) or run with `PYTHONPATH=<repo>/src`; the checker prints `(site-packages copy)` for this case | – |
| `ModuleNotFoundError: No module named 'medvision_ds'` when parsing, summarizing, training or loading data | dataset package not installed (it is not on PyPI) | `mvbm install mvds -d <data_dir>` or `python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>` (needs network to the HF dataset repo) | no network: install later from a local copy of the dataset repo's `src/` |
| `ImportError: cannot import name 'is_offline_mode' from huggingface_hub` | transformers 4.x with `huggingface_hub>=1.0` (something lifted the pin, often the dataset-package installer or an unpinned `pip install`) | `pip install "huggingface_hub==0.36.0"` (or the value in the model's requirements file); re-check with the sub-skill pin checker `../sub-skills/environment-setup/scripts/check_env_pins.py` | – |
| `AttributeError: ... TokenizersBackend ... all_special_tokens_extended` at vLLM `LLM()` init | transformers 5.x in an environment whose vLLM expects 4.x | downgrade transformers to the pin in the model's requirements file (e.g. 4.57.x); GLM-4.6V / Gemma-4 stacks deliberately use transformers 5 with vLLM 0.19 | – |
| `trust_remote_code` errors / dataset script refused | `datasets>=4` installed | `pip install "datasets==3.6.0"` (pinned by `medvision_bm`) | – |
| `operator torchvision::nms does not exist` or `undefined symbol` on import | torch/torchvision ABI mismatch after a `--force-reinstall` from a wheelhouse or a partial upgrade | reinstall the *pair* from the model's requirements file (e.g. torch 2.6.0 + torchvision 0.21.0) | – |
| `libGL.so.1: cannot open shared object file` | `opencv-python` (pulled by `medvision_ds`) on a headless host | `pip install opencv-python-headless` and uninstall `opencv-python`, or install the system GL library | – |
| `pip`/`conda` break after installing `medvision_ds` | its `gdrive` dependency pins `setuptools~=59.6.0` | `pip install "setuptools>=60,<81"` afterwards (`<81` keeps `pkg_resources` for `pylidc`); the remaining `pip check` complaint about gdrive is harmless | – |
| `could not create '.../build/...': No such file or directory` while building the wheel on a shared network filesystem | setuptools build cache race on CephFS-like storage | build the wheel in a node-local temp dir and install it under a lock — `../sub-skills/environment-setup/scripts/build_local_wheel.sh` does exactly that | – |
| `cannot import name 'Imports' from wandb.proto...` at SFT start | protobuf too old for `wandb>=0.21` after `sft.env_setup` | `pip install "protobuf==6.33.0"` (the launchers do this) | – |
| conda solver hangs or fails inside launchers | libmamba solver on this conda version | `conda config --set solver classic` (launchers also pin `conda=26.1.1`) | – |

More: `../sub-skills/environment-setup/references/troubleshooting.md`.

## 2. Dataset package and environment variables

| Symptom | Likely cause | Fix / check | Stop when |
| --- | --- | --- | --- |
| Loader raises `MedVision: annotation version selection required` (or similar banner) | `MedVision_PLANNER_VERSION` unset (required since dataset package 1.1.0) | `export MedVision_PLANNER_VERSION=1.0.0` (leaderboard) or `latest`; set it **before** the first `datasets` import | – |
| Loader refuses an older pin / asks for an acknowledgement | pinning below the newest annotation release without `MedVision_ACK_RELEASE` | `export MedVision_ACK_RELEASE=1.4.0` (must equal the latest release; T/L is the only family whose annotations changed) | – |
| Sample counts differ from the leaderboard tables | a different planner version (only T/L changes), or `MedVision_DISABLE_SAMPLE_FILTERING=true` (multi-instance samples) | pin `1.0.0`, unset the filtering override; compare with the per-version `all_tasks__ds_v*` lists | – |
| `401`/`403` on a gated dataset | missing token, or a token with a trailing newline injected by the pod/k8s secret | `export HF_TOKEN="$(printf '%s' "$HF_TOKEN" \| tr -d '[:space:]')"`; FeTA24 needs `SYNAPSE_TOKEN`; SKM-TEA/ToothFairy2 need a private HF mirror id | you do not hold access to the source dataset |
| Raw data was updated but the loader still returns old values | a warm Arrow cache never re-runs the loading script | `download_mode="force_redownload"` **and** (`MedVision_FORCE_DOWNLOAD_DATA=True` or drop the entry from `<data_dir>/.downloaded_datasets.json`) | – |
| `Datasets/<name>` missing plans for the requested version | `resolve_plan_path` applies a *ceiling*: each dataset loads its newest plan at or below the pin; a dataset never regenerated still resolves to an older version | inspect with `../sub-skills/dataset-and-tasks/scripts/inspect_benchmark_plan.py` | – |
| `BuilderConfig '<task>-CoT_Test' not found` from `download_datasets --tasks_json` | eval task lists carry the `-CoT` suffix, which `tasks_to_configs` does not strip (it only rewrites `BoxCoordinate`→`BoxSize` and appends the split) | pass an SFT-style list, a configs CSV, or use `../sub-skills/dataset-and-tasks/scripts/download_datasets.sh`, which strips the suffix into a temp copy | – |
| Disk fills up during download | QC figures (~298 GB) were enabled, or a whole source dataset was fetched for one config (always the case) | keep `MedVision_DOWNLOAD_QC_FIGURES` unset; budget ~1 TB for the full collection | – |

More: `../sub-skills/dataset-and-tasks/references/troubleshooting.md`.

## 3. Host, GPU and memory

| Symptom | Likely cause | Fix / check | Stop when |
| --- | --- | --- | --- |
| `torch.cuda.is_available()` is False / `nvidia-smi` missing | CPU-only host or no GPU passthrough | parsing, summarizing, task-list work, parquet inspection and analyses still run; evaluation, SFT/RFT and the LLM judge need a CUDA host | a GPU is required for the requested step |
| CUDA out of memory in vLLM | `tensor_parallel_size` = visible GPUs is too small for the weights, or `gpu_memory_utilization` too high with a large batch | expose more GPUs via `CUDA_VISIBLE_DEVICES`, lower `batch_size_per_gpu`, check `model-roster.md` for footprints | model does not fit the available cards |
| Process killed with no traceback during dataset construction | cgroup memory limit (containers report host RAM in `free -g`, but the ceiling is `memory.max`) | reduce `num_workers_concat_datasets`/`num_workers_format_dataset`; use the checkpointed parquet builder; check `/sys/fs/cgroup/memory.max` | – |
| `NCCL` timeouts / hangs at checkpoint save | slow shared filesystem or a rank waiting on a long save | raise `NCCL_TIMEOUT`, save less often, keep `main_process_port` unique per job | – |

More: `../sub-skills/benchmark-evaluation/references/troubleshooting.md`, `../sub-skills/sft/references/troubleshooting.md`.

## 4. Result trees and bookkeeping

| Symptom | Likely cause | Fix / check | Stop when |
| --- | --- | --- | --- |
| Eval says a task is done and skips it | `completed_tasks/completed_tasks_<task_tag>.json` marks it complete | remove the entry, or point `--task_status_json_path` at a scratch file — the skip is unconditional; `--skip_update_status` only stops the *current* run from recording completion, it does not re-run a completed task. Never hand-edit result JSONLs | – |
| Re-run regenerates nothing / everything | `response_cache/` keys include a prompt hash: unchanged prompt → resume; edited prompt → automatic invalidation; `MEDVISION_RESP_CACHE=0` disables the cache | decide whether you want resume or a clean rerun before starting | – |
| Summaries look wrong after re-parsing | mixed `parsed/` vs `llm-parsed_<judge>/` inputs, or `--limit` differing across models | keep `--parsed_dirname`, `--resps_key`, `--limit`, `--removed_samples_dir` identical across the models of one report | – |
| A metric threshold is lower than the mean suggests | thresholds (`IoU>k`, `MRE<k`) divide by the **total** sample count while means exclude/zero failures | see `../sub-skills/results-parsing-and-metrics/references/metrics.md` | – |

More: `../sub-skills/results-parsing-and-metrics/references/troubleshooting.md`, `../sub-skills/llm-judge-parsing/references/troubleshooting.md`.

## 5. When to stop and ask

- The step needs a GPU, model weights, API credits, or a gated dataset you do not have.
- A fix requires changing a pinned version in a shared environment used by other runs.
- Reproducing a leaderboard number requires the exact planner version and roster; ask which version the user wants before re-running.
