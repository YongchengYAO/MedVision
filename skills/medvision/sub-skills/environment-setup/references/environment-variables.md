# Environment Variables

Two components read environment variables: the `medvision_bm` package (installers, eval wrappers, vendored `lmms_eval`) and the Hugging Face dataset loading script `MedVision.py` of the `YongchengYAO/MedVision` dataset repo, which is what `datasets.load_dataset("YongchengYAO/MedVision", ...)` executes and which in turn installs and imports `medvision_ds`. Every entry below was verified by grepping those sources; "set by" means the code assigns it, "read by" means the code consumes it.

Casing matters: the dataset variables are `MedVision_*` (mixed case), the benchmark-runtime ones are `MEDVISION_*` (upper case).

## Quick start block

```bash
export MedVision_DATA_DIR=<data_dir>            # where datasets, src/, caches live (~1 TB for everything)
export MedVision_PLANNER_VERSION=latest         # REQUIRED; or a pinned version such as 1.0.0
# export MedVision_ACK_RELEASE=1.4.0            # only when pinning below a dataset's newest annotation
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN="$(printf '%s' "${HF_TOKEN}" | tr -d '[:space:]')"  # strip trailing newline
```

`setup_env_hf_medvision_ds(<data_dir>)` (called by every eval wrapper and by `install_medvision_ds`) sets `MedVision_DATA_DIR`, `MedVision_FORCE_INSTALL_CODE=true`, `HF_HOME`, `HF_DATASETS_CACHE` for you; it never sets the planner version, so **export `MedVision_PLANNER_VERSION` yourself before any dataset load** (the repository launchers put `export MedVision_PLANNER_VERSION='1.0.0'` right after the wheel-build block).

## Dataset loader variables (`MedVision_*`)

| Variable | Default | Read by | Semantics |
| --- | --- | --- | --- |
| `MedVision_DATA_DIR` | none — **required** | `MedVision.py` at import (`ValueError: Environment variable MedVision_DATA_DIR must be set to specify download directory`), `_data_root()` (also rejects an empty string); `medvision_bm.dataset.ds_utils`, `medvision_bm.sft.sft_utils`, the RFT parquet builders | Root for `Datasets/<dataset>/`, `src/` (the `medvision_ds` source), `.downloaded_datasets.json` (download/install tracker), `.cache/`. Normalised to an absolute path. An 8-char hash of the root is folded into the Arrow cache config id, so two roots never share a cache. Set by `setup_env_medvision_ds(data_dir)` (absolute path). |
| `MedVision_PLANNER_VERSION` | none — **required** since dataset v1.1.0 | `MedVision.py::_split_generators` via `_normalize_requested`; also part of the Arrow cache fingerprint (resolved version) and of the QC-figure check | The **newest annotation version you are willing to load — a ceiling, not a selection**: each (dataset, plan kind) loads the newest annotation published at or before it. Accepted: `latest` (resolves to the `medvision_ds` release, `1.4.0` at generation time), a published annotation version — `1.0.0`, `1.1.0`, `1.1.1`, `1.2.0`, `1.2.1`, `1.3.0`, `1.4.0` — or the release version itself. Malformed values (`v1.1.1`, `1.2`) and unpublished ones (`1.1.5`) raise instead of being guessed. Datasets introduced after the pinned version cannot be loaded at that pin ("annotation not published at the selected version"); a withdrawn annotation (MAMA-MIA / PI-CAI at 1.2.0) raises "annotation WITHDRAWN at the selected version". Unset → `EnvironmentError` whose banner reads `MedVision: annotation version selection required` (in eval logs it may surface wrapped in a `tenacity.RetryError[... OSError]`; the banner is in the first inner traceback). The repository launchers pin `'1.0.0'`. Changing it changes the T/L sample set. |
| `MedVision_ACK_RELEASE` | unset | `MedVision.py::_enforce_release_ack` (called from `_split_generators`, before any download) | Required **only** when the pinned `MedVision_PLANNER_VERSION` is **below the newest annotation published for the (dataset, plan kind) being loaded**; a release that did not regenerate that dataset never triggers it. Accepted values: that pair's newest annotation version (expires when the dataset is regenerated) **or** the release version as a blanket acknowledgement (`1.4.0`; use this for catalogue sweeps). Otherwise `EnvironmentError` with banner `MedVision: outdated annotation version — acknowledgement required`, which prints both accepted values. Example: `MedVision_PLANNER_VERSION=1.0.0` on a T/L config regenerated at 1.4.0 needs `MedVision_ACK_RELEASE=1.4.0`; the same pin on a Detection config whose newest annotation is still 1.0.0 needs nothing. A stale value from an older release (e.g. `1.1.1`) stops working as soon as the dataset is regenerated again. |
| `MedVision_FORCE_INSTALL_CODE` | `True` (loader default) | `MedVision.py::_split_generators` | When true (or when the tracker's `medvision_ds`/`medvision_ds_installed` entry is not the current release version) the loader re-downloads `src/*` from the Hub and runs `pip install .` in `<data_dir>/src` **inside the calling process**, serialised on `<data_dir>/src/.build.lock`, on **every** dataset build. `setup_env_medvision_ds(force_install_code=True)` (the default) sets it to `true`. Set it to `false` **after** your explicit `install_medvision_ds` run if you must prevent the loader from re-resolving `medvision_ds` dependencies mid-run (the GLM-4.6V wrapper does exactly this because the reinstall can shift `huggingface_hub` under a running transformers-5 stack). The published guidance is to leave it `True` so release notices are seen. |
| `MedVision_FORCE_DOWNLOAD_DATA` | `False` | `MedVision.py` (download step and QC-figure step); set to `true` by `setup_env_medvision_ds(force_download_data=True)` | Force re-download of raw images/annotations for the dataset being built, and force re-fetch of QC figures. Only consulted when the loading script actually runs — pair it with `load_dataset(..., download_mode="force_redownload")`, otherwise a valid Arrow cache short-circuits the script. Also the documented fix for "annotation file missing after download". |
| `MedVision_DISABLE_SAMPLE_FILTERING` | `False` | `MedVision.py::_generate_examples`; when `true` it is added to the Arrow config id | Bypass the per-sample quality/size filters and return every planner sample (multi-instance cases included). The default config id is unchanged, so existing caches stay valid. |
| `MedVision_DOWNLOAD_QC_FIGURES` | `False` | `MedVision.py` (`_info` and `_split_generators`) | Opt-in download of per-slice QC figure archives (`Datasets/<dataset>_fig.zip` / `_fig.partNN.zip`; ~298 GB in total, no task reads them). Checked on every load; only figures are fetched when the dataset is already present; state tracked per annotation version as `qc_figures_<dataset>` in `.downloaded_datasets.json`. |
| `MedVision_PLANNER_WORKERS` | `1` | `medvision_ds.utils.benchmark_planner` | Parallel workers for benchmark-plan generation (`max(1, int(value))`). |
| `MedVision_SKMTEA_HF_ID` | `YongchengYAO/SKM-TEA-nii` | `medvision_ds.datasets.SKM_TEA.download` | Your private HF dataset repo holding the preprocessed SKM-TEA data (registration-only dataset). |
| `MedVision_ToothFairy2_HF_ID` | see `medvision_ds.datasets.ToothFairy2.download` | same pattern | Private HF repo for preprocessed ToothFairy2 data. |
| `BiometricVQA_KiPA22_HF_ID` | `YongchengYAO/KiPA22` | `medvision_ds.datasets.KiPA22.download` | Legacy-named override for the KiPA22 mirror. |
| `SYNAPSE_TOKEN` | unset | `medvision_ds.datasets.FeTA24.download` (and the `download_raw.py` rebuild scripts of BraTS24, BCV15, MAMA-MIA) | Synapse personal access token for FeTA24. |

## Hugging Face variables

| Variable | Set by | Notes |
| --- | --- | --- |
| `HF_HOME` | `setup_env_hf(data_dir)` → `<abs data_dir>/.cache/huggingface` | Model and hub cache; also read by the vendored `lmms_eval/api/task.py`. |
| `HF_DATASETS_CACHE` | `setup_env_hf(data_dir)` → `<abs data_dir>/.cache/huggingface/datasets` | Arrow cache of every MedVision config (large). |
| `HF_TOKEN` | you | Needed for gated/private sources: the private HF repos for SKM-TEA and ToothFairy2, the gated `AbdomenAtlas/AbdomenAtlas1.0Mini` imaging files (accept the terms first), and any gated model weights. The vendored `lmms_eval/__main__.py` appends `token=$HF_TOKEN` to the evaluation-tracker args when set (redacted in its log line). Container/pod-injected secrets often carry a **trailing newline** that is legal for `huggingface_hub` (it strips it) but fatal for vLLM's own header (`Illegal header value b'Bearer hf_...\n'`), so always sanitise: `export HF_TOKEN="$(printf '%s' "$HF_TOKEN" \| tr -d '[:space:]')"`. FeTA24 uses `SYNAPSE_TOKEN`, not `HF_TOKEN`. |
| `HF_HUB_ENABLE_HF_TRANSFER` | `lmms_eval/models/__init__.py` sets `"1"` unconditionally | Requires the `hf_transfer` package (a base dependency of the vendored `lmms_eval`). |

## Benchmark-runtime variables (`MEDVISION_*` and friends)

| Variable | Default | Set/read by | Notes |
| --- | --- | --- | --- |
| `MEDVISION_RESP_CACHE` | `1` | read by vendored `lmms_eval/api/model.py::init_response_cache` | `0` disables the per-sample response cache written to `<output_path>/response_cache/<task>_rank<r>.jsonl`; the cache key is `task::split::doc_id::<16-hex prompt hash>`, so a changed prompt misses instead of returning stale text. Resume semantics: `../../benchmark-evaluation/SKILL.md`. |
| `MEDVISION_SCALED_PS_LOW` / `MEDVISION_SCALED_PS_HIGH` | `0.5` / `3.0` | set by every eval wrapper from `--scaled_ps_low/--scaled_ps_high`; read by `lmms_eval/tasks/medvision/medvision_utils.py` | Pixel-size scaling range for `-scaledPS` task variants (`../../benchmark-evaluation/SKILL.md`). |
| `VLLM_WORKER_MULTIPROC_METHOD` | — | `setup_env_vllm(data_dir)` → `spawn` | Applied by `install_vllm` and in the `--skip_env_setup` branch of vLLM wrappers. |
| `XDG_CACHE_HOME` | — | `setup_env_vllm(data_dir)` → `<abs data_dir>/.cache/vllm` | vLLM compile/torch-inductor caches under the data dir. |
| `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH` | — | `setup_env_cuda()` → `CUDA_HOME=$CONDA_PREFIX`, prepends `$CUDA_HOME/bin`, `$CUDA_HOME/lib64`, `$CUDA_HOME/lib` | Run after `install_cuda_toolkit` / `install_torch_cu124`; only meaningful inside a conda env. |
| `PIP_DISABLE_PIP_VERSION_CHECK` | `1` | `run_pip_install` (setdefault) | Cosmetic. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY` | unset | API eval wrappers (`eval__claude`, `eval__openai`, `eval__gemini`, `eval__kimi`) and the matching `lmms_eval/models/*.py` | Credentials for API models; apply the same whitespace sanitising as `HF_TOKEN`. Details in `../../benchmark-evaluation/SKILL.md`. |
| `MEDVISION_SFT_*` (`_ATTN`, `_LR`, `_OPTIM`, `_USE_LIGER`, `_SAVE_ONLY_MODEL`, `_SYNC_EACH_BATCH`, `_MEMPROBE`, `_MEMSNAPSHOT`, `_COMPLETION_ONLY`) | unset | `medvision_bm.sft.sft_utils`, `gemma4_utils`, `medgemma_utils` | Training knobs; documented in `../../sft/SKILL.md`. |

## Where the variables must be visible

- `MedVision_*` and `HF_*` must be in the environment of the **process that calls `load_dataset`** — the eval wrapper, the `lmms_eval` subprocess it spawns (inherits), SFT dataset preparation, parquet builders, and any ad-hoc Python session. Exporting them in the shell before `python -m ...` is the simplest way.
- A pinned planner version is part of the experiment's identity: record `MedVision_PLANNER_VERSION` (and the resolved `medvision_ds` version) next to results. Dataset/task semantics of the versions: `../../dataset-and-tasks/SKILL.md`.
