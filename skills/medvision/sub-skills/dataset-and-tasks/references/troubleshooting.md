# Troubleshooting: Dataset, Versions, Downloads, Task Lists

Quick triage: (1) print `MedVision_DATA_DIR`, `MedVision_PLANNER_VERSION`, `MedVision_ACK_RELEASE`, `MedVision_DISABLE_SAMPLE_FILTERING`; (2) `python -c "import datasets; print(datasets.__version__)"` must say `3.6.0`; (3) run `scripts/list_tasks.py` on the task list to see the configs actually requested; (4) run `scripts/inspect_benchmark_plan.py` on the dataset directory to see which plan versions are on disk.

## Loader banners (all `EnvironmentError` unless noted)

| Error fragment | Cause | Fix | Stop when |
| --- | --- | --- | --- |
| `MedVision: annotation version selection required` | `MedVision_PLANNER_VERSION` unset (required since dataset v1.1.0) | `export MedVision_PLANNER_VERSION=latest` (new work) or `=1.0.0` (leaderboard) **before** `import datasets`; the banner lists accepted values | never; this is configuration |
| `MedVision: invalid MedVision_PLANNER_VERSION` ... `Got: 'v1.1.1'` | not `latest` or `X.Y.Z` (`v1.1.1`, `1.2`, padded values) | use a three-component version exactly as published | never |
| `MedVision: unknown MedVision_PLANNER_VERSION` | value never published and not the release (`1.1.5`, `2.0.0`) | choose from the accepted list: `1.0.0, 1.1.0, 1.1.1, 1.2.0, 1.2.1, 1.3.0, 1.4.0` | never |
| `MedVision: outdated annotation version — acknowledgement required` (`Dataset: X (plan kind: biometry)`) | pin below this dataset/kind's newest annotation | `export MedVision_ACK_RELEASE=1.4.0` (blanket) or the dataset value printed in the banner; or move to `latest` | never |
| `MedVision: annotation not published at the selected version` (`This dataset/task first ships at vX`) | dataset newer than the pin (e.g. MSWAL at `1.1.1`) | raise the pin, or skip the dataset by filtering the task list with the version's `ConfigurationsList_*.csv`; ACK does not help | never |
| `MedVision: annotation WITHDRAWN at the selected version` | MAMA-MIA / PI-CAI pinned at `1.2.0` | pin `1.2.1` or later; discard results and Arrow caches built from `1.2.0` | never |
| `MedVision: annotation paused pending investigation` (`RuntimeError`, raised even with a warm cache) | AFIDs / PDDCA / VerSe biometry (`1.2.0`) | no fix; drop these configs; their segmentation/detection configs (PDDCA, VerSe) still load | always for these three biometry plans |
| `MedVision: annotation file missing after download` (`FileNotFoundError`) | interrupted download or dataset still being generated | `export MedVision_FORCE_DOWNLOAD_DATA=True` and reload with `download_mode="force_redownload"` | network unavailable |
| `Environment variable MedVision_DATA_DIR must be set to specify download directory` (at import of the script) | variable missing | set it before `load_dataset`; the `medvision_bm` CLIs set it from `--data_dir` | never |
| `MedVision: dataset 'X' is registered as biometry family ... implies ...` (`RuntimeError`) | loader tables inconsistent (maintainer bug) | update the loader script; not a user problem | report upstream |

## Config-name errors

| Error fragment | Cause | Fix |
| --- | --- | --- |
| `ValueError: BuilderConfig 'KiPA22_TumorLesionSize_Task01_Axial-CoT_Test' not found. Available: [...]` | an eval `-CoT` list (or any `-<variant>` name) was passed to `download_datasets --tasks_json` / `tasks_to_configs`; configs never carry the suffix | strip everything after the plane token (`scripts/list_tasks.py` shows the corrected configs; `scripts/download_datasets.sh` strips automatically), or use the `__train_SFT` list / a `ConfigurationsList_*.csv` |
| `BuilderConfig 'CrossMoDA_BoxCoordinate_Task01_Axial_Test' not found` | config built by hand without the `BoxCoordinate -> BoxSize` rewrite | use `tasks_to_configs`, or pass `tag_ds="BoxSize"` to the SFT loader (it rewrites `_BoxCoordinate_`) |
| `BuilderConfig 'X_Train' not found` for a dataset added in a later release, while `latest` is set | config name not in the loader you have cached | `download_mode="force_redownload"` pulls the current `MedVision.py`; compare against the newest `ConfigurationsList_All.csv` |
| `AssertionError: Split must be 'train' or 'test'` | `tasks_to_configs` split argument | pass `train` or `test` (case-insensitive) |
| `ValueError: ... split ... not found` / empty split | `split="train"` requested on a `*_Test` config or vice versa | match the split to the config suffix |
| `tag_ds ... must be provided` / `dataset_<name>` never appears in the tracker | SFT/parquet loader given the wrong family tag, or task names whose dataset token is not followed by `_<tag>` | use the family token of the list (`BoxSize`, `TumorLesionSize`, `BiometricsFromLandmarks`) |

## Downloads, credentials, network

| Symptom | Cause | Fix | Stop when |
| --- | --- | --- | --- |
| `401`/`403`/`GatedRepoError` on `YongchengYAO/SKM-TEA-nii` or `YongchengYAO/ToothFairy2` | default ids are private mirrors; you must supply your own preprocessed copy | apply to the owners, upload `SKM-TEA-nii.zip` / `ToothFairy2.zip` to a private HF repo, set `MedVision_SKMTEA_HF_ID` / `MedVision_ToothFairy2_HF_ID`, `HF_TOKEN`, `hf auth login --token $HF_TOKEN --add-to-git-credential` | credentials unavailable |
| `ValueError: SYNAPSE_TOKEN environment variable not set` | FeTA24 raw download via Synapse | obtain Synapse access, `export SYNAPSE_TOKEN=...` | credentials unavailable |
| `401` on `AbdomenAtlas/AbdomenAtlas1.0Mini` | gated HF repo | accept the terms on the dataset page, set `HF_TOKEN`, log in | credentials unavailable |
| Token accepted locally but HF returns 401 | trailing newline/space in the token exported from a secret store | `export HF_TOKEN="$(printf '%s' "$HF_TOKEN" \| tr -d '[:space:]')"` (the bundled shell wrappers do this) | never |
| One `_Test` config requested, but several hundred GB downloaded | by design: any config fetches the whole source dataset (all planes and splits) | plan disk per **dataset**, not per config; download `_Test` first and later `_Train` builds are local | disk insufficient |
| Every build re-downloads `src/` and runs `pip install .` (minutes per config) | `MedVision_FORCE_INSTALL_CODE` defaults to `True` in the loader and the download CLI forces it | after the first successful install `export MedVision_FORCE_INSTALL_CODE=false` for plain `load_dataset` sessions | never |
| Rows point at files that do not exist; `image_file` under an old data root | Arrow cache from a previous `MedVision_DATA_DIR` (pre-1.2.0 loaders) or the data directory was deleted while the cache was kept | `MedVision_FORCE_DOWNLOAD_DATA=True` + `download_mode="force_redownload"`; since 1.2.0 the data root is part of the cache id | never |
| Stale annotations after a release; `load_dataset` returns old rows | valid Arrow cache short-circuits the script | lever [1] (`force_redownload`) **plus** [2] (`MedVision_FORCE_DOWNLOAD_DATA=True`) or [3] (delete `dataset_<name>` from `.downloaded_datasets.json`) | never |
| `NonMatchingSplitsSizesError` after switching `MedVision_PLANNER_VERSION` | loader older than 1.1.1 shared one cache across versions | `download_mode="force_redownload"` to fetch the current script; current loaders key the cache on the resolved version | never |
| `FileNotFoundError ... Datasets/<name>.zip` or `Errno 17` during parallel loads | two configs of one dataset prepared concurrently by an old loader | current loader serialises with `.<dataset>.zip.lock`; keep `--num_workers_concat_datasets 1` for a dataset not yet in the tracker (the loaders do this automatically) | never |
| `FileNotFoundError: .../.downloaded_datasets.json` from `build_parquet_ds` / SFT prep | fresh data root: nothing downloaded yet | download at least one config first | never |
| `trust_remote_code` `TypeError` or `RuntimeError: Dataset scripts are no longer supported` | `datasets>=4.0.0` | `pip install "datasets==3.6.0"` (warn: this may alter a user env; `medvision_bm` already pins it) | env is not yours to change |

## Versions, counts, filtering

| Symptom | Cause | Fix |
| --- | --- | --- |
| Sample counts differ from the leaderboard / paper | leaderboard uses `1.0.0`; `latest` = `1.4.0` has ~20x more T/L samples and re-split six datasets (HNTSMRG24, KiPA22, KiTS23, MSD, autoPET-III, BraTS24) | pin `1.0.0` + `MedVision_ACK_RELEASE=1.4.0`, or compare against the `all_tasks__ds_v<pin>` list for your pin; never mix versions in one comparison |
| Detection or A/D counts "changed" between versions | they did not; only T/L plans were regenerated | check the pin and whether a dataset was added (v1.2.0/v1.3.0) rather than changed |
| A `1.2.0` pin loads MAMA-MIA rows from an old cache | cache built before the withdrawal | delete that cache and the results; pin `1.2.1+` |
| `MedVision_DISABLE_SAMPLE_FILTERING=1` has no effect | value must compare equal to `"true"` case-insensitively | use `true`/`True`; the loader logs `quality/size sample filters bypassed`; a new cache id is used |
| Multi-instance numbers used for a leaderboard comparison | not comparable; V0 not trained for it | rerun with filtering on (unset the variable) |
| Coronal/sagittal T/L task lists have far fewer or zero samples at old pins | pre-1.4.0 pixel-count floor was plane-dependent; anisotropic slices lost ~22 % at 1.1.1 | use `1.1.1` for plane-OOD ablations (as the repository did) or `latest`; zero-sample subtasks are omitted from the catalogues |
| `[plan_utils] X: biometry plan v1.1.1 not found; using benchmark_plan_biometry_v1.0.0.json.gz instead` on stderr | ceiling fallback: the dataset never published that version (Ceph, FeTA) | expected; only investigate if the dataset should have the version |
| A dataset is missing from an old-version summary | it did not exist at that version (`dataset_exists_at` false) | expected; that is what the ceiling rule protects |
| `summarize_datasets` or a detection plan load is killed (OOM) | detection plans of whole-body CT datasets are hundreds of MB compressed | `--no_detection`, `--datasets <subset>`, or `inspect_benchmark_plan.py --no-load`; use the segmentation plan for image sizes (`FAMILY_TO_PLAN_TYPE`) |

## Parquet and visualization

| Symptom | Cause | Fix |
| --- | --- | --- |
| `limit_val_sample must be greater than 0` / `limit_train_sample cannot be 0` | `--val_sample_limit_per_task 0` or `--train_sample_limit_per_task 0` | use `-1` for no limit, `> 0` to cap; validation must be `> 0` |
| `Expected a Dataset object, but got <class 'datasets.dataset_dict.DatasetDict'>` | `--parquet_ds_path` pointed at a `save_to_disk` directory with several splits | pass the single `test.parquet` file |
| `Error processing sample i: ...` for many rows in `visualize_samples` | mixed-family parquet or wrong `--task_type` | build one parquet dir per family; use `Angle` and `Distance` separately for A/D |
| `High memory usage: 8x%` warnings, then a kill | too many parallel config loads | `--num_workers_concat_datasets 1`, per-subset limits |
| `wordcloud package not installed; skipping` | optional dependency for `summarize_datasets --viz` | `pip install wordcloud` only if the figure is needed |
| `refusing to compile from a possibly stale installed copy` | `compile_dataset_info.py` imported `medvision_ds` from site-packages | pass `--medvision_ds_src <source tree>`; verify `medvision_ds.__file__` |

## When to stop and report

- Any step that needs `SYNAPSE_TOKEN`, a private HF mirror, or gated-repo acceptance you do not have.
- Disk below the dataset's footprint (whole-body CT detection datasets are the largest) or a request to enable QC figures (~298 GB).
- A paused annotation (AFIDs/PDDCA/VerSe biometry): there is no workaround.
- A request to compare results across annotation versions or with filtering disabled: explain, do not compute.
