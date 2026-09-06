---
name: dataset-and-tasks
description: "Selects, downloads, inspects and builds MedVision data: dataset-config naming, task-list JSONs (eval -CoT vs SFT names, the BoxCoordinate/BoxSize bridge), the download_datasets CLI and load_dataset download modes, annotation versions 1.0.0-1.4.0 with the required MedVision_PLANNER_VERSION pin, the Data/ layout and benchmark plans, parquet building and sample visualization."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision Dataset and Task Lists

Use this sub-skill when an agent must decide *which* MedVision data to use, name it correctly, fetch it, verify what landed on disk, or turn it into parquet for inspection. It covers the Hugging Face dataset `YongchengYAO/MedVision` (annotations only; the loader script `MedVision.py` fetches raw images through `medvision_ds`), the repository's task-list JSONs, and the `medvision_bm` dataset tooling.

## Route Here

- Translate between a **task name** (`KiPA22_TumorLesionSize_Task01_Axial-CoT`) and a **dataset config** (`KiPA22_TumorLesionSize_Task01_Axial_Test`), including the `BoxCoordinate` (task) vs `BoxSize` (config) bridge and the `-CoT` suffix.
- Pick or build a task list (eval `-CoT`, SFT `__train_SFT`, plane/target OOD, per-version `all_tasks__ds_v*` catalogues) and derive configs for a split without downloading.
- Download data with `python -m medvision_bm.benchmark.download_datasets` or `load_dataset(...)`, choose `download_mode`, force a raw re-download, handle gated datasets and QC figures.
- Pin an annotation version (`MedVision_PLANNER_VERSION`, `MedVision_ACK_RELEASE`), explain why sample counts differ from the leaderboard (v1.0.0), and read the per-version release changes.
- Inspect `<data_dir>/Datasets/<dataset>/benchmark_plan_*.json.gz` offline with the `plan_utils` API, or build/visualize a parquet snapshot of a task list.

## Route Elsewhere

- Installing `medvision_bm`, `medvision_ds` (`mvbm install mvds -d <data_dir>`), pins and requirement files: `../environment-setup/SKILL.md`.
- Running evaluations with the task lists: `../benchmark-evaluation/SKILL.md`.
- SFT dataset construction internals (`load_split_limit_dataset`, PNG cache, `prepared_ds_dir`): `../sft/SKILL.md`.
- verl parquet builders for RFT: `../rft/SKILL.md`.
- Adding new task YAMLs or datasets to the evaluator: `../extending-models-and-tasks/SKILL.md`.
- Shared vocabulary and cross-cutting fixes: `../../references/concepts-and-glossary.md`, `../../references/troubleshooting.md`.

## Ten Facts to Hold Before Acting

1. Config name = `{dataset}_{annotation-type}_{task-ID}_{slice}_{split}`; annotation types `BoxSize`, `TumorLesionSize`, `BiometricsFromLandmarks`, `MaskSize`; slices `Sagittal|Coronal|Axial`; splits `Train|Test`; `Task01` is dataset-local.
2. Task lists are JSON `{"task name": count}`; only the keys are read (`load_tasks`); counts are informational.
3. `tasks_to_configs(tasks, split)` appends `_Train`/`_Test` and rewrites `BoxCoordinate`->`BoxSize`; it does **not** strip `-CoT`, and no HF config carries `-CoT`, so eval lists must have the suffix removed before deriving configs (the bundled scripts do this).
4. Eval task names end in `-CoT`; SFT names do not and already use `BoxSize`; both point at the same configs. The SFT loader also accepts `_BoxCoordinate_` keys and rewrites them.
5. `MedVision_PLANNER_VERSION` is **required** (loader raises `MedVision: annotation version selection required`). It is a per-dataset *ceiling*: each dataset loads its newest annotation at or below the pin. `latest` currently resolves to `1.4.0`.
6. Leaderboard numbers use annotation **v1.0.0**. Pinning below a dataset's newest annotation needs `MedVision_ACK_RELEASE=1.4.0` (blanket) or the dataset's own newest version. Only T/L annotations change across versions; Detection and A/D are byte-identical.
7. Any config download fetches the **whole source dataset** (both splits, all planes); `--split` only selects which Arrow build to materialise.
8. Raw data refresh requires `download_mode="force_redownload"` **and** (`MedVision_FORCE_DOWNLOAD_DATA=True` or removing the dataset's entry from `<data_dir>/.downloaded_datasets.json`); a warm Arrow cache never re-runs the script.
9. Gated sources: FeTA24 needs `SYNAPSE_TOKEN`; SKM-TEA and ToothFairy2 need your own private HF mirror via `MedVision_SKMTEA_HF_ID` / `MedVision_ToothFairy2_HF_ID` plus `HF_TOKEN`; AbdomenAtlas1.0Mini is HF-gated (accept terms + `HF_TOKEN`).
10. QC figures are opt-in (`MedVision_DOWNLOAD_QC_FIGURES=True`, default off) and weigh ~298 GB against ~3 GB of annotations. `datasets` must stay at `3.6.0` (`trust_remote_code` is gone in 4.x).

## Fast Path

1. Set the environment before any `datasets` import: `export MedVision_DATA_DIR=<data_dir>` and `export MedVision_PLANNER_VERSION=latest` (or `1.0.0` + `MedVision_ACK_RELEASE=1.4.0` to reproduce the leaderboard).
2. Check the task list and the configs it implies: `python scripts/list_tasks.py --tasks-json <tasks.json> --split test`.
3. Preview, then run the download: `bash scripts/download_datasets.sh --data-dir <data_dir> --tasks-json <tasks.json> --dry-run` (drop `--dry-run` to execute; network + disk heavy).
4. Verify what landed: `python scripts/inspect_benchmark_plan.py --dataset-dir <data_dir>/Datasets/<dataset> --plan-type biometry` (offline).
5. Optional snapshot: `bash scripts/build_parquet_ds.sh --data-dir <data_dir> --out-dir <out> --tasks-json-tl <tasks.json> --visualize --dry-run`.

## References

- Read `references/concepts.md` for the naming convention, loader task types, the fields each config returns, single- vs multi-instance filtering rules, and the annotation-version history (1.0.0-1.4.0, paused/withdrawn entries).
- Read `references/task-lists.md` for the JSON shape, every shipped list and its role, name-to-config derivation, the `-CoT` and `BoxCoordinate` rules, SFT namespace, OOD lists and the per-version `all_tasks__ds_v*` catalogues.
- Read `references/downloading.md` for the download CLI flags, `load_dataset` examples, `download_mode` semantics, every environment variable the loader reads, the `.downloaded_datasets.json` tracker, tokens for gated datasets and QC figures.
- Read `references/data-layout.md` for the on-disk `Data/` tree, benchmark-plan file naming and schema, and the `medvision_bm.utils.plan_utils` API including the `resolve_plan_path` ceiling rule.
- Read `references/parquet-and-visualization.md` for `build_parquet_ds` flags, the four-level sample-limit hierarchy, `visualize_samples`, and the `ds_utils` functions behind them.
- Read `references/maintainer-workflows.md` when regenerating `dataset-info/` catalogues (`configs_to_tasks`, size probes, `summarize_datasets`, `regen_all_tasks`, `compile_dataset_info` and its source-tree guard).
- Read `references/troubleshooting.md` for loader error banners, config-not-found errors, 401/403 on gated data, stale caches, version and filtering confusion, and when to stop.

## Scripts

- Run `python scripts/list_tasks.py --help`; use it to print task names, counts and derived configs for a split, rewrite the plane (`--plane Coronal`) or the `-CoT` suffix (`--cot add|strip`), and emit `--json`. Offline.
- Run `bash scripts/download_datasets.sh --help`; it wraps `python -m medvision_bm.benchmark.download_datasets`, checks the required environment, trims token whitespace, strips `-CoT` suffixes from eval lists into a temp copy, and prints the command (`--dry-run`). Network + large disk.
- Run `python scripts/inspect_benchmark_plan.py --help`; it lists plan files and versions for a dataset directory, resolves the plan for a pin, and prints per-task train/test case counts and first-case keys. Offline; large detection plans are skipped unless `--max-load-mb` is raised.
- Run `bash scripts/build_parquet_ds.sh --help`; it wraps `medvision_bm.dataset.build_parquet_ds` with explicit paths and small default limits, optionally followed by `visualize_samples`. Needs downloaded data (or network).

## Safe Operating Rules

- Never start a download, parquet build or `load_dataset` call unless the user asked for it and named `<data_dir>`; state the disk cost (a full copy is ~1 TB) and that one config pulls the whole dataset.
- Never export `MedVision_DOWNLOAD_QC_FIGURES=True` on the user's behalf.
- Do not set `MedVision_DISABLE_SAMPLE_FILTERING=true` for leaderboard comparisons; multi-instance samples are not comparable.
- Treat `<data_dir>/Datasets/`, `.downloaded_datasets.json` and the HF cache as data: read them, do not edit them by hand except the documented tracker-entry removal.
- Keep `MedVision_PLANNER_VERSION` fixed for the life of a study and record it with results.

## Verification Mindset

- Before download: `list_tasks.py` shows the expected configs and no name lacks a plane token.
- After download: `inspect_benchmark_plan.py` finds the expected plan version and non-zero test cases; `.downloaded_datasets.json` has `dataset_<name>` set.
- Before analysis: confirm the loader banner printed the intended `MedVision_PLANNER_VERSION` and that `len(ds)` matches the count in the matching `all_tasks__ds_v<version>` list.
