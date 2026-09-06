# Parquet Snapshots and Sample Visualization

Use these when you want a self-contained, shuffled train/validation/test snapshot of one or more task lists (for inspection, quick experiments, or figures) rather than the SFT/RFT-specific builders (`../../sft/SKILL.md`, `../../rft/SKILL.md`).

## `python -m medvision_bm.dataset.build_parquet_ds`

Requires `MedVision_DATA_DIR` (asserted), `MedVision_PLANNER_VERSION`, and an existing `<data_dir>/.downloaded_datasets.json` (read to decide worker count; a fresh data root fails with `FileNotFoundError` until at least one config was downloaded). Loads the `_Train` configs of every task, carves a group-aware validation split (volumes kept together via `image_file`, stratified by `dataset_name`, `SEED=1024`), then loads the `_Test` configs, shuffles, and writes `train.parquet`, `validation.parquet`, `test.parquet` into `--parquet_ds_dir`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--parquet_ds_dir` | none | output directory |
| `--ds_download_mode` | `reuse_dataset_if_exists` | passed to `load_dataset` (`reuse_cache_if_exists`, `force_redownload`) |
| `--tasks_list_json_path_AD` / `_TL` / `_detect` | none | task lists per family (SFT-style names); each present list adds a family; `tag_ds` is fixed per family (`BiometricsFromLandmarks`, `TumorLesionSize`, `BoxSize`) |
| `--num_workers_concat_datasets` | 4 | parallel config loads, clamped to CPUs and task count, forced to 1 if any dataset is not yet in the tracker |
| `--train_sample_limit_per_subset` / `--test_sample_limit_per_subset` | -1 | Level 1: cap per config before merging |
| `--train_sample_limit_per_task` / `--val_sample_limit_per_task` / `--test_sample_limit_per_task` | -1 / **100** / -1 | Level 2: cap per family after merging; validation size must be > 0 |
| `--{train,val,test}_sample_limit_task_{AD,Detection,TL}` | -1 | Level 3: per-family override when > 0 |
| `--train_sample_limit` / `--val_sample_limit` / `--test_sample_limit` | -1 | Level 4: cap on the concatenated total |

Order: Level 1 while loading -> validation carve-out -> Level 2/3 caps -> concatenate families -> Level 4. `-1` means no limit; `0` for the train limit is rejected (`limit_train_sample cannot be 0`). The repository's own recipe builds one parquet dir per family (`medvision_Detection`, `medvision_AD`, `medvision_TL`) with `--num_workers_concat_datasets 1`.

`scripts/build_parquet_ds.sh` wraps this with explicit `--data-dir`/`--out-dir`, small defaults (20 train / 10 test per subset, 5 validation per family), one worker, and `--dry-run`.

## `python -m medvision_bm.dataset.visualize_samples`

| Flag | Default | Meaning |
| --- | --- | --- |
| `--parquet_ds_path` (required) | | a `.parquet` file (loaded with `load_dataset("parquet", split="train")`) or a `save_to_disk` directory; must yield a single `Dataset` |
| `--fig_dir` (required) | | output directory |
| `--num_samples` | 10 | first N rows |
| `--task_type` | `Detection` | `Detection` (boxes), `Distance`, `Angle` (landmark visual prompts), `TL` (ellipse axes) |

Uses the evaluator's `doc_to_visual_*` renderers from the vendored `lmms_eval` task utilities, so the figure is exactly what a model sees. Output names: `<dataset_name>__<image basename>__dim<slice_dim>__idx<slice_idx>[_k].png`. Rows of another family raise per-sample errors that are printed and skipped, so run it on a single-family parquet. A/D parquet files need two runs (`Angle`, `Distance`).

## `medvision_bm.dataset.ds_utils`

- `parse_sample_limits_tr_val_ts(**kwargs)` -> 9-tuple `(train, val, test) x (AD, detect, TL)`: task-specific limit if `> 0` else per-task limit; a family whose task JSON is `None` gets train/val limits 0.
- `load_split_limit_dataset_tr_val_ts(tasks_list_json_path, limit_train_sample, limit_val_sample, limit_test_sample, limit_train_sample_per_subset=None, limit_test_sample_per_subset=None, num_workers_concat_datasets=4, tag_ds=None, download_mode="reuse_dataset_if_exists")` -> `DatasetDict{train, validation, test}`; asserts `limit_val_sample > 0`, `limit_train_sample != 0`, `tag_ds` given, `MedVision_DATA_DIR` set; raises `RuntimeError` if any config failed (after 5 retries with exponential backoff inside `_load_single_dataset`).
- `build_parquet_dataset(*, tasks_list_json_path, limit_train_sample, limit_val_sample, limit_test_sample, limit_train_sample_per_subset, limit_test_sample_per_subset, num_workers_concat_datasets=4, tag_ds=None, download_mode=...)` -> the same `DatasetDict` (thin keyword-only wrapper used by the CLI).

`_load_single_dataset` (shared with SFT) calls `load_dataset("YongchengYAO/MedVision", name=config, trust_remote_code=True, split=..., streaming=False, download_mode=...)` and applies the per-subset `limit` with `select(range(limit))`.
