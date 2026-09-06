# Task Lists, Task Names and Dataset Configs

## JSON shape

Every task list is a flat JSON object mapping a task name to an informational sample count:

```json
{
    "KiPA22_TumorLesionSize_Task01_Axial-CoT": 866,
    "KiTS23_TumorLesionSize_Task01_Axial-CoT": 1859
}
```

`medvision_bm.utils.utils.load_tasks(json_file_path)` returns `list(json.load(f).keys())` in file order and prints `Found N tasks to process: [...]`. The values are never read by the pipeline (they are counts at the time the list was generated and go stale when the annotation version changes).

## Shipped lists (repository `tasks_list/`)

| File | Role | Names |
| --- | --- | --- |
| `tasks_MedVision-AD-CoT.json` | A/D evaluation (5 tasks: Ceph-Biometrics-400 distance + angle sagittal; FeTA24 sagittal/coronal/axial) | `-CoT` suffix |
| `tasks_MedVision-TL-CoT.json` | T/L evaluation (10 axial tasks: autoPET-III, BraTS24 x3, HNTSMRG24, KiPA22, KiTS23, MSD x3) | `-CoT` |
| `tasks_MedVision-detect-CoT.json` | Detection evaluation (28 axial tasks) | `-CoT`, `BoxCoordinate` |
| `tasks_MedVision-{AD,TL,detect}__train_SFT.json` | SFT training data (same tasks, no suffix; detection uses `BoxSize`) | plain |
| `OOD/tasks_MedVision-{TL,detect}-CoT-planeOOD.json` | plane-OOD evaluation: axial tasks re-issued on `Sagittal` and `Coronal` — all 10 for T/L (20 entries), 24 of the 28 for detection (48 entries; BraTS24 Task06, CAMUS Task01 and MSD Task06/Task12 excluded) | `-CoT` |
| `OOD/tasks_MedVision-{TL,detect}-CoT-taskOOD.json` | target-OOD evaluation: unseen task IDs (e.g. `BraTS24_TumorLesionSize_Task02_Axial-CoT`); the detection list mixes `-CoT` sagittal/coronal names with plain axial names | mixed |
| `experimental/*.json` | legacy and ablation lists (`-scaledPS`, `-woInstruct`, `-VP`, `-VP-woMedImg`, `-OOD`, ...) not part of the main pipeline | various |

Per-version catalogues live in `dataset-info/all_tasks__ds_v{1.0.0,1.1.0,1.1.1,1.2.0,1.3.0,1.4.0}/`: `tasks_MedVision-{detect,TL}-CoT__{Axial,Coronal,Sagittal,AllSlices}__{Test,Train}.json` and `tasks_MedVision-AD-CoT__AllSlices__{Test,Train}.json`. Despite the `-CoT` in every filename, only the `__Test` lists carry the `-CoT` suffix on names; `__Train` lists are plain (SFT-style). A subtask appears only if that pin can load it (zero-sample subtasks are omitted; paused datasets are absent). The `readme.md` in each folder states coverage; v1.4.0: Detection 29 datasets / 378 subtasks, T/L 12 datasets / 228 subtasks, A/D 2 datasets / 10 subtasks.

## Task name -> dataset config

`medvision_bm.utils.data_utils.tasks_to_configs(tasks, split)`:

1. asserts `split.lower() in {"train", "test"}`;
2. appends `_Train` or `_Test` to each name;
3. replaces `BoxCoordinate` with `BoxSize`.

It does **not** remove `-CoT` or any other variant suffix, and the loader defines no config containing `-CoT`. Feeding an eval list straight through therefore yields `KiPA22_TumorLesionSize_Task01_Axial-CoT_Test`, which `load_dataset` rejects with `ValueError: BuilderConfig '...' not found. Available: [...]`. Rule: the config is the name **up to and including the plane token**, plus `_<Split>`, with `BoxCoordinate -> BoxSize`. `scripts/list_tasks.py` applies exactly this rule; `scripts/download_datasets.sh` writes a stripped copy of an eval list before calling the CLI.

Why task names say `BoxCoordinate` while configs say `BoxSize`: in the task namespace `BoxSize` is reserved for (future) mask-size estimation tasks, so detection tasks were named `BoxCoordinate`; the dataset never adopted that name.

The reverse mapping (`medvision_bm.utils.configs_to_tasks.config_to_task(config, cot)`) strips the trailing `_Train`/`_Test`, replaces `BoxSize -> BoxCoordinate`, and appends `-CoT` when `cot=True`.

## What `-CoT` means

Both eval and SFT names point at the same configs; the suffix selects a different **task YAML** (prompt with chain-of-thought instructions, GT formatting, metrics), not different data. Each eval YAML declares the config explicitly, for example `task: KiPA22_TumorLesionSize_Task01_Axial-CoT-woInstruct` with `dataset_name: KiPA22_TumorLesionSize_Task01_Axial_Test`. MedVision-V0 also trains with CoT; the missing suffix on SFT names is a legacy inconsistency, not a functional difference.

## SFT namespace

SFT lists use the dataset's own family tokens: `BoxSize`, `TumorLesionSize`, `BiometricsFromLandmarks`. The SFT/parquet loaders (`load_split_limit_dataset` and `load_split_limit_dataset_tr_val_ts`) take `tag_ds=<family token>` and

- build config names as `task + "_Train"` / `task + "_Test"` (no `-CoT` handling: never pass an eval list);
- recover the dataset name as `task.split(f"_{tag_ds}")[0]` (used to check `dataset_<name>` in `.downloaded_datasets.json`);
- **`load_split_limit_dataset` only** (SFT, `sft_utils.py:2133-2141`): when `tag_ds == "BoxSize"`, rewrite
  `_BoxCoordinate_ -> _BoxSize_` in the keys, so an `all_tasks__ds_v*` detection `__Train` list can be passed
  straight in.

> The parquet loader does **not** do this. `load_split_limit_dataset_tr_val_ts` reads the keys verbatim
> (`ds_utils.py:181-183`) and appends `_Train` (`ds_utils.py:231`); there is no `BoxCoordinate` handling anywhere
> under `src/medvision_bm/dataset/`. Feeding it a detection `__Train` list whose keys still say `_BoxCoordinate_`
> fails with `BuilderConfig '<dataset>_BoxCoordinate_Task01_Axial_Train' not found`. Rewrite the keys to
> `_BoxSize_` first — `../scripts/list_tasks.py` does it (line 65) — before passing the list to
> `--tasks_list_json_path_detect`.

`build_parquet_ds` hard-codes the tag per family: A/D `BiometricsFromLandmarks`, T/L `TumorLesionSize`, detection `BoxSize`.

## Deriving lists without a download

- Eval list for a split you do not have: take the `__Test` list of the matching `all_tasks__ds_v<version>/` folder for the plane you need (Axial/Coronal/Sagittal/AllSlices).
- SFT list -> eval names for another plane: rewrite the plane token and add `-CoT` (`scripts/list_tasks.py --plane Coronal --cot add`). Check that the resulting subtask exists in the version catalogue: a plane can have zero samples at an older pin (the pre-1.4.0 pixel floor rejected every off-axial cluster on some tasks), and the six datasets re-split in v1.4.0 have different test cases than before.
- Configs for a whole family/plane: filter a `ConfigurationsList_*.csv` on `parts[1]` (family), `parts[-2]` (plane), `parts[-1]` (split), the same positional rule `configs_to_tasks` uses.
