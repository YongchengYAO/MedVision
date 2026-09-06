# CLI Reference - `src/*.py` of the ablation folder

All programs are plain argparse scripts under `${ABLATION_DIR}/src/`. They are **reference-only** (not bundled):
they import the upstream BiomedParse package (`inference`, `utils`, `src.*` from `BIOMEDPARSE_DIR`), detectron2,
Hydra, Lightning, and `medvision_bm` / `medvision_ds` from the repository layout resolved by `_paths.py`. Flags
below were read from the argparse blocks and confirmed with `--help` for the six scripts whose module imports do
not need the upstream package (`finetune.py --help` needs `lightning` and the upstream `src` package).

## `_paths.py` (imported by every script)

| Symbol | Value / behaviour |
|---|---|
| `ABLATION_DIR` | parent of `src/` (the ablation folder) |
| `REPO_ROOT` | `ABLATION_DIR/../../..` |
| `BIOMEDPARSE_DIR` | `os.environ["BIOMEDPARSE_DIR"]` or `ABLATION_DIR/third_party/BiomedParse` |
| `add_medvision_to_path()` | prepends `REPO_ROOT/Data/src` then `REPO_ROOT/src` to `sys.path` (source first) when they exist |
| `add_biomedparse_to_path()` | prepends `BIOMEDPARSE_DIR`; exits with `Upstream BiomedParse not found at <dir>. Run setup.sh, or point BIOMEDPARSE_DIR at an existing checkout.` when the directory is missing |

## `prepare_test_data_detect.py` / `prepare_test_data_tl.py`

Export the MedVision test set to BiomedParse NPZ format, first N rows per subtask in HF native order.

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--tasks_json PATH` | yes | - | task list JSON (`tasks_MedVision-detect__train_SFT.json` / `tasks_MedVision-TL__train_SFT.json`) |
| `--output_dir DIR` | yes | - | directory for the `.npz` files (created) |
| `--processes N`, `-p N` | no | `None` (sequential) | multiprocessing workers (`> 1` enables the pool) |
| `--limit_per_subtask N` | no | `-1` (no limit) | first N rows of each `*_Test` config |
| `--filter_dataset LIST` | no | `None` | comma-separated dataset names; other tasks are skipped before loading |

Behaviour: for every key, loads `<key>_Test` (`split="test"`), concatenates, filters by `dataset_name`, and writes
one NPZ per row. Errors inside a worker raise `ValueError("Error processing row: ...")` (the run aborts).
Detection uses its own HU-window / percentile normalization; T/L uses `medvision_bm.sft.sft_utils.normalize_img`.
Exits with `No task in <json> matches dataset '<filter>'` when the filter removes everything.

**NPZ format**

| Key | Content |
|---|---|
| `imgs` | uint8 array `(1, H, W)` — the 2-D slice normalized to 0-255 with a leading singleton axis (`np.expand_dims(..., axis=0)`); consumers `np.squeeze` it first |
| `text_prompts` | dict `{"<label_id>": "<label name>", "instance_label": 0}` (pickled object) |
| `pixel_size` | in-plane pixel spacing (from the HF row) |
| `slice_dim`, `slice_idx` | slice axis and index |

File name: `<dataset_name>__<image basename without .nii.gz>__dim<slice_dim>__idx<slice_idx>__lbl<label>.npz`.
The same base name is used for masks (`<base>_pred_mask.nii.gz`), figures (`<base>.png`) and the `file` column of
the result CSVs; the evaluators use it as the key into the HF lookup.

## `prepare_finetune_data.py`

Builds the Track B training set with the SFT selection (see `overview-and-fairness.md`).

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--tasks_json PATH` | yes | - | detection SFT task list |
| `--output_dir DIR` | yes | - | PNG + JSON output root |
| `--train_limit N` | no | `110000` | training samples selected after the val carve-out |
| `--val_limit N` | no | `105` (launcher passes `1000`) | validation samples carved out with `group_train_test_split` |
| `--processes N`, `-p N` | no | `None` | multiprocessing workers |
| `--filter_dataset LIST` | no | `None` | comma-separated datasets whose `*_Train` configs are loaded (breaks the identical-110k guarantee) |

Output layout: `<output_dir>/{train,train_mask,val,val_mask}/<base>.png`, `<output_dir>/{train,val}.json` with
`{"annotations": [{"file_name": "<base>.png", "mask_file": "<base>.png", "class_prompts": {"1": "<label name>"},
"instance_label": true}, ...]}`. Images 512x512x3 uint8 (`cv2.INTER_LINEAR`), masks 512x512 uint8 with values
0/1 (`cv2.INTER_NEAREST`). Worker errors are collected (first three printed) and the row is skipped.

## `run_inference.py`

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--npz_dir DIR` | yes | - | prepared `.npz` files |
| `--seg_dir DIR` | yes | - | output directory for `.nii.gz` (created) |
| `--gpu STR` | no | `"0"` | value for `CUDA_VISIBLE_DEVICES` (set before torch import) |
| `--slice_batch_size N` | no | `4` | forwarded to `model(..., mode="eval", slice_batch_size=N)` |
| `--skip_existing` | no | off | skip samples whose `<base>_pred_mask.nii.gz` already exists |
| `--checkpoint PATH` | no | `None` | local `.ckpt`; when omitted downloads `biomedparse_v2.ckpt` from HF `microsoft/BiomedParse` into the HF cache |
| `--filter_dataset NAME` | no | `None` | single dataset name; keeps files starting with `<NAME>__` |

Pipeline per sample: Hydra `compose("biomedparse_3D")` from `<BIOMEDPARSE_DIR>/configs/model` ->
`model.load_pretrained(ckpt)` -> `process_input(imgs, 512)` -> forward -> `pred_gmasks` bicubic-resized to
512x512 (antialias) -> `postprocess(masks, object_existence)` -> `merge_multiclass_masks(masks, ids)` ->
`process_output(...)` -> saves `<base>.nii.gz` (image) and `<base>_pred_mask.nii.gz` (float32, identity affine).
Unreadable NPZ files are skipped with a warning.

## `eval_detect.py`

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--pred_dir DIR` | yes | - | `*_pred_mask.nii.gz` from inference |
| `--npz_dir DIR` | yes | - | prepared test NPZ |
| `--tasks_json PATH` | no | `<REPO_ROOT>/tasks_list/tasks_MedVision-detect__train_SFT.json` | task list used to rebuild the GT lookup (all rows) |
| `--output_dir DIR` | yes | - | metric files |
| `--fig_dir DIR` | yes | - | per-sample bounding-box figures |

No `--filter_dataset`. Every run rescans `pred_dir` and rewrites all outputs.

## `eval_tl.py`

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--pred_dir DIR` | yes | - | `*_pred_mask.nii.gz` |
| `--npz_dir DIR` | yes | - | prepared test NPZ |
| `--tasks_json PATH` | no | `<REPO_ROOT>/tasks_list/tasks_MedVision-TL__train_SFT.json` | GT lookup source |
| `--output_dir DIR` | yes | - | metric files |
| `--fig_dir DIR` | yes | - | ellipse figures, bucketed by MRE (`MRE01` ... `MRE09`) |
| `--filter_dataset NAME` | no | `None` | single dataset; results merged into existing output files (see `tracks.md`) |

## `finetune.py`

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--data_dir DIR` | yes | - | output of `prepare_finetune_data.py` (`train/`, `train_mask/`, `train.json`, `val/`, `val_mask/`, `val.json`) |
| `--checkpoint PATH` | yes | - | `biomedparse_v2.ckpt` |
| `--output_dir DIR` | yes | - | checkpoints + Lightning logs |
| `--batch_size N` | no | `4` | per GPU |
| `--lr F` | no | `1e-5` | AdamW learning rate (weight decay fixed at 0.01) |
| `--epochs N` | no | `10` | |
| `--gpus N` | no | `1` | devices; `> 1` selects `ddp_find_unused_parameters_true` (launch with `torchrun --nproc_per_node=N`) |
| `--num_workers N` | no | `4` | DataLoader workers |
| `--cls_coeff F` | no | `1.0` | loss coefficient (upstream `finetune_biomedparse.yaml`) |
| `--pos_weight F` | no | `3.0` | |
| `--edge_coeff F` | no | `1.0` | must be > 0 (upstream loss always sums `edge_loss`) |
| `--save_top_k N` | no | `-1` | `-1` keeps every epoch, `N` keeps the N best by `val_loss`, `0` keeps only `last.ckpt` |
| `--resume_from_checkpoint PATH` | no | `None` | Lightning `ckpt_path` for resuming |

Fixed: `precision="bf16-mixed"`, `gradient_clip_val=5.0`, `L.seed_everything(SEED, workers=True)`, Hydra
`compose("biomedparse", overrides=["+edge_queries=4"])` from `<BIOMEDPARSE_DIR>/configs/model`, checkpoint
filenames `biomedparse_medvision_{epoch:02d}_{val_loss:.4f}.ckpt` + `last.ckpt`.

## Output-file inventory

### `results/detect/<model>/`

| File | Content |
|---|---|
| `seg_masks/<base>.nii.gz`, `seg_masks/<base>_pred_mask.nii.gz` | inference inputs/outputs |
| `eval_biomedparse_medvision_detect_success_predictions.txt` / `..._failure_predictions.txt` | absolute mask paths; failure = all-zero mask |
| `eval_biomedparse_medvision_detect_results.csv` | one row per success: `file, label_name, region_key, label_group, coords_model, coords_gt, slice_dim, slice_idx, pixel_size, avgMAE, F1, IoU, Precision, Recall` |
| `eval_biomedparse_medvision_detect_group_summary.csv` | per `label_group` (anatomy / tumor_lesion / miscellaneous): counts, success_rate, F1/IoU/Precision/Recall mean+std over successes, `IoU_gt_0.5_rate` (over successes) |
| `eval_biomedparse_medvision_detect_metrics_dist.png` | 2x2 histograms (F1, IoU, Precision, Recall) |
| `summary_metrics_detect_Task.json` | per region key: `avgMAE`, `IoU`, `F1`, `Precision`, `Recall`, `SuccessRate`, `num_samples`, `MAE<0.1..1.0`, `IoU>0.5..0.9`, `F1>...`, `Precision>...`, `Recall>...` (benchmark denominators) |
| `summary_metrics_anatomy_vs_lesion_detect_Task.json` | `anatomy` / `T/L` groups (regions with >= 50 samples): weighted `mean_metrics`, `regions`, `detailed_data` |
| `summary_detection_task.txt` | human-readable summary in the benchmark's format (`Model: <model>`, `ANATOMY (...)`, `T/L (...)` lines) |

### `results/tl/<model>/`

| File | Content |
|---|---|
| `eval_biomedparse_medvision_tl_success_predictions.txt` / `..._failure_predictions.txt` | mask paths (ellipse-fit failures are moved to the failure list) |
| `eval_biomedparse_medvision_tl_results.csv` | per success: `file, label_name, major_axis_model, minor_axis_model, major_axis_gt, minor_axis_gt, slice_dim, slice_idx, pixel_size, mae, mre` |
| `eval_biomedparse_medvision_tl_group_summary.csv` | one row: `mae_*`, `mre_*` stats, `mre_lt_0_{1,2,3}_pct` (over successes), `total_files`, `success_files`, `success_pct`, `failure_files`, `failure_pct` |
| `eval_biomedparse_medvision_tl_metrics_dist.png` | MAE and MRE histograms |
| `summary_metrics_tl_Task.json` | per label name: `avgMAE`, `avgMRE`, `SuccessRate`, `num_samples`, `MRE<0.1..0.5` (total-count denominator) |
| `summary_tl_task.txt` | labels merged with `label_map_rename`; `Weighted Average MAE/MRE/SR`, `Weighted MRE<0.1/0.2/0.3`, per-label table (`nMAE` reported as `N/A`) |

### Figures

- Detection: `figures/detect/<model>/<base>.png` - GT box green, predicted box red, title with P/R/F1/IoU.
- T/L: `figures/tl/<model>/MRE0<k>/<base>.png`, `k = min(int(mre / 0.1) + 1, 9)` - predicted mask contour
  (`#97D540`), GT mask contour (cyan, when the local GT NIfTI exists), major axis `#F37020`, minor axis `#FBBC05`,
  white landmark dots, L-shaped scale bar. Both at `dpi=100`, `bbox_inches="tight"`.
