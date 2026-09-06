# Tracks A and B - launchers, inputs, outputs

The nine launchers live in `${ABLATION_DIR}/scripts/{eval,finetune}/` of the repository's
`script/ablation/biomedparse/` folder. They are documented here **reference-only** and not bundled: each one
resolves `ABLATION_DIR`/`REPO_ROOT` from its own location, sources the shared `_env.sh` from its parent directory (conda activation), and calls
`${ABLATION_DIR}/src/*.py`, which in turn import the upstream checkout and `medvision_bm` from the repository
source tree. Copy the whole folder (inside a MedVision checkout) and run the launchers from any working directory.

Every launcher: `TASK` must be set before `source _env.sh`; all steps **require a CUDA GPU** (inference/training) or
at least the conda env + data directory (prepare/eval). Inference is single-GPU only (no DP/DDP path upstream).

Common variables: `${ABLATION_DIR}` (ablation folder), `${REPO_ROOT}` (`${ABLATION_DIR}/../../..`),
`${PRETRAINED_CKPT}` = `${ABLATION_DIR}/models/biomedparse_v2.ckpt`, task JSON =
`${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json` (detect) or
`${REPO_ROOT}/tasks_list/tasks_MedVision-TL__train_SFT.json` (tl).

## Track A - evaluate the pretrained model

Run the three steps for `TASK=detect`, then again with `TASK=tl`.

### A1 `scripts/eval/1_prepare_test_data.sh` - HF test configs -> NPZ

| | |
|---|---|
| Launcher | `bash ${ABLATION_DIR}/scripts/eval/1_prepare_test_data.sh` / `TASK=tl bash ...` |
| Fixed settings | `LIMIT_PER_SUBTASK=1000`, `N_PROCESSES=32` |
| Command | `python ${ABLATION_DIR}/src/prepare_test_data_${TASK}.py --tasks_json <task json> --output_dir ${ABLATION_DIR}/data/test_npz/${TASK} --limit_per_subtask 1000 -p 32` |
| Inputs | HF `YongchengYAO/MedVision` `*_Test` configs (network or cache), local NIfTI under `MedVision_DATA_DIR` |
| Outputs | `data/test_npz/${TASK}/<dataset>__<image basename>__dim<slice_dim>__idx<slice_idx>__lbl<label>.npz` (keys `imgs` uint8 (1, H, W), `text_prompts`, `pixel_size`, `slice_dim`, `slice_idx`) |
| Needs | conda env, data directory, `MedVision_PLANNER_VERSION`, `MedVision_ACK_RELEASE` (tl) |

### A2 `scripts/eval/2_inference.sh` - masks with the pretrained weights

| | |
|---|---|
| Launcher | `bash ${ABLATION_DIR}/scripts/eval/2_inference.sh` / `TASK=tl GPU=1 bash ...` |
| Fixed settings | `GPU=${GPU:-0}`, `SLICE_BATCH_SIZE=4`; calls `ensure_pretrained_ckpt` first |
| Command | `python ${ABLATION_DIR}/src/run_inference.py --checkpoint ${PRETRAINED_CKPT} --npz_dir ${ABLATION_DIR}/data/test_npz/${TASK} --seg_dir ${ABLATION_DIR}/results/${TASK}/pretrained/seg_masks --gpu ${GPU} --slice_batch_size 4 --skip_existing` |
| Inputs | `data/test_npz/${TASK}/*.npz`, `models/biomedparse_v2.ckpt`, upstream `configs/model/biomedparse_3D.yaml` |
| Outputs | `results/${TASK}/pretrained/seg_masks/<basename>.nii.gz` (input image) and `<basename>_pred_mask.nii.gz` (float32 mask, identity affine) |
| Resume | `--skip_existing` skips samples whose `_pred_mask.nii.gz` exists |
| Needs | **GPU** (falls back to CPU if `torch.cuda.is_available()` is False, impractically slow) |

### A3 `scripts/eval/3_eval.sh` - MedVision metrics

| | |
|---|---|
| Launcher | `bash ${ABLATION_DIR}/scripts/eval/3_eval.sh` / `TASK=tl bash ...` |
| Fixed settings | `MODEL=pretrained` |
| Command | `python ${ABLATION_DIR}/src/eval_${TASK}.py --pred_dir ${ABLATION_DIR}/results/${TASK}/pretrained/seg_masks --npz_dir ${ABLATION_DIR}/data/test_npz/${TASK} --output_dir ${ABLATION_DIR}/results/${TASK}/pretrained --fig_dir ${ABLATION_DIR}/figures/${TASK}/pretrained` (the launcher relies on the script's default `--tasks_json`) |
| Inputs | masks from A2, NPZ from A1, HF `*_Test` rows (all rows, for GT/pixel sizes/modality), local GT masks (T/L figure overlay only) |
| Outputs (detect) | `eval_biomedparse_medvision_detect_{success,failure}_predictions.txt`, `..._detect_results.csv`, `..._detect_group_summary.csv`, `..._detect_metrics_dist.png`, `summary_metrics_detect_Task.json`, `summary_metrics_anatomy_vs_lesion_detect_Task.json`, `summary_detection_task.txt`; figures `figures/detect/pretrained/<basename>.png` |
| Outputs (tl) | `eval_biomedparse_medvision_tl_{success,failure}_predictions.txt`, `..._tl_results.csv`, `..._tl_group_summary.csv`, `..._tl_metrics_dist.png`, `summary_metrics_tl_Task.json`, `summary_tl_task.txt`; figures `figures/tl/pretrained/MRE0<k>/<basename>.png` |
| Needs | conda env, data directory, network/cache for HF rows; CPU only |

## Track B - fine-tune, then re-evaluate

### B1 `scripts/finetune/1_prepare_finetune_data.sh` - HF train configs -> PNG + masks + JSON

| | |
|---|---|
| Launcher | `bash ${ABLATION_DIR}/scripts/finetune/1_prepare_finetune_data.sh` |
| Fixed settings | `TRAIN_LIMIT=110000`, `VAL_LIMIT=1000`, `N_PROCESSES=64` |
| Command | `python ${ABLATION_DIR}/src/prepare_finetune_data.py --tasks_json ${REPO_ROOT}/tasks_list/tasks_MedVision-detect__train_SFT.json --output_dir ${ABLATION_DIR}/data/finetune/detect --train_limit 110000 --val_limit 1000 --processes 64` |
| Inputs | all 28 axial Detection `*_Train` configs (full pool, no limit), local NIfTI images and masks |
| Outputs | `data/finetune/detect/{train,train_mask,val,val_mask}/<same basename>.png` (512x512; images 3-channel, masks 0/1 uint8) and `train.json`, `val.json` (`{"annotations":[{"file_name","mask_file","class_prompts":{"1":"<label name>"},"instance_label":true}]}`) |
| Needs | conda env, data directory, lots of disk (110k PNG pairs) and CPU |

### B2 `scripts/finetune/2_finetune.sh` - PyTorch Lightning fine-tuning

| | |
|---|---|
| Launcher | `bash ${ABLATION_DIR}/scripts/finetune/2_finetune.sh` / `CUDA_VISIBLE_DEVICES=2,3 bash ...` |
| Fixed settings | `DATA_DIR=data/finetune/detect`, `OUTPUT_DIR=models/finetuned-detect`, `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}`, `N_GPUS=2`, `BATCH_SIZE=4` (per GPU), `LR=1e-5`, `EPOCHS=10`, `NUM_WORKERS=4`, `CLS_COEFF=1.0`, `POS_WEIGHT=3.0`, `EDGE_COEFF=1.0`, `SAVE_TOP_K=-1`, `RESUME_FROM_CHECKPOINT=""`; calls `ensure_pretrained_ckpt` |
| Command | `torchrun --nproc_per_node=2 ${ABLATION_DIR}/src/finetune.py --data_dir ... --checkpoint ${PRETRAINED_CKPT} --output_dir ... --batch_size 4 --lr 1e-5 --epochs 10 --gpus 2 --num_workers 4 --cls_coeff 1.0 --pos_weight 3.0 --edge_coeff 1.0 --save_top_k -1` (plain `python` when `N_GPUS=1`; `--resume_from_checkpoint <ckpt>` appended when set) |
| Inputs | B1 output, `models/biomedparse_v2.ckpt`, upstream `configs/model/biomedparse.yaml` (+`edge_queries=4`), upstream `src.datasets.biomedparse_dataset.BiomedParseDataset`, `src.losses.{biomedparse_loss.BiomedParseLossCLS, medsam_loss.MedSamLoss}` |
| Outputs | `models/finetuned-detect/biomedparse_medvision_epoch=XX_val_loss=Y.YYYY.ckpt` for every epoch (`save_top_k=-1`) plus `last.ckpt` (safe resume target), Lightning logs in the same folder |
| Needs | **2 GPUs** (bf16-mixed); `N_GPUS` in the launcher must equal the number of visible GPUs |

Fine-tuning recipe as implemented in `finetune.py`: `L.seed_everything(SEED, workers=True)`; validation set = the
prepared `val/` split (never a random split of `train/`); AdamW (`lr`, `weight_decay=0.01`); loss
`BiomedParseLossCLS(MedSamLoss(reduction="none"), cls_coeff, pos_weight, edge_coeff)`; `precision="bf16-mixed"`;
`gradient_clip_val=5.0`; strategy `ddp_find_unused_parameters_true` when `--gpus > 1`; `ModelCheckpoint(monitor
="val_loss", mode="min", save_last=True, filename="biomedparse_medvision_{epoch:02d}_{val_loss:.4f}")` and a
`LearningRateMonitor`. The pretrained state dict is loaded with the `model.` prefix stripped and `strict=False`.
Loss coefficients match upstream `finetune_biomedparse.yaml`; `edge_coeff` must stay > 0 because the upstream
loss references `edge_loss` unconditionally.

### B3 `scripts/finetune/3_inference.sh` - masks with a fine-tuned checkpoint

| | |
|---|---|
| Launcher | `bash ${ABLATION_DIR}/scripts/finetune/3_inference.sh` / `TASK=tl bash ...` / `CHECKPOINT=${ABLATION_DIR}/models/finetuned-detect/biomedparse_medvision_epoch=03_val_loss=0.4460.ckpt bash ...` |
| Fixed settings | `GPU=${GPU:-0}`, `SLICE_BATCH_SIZE=4`, `CHECKPOINT=${CHECKPOINT:-${ABLATION_DIR}/models/finetuned-detect/last.ckpt}` |
| Command | `python ${ABLATION_DIR}/src/run_inference.py --checkpoint ${CHECKPOINT} --npz_dir ${ABLATION_DIR}/data/test_npz/${TASK} --seg_dir ${ABLATION_DIR}/results/${TASK}/finetuned/seg_masks --gpu ${GPU} --slice_batch_size 4 --skip_existing` |
| Inputs | Track A step A1 NPZ for the task (run A1 for `tl` too), the chosen checkpoint |
| Outputs | `results/${TASK}/finetuned/seg_masks/*` |
| Needs | **GPU** |

The detection-fine-tuned model is scored on **both** tasks (`TASK=detect` and `TASK=tl`); there is no
T/L-specific training.

### B4 `scripts/finetune/4_eval.sh` - metrics for the fine-tuned model

Identical to A3 with `MODEL=finetuned`: outputs in `results/${TASK}/finetuned/` and `figures/${TASK}/finetuned/`.

## Checkpoint choice

- Default of B3 is `last.ckpt` (epoch 10 of a completed run).
- The shipped/paper numbers use the **best-validation** checkpoint `epoch=03` (val_loss 0.446):
  `CHECKPOINT=${ABLATION_DIR}/models/finetuned-detect/biomedparse_medvision_epoch=03_val_loss=0.4460.ckpt`.
  Pass it explicitly for both `TASK=detect` and `TASK=tl`. Because `--skip_existing` reuses masks already present
  in `results/${TASK}/finetuned/seg_masks/`, delete (or move) that folder before switching checkpoints, otherwise
  the old masks are kept.

## Smoke tests (code-path validation only)

Both smoke tests write **everything** under `${ABLATION_DIR}/smoke_test/<scope>/` (git-ignored), where `<scope>` is
the `DATASET` list with commas replaced by `+`, or `all` when `DATASET=` (empty). The real `data/`, `results/`,
`figures/`, `models/finetuned-detect/` are never touched; only `models/biomedparse_v2.ckpt` is shared (downloaded
if missing). Each script is one `{ ... }` block so bash parses the whole file before executing (safe against edits
during a run) and uses `set -euo pipefail` with `fail` messages `SMOKE TEST FAILED - ...`.

### Track A smoke - `scripts/eval/smoke_test.sh`

| Variable | Default | Meaning |
|---|---|---|
| `TASKS` | `detect tl` | space-separated tasks |
| `DATASET` | `AMOS22,BraTS24,CAMUS,FeTA24,HNTSMRG24,KiPA22,KiTS23,MSD,OAIZIB-CM,autoPET-III` | comma-separated; datasets absent from a task's list are skipped (only 6 of the 10 exist for T/L); `DATASET=` runs all |
| `LIMIT_PER_SUBTASK` | `100` in the script body (its header comment and the README describe 10 per subtask - override explicitly if you need a specific size) | samples per subtask |
| `GPU` | `0` | inference GPU |
| fixed | `N_PROCESSES=8`, `SLICE_BATCH_SIZE=4` | |

Per task it runs prepare (`--filter_dataset ${DATASET}` when non-empty) -> inference -> eval with explicit
`--tasks_json`, into `smoke_test/<scope>/data/test_npz/${TASK}`, `smoke_test/<scope>/results/${TASK}/pretrained`
and `smoke_test/<scope>/figures/${TASK}/pretrained`. It asserts: `.npz` count > 0; `_pred_mask.nii.gz` count ==
npz count; the five files `summary_*_task.txt`, `summary_metrics_*_Task.json`,
`eval_biomedparse_medvision_<task>_group_summary.csv`, `..._results.csv`, `..._metrics_dist.png` exist and are
non-empty; at least one figure PNG. On success prints `SMOKE TEST PASSED` and the first 8 lines of each summary
txt. Re-runs reuse existing masks; delete `smoke_test/<scope>/` to start clean.

Examples:

```bash
bash ${ABLATION_DIR}/scripts/eval/smoke_test.sh                                   # detect + tl, 10 datasets, GPU 0
DATASET=KiPA22 bash ${ABLATION_DIR}/scripts/eval/smoke_test.sh                    # single dataset -> smoke_test/KiPA22/
DATASET=BraTS24,KiTS23 TASKS=tl GPU=1 bash ${ABLATION_DIR}/scripts/eval/smoke_test.sh
DATASET= bash ${ABLATION_DIR}/scripts/eval/smoke_test.sh                          # all datasets -> smoke_test/all/
```

### Track B smoke - `scripts/finetune/smoke_test.sh`

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `KiPA22` | one dataset keeps it small; `DATASET=` runs all |
| `TASKS` | `detect tl` | tasks the tiny fine-tuned checkpoint is evaluated on |
| `LIMIT_PER_SUBTASK` | `10` | test samples per subtask |
| `TRAIN_LIMIT` / `VAL_LIMIT` | `32` / `8` | fine-tune pool sizes |
| `EPOCHS` | `1` | |
| `GPU` | **`1`** (not 0) | single GPU for training and inference |
| fixed | `BATCH_SIZE=2`, `N_PROCESSES=8`, `SLICE_BATCH_SIZE=4`, `--num_workers 2`, `--save_top_k 0` (only `last.ckpt`, ~4 GB) | |

Stages: B1 into `smoke_test/<scope>/data/finetune/detect` (reused if `train.json` exists; asserts PNGs in
`train`, `train_mask`, `val`, `val_mask`), B2 into `smoke_test/<scope>/models/finetuned/last.ckpt`, then per task:
prepare test NPZ into `smoke_test/<scope>/data/test_npz/${TASK}` **only if empty** (shared with the Track A smoke of
the same scope), inference, eval with the same five-file assertion. Prints `TRACK B SMOKE TEST PASSED` and
"(metrics from a 32-sample, 1-epoch model - not meaningful)".

What the smoke tests validate: the end-to-end code path (HF loading, normalization, upstream model construction,
checkpoint loading, mask I/O, metric files, figures). What they do **not** validate: the numbers (tiny pools,
1 epoch) and the identical-110k-sample guarantee (needs the full multi-dataset pool).

## Re-running a single dataset (`--filter_dataset`)

| Program | `--filter_dataset` semantics |
|---|---|
| `prepare_test_data_detect.py`, `prepare_test_data_tl.py`, `prepare_finetune_data.py` | comma-separated list; tasks of other datasets are skipped before loading |
| `run_inference.py` | **single** name; keeps NPZ files whose name starts with `<name>__` |
| `eval_tl.py` | **single** name; scores only `<name>__*` masks and **merges** into existing outputs |
| `eval_detect.py` | not available - re-run the full detection eval (it rescans `seg_masks/` and rewrites every file) |

T/L merge behaviour (`eval_tl.py --filter_dataset <name>`): the `<name>__` entries are replaced in
`eval_biomedparse_medvision_tl_{success,failure}_predictions.txt`; stale `<name>__*.png` figures are deleted from
all `MRE0k/` buckets and regenerated; rows with `file` starting with `<name>__` are replaced in
`eval_biomedparse_medvision_tl_results.csv`. `summary_metrics_tl_Task.json` and `summary_tl_task.txt` are **not**
refreshed, and `eval_biomedparse_medvision_tl_group_summary.csv` is rewritten from the filtered subset only - run
the full `eval_tl.py` (no filter) afterwards to refresh the aggregate files. Example (fine-tuned `epoch=03`,
dataset KiPA22, T/L):

```bash
export TASK=tl MedVision_PLANNER_VERSION=1.0.0 MedVision_ACK_RELEASE=1.4.0
CKPT=${ABLATION_DIR}/models/finetuned-detect/biomedparse_medvision_epoch=03_val_loss=0.4460.ckpt
rm -f ${ABLATION_DIR}/results/tl/finetuned/seg_masks/KiPA22__*          # otherwise --skip_existing keeps old masks
python ${ABLATION_DIR}/src/run_inference.py --checkpoint "$CKPT" --npz_dir ${ABLATION_DIR}/data/test_npz/tl \
    --seg_dir ${ABLATION_DIR}/results/tl/finetuned/seg_masks --gpu 0 --slice_batch_size 4 --skip_existing \
    --filter_dataset KiPA22                                              # GPU
python ${ABLATION_DIR}/src/eval_tl.py --pred_dir ${ABLATION_DIR}/results/tl/finetuned/seg_masks \
    --npz_dir ${ABLATION_DIR}/data/test_npz/tl --output_dir ${ABLATION_DIR}/results/tl/finetuned \
    --fig_dir ${ABLATION_DIR}/figures/tl/finetuned --filter_dataset KiPA22    # merge rows
python ${ABLATION_DIR}/src/eval_tl.py --pred_dir ... --npz_dir ... --output_dir ... --fig_dir ...   # full pass: refresh JSON/TXT/CSV
```

If the NPZ for that dataset must be regenerated (e.g. a corrected annotation), run
`prepare_test_data_tl.py --filter_dataset KiPA22 --output_dir ${ABLATION_DIR}/data/test_npz/tl ...` first - it
overwrites the `KiPA22__*.npz` files in place and leaves the others untouched.

## Inference speed / VRAM

`run_inference.py` processes one sample at a time; `--slice_batch_size` (default 4) is forwarded to
`model(input, mode="eval", slice_batch_size=N)` and is the only speed/VRAM knob. Lower it on small GPUs, raise it
when memory allows. `--gpu` sets `CUDA_VISIBLE_DEVICES` before torch is imported; giving `0,1` does not
parallelize (the upstream model has no DP/DDP path).
