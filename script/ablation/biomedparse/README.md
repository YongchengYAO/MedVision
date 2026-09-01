# BiomedParse Ablation — a Segmentation Specialist on MedVision

The MedVision paper compares MedVision-V0 not only with general and medical VLMs but also
with a **segmentation specialist**. This folder is that comparison. It runs
[BiomedParse v2](https://github.com/microsoft/BiomedParse) (Zhao et al., *Nature Methods* 2025;
BoltzFormer, *CVPR* 2025) on the MedVision **Detection** and **Tumor/Lesion (T/L) size** test
sets, turns its predicted masks into the quantities the VLMs are asked for, and scores them
with MedVision's own metrics. Two tracks:

| Track | What it measures | Scripts |
|---|---|---|
| **A · Evaluate** | BiomedParse v2 off-the-shelf (pretrained weights) | `scripts/eval/` |
| **B · Fine-tune** | BiomedParse v2 after fine-tuning on MedVision's detection training data, then re-evaluated on both tasks | `scripts/finetune/` |

**How the comparison is kept fair**

- *Same test samples as the VLM benchmark.* Each `*_Test` config is loaded from
  HuggingFace in native order and the first 1 000 rows per subtask are kept — the same
  selection lmms-eval makes for the VLMs. 28 Detection subtasks (18 datasets) and 10 T/L
  subtasks, all axial.
- *Same training samples as MedVision-V0.* Track B fine-tunes on the 28 axial Detection
  `*_Train` configs with the same group-aware split and `shuffle(SEED).select(110k)` used
  by the SFT pipeline, so the 110 000 images are identical to V0's SFT detection set.
- *Prompting.* The text prompt is the target's label name. Images follow the upstream
  preprocessing recommendation: CT is windowed by anatomical group, everything else is
  percentile-clipped, then rescaled to `[0, 255]`.
- *From masks to MedVision quantities.* Detection fits a bounding box to the predicted
  mask (IoU, F1, Precision, Recall, MAE). T/L fits an ellipse to the mask and reports its
  major/minor axes in mm (MAE, MRE). A sample counts as a *failure* when the model returns
  no mask; as in the benchmark, `IoU>k` / `MRE<k` use the total sample count as denominator.

## Layout

```
script/ablation/biomedparse/
├── README.md
├── setup.sh                 one-time: pinned upstream clone + conda env
├── requirements.txt
├── src/                     MedVision-side Python
│   ├── _paths.py            path resolution (repo root, upstream checkout)
│   ├── prepare_test_data_detect.py / prepare_test_data_tl.py
│   ├── prepare_finetune_data.py
│   ├── run_inference.py     BiomedParse v2 inference on prepared .npz samples
│   ├── eval_detect.py / eval_tl.py
│   └── finetune.py          PyTorch Lightning fine-tuning (no AzureML dependency)
├── scripts/
│   ├── _env.sh              shared: paths, dataset pin, conda activation
│   ├── eval/                Track A   1_prepare_test_data → 2_inference → 3_eval
│   └── finetune/            Track B   1_prepare_finetune_data → 2_finetune → 3_inference → 4_eval
├── docs/visualization.md    convention for the per-sample figures
├── third_party/BiomedParse  upstream checkout @ e02096c            (created by setup.sh, git-ignored)
├── data/                    test_npz/{detect,tl}/  finetune/detect/ (git-ignored)
├── models/                  biomedparse_v2.ckpt  finetuned-detect/  (git-ignored)
├── results/<task>/<model>/  seg masks + metric files                (git-ignored)
└── figures/<task>/<model>/  per-sample bounding-box / ellipse figures (git-ignored)
```

`<task>` is `detect` or `tl`; `<model>` is `pretrained` (Track A) or `finetuned` (Track B).

## Setup

Requirements: conda, a CUDA 12.4 GPU with `nvcc` (detectron2 is built from source),
network access to GitHub and HuggingFace, and the MedVision data directory
(`MedVision_DATA_DIR`, default `<repo>/Data`).

```bash
cd script/ablation/biomedparse
bash setup.sh
```

This clones `microsoft/BiomedParse` into `third_party/` at the pinned commit, creates the
`biomedparse` conda env from `requirements.txt` (+ detectron2), and installs the MedVision
dataset package (`medvision_ds`). The pretrained weights (`biomedparse_v2.ckpt`, 4.2 GB) are
downloaded into `models/` the first time a launcher needs them.

All launchers resolve paths relative to their own location and can be run from any directory.
Machine-specific settings (e.g. `export MedVision_DATA_DIR=/data/MedVision`) belong in
`scripts/_env.local.sh` — git-ignored and sourced by `setup.sh` and every launcher when present.

## Track A — evaluate the pretrained model

```bash
bash scripts/eval/1_prepare_test_data.sh     # HF test configs → data/test_npz/detect/
bash scripts/eval/2_inference.sh             # masks → results/detect/pretrained/seg_masks/
bash scripts/eval/3_eval.sh                  # metrics → results/detect/pretrained/, figures → figures/detect/pretrained/

TASK=tl bash scripts/eval/1_prepare_test_data.sh   # same three steps for Tumor/Lesion size
TASK=tl bash scripts/eval/2_inference.sh
TASK=tl bash scripts/eval/3_eval.sh
```

Quick check without touching the shipped artifacts — prepare → inference → eval on a
comma-separated dataset list (default: **10 datasets** spanning CT/MR/US/PET;
~200 detect + 100 T/L samples at 10 per subtask; datasets absent from a task's list are
skipped), everything written under `smoke_test/<scope>/` (never tracked by git):

```bash
bash scripts/eval/smoke_test.sh                                   # detect + tl, 10 datasets, GPU 0
DATASET=KiPA22 bash scripts/eval/smoke_test.sh                    # single dataset
DATASET= bash scripts/eval/smoke_test.sh                          # all datasets
```

## Track B — fine-tune, then re-evaluate

```bash
bash scripts/finetune/1_prepare_finetune_data.sh   # HF train configs → data/finetune/detect/ (110k PNG + masks)
bash scripts/finetune/2_finetune.sh                # → models/finetuned-detect/*.ckpt
bash scripts/finetune/3_inference.sh               # → results/detect/finetuned/seg_masks/
bash scripts/finetune/4_eval.sh                    # → results/detect/finetuned/, figures/detect/finetuned/

TASK=tl bash scripts/finetune/3_inference.sh       # the detection-fine-tuned model is also scored on T/L
TASK=tl bash scripts/finetune/4_eval.sh
```

**Smoke test.** `bash scripts/finetune/smoke_test.sh` exercises all four stages on one
dataset (default `KiPA22`; 32 train / 8 val samples, 1 epoch, ~4 GB checkpoint under
`smoke_test/<dataset>/models/`) and re-uses the Track A smoke's test NPZ when present.
It validates the code path only — the tiny model and the single-dataset pool are not
meaningful (the identical-110k-sample guarantee needs the full multi-dataset pool).

Fine-tuning is detection-only; there is no T/L-specific training. The shipped results use
the best-validation checkpoint (`epoch=03`, val_loss 0.446) for both tasks — pass it as
`CHECKPOINT=...` to step 3 to reproduce them (the default is `last.ckpt`).

## Knobs

Every launcher has its settings at the top of the file. The ones you are most likely to
touch can also be given as environment variables:

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `TASK` | all task launchers | `detect` | `detect` or `tl` |
| `GPU` | inference | `0` | `CUDA_VISIBLE_DEVICES` for the (single-GPU) inference |
| `CUDA_VISIBLE_DEVICES` | `2_finetune.sh` | `0,1` | GPUs for DDP training (`N_GPUS` in the script must match) |
| `CHECKPOINT` | `finetune/3_inference.sh` | `models/finetuned-detect/last.ckpt` | fine-tuned checkpoint to evaluate |
| `ENV_NAME` | all | `biomedparse` | conda env |
| `BIOMEDPARSE_DIR` | all | `third_party/BiomedParse` | upstream checkout |
| `MedVision_DATA_DIR` | all | `<repo>/Data` | MedVision data directory — the folder holding `Datasets/`, `src/` and `.downloaded_datasets.json` |
| `MedVision_PLANNER_VERSION` | prepare / eval | `1.0.0` | MedVision annotation version (the paper's) |
| `MedVision_ACK_RELEASE` | prepare / eval (T/L only) | `1.4.0` | acknowledges the newest T/L annotation release when pinning an older one; must equal the latest release |

Fine-tuning hyper-parameters (`2_finetune.sh`): 10 epochs, batch 4 per GPU × 2 GPUs,
AdamW lr 1e-5, bf16-mixed, gradient clip 5.0, loss coefficients as in upstream
`finetune_biomedparse.yaml`. The validation split is carved out group-aware on the source
volume (`VAL_LIMIT=1000`, giving 1 376 slices in our run). `SEED` comes from
`medvision_bm.utils.configs`.

Re-running a single dataset: `prepare_test_data_tl.py`, `run_inference.py` and `eval_tl.py`
accept `--filter_dataset <name>` (e.g. `KiPA22`); the T/L evaluator merges the refreshed
rows back into the existing result files.

## Notes

- **Inference is single-GPU.** The upstream model has no DP/DDP path; `--slice_batch_size`
  (default 4) is the only speed/VRAM knob. Re-runs skip samples that already have a mask.
- **`pip check` after setup** may still list `opencv-python`/`numpy`, `gdrive`/`setuptools` and
  `build`/`packaging` metadata conflicts — they come from the dataset package's own
  dependencies and do not affect this pipeline (`cv2` and the rest import and run with the
  pinned versions). `huggingface-hub` and `packaging` are re-pinned by `setup.sh` on purpose:
  the dataset-package installer would otherwise lift them past what `transformers 4.40` /
  `lightning 2.3` accept.
- **Upstream is used unmodified.** `setup.sh` pins `microsoft/BiomedParse` to commit
  `e02096c`; the study only needs its `src/`, `configs/model/`, `utils.py` and
  `inference.py`. Set `BIOMEDPARSE_DIR` to reuse an existing checkout.
- **Per-sample figures** (`figures/`) follow the convention in
  [`docs/visualization.md`](docs/visualization.md); T/L figures are bucketed by MRE.

## Citation

If you use this ablation, please also cite BiomedParse:

```bibtex
@article{zhao2025foundation,
  title={A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities},
  author={Zhao, Theodore and Gu, Yu and Yang, Jianwei and Usuyama, Naoto and Lee, Ho Hin and Kiblawi, Sid and Naumann, Tristan and Gao, Jianfeng and Crabtree, Angela and Abel, Jacob and others},
  journal={Nature Methods}, volume={22}, number={1}, pages={166--176}, year={2025}
}
@inproceedings{zhao2025boltzmann,
  title={Boltzmann Attention Sampling for Image Analysis with Small Objects},
  author={Zhao, Theodore and Kiblawi, Sid and Usuyama, Naoto and Lee, Ho Hin and Preston, Sam and Poon, Hoifung and Wei, Mu},
  booktitle={CVPR}, pages={25950--25959}, year={2025}
}
```

BiomedParse is released under the Apache-2.0 license (see `third_party/BiomedParse/LICENSE`
after setup).
