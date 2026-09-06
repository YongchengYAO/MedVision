# Overview and Fairness Rules

Source of truth: the repository's `script/ablation/biomedparse/` folder (README, `src/*.py`, `scripts/**`).
Every statement below was checked against those files; nothing is inferred from the paper text alone.

## Purpose

MedVision benchmarks VLMs on *quantitative* medical image analysis. A natural question is how a dedicated
segmentation model fares when its masks are converted into the same quantities. The ablation answers it with
**BiomedParse v2** (text-prompted biomedical segmentation across nine modalities), used unmodified from the pinned
upstream checkout `microsoft/BiomedParse` @ `e02096c03af0d79c6994ffc2d60a49eeb0361e1f` (v2 branch, 2026-01-20).
Only its `src/`, `configs/model/`, `utils.py` and `inference.py` are used.

| Track | Model | Evaluated on | Result folders |
|---|---|---|---|
| A - Evaluate | pretrained `biomedparse_v2.ckpt` | Detection + T/L | `results/<task>/pretrained/`, `figures/<task>/pretrained/` |
| B - Fine-tune | `biomedparse_v2.ckpt` fine-tuned on MedVision detection training data (detection only, no T/L-specific training) | Detection + T/L | `results/<task>/finetuned/`, `figures/<task>/finetuned/` |

`<task>` is `detect` or `tl`.

## Fairness rules

### 1. Same test samples as the VLM benchmark

- `prepare_test_data_detect.py` / `prepare_test_data_tl.py` load each task key of the task JSON with the
  `_Test` suffix from HF `YongchengYAO/MedVision` via `medvision_bm.sft.sft_utils._load_single_dataset(...,
  split="test", limit=1000)`. That helper calls `load_dataset(..., streaming=False)` and then
  `ds.select(range(limit))`: the **first N rows in native HF order**, which is exactly what lmms-eval's
  `islice(iterator, 0, limit, 1)` gives the VLMs.
- Task JSONs (the same files the SFT pipeline uses):
  - Detection: `tasks_MedVision-detect__train_SFT.json` - 28 keys of the form `<Dataset>_BoxSize_TaskNN_Axial`,
    18 datasets (AMOS22, AbdomenAtlas1.0Mini, AbdomenCT-1K, BCV15, BraTS24, CAMUS, CrossMoDA, FLARE22, FeTA24,
    HNTSMRG24, ISLES24, KiPA22, KiTS23, MSD, OAIZIB-CM, SKM-TEA, TopCoW24, autoPET-III).
  - T/L: `tasks_MedVision-TL__train_SFT.json` - 10 keys `<Dataset>_TumorLesionSize_TaskNN_Axial`, 6 datasets
    (BraTS24, HNTSMRG24, KiPA22, KiTS23, MSD, autoPET-III).
  - All keys are axial. The dataset name is `key.split("_BoxSize_")[0]` / `key.split("_TumorLesionSize_")[0]`.
- The annotation version is pinned like the VLM launchers: `MedVision_PLANNER_VERSION=1.0.0`; T/L additionally
  sets `MedVision_ACK_RELEASE=1.4.0` (see `../../dataset-and-tasks/SKILL.md` for the semantics).

### 2. Same training samples as MedVision-V0 (Track B)

`prepare_finetune_data.py` loads all 28 axial Detection `*_Train` configs (no per-task limit), concatenates them,
then reproduces the SFT selection:

```python
split_ds = group_train_test_split(combined, group_column="image_file", test_size=val_limit,
                                  seed=SEED, stratify_column="dataset_name")
train_ds = split_ds["train"].shuffle(seed=SEED).select(range(min(train_limit, len(split_ds["train"]))))
val_ds   = split_ds["validation"]
```

with `SEED` imported from `medvision_bm.utils.configs` (value 1024) and, in the launcher, `TRAIN_LIMIT=110000`,
`VAL_LIMIT=1000` (the script's own default `--val_limit` is 105). The validation carve-out is group-aware on the
source volume (`image_file`); the README reports 1,376 validation slices for `VAL_LIMIT=1000` in the paper run.
A `--filter_dataset` pool does **not** reproduce the identical-110k guarantee (different pool -> different shuffle).

### 3. Prompting

The text prompt is the target's label name: `labels_map[str(label)]` from the dataset module's
`preprocess_detection.benchmark_plan` (Detection) or `preprocess_biometry.benchmark_plan` (T/L) in
`medvision_ds.datasets.<package>`; the misspelling `"arota"` is mapped to `"aorta"` on the Detection paths only (`prepare_test_data_detect.py`, `prepare_finetune_data.py`); the T/L preparer applies no such remap. Each NPZ stores
`text_prompts = {"<label_id>": "<label name>", "instance_label": 0}`; `run_inference.py` joins the prompts of the
sorted label ids with `"[SEP]"` (one label per sample in practice).

### 4. Image normalization (same convention as the VLM inputs)

| Script | Function | Rule |
|---|---|---|
| `prepare_test_data_detect.py` | local `normalize_ct` / `normalize_general` | CT with a known group: HU window `W,L = CT_HU_windows_WL[label_map_regroup.get(label_name, "Others")]`, clip to `[L-W/2, L+W/2]`; otherwise clip to the 0.5-99.5 percentiles; both scaled to uint8 `[0,255]` |
| `prepare_test_data_tl.py`, `prepare_finetune_data.py` | `medvision_bm.sft.sft_utils.normalize_img(doc, img_2d)` | same HU-window rule, except general normalization is forced when the label regroups to `"Others"` or the task is in `TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION` (KiPA22 Task01 - contrast CT) |

The two test-data preparers read slices with `_load_resize_nifti_2d(image_file, slice_dim, slice_idx)` (no resize); `prepare_finetune_data.py` uses its own `load_nifti_2d` instead (same slice indexing, returns only the array). Fine-tuning images are
additionally resized to 512x512 (`cv2.INTER_LINEAR`) and replicated to 3 channels; masks are resized with
`cv2.INTER_NEAREST` and saved as **0/1 uint8** (upstream's `BiomedParseDataset` zeroes masks stored as 0/255).

### 5. From masks to MedVision quantities

**Detection (`eval_detect.py`).** The predicted mask (`*_pred_mask.nii.gz`) is labelled into connected
components (`scipy.ndimage.label` + `find_objects`); the first component's box `[dim0_min, dim1_min, dim0_max,
dim1_max]` is the prediction (test cases with multiple objects were filtered out at dataset construction). Ground
truth comes from the HF row (`bounding_boxes.min_coords[0]`, `max_coords[0]`). Metrics:

- `avgMAE = mean(|pred/[H,W,H,W] - gt/[H,W,H,W]|)` (normalized coordinates, successes only),
- `IoU`, `F1`, `Precision`, `Recall` via `medvision_bm.utils.parse_utils.cal_IoU/cal_F1/cal_Precision/cal_Recall`.

Region key `"<label_map_regroup[label]> @ <modality> (A)"` mirrors the VLM summaries; labels are grouped into
`anatomy` / `tumor_lesion` (any of `TUMOR_LESION_GROUP_KEYS`) / `miscellaneous` (any of `EXCLUDED_KEYS`).

**T/L (`eval_tl.py`).** Connected components with >= 10 pixels (dataset construction used 200; inference keeps
small ROIs) are fitted with `cv2.fitEllipse` on the external contour expressed in **mm** (`contour * pixel_sizes`).
The ellipse gives four landmarks P1-P4 (major axis P1-P2, minor axis P3-P4, ordered so the major axis is the
longer one); axis lengths are Euclidean distances of the landmarks in physical space (`voxel_size`). Ground truth
is `biometric_profile.metric_value_major_axis[0]` / `metric_value_minor_axis[0]`.

```
MAE = (|d1 - major_gt| + |d2 - minor_gt|) / 2          # mm
MRE = (|d1 - major_gt| / major_gt + |d2 - minor_gt| / minor_gt) / 2
```

Only the largest cluster is scored (multi-cluster cases were filtered out during data loading).

### 6. Failures and denominators

- A sample is a **failure** when its predicted mask is all zeros (`analyze_predictions`), or, for T/L, when no
  cluster yields a valid ellipse (moved from the success list to the failure list during evaluation).
- Detection: `IoU`, `F1`, `Precision`, `Recall` means include failures as 0; `avgMAE` averages successes only;
  `IoU>k`, `F1>k`, `Precision>k`, `Recall>k` (k = 0.5 ... 0.9, `>=` comparison) and `MAE<k` (k = 0.1 ... 1.0)
  divide by the **total** sample count of the region.
- T/L: `avgMAE`, `avgMRE` average successes; `MRE<0.1 ... 0.5` divide by the total count; `SuccessRate = n_success
  / n_total`. This matches how the benchmark treats VLM parsing failures (see
  `../../results-parsing-and-metrics/SKILL.md`).
- The grouped detection file keeps only regions with >= 50 samples (`MINIMUM_GROUP_SIZE`) and reports
  sample-weighted means.

## Folder layout (inside `${ABLATION_DIR}`)

```
README.md
setup.sh                    one-time: pinned upstream clone + conda env
requirements.txt
src/                        _paths.py, prepare_test_data_{detect,tl}.py, prepare_finetune_data.py,
                            run_inference.py, eval_{detect,tl}.py, finetune.py
scripts/_env.sh             shared: paths, dataset pin, conda activation, ensure_pretrained_ckpt
scripts/_env.local.sh       optional machine-specific overrides (git-ignored)
scripts/eval/               Track A: 1_prepare_test_data.sh, 2_inference.sh, 3_eval.sh, smoke_test.sh
scripts/finetune/           Track B: 1_prepare_finetune_data.sh, 2_finetune.sh, 3_inference.sh, 4_eval.sh, smoke_test.sh
docs/visualization.md       per-sample figure convention
third_party/BiomedParse     upstream checkout (created by setup.sh)            git-ignored
data/test_npz/{detect,tl}/  prepared test samples (.npz)                      git-ignored
data/finetune/detect/       train/, train_mask/, train.json, val/, val_mask/, val.json   git-ignored
models/                     biomedparse_v2.ckpt, finetuned-detect/*.ckpt      git-ignored
results/<task>/<model>/     seg_masks/ + metric files                         git-ignored
figures/<task>/<model>/     per-sample figures                                git-ignored
smoke_test/<scope>/         smoke-test copies of data/, results/, figures/, models/   git-ignored
```

`.gitignore` covers `third_party/ data/ models/ figures/ smoke_test/ scripts/_env.local.sh *.log __pycache__/
results/`. A fresh clone therefore contains only code and docs.

## Citation and licence

BiomedParse is released under the **Apache-2.0** licence (`LICENSE` in the upstream checkout after setup). When
reporting the ablation, cite BiomedParse in addition to MedVision:

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
