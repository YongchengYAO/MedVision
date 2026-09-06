---
name: biomedparse-ablation
description: "Reproduces or extends MedVision's segmentation-specialist ablation: evaluating pretrained BiomedParse v2 (Track A) and a detection-fine-tuned BiomedParse (Track B) on the MedVision Detection and Tumor/Lesion-size test sets with MedVision's own metrics. Use it for the BiomedParse setup (pinned upstream clone, conda env, detectron2 source build, medvision_ds re-pin), the prepare -> inference -> eval launchers and their env knobs, the fine-tuning recipe and checkpoint choice, smoke tests, --filter_dataset re-runs with T/L row merging, mask-to-bounding-box and mask-to-ellipse scoring, and the per-sample figure convention."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# BiomedParse Ablation (segmentation specialist on MedVision)

The MedVision paper compares MedVision-V0 not only with VLMs but with a **segmentation specialist**:
BiomedParse v2 (Zhao et al., *Nature Methods* 2025; BoltzFormer, *CVPR* 2025). This sub-skill covers the
repository's self-contained ablation folder (`script/ablation/biomedparse/` in the public GitHub repo, referred
to below as `${ABLATION_DIR}`): it runs BiomedParse on the same Detection and Tumor/Lesion (T/L) test samples the
VLMs see, converts predicted masks into the quantities the VLMs are asked for, and scores them with MedVision
metrics.

| Track | What it measures | Steps |
|---|---|---|
| **A - Evaluate** | BiomedParse v2 off-the-shelf (`biomedparse_v2.ckpt`, 4.2 GB from HF `microsoft/BiomedParse`) | prepare test NPZ -> inference -> eval |
| **B - Fine-tune** | BiomedParse v2 fine-tuned on MedVision's 110k-image detection SFT set, then re-scored on **both** tasks | prepare fine-tune data -> fine-tune -> inference -> eval |

## Route here for

- Setting up the ablation: pinned upstream clone (`microsoft/BiomedParse` @ `e02096c`), conda env `biomedparse`
  (Python 3.11, torch 2.6.0+cu124, transformers 4.40.0, lightning 2.3.0, huggingface-hub 0.36.0), detectron2 built
  from source, `medvision_ds` install followed by a re-pin.
- Running or re-running Track A / Track B, choosing `TASK=detect|tl`, `GPU`, `CHECKPOINT`, `--slice_batch_size`,
  `--filter_dataset`, and understanding where inputs/outputs land (`data/`, `models/`, `results/`, `figures/`,
  `smoke_test/`).
- Explaining the fairness rules (same 1000-per-subtask selection, same 110k training images, prompt = label name,
  identical image normalization, failure = empty mask) or the mask -> box / mask -> ellipse conversions.
- Reproducing the shipped numbers (fine-tuned checkpoint `epoch=03`, val_loss 0.446) or extending the study to a
  new dataset.
- Drawing per-sample figures with the shared 90-degree-CCW rotation convention.

## Do not use for

- MedVision metric definitions and denominators for VLM outputs -> `../results-parsing-and-metrics/SKILL.md`.
- Dataset downloads, `MedVision_PLANNER_VERSION` / `MedVision_ACK_RELEASE` semantics, task JSON namespace ->
  `../dataset-and-tasks/SKILL.md`.
- Why the SFT pipeline selects exactly those 110k detection images (`group_train_test_split`, `shuffle(SEED)`) ->
  `../sft/SKILL.md`.
- General `medvision_bm` install traps (pins lifted by installers, shadowed editable installs) ->
  `../environment-setup/SKILL.md`; cross-cutting symptoms -> `../../references/troubleshooting.md`.
- Task/metric vocabulary -> `../../references/concepts-and-glossary.md`; the benchmark's own figure scripts ->
  `../../references/visualization-catalog.md`.

## Quick facts (verified from the ablation sources)

| Item | Value |
|---|---|
| Test selection | each `*_Test` config loaded from HF `YongchengYAO/MedVision` in native order, first **1000** rows per subtask (`ds.select(range(limit))`) |
| Subtasks | Detection: 28 axial subtasks / 18 datasets (`tasks_MedVision-detect__train_SFT.json`); T/L: 10 axial subtasks / 6 datasets (`tasks_MedVision-TL__train_SFT.json`) |
| Prompt | the target's label name from the dataset `benchmark_plan` (`"arota"` fixed to `"aorta"`) |
| Normalization | CT: HU window per anatomical group (`CT_HU_windows_WL[label_map_regroup[label]]`); everything else: 0.5-99.5 percentile clip; both -> uint8 [0,255] |
| Detection score | bounding box of the predicted mask -> IoU, F1, Precision, Recall (failures count 0), avgMAE (successes only) |
| T/L score | ellipse fitted to the mask (`cv2.fitEllipse` in mm) -> major/minor axes -> MAE (mm), MRE; `MRE<k` uses total samples as denominator |
| Failure | mask all zeros (or no ellipse with >= 10 px) |
| Fine-tuning | 10 epochs, batch 4/GPU x 2 GPUs, AdamW lr 1e-5 wd 0.01, bf16-mixed, grad-clip 5.0, `SEED` from `medvision_bm.utils.configs` |
| Shipped checkpoint | `biomedparse_medvision_epoch=03_val_loss=0.4460.ckpt` (launcher default is `last.ckpt`) |
| Inference | single GPU only; `--slice_batch_size` (default 4) is the only speed/VRAM knob; `--skip_existing` resumes |

## Workflow map

1. **Environment** - read `references/setup.md`; run `scripts/check_biomedparse_env.py` (CPU-safe) to compare an
   interpreter against the pins and to confirm the upstream clone, checkpoint and env vars.
2. **Track A** - `TASK=detect` then `TASK=tl`: `1_prepare_test_data` -> `2_inference` -> `3_eval`
   (`references/tracks.md`, section A). Requires GPU, network (HF) and the MedVision data directory.
3. **Track B** - `1_prepare_finetune_data` -> `2_finetune` (2 GPUs, ~4 GB checkpoints per epoch) ->
   `3_inference` (both tasks, set `CHECKPOINT`) -> `4_eval` (`references/tracks.md`, section B).
4. **Validate the code path first** with the smoke tests (`smoke_test/<scope>/`, never touches shipped results).
5. **Re-run one dataset** with `--filter_dataset`; T/L rows are merged back, Detection is not (see tracks.md).
6. **Figures** follow `references/visualization-convention.md`.

## References and scripts

- Read `references/overview-and-fairness.md` for the study design, the fairness rules, the mask-to-quantity
  conversions, the folder layout and the citation/licence facts.
- Read `references/setup.md` for `setup.sh` step by step, the `requirements.txt` pins, the env-knob table
  (`TASK`, `GPU`, `CUDA_VISIBLE_DEVICES`, `CHECKPOINT`, `ENV_NAME`, `BIOMEDPARSE_DIR`, `MedVision_DATA_DIR`,
  `MedVision_PLANNER_VERSION`, `MedVision_ACK_RELEASE`, `_env.local.sh`) and the weights download.
- Read `references/tracks.md` for every launcher's exact command, variables, inputs and outputs (Track A, Track B,
  both smoke tests), fine-tuning hyper-parameters, `--filter_dataset` re-runs and checkpoint choice.
- Read `references/cli-reference.md` for the argparse tables of the seven `src/*.py` programs, the NPZ / PNG data
  formats and the complete output-file inventory.
- Read `references/visualization-convention.md` for the 90-degree-CCW rotation, coordinate transform, aspect-ratio
  and scale-bar formulas shared by the per-sample figures.
- Read `references/troubleshooting.md` when detectron2 will not build, `pip check` complains, the conda env is
  missing, the checkpoint download fails, T/L aborts on a release ACK, or a re-run changes the wrong files.
- Run `scripts/check_biomedparse_env.py [--python <interpreter>] [--ablation-dir DIR]` to report installed versions
  vs pins, the upstream commit, checkpoint presence/size and env vars (exit 1 when a required import is missing).
- Source `scripts/env_template.sh` (adapted from the ablation's shared `_env.sh`) from your own launcher after
  setting `TASK`; `DRY_RUN=1` prints the resolved knobs without activating conda.

## Safe operating rules

1. Inference (`run_inference.py`) and fine-tuning (`finetune.py`) **require a CUDA GPU**; `setup.sh` additionally needs `nvcc` (detectron2 is built from source), network access to GitHub and Hugging Face, and a populated MedVision data directory. The prepare and eval steps import no torch and run on CPU, `nvcc`
   for the detectron2 build, network access to GitHub and HuggingFace, and a populated MedVision data directory.
   Do not launch training or inference unless the user asked for it and the hardware is confirmed.
2. `setup.sh` creates/mutates a conda env and re-pins packages; never run it against a shared env or the
   benchmark's own environments.
3. The ablation launchers resolve paths relative to their own location and the upstream checkout - copy the
   repository's `script/ablation/biomedparse/` folder as a whole; the launchers are documented here but not bundled.
4. `third_party/`, `data/`, `models/`, `results/`, `figures/`, `smoke_test/` and `_env.local.sh` are git-ignored:
   a fresh clone contains none of them.
5. Smoke-test metrics are not meaningful (tiny pools, 1-epoch model); use them only to validate the code path.
