# MedVision Concepts and Glossary

## Purpose

Read this when a request uses MedVision vocabulary you need to pin down
(task vs. dataset config, planes, versions, metric names, CoT tags) or when
you must write text that has to match the paper and README. Every sub-skill
links here instead of redefining terms.

## The project in one paragraph

MedVision is a dataset and benchmark for *quantitative* medical image analysis
with vision-language models (VLMs). Instead of classification or report
generation it asks a model for numbers in real-world units: bounding boxes
(**Detection**), tumor/lesion ellipse axes in millimetres (**T/L**, "Tumor/Lesion
size"), and angles/distances from landmarks (**A/D**, "Angle/Distance",
degrees / mm). The code package is `medvision_bm` (PyPI `medvision-bm`, CLI
`mvbm`); the dataset package is `medvision_ds` (shipped in the `src/` folder of
the Hugging Face dataset `YongchengYAO/MedVision`). The released 7B model is
**MedVision-V0** (`YongchengYAO/MedVision-V0-7B`, Qwen2.5-VL-7B after SFT then
GRPO RFT). Post-training is supported through SFT entry points in
`medvision_bm.sft` and verl parquet builders in `medvision_bm.rft.verl`; the
GRPO stage itself runs in a verl fork (branch `medvision-rl`).

## Three task families

| Family | Short name | Model output | Ground truth source | Units | Answer numbers (k) |
| --- | --- | --- | --- | --- | --- |
| Detection | `detect` / `BoxCoordinate` task names / `BoxSize` dataset configs | one bounding box as relative `[x_min, y_min, x_max, y_max]` in [0, 1], origin at the lower-left corner | box fitted to each mask cluster on a 2D slice (no size test at generation time; the load-time single-instance filter drops a slice-sample with > 1 box, or whose only box is < 10 px on either side) | relative coordinates | 4 |
| Tumor/Lesion size | `TL` / `TumorLesionSize` | major and minor axis lengths | ellipse fitted to the mask in the real-world coordinate system (indices × pixel size); since v1.4.0 a cluster is measured when its major axis clears `max(2.0 mm, 2 × coarser in-plane spacing)`, with four guards on the fit (contour < 5 points, non-finite conic, minor axis < 1 voxel, major axis > 1.5× bbox diagonal) | mm | 2 |
| Angle/Distance | `AD` / `BiometricsFromLandmarks` | one angle or one distance | human-annotated landmark coordinates (Ceph-Biometrics-400 angles+distances, FeTA24 distances) | degrees / mm | 1 |
| Mask size (not benchmarked) | `MaskSize` | area | mask area | mm² | – |

For the **T/L and A/D** families the prompt states the image size and the **pixel size**
(physical spacing) and the model must do the pixel→mm arithmetic itself; **detection** prompts omit
both, because the answer is relative coordinates. Where the spacing is given, so the pixel size in the prompt is
re-adjusted to the resolution the model actually perceives after its internal
resize (see `../sub-skills/benchmark-evaluation/references/image-processing-and-token-budgets.md`).

## Names you will meet

- **dataset**: a public source dataset such as `BraTS24`, `MSD`, `KiPA22`, `Ceph-Biometrics-400`.
- **data-config** (Hugging Face config name): `{dataset}_{annotation-type}_{TaskID}_{plane}_{split}`, e.g. `BraTS24_BoxSize_Task01_Axial_Test`. Annotation types: `BoxSize`, `TumorLesionSize`, `BiometricsFromLandmarks`, `MaskSize`. `TaskID` is dataset-local (`Task01`, `Task02`, …) and defined in `medvision_ds/datasets/<dataset>/preprocess_*.py`, where the package directory replaces `-` and `.` with `_` (e.g. `Ceph-Biometrics-400` -> `Ceph_Biometrics_400`). Planes: `Axial`, `Coronal`, `Sagittal`. Splits: `Train`, `Test` (70/30 at the subject level).
- **task** (lmms_eval task name): `{dataset}_{TaskType}_{TaskID}_{plane}[-CoT]`, e.g. `BraTS24_BoxCoordinate_Task01_Axial-CoT`. Detection tasks say `BoxCoordinate` while the dataset config says `BoxSize`; `tasks_to_configs()` rewrites the token and appends `_Train`/`_Test`. `-CoT` selects the chain-of-thought prompt/target variant, not different data.
- **task list**: a JSON under `tasks_list/` whose top-level keys are task names (values are informational sample counts). Eval lists: `tasks_MedVision-{AD,TL,detect}-CoT.json`; SFT lists: `tasks_MedVision-{AD,TL,detect}__train_SFT.json` (keys use the dataset-config vocabulary, e.g. `BoxSize`, and the loader appends `_Train`); `OOD/` holds plane-OOD and target-OOD lists; `experimental/` is legacy.
- **task_tag / result_dir**: the launcher's `task_tag` (e.g. `MedVision-detect-CoT`) names `Results/<task_tag>/` and `completed_tasks/completed_tasks_<task_tag>.json`.
- **model_name**: the user-chosen folder name under `Results/<task_tag>/`; **model_hf_id**: the Hugging Face id used to load the model and its image processor.
- **lmms_eval model key**: the `--model` value understood by the vendored `lmms_eval` (`vllm_qwen25vl`, `medgemma`, `claude`, …); it must match a key in `AVAILABLE_MODELS` and a branch of `get_resized_img_shape()`.
- **model_family_name** (SFT/RFT): the same vocabulary without the `vllm_` prefix (`qwen25vl`, `gemma4`, `medgemma`, `qwen3vl`).
- **benchmark plan**: `benchmark_plan_<type>_v<version>.json.gz` files under `<data_dir>/Datasets/<dataset>/` that hold the annotations; `medvision_bm.utils.plan_utils` resolves them with a version ceiling.
- **annotation / planner version**: `1.0.0` … `1.4.0`. Leaderboard numbers use `1.0.0`; only T/L annotations change across versions. `MedVision_PLANNER_VERSION` must be set for the loader; pinning an older version than the newest release additionally needs `MedVision_ACK_RELEASE=<latest>`.
- **single- vs multi-instance**: benchmark samples are single-instance (one box / one cluster per target per slice); `MedVision_DISABLE_SAMPLE_FILTERING=true` returns everything.
- **CoT tags**: reasoning inside `<think></think>`, final values inside `<answer></answer>`. The regex parser only reads the first `<answer>` block; the LLM-judge pass (benchmark step 4) recovers answers written elsewhere.
- **Plane-OOD / Target-OOD**: evaluation on a plane (coronal/sagittal) or anatomical target unseen in training; Detection and T/L only.
- **scaledPS**: an experimental variant that scales the stated pixel size to probe arithmetic vs. perception.
- **MedVision-V0 recipe**: SFT on 121K CoT samples (110K detection, 5.5K T/L, 5.5K A/D, axial only, 512×512), then GRPO RFT sequentially A/D → T/L → detection with reward `r = r_format + r_process · r_answer` (detection: `r = r_format + r_answer`, no process reward).

## Metric vocabulary (details in `../sub-skills/results-parsing-and-metrics/references/metrics.md`)

- **SR / SuccessRate**: fraction of responses with k parseable numbers inside `<answer>`.
- **MAE, MRE, nMAE**: absolute / relative / diagonal-normalised error, computed over successful samples only (A/D, T/L).
- **IoU, F1, Precision, Recall** (Detection): region-based; failures count as 0, so means are over all samples.
- **MRE<k, IoU>k, Acc@IoU**: threshold accuracies with the **total** sample count as denominator.
- **AD near-zero GT**: A/D ground truths below 0.1 are dropped by the summarizer.
- **CDA**: Clinical Decision Agreement, weighted κ after mapping measurements to clinical categories.

## Benchmark pipeline (4 steps)

1. `python -m medvision_bm.benchmark.eval__<model>` → `Results/<task_tag>/<model_name>/*.jsonl` (GPU or API).
2. `python -m medvision_bm.benchmark.parse_outputs` → `parsed/`.
3. `python -m medvision_bm.benchmark.summarize_{AD,TL,detection}_task` → `summary_*`.
4. LLM-judge re-parse (`script/llm-parsing` driver in the repository; see `../sub-skills/llm-judge-parsing/`) → `llm-parsed_<judge>/` + `__llm-parsed_<judge>` summaries.

## Standard working-directory layout

```
<benchmark_dir>/            # the repository checkout, used as working directory
  Data/                     # <data_dir>: Datasets/, src/ (medvision_ds), .cache/, .downloaded_datasets.json
  Results/<task_tag>/<model_name>/
  completed_tasks/completed_tasks_<task_tag>.json
  SFT/<run_name>/           # checkpoints, merged models, wandb
  tasks_list/, requirements/, dockerfile/, script/
```

## Software versions used in the paper

`medvision_bm` 1.1.0, `medvision_ds` 1.1.0, verl 0.7.0; evaluation on 4× H100 80 GB; MedVision-V0 trained on 4× H200 140 GB. Intended for research only, not clinical use.
