# Routing decision — `medvision`

- Repository: `YongchengYAO/MedVision` (https://github.com/YongchengYAO/MedVision)
- Source commit: `a2c6482e0dbeea7f5cd5a8eddac7c7581f30608c` (the original classification run; `classification.json` carries the refreshed handoff at `780e247`) (branch `master`, working tree dirty — see the skill's `references/repo-provenance.md`)
- Skill id / root: `medvision` → `repo-skills/medvision`
- Taxonomy: `f8c306386015711634ddbb43a5eb95d1f58909c3513ce2063ba42efdd583a431`
- Status: **classified**, 3 assignments, all high confidence
- Primary evidence source: the repository checkout. The generated skill was used only as navigation context.

## What the repository is

MedVision is a dataset and benchmark for *quantitative* medical image analysis with
vision-language models, plus the post-training stack that produced the released
MedVision-V0 model. A model is asked for numbers in real-world units on a 2D medical
slice: a bounding box (Detection), tumour/lesion ellipse axes in millimetres (T/L), or an
angle/distance from landmarks (A/D). The repository ships the evaluation harness (a
vendored lmms-eval fork), the metric definitions, the launchers, an LLM-as-judge second
parse, SFT trainers, verl parquet builders for GRPO, analysis tools, and a
segmentation-specialist ablation.

## Assignment 1 — LLM Models, Training, and Alignment → Model Evaluation Benchmarks (high)

Family scope: *model-centric benchmark suites, evaluation harnesses, datasets, and
reproducible metrics for language and vision-language models.*

This is the repository's centre of gravity. The evidence is structural, not keyword-based:

- `src/medvision_bm/medvision_lmms_eval/` is a vendored fork of an evaluation harness, with
  `lmms_eval/models/__init__.py:19-65` registering 20 model keys, with 5 more commented out (vLLM-served, Hugging Face
  local, and API providers) and `lmms_eval/tasks/<dataset>/*.yaml` defining every subtask
  with its prompt hooks and metric aggregation functions.
- `src/medvision_bm/benchmark/` holds 21 `eval__<model>.py` entry points plus the
  `parse_outputs.py` scoring stage and three task summarizers.
- `script/benchmark-{detect,TL,AD}/` holds 72 launchers (24 models × 3 task families) that
  fix the protocol: sample limits (1000 open-weight, 100 for the API pilot), token budgets,
  annotation-version pins, and the result-tree layout.
- `README.md:689-720` documents the four-step protocol and the published leaderboard.
- `pyproject.toml:1-30` describes the distribution as the MedVision benchmark codebase.

Rejected alternative in the same area: **Multi-Stage LLM Training**. Although the
repository covers both SFT and RL data preparation, it is not an integrated multi-stage
training system; the RL stage runs in an external framework fork (see the rejected
assignment below).

## Assignment 2 — Biomedical AI → Medical Object Detection (high)

Family scope: *localization and detection of lesions, abnormalities, anatomical structures,
and clinical findings in 2D or 3D medical images.*

Detection is one of three first-class tasks and by far the largest (110K of the 121K
post-training samples). The repository owns the detection semantics end to end:

- Ground truth: boxes fitted per binary-mask cluster on each 2D slice, with clusters under
  10 px excluded and a single-instance filter; predictions are relative `[x_min, y_min,
  x_max, y_max]` coordinates in [0, 1].
- Metrics: `src/medvision_bm/utils/parse_utils.py:238-540` implements region-based IoU, F1,
  Precision and Recall together with the detection metric aggregator.
- Reporting: `src/medvision_bm/benchmark/summarize_detection_task.py` aggregates by anatomy,
  imaging modality and slice plane; `analyze_detection_task_boxsize.py` stratifies by
  box-to-image ratio to expose small-target behaviour.
- Roster: `tasks_list/tasks_MedVision-detect-CoT.json` enumerates the detection subtasks
  across anatomical and lesion targets.
- The `script/ablation/biomedparse/` study compares the benchmark against a segmentation
  specialist on the same detection test set, converting predicted masks to boxes.

Rejected alternative in the same area: **Medical Imaging Toolkits**. The medical-image
loading, spacing and annotation-generation code lives in the separate `medvision_ds`
package distributed with the Hugging Face dataset, not in this repository; what remains here
(NIfTI slice reading, benchmark-plan resolution) is supporting code for the benchmark rather
than a task-spanning imaging library.

Rejected alternative: **Computer Vision → Object Detection Models**, whose scope is
implementations of closed-set detection architectures. MedVision implements no detector; it
evaluates and post-trains vision-language models. Assigning it would be a context collision.

## Assignment 3 — LLM Models, Training, and Alignment → LLM Fine-Tuning (high)

Family scope: *libraries for supervised and parameter-efficient fine-tuning of language and
vision-language models, including full SFT, LoRA, and QLoRA.*

- `src/medvision_bm/sft/` contains ten trainer entry points (eight CoT LoRA/full-FT, one non-CoT, one tool-use) spanning four model families,
  in both LoRA chain-of-thought and full-parameter variants.
- `sft_utils.py:2696-2960` provides the two trainer builders: `prepare_trainer` (LoRA) and
  `prepare_trainer_fullFT` (full-parameter, FSDP), alongside multi-task dataset
  construction, sample-limit resolution, group-aware validation splitting, per-family
  collation and loss masking, a temperature-scaled multi-task sampler, checkpoint merging
  and Hub publishing.
- `train__SFT-CoT__qwen2_5_vl.py` and `train__fullFT-CoT__gemma4.py` show both paths are
  supported entry points, not examples.
- `README.md:873-900` documents the fine-tuning workflow and its resource-constrained
  configuration; the released MedVision-V0 model was produced with this stack.

Rejected alternative in the same area: **Preference and Reinforcement Alignment**. The GRPO
stage that produced MedVision-V0 is real, but it executes in an *external* fork of the verl
framework (branch `medvision-rl`), where the reward functions, curriculum and recipes live.
This repository contributes RL *data preparation* (`src/medvision_bm/rft/verl/`: parquet
builders, prompts, schema helpers) and documentation of the recipe. Under the rule that
dependency-only and external-integration matches are rejected, the alignment family is not
claimed. The generated skill still documents the recipe so an agent can reproduce it.

## Rejected assignments, summarised

| Candidate | Why rejected |
| --- | --- |
| LLM Models, Training, and Alignment → Preference and Reinforcement Alignment | GRPO training runs in an external verl fork; this repository only builds the RL datasets and documents the recipe. |
| LLM Models, Training, and Alignment → Multi-Stage LLM Training | Not an integrated multi-stage training system; the stages are split across this repository and an external framework. |
| Biomedical AI → Medical Imaging Toolkits | Imaging IO and preprocessing live in the separate `medvision_ds` package shipped with the dataset, not here. |
| Biomedical AI → Medical Segmentation | No segmentation model or training path; masks are ground-truth inputs, and the only segmentation model involved is an external baseline in an ablation. |
| Computer Vision → Object Detection Models | No detection architecture is implemented; context collision with the detection *task*. |
| Computer Vision → Vision-Language Understanding | The repository evaluates and fine-tunes VLMs but implements none; it is a benchmark, not a model family. |

## Import note

`classification.json` carries a recorded `skill_content_sha256` that is stale relative to the
current tree. The digest covers the
final runtime tree and must be recomputed immediately before import with
`skills/tests/medvision/reports/routing/compute_skill_digest.py <runtime-skill-dir> --write
skills/disco/routing_decision/classification.json`, which reproduces the importer's own
`digestPortableTree` algorithm. The importer rejects a stale digest.
