---
name: medvision
description: "Use the MedVision benchmark and toolkit for quantitative medical image analysis with vision-language models: bounding-box detection, tumor/lesion size in millimetres, and angle/distance measurement from landmarks. Covers the medvision_bm package and mvbm CLI, the medvision_ds dataset and its task lists and annotation versions, running evaluations of local vLLM/HF or API VLMs, parsing outputs and computing SuccessRate / MAE / MRE / nMAE / IoU / F1 metrics, LLM-as-judge re-parsing, SFT and GRPO reinforcement fine-tuning of MedVision-V0, clinical-decision and process/equation accuracy analyses, adding new models or tasks, and the BiomedParse segmentation-specialist ablation."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision

Use this repo skill when a task involves the **MedVision** dataset, benchmark, or
post-training stack. MedVision measures whether a vision-language model can
produce *quantitative* answers about medical images in real-world units, rather
than classifications or free-text reports. Three task families:

| Family | Short name | Model must output | Units |
| --- | --- | --- | --- |
| Detection | `detect` | one bounding box, relative `[x_min, y_min, x_max, y_max]` in [0, 1] | – |
| Tumor/Lesion size | `TL` | major and minor ellipse axis lengths | mm |
| Angle/Distance | `AD` | one angle or one distance from landmarks | degrees / mm |

The **T/L and A/D** prompts state the image size and the pixel size, and the model does the
pixel-to-millimetre arithmetic itself; the **detection** prompt states neither, because its answer
is relative coordinates in [0, 1]. Answers go inside `<answer></answer>`,
reasoning inside `<think></think>`. The released model is **MedVision-V0**
(Qwen2.5-VL-7B after SFT then GRPO). Two packages matter: `medvision_bm`
(this repository, PyPI `medvision-bm`, CLI `mvbm`) and `medvision_ds` (the
dataset codebase, installed from the Hugging Face dataset repository).

## Start here

1. Confirm the request is about MedVision itself, not generic medical imaging,
   segmentation, or another VLM benchmark.
2. Check the environment before proposing commands:
   `python scripts/check_medvision_env.py` (add `--data-dir <data_dir>` to also
   inspect a data directory, `--json` for machine output). It reports both
   packages, the pinned foundation versions, GPU visibility, the `mvbm` CLI, and
   the seven `MedVision_*` dataset/loader variables plus `MEDVISION_RESP_CACHE`, and it names the pin traps
   it finds. It does **not** report the `MEDVISION_SFT_*`, `MEDVISION_SCALED_PS_*` or `MEDVISION_DS_SRC`
   variables — check those by hand.
3. Route to the narrowest sub-skill below before writing commands or code.
4. Read `references/concepts-and-glossary.md` whenever task names, dataset
   config names, planes, annotation versions, or metric names appear in the
   request — the vocabulary is precise and easy to get wrong.

## Minimal install and import check

```bash
pip install medvision-bm                       # or: pip install .   (from a checkout)
mvbm install mvds -d Data                      # installs medvision_ds into <data_dir>
python -c "import medvision_bm, medvision_ds; print(medvision_bm.__version__, medvision_ds.__version__)"
```

Set these before any dataset call: `MedVision_DATA_DIR=<data_dir>` and
`MedVision_PLANNER_VERSION=<annotation version>` (the loader fails without the
latter; leaderboard numbers use `1.0.0`, which also needs
`MedVision_ACK_RELEASE=1.4.0`). Evaluating open-weight models, fine-tuning, and
the LLM judge additionally need per-model dependency stacks and CUDA GPUs;
`environment-setup` owns that.

## Route by task

- `sub-skills/environment-setup/SKILL.md` — install or repair the stack: the
  three ways to install `medvision_bm`, the dataset package, the vendored
  `lmms_eval` engine and its per-model extras, `benchmark.env_setup` /
  `sft.env_setup`, the 25 frozen requirements files, Docker images, the
  load-bearing install order, and the version-pin traps.
- `sub-skills/dataset-and-tasks/SKILL.md` — choose and name data: dataset config
  naming, task-list JSONs, downloading, `download_mode` semantics, annotation
  versions and the planner pin, the `Data/` layout and benchmark plans, parquet
  snapshots.
- `sub-skills/benchmark-evaluation/SKILL.md` — run an evaluation: the 21
  `eval__<model>` entry points and their 24 launcher stems per task family, launcher anatomy, sample limits and token
  budgets, tensor/data parallelism, the crash-safe resume cache, the perceived
  image-size invariant, and where results land.
- `sub-skills/results-parsing-and-metrics/SKILL.md` — turn raw outputs into
  numbers: `parse_outputs`, the three summarizers, and the exact definition,
  denominator and failure handling of every metric.
- `sub-skills/llm-judge-parsing/SKILL.md` — the format-robust second parse: the
  judge pipeline's stages, environment, roster YAMLs, artifacts, and the
  reproducibility caveats.
- `sub-skills/sft/SKILL.md` — supervised fine-tuning: dataset construction and
  sample-limit semantics, LoRA and full-parameter entry points per model family,
  the two-phase launcher pattern, merging, resuming and pushing.
- `sub-skills/rft/SKILL.md` — reinforcement fine-tuning: building verl-ready
  parquet datasets and the GRPO recipe (rewards, task mixing, curriculum) that
  produced MedVision-V0.
- `sub-skills/analysis/SKILL.md` — post-hoc studies on parsed results: clinical
  decision agreement, process accuracy, equation accuracy, detection by target
  size.
- `sub-skills/extending-models-and-tasks/SKILL.md` — maintainer work: add a new
  local or API model across all required sites, or add a new task/dataset YAML
  pair and register it.
- `sub-skills/biomedparse-ablation/SKILL.md` — the segmentation-specialist
  comparison: evaluating and fine-tuning BiomedParse v2 on the MedVision test
  sets with MedVision's own metrics.

## Verifying a change

The checkout ships `unit-test/` — 47 Python files across 14 areas (`claude-image-resize`,
`detection-metric-failure`, `detection-verl-nocot`, `equation-accuracy`, `gemini-image-resize`,
`kimi-image-resize`, `llm-parsing`, `medvision-ds-planner-version`, `nMAE`, `openai-image-resize`,
`perceived-size-resize`, `scaledPS`, `sft-loss-masking`, `tool-use`). There is no runner: execute a
file directly, e.g. `python unit-test/llm-parsing/test-1.py`. Run the area matching whatever you
changed — image-resize and `perceived-size-resize` for a new model branch, `scaledPS`/`nMAE` for
parsing or metrics, `sft-loss-masking` for collator work.

## Cross-cutting references

- Read `references/concepts-and-glossary.md` for the vocabulary: task families,
  task names vs dataset configs, planes, splits, annotation versions, CoT tags,
  metric names, and the standard working-directory layout.
- Read `references/model-roster.md` before naming a model: which VLMs are
  supported, their evaluation entry point and `lmms_eval` key, dependency stack,
  parallelism, hardware footprint, and perceived image size.
- Read `references/troubleshooting.md` for failures that are not specific to one
  workflow: install and pin errors, dataset-package and environment-variable
  problems, GPU/memory limits, and result-tree hygiene.
- Read `references/visualization-catalog.md` when a figure is requested; the
  figure scripts stay in the repository and are catalogued, not bundled.
- Read `references/repo-provenance.md` to decide whether this skill still matches
  a checkout before trusting its details or running a refresh.
- `references/repo-routing-metadata.json` is skill-library infrastructure, not
  guidance. It records where this skill sits in the library taxonomy and is read
  by the importer; there is nothing in it for a task.

## Bundled script

- Run `scripts/check_medvision_env.py` to collect environment facts safely. It
  imports nothing heavier than the packages it reports on, never installs,
  downloads or touches a GPU, prints no secret values, and exits non-zero when
  `medvision_bm` is missing.

## Safe operating rules

- Never start an evaluation, fine-tuning run, dataset download, or judge sweep
  unless the user asked for it. These cost GPU-hours, API credits, or hundreds
  of gigabytes; state the cost first. One dataset config downloads the whole
  source dataset.
- Treat `Data/`, `Results/`, `SFT/` and `completed_tasks/` as data. Read them;
  do not hand-edit result JSONLs. Deduplicate into a new directory instead.
- Keep `MedVision_PLANNER_VERSION` fixed for the life of a study and record it
  with the results; annotation versions change the T/L sample set.
- Never mix model dependency stacks in one environment. Each model has a frozen
  requirements file and the launchers create one environment per model.
- Sanitize secrets before use: pod-injected tokens carry a trailing newline,
  which is an illegal HTTP header value.
- Report a metric only with its denominator convention: threshold metrics divide
  by the total sample count, while means exclude or zero out failures.
- MedVision is a research benchmark. Its models are not clinically accurate and
  must not be presented as clinical tools.
