## Release v1.2.1

---
### SFT Dataset Preparation

- Preparation is now resumable: each task's formatted splits go to a content-addressed cache, and splits are written in 20,000-row chunks saved atomically, so an interrupted build loses at most one chunk instead of everything (c2ac89c)
- The cache key covers the task list, sample limits, model, image settings, planner version, seed and the formatting code itself, so any change invalidates it (c2ac89c)
- Arrow builds now use an explicit cache directory; the `HF_DATASETS_CACHE` export was a no-op, so builds landed on the container overlay and were lost on pod restart (c2ac89c)
- Faster formatting: the last decoded NIfTI volume is cached per worker with rows sorted by volume, existing slice PNGs are reused, cleanup is schema-only and saving runs in parallel. Output stays bit-identical (a2c6482)
- Prepared datasets are named from their true train sizes, and the launchers pass the path from the prep run to training so rank 0 never repeats the load (980e9df)
- An unset sample limit now means the full dataset; an explicit 0 is rejected as ambiguous (10b3b6c)
- Each task cache now lives beside the prepared dataset it feeds; the non-CoT entry point was filing under the CoT directory (b179ed0)
- The SFT loader crashed on detection task lists that use the evaluation-side `_BoxCoordinate_` task names, such as the AllSlices lists, because the Hugging Face dataset exposes that data only under `_BoxSize_` configs. The loader now renames them before looking up the config (5705597)

---
### SFT Training Fixes

- Resuming a full fine-tune under FSDP crashed on the first step with mismatched optimizer dtypes; optimizer moments are now recast to the flat-parameter dtype after loading (94a3397)
- Training on a prepared dataset ignored the validation limits and evaluated the whole split (1.8M rows for full v1.4.0); all ten entry points now cap it, seeded and identical across ranks (94a3397)
- New CPU-only tests for the cache, the resume recast, the validation cap and all ten entry points (c2ac89c, b179ed0, 94a3397)

---
### Agent Skills

- New `skills/medvision/`: a `/medvision` router over ten sub-skills (setup, datasets, evaluation, parsing, LLM-judge, SFT, RFT, analysis, extending, BiomedParse ablation) with references and helper scripts, plus the `medvision-paper` and `medvision-pipeline` companions (39dd6bf)
- Install instructions for Claude Code, Codex, OpenCode and Pi in `skills/README.md`, and a README section with worked examples (39dd6bf, 8f8d0f2, 604f30f, 37e802b)

---
### BiomedParse Ablation

- Two-track segmentation-specialist ablation: pretrained BiomedParse v2 on the Detection and T/L test sets, then fine-tuned on the 110K SFT detection samples; masks are scored with the benchmark's own metrics (a2c1320)

---
### Figures and Task Lists

- New `figure_concat`, a vector-preserving PDF/PNG panel compositor, with a worked sample script (6583385)
- Website export scripts updated; `viz_label_cloud` renamed `viz_OOD_label` (684c036, 2a4f39d)
- Merged AllSlices task lists for Detection and T/L across all six dataset versions; totals reconcile with the README counts (b6cbfae)

---
### Setup and Install

- `medvision_ds` now installs in two steps, so `--force-reinstall` no longer upgrades `huggingface_hub` past what `transformers` 4.x accepts (780e247)
- Requirement pins drop dev-machine wheel paths; a stale package-data entry removed (8f8d0f2, 780e247)

---
### Other

- `Paper-results-backup/` snapshots the LLM-parsed leaderboard summaries for the 18 evaluated models (755a7bb)
- nMAE unit tests were passing vacuously on parse failures; fixed along with the scaled-PS test regex (780e247)

---
### Documentation

- Docsite reconciled with the shipped code: metric and flag tables corrected, contradicted claims dropped, placeholder paths replaced (8f8d0f2)
- RFT page documents the verl-fork rewards, task mixing, curriculum and recipes (2a4f39d)
- Corrections: Qwen3-VL resizes to a multiple of 32, GLM-4.6V is ~106B, CDA uncertainty pairs with weighted kappa, renal cohort counts split into unfiltered and post-exclusion (37e802b)
