## Release v1.1.0
---
### New Benchmark Variants

- CoT + scaledPS variants for TL and AD tasks (-CoT-scaledPS) — combines chain-of-thought prompting with pixel-size-scaled inputs (commits aa708db, 2bfec27)
- CoT variants for Detection tasks (3ddbef0)
- System prompt support added for 7 models (Gemma3, InternVL3, Llama3.2-Vision, LLaVA-OneVision, Qwen2.5VL, Lingshu, MedGemma) + --stop_strings CLI arg (bc9f636, 6006fdc)
- --log-sys-prompt CLI arg to record system prompt in JSONL output (eb90fd1)

---
### New Metrics

- nMAE (normalized MAE) added for TL and AD tasks (c5571f6, 991c7b3)
- nMAE 3-tier fallback fix for scaledPS tasks at summarization (daa4c4d)

---
### Key Bugfixes/Improvement

- float64 cast for IoU/F1 computations to prevent precision errors (52d1bc7)
- TL task metric fixes + ambiguous samples exclusion (692468e)
- Near-zero GT filtering for AD tasks (threshold AD_NEAR_ZERO_GT_THRESHOLD=0.1) (b9d871c)
- Val set diversity fix when val sample limit is small (f4b829c)

---
### New Analyses

- Equation-accuracy analysis scripts for AD and TL tasks, with test suite (c85af94)
- Process-accuracy analysis scripts for AD and TL tasks (655561d)
- Per-label + cross-model summary for process/equation-accuracy analyses (3b75e02)

---
### New Training Method

- Tool-use SFT training for Qwen2.5VL, with matching benchmark eval scripts (6ecf738)
- Tool-use model output parsing support extended to AD and TL tasks (b9d871c, 692468e)

---
### New Benchmark Tasks (Experimental)

- Visual-prompt (VP) tasks: overlay landmark/box annotations directly on the input image as a visual prompt (b1e67aa)
- VP-woMedImg tasks: visual-prompt variant where the medical image is omitted, isolating the model's ability to read annotation-only prompts (612e895)

---
### New CLI Arguments / Inference Features

- `--sample_indices` for partial inference over a dataset index range; accepts `[start:stop]` or `[start,stop,step]` formats; indices take precedence over `--limit` when both are set (ae6e130)
- `--max_new_tokens` to control maximum generated tokens across all eval scripts (default 4096) (da9124a)
- `--reshape_image_hw` to override input image resolution before feeding to the model; omit to pass original images unchanged (b7c24da)
- `--rm_old` argument added to `parse_outputs` to remove stale parsed files before re-parsing (4e2dec0)
- Image spatial dimensions (height × width) now injected into A/D and T/L task prompts (fbf771c)
- Default output parser changed from `extract_last_k_nums` to `extract_last_k_nums_within_answer_tag` (df37243)

---
### CT Image Preprocessing

- HU-window-based normalization applied to CT images in both benchmark and SFT pipelines; tissue-type-to-HU-window mapping added to `configs.py` as `CT_HU_windows_WL` (eadae72)
- Standard (non-HU) normalization enforced for KiPA22, which uses contrast CT images (c6b60d8)
- HU-based normalization skipped for miscellaneous/other labels in CT scans (15e5d76)

---
### Task Config Refactoring

- `lmms_eval_specific_kwargs` and per-task metadata centralized into shared YAML base configs, removing duplication across task YAML files (c501360, 9075795)
- `model_hf` and `model_name` now injected from CLI args into task `lmms_eval_specific_kwargs` at runtime (6641398)

---
### SFT Updates

- Fix: incorrect relative coordinate normalization in CoT ground-truth construction for A/D and T/L tasks — landmark indices were divided by the resized image shape (wrong) instead of the raw NIfTI shape (correct (5713309)
- Feat: full-parameter SFT with CoT reasoning now supported (5713309)
- Feat: SFT-CoT added for Detection tasks; per-dataset resampling for multi-task training (8945cab)
- Feat: dataset oversampling support — sample limit can now exceed dataset size via repeat sampling (7fd0ef5)
- Fix: `writer_batch_size=50` in dataset formatting to prevent OOM during large dataset builds; removed `is_main_process()` guard on `save_model` to fix multi-GPU checkpoint hang (b868189)
- Chore: skip samples where image loading fails during SFT dataset construction (df461b6)

---
### RFT / Parquet Dataset

- Feat: parquet dataset builders for RL finetuning in verl (`medvision_bm.rft.verl`) (58ba6b1)
- Feat: checkpointed parquet builders using shard-level checkpointing and PyArrow stream-merge to avoid OOM on large datasets (e.g. 1M detection samples); separate scripts for train+val and train+val+test splits (21af636)
- Feat: `medvision_bm.dataset.build_parquet_ds` for building parquet datasets; `medvision_bm.dataset.visualize_samples` for visualizing images with annotations (4d5255a)

---
### Infrastructure / Multi-GPU

- Standardized GPU assignment across eval scripts; enabled full GPU utilization (8d0a2a7)
- Serialized `medvision_bm` install with `flock` to prevent race conditions in parallel multi-GPU launches; removed hardcoded `CUDA_VISIBLE_DEVICES` (177bc80)
- More robust environment setup via requirement files; fixed module import failure in distributed subprocesses (fb317a6, 1e9b02d)
- LoRA + base model inference now supported in vLLM (7543f46)

---
### Key Bugfixes

- Replace hardcoded `.cuda()` with dynamic device detection in MedDr to enable DDP/multi-GPU (e1691fc)
- Fix default dtype for LLaVA-Med and MedDr: FP16 → BF16 (518bc07)
- Force fp32 when saving LoRA-merged models to prevent precision loss from the fp16 delta representation (e7b10a8)
- Resolve missing `model_name` in evaluation tasks; add answer-tag parser (`extract_last_k_nums_within_answer_tag`) (2fc5133)
- Guard against `inf` values in metric counter updates in `summarize_*_task` (1e97f3a)
- Fix `get_resized_img_shape()` if-condition error introduced in an earlier refactor (9533fc2)
- Fix import error in `lmms_eval/evaluator.py` (c8ac102)
- Remove duplicate samples in JSONL output files (e04d319)
- Sort samples in JSONL files by `doc_id` ascending during parsing (e8644b5)
- Add limit suffix to summary output filenames for AD, TL, and Detection tasks (b9c5dbf)
- Fix RFT parquet dataset field: retain `images` column instead of `image_file` (dac39f8)
