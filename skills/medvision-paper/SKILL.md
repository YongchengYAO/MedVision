---
name: medvision-paper
description: Authoritative facts from the MedVision paper (dataset scale, annotation-generation rules, metric definitions, MedVision-V0 SFT/RFT recipe, evaluated VLM roster). Use when writing README/webpage/docs text, when terminology or scale numbers matter, or before relying on a benchmark design decision.
---

## Research Context (from the MedVision paper)

Authoritative facts from the accompanying paper ("MedVision: Benchmarking Quantitative Medical Image Analysis"). Use these as the source of truth for terminology, scale, and design decisions; verify against code before relying on any specific path/flag.

**Motivation:** Existing medical VLMs/benchmarks are categorical ("normal vs abnormal") or qualitative (report generation); MedVision targets the underexplored *quantitative* axis (lengths, areas, angles in real-world units) that clinicians actually use to stage disease and plan treatment.

**Dataset (v1.0.0):** 22 public datasets comprising **29.0K 3D images, 11.2M annotated 2D slices and 24.3M single-instance annotations** (45.3M multi-instance), multi-anatomy + multi-modality (XR, CT, MRI, US, PET). Restricted to modalities with physical spacing in the header (needed for real-world GT). Images stored as 3D volumes in **RAS+** orientation; loader supports slicing along any of the 3 planes (axial/coronal/sagittal). Train/test split is **7:3 per dataset, over 3D image volumes** (`_split_niigz_dataset` splits `.nii.gz` files) — not subject-disjoint, so a subject with several volumes can appear on both sides.

**Single- vs multi-instance.** A benchmark sample is a *(2D slice, target)* pair. **Single-instance** keeps a
target only when it appears as a single, sufficiently large instance on the slice — that is the set used for
benchmarking and fine-tuning (24.3M); **multi-instance** lifts those per-sample filters (45.3M).

**Annotation generation** (the current **v1.4.0** rules):
- *Bounding box:* fitted per binary-mask cluster on each 2D slice, with **no size test at generation time** (`benchmark_planner._find_bounding_boxes_2D` records every component). The `<10 px` rule is a **load-time** single-instance filter (`MedVision.py:12300`): a slice-sample is dropped when it holds more than one box, or when its only box is under 10 px on either side.
- *T/L size:* ellipse fitted to each T/L mask **in the real-world coordinate system** (array indices × pixel size), reporting major + minor axis physical lengths. A cluster is measured when its fitted **major axis** clears a physical floor of `max(2.0 mm, 2 × the coarser in-plane spacing of the plane being measured)` — a *resolution* floor, not a clinical one (per-lesion thresholds such as RECIST 10 mm are left to downstream consumers). Four guards reject degenerate `cv2.fitEllipse` results: a contour under 5 points, a non-finite conic, a minor axis under one voxel, and a major axis over 1.5× the cluster's own bounding-box diagonal.
- *A/D:* angle/distance computed from human-annotated landmark coordinates (Ceph-Bio-400 for both; FeTA24 for distance).

> **Paper T/L numbers predate v1.4.0.** The published T/L results were produced under an earlier annotation version and are **not** comparable to a v1.4.0 run. Pin `MedVision_PLANNER_VERSION=1.0.0` (plus `MedVision_ACK_RELEASE`) to reproduce them.

**Real-world units:** T/L and A/D targets are in **mm** (degrees for angles), NOT pixels, and only those two prompts carry the image size + physical pixel size — the **detection prompt omits the spacing block** because its answer is relative coordinates in [0, 1] (paper.tex:838). For T/L and A/D, because each VLM resizes images differently, pixel size is **re-adjusted at every resize step** so image-size × pixel-size always reflects the true physical extent. Models must do the pixel→mm arithmetic themselves. (See per-model resize strategies; matches the scaledPS pixel-size pipeline.)

**Metrics:** SR (success rate) = the fraction of responses from which the required numeric values could be parsed (paper.tex:786 defines it with no tag restriction). The **strict** parser reads only `<answer></answer>`, so an answer given as `\boxed{...}`, "Answer: ..." or plain prose scores as a failure; the **reported** metrics come from the LLM-parsed records, where an answer stated anywhere in the response counts once the located quote is span-verified and its numbers re-extracted. That is why pooled SR rises 92.2% → 98.6% (detection), 81.7% → 94.5% (T/L) and 79.0% → 93.9% (A/D) — the strict parser rejects 7.8% / 18.3% / 21.0% of responses respectively. Detection: region-based R, P, F1, IoU, IoU>0.5, plus the COCO-style localization hit-rate sweep `Acc@IoU>=0.50` … `Acc@IoU>=0.95` in 0.05 steps and its mean `Acc@IoU[0.50:0.95]` (a hit-rate, **not** a mAP integral — one box per sample with no confidence score, so classic mAP is undefined). Measurement: MAE, MRE, MRE<0.1 (T/L: n=2 axes; A/D: n=1). MAE/MRE are computed **only on successfully parsed samples** (failures are NaN-excluded, `summarize_TL_task.py:262-271`), but the detection region metrics are **not**: a failed parse scores IoU/F1/P/R = **0** rather than NaN (`parse_utils.py:497-523`), so their denominator is every sample. The paper's blanket wording is loose here; the threshold metrics **MRE<k / IoU>k use total sample count as denominator**, so they reflect instruction-following *and* accuracy. Reported numbers are computed from the **LLM-judge re-parsed** records (benchmark step 4, `script/llm-parsing/`, reader `gemma-4-31b`), not from the strict-regex pass alone.

**CDA (clinical decision agreement)** is a second reported metric family (paper §4.5, App. Clinical Decision
Agreement Analysis): each predicted measurement is mapped to a clinical category through a published cutoff
table — Steiner SNA 82±2°, SNB 80±2°; AJCC kidney T-staging at 40/70/100 mm — and agreement with the GT
category is scored by **quadratic-weighted Cohen's κ_w**, with a 4,000-resample bootstrap over whole volumes.
No re-inference is involved; it re-reads the existing predictions.

**MedVision-V0 (the released 7B model):** base = **Qwen2.5-VL-7B**, two-stage post-training.
- *SFT:* 121K multi-task CoT samples (**110K detection, 5.5K T/L, 5.5K A/D**), **axial slices only** — coronal/sagittal held out to test plane-OOD generalization. Weighted random sampler with temperature-scaled task mixing, `P(t) ∝ N_t^(1/T)`, **T=5** → ~47.7% detection / 26.2% T/L / 26.2% A/D against natural proportions 90.9/4.5/4.5, sampled with replacement. It oversamples minority tasks. Images reshaped to **512×512**. Answers: reasoning in `<think></think>`, final values in `<answer></answer>`; CoT text filled from intermediate GT (e.g., landmark coords).
- *RFT:* GRPO (via **verl**, fork branch `medvision-rl`), full-parameter on the SFT checkpoint, run **sequentially A/D → T/L → detection** with the CoT answers removed from the targets; The **paper** states RFT reuses the same 121K SFT samples (paper.tex:210). The **as-run recipes** instead use single-task parquets — stages 1–2 the 5.5K A/D and T/L sets, stage 3 the 1M-cap detection set `ds__AD0_D1000000_TL0_all1000000` — and the released model is that stage's `global_step_250` (`train__rft-sequential__{1,2,3}*.sh` headers; **code, not paper**). Reward is `r = r_format + r_process · r_answer` for T/L and A/D, and `r = r_format + r_answer` for detection (a box *is* the answer, so there are no intermediate steps to supervise). Each component ∈ [0,1]; errors map through `ρ(e) = exp(−e)`.
  - `r_format` — `0.8 × reasoning-structure score + 0.2 × binary <answer> check` (detection: the binary check alone).
  - `r_process` — mean over the CoT steps (T/L 4, A/D 3) of `ρ(step error)`: worst-point localization error (displacement ÷ √2, bounded in [0,1]) for landmark steps, relative error for measurement steps. This is what encourages accurate intermediate landmark estimates.
  - `r_answer` — `ρ(MRE)` for T/L and A/D; `ρ((1 − CIoU)/2)` for detection, CIoU being position- and scale-fair where coordinate MRE is biased toward boxes near the origin.
  - GRPO config: 8 rollouts/prompt, train batch 256, mini-batch 128, lr 3e-6 constant, KL coef 0.01 (`low_var_kl`), 4096 max prompt/response, vLLM rollouts. A multi-task ablation instead runs a single stage over the mixture (T=8 task sampling + an epoch-level curriculum that mines hard examples). Code: rewards in `verl/utils/reward_score/medvision_rewards/` (one `compute_score` entry point, configured through `reward_kwargs`); recipes `examples/grpo_trainer/train__rft-*.sh`.
- *Headline numbers (v1.0.0 annotations, LLM-parsed — quote these, not a v1.4.0 run):* detection **F1 79.1**
  (anatomy) / **46.9** (tumour-lesion); Measurement MRE **26.0** (T/L), **6.4** (distance), **52.1** (angle).
- Result: MedVision-V0 outperforms all 17 evaluated off-the-shelf VLMs across all three tasks, and a segmentation specialist (BiomedParse) on detection + T/L. SFT gives the large jump; RFT adds consistent gains and better OOD generalization (Plane-OOD = unseen plane for a target; Target-OOD = unseen anatomical target).

**Error analysis frame:** end-to-end *measurement error* decomposes into *localization error* (normalized L2 of landmark coords, [0,1]) and *arithmetic error* (MRE vs. a Python execution of the same formula). T/L localization is harder (irregular structures); A/D arithmetic is harder (angle formulas).

**Evaluated off-the-shelf VLMs (17):** general — Qwen2.5-VL (7B/32B), Qwen3-VL-Thinking (32B), InternVL3 (38B), Gemma3 (27B), Gemma4 (31B), GLM-4.6V (106B MoE), GLM-4.6V-Flash (9B), MiniMax-M3 (428B MoE, AWQ-INT4), Llama3.2-Vision (11B), LLaVA-OneVision (72B); medical — Lingshu (32B), MedGemma (4B/27B), MedDr (40B), HuatuoGPT-Vision (34B), HealthGPT-L14 (14B).

**Software versions (paper):** `medvision_bm` v1.1.0 (benchmarking), `medvision_ds` v1.1.0 (dataset access), verl v0.7.0 (RFT). Eval on 4× H100 (80GB); MedVision-V0 trained on 4× H200 (140GB) with data + tensor parallelism.

**Intended use:** research/education only; not for clinical decision-making. The paper explicitly notes MedVision-V0 is far from clinical accuracy.
