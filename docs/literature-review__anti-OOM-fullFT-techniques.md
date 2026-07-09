# Anti-OOM Techniques for 27–31B Full-Parameter SFT — Primary-Source Review

_Compiled 2026-07-08. Scope: the memory-saving techniques composed in the `script/sft/train__fullSFT-CoT__*__512x512.sh` launchers to fit **full-parameter** (not LoRA) supervised fine-tuning of 27–31B vision-language models onto **4×80GB** GPUs with PyTorch FSDP `FULL_SHARD` + HuggingFace transformers/TRL. For each technique this maps the launcher knob to the **original paper or authoritative official documentation**, states what that source establishes, and records the caveat the source itself raises._

> **How this was built.** Sources were gathered by an automated deep-research sweep (5 angles → 24 primary sources → 119 extracted claims → adversarial 3-vote verification). Every citation below was fetched from its live primary page (arXiv abstract, official PyTorch / HuggingFace / bitsandbytes / torchtune docs, or ICLR/NeurIPS record) and the load-bearing claim quoted verbatim. Provenance of each claim is tagged: **✓✓ verified** (adversarially confirmed by the workflow), **✓ fetched** (quoted directly from the primary page during compilation), so a reader can see which statements had the stronger check.

---

## 1. The recipe & knob → technique → source map

The launchers turn the default fp32-master mixed-precision recipe (which costs a fixed **~67.5 GB/GPU at 27B / ~77.5 GB/GPU at 31B across 4 ranks** and cannot fit 80 GB) into a **pure-bf16** recipe that peaks ~52–55 GB/GPU at 27B. It composes eight techniques, each behind an env knob or launch flag:

| Recipe lever (launcher) | Technique | Primary source(s) |
|---|---|---|
| `accelerate launch --use_fsdp --fsdp_sharding_strategy FULL_SHARD` | Shard params + grads + optimizer state across ranks | ZeRO (arXiv:1910.02054); PyTorch FSDP (arXiv:2304.11277); PyTorch FSDP docs |
| `MEDVISION_SFT_SYNC_EACH_BATCH=1` | Disable `no_sync` so grads reduce-scatter every micro-batch (accumulate **sharded**, not full-unsharded) | PyTorch FSDP paper §grad-accum; FSDP `no_sync` docs; Accelerate `sync_each_batch` docs |
| `MEDVISION_SFT_PURE_BF16=1` + **no** `--mixed_precision` | Pure bf16 — no fp32 master weights | Kalamkar et al. (arXiv:1905.12322); Dobler & de Melo (arXiv:2408.15793); torchtune docs |
| `MEDVISION_SFT_LR=4e-5` | Mitigate the bf16 update rounding floor | Zamirai et al. "Revisiting BFloat16 Training" (arXiv:2010.06192) — *canonical fix is stochastic rounding / Kahan, **not** raising LR* |
| `MEDVISION_SFT_OPTIM=adamw_bnb_8bit` (non-paged) | 8-bit block-wise quantized AdamW; **reject** the paged variant | Dettmers et al. ICLR 2022 (arXiv:2110.02861); QLoRA NeurIPS 2023 (arXiv:2305.14314); bitsandbytes optimizer docs |
| `MEDVISION_SFT_USE_LIGER=1` (MedGemma only) | Fused linear cross-entropy — never materialize the seq×vocab logits | Liger Kernel (arXiv:2410.10989); Cut Cross-Entropy (arXiv:2411.09009) |
| `--gradient_checkpointing` | Recompute activations in backward | Chen et al. 2016 (arXiv:1604.06174); PyTorch `checkpoint` docs |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce allocator fragmentation | PyTorch CUDA semantics docs |

---

## 2. Technique 1 — Fully sharded data parallelism (`FULL_SHARD`)

**[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** — Rajbhandari, Rasley, Ruwase, He (Microsoft) · arXiv:1910.02054 · 2019, published at **SC '20**
- *Establishes (✓✓ verified):* ZeRO-DP stage 3 (P_os+g+p) partitions **optimizer states, gradients, and parameters** across the N_d data-parallel ranks. *"Memory reduction is linear with DP degree N_d… splitting across 64 GPUs will yield a 64× memory reduction. There is a modest 50% increase in communication volume."* This is the direct antecedent of FSDP `FULL_SHARD`.
- *Also establishes (✓✓ verified):* the baseline the whole recipe attacks — *"Mixed-precision Adam has K=12. In total, this results in 2Ψ+2Ψ+KΨ=16Ψ bytes"* (2 B fp16 param + 2 B fp16 grad + 12 B fp32 optimizer state = **16 bytes/param**).
- *Caveat:* the ~50% extra communication volume is the price of stage-3 sharding.

**[PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)** — Zhao et al. (Meta) · arXiv:2304.11277 · **PVLDB 16(12):3848–3860, 2023**
- *Establishes (✓✓ verified):* the exact `FULL_SHARD` semantics the recipe relies on — *"During forward and backward computation, FSDP only materializes unsharded parameters and gradients of one unit at a time, and otherwise, it keeps parameters and gradients sharded. Throughout the training loop, the optimizer states are kept sharded."* Only one unit's full weights are all-gathered on device at a time and freed after, which is what makes 27–31B full-FT feasible on 4×80GB.

**[PyTorch FSDP API docs](https://docs.pytorch.org/docs/main/fsdp.html)** (official)
- *Establishes (✓✓ verified):* *"Parameters, gradients, and optimizer states are sharded. …unshards (via all-gather) before the forward, reshards after… For gradients, it synchronizes and shards them (via reduce-scatter) after the backward… The sharded optimizer states are updated locally per rank."* The official definition of `FULL_SHARD` (≈ ZeRO-3).

---

## 3. Technique 2 — Sharded gradient accumulation (disable `no_sync`)

This is the fix for the step-0 first-backward OOM that motivated the whole campaign.

**[PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)** (arXiv:2304.11277) — the tradeoff, stated primary-source
- *Establishes (✓✓ verified):* *"With communication, FSDP still reduces gradients across ranks, and each rank saves the sharded gradients… Without communication, FSDP does not reduce gradients across ranks, and each rank saves the **unsharded** gradients. This latter variation trades off increased memory usage with decreased communication."* Skipping the reduce-scatter (i.e. `no_sync`) forfeits gradient sharding — at 27B that is ~48 GB/rank of full-unsharded grads accumulating during the first backward.

**[PyTorch FSDP `no_sync` docs](https://docs.pytorch.org/docs/stable/fsdp.html)** (official)
- *Establishes (✓ fetched):* the `no_sync()` context "will require additional memory" under FSDP because gradients are held in full (unsharded) form until the deferred sync. Outside `no_sync`, FSDP reduce-scatters gradients on **every** backward — i.e. sync-each-micro-batch is FSDP's default, and `no_sync` is an opt-in communication-saving mode.

**[Accelerate — Gradient Synchronization](https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization)** (official) — the decisive corroboration
- *Establishes (✓ fetched):* *"in memory intensive situations while using FSDP, we recommend to set `sync_each_batch` to `True` in the GradientAccumulationPlugin to disable `no_sync`."* Backed by a concrete benchmark fine-tuning **Mixtral 8×7B (47B) on 8×A100-80GB**: with `no_sync` enabled the run goes **OOM already at `gradient_accumulation_steps=2`**, whereas with `no_sync` disabled (`sync_each_batch=True`) the memory at **`accum=16` reverts to the `accum=1` footprint of 69 GB**.
- *Caveat (stated):* *"Disabling `no_sync` means there will be slowdown due to the extra data syncs"* — one reduce-scatter per micro-batch instead of one per optimizer step. (In the launchers this is negligible over NVLink.)
- *Implementation note:* transformers ≤5.5 hardcodes the `no_sync` wrap and does **not** consult `sync_each_batch`; the launcher's `MEDVISION_SFT_SYNC_EACH_BATCH` knob therefore monkeypatches `accelerator.no_sync` → `nullcontext` to achieve the same effect. See also [transformers#29425](https://github.com/huggingface/transformers/issues/29425).

---

## 4. Technique 3 — Pure bf16 training (no fp32 master weights)

**[A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)** — Kalamkar et al. (Intel/Facebook/…) · arXiv:1905.12322 · 2019
- *Establishes (✓ fetched):* first broad empirical study that bf16 tensors reach FP32-equivalent accuracy across image classification, speech, LMs, GANs, and recommendation, in the same iterations and hyperparameters, because bf16 has FP32-equivalent dynamic range (no loss-scaling needed, unlike fp16).
- **Critical caveat for this recipe (✓ fetched):** this is a **mixed-precision** study — it keeps an **fp32 master copy** of the weights for the optimizer update. Its parity result does **not** by itself validate *pure* bf16 (bf16 params/grads/optimizer, no master). That gap is filled by the next paper.

**[Language Adaptation on a Tight Academic Compute Budget: Tokenizer Swapping Works and Pure bfloat16 Is Enough](https://arxiv.org/abs/2408.15793)** — Dobler & de Melo · arXiv:2408.15793 · **WANT@ICML 2024**
- *Establishes (✓✓ verified):* *"pure bfloat16 training is a viable alternative to mixed-precision training, while being much faster when only using a few GPUs"* — matching loss and even outperforming on some downstream evals for continued LLM pretraining. Reports **39.2% faster on 2 GPUs, 31.0% on 4 GPUs**, and — directly on point — mixed precision hits **OOM on a single 80GB GPU where pure bf16 fits**. This is the closest published validation of the recipe's pure-bf16 choice.

**[torchtune memory optimizations](https://meta-pytorch.org/torchtune/0.5/tutorials/memory_optimizations.html)** (official)
- *Establishes (✓ fetched):* *"if your hardware supports training with `bfloat16`, we recommend using it — this is the default setting for our recipes"* (2 bytes per model **and optimizer** parameter). A production framework trains full-bf16 by default.
- *Caveat (this recipe's own note):* pure bf16 is validated for **short SFT**; long runs risk stale weights (Technique 4). The launchers watch the early wandb loss curve and can revert.

---

## 5. Technique 4 — Mitigating the bf16 update rounding floor

The launchers raise LR 2e-5 → 4e-5 so AdamW updates clear bf16's ~0.4% relative resolution. This is an **engineering mitigation, not a validated recipe** — the primary source below establishes the *problem* and the *canonical* fixes, which are different.

**[Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)** — Zamirai, Zhang, Aberger, De Sa · arXiv:2010.06192 · 2020
- *Establishes (✓ fetched):* the root cause — *"nearest rounding for model weight updates often cancels small updates, which degrades the convergence and model accuracy"* (the "stale weights" / swamping problem when a small bf16 update is added to a large bf16 weight).
- *Canonical fixes (✓ fetched):* **stochastic rounding** and **Kahan summation**, delivering "up to 7% absolute validation accuracy gain in 16-bit-FPU training" and closing to within 0.1–0.2% of fp32. These — not LR-raising — are the literature-endorsed mitigations; e.g. torchao's `AnyPrecisionAdamW` implements Kahan-summation-based low-precision Adam.
- *Implication:* the launcher's LR bump is a pragmatic stopgap; if the loss curve misbehaves, the principled path is stochastic rounding / Kahan, or restoring fp32 masters at ≥8 GPUs. The launcher comments flag this honestly.

---

## 6. Technique 5 — 8-bit optimizers (and why the recipe rejects the *paged* variant)

**[8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861)** — Dettmers, Lewis, Shleifer, Zettlemoyer · arXiv:2110.02861 · **ICLR 2022 (spotlight)**
- *Establishes (✓ fetched):* *"Our 8-bit optimizers maintain 32-bit performance with a small fraction of the memory footprint… without changes to the original optimizer hyperparameters,"* via block-wise quantization (independently quantized blocks isolate outliers), dynamic quantization, and a stable embedding layer. This is `adamw_bnb_8bit` — ~13.5 GB/rank optimizer state at 27B/4 instead of ~54 GB fp32.

**[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** — Dettmers, Pagnoni, Holtzman, Zettlemoyer · arXiv:2305.14314 · **NeurIPS 2023**
- *Establishes (✓ fetched):* introduces *"paged optimizers to manage memory spikes"* (as one of QLoRA's three innovations), built on CUDA unified memory to page optimizer state to CPU during gradient-checkpointing spikes.

**[bitsandbytes — 8-bit optimizers](https://huggingface.co/docs/bitsandbytes/main/en/explanations/optimizers)** (official) — the reason the recipe uses **non-paged**
- *Establishes (✓ fetched):* *"Paged optimizers work like regular CPU paging, which means that it **only becomes active if you run out of GPU memory**. When that happens, memory is transferred page-by-page from GPU to CPU."* So paged state is **GPU-resident until pressure** — exactly the observed failure mode: on the twin MedGemma-27B run the paged state squatted on-device as unified-memory pages *outside* the torch allocator (`device_used − allocated` = 13–27 GB), which torch's `cudaMalloc` could not reclaim, OOMing the next micro-batch. `adamw_bnb_8bit` keeps the same 8-bit state **inside** the torch pool. (Corroborated by torchtune's note that `PagedAdamW8bit` "will also offload to CPU if there isn't enough GPU memory available.")
- *Caveat (stated):* bnb 8-bit optimizer state cannot be gathered by FSDP `FULL_STATE_DICT`, so the launchers set `MEDVISION_SFT_SAVE_ONLY_MODEL=1` (weights-only checkpoints; resume restarts the optimizer).

---

## 7. Technique 6 — Fused linear cross-entropy (large-vocab logits)

Relevant because Gemma-3/MedGemma has a **262k** vocabulary; the seq×vocab logits tensor is a major fp32 spike at the loss.

**[Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989)** — Hsu, Dai, Kothapalli, Song, Tang, Zhu, Shimizu, Sahni, Ning, Chen (LinkedIn) · arXiv:2410.10989 · 2024
- *Establishes (✓ fetched):* Liger's fused kernels (most importantly fused linear cross-entropy) deliver *"on average a 20% increase in training throughput and a 60% reduction in GPU memory usage for popular LLMs compared to HuggingFace implementations."* Enabled via `use_liger_kernel=True` (the `MEDVISION_SFT_USE_LIGER` knob).
- *Recipe note:* enabled only for **MedGemma** (Gemma-3, 262k vocab). **Not** enabled for Qwen3.5/3.6 (no Liger kernels for the qwen3_5 hybrid arch) or Gemma-4 (arch support unverified) — with Liger installed but the arch unsupported, transformers warns and runs the stock unfused loss.

**[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)** — Wijmans, Huval, Hertzberg, Koltun, Krähenbühl (Apple) · arXiv:2411.09009 · 2024
- *Establishes (✓ fetched):* the mechanism — *"CCE only computes the logit for the correct token and evaluates the log-sum-exp over all logits on the fly"*, never materializing the full logit matrix. For **Gemma-2 (2B)** it *"reduces the memory footprint of the loss computation from 24 GB to 1 MB, and the total training-time memory consumption of the classifier head from 28 GB to 1 GB"* — *"without sacrificing training speed or convergence."* Quantifies why the large-vocab logits spike is the marginal consumer this technique removes.

---

## 8. Technique 7 — Gradient (activation) checkpointing

**[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)** — Chen, Xu, Zhang, Guestrin · arXiv:1604.06174 · 2016
- *Establishes (✓ fetched):* the original algorithm achieving *"O(√n) memory to train an n-layer network, with only the computational cost of an extra forward pass per mini-batch"* — e.g. a 1,000-layer ResNet from 48 GB to 7 GB at ~30% extra runtime.
- *Caveat (stated):* it is a compute-for-memory trade; the extra forward recompute costs throughput.

**[PyTorch `torch.utils.checkpoint`](https://docs.pytorch.org/docs/main/checkpoint.html)** & **[torchtune memory docs](https://meta-pytorch.org/torchtune/0.5/tutorials/memory_optimizations.html)** (official)
- *Establishes (✓ fetched):* checkpointed regions "omit saving tensors for backward and recompute them during the backward pass"; *"these savings in memory come at the cost of training speed."* The launchers set `gradient_checkpointing=true` with `use_reentrant=False`.

---

## 9. Technique 8 — `expandable_segments`

**[PyTorch CUDA semantics — Memory management](https://docs.pytorch.org/docs/main/notes/cuda.html)** (official)
- *Establishes (✓ fetched):* with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` the caching allocator creates one large segment per stream that **grows as needed** instead of many fixed allocations, so variable batch/sequence sizes across iterations no longer leave unusable "slivers" — reducing fragmentation and reserved-but-unallocated memory, making it likelier the run avoids OOM.
- *Caveat (stated):* **experimental**, defaults to `False`, and only works with `backend:native` (ignored under `backend:cudaMallocAsync`).

---

## 10. The one honest gap

Seven of the eight techniques map cleanly to a peer-reviewed paper or official doc. The exception is **Technique 4's specific choice** — raising LR to clear the bf16 rounding floor. The primary literature (Zamirai et al.) validates the *problem* and endorses *stochastic rounding / Kahan summation* as the fix; **no authoritative source validates LR-raising as the mitigation.** The launchers therefore treat it as a monitored heuristic (watch the loss curve; revert to 2e-5 or restore fp32 masters at ≥8 GPUs), and this document records that limitation rather than papering over it.

---

## Appendix — full citation list

| # | Technique | Citation | ID / URL |
|---|---|---|---|
| 1 | FULL_SHARD | Rajbhandari et al., *ZeRO*, SC'20 | arXiv:1910.02054 |
| 2 | FULL_SHARD | Zhao et al., *PyTorch FSDP*, PVLDB 2023 | arXiv:2304.11277 |
| 3 | FULL_SHARD | PyTorch FSDP API docs | docs.pytorch.org/docs/main/fsdp.html |
| 4 | Sharded grad-accum | Accelerate — Gradient Synchronization | huggingface.co/docs/accelerate/…/gradient_synchronization |
| 5 | Sharded grad-accum | transformers no_sync/sync_each_batch issue | github.com/huggingface/transformers/issues/29425 |
| 6 | Pure bf16 | Kalamkar et al., *A Study of BFLOAT16* | arXiv:1905.12322 |
| 7 | Pure bf16 | Dobler & de Melo, WANT@ICML 2024 | arXiv:2408.15793 |
| 8 | Pure bf16 | torchtune memory optimizations | meta-pytorch.org/torchtune/0.5/… |
| 9 | Rounding floor | Zamirai et al., *Revisiting BFloat16 Training* | arXiv:2010.06192 |
| 10 | 8-bit optim | Dettmers et al., *8-bit Optimizers*, ICLR 2022 | arXiv:2110.02861 |
| 11 | Paged optim | Dettmers et al., *QLoRA*, NeurIPS 2023 | arXiv:2305.14314 |
| 12 | Paged optim | bitsandbytes optimizer docs | huggingface.co/docs/bitsandbytes/…/optimizers |
| 13 | Fused linear CE | Hsu et al., *Liger Kernel* | arXiv:2410.10989 |
| 14 | Fused linear CE | Wijmans et al., *Cut Cross-Entropy* (Apple) | arXiv:2411.09009 |
| 15 | Activation checkpointing | Chen et al., *Sublinear Memory Cost* | arXiv:1604.06174 |
| 16 | Activation checkpointing | PyTorch `torch.utils.checkpoint` docs | docs.pytorch.org/docs/main/checkpoint.html |
| 17 | expandable_segments | PyTorch CUDA semantics docs | docs.pytorch.org/docs/main/notes/cuda.html |
