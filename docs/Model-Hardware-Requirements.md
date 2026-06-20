# Model Hardware Requirements

VRAM and GPU-count estimates for serving each benchmark model under the MedVision eval pipeline
(vLLM inference unless noted). Figures are estimates derived from each model's config/card; exact
numbers depend on the vLLM build, the quantization kernel, batch size, and context length.

> [!NOTE]
> **How to read these tables.** The binding constraint is **per-GPU footprint**, not aggregate VRAM:
> with tensor parallelism, weights are sharded across GPUs, so each GPU must hold
> `total_weight_VRAM ÷ tensor_parallel_size` plus its share of the KV cache and activations.
> The eval scripts set `tensor_parallel_size = num_visible_GPUs` and `gpu_memory_utilization = 0.99`
> by default, so expose exactly the GPUs you intend to shard across via `CUDA_VISIBLE_DEVICES`.

> [!IMPORTANT]
> **Mixture-of-Experts (MoE) caveat.** For MoE models, the "activated parameters" figure describes
> **compute per token**, not memory. vLLM must keep **all** experts resident in VRAM (any expert can be
> routed to on the next token), so you pay for the **total** parameter count, not the activated subset.
> Sparsity buys FLOPs/latency, not footprint.

---

## MiniMax-M3

**Model:** [`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3) — native vision-language
model; **428B total parameters, ~23B activated** (sparse MoE: `num_local_experts=128`,
`num_experts_per_tok=4`, `n_shared_experts=1`), 60 text layers with MiniMax Sparse Attention (MSA),
CLIP-style vision tower, native dtype **bfloat16**. lmms-eval module: `vllm_minimax_m3`.

### 1. Weights (dominant term)

428B params × bytes-per-param:

| Precision | Weight VRAM | Per-GPU @ TP=8 | Fits 80 GB card? |
|---|---|---|---|
| **BF16** (native) | **~856 GB** | ~107 GB | ❌ exceeds 80 GB |
| **FP8** | **~428 GB** | ~54 GB | ✅ |
| INT4 / AWQ | ~214 GB | ~27 GB | ✅ |

### 2. KV cache

From `config.json`: 60 layers, `num_key_value_heads=4`, `head_dim=128`, bf16 →
`2 × 60 × 4 × 128 × 2 B ≈ 0.12 MB/token`.

MedVision prompts are short: MiniMax-M3 emits `image_seq_length = 576` tokens per image plus a
~500-token text prompt ≈ **~1.1K tokens/sample → ~130 MB**. Even batching 100 concurrent sequences is
only ~13 GB. **Negligible** next to weights. (MSA would only reduce resident KV further; the model's
1M-context capability is irrelevant to MedVision's short prompts.)

### 3. Overhead

Vision tower + projector (~1–2 GB; `vision_config` is small — 1280-dim, 32 layers) plus vLLM workspace,
CUDA graphs, and allocator fragmentation (~5–15% of weights).

### 4. Aggregate targets

| Precision | Aggregate VRAM | Practical deployment |
|---|---|---|
| **BF16** | **~900 GB – 1 TB** | **8× H200 (141 GB = 1128 GB), TP=8** — comfortable; or 12–16× 80 GB |
| **FP8** | ~470–520 GB | **8× H100 (80 GB), TP=8** — fits; or **4× H200 (141 GB)** — tight on KV/batch |

### 5. Deployment notes

- **BF16 will not fit 80 GB cards** at TP=8 (~107 GB/GPU). Use H200/B200, increase GPU count, or run FP8.
- The MedVision-V0 training rig (**4× H200 = 564 GB**) is **not enough for BF16** (856 GB); it would fit
  **FP8** (~428 GB) with a modest batch size.
- **vLLM architecture support is a hard prerequisite.** The checkpoint ships no HF modeling file (only
  `configuration_*.py`), so the forward pass must come from a **native vLLM/SGLang implementation** of the
  `minimax_m3_vl` architecture (`MiniMaxM3SparseForConditionalGeneration`). If `vllm` raises an
  "unsupported architecture" error, set `eval__minimax_m3.py --vllm_version` to a release that registers it.

---

## GLM-4.6V

**Model:** [`zai-org/GLM-4.6V`](https://huggingface.co/zai-org/GLM-4.6V) — native vision-language model;
**107.7B total parameters, ~12B activated** (sparse MoE: `n_routed_experts=128`, `num_experts_per_tok=8`,
`n_shared_experts=1`, `first_k_dense_replace=1` so 1 dense + 45 MoE of 46 text layers), GLM-4V vision tower
(24 layers, 1536-dim), native dtype **bfloat16** (HF safetensors report BF16; no shipped FP8 checkpoint).
lmms-eval module: `vllm_glm4v`; eval driver `eval__glm4v.py` (`--model_hf_id zai-org/GLM-4.6V`).

### 1. Weights (dominant term)

107.7B params × bytes-per-param:

| Precision | Weight VRAM | Per-GPU @ TP=4 | Fits 80 GB card? |
|---|---|---|---|
| **BF16** (native) | **~215 GB** | ~54 GB | ✅ at TP≥4 (TP=2 → ~108 GB/GPU ❌) |
| **FP8** (runtime quant) | **~108 GB** | ~27 GB | ✅ |
| INT4 / AWQ | ~54 GB | ~13.5 GB | ✅ |

### 2. KV cache

From `config.json`: 46 layers, `num_key_value_heads=8`, `head_dim=128`, bf16 →
`2 × 46 × 8 × 128 × 2 B ≈ 0.18 MB/token`.

MedVision prompts are short: a GLM-4V image (smart-resized, 14-px patches / merge-2 → 28-px effective) plus a
~500-token text prompt ≈ **~1K tokens/sample → ~0.18 GB**. Batching 64 concurrent sequences ≈ **~12 GB
aggregate** (~3 GB/GPU at TP=4, since the 8 KV heads shard across GPUs). Small next to weights; the model's
128K context is irrelevant to MedVision's short prompts.

### 3. Overhead

GLM-4V vision tower + projector (~1–2 GB; 24 layers @ 1536-dim) plus vLLM workspace, CUDA graphs, and allocator
fragmentation (~5–15% of weights).

### 4. Aggregate targets

| Precision | Aggregate VRAM | Practical deployment |
|---|---|---|
| **BF16** | **~230–250 GB** | **4× H100 (80 GB = 320 GB), TP=4** — comfortable (matches the paper's eval rig); or 4× H200 |
| **FP8** | ~120–140 GB | **2× H100 (80 GB), TP=2** — fits; 4× for batch/KV headroom |

The MedVision-V0 training rig (**4× H200 = 564 GB**) fits BF16 (~215 GB) with generous batch headroom.

### 5. Deployment notes

- **BF16 needs TP≥4 on 80 GB cards** — at TP=2 the ~108 GB/GPU shard exceeds 80 GB. The eval script sets
  `tensor_parallel_size = num_visible_GPUs`, so expose exactly 4 GPUs via `CUDA_VISIBLE_DEVICES`.
- **MoE memory caveat applies** (see top): all 128 experts stay resident; the ~12B activated figure is
  compute-per-token, not footprint.
- **Version constraint is tight — use vLLM 0.19.x + transformers 5.12.1 (the eval driver pins these).**
  GLM-4.6V's `preprocessor_config.json` declares `Glm46VImageProcessor`/`Glm46VProcessor`, which exist
  ONLY in **transformers ≥ 5.2.0** (transformers 4.57.x has just `Glm4vImageProcessor` → "Unrecognized
  image processor"). transformers 5.x in turn needs a vLLM that accepts it: **vLLM 0.12.0 pins
  `transformers<5` and CANNOT run GLM-4.6V**, but **vLLM 0.19.x** excludes only transformers 5.0–5.5.0
  (no upper bound) and ships the `ALLOWED_LAYER_TYPES` fallback, so it imports on transformers 5.x. The
  working window is the intersection: **transformers ≥ 5.6** (≥5.2.0 for GLM ∩ ≥5.6 for vLLM 0.19.x) —
  5.12.1 is used. Unlike Qwen3-VL, no `hf_overrides` are needed — vLLM supports GLM-4V natively.
  See [New-Models-Guide](New-Models-Guide.md).
- **Known ecosystem risk:** GLM-4.6V + transformers 5.x on vLLM is bleeding-edge; multimodal serving has
  had open bugs (e.g. vLLM #30584, the server passing `MediaWithBytes` into `Glm46VProcessor`). The
  MedVision path uses offline `LLM.chat`, which may avoid it, but verify the first run produces image
  outputs (not 400s / empty responses).
- The launcher `script/benchmark-{AD,TL,detect}/eval__GLM-4.6V__*.sh` defaults to `batch_size_per_gpu=1`
  (108B is heavy); raise it if VRAM allows.

---

## GLM-4.6V-Flash

**Model:** [`zai-org/GLM-4.6V-Flash`](https://huggingface.co/zai-org/GLM-4.6V-Flash) — native vision-language
model; **10.3B total parameters, dense** (`Glm4vForConditionalGeneration`, 40 text layers, 4096-dim, GLM-4V
vision tower), native dtype **bfloat16**. The lightweight, low-latency sibling of GLM-4.6V; shares the same
lmms-eval module `vllm_glm4v` and eval driver (`eval__glm4v.py --model_hf_id zai-org/GLM-4.6V-Flash`).

### 1. Weights (dominant term)

10.3B params × bytes-per-param:

| Precision | Weight VRAM | Fits single 80 GB? | Fits single 32 GB? |
|---|---|---|---|
| **BF16** (native) | **~21 GB** | ✅ | ✅ (≈21 GB + KV + overhead ≈ 26 GB) |
| **FP8** (runtime quant) | ~10 GB | ✅ | ✅ |

### 2. KV cache

From `config.json`: 40 layers, `num_key_value_heads=2`, `head_dim=128`, bf16 →
`2 × 40 × 2 × 128 × 2 B ≈ 0.04 MB/token`. A ~1K-token MedVision sample ≈ ~40 MB; batching 100 ≈ **~4 GB**.
Negligible.

### 3. Overhead

GLM-4V vision tower + projector (~1–2 GB) plus vLLM workspace, CUDA graphs, and fragmentation.

### 4. Aggregate targets

| Precision | Aggregate VRAM | Practical deployment |
|---|---|---|
| **BF16** | **~26–30 GB** | **1× H100 / H200 / A100-80GB, TP=1** — comfortable; fits a single **≥32 GB** card with MedVision's short prompts (24 GB is tight once KV + overhead are added) |

### 5. Deployment notes

- **Single GPU (TP=1) is the intended deployment.** The eval script sets
  `tensor_parallel_size = num_visible_GPUs`, so expose exactly **one** GPU via `CUDA_VISIBLE_DEVICES`; exposing
  more shards a 10B model needlessly (works, but wastes inter-GPU bandwidth).
- **vLLM 0.19.x + transformers 5.12.1** (the `glm4v` / `Glm4vForConditionalGeneration` architecture) —
  same constraint as GLM-4.6V above: GLM-4.6V-Flash needs transformers ≥5.2.0 for `Glm46VImageProcessor`,
  and vLLM 0.19.x is the released engine that accepts transformers 5.6+. vLLM 0.12.0 cannot run it. No
  `hf_overrides` needed.
- The launcher `script/benchmark-{AD,TL,detect}/eval__GLM-4.6V-Flash__*.sh` defaults to `batch_size_per_gpu=2`;
  this small model can take a much larger batch on an 80 GB card.
