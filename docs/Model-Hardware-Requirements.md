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
