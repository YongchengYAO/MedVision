# Model Parallelism Summary

This document summarizes the parallelism strategy used by each model in the benchmark pipeline.

## Summary Table

| Model | Parallelism Type | DDP Wrapper | Notes |
|---|---|---|---|
| LLaVA-Med | Data Parallel | Yes | `accelerator.prepare_model()` wraps for DDP |
| HealthGPT L14 | Data Parallel | Yes | `accelerator.prepare_model()` wraps for DDP |
| HealthGPT XL32 | Data Parallel | Yes | `accelerator.prepare_model()` wraps for DDP |
| MedGemma | Data Parallel | No | Pipeline path; per-process `device_map={"": local_rank}` |
| Lingshu | Data Parallel | No | `from_pretrained(device_map=f"cuda:{local_rank}")` |
| HuatuoGPT-Vision | Data Parallel | No | Per-process device passed to `HuatuoChatbot(device=self._device)` |
| MedDr | Data Parallel | Yes | Loaded to per-process GPU via `.to(device)`; `accelerator.prepare_model()` wraps for DDP |
| InternVL3 | Tensor Parallel | N/A | vLLM `tensor_parallel_size=num_gpus` |
| LLaMA-3.2-Vision | Tensor Parallel | N/A | vLLM `tensor_parallel_size=num_gpus` |
| LLaVA-OneVision | Tensor Parallel | N/A | vLLM `tensor_parallel_size=num_gpus` |
| Qwen2.5-VL | Tensor Parallel | N/A | vLLM `tensor_parallel_size=num_gpus` |
| Gemma3 | Tensor Parallel | N/A | vLLM `tensor_parallel_size=num_gpus` |
| Gemini 2.5 | API | N/A | No local inference |

## Data Parallelism: With vs. Without DDP Wrapper

In all multi-process (`accelerate launch`) inference runs, each process loads a full copy of the model on its own GPU. lmms-eval shards the evaluation dataset by rank, so each process sees a disjoint subset of samples — this is the core of data parallelism and works regardless of whether a DDP wrapper is present.

**Why the DDP wrapper does not affect inference**

The DDP wrapper (`accelerator.prepare_model()`) exists to synchronize gradients across processes during training. At inference time, no backward pass runs, so the synchronization hooks are never triggered. The practical difference is cosmetic: models with the wrapper must call `accelerator.unwrap_model()` before generation to avoid the DDP shell intercepting generation APIs; models without it call `generate()` directly.

**With `accelerator.prepare_model()` — LLaVA-Med, HealthGPT, MedDr**

Model is assigned to the per-process GPU, then wrapped in a DDP shell. The `model` property unwraps it via `accelerator.unwrap_model()` before each generation call. This pattern is inherited from training pipelines where DDP is required.

**Without `accelerator.prepare_model()` — MedGemma, Lingshu, HuatuoGPT-Vision**

Model is assigned directly to the per-process GPU; no DDP shell is added. Cleaner for inference-only use but functionally equivalent.

Both strategies achieve the same inference throughput for a given number of GPUs.

## Details by Model

### Data Parallel — with DDP wrapper

These models use `accelerate launch --num_processes=N` and wrap the model with `accelerator.prepare_model()`, enabling proper DDP gradient synchronization (not needed for inference, but the wrapper is present).

**LLaVA-Med**
- Third-party loader: `load_pretrained_model()` from `third_party/LLaVA-Med/llava/model/builder.py`
- With default `device="cuda"`, no `device_map` is set — model loads to CPU then moved to `"cuda"`
- `llava_med.py` then calls `.to(self.device)` (per-process GPU) followed by `accelerator.prepare_model()`

**HealthGPT L14 / XL32**
- Third-party model: `LlavaPhiForCausalLM` from the HealthGPT codebase
- Loaded with `device_map={"": self.device}` (per-process GPU) directly at `from_pretrained` time
- `accelerator.prepare_model()` wraps for DDP

**MedDr**
- Third-party model: `InternVLChatModel` from the MedDr codebase (based on InternVL)
- Loaded to CPU at `from_pretrained` time (no `device_map`), then moved to per-process GPU via `.to(self.device)`
- `accelerator.prepare_model(evaluation_mode=True)` wraps for DDP; generation uses `self._model` directly (unwrapped via the `model` property)

### Data Parallel — without DDP wrapper

These models use `accelerate launch --num_processes=N` and assign each process to a dedicated GPU, but do not call `accelerator.prepare()`. This is sufficient for inference since lmms-eval shards data by rank regardless.

**MedGemma** (`use_pipeline=True`)
- `_setup_distributed_inference()` sets `self.device_map = {"": local_process_index}` for multi-GPU
- `pipeline("image-text-to-text", model_kwargs={"device_map": self.device_map})` — each process loads the full model on its own GPU
- No `accelerator.prepare()` called on the pipeline path

**Lingshu**
- `Qwen2_5_VLForConditionalGeneration.from_pretrained(device_map=f"cuda:{local_process_index}")`
- Accelerator is used only for device string resolution; no `accelerator.prepare()` called

**HuatuoGPT-Vision**
- `huatuogpt_vision.py` resolves `self._device = f"cuda:{local_process_index}"` via `setup_device_with_accelerate()`
- Passes `device=self._device` to `HuatuoChatbot()`; `cli.py` does `model.to(self.device)` with that device
- No `accelerator.prepare()` called; Accelerator is used solely for device assignment

### Tensor Parallel (vLLM)

**InternVL3, LLaMA-3.2-Vision, LLaVA-OneVision, Qwen2.5-VL, Gemma3**
- Launched via vLLM with `tensor_parallel_size=num_gpus`
- vLLM handles all inter-GPU communication internally

### API

**Gemini 2.5 (w/ tool and w/o tool)**
- Inference via Google API; no local model loading or GPU parallelism

## DDP Effectiveness Analysis

### How DDP Works in lmms-eval

Data parallelism in this benchmark is **sample-level**, not model-level. The lmms-eval evaluator (`evaluator.py`) splits samples across ranks using `doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)`, meaning each process receives a disjoint 1/N subset of the evaluation set. Results are gathered via `torch.distributed.gather_object()`. This mechanism is independent of whether a DDP wrapper is present — it's built into the evaluator.

**Throughput scales linearly with GPU count.** Each GPU processes its shard simultaneously. With N GPUs, N distinct samples are being processed in parallel.

### Per-Model Verification

All four data-parallel models are **DDP-effective** for inference:

| Model | DDP Method | Conflict Status | Notes |
|---|---|---|---|
| **LLaVA-Med** | `accelerator.prepare_model()` | ✅ Clean | DDP wrapper properly synchronized; data sharding works |
| **MedDr** | `accelerator.prepare_model(evaluation_mode=True)` | ✅ Clean | Wrapper prevents unwanted gradient hooks; data sharding works |
| **HealthGPT L14/XL32** | `accelerator.prepare_model()` | ✅ Clean | Same as MedDr; both use `evaluation_mode=True` |
| **HuatuoGPT-Vision** | Per-rank `device` assignment | ✅ Clean | No `device_map="auto"`; explicit `.to(device)` in `cli.py:81` binds model to per-rank GPU |

Each process loads a full model copy on its own GPU. Accelerator or explicit device assignment ensures no cross-rank interference. Data sharding happens in the evaluator, not in the model wrapper.

### Known Limitations

**1. Batch size is cosmetic**

The `--batch_size` argument passed to lmms_eval is calculated as `batch_size_per_gpu * num_processes`. However, all four models' `generate_until()` methods loop through requests **one sample per iteration** — there is no intra-GPU batching. The batch_size argument only affects how lmms_eval internally pads request lists for even rank distribution (padding for FSDP/DDP requirements), but it does not increase throughput per GPU. Each model processes `bs=1` effectively.

**2. HuatuoGPT-Vision: CPU RAM during startup**

`cli.py` loads the model via `from_pretrained(..., torch_dtype=torch.bfloat16)` without `device_map`, which loads the entire model to CPU RAM first, then moves it to the target GPU via `.to(self.device)`. When N DDP processes launch simultaneously, N × 34B of CPU RAM is consumed during initialization. This can cause host OOM even if GPU memory is sufficient. Mitigation: stagger process startup or use a single-process loader.

### Conclusion

DDP data parallelism is properly implemented and effective across all four models. The key insight is that the evaluator handles data distribution — each rank automatically gets its subset. Model wrappers exist for training compatibility (LLaVA-Med, MedDr, HealthGPT) or device binding (HuatuoGPT-Vision), not for inference parallelism. Both approaches achieve the same sample-level throughput scaling.
