# Hardware, Parallelism and Batch Sizing

Per-model VRAM tables and foundation pins live in the root roster `../../../references/model-roster.md`. This page is
about the run-time knobs: how many GPUs a run uses, what `batch_size_per_gpu` actually buys, and how to size
`gpu_memory_utilization`.

## `CUDA_VISIBLE_DEVICES` is the only GPU control

`set_cuda_num_processes()` reads `CUDA_VISIBLE_DEVICES` and returns the number of ids listed (at least 1); when the
variable is unset it returns `torch.cuda.device_count()` — i.e. **every** GPU on the host. That single number drives
three things:

```
num_processes  = len(CUDA_VISIBLE_DEVICES.split(",")) or torch.cuda.device_count()
batch_size     = batch_size_per_gpu * num_processes          # -> lmms_eval --batch_size
vLLM:  tensor_parallel_size = num_processes,  max_num_seqs = batch_size
HF:    accelerate launch --num_processes=<num_processes>
```

There is **no** `--tensor_parallel_size` flag and no `--num_processes` flag. To shard a model across 4 of 8 GPUs you
export `CUDA_VISIBLE_DEVICES=0,1,2,3`; exposing 8 shards it across 8. Always set the variable explicitly — an unset
variable on a shared node silently claims every GPU.

## Tensor parallel (vLLM keys)

- Weights are sharded, so the per-GPU footprint is `total_weights / TP + KV + activations`. The binding constraint is
  per-GPU, not aggregate.
- Some models have a **minimum** TP on 80 GB cards: GLM-4.6V (≈215 GB BF16) needs TP ≥ 4; TP = 2 gives ≈108 GB/GPU and
  will not fit. MiniMax-M3 in BF16 (≈856 GB) does not fit 80 GB cards at any practical TP — the benchmark uses the
  AWQ-INT4 checkpoint (≈214 GB, ≈54 GB/GPU at TP = 4) or MXFP8.
- Some have a natural **maximum**: GLM-4.6V-Flash is a dense ~10 B model (~21 GB BF16), so TP = 1 is the intended
  deployment; exposing more GPUs works but wastes interconnect.
- MoE caveat: for a mixture-of-experts model, "activated parameters" is compute per token, not memory. Every expert
  stays resident, so you pay for the total parameter count.
- KV cache is negligible for MedVision. Prompts are ~1.1 K tokens (image tokens + ~500-token text), so even 100
  concurrent sequences cost single-digit GB. Long-context capability is irrelevant here.

`gpu_memory_utilization` is the fraction of each GPU vLLM claims at engine start:

| Value | Where it is used | Why |
|---|---|---|
| 0.99 | module default | maximal KV; leaves almost nothing for fragmentation |
| 0.95 | Qwen3-VL, Gemma-4, GLM-4.6V(-Flash) launchers | |
| 0.90 | Qwen2.5-VL, Gemma-3, InternVL3, LLaVA-OneVision, Llama-3.2, MedVision-V0 launchers | |
| 0.90 | MiniMax-M3 launchers, **deliberately not 0.95** | a dead vLLM engine leaks ~4 GiB VRAM; at 0.95 the driver's auto-retry then fails the startup memory check on the remaining ~75 GiB and the run wedges in a restart loop |

Lower it when the engine OOMs at start-up or during vLLM's post-load profiling pass; raise it only when you know KV is
the limit. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the documented mitigation for the specific case where
profiling OOMs by a fraction of a GiB while a couple of GiB sit reserved-but-unallocated.

`cpu_offload_gb` (MiniMax-M3 only) is **per GPU worker**: with TP = 4 a value of N demands 4 × N GiB of host RAM, and
containers are usually capped far below the node's RAM (`memory.max` in the cgroup, not what `free -g` reports). The
repository sets 0 and warns against raising it; offload also streams weights every forward pass.

## Data parallel (HF keys)

`eval__medgemma`, `eval__lingshu`, `eval__meddr`, `eval__huatuogpt_vision`, `eval__llava_med` and `eval__healthgpt`
wrap `lmms_eval` in `python3 -m accelerate.commands.launch --num_processes=<GPUs> --main_process_port=29501` (29502 for `eval__llava_med`) — the port is hard-coded, so two concurrent HF runs on one node collide. Each process loads a **full replica**
on its own GPU and `lmms_eval` shards the samples by rank (`doc_iterator(rank, limit, world_size)`); results are
gathered with `torch.distributed.gather_object`. Throughput scales linearly with GPU count; VRAM does not shard, so
the whole model must fit one GPU.

Two variants exist and are functionally equivalent for inference:

- **With a DDP wrapper** (`accelerator.prepare_model()`): LLaVA-Med, HealthGPT, MedDr. The wrapper is inherited from
  training; the model property unwraps it before `generate`.
- **Without** (per-process `device_map`): MedGemma (`{"": local_rank}` on the pipeline path), Lingshu
  (`device_map=f"cuda:{local_rank}"`), HuatuoGPT-Vision (explicit `.to(device)`).

**`batch_size` is cosmetic for four of the six** — MedDr, HuatuoGPT-Vision, LLaVA-Med and HealthGPT loop one sample
per iteration inside `generate_until`, so `batch_size_per_gpu` only affects how `lmms_eval` pads request lists for
even rank distribution. **MedGemma and Lingshu genuinely batch** (MedGemma passes `batch_size` into its HF pipeline,
`medgemma.py:211`; Lingshu builds a padded batch), so raising it there raises both throughput and VRAM.
For the cosmetic four, raising it does not raise throughput — but each replica is a full model, so keep it small (the repository uses 2 for
MedGemma-27B, MedDr, Lingshu and HuatuoGPT-Vision) and add GPUs instead.

One host-RAM trap: HuatuoGPT-Vision's third-party loader materialises the model on CPU before moving it to the GPU, so
N simultaneous ranks briefly need N × 34 B of host RAM. On a memory-capped container that is an OOM kill with no
traceback. Stagger startup or use fewer ranks.

## API models

No local inference and no GPU. `--batch_size 1`; each wrapper issues one request per sample with exponential backoff
(10 tries, immediate give-up on HTTP 400). Throughput is provider-side rate limits, and the cost driver is
`--sample_limit` × `--max_tokens`.

## Choosing values in practice

1. **Decide TP first** from the model's per-GPU footprint (root roster), then export exactly those ids.
2. **Start from the launcher's `batch_size_per_gpu`** for that model (see `model-catalog.md`); it was chosen against
   the paper's 4 × H100 80 GB rig.
3. **On a vLLM OOM**: lower `gpu_memory_utilization` first (it is claimed at engine start), then
   `batch_size_per_gpu` (it is `max_num_seqs`, i.e. concurrency), then add GPUs. A model that does not fit at the
   maximum TP you have is not runnable on that machine — say so instead of shrinking the batch to 1 and hoping.
4. **On an HF data-parallel OOM**: only `batch_size_per_gpu` and the number of ranks matter; a replica that does not
   fit one GPU cannot be run by this path at all.
5. **Reference rig**: the paper evaluated every open-weight model on 4 × H100 80 GB (MedVision-V0 was trained on
   4 × H200 140 GB). Numbers quoted in `model-catalog.md` assume that machine.
