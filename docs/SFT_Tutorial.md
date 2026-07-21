# Supervised Fine-Tuning (SFT) for VLMs on Medical Image Data

This tutorial gets you from a fresh clone to a running SFT job in a few minutes, using the `medvision_bm` codebase.

> [!TIP]
> **Useful Resources**
> *   **Code**: [`medvision_bm`](https://github.com/YongchengYAO/MedVision) - The codebase for benchmarking and fine-tuning VLMs on medical image data.
> *   **Dataset**: [YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision) - A dataset for quantitative medical image analysis: 29K 3D images, 11.2M 2D slices, 24.3M single-instance annotations.
> *   **Project**: [MedVision Project Page](https://medvision-vlm.github.io)
> *   **Full SFT guide**: [medvision.readthedocs.io — SFT](https://medvision.readthedocs.io/en/latest/fine-tuning/sft.html) - Every parameter, recipe, and memory-tuning detail deferred from this page.

## 1. 📖 Introduction

Supervised Fine-Tuning (SFT) adapts a Vision-Language Model to a task by training it on **image-instruction-response triplets**. In MedVision, we use SFT to teach VLMs — currently **Qwen2.5-VL**, **Qwen3-VL** (Qwen3.5 / Qwen3.6), **Gemma-4**, and **MedGemma** — three quantitative medical tasks: **Angle/Distance (A/D)** estimation, **Detection** (bounding boxes), and **Tumor/Lesion Size (T/L)** estimation. All current recipes use **chain-of-thought (CoT) supervision**: the target response reasons inside `<think></think>` (filled from intermediate ground truth, e.g., landmark coordinates) and gives the final values inside `<answer></answer>` — teaching the model the measurement *procedure*, not just the number. Training combines all three tasks: 110K detection + 5.5K A/D + 5.5K T/L = 121K samples.

New to SFT in general? See the [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1).

## 2. 🚀 Quick Start

First, install the package (see the [installation guide](https://medvision.readthedocs.io/en/latest/getting-started/installation.html) for details):

```bash
git clone https://github.com/YongchengYAO/MedVision.git MedVision
cd MedVision
pip install .
```

Training runs are launched via the shell scripts in **`script/sft/`** — each one is a documented, ready-to-edit recipe that handles conda env setup, data preparation, and (distributed) training. The file names tell you what you get:

```text
train__{SFT|fullSFT}-CoT__{model}__D110k-AD5.5k-TL5.5k[__512x512][__4xGPU-140G-fp32master][__cmplLoss].sh
```

*   `SFT-CoT` = **LoRA**; `fullSFT-CoT` = **full-parameter** (FSDP).
*   `__512x512` = images reshaped to 512×512 (the MedVision-V0 recipe).
*   `__4xGPU-140G-fp32master` = full-FT recipe for 140GB-class GPUs with fully resumable checkpoints; without it, full-FT scripts use an anti-OOM pure-bf16 recipe that fits 80GB cards.
*   `__cmplLoss` = completion-only loss masking for the Gemma families.

Available scripts by model family:

| Model | LoRA | Full-FT (80GB) | Full-FT (140GB) |
| :-- | :-: | :-: | :-: |
| Qwen2.5-VL-7B | ✅ (+ native-res) | ✅ | — |
| Qwen3.5-27B | ✅ | ✅ | ✅ |
| Qwen3.6-27B | ✅ | ✅ | ✅ |
| Gemma-4-31B-it | ✅ | ✅ | ✅ |
| MedGemma-27B-it | ✅ | ✅ | ✅ |

Gemma-4 and MedGemma additionally ship `__cmplLoss` twins of their LoRA and full-FT scripts.

Open your chosen script and set the variables at the top — `benchmark_dir` (repo root), `data_dir` (data cache), `base_model_hf`, `model_family_name`, `run_name` / `wandb_*`, and the batch-size / GPU settings — then run it from the project root:

```bash
# LoRA SFT of Qwen2.5-VL-7B at 512x512 (the MedVision-V0 recipe)
bash script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh

# Full-parameter SFT of a 27B model on 4x 140GB GPUs, fully resumable
bash script/sft/train__fullSFT-CoT__Qwen3.5-27B__D110k-AD5.5k-TL5.5k__512x512__4xGPU-140G-fp32master.sh
```

Each script runs two phases:

1.  **Dataset processing** (CPU-heavy): downloads the data, formats every sample, and saves the prepared dataset to disk — kept separate so long preprocessing can't time out distributed training.
2.  **Training** (GPU): launches the training module with `accelerate launch`, loading the prepared dataset directly from disk. Re-runs and interrupted jobs reuse the cache and can resume from the last checkpoint.

> [!NOTE]
> `script/sft/v1.0.0/` and `script/sft/dev_medvision-sft/` contain legacy and development scripts; use the top-level scripts.

## 3. 🎯 Customizing a Run

Everything you routinely change lives in the shell script's variable block:

*   **Compute**: `per_device_train_batch_size`, `gradient_accumulation_steps`, `CUDA_VISIBLE_DEVICES` / `--num_processes` — match these to your GPUs.
*   **Data mix**: per-task sample limits (`train_sample_limit_task_AD` / `..._Detection` / `..._TL`, currently 5.5K / 110K / 5.5K). Because detection dominates, the scripts enable a temperature-based multi-task sampler that oversamples the minority tasks during training.
*   **Schedule & checkpoints**: `epoch` (10 for LoRA, 3 for full-FT), `save_steps`, `eval_steps`, `save_total_limit`, and `resume_from_checkpoint=true` to relaunch interrupted runs.
*   **Image size**: `--new_shape_hw 512 512` resizes images during preparation *and* re-adjusts the pixel size in the prompt, so the physical-spacing information always matches what the model sees.

**LoRA vs. full-parameter**: LoRA scripts train adapters on a frozen base model with plain DDP; `fullSFT` scripts update all weights and shard everything with FSDP. At 27–31B, full-FT further splits into the 80GB pure-bf16 recipe (weights-only checkpoints) and the 140GB fp32-master recipe (fully resumable). The memory math, `MEDVISION_SFT_*` environment knobs, and per-family loss-masking behavior are covered in the [full SFT guide](https://medvision.readthedocs.io/en/latest/fine-tuning/sft.html).

If you want to change *how* data is formatted or the training loop itself, the logic lives in `src/medvision_bm/sft/`: `sft_utils.py` (data preparation, samplers, trainer setup), `sft_prompts.py` (prompt and CoT templates), and the `train__*.py` entry points (one per model family and FT mode).

## 4. 📊 Logging & Model Uploads

Log in once before launching:

```bash
wandb login      # training metrics
hf auth login    # pushing models to the Hugging Face Hub
```

WandB behavior (`wandb_project`, `wandb_run_name`, resume settings) is configured in each script. The LoRA scripts can optionally push adapters and merged models to the Hub after training; full-parameter scripts save complete checkpoints directly. See the [full SFT guide](https://medvision.readthedocs.io/en/latest/fine-tuning/sft.html) for the option list.

## 5. 📚 References

*   [Full SFT guide (Read the Docs)](https://medvision.readthedocs.io/en/latest/fine-tuning/sft.html)
*   [Installation guide (Read the Docs)](https://medvision.readthedocs.io/en/latest/getting-started/installation.html)
*   [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1)
*   [TRL Documentation: SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
*   [MedGemma Fine-tuning Notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
