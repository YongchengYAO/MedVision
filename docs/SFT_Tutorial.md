# Supervised Fine-Tuning (SFT) for VLMs on Medical Image Data

This tutorial guides you through the process of Supervised Fine-Tuning (SFT) for Vision-Language Models (VLMs) on medical image data, using the `medvision_bm` codebase.

> [!TIP]
> **Useful Resources**
> *   **Code**: [`medvision_bm`](https://github.com/YongchengYAO/MedVision) - The codebase for benchmarking and fine-tuning VLMs on medical image data.
> *   **Dataset**: [YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision) - A dataset for quantitative medical image analysis with 30.8M annotated samples.
> *   **Project**: [MedVision Project Page](https://medvision-vlm.github.io)

## 1. 📖 Introduction to SFT

Supervised Fine-Tuning (SFT) is a crucial step in adapting Large Language Models (LLMs) and Vision-Language Models (VLMs) to specific tasks or domains. It involves training the model on a dataset of instruction-response pairs (or in our case, **image-instruction-response triplets**) to learn how to follow instructions and generate desired outputs.

For a deeper dive into the concepts of SFT, we recommend the [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1).

In the context of `medvision_bm`, we use SFT to teach **Qwen2.5-VL** to perform specific medical tasks:

*   **Angle/Distance (A/D)**: Estimating angles or distances in medical images.
*   **Detection**: Identifying bounding boxes for anatomical structures.
*   **Tumor/Lesion Size (T/L)**: Estimating the major and minor axes of tumors or lesions.

All current recipes use **chain-of-thought (CoT) supervision**: the target response contains reasoning inside `<think></think>` followed by the final values inside `<answer></answer>`. The CoT text is filled from intermediate ground truth (e.g., landmark coordinates), teaching the model the measurement *procedure*, not just the final number.

> [!NOTE]
> Detailed descriptions of these tasks can be found on the [MedVision Project Page](https://medvision-vlm.github.io).


## 2. 🚀 Quick Start

The recommended way to run training is using the provided shell scripts in `script/sft/`. These scripts handle environment setup, variable definition, and launching the training process (including distributed training).

All current scripts fine-tune **Qwen2.5-VL-7B** with CoT supervision on the combined detection / A-D / T-L data (110K + 5.5K + 5.5K = 121K samples):

*   `script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh`: **LoRA** SFT at the model's native (dynamic) resolution.
*   `script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh`: **LoRA** SFT with images reshaped to **512×512**.
*   `script/sft/train__fullSFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh`: **Full-parameter** SFT with images reshaped to **512×512** (FSDP).

```bash
# Install the package
git clone https://github.com/YongchengYAO/MedVision.git MedVision
cd MedVision
pip install .
```

Before launching, set these variables at the top of the script:

*   `benchmark_dir`: the working directory (the repo root).
*   `data_dir`: where the MedVision data will be downloaded/cached.
*   `base_model_hf`: Hugging Face ID of the base model, or a path to a local model folder.
*   `model_family_name`: must be a model name registered in `AVAILABLE_MODELS` in `src/medvision_bm/medvision_lmms_eval/lmms_eval/models/__init__.py`; the `vllm_` prefix may be omitted (e.g., `qwen25vl` matches `vllm_qwen25vl`). Validated at startup by `check_model_supported()` in `sft_utils.py`.
*   `run_name`, `wandb_*`: identifiers for the run and logging.
*   Resource configs: `per_device_train_batch_size`, `gradient_accumulation_steps`, and `CUDA_VISIBLE_DEVICES` / `--num_processes` to match your GPUs.

Then simply execute the shell script from the project root:

```bash
# LoRA SFT, 512x512 (the MedVision-V0 recipe)
bash script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh
# or LoRA SFT at native resolution
bash script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh
# or full-parameter SFT, 512x512
bash script/sft/train__fullSFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh
```

### What the scripts do

Each script runs the same pipeline:

1.  **Environment setup**: creates (or reuses) a dedicated conda env (`sft-qwen25vl`), builds `medvision_bm` as a wheel, and installs SFT dependencies via `python -m medvision_bm.sft.env_setup`.
2.  **Phase 1 — dataset processing** (CPU-heavy): the training module is first invoked with `--process_dataset_only true`, which downloads the data, formats every sample, and saves the prepared dataset to disk. Offloading this to a separate run avoids distributed-training timeouts during long preprocessing.
3.  **Phase 2 — training** (GPU): the same module is launched with `accelerate launch` and `--skip_process_dataset true`, loading the prepared dataset directly from disk.

> [!NOTE]
> `script/sft/v1.0.0/` and `script/sft/dev_medvision-sft/` contain legacy and development scripts; use the three top-level scripts above.


## 3. 💿 Data Preparation

The data preparation pipeline is handled by the `prepare_dataset` function in `src/medvision_bm/sft/sft_utils.py`. This orchestrates the loading, formatting, and cleaning of data for each task.

### 3.1 Loading and Splitting
*   **Loading**: Reads task configurations from JSON files in `tasks_list/` (e.g., `--tasks_list_json_path_AD`, `--tasks_list_json_path_detect`, `--tasks_list_json_path_TL`). At least one must be provided; set multiple for multi-task training.
*   **Concatenation**: Combines datasets from multiple sources if specified.
*   **Splitting**: Splits the combined dataset into training and validation sets based on `val_sample_limit`.
*   **Limiting**: Applies sample limits (see Section 4) to balance the dataset or reduce size for debugging.

### 3.2 Formatting Logic
This is the most critical step where raw data is converted into the model's expected input format. The `prepare_dataset` function uses a `mapping_func` to process each sample. The CoT entry point (`train__SFT-CoT__qwen2_5_vl.py`) uses:

*   **Angle/Distance Task**: `_format_data_AngleDistanceTask_CoT`
*   **Detection Task**: `_format_data_DetectionTask_CoT`
*   **Tumor/Lesion Task**: `_format_data_TumorLesionTask_CoT`

Each builds the chat messages, fills the CoT template with intermediate ground truth, and wraps the final values in `<answer></answer>`. (Non-CoT variants without the `_CoT` suffix also exist, used by `train__SFT__qwen2_5_vl.py`.)

> [!CAUTION]
> **Physical Spacing Information**: VLMs need physical spacing info (pixel size) to perform measurement tasks (A/D and T/L estimation). `medvision_bm` handles model-specific image processing to ensure the pixel size in the prompt always matches the resolution the model actually perceives. When `--new_shape_hw <height> <width>` is set (e.g., `--new_shape_hw 512 512`), images are resized during dataset preparation and the pixel size is re-adjusted accordingly.

### 3.3 Caching Mechanism
Dataset processing for 121K samples is expensive, so the scripts cache aggressively:

*   `save_processed_img_to_disk=true`: saves processed images as PNG files during dataset processing, so training loads them directly instead of re-slicing 3D volumes.
*   `skip_process_dataset=true`: skips processing entirely and loads the prepared dataset from disk (used in Phase 2, or for re-runs).
*   `prepared_ds_dir`: where the prepared dataset lives; defaults to a path derived from the sample limits (e.g., `<data_dir>/tmp_prepared_ds_AD5500_D110000_TL5500_all121000`).


## 4. 🎯 Training Settings

Training configuration is controlled via variables in the shell scripts. Key parameters include:

### Hyperparameters
*   `epoch`: Number of training epochs (`10` for the LoRA scripts, `3` for full-parameter SFT).
*   `per_device_train_batch_size` / `per_device_eval_batch_size`: Batch size per GPU.
*   `gradient_accumulation_steps`: Steps to accumulate gradients (effective batch size = per-device batch × accumulation steps × number of GPUs).
*   `gradient_checkpointing`: Trades compute for memory; required for full-parameter SFT at 7B scale.
*   `use_flash_attention_2`: Enables Flash Attention 2 for faster training (requires compatible GPU).

### Checkpointing & Evaluation
*   `save_steps`: Frequency of saving checkpoints.
*   `eval_steps`: Frequency of evaluation on the validation set.
*   `logging_steps`: Frequency of logging metrics.
*   `save_total_limit`: Maximum number of kept checkpoints (older ones are pruned).
*   `resume_from_checkpoint`: If `true`, resumes from the last checkpoint of the same `run_name` — interrupted runs can simply be relaunched.

### Sample Limits
*   `train_sample_limit` / `val_sample_limit`: Global limits (required).
*   `train_sample_limit_per_task` / `val_sample_limit_per_task`: Approximately balanced sampling across the 3 tasks (Option 1).
*   `train_sample_limit_task_AD` / `..._Detection` / `..._TL` (and `val_...` counterparts): Task-specific limits (Option 2, the current setting: 5.5K / 110K / 5.5K).

If a limit exceeds the dataset size, the full dataset is used.

### Temperature-Based Multi-Task Sampling
With heavily imbalanced tasks (110K detection vs. 5.5K A/D and T/L), uniform sampling would let detection dominate every batch. Setting `enable_temperature_sampler=true` switches to a `TemperatureSamplerSFTTrainer` that re-weights task sampling probabilities by a temperature `temperature_sampler_T` (default `3`; the current scripts use `5`):

*   `T = 1`: sampling proportional to task counts (no re-weighting).
*   Larger `T`: flatter task probabilities, i.e., minority tasks (A/D, T/L) are oversampled.

This only affects training; it has no effect during dataset processing.

### LoRA vs. Full-Parameter SFT
The LoRA scripts train adapters on top of the frozen base model and run with plain DDP (`accelerate launch --num_processes=4 --mixed_precision=bf16`).

The full-parameter script (`train__fullSFT-CoT__...`) updates all weights, which does not fit in DDP on 80GB GPUs: weights (14GB) + gradients (14GB) + AdamW FP32 states (56GB) ≈ 84GB per GPU before activations. It therefore launches with **FSDP** (`--use_fsdp --fsdp_sharding_strategy FULL_SHARD`), sharding all three components across GPUs (~31GB per GPU on 4 GPUs).


## 5. 📊 WandB and Hugging Face Logging

To use **Weights & Biases (WandB)** and the **Hugging Face Hub**, you must first log in:

```bash
# Login to WandB
wandb login

# Login to Hugging Face
huggingface-cli login
```

### Weights & Biases (WandB)
*   `wandb_project`: Your project name.
*   `wandb_run_name`: Name for the current run.
*   `wandb_resume`: Set to "allow" to resume interrupted runs.
*   `wandb_run_id`: Unique ID for resuming specific runs.

### Hugging Face Hub (LoRA scripts)
*   `push_LoRA`: If `true`, pushes the LoRA adapter after each save.
*   `push_merged_model`: If `true`, pushes the merged (adapter + base) model to the Hub.
*   `merge_model`: If `true`, merges after training completes.
*   `merge_only`: If `true`, skips training and only performs the merge/push of the last checkpoint.

The full-parameter script saves complete model checkpoints directly, so it has no merge/push options.

## 6. 📚 References

*   [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1)
*   [TRL Documentation: SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
*   [MedGemma Fine-tuning Notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
