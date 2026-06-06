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

> [!NOTE]
> Detailed descriptions of these tasks can be found on the [MedVision Project Page](https://medvision-vlm.github.io).


## 2. 🚀 Quick Start

The recommended way to run training is using the provided shell scripts in `script/sft/`. These scripts handle environment setup, variable definition, and launching the training process (including distributed training).

All current scripts fine-tune **Qwen2.5-VL-7B** with chain-of-thought (CoT) supervision on the combined detection / A-D / T-L data:

*   `script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh`: **LoRA** SFT at the model's native (dynamic) resolution.
*   `script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh`: **LoRA** SFT with images reshaped to **512×512**.
*   `script/sft/train__fullSFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh`: **Full-parameter** SFT with images reshaped to **512×512**.

```bash
# Install the package
git clone https://github.com/YongchengYAO/MedVision.git MedVision
cd MedVision
pip install .
```

Simply execute the shell script from the project root:

```bash
# LoRA SFT, 512x512 (the MedVision-V0 recipe)
bash script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh
# or LoRA SFT at native resolution
bash script/sft/train__SFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k.sh
# or full-parameter SFT, 512x512
bash script/sft/train__fullSFT-CoT__Qwen2.5VL7B__D110k-AD5.5k-TL5.5k__512x512.sh
```

## 3. 💿 Data Preparation

The data preparation pipeline is handled by the `prepare_dataset` function in `src/medvision_bm/sft/sft_utils.py`. This orchestrates the loading, formatting, and cleaning of data for each task.

### 3.1 Loading and Splitting
*   **Loading**: Reads task configurations from JSON files (e.g., `tasks_list_json_path_AD`).
*   **Concatenation**: Combines datasets from multiple sources if specified.
*   **Splitting**: Splits the combined dataset into training and validation sets based on `limit_val_sample`.
*   **Limiting**: Applies sample limits (`limit_train_sample`, `limit_val_sample`) to balance the dataset or reduce size for debugging.

### 3.2 Formatting Logic
This is the most critical step where raw data is converted into the model's expected input format. The `prepare_dataset` function uses a `mapping_func` to process each sample:

*   **Angle/Distance Task** `_format_data_AngleDistanceTask`
*   **Detection Task** `_format_data_DetectionTask`
*   **Tumor/Lesion Task** `_format_data_TumorLesionTask`

> [!CAUTION]
> **Physical Spacing Information**: VLMs need physical spacing info (pixel size) to perform measurement tasks (A/D and T/L estimation). `medvision_bm` handles model-specific image processing (e.g., resizing for MedGemma, dynamic processing for Qwen2.5-VL) to ensure accurate spatial metadata in the formatting.

### 3.3 Caching Mechanism
With `save_processed_img_to_disk=true`, the processed dataset is saved to disk. It ensures that subsequent runs with the same settings load data instantly without re-processing.

## 4. 🎯 Training Settings

Training configuration is controlled via variables in the shell scripts. Key parameters include:

### Hyperparameters
*   `epoch`: Number of training epochs (default: `10`).
*   `learning_rate`: learning rate for the optimizer.
*   `per_device_train_batch_size`: Batch size per GPU.
*   `gradient_accumulation_steps`: Steps to accumulate gradients before updating weights.
*   `use_flash_attention_2`: Enables Flash Attention 2 for faster training (requires compatible GPU).

### Checkpointing & Evaluation
*   `save_steps`: Frequency of saving checkpoints.
*   `eval_steps`: Frequency of evaluation on the validation set.
*   `logging_steps`: Frequency of logging metrics.
*   `save_total_limit`: Maximum number of kept checkpoints (older ones are pruned).

### Sample Limits
Useful for debugging or balancing:
*   `train_sample_limit` / `val_sample_limit`: Global limits.
*   `train_sample_limit_task_*`: Task-specific limits (e.g., AD, Detection, TL).

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

### Hugging Face Hub
*   `push_LoRA`: If `true`, pushes the LoRA adapter after each save.
*   `push_merged_model`: If `true`, merges adapter + base model and pushes the full model.
*   `merge_only`: If `true`, skips training and only performs the merge/push.

## 6. 📚 References

*   [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1)
*   [TRL Documentation: SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
*   [MedGemma Fine-tuning Notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
