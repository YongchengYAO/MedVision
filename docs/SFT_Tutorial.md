# Supervised Fine-Tuning (SFT) for VLMs on Medical Image Data

This tutorial guides you through the process of Supervised Fine-Tuning (SFT) for Vision-Language Models (VLMs) on medical image data, using the `medvision_bm` codebase.

☀️ Useful resources:
- Code: ['medvision_bm'](https://github.com/YongchengYAO/MedVision), the codebase for benchmarking and fine-tuning VLMs on medical image data.
- Dataset: [YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision), a dataset for quantitative medical image analysis with 30.8M annotated samples.
- Project: [MedVision](https://medvision-vlm.github.io)

## 1. Introduction to SFT

Supervised Fine-Tuning (SFT) is a crucial step in adapting Large Language Models (LLMs) and Vision-Language Models (VLMs) to specific tasks or domains. It involves training the model on a dataset of instruction-response pairs (or in our case, image-instruction-response triplets) to learn how to follow instructions and generate desired outputs.

For a deeper dive into the concepts of SFT, we recommend the [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1).

In the context of `medvision_bm`, we use SFT to teach models like **MedGemma** and **Qwen2.5-VL** to perform specific medical tasks such as:
- **Angle/Distance (A/D)**: Estimating angles/distances from landmarks.
- **Detection**: Identifying bounding boxes for anatomical structures.
- **Tumor/Lesion Size (T/L)**: Estimating the major and minor axes of tumors/lesions.

🌟 Details of these tasks can be found in [MedVision](https://medvision-vlm.github.io).

## 2. Environment Setup

To set up the environment, you need to install the `medvision_bm` package and run the setup script.

```bash
# 1. Install the package (run from the project root)
pip install .

# 2. Setup training environment (downloads necessary resources)
# Replace ./Data with your actual data directory
python -m medvision_bm.sft.env_setup --data_dir ./Data
```

## 3. Quick Start

The recommended way to run training is using the provided shell scripts in `script/sft/`. These scripts handle environment setup, variable definition, and launching the training process (including distributed training).

- `script/sft/train__SFT__MedGemma.sh`: For MedGemma models.
- `script/sft/train__SFT__Qwen2.5-VL.sh`: For Qwen2.5-VL models.

To run the training, simply execute the shell script from the project root:

```bash
bash script/sft/train__SFT__MedGemma.sh
# or
bash script/sft/train__SFT__Qwen2.5-VL.sh
```

## 4. Data Preparation

The data preparation pipeline is handled by the `prepare_dataset` function in `src/medvision_bm/sft/sft_utils.py`. This function orchestrates the loading, formatting, and cleaning of data for each task.

### 4.1 Loading and Splitting
-   **Loading**: Reads task configurations from JSON files (e.g., `tasks_list_json_path_AD`).
-   **Concatenation**: Combines datasets from multiple sources if specified.
-   **Splitting**: Splits the combined dataset into training and validation sets based on `limit_val_sample`.
-   **Limiting**: Applies sample limits (`limit_train_sample`, `limit_val_sample`) to balance the dataset or reduce size for debugging.

### 4.2 Formatting Logic
This is the most critical step where raw data is converted into the model's expected input format. The `prepare_dataset` function takes a `mapping_func` argument, which defines how each sample is processed.

*   **Angle/Distance Task (`_format_data_AngleDistanceTask`)**:
    *   **Input**: Raw data with landmarks.
    *   **Process**: Calculates angles or distances between specified points.
    *   **Output**: Generates a text prompt asking for the measurement and provides the calculated value as the target.
*   **Detection Task (`_format_data_DetectionTask`)**:
    *   **Input**: Raw data with bounding box annotations.
    *   **Process**: Normalizes bounding box coordinates to [0, 1].
    *   **Output**: Generates a prompt asking for the bounding box of a specific structure and provides the normalized coordinates `[x1, y1, x2, y2]` as the target.
*   **Tumor/Lesion Task (`_format_data_TumorLesionTask`)**:
    *   **Input**: Raw data with tumor/lesion measurements.
    *   **Process**: Extracts major and minor axis lengths.
    *   **Output**: Generates a prompt asking for the dimensions and provides the values as the target.

⚠️ VLMs need to acquire the physical spacing information (i.e., pixel size) to perform measurement tasks (A/D and T/L estimation). This information is usually stored in the image metadata or can be provided through prompts. The `medvision_bm` codebase handles the model-specific image processing logic (e.g., for MedGemma, it resizes images to fixed dimensions; for Qwen2.5-VL, it processes images dynamically) to ensure accurate physical spacing information in the formatted prompts.

### 4.3 Caching Mechanism
With `save_processed_img_to_disk=true`, the processed dataset is saved to disk in a directory that uniquely encodes the configuration (model ID, sample limits). This ensures that subsequent runs with the same settings can load the data instantly without re-processing.

## 5. Training Settings

The training configuration is controlled by variables in the shell scripts. Key parameters include:

### Hyperparameters
-   `epoch`: Number of training epochs (default: 10).
-   `learning_rate`: Learning rate for the optimizer.
-   `per_device_train_batch_size`: Batch size per GPU.
-   `gradient_accumulation_steps`: Number of steps to accumulate gradients before updating weights.
    -   *Effective Batch Size* = `per_device_train_batch_size` * `gradient_accumulation_steps` * `num_gpus`.
-   `use_flash_attention_2`: Enables Flash Attention 2 for faster training (requires compatible GPU).

### Checkpointing & Evaluation
-   `save_steps`: Frequency of saving checkpoints (in steps).
-   `eval_steps`: Frequency of running evaluation on the validation set.
-   `logging_steps`: Frequency of logging metrics to WandB/console.
-   `save_total_limit`: Maximum number of checkpoints to keep (older ones are deleted).

### Sample Limits
To balance the dataset or for debugging purposes, you can limit the number of samples:
-   `train_sample_limit` / `val_sample_limit`: Total limit for the combined dataset.
-   `train_sample_limit_task_*`: Limits for specific tasks (AD, Detection, TL).

## 6. WandB and Hugging Face Logging

To use Weights & Biases (WandB) for logging and Hugging Face Hub for model pushing, you must first log in to these services.

```bash
# Login to WandB
wandb login

# Login to Hugging Face
huggingface-cli login
```

### Weights & Biases (WandB)
The scripts are configured to log training metrics to WandB.
-   `wandb_project`: The project name in WandB.
-   `wandb_run_name`: The name of the current run.
-   `wandb_resume`: Set to "allow" to resume logging for interrupted runs.
-   `wandb_run_id`: Unique ID in the current `wandb_project` for the run (useful for resuming).

### Hugging Face Hub
You can push models and adapters to the Hugging Face Hub:
-   `push_LoRA`: If `true`, pushes the LoRA adapter to the Hub after each save.
-   `push_merged_model`: If `true`, merges the adapter with the base model and pushes the full model to the Hub after training.
-   `merge_only`: If `true`, skips training and only performs the merge and push operation.

## 7. References

- [Hugging Face SFT Tutorial](https://huggingface.co/learn/llm-course/en/chapter11/1)
- [TRL Documentation: SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [MedGemma Fine-tuning Notebook](https://github.com/Google-Health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
