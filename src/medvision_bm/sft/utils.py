import gc
import importlib
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import psutil
import torch
from accelerate import PartialState
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          BitsAndBytesConfig)
from trl import SFTConfig, SFTTrainer

from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE, SEED


# Add small helpers
def is_main_process() -> bool:
    try:
        ps = PartialState()
        # Some versions/contexts may not expose the attribute; guard against that.
        if hasattr(ps, "is_main_process"):
            return bool(ps.is_main_process)
    except Exception:
        # If PartialState can't be instantiated or accessed, fall back to True.
        # This avoids importing torch.distributed (heavy) and keeps the check lightweight.
        pass
    return True


def safe_print(*args, force: bool = False, **kwargs):
    """Print only on main process unless force=True."""
    if force or is_main_process():
        print(*args, **kwargs)


def broadcast_int_from_main(value: int, src: int = 0):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        obj = [int(value) if dist.get_rank() == src else 0]
        dist.broadcast_object_list(obj, src=src)
        return int(obj[0])
    return int(value)


def get_cgroup_limited_cpus():
    # cgroup v1
    try:
        base = Path("/sys/fs/cgroup/cpu")
        q = base / "cpu.cfs_quota_us"
        p = base / "cpu.cfs_period_us"
        if q.exists() and p.exists():
            quota = int(q.read_text().strip())
            period = int(p.read_text().strip())
            if quota > 0 and period > 0:
                return math.floor(quota / period)
    except (ValueError, OSError):
        pass

    # cgroup v2
    try:
        line = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        quota, period = line.split()
        if quota != "max":
            return math.floor(int(quota) / int(period))
    except (ValueError, OSError):
        pass

    # fallback to host-wide CPU count
    return os.cpu_count()


def _load_nifti_2d(img_path, slice_dim, slice_idx):
    """Map function to load 2D slice from a 3D NIFTI images."""
    img_nib = nib.load(img_path)
    voxel_size = img_nib.header.get_zooms()
    image_3d = img_nib.get_fdata().astype("float32")
    if slice_dim == 0:
        image_2d = image_3d[slice_idx, :, :]
        pixel_size = voxel_size[1:3]
    elif slice_dim == 1:
        image_2d = image_3d[:, slice_idx, :]
        pixel_size = voxel_size[0:1] + voxel_size[2:3]
    elif slice_dim == 2:
        image_2d = image_3d[:, :, slice_idx]
        pixel_size = voxel_size[0:2]
    else:
        raise ValueError("slice_dim must be 0, 1 or 2")
    return (pixel_size, image_2d)


def _doc_to_visual(doc):
    """Convert document to image with scale bar added."""
    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)
    # Normalize the image to 0-255 range
    if img_2d.max() > img_2d.min():
        img_2d_normalized = (
            (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min()) * 255
        ).astype(np.uint8)
    else:
        img_2d_normalized = np.zeros_like(img_2d, dtype=np.uint8)
    # Convert to PIL Image
    pil_img = Image.fromarray(img_2d_normalized)
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")
    return [pil_img]


def _get_biometric_prompt_angle(biometrics_name, l1p1, l1p2, l2p1, l2p2, metric_unit):
    """Prepare prompt for angle estimate VQA. Inputs are names."""
    if biometrics_name is not None and biometrics_name != "":
        return (
            f"estimate the angle of {biometrics_name} in {metric_unit}, "
            f"which is the angle between 2 lines: "
            f"(line 1) the line connecting {l1p1} and {l1p2}, "
            f"(line 2) the line connecting {l2p1} and {l2p2}.\n"
        )
    else:
        return (
            f"estimate the angle between 2 lines in {metric_unit}: "
            f"(line 1) the line connecting {l1p1} and {l1p2}, "
            f"(line 2) the line connecting {l2p1} and {l2p2}.\n"
        )


def _get_biometric_prompt_distance(biometrics_name, p1, p2, metric_unit):
    """Prepare prompt for distance estimate VQA. Inputs are names."""
    metric_unit = metric_unit.strip().replace("mm", "millimeters")
    if biometrics_name is not None and biometrics_name != "":
        return (
            f"estimate the distance of {biometrics_name} in {metric_unit}, "
            f"which is the distance between 2 landmark points: "
            f"(landmark 1) {p1}, "
            f"(landmark 2) {p2}.\n"
        )
    else:
        return (
            f"estimate the distance between 2 landmark points in {metric_unit}: "
            f"(landmark 1) {p1}, "
            f"(landmark 2) {p2}.\n"
        )


def _doc_to_text_BiometricsFromLandmarks(doc, img_processor=None, reshape_size=None):
    """Convert document to text."""
    # Early assertions
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    format_prompt = "The answer should be a single decimal number without any units or additional text."

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(
            f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry_module = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_biometry"
    )

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry_module.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_type = biometric_profile["metric_type"]
    metric_map_name = biometric_profile["metric_map_name"]
    metric_key = biometric_profile["metric_key"]
    metric_unit = biometric_profile["metric_unit"]

    # Get 2D image info
    image_description = task_info["image_description"]

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)
    img_shape = img_2d_raw.shape

    # -------------
    # NOTE: If img_processor is provided, a model-specific processing is applied to get the reshaped image size
    # -------------
    if img_processor is not None:
        # Get reshaped image size so that we can adjust the pixel size dynamically
        img_PIL = Image.fromarray(img_2d_raw)
        processed_visual = img_processor([img_PIL])
        image_grid_thw = processed_visual["image_grid_thw"][0]
        patch_size = img_processor.patch_size
        img_shape_resized = (
            image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
    elif reshape_size is not None:
        assert len(reshape_size) == 2, "reshape_size should be of length 2"
        img_shape_resized = reshape_size
    # -------------

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape
    pixel_height, pixel_width = pixel_size_hw
    resize_ratio_h = img_shape_resized[0] / original_height
    resize_ratio_w = img_shape_resized[1] / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w
    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} mm (width) x {adjusted_pixel_height:.3f} mm (height)."

    # Question
    if metric_type == "distance":
        lines_map = task_info[metric_map_name]
        line_dict = lines_map[metric_key]
        lms_map_name = line_dict["element_map_name"]
        lms_map = task_info[lms_map_name]
        lms = line_dict[
            "element_keys"
        ]  # list of 2 strings -- names of points (landmarks)
        p1_name = lms_map[lms[0]]
        p2_name = lms_map[lms[1]]
        biometrics_name = line_dict["name"]
        task_prompt = _get_biometric_prompt_distance(
            biometrics_name, p1_name, p2_name, metric_unit
        )
    if metric_type == "angle":
        angles_map = task_info[metric_map_name]
        angle_dict = angles_map[metric_key]
        lines_map_name = angle_dict["element_map_name"]
        # list of 2 strings -- names of lines
        line_keys = angle_dict["element_keys"]
        lines_map = task_info[lines_map_name]
        line1_dict = lines_map[line_keys[0]]
        line1_lms = line1_dict[
            "element_keys"
        ]  # list of 2 strings -- names of points (landmarks)
        line1_lms_map_name = line1_dict["element_map_name"]
        line1_lms_map = task_info[line1_lms_map_name]
        line1_p1_name = line1_lms_map[line1_lms[0]]
        line1_p2_name = line1_lms_map[line1_lms[1]]
        line2_dict = lines_map[line_keys[1]]
        line2_lms = line2_dict[
            "element_keys"
        ]  # list of 2 strings -- names of points (landmarks)
        line2_lms_map_name = line2_dict["element_map_name"]
        line2_lms_map = task_info[line2_lms_map_name]
        line2_p1_name = line2_lms_map[line2_lms[0]]
        line2_p2_name = line2_lms_map[line2_lms[1]]
        biometrics_name = angle_dict["name"]
        task_prompt = _get_biometric_prompt_angle(
            biometrics_name,
            line1_p1_name,
            line1_p2_name,
            line2_p1_name,
            line2_p2_name,
            metric_unit,
        )

    question = (
        f"Task:\n"
        f"Given the input medical image: {image_description}, "
        f"{task_prompt}"
        f"Additional information:\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{format_prompt}"
    )
    return question


def _doc_to_target_BiometricsFromLandmarks(doc):
    """Get ground truth biometrics."""
    biometric_profile = doc["biometric_profile"]
    return biometric_profile["metric_value"]


# NOTE: This is specific to the BiometricVQA dataset
def _format_data_distance_angle(
    example: dict[str, Any], img_processor=None, reshape_size=None
) -> dict[str, Any]:
    # Early assertions
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    target_str = str(_doc_to_target_BiometricsFromLandmarks(example))
    prompt = _doc_to_text_BiometricsFromLandmarks(
        example, img_processor, reshape_size)

    example["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": target_str,
                },
            ],
        },
    ]

    return example


def _load_single_dataset(task, tag_ds):
    """Load a single dataset configuration with improved error handling."""
    try:
        print(f"Loading dataset for task: {task}")
        config = task + "_Train"

        # Add timeout and retry logic for dataset loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    "YongchengYAO/MedVision",
                    name=config,
                    trust_remote_code=True,
                    split="train",
                    streaming=False,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    print(
                        f"Attempt {attempt + 1} failed for {task}, retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        # Add dataset name column
        # Extract dataset name (part before "_BiometricsFromLandmarks")
        dataset_name = task.split(f"_{tag_ds}")[0]
        ds = ds.add_column("dataset_name", [dataset_name] * len(ds))

        print(
            f"Successfully loaded {len(ds)} samples from config {config} (dataset: {dataset_name})"
        )
        return ds

    except Exception as e:
        print(f"Failed to load dataset for task {task}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Task {task} failed: {str(e)}")


def prepare_dataset_angle_distance(
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    num_workers_concat_datasets=4,
    num_workers_format_dataset=32,
    model_hf=None,
    tag_ds=None,
    img_processor=None,
    reshape_size=None,
):
    # Early assertions
    assert tag_ds is not None, "\n [Error] tag_ds (i.e., the string in tasks names: <dataset_name>_<tag_ds>) must be provided."
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    print(f"Starting dataset preparation from {tasks_list_json_path}")
    print(
        f"Memory usage before loading: {psutil.virtual_memory().percent}%")

    assert model_hf is not None, "Model HF ID must be provided for dataset preparation."

    # Load tasks list from JSON file
    with open(tasks_list_json_path, "r") as f:
        tasks_dict = json.load(f)
    tasks = list(tasks_dict.keys())

    print(f"Found {len(tasks)} tasks to process")

    # Reduce parallelism to avoid memory issues - use fewer workers
    available_cpus = get_cgroup_limited_cpus()
    concat_workers = min(num_workers_concat_datasets,
                         available_cpus, len(tasks))
    print(
        f"Using {concat_workers} workers for dataset loading (available CPUs: {available_cpus})"
    )

    datasets_list = []
    failed_tasks = []

    # Process datasets with controlled parallelism
    with ProcessPoolExecutor(max_workers=concat_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_load_single_dataset, task, tag_ds): task for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                ds = future.result(timeout=120)  # 5 minute timeout per task
                datasets_list.append(ds)
                print(
                    f"✓ Completed {task} ({len(datasets_list)}/{len(tasks)})")

                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    print(f"⚠️  High memory usage: {memory_percent}%")

            except Exception as exc:
                error_msg = f"Task {task} generated an exception: {exc}"
                print(error_msg)
                failed_tasks.append((task, str(exc)))
                # Continue with other tasks instead of failing completely

    # Report results
    if failed_tasks:
        print(f"❌ Failed to load {len(failed_tasks)} tasks:")
        for task, error in failed_tasks:
            print(f"  - {task}: {error}")

        raise RuntimeError(
            "❌ ERROR: Some tasks failed to load. Check the logs above for details."
        )

    # Combine all datasets
    print("Combining datasets...")
    combined_dataset = concatenate_datasets(datasets_list)
    print(f"Combined dataset has {len(combined_dataset)} total samples")

    # Clear intermediate datasets to free memory
    del datasets_list
    gc.collect()

    # Split the dataset into training and validation sets
    dataset = combined_dataset.train_test_split(
        train_size=len(combined_dataset) - limit_val_sample,
        test_size=limit_val_sample,
        shuffle=True,
        seed=SEED,
    )
    dataset["validation"] = dataset.pop("test")

    # Limit the number of training and validation samples if specified
    if limit_train_sample and limit_train_sample < len(dataset["train"]):
        print(
            f"Limiting training samples to {limit_train_sample} (original: {len(dataset['train'])})"
        )
        dataset["train"] = (
            dataset["train"].shuffle(seed=SEED).select(
                range(limit_train_sample))
        )

    # Format the dataset with parallelism
    # Use conservative parallelism for formatting to avoid OOM
    format_workers = min(num_workers_format_dataset, available_cpus)
    print(f"Formatting dataset with {format_workers} workers...")
    if img_processor is not None:
        dataset = dataset.map(
            _format_data_distance_angle,
            fn_kwargs={"img_processor": img_processor},
            num_proc=format_workers,
            desc="Formatting dataset",
        )
    elif reshape_size is not None:
        dataset = dataset.map(
            _format_data_distance_angle,
            fn_kwargs={"reshape_size": reshape_size},
            num_proc=format_workers,
            desc="Formatting dataset",
        )
    print(f"Dataset length after formatting: {len(dataset)}")

    return dataset

# TODO: debug
# FIXME: This is wrong in multi-GPU setting. The len(train_dl) may not reflect the actual number of samples.
# --- NEW Training BLOCK: handle resume logic & recompute max_steps if user changed epochs ---


def recompute_total_max_steps(trainer, *, gradient_accumulation_steps=None, num_train_epochs=None):
    """Recompute total planned update steps based on global dataset size, world size and desired epochs."""
    args = trainer.args
    state = PartialState()

    # Prefer accelerate's world size; fallback to Trainer args/env
    world_size = getattr(state, "num_processes", None) or getattr(
        args, "world_size", None)
    if not world_size or world_size < 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    per_device_bsz = getattr(args, "per_device_train_batch_size", 1)
    grad_accum = gradient_accumulation_steps or getattr(
        args, "gradient_accumulation_steps", 1)
    epochs = int(num_train_epochs or getattr(
        args, "num_train_epochs", 1))

    new_max_steps = 0
    dataset_n = None
    if is_main_process():
        # Prefer sized dataset to avoid per-process dataloader length in DDP.
        try:
            dataset_n = len(trainer.train_dataset)  # global length
            if dataset_n is None:
                raise TypeError
            effective_bsz = max(
                1, per_device_bsz * world_size * grad_accum)
            if getattr(args, "dataloader_drop_last", False):
                steps_per_epoch = max(1, dataset_n // effective_bsz)
            else:
                steps_per_epoch = max(
                    1, math.ceil(dataset_n / effective_bsz))
        except Exception:
            # Fallback if dataset is unsized (e.g., IterableDataset)
            train_dl = trainer.get_train_dataloader()
            steps_per_epoch = max(
                1, math.ceil(len(train_dl) / grad_accum))

        new_max_steps = steps_per_epoch * epochs

        # Main-process-only logs
        print(f"[resume] world_size: {world_size}")
        print(f"[resume] dataset size (global): {dataset_n}")
        print(
            f"[resume] per_device_train_batch_size: {per_device_bsz}")
        print(f"[resume] gradient_accumulation_steps: {grad_accum}")
        print(f"[resume] num_train_epochs: {epochs}")
        print(
            f"[resume] steps_per_epoch (computed): {steps_per_epoch}")
        print(
            f"[resume] Recomputed new_max_steps (epochs based): {new_max_steps}")

    # Share the computed value to all processes
    new_max_steps = broadcast_int_from_main(new_max_steps)
    return new_max_steps


# Safer alternative: build a collate_fn bound to a specific processor (avoids relying on a global in multi-process contexts).
def make_collate_fn(proc):
    def _collate_fn_local(examples: list[dict[str, Any]]):
        texts = []
        images = []
        for example in examples:
            if "processed_images" in example:
                images.append(example["processed_images"])
            else:
                pil_images = _doc_to_visual(example)
                images.append(pil_images)
            texts.append(
                proc.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
            )

        batch = proc(text=texts, images=images,
                     return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
        image_begin_token_id = [
            proc.tokenizer.convert_tokens_to_ids("<|im_start|>")]
        image_end_token_id = [
            proc.tokenizer.convert_tokens_to_ids("<|im_end|>")]

        labels[labels == proc.tokenizer.pad_token_id] = -100
        labels[labels == image_begin_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == image_end_token_id] = -100

        batch["labels"] = labels
        return batch
    return _collate_fn_local


def prepare_trainer(
    run_name,
    base_model_hf,
    lora_checkpoint_dir,
    data,
    per_device_train_batch_size=14,
    per_device_eval_batch_size=14,
    gradient_accumulation_steps=6,
    use_flash_attention_2=True,
    num_train_epochs=1,
    save_steps=100,
    eval_steps=50,
    logging_steps=50,
    save_total_limit=10,
    dataloader_num_workers=8,
    gradient_checkpointing=False,
    dataloader_pin_memory=True,
    push_LoRA=False,
):
    # Check if GPU supports bfloat16
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError(
            "GPU does not support bfloat16, please use a GPU that supports bfloat16."
        )

    # Set the device string for multi-gpu training using accelerate's PartialState
    # ref: https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md#multi-gpu-training
    if use_flash_attention_2:
        model_kwargs = dict(
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map={"": PartialState().process_index},
            trust_remote_code=True,
        )
    else:
        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map={"": PartialState().process_index},
            trust_remote_code=True,
        )
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    # Load the model with the specified configuration
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_hf, **model_kwargs)

    # Initialize processor
    processor = AutoProcessor.from_pretrained(base_model_hf)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    # PEFT configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    learning_rate = 2e-4

    args = SFTConfig(
        run_name=run_name,
        output_dir=lora_checkpoint_dir,
        num_train_epochs=num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Enable gradient checkpointing to reduce memory usage
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch_fused",  # Use fused AdamW optimizer for better performance
        logging_steps=logging_steps,  # Number of steps between logs
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,  # Maximum number of checkpoints to save
        eval_strategy="steps",  # Evaluate every `eval_steps`
        eval_steps=eval_steps,  # Number of steps between evaluations
        learning_rate=learning_rate,  # Learning rate based on QLoRA paper
        bf16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # Warmup ratio based on QLoRA paper
        lr_scheduler_type="linear",  # Use linear learning rate scheduler
        push_to_hub=push_LoRA,  # Push model to Hub
        hub_private_repo=True,  # Push to a private repository
        report_to="wandb",  # Report metrics to tensorboard
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Set gradient checkpointing to non-reentrant to avoid issues
        dataset_kwargs={
            "skip_prepare_dataset": True
        },  # Skip default dataset preparation to preprocess manually
        # Columns are unused for training but needed for data collator
        remove_unused_columns=False,
        label_names=[
            "labels"
        ],  # Input keys that correspond to the labels. This is defined by batch["labels"] in _collate_fn_local()
        dataloader_num_workers=dataloader_num_workers,
        # Pin memory for faster GPU transfer
        dataloader_pin_memory=dataloader_pin_memory,
        # Disable persistent workers to avoid OOM issues
        dataloader_persistent_workers=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_config,
        processing_class=processor,
        data_collator=make_collate_fn(processor),
    )
    return trainer


def merge_models(
    base_model_hf,
    lora_checkpoint_dir,
    merged_model_hf=None,
    merged_model_dir=None,
    push_to_hub=False,
):
    """
    Merge LoRA adapter with base model and optionally save locally and/or push to Hugging Face Hub.

    Args:
        base_model_hf (str): Base model ID or path (e.g., "google/medgemma-4b-it")
        lora_checkpoint_dir (str): Path or HF repository ID of the trained LoRA adapter
        merged_model_hf (str, optional): Hugging Face model repository ID for pushing.
                                       Required when push_to_hub=True. Defaults to None.
        merged_model_dir (str): Local directory to save merged model. Defaults to "merged_model".
        push_to_hub (bool): Whether to push to Hugging Face Hub. Defaults to False.

    Raises:
        ValueError: If merged_model_hf is None when push_to_hub is True.
    """
    # Ensure only the main process runs the merge and synchronize with others.
    from accelerate import Accelerator
    acc = Accelerator()
    if not acc.is_main_process:
        acc.wait_for_everyone()
        return

    # Aggressive memory cleanup before loading models
    torch.cuda.empty_cache()
    gc.collect()

    # Print initial memory state
    print("Starting model merge process...")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(
                f"GPU {i} before merge - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
            )

    # Load base model with aggressive memory optimization
    print("Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_hf, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load LoRA adapter and merge with base model
    peft_model = PeftModel.from_pretrained(model, lora_checkpoint_dir)
    merged_model = peft_model.merge_and_unload()

    # Clear intermediate references immediately
    del model
    del peft_model
    torch.cuda.empty_cache()
    gc.collect()

    # Load processor from the adapter (includes any training-specific configurations)
    processor = AutoProcessor.from_pretrained(lora_checkpoint_dir)

    # Save merged model locally if requested
    if merged_model_dir is not None:
        print(f"Saving merged model to: {merged_model_dir}")
        merged_model.save_pretrained(
            merged_model_dir, safe_serialization=True, max_shard_size="2GB"
        )
        processor.save_pretrained(merged_model_dir)
        print(f"Merged model saved to: {merged_model_dir}")

    # Push to Hugging Face Hub if requested
    if push_to_hub:
        if merged_model_hf is None:
            raise ValueError(
                "merged_model_hf must be specified when push_to_hub is True."
            )
        else:
            print(
                f"Pushing merged model to Hugging Face Hub: {merged_model_hf}")
            merged_model.push_to_hub(
                merged_model_hf, private=True, max_shard_size="2GB"
            )
            processor.push_to_hub(merged_model_hf, private=True)
            print(
                f"Successfully pushed merged model to: {merged_model_hf}")

    # Clean up merged model references
    del merged_model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    print("Model merge completed with memory cleanup.")

    # Synchronize so other processes can proceed
    acc.wait_for_everyone()


def cleanup_all_gpu():
    # Step 1: delete or unset GPU-resident objects
    # del model, optimizer, tensors...

    # Step 2: collect Python garbage
    gc.collect()

    # Step 3: empty cache for each CUDA device
    for d in range(torch.cuda.device_count()):
        with torch.cuda.device(d):
            torch.cuda.empty_cache()

    # Optional: get memory summary
    for d in range(torch.cuda.device_count()):
        print(
            f"Device {d}: allocated={torch.cuda.memory_allocated(d)/1024**2:.1f}MB, "
            f"reserved={torch.cuda.memory_reserved(d)/1024**2:.1f}MB"
        )


def train_resume_from_checkpoint(trainer, last_checkpoint, gradient_accumulation_steps, num_train_epochs):
    safe_print("[resume] Requested resume_from_checkpoint=True")

    from transformers.trainer_utils import get_last_checkpoint

    assert last_checkpoint is not None, f"No checkpoint found in {last_checkpoint}"
    safe_print(f"[resume] Found checkpoint: {last_checkpoint}")

    new_max_steps = recompute_total_max_steps(
        trainer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
    )

    # --- load previous trainer_state.json directly (avoid non-existent _load_state) ---
    trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")
    try:
        with open(trainer_state_path, "r", encoding="utf-8") as f:
            _prev_state = json.load(f)
        prev_global = _prev_state.get("global_step")
        prev_recorded_max = _prev_state.get("max_steps")
        safe_print(
            f"[resume] Loaded previous trainer_state.json: global_step={prev_global}, max_steps={prev_recorded_max}"
        )
    except Exception as e:
        raise RuntimeError(
            f"[resume] Failed to read trainer_state.json ({e}); cannot resume training.")
    # -------------------------------------------------------------------------------

    if new_max_steps <= prev_recorded_max:
        if prev_global >= new_max_steps:
            safe_print(
                "[resume] Training already satisfies (or exceeds) the new reduced horizon."
                " Nothing further to do. If you intended more training, increase num_train_epochs."
            )
            trainer.state.is_finished = True
        else:
            safe_print(
                "[resume] Horizon reduced (or unchanged) and progress not past new_max_steps; continuing."
            )
            trainer.args.max_steps = new_max_steps
            trainer.state.max_steps = new_max_steps
    else:
        safe_print("[resume] Extending training horizon.")
        trainer.args.max_steps = new_max_steps
        trainer.state.max_steps = new_max_steps
        trainer.state.is_finished = False

    safe_print("Resuming training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
