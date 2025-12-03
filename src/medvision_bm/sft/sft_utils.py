import argparse
import gc
import gzip
import importlib
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import psutil
import torch
from accelerate import PartialState
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from medvision_bm.utils import str2bool
from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE, SEED


def is_main_process():
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


def safe_print(*args, force=False, **kwargs):
    """Print only on main process unless force=True."""
    if force or is_main_process():
        print(*args, **kwargs)


def broadcast_int_from_main(value, src=0):
    import torch.distributed as dist
    """
    Broadcast an integer value from the source (main) process to all other processes.

    Why we need this:
    - In distributed training (DDP / multi-process setups) only one process (commonly the
      main process) should perform certain global computations (e.g., computing the total
      number of training steps based on the global dataset size).
    - Other processes may have only a local view (sharded dataset/dataloader) and would
      compute different step counts if they tried independently. Broadcasting ensures every
      process receives the exact same integer so training logic stays consistent across
      processes (same max_steps, scheduling, checkpointing decisions, etc.).
    - Without this synchronization, processes could diverge: some may stop earlier/later,
      produce inconsistent checkpoint/state, or deadlock during collective operations.
    """
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


def _get_prompt_angle(biometrics_name, l1p1, l1p2, l2p1, l2p2, metric_unit):
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


def _get_prompt_distance(biometrics_name, p1, p2, metric_unit):
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


def _doc_to_text_AngleDistanceTask(doc, img_processor=None, reshape_size=None):
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
        # ====== Qwen2.5VL specific processing ======
        # FIXME: This block only works for Qwen2.5VL
        # TODO: Generalize to other models; reuse code if possible
        # ---
        # Get reshaped image size so that we can adjust the pixel size dynamically
        img_PIL = Image.fromarray(img_2d_raw)
        processed_visual = img_processor([img_PIL])
        image_grid_thw = processed_visual["image_grid_thw"][0]
        patch_size = img_processor.patch_size
        img_shape_resized = (
            image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
        # ===== End of Qwen2.5VL specific processing ======
    elif reshape_size is not None:
        # NOTE: For all models that have a fixed reshape size
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
        task_prompt = _get_prompt_distance(
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
        task_prompt = _get_prompt_angle(
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


def _doc_to_target_AngleDistanceTask(doc):
    """Get ground truth biometrics."""
    biometric_profile = doc["biometric_profile"]
    return biometric_profile["metric_value"]


# NOTE: This is specific to the MedVision dataset
def _format_data_AngleDistanceTask(
    example, img_processor=None, reshape_size=None, process_img=False, save_processed_img_to_disk=False,
):
    # Early assertions
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    target_str = str(_doc_to_target_AngleDistanceTask(example))
    prompt = _doc_to_text_AngleDistanceTask(
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

    # [Not recommended] Save processed images to dataset, making the cached dataset very large
    if process_img:
        example["processed_images"] = _doc_to_visual(example)

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        # Process image: read from nii.gz file and extract 2D slice
        pil_img = _doc_to_visual(example)[0]

        # Save tmp PNGs next to the source image inside a tmp_prepared_png folder
        img_path = example["image_file"]
        slice_dim = example["slice_dim"]
        slice_idx = example["slice_idx"]
        png_basename = Path(img_path).name.split(".", 1)[0]
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}.png"
        png_dir = os.path.join(os.path.dirname(img_path), "tmp_prepared_png")
        png_path = os.path.join(png_dir, png_filename)
        os.makedirs(png_dir, exist_ok=True)
        pil_img.save(png_path)
        example["image_file_png"] = [png_path]

    return example


def _format_data_AngleDistanceTask_CoT():
    raise NotImplementedError(
        "CoT formatting for AngleDistanceTask is not implemented yet. "
        "Please use the non-CoT version for now."
    )


def _doc_to_text_TumorLesionTask(doc, img_processor=None, reshape_size=None):
    """Convert document to text."""
    # Early assertions
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    format_prompt_tl = (
        "The answer should be two decimal numbers separated by a comma without any units or additional text. "
        "The first is the major axis length, and the second is the minor axis length."
    )

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(
            f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_biometry"
    )

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get label info
    label = str(doc["label"])
    labels_map = task_info["labels_map"]
    if label not in labels_map:
        raise ValueError(f"Label {label} not found in labels_map.")
    else:
        label_name = labels_map.get(label)

    # Get 2D image info
    image_description = task_info["image_description"]

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)
    img_shape = img_2d_raw.shape

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_unit = biometric_profile["metric_unit"]
    if isinstance(metric_unit, list):
        assert len(
            metric_unit) == 1, "metric_unit list should have only one element."
        metric_unit = metric_unit[0]
    elif isinstance(metric_unit, str):
        if metric_unit == "mm":
            metric_unit = "millimeters"
        elif metric_unit == "cm":
            metric_unit = "centimeters"
    else:
        raise ValueError(f"Unsupported metric_unit type: {type(metric_unit)}")

    # -------------
    # FIXME: This implementation only works for Qwen2.5VL
    # TODO: Generalize to other models, reuse code if possible
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
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image: {image_description}, "
        f"estimate the major and minor axis lengths of the ellipse enclosing the {label_name}, in {metric_unit}.\n"
        f"Additional information:\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{format_prompt_tl}"
    )
    return question, label_name


def _load_json(path: str):
    """
    Load a landmark file from .json or .json.gz format.
    Returns the parsed JSON object (usually dict or list).
    """
    path = Path(path)

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:  # assume plain .json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data


def _extract_3dCoor_to_2dCoor(coor_3d, slice_dim):
    if slice_dim == 0:
        return coor_3d[1:3]
    elif slice_dim == 1:
        return [coor_3d[0], coor_3d[2]]
    elif slice_dim == 2:
        return coor_3d[0:2]
    else:
        raise ValueError("slice_dim must be 0, 1, or 2")


def _get_TL_landmarks_coords(example):
    # Used in reasoning process reward
    landmark_data = _load_json(example["landmark_file"])
    slice_dim = example["slice_dim"]
    if slice_dim == 0:
        lm_key = "slice_landmarks_x"
    elif slice_dim == 1:
        lm_key = "slice_landmarks_y"
    elif slice_dim == 2:
        lm_key = "slice_landmarks_z"
    slice_idx = example["slice_idx"]
    lm_slice_ls = landmark_data[lm_key]

    matched_entry = next(
        (itm for itm in lm_slice_ls if itm.get("slice_idx") == slice_idx), None
    )
    if matched_entry is not None:
        lm_slice = matched_entry
    else:
        raise ValueError(
            f"No landmark entry found for slice_dim: {slice_dim} and slice_idx: {slice_idx}"
        )

    landmark_coords = {}
    for p_name in ("P1", "P2", "P3", "P4"):
        coor_2d = _extract_3dCoor_to_2dCoor(
            lm_slice["landmarks"][0][p_name], slice_dim)
        key = f"landmark_{p_name}"
        landmark_coords[key] = coor_2d
    return landmark_coords


def _doc_to_text_TumorLesionTask_CoT(doc, img_processor=None, reshape_size=None):
    """Convert document to text."""
    # Early assertions
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    from medvision_bm.sft.sft_prompts import (
        COT_INSTRUCT_TL_NORM,
        FORMAT_PROMPT_TL_REASONING,
    )

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(
            f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_biometry"
    )

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get label info
    label = str(doc["label"])
    labels_map = task_info["labels_map"]
    if label not in labels_map:
        raise ValueError(f"Label {label} not found in labels_map.")
    else:
        label_name = labels_map.get(label)

    # Get 2D image info
    image_description = task_info["image_description"]

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)
    img_shape = img_2d_raw.shape

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_unit = biometric_profile["metric_unit"]
    if isinstance(metric_unit, list):
        assert len(
            metric_unit) == 1, "metric_unit list should have only one element."
        metric_unit = metric_unit[0]
    elif isinstance(metric_unit, str):
        if metric_unit == "mm":
            metric_unit = "millimeters"
        elif metric_unit == "cm":
            metric_unit = "centimeters"
    else:
        raise ValueError(f"Unsupported metric_unit type: {type(metric_unit)}")

    # -------------
    # FIXME: This implementation only works for Qwen2.5VL
    # TODO: Generalize to other models, reuse code if possible
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
    resized_img_h, resized_img_w = img_shape_resized
    resize_ratio_h = resized_img_h / original_height
    resize_ratio_w = resized_img_w / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image: {image_description}, "
        f"estimate the major and minor axis lengths of the ellipse enclosing the {label_name}, in {metric_unit}.\n"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_TL_REASONING}\n"
        f"Reasoning steps:\n"
        f"{COT_INSTRUCT_TL_NORM}\n"
        f"Follow the reasoning steps to get the final answer in the required format."
    )

    # Gather values to fill in the CoT template
    # NOTE: The keys must be in the COT_TEMPLATE_TL_NORM from medvision_bm.sft.sft_prompts
    landmarks_coords = _get_TL_landmarks_coords(doc)
    # Caveat: 
    # 1. x is the width direction, y is the height direction
    # 2. use relative coordinates
    # 3. recalculate the major and minor axis lengths based on adjusted pixel size and resized image size; marginal error may exist compared to the original values due to rounding errors
    x1_major = landmarks_coords["landmark_P1"][1]/original_width
    y1_major = landmarks_coords["landmark_P1"][0]/original_height
    x2_major = landmarks_coords["landmark_P2"][1]/original_width
    y2_major = landmarks_coords["landmark_P2"][0]/original_height
    x1_minor = landmarks_coords["landmark_P3"][1]/original_width
    y1_minor = landmarks_coords["landmark_P3"][0]/original_height
    x2_minor = landmarks_coords["landmark_P4"][1]/original_width
    y2_minor = landmarks_coords["landmark_P4"][0]/original_height
    major_axis_length = math.sqrt(
        ((x2_major - x1_major) * resized_img_w * adjusted_pixel_width) ** 2
        + ((y2_major - y1_major) * resized_img_h * adjusted_pixel_height) ** 2
    )
    minor_axis_length = math.sqrt(
        ((x2_minor - x1_minor) * resized_img_w * adjusted_pixel_width) ** 2
        + ((y2_minor - y1_minor) * resized_img_h * adjusted_pixel_height) ** 2
    )
    values_dict = {
        "<label>": label_name,
        "<image_description>": image_description,
        "<image_width>": f"{resized_img_w}",
        "<image_height>": f"{resized_img_h}",
        "<pixel_width>": f"{adjusted_pixel_width:.3f}",
        "<pixel_height>": f"{adjusted_pixel_height:.3f}",
        "<metric_unit>": metric_unit,
        "<x1_major>": f'{x1_major:.3f}',
        "<y1_major>": f'{y1_major:.3f}',
        "<x2_major>": f'{x2_major:.3f}',
        "<y2_major>": f'{y2_major:.3f}',
        "<x1_minor>": f'{x1_minor:.3f}',
        "<y1_minor>": f'{y1_minor:.3f}',
        "<x2_minor>": f'{x2_minor:.3f}',
        "<y2_minor>": f'{y2_minor:.3f}',
        "<major_axis_length>": f'{major_axis_length:.3f}',
        "<minor_axis_length>": f'{minor_axis_length:.3f}',
    }
    return question, values_dict


def _doc_to_target_TumorLesionTask(doc):
    """Get ground truth biometrics."""
    biometric_profile = doc["biometric_profile"]
    return [
        biometric_profile["metric_value_major_axis"][0],
        biometric_profile["metric_value_minor_axis"][0],
    ]


def _doc_to_target_TumorLesionTask_CoT(values_dict):
    """Get ground truth biometrics."""
    from medvision_bm.sft.sft_prompts import COT_TEMPLATE_TL_NORM, fill_in_template

    # Prepare values to fill in the CoT template
    target_outputs_cot = fill_in_template(COT_TEMPLATE_TL_NORM, values_dict)

    return target_outputs_cot


# NOTE: This is dataset-specific formatting function
def _format_data_TumorLesionTask(
    example, img_processor=None, reshape_size=None, process_img=False, save_processed_img_to_disk=False,
):
    target = _doc_to_target_TumorLesionTask(example)
    target_str = ", ".join([f"{value:.3f}" for value in target])
    prompt, _ = _doc_to_text_TumorLesionTask(
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

    # [Not recommended] Save processed images to dataset, making the cached dataset very large
    if process_img:
        example["processed_images"] = _doc_to_visual(example)

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        # Process image: read from nii.gz file and extract 2D slice
        pil_img = _doc_to_visual(example)[0]

        # Save tmp PNGs next to the source image inside a tmp_prepared_png folder
        img_path = example["image_file"]
        slice_dim = example["slice_dim"]
        slice_idx = example["slice_idx"]
        png_basename = Path(img_path).name.split(".", 1)[0]
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}.png"
        png_dir = os.path.join(os.path.dirname(img_path), "tmp_prepared_png")
        png_path = os.path.join(png_dir, png_filename)
        os.makedirs(png_dir, exist_ok=True)
        pil_img.save(png_path)
        example["image_file_png"] = [png_path]

    return example


# NOTE: This is dataset-specific formatting function
def _format_data_TumorLesionTask_CoT(
    example, img_processor=None, reshape_size=None, process_img=False, save_processed_img_to_disk=False,
):
    """
    Format data for TumorLesionTask with CoT reasoning.
    Compared to the non-CoT version, this function:
    1. Uses a different prompt template that includes reasoning steps.
    2. Returns a target string that includes reasoning steps.
    """
    prompt, values_dict = _doc_to_text_TumorLesionTask_CoT(
        example, img_processor, reshape_size)
    target_str = _doc_to_target_TumorLesionTask_CoT(values_dict)

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

    # [Not recommended] Save processed images to dataset, making the cached dataset very large
    if process_img:
        example["processed_images"] = _doc_to_visual(example)

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        # Process image: read from nii.gz file and extract 2D slice
        pil_img = _doc_to_visual(example)[0]

        # Save tmp PNGs next to the source image inside a tmp_prepared_png folder
        img_path = example["image_file"]
        slice_dim = example["slice_dim"]
        slice_idx = example["slice_idx"]
        png_basename = Path(img_path).name.split(".", 1)[0]
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}.png"
        png_dir = os.path.join(os.path.dirname(img_path), "tmp_prepared_png")
        png_path = os.path.join(png_dir, png_filename)
        os.makedirs(png_dir, exist_ok=True)
        pil_img.save(png_path)
        example["image_file_png"] = [png_path]

    return example


def _doc_to_text_DetectionTask(doc):
    """Convert document to text."""
    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(
            f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_detection = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_detection"
    )

    # Get task infoG
    taskID = doc["taskID"]
    bm_plan = preprocess_detection.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]
    # Get label info
    label = str(doc["label"])
    labels_map = task_info["labels_map"]
    if label not in labels_map:
        raise ValueError(f"Label {label} not found in labels_map.")
    else:
        label_name = labels_map.get(label)
    # Get image info
    image_description = task_info["image_description"]

    # Format prompt for box coordinates
    format_prompt_BoxCoordinates = "Return the coordinates as [x1, y1, x2, y2] where (x1, y1) is the lower-left corner and (x2, y2) is the upper-right corner, with coordinates normalized to [0, 1]."

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image: {image_description}, "
        f"return the coordinates of the lower-left and upper-right corner of the bounding box for the {label_name}.\n"
        f"Format requirement:\n"
        f"{format_prompt_BoxCoordinates}"
    )
    return question


def _doc_to_target_DetectionTask(doc):
    """
    Get bounding coordinates.
    NOTE:
    Definition of the output bounding box coordinates:
    1.  The origin of the coordinates is at the [lower-left corner] of the image.
    2.  The first two numbers are the coordinates of the [lower-left] corner and
        the last two numbers are the coordinates of the [upper-right] corner of the bounding box.
    3.  The coordinates are expected to be in the format of [coor0_dim1, coor0_dim0, coor1_dim1, coor1_dim0], where:
        - coor0: lower-left corner of the bounding box
        - coor1: upper-right corner of the bounding box
        - dim0: the first dimension of the image (height)
        - dim1: the second dimension of the image (width)
    NOTE:
    Definition of bounding box coordinates in the benchmark planner:
    1. The origin of the coordinates is at the [top-left corner] of the image.
    2. The first two numbers are the coordinates of the [upper-left] corner and
       the last two numbers are the coordinates of the [lower-right] corner of the bounding box.
    NOTE: CAVEAT!
    !!! We need to convert the coordinates from the benchmark planner format to the output format. !!!
    """
    # Read NIfTI image
    img_size = doc["image_size_2d"]
    # Convert the coordinates from the benchmark planner format to the output format.
    bm_coor0_dim0, bm_coor0_dim1 = doc["bounding_boxes"]["min_coords"][0]
    bm_coor1_dim0, bm_coor1_dim1 = doc["bounding_boxes"]["max_coords"][0]
    img_coor0_dim0 = bm_coor0_dim1
    img_coor0_dim1 = img_size[0] - bm_coor1_dim0
    img_coor1_dim0 = bm_coor1_dim1
    img_coor1_dim1 = img_size[0] - bm_coor0_dim0
    # Convert bounding box coordinates to relative coordinates
    coor0_dim1 = img_coor0_dim1 / img_size[0]
    coor0_dim0 = img_coor0_dim0 / img_size[1]
    coor1_dim1 = img_coor1_dim1 / img_size[0]
    coor1_dim0 = img_coor1_dim0 / img_size[1]
    # Return the relative coordinates in the image space (the origin is at the lower-left corner)
    return [coor0_dim0, coor0_dim1, coor1_dim0, coor1_dim1]


# NOTE: This is dataset-specific formatting function
# NOTE: img_processor and reshape_size are not used for detection task, but kept for API consistency
def _format_data_DetectionTask(example, img_processor=None, reshape_size=None, process_img=False, save_processed_img_to_disk=False):
    target_coords = _doc_to_target_DetectionTask(example)
    coord_str = f"{target_coords[0]:.3f}, {target_coords[1]:.3f}, {target_coords[2]:.3f}, {target_coords[3]:.3f}"

    example["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": _doc_to_text_DetectionTask(example),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": coord_str,
                },
            ],
        },
    ]

    # [Not recommended] Save processed images to dataset, making the cached dataset very large
    if process_img:
        example["processed_images"] = _doc_to_visual(example)

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        # Process image: read from nii.gz file and extract 2D slice
        pil_img = _doc_to_visual(example)[0]

        # Save tmp PNGs next to the source image inside a tmp_prepared_png folder
        img_path = example["image_file"]
        slice_dim = example["slice_dim"]
        slice_idx = example["slice_idx"]
        png_basename = Path(img_path).name.split(".", 1)[0]
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}.png"
        png_dir = os.path.join(os.path.dirname(img_path), "tmp_prepared_png")
        png_path = os.path.join(png_dir, png_filename)
        os.makedirs(png_dir, exist_ok=True)
        pil_img.save(png_path)
        example["image_file_png"] = [png_path]

    return example


def _format_data_DetectionTask_CoT():
    raise NotImplementedError(
        "CoT formatting for DetectionTask is not implemented yet. "
        "Please use the non-CoT version for now."
    )


def _load_single_dataset(task, tag_ds):
    """Load a single dataset configuration with improved error handling."""
    try:
        print(f"\n[Info] Loading dataset for task: {task}")
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
                        f"[Warning] Attempt {attempt + 1} failed for {task}, retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        # NOTE: This is specific to the MedVision dataset and configs
        # Add dataset name column
        # Extract dataset name (part before "_BiometricsFromLandmarks")
        dataset_name = task.split(f"_{tag_ds}")[0]
        ds = ds.add_column("dataset_name", [dataset_name] * len(ds))

        print(
            f"\n[Info] Successfully loaded {len(ds)} samples from config {config} (dataset: {dataset_name})"
        )
        return ds

    except Exception as e:
        print(f"[Error] Failed to load dataset for task {task}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Task {task} failed: {str(e)}")


def safe_concat_align_top_keys(datasets_list, fill_value=None):
    """
    Concatenate Hugging Face datasets even if they have different top-level keys.
    Missing columns are added and filled with `fill_value`.
    """
    # get union of all column names
    all_columns = set()
    for ds in datasets_list:
        all_columns.update(ds.column_names)
    all_columns = sorted(all_columns)

    # ensure all datasets have the same columns
    aligned = []
    for ds in datasets_list:
        missing = [c for c in all_columns if c not in ds.column_names]
        for c in missing:
            ds = ds.add_column(c, [fill_value] * len(ds))
        aligned.append(ds)

    return aligned


def safe_concat_align_dict_keys(datasets_list, dict_cols=None, fill_value=None):
    """
    Concatenate Hugging Face datasets even if dict columns have different keys.

    Args:
        datasets_list (list[Dataset]): list of datasets to concatenate
        dict_cols (list[str] | None): names of columns containing dicts
                                      (if None, auto-detects)
        fill_value (any): value used to fill missing keys (default: None)
    Returns:
        Dataset: concatenated dataset
    """
    if not datasets_list:
        raise ValueError("datasets_list cannot be empty.")

    # auto-detect dict columns if not provided
    if dict_cols is None:
        sample = datasets_list[0][0]
        dict_cols = [k for k, v in sample.items() if isinstance(v, dict)]

    # unify keys across all dict columns
    key_union = {}
    for col in dict_cols:
        keys = set()
        for ds in datasets_list:
            for d in ds[col]:
                keys.update(d.keys())
        key_union[col] = sorted(keys)

    # normalize each dataset
    def pad_dict_keys(example):
        for col in dict_cols:
            d = example[col]
            for k in key_union[col]:
                if k not in d:
                    d[k] = fill_value
            example[col] = {k: d[k] for k in key_union[col]}
        return example

    normalized = [ds.map(pad_dict_keys, desc="Normalizing dict columns")
                  for ds in datasets_list]
    return normalized


def safe_concatenate_datasets(datasets_list):
    datasets_list = safe_concat_align_top_keys(
        datasets_list, fill_value=None)
    datasets_list = safe_concat_align_dict_keys(
        datasets_list, dict_cols=None, fill_value=None
    )
    combined_dataset = concatenate_datasets(datasets_list)
    return combined_dataset


def load_split_limit_dataset(
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    num_workers_concat_datasets=4,
    tag_ds=None,
):
    # NOTE:
    # - limit_val_sample must be greater than 0 to ensure validation set is not empty.
    # - limit_train_sample can be <0 (no limit) or >0 (limited training set).
    assert limit_val_sample > 0, "\n [Error] limit_val_sample must be greater than 0."
    assert limit_train_sample != 0, "\n [Error] limit_train_sample cannot be 0. Use <0 for no limit or >0 for limited training set."

    # Early assertions
    assert tag_ds is not None, "\n [Error] tag_ds (i.e., the string in tasks names: <dataset_name>_<tag_ds>) must be provided."

    print(f"\n[Info] Starting dataset preparation from {tasks_list_json_path}")

    # Load tasks list from JSON file
    with open(tasks_list_json_path, "r") as f:
        tasks_dict = json.load(f)
    tasks = list(tasks_dict.keys())

    print(f"[Info] Found {len(tasks)} tasks to process")

    # Reduce parallelism to avoid memory issues - use fewer workers
    available_cpus = get_cgroup_limited_cpus()
    concat_workers = min(num_workers_concat_datasets,
                         available_cpus, len(tasks))

    # NOTE: Force single-threaded loading for new datasets to avoid conflicts, otherwise errors may occur.
    data_dir = os.environ.get("MedVision_DATA_DIR")
    assert data_dir is not None, "\n [Error] MedVision_DATA_DIR environment variable must be set."
    # Read the .downloaded_datasets.json file in data_dir
    with open(os.path.join(data_dir, ".downloaded_datasets.json"), "r") as f:
        downloaded_datasets = list(json.load(f).keys())
    for task in tasks:
        # NOTE: This is specific to the MedVision dataset and configs: extract dataset name (part before "_<tag_ds>")
        dataset_name = task.split(f"_{tag_ds}")[0]
        if f"dataset_{dataset_name}" not in downloaded_datasets:
            concat_workers = 1
            print(
                f"[Info] Dataset {dataset_name} is newly downloaded. Using single-threaded loading to avoid conflicts."
            )
            break

    print(
        f"[Info] Using {concat_workers} workers for dataset loading (available CPUs: {available_cpus})"
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
                ds = future.result(timeout=120)  # 2 minute timeout per task
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
    print("\n[Info] Combining datasets...")
    combined_dataset = concatenate_datasets(datasets_list)
    print(f"[Info] Combined dataset has {len(combined_dataset)} total samples")

    # Clear intermediate datasets to free memory
    del datasets_list
    gc.collect()

    # Split the dataset into training and validation sets
    print(
        f"\n[Info] Splitting dataset into training (size: {len(combined_dataset) - limit_val_sample}) and validation (size: {limit_val_sample}) sets")
    dataset = combined_dataset.train_test_split(
        train_size=len(combined_dataset) - limit_val_sample,
        test_size=limit_val_sample,
        shuffle=True,
        seed=SEED,
    )
    dataset["validation"] = dataset.pop("test")

    # Limit the number of training and validation samples if specified
    if limit_train_sample > 0 and limit_train_sample < len(dataset["train"]):
        print(
            f"\n[Info][Warning] Limiting training samples to {limit_train_sample} (original: {len(dataset['train'])})"
        )
        dataset["train"] = (
            dataset["train"].shuffle(seed=SEED).select(
                range(limit_train_sample))
        )
    return dataset


def format_dataset(dataset, mapping_func, mapping_func_args, num_workers_format_dataset):
    img_processor = mapping_func_args.get("img_processor")
    reshape_size = mapping_func_args.get("reshape_size")
    assert img_processor is not None or reshape_size is not None, "\n [Error] Either img_processor or reshape_size must be provided."
    assert not (
        img_processor is not None and reshape_size is not None), "\n [Error] Provide only one of img_processor or reshape_size, not both."

    # Format the dataset with parallelism
    # Use conservative parallelism for formatting to avoid OOM
    available_cpus = get_cgroup_limited_cpus()
    format_workers = min(num_workers_format_dataset, available_cpus)
    print(f"\n[Info] Formatting dataset with {format_workers} workers...")
    dataset = dataset.map(
        mapping_func,
        fn_kwargs=mapping_func_args,
        num_proc=format_workers,
        desc="Formatting dataset",
    )
    return dataset


def clean_dataset(dataset, keys_to_keep):
    def _clean_dataset_map(example, keys_to_keep):
        for key in list(example.keys()):
            if key not in keys_to_keep:
                del example[key]
        return example

    dataset = dataset.map(
        _clean_dataset_map,
        fn_kwargs={"keys_to_keep": keys_to_keep},
        desc="Cleaning dataset",
    )
    return dataset


def prepare_dataset(
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    mapping_func,
    num_workers_concat_datasets=4,
    num_workers_format_dataset=32,
    tag_ds=None,
    img_processor=None,
    reshape_size=None,
    process_img=False,
    save_processed_img_to_disk=False,
):
    # Load and split dataset
    dataset = load_split_limit_dataset(
        tasks_list_json_path=tasks_list_json_path,
        limit_train_sample=limit_train_sample,
        limit_val_sample=limit_val_sample,
        num_workers_concat_datasets=num_workers_concat_datasets,
        tag_ds=tag_ds,
    )

    # Format dataset
    mapping_func_args = {
        "img_processor": img_processor,
        "reshape_size": reshape_size,
        "process_img": process_img,
        "save_processed_img_to_disk": save_processed_img_to_disk,
    }
    dataset = format_dataset(
        dataset=dataset,
        mapping_func=mapping_func,
        mapping_func_args=mapping_func_args,
        num_workers_format_dataset=num_workers_format_dataset
    )

    # Clean dataset to keep only necessary keys
    keys_to_keep = ["messages", "labels",
                    "image_file", "slice_dim", "slice_idx"]
    if process_img:
        keys_to_keep.append("processed_images")
    if save_processed_img_to_disk:
        keys_to_keep.append("image_file_png")
    dataset = clean_dataset(dataset, keys_to_keep)

    return dataset


def recompute_total_max_steps(trainer):
    """Recompute total planned update steps based on global dataset size, world size and desired epochs."""
    args = trainer.args
    grad_accum = args.gradient_accumulation_steps
    epoch = args.num_train_epochs
    per_device_bsz = args.per_device_train_batch_size

    # Prefer accelerate's world size; fallback to Trainer args/env
    state = PartialState()
    world_size = getattr(state, "num_processes", None) or getattr(
        args, "world_size", None)
    if not world_size or world_size < 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

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

        new_max_steps = steps_per_epoch * epoch

        # Main-process-only logs
        print(f"[Resume] world_size: {world_size}")
        print(f"[Resume] dataset size (global): {dataset_n}")
        print(f"[Resume] per_device_train_batch_size: {per_device_bsz}")
        print(f"[Resume] gradient_accumulation_steps: {grad_accum}")
        print(f"[Resume] num_train_epochs: {epoch}")
        print(f"[Resume] steps_per_epoch (computed): {steps_per_epoch}")
        print(
            f"[Resume] Recomputed new_max_steps (epochs based): {new_max_steps}")

    # Share the computed value to all processes so every worker uses the exact same max_steps.
    # This prevents mismatched training horizons, inconsistent checkpointing, or hangs in collective ops.
    new_max_steps = broadcast_int_from_main(new_max_steps)
    return new_max_steps


def prepare_trainer(
    run_name,
    base_model_hf,
    lora_checkpoint_dir,
    data,
    make_collate_fn,
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
    merged_model_hf,
    merged_model_dir,
    push_to_hub,
):
    """
    Merge LoRA adapter with base model and optionally save locally and/or push to Hugging Face Hub.
    This function is intended to be called **only on the main process**.
    """

    print("\n[Info] Starting model merge process (CPU-only)...")

    # 1) Load base model on CPU
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_hf,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,  # or float16/float32 as appropriate
        device_map="cpu",
    )

    # 2) Load LoRA adapter and merge
    peft_model = PeftModel.from_pretrained(model, lora_checkpoint_dir)
    merged_model = peft_model.merge_and_unload()

    # Drop references to base + peft wrapper
    del model, peft_model
    gc.collect()

    # 3) Load processor from the adapter
    processor = AutoProcessor.from_pretrained(lora_checkpoint_dir)

    # 4) Save locally (optional)
    if merged_model_dir is not None:
        print(f"[Info] Saving merged model to: {merged_model_dir}")
        merged_model.save_pretrained(
            merged_model_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        processor.save_pretrained(merged_model_dir)
        print(f"[Info] Merged model saved to: {merged_model_dir}")

    # 5) Push to Hub (optional)
    if push_to_hub:
        if merged_model_hf is None:
            raise ValueError(
                "[Error] merged_model_hf must be specified when push_to_hub is True."
            )
        print(
            f"[Info] Pushing merged model to Hugging Face Hub: {merged_model_hf}")
        merged_model.push_to_hub(
            merged_model_hf,
            private=True,
            max_shard_size="2GB",
        )
        processor.push_to_hub(merged_model_hf, private=True)
        print(f"[Info] Successfully pushed merged model to: {merged_model_hf}")

    # 6) Final cleanup
    del merged_model, processor
    gc.collect()

    print("[Info] Model merge completed.")


def train_resume_from_checkpoint(trainer, last_checkpoint):
    safe_print("[Resume] Requested resume_from_checkpoint=True")

    assert last_checkpoint is not None, f"No checkpoint found in {last_checkpoint}"
    safe_print(f"[Resume] Found checkpoint: {last_checkpoint}")

    # recompute_total_max_steps already broadcasts the integer so every process
    # receives the same `new_max_steps` value in its local variable.
    new_max_steps = recompute_total_max_steps(
        trainer
    )

    # --- load previous trainer_state.json directly (avoid non-existent _load_state) ---
    trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")
    try:
        # Only main process reads the checkpoint file and computes the decision.
        prev_global = None
        prev_recorded_max = None
        should_finish_int = 0  # 0 -> False, 1 -> True
        if is_main_process():
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                _prev_state = json.load(f)
            prev_global = _prev_state.get("global_step")
            prev_recorded_max = _prev_state.get("max_steps")
            print(
                f"[Resume] Loaded previous trainer_state.json: global_step={prev_global}, max_steps={prev_recorded_max}"
            )

            # Decide whether training is already finished relative to the new horizon.
            if new_max_steps <= prev_recorded_max and prev_global >= new_max_steps:
                should_finish_int = 1
        else:
            # Non-main processes don't read the file.
            prev_global = None
            prev_recorded_max = None

        # Broadcast the boolean decision (as int) so every process knows whether to mark finished.
        should_finish_int = broadcast_int_from_main(should_finish_int)
        should_finish = bool(should_finish_int)

    except Exception as e:
        raise RuntimeError(
            f"[Resume] Failed to read trainer_state.json ({e}); cannot resume training.")
    # -------------------------------------------------------------------------------

    # Apply the new_max_steps and is_finished flag on every process for consistency.
    # This ensures all processes have identical trainer args/state before training resumes.
    trainer.args.max_steps = new_max_steps
    trainer.state.max_steps = new_max_steps
    trainer.state.is_finished = should_finish

    # Main-process-only logs (kept for visibility)
    if is_main_process():
        if new_max_steps <= (prev_recorded_max or -1):
            if should_finish:
                print(
                    "[Resume] Training already satisfies (or exceeds) the new reduced horizon."
                    " Nothing further to do. If you intended more training, increase num_train_epochs."
                )
            else:
                print(
                    "[Resume] Horizon reduced (or unchanged) and progress not past new_max_steps; continuing."
                )
        else:
            print("[Resume] Extending training horizon.")

        print(
            f"[Resume] Applied new_max_steps={new_max_steps} on all processes.")
        print(
            f"[Resume] Marked is_finished={trainer.state.is_finished} on all processes.")

    safe_print("Resuming training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)


def parse_args_multiTask():
    """
    Add this block to add sample limit fallbacks: when task-specific limit is not set (default to -1), fallback to use 
    the --val_sample_limit_per_task (default to 100) and --train_sample_limit_per_task (default to -1 meaning no limit). 

    # Get command-line arguments
    args = parse_args_Qwen25VL_multiTask()
    kwargs = vars(args)

    # Determine sample limits for each task
    if kwargs.get("train_sample_limit_task_AD") > 0:
        train_limit_AD = kwargs.get("train_sample_limit_task_AD")
    else:
        train_limit_AD = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_AD") > 0:
        val_limit_AD = kwargs.get("val_sample_limit_task_AD")
    else:
        val_limit_AD = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("train_sample_limit_task_Detection") > 0:
        train_limit_detect = kwargs.get(
            "train_sample_limit_task_Detection")
    else:
        train_limit_detect = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_Detection") > 0:
        val_limit_detect = kwargs.get(
            "val_sample_limit_task_Detection")
    else:
        val_limit_detect = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("train_sample_limit_task_TL") > 0:
        train_limit_TL = kwargs.get("train_sample_limit_task_TL")
    else:
        train_limit_TL = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_TL") > 0:
        val_limit_TL = kwargs.get("val_sample_limit_task_TL")
    else:
        val_limit_TL = kwargs.get("val_sample_limit_per_task")
    train_limit_total = kwargs.get("train_sample_limit")
    """

    parser = argparse.ArgumentParser(
        description="SFT on the MedVision dataset"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the run",
    )

    # -- Model arguments
    parser.add_argument(
        "--base_model_hf",
        type=str,
        help="Hugging Face model ID for the base model",
    )
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        help="Local directory path for LoRA checkpoint",
    )
    parser.add_argument(
        "--merged_model_hf",
        type=str,
        help="Hugging Face repository ID for merged model",
    )
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        help="Local directory path for merged model",
    )

    # -- wandb logging arguments
    parser.add_argument(
        "--wandb_resume",
        type=str,
        default="allow",
        help="Wandb resume mode (e.g., 'allow', 'must', 'never')",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        help="Directory for wandb logs",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        help="Wandb run ID for resuming",
    )

    # -- Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Dataset folder",
    )
    parser.add_argument(
        "--tasks_list_json_path_AD",
        type=str,
        help="Path to the tasks list JSON file for angle distance task",
    )
    parser.add_argument(
        "--tasks_list_json_path_detect",
        type=str,
        help="Path to the tasks list JSON file for detection task",
    )
    parser.add_argument(
        "--tasks_list_json_path_TL",
        type=str,
        help="Path to the tasks list JSON file for tumor lesion size task",
    )
    parser.add_argument(
        "--process_img",
        type=str2bool,
        default=False,
        help="Whether to process images during dataset formatting",
    )
    parser.add_argument(
        "--process_dataset_only",
        type=str2bool,
        default=False,
        help="Only process dataset without training",
    )
    parser.add_argument(
        "--skip_process_dataset",
        type=str2bool,
        default=False,
        help="Skip dataset processing and directly load from disk",
    )
    parser.add_argument(
        "--prepared_ds_dir",
        type=str,
        help="Path to the prepared dataset directory to load from disk",
    )
    parser.add_argument(
        "--save_processed_img_to_disk",
        type=str2bool,
        default=False,
        help="Whether to save processed images to PNG files on disk during dataset formatting",
    )

    # -- Training arguments
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between model saves",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=10,
        help="Maximum number of checkpoints to save",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=20,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=20,
        help="Batch size per device during evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps before performing a backward/update pass",
    )
    parser.add_argument(
        "--use_flash_attention_2",
        type=str2bool,
        default=True,
        help="Use Flash Attention 2 for training",
    )
    parser.add_argument(
        "--num_workers_concat_datasets",
        type=int,
        default=4,
        help="Number of workers for concatenating datasets, should be <= number of tasks",
    )
    parser.add_argument(
        "--num_workers_format_dataset",
        type=int,
        default=32,
        help="Number of workers for formatting datasets",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    # This is only for multi-task training to limit the number of samples per task
    parser.add_argument(
        "--train_sample_limit_per_task",
        type=int,
        default=-1,
        help="Limit the number of training samples per task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_per_task",
        type=int,
        default=100,
        help="Limit the number of validation samples per task",
    )
    # Task-specific sample limit
    parser.add_argument(
        "--train_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of training samples for angle distance task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_AD",
        type=int,
        default=-1,
        help="Limit the number of validation samples for angle distance task, -1 means no limit",
    )
    parser.add_argument(
        "--train_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of training samples for detection task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_Detection",
        type=int,
        default=-1,
        help="Limit the number of validation samples for detection task, -1 means no limit",
    )
    parser.add_argument(
        "--train_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of training samples for tumor lesion task, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit_task_TL",
        type=int,
        default=-1,
        help="Limit the number of validation samples for tumor lesion task, -1 means no limit",
    )
    # This is to limit the number of samples in total
    parser.add_argument(
        "--train_sample_limit",
        type=int,
        default=-1,
        help="Limit the number of training samples, -1 means no limit",
    )
    parser.add_argument(
        "--val_sample_limit",
        type=int,
        default=100,
        help="Limit the number of validation samples",
    )
    parser.add_argument(
        "--push_LoRA",
        type=str2bool,
        default=False,
        help="Push LoRA checkpoint to HF Hub after each save",
    )
    parser.add_argument(
        "--push_merged_model",
        type=str2bool,
        default=False,
        help="Push merged model to HF Hub after merging",
    )
    parser.add_argument(
        "--merge_model",
        type=str2bool,
        default=False,
        help="Merge LoRA with base model after training",
    )
    parser.add_argument(
        "--merge_only",
        type=str2bool,
        default=False,
        help="ONLY Merge LoRA with base model and push to HF Hub, no training",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str2bool,
        default=False,
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str2bool,
        default=False,
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        type=str2bool,
        default=True,
        help="Pin memory for faster GPU transfer",
    )
    args = parser.parse_args()
    return args


def parse_validate_args_multiTask():
    args = parse_args_multiTask()

    # Arguments
    # ------------------------------------------------------------
    # -- wandb logging
    wandb_resume = args.wandb_resume
    wandb_dir = args.wandb_dir
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_run_id = args.wandb_run_id
    # -- Data
    tasks_list_json_path_AD = args.tasks_list_json_path_AD
    tasks_list_json_path_detect = args.tasks_list_json_path_detect
    tasks_list_json_path_TL = args.tasks_list_json_path_TL
    # ------------------------------------------------------------

    # Ensure at least one task JSON path is provided (they don't all have to be present).
    if (
        tasks_list_json_path_AD is None
        and tasks_list_json_path_detect is None
        and tasks_list_json_path_TL is None
    ):
        raise AssertionError(
            "\n[Error] At least one of --tasks_list_json_path_AD, "
            "--tasks_list_json_path_detect, or --tasks_list_json_path_TL must be provided.\n"
        )

    # Set wandb environment variables
    os.environ["WANDB_RESUME"] = wandb_resume
    if wandb_dir is not None:
        os.environ["WANDB_DIR"] = wandb_dir
        os.makedirs(wandb_dir, exist_ok=True)
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name is not None:
        os.environ["WANDB_NAME"] = wandb_run_name
    if wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = wandb_run_id

    return vars(args)


def parse_sample_limits(**kwargs):
    # Determine sample limits for each task
    # Angle/distance task
    if kwargs.get("train_sample_limit_task_AD") > 0:
        train_limit_AD = kwargs.get("train_sample_limit_task_AD")
    else:
        train_limit_AD = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_AD") > 0:
        val_limit_AD = kwargs.get("val_sample_limit_task_AD")
    else:
        val_limit_AD = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("tasks_list_json_path_AD") is None:
        train_limit_AD = 0
        val_limit_AD = 0
    # Detection task
    if kwargs.get("train_sample_limit_task_Detection") > 0:
        train_limit_detect = kwargs.get(
            "train_sample_limit_task_Detection")
    else:
        train_limit_detect = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_Detection") > 0:
        val_limit_detect = kwargs.get(
            "val_sample_limit_task_Detection")
    else:
        val_limit_detect = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("tasks_list_json_path_detect") is None:
        train_limit_detect = 0
        val_limit_detect = 0
    # Tumor lesion size task
    if kwargs.get("train_sample_limit_task_TL") > 0:
        train_limit_TL = kwargs.get("train_sample_limit_task_TL")
    else:
        train_limit_TL = kwargs.get("train_sample_limit_per_task")
    if kwargs.get("val_sample_limit_task_TL") > 0:
        val_limit_TL = kwargs.get("val_sample_limit_task_TL")
    else:
        val_limit_TL = kwargs.get("val_sample_limit_per_task")
    if kwargs.get("tasks_list_json_path_TL") is None:
        train_limit_TL = 0
        val_limit_TL = 0
    # Total sample limit across all tasks
    train_limit_total = kwargs.get("train_sample_limit")

    return (train_limit_AD, val_limit_AD,
            train_limit_detect, val_limit_detect,
            train_limit_TL, val_limit_TL,
            train_limit_total)
