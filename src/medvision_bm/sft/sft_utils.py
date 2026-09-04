import argparse
import gc
import gzip
import importlib
import io
import json
import math
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import psutil
import torch
from accelerate import PartialState
from datasets import DatasetDict, concatenate_datasets, load_dataset
from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import WeightedRandomSampler

from medvision_bm.sft.sft_prompts import (
    _get_prompt_angle,
    _get_prompt_distance,
    fill_in_template,
)
from medvision_bm.utils import str2bool
from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE, SEED
from medvision_bm.utils.tool_execution import safe_exec_python


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


def broadcast_object_from_main(value, src=0):
    """Broadcast any picklable object from the main process to all other processes.

    Same rationale as :func:`broadcast_int_from_main`; used e.g. for a prepared-dataset
    path that only the main process can resolve. Returns ``value`` unchanged when not
    running distributed.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        obj = [value if dist.get_rank() == src else None]
        dist.broadcast_object_list(obj, src=src)
        return obj[0]
    return value


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


def get_cgroup_memory_percent():
    """Return ``(used_gib, limit_gib, percent)`` for the memory cgroup.

    Reads cgroup v2 (``memory.current`` / ``memory.max``), falling back to
    cgroup v1 and finally to host-wide psutil numbers. On pods psutil reports
    the HOST total, so the cgroup files are the only trustworthy signal there.
    """
    # cgroup v2
    try:
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        limit_txt = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit_txt != "max":
            limit = int(limit_txt)
            return used / 2**30, limit / 2**30, 100.0 * used / limit
    except (ValueError, OSError):
        pass

    # cgroup v1
    try:
        base = Path("/sys/fs/cgroup/memory")
        used = int((base / "memory.usage_in_bytes").read_text().strip())
        limit = int((base / "memory.limit_in_bytes").read_text().strip())
        # v1 reports a huge sentinel value when unlimited
        if 0 < limit < 1 << 60:
            return used / 2**30, limit / 2**30, 100.0 * used / limit
    except (ValueError, OSError):
        pass

    # fallback to host-wide memory
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 2**30, vm.total / 2**30, vm.percent


# One-slot per-process cache of the last decompressed volume. Rows reach the
# formatting map sorted by image_file (see format_dataset), so consecutive rows
# in a worker share a volume and each .nii.gz is decompressed once, not per slice.
_NIFTI_CACHE = {"path": None, "voxel_size": None, "image_3d": None}


def _load_nifti_2d(nii_path, slice_dim, slice_idx):
    """Map function to load 2D slice from a 3D NIFTI images."""
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Image file {nii_path} does not exist.")
    if _NIFTI_CACHE["path"] != nii_path:
        img_nib = nib.load(nii_path)
        _NIFTI_CACHE["voxel_size"] = img_nib.header.get_zooms()
        _NIFTI_CACHE["image_3d"] = img_nib.get_fdata().astype("float32")
        _NIFTI_CACHE["path"] = nii_path
    voxel_size = _NIFTI_CACHE["voxel_size"]
    image_3d = _NIFTI_CACHE["image_3d"]
    # NOTE: .copy() detaches the slice from the cached volume so downstream
    # in-place ops can never corrupt the cache.
    if slice_dim == 0:
        image_2d = image_3d[slice_idx, :, :].copy()
        pixel_size = voxel_size[1:3]
    elif slice_dim == 1:
        image_2d = image_3d[:, slice_idx, :].copy()
        pixel_size = voxel_size[0:1] + voxel_size[2:3]
    elif slice_dim == 2:
        image_2d = image_3d[:, :, slice_idx].copy()
        pixel_size = voxel_size[0:2]
    else:
        raise ValueError("slice_dim must be 0, 1 or 2")
    return (pixel_size, image_2d)


def _load_resize_nifti_2d(nii_path, slice_dim, slice_idx, new_shape_hw=None):
    """Map function to load 2D slice from a 3D NIFTI images and maybe resize."""
    pixel_size_hw, image_2d = _load_nifti_2d(nii_path, slice_dim, slice_idx)

    # Reshape image and update pixel size if new_shape_hw is provided
    if new_shape_hw is not None:
        original_shape_hw = image_2d.shape
        # Calculate zoom factors for each dimension
        zoom_factors = (
            new_shape_hw[0] / original_shape_hw[0],
            new_shape_hw[1] / original_shape_hw[1],
        )
        # Use scipy.ndimage.zoom for resizing (order=1 for bilinear interpolation)
        image_2d = zoom(image_2d, zoom_factors, order=1)

        # Update pixel size based on zoom factors
        pixel_size_hw = (
            pixel_size_hw[0] / zoom_factors[0],
            pixel_size_hw[1] / zoom_factors[1],
        )

    return (pixel_size_hw, image_2d)


# NOTE: This function only works for MedVision dataset
def get_image_info_for_medvision_dataset(doc):
    """Get the image modality and label name for a MedVision sample.

    The sample's ``taskType`` (defined in ``MedVision.py`` on the dataset repo)
    selects the dataset-specific preprocessing module, from which the image
    modality and the human-readable label name are looked up.

    Args:
        doc: A data sample from the MedVision dataset. Must contain
            ``taskType``, ``dataset_name`` and ``taskID``; ``label`` is present
            for all task types except the Biometrics-From-Landmarks (angle /
            distance) tasks.

    Returns:
        tuple: ``(image_modality, label_name)``, where ``label_name`` is
        ``None`` for tasks that have no ``label`` (the angle / distance tasks).

    Raises:
        ValueError: If ``taskType`` is not one of the valid task types, or the
            dataset is not registered in ``DATASETS_NAME2PACKAGE``.
    """
    # Validate taskType
    valid_task_types = [
        "Mask-Size",
        "Box-Size",
        "Tumor-Lesion-Size",
        "Biometrics-From-Landmarks",
        "Biometrics-From-Landmarks-Distance",
        "Biometrics-From-Landmarks-Angle",
    ]
    task_type = doc["taskType"]
    if task_type not in valid_task_types:
        raise ValueError(
            f"Invalid taskType: {task_type}. Must be one of {valid_task_types}."
        )

    # Get data info
    dataset_name = doc["dataset_name"]

    # Import the dataset-specific module from medvision_ds.datasets
    if task_type in ["Box-Size"]:
        processor_module_name = "preprocess_detection"
    elif task_type in ["Mask-Size"]:
        processor_module_name = "preprocess_segmentation"
    elif task_type in [
        "Tumor-Lesion-Size",
        "Biometrics-From-Landmarks",
        "Biometrics-From-Landmarks-Distance",
        "Biometrics-From-Landmarks-Angle",
    ]:
        processor_module_name = "preprocess_biometry"
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    processor_module = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.{processor_module_name}"
    )

    # Get task info
    taskID = doc["taskID"]
    bm_plan = processor_module.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get label_name from label and labels_map
    # NOTE: Biometrics-From-Landmarks* (A/D) tasks do not have "label", check class MedVision(GeneratorBasedBuilder) in MedVision.py (https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/MedVision.py)
    # NOTE: Biometrics-From-Landmarks* (A/D) tasks do not have "labels_map" in task_info, check preprocess_biometry.py in dataset folder at https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/src/medvision_ds/datasets
    if "label" in doc and "labels_map" in task_info:
        label = str(doc["label"])
        labels_map = task_info["labels_map"]
        assert label in labels_map, f"Label {label} not found in labels_map."
        label_name = labels_map.get(label)
    else:
        label_name = None

    # Get image modality
    image_modality = task_info["image_modality"]

    return image_modality, label_name


def normalize_ct_img(img, window_width, window_level):
    """
    Normalizes CT Hounsfield Units to [0, 255] based on W and L.
    """
    v_min = window_level - (window_width / 2)
    v_max = window_level + (window_width / 2)

    # Clip values to the window range
    img_normalized = np.clip(img, v_min, v_max)

    # Map to [0, 255]
    img_normalized = ((img_normalized - v_min) / (v_max - v_min)) * 255.0
    return img_normalized.astype(np.uint8)


def normalize_general_img(img):
    """
    Standard min-max normalization to [0, 255] for MR, PET, etc.
    """
    v_min = np.percentile(img, 0.5)
    v_max = np.percentile(img, 99.5)

    if v_max - v_min == 0:
        # If the image is flat/uniform, return black image
        return np.zeros_like(img, dtype=np.uint8)

    img_normalized = np.clip(img, v_min, v_max)
    img_normalized = ((img_normalized - v_min) / (v_max - v_min)) * 255.0
    return img_normalized.astype(np.uint8)


def normalize_img(doc, img_2d):
    """Convert document to image with scale bar added."""
    from medvision_bm.utils.configs import (
        TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION,
        CT_HU_windows_WL,
        label_map_regroup,
    )

    # Get image info
    # NOTE: For Biometrics-From-Landmarks* (A/D) tasks, label_name would be None
    image_modality, label_name = get_image_info_for_medvision_dataset(doc)

    # Check if this task requires standard image normalization (i.e., skip HU-based CT normalization)
    is_standard_normalization_required = False
    for task in TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION:
        if (
            task["dataset_name"] == doc["dataset_name"]
            and task["taskID"] == doc["taskID"]
            and task["taskType"] == doc["taskType"]
        ):
            is_standard_normalization_required = True
            break

    # Adaptive normalization
    # NOTE: A/D tasks in CT image do not have the optimal image normalization due to missing label_name used to decide the HU window
    # TODO: Could be improved by adding label_name or HU window info for A/D tasks in MedVision
    if image_modality.lower() in ["ct"]:
        # Use HU window-based normalization if: 1) label_name is not None, 2) label is not regrouped to "others", and 3) standard image normalization is not forced in this task
        if (
            label_name is not None
            and not is_standard_normalization_required
            and label_map_regroup[label_name].lower() != "others"
        ):
            hu_window_WL = CT_HU_windows_WL.get(label_map_regroup[label_name], None)
            assert (
                hu_window_WL is not None
            ), f"Fail to set HU window for label_name {label_name}. Check CT_HU_windows_WL in medvision_bm/utils/configs.py"
            img_2d_normalized = normalize_ct_img(
                img_2d, hu_window_WL[0], hu_window_WL[1]
            )
        else:
            if label_name is None:
                print(
                    "[Info] label_name is None, using general normalization (which does not use HU windows) for CT image."
                )
            if is_standard_normalization_required:
                print(
                    "[Info] standard image normalization is forced for this task, using general normalization (which does not use HU windows)"
                )
            if (
                label_name is not None
                and label_map_regroup[label_name].lower() == "others"
            ):
                print(
                    f"[Info] label_name {label_name} is regrouped to 'others', using general normalization (which does not use HU windows)"
                )
            img_2d_normalized = normalize_general_img(img_2d)
    else:
        img_2d_normalized = normalize_general_img(img_2d)

    return img_2d_normalized


def _doc_to_visual(doc, new_shape_hw=None):
    """Convert document to image with scale bar added."""
    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    # Load and maybe resize
    _, img_2d = _load_resize_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw)
    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d)
    # Convert to PIL Image
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")
    return [pil_img]


def _doc_to_text_AngleDistanceTask(doc, model_name, model_hf, new_shape_hw=None):
    """Convert document to text."""
    from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
        _normalize_metric_unit,
        get_resized_img_shape,
    )
    from medvision_bm.sft.sft_prompts import FORMAT_PROMPT_1_DECIMAL_NUMBER

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
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
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # Get 2D image info
    image_description = task_info["image_description"]

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    pixel_size_hw, img_2d_raw = _load_resize_nifti_2d(
        img_path, slice_dim, slice_idx, new_shape_hw
    )  # explicit resizing
    img_shape = img_2d_raw.shape

    # Get resized image shape
    img_shape_resized, img_shape_content_hw = get_resized_img_shape(
        model_name, img_2d_raw, {"model_hf": model_hf}
    )  # implicit/dynamic resizing from VLM

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_resized
    resize_ratio_h = img_shape_content_hw[0] / original_height
    resize_ratio_w = img_shape_content_hw[1] / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

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

    if image_description != "" and image_description is not None:
        image_prompt = ": " + image_description
    else:
        image_prompt = ""

    question = (
        f"Task:\n"
        f"Given the input medical image{image_prompt}, "
        f"{task_prompt}"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_1_DECIMAL_NUMBER}"
    )
    return question


def _doc_to_text_AngleDistanceTask_CoT(doc, model_name, model_hf, new_shape_hw=None):
    """Convert document to text."""
    from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
        _normalize_metric_unit,
        get_resized_img_shape,
    )
    from medvision_bm.sft.sft_prompts import (
        COT_INSTRUCT_ANGLE,
        COT_INSTRUCT_DISTANCE,
        FORMAT_PROMPT_AD_REASONING,
    )

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
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
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # Get 2D image info
    image_description = task_info["image_description"]

    # [!] Read NIfTI image with explicit resizing to the specified new_shape_hw (if not None)
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    pixel_size_hw, img_explicit_resize_2d = _load_resize_nifti_2d(
        img_path, slice_dim, slice_idx, new_shape_hw
    )
    img_shape = img_explicit_resize_2d.shape

    # [!] Get resized image shape (implicit/dynamic resizing from VLM)
    img_shape_implicit_resize, img_shape_content_hw = get_resized_img_shape(
        model_name, img_explicit_resize_2d, {"model_hf": model_hf}
    )

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_implicit_resize
    resize_ratio_h = img_shape_content_hw[0] / original_height
    resize_ratio_w = img_shape_content_hw[1] / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

    # Question
    if metric_type == "distance":
        # CoT instruction - reasoning step description
        cot_instruction = COT_INSTRUCT_DISTANCE
        # Task prompt - task description
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
        # CoT instruction - reasoning step description
        cot_instruction = COT_INSTRUCT_ANGLE
        # Task prompt - task description
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

    if image_description != "" and image_description is not None:
        image_prompt = ": " + image_description
    else:
        image_prompt = ""

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image{image_prompt}, "
        f"{task_prompt}"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_AD_REASONING}\n"
        f"Reasoning steps:\n"
        f"{cot_instruction}\n"
        f"Follow the reasoning steps to get the final answer in the required format."
    )

    # ------------------------------------------------------------------
    # NOTE: CAVEAT!
    # !!! We need to convert the coordinates from the benchmark planner format to the output format. !!!
    #
    # Warning:
    # If you use this function, make sure you do not rotate the image in _doc_to_visual().
    #
    #              #---------------+   --
    #              |   * (P1)      |    |
    #              |               |    | -> image_size_height
    #              |               |    |
    #              &---------------+   --
    #
    # #: array space origin (upper-left corner)
    # &: image space origin (lower-left corner)
    # The point * can be written in array space as P1 and in image space as P1':
    #   - P1: (idx_dim0, idx_dim1)
    #   - P1': (x_1, y_1) = (idx_dim1, image_size_height - idx_dim0)
    # --------------------------------------

    # [!] Get the raw image shape (before explicit and implicit resizing)
    _, img_raw_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)
    raw_img_h, raw_img_w = img_raw_2d.shape

    # NOTE: keys should be in the "COT_TEMPLATE_DISTANCE" or "COT_TEMPLATE_ANGLE" from medvision_bm.sft.sft_prompts
    if metric_type == "distance":
        # Gather values to fill in the CoT template
        landmarks_coords = _get_landmarks_coords(
            doc, lms
        )  # this coordinates are indices in array space
        # Convert to relative coordinates in image space
        x1_relative_coord = landmarks_coords["landmark_" + lms[0]][1] / raw_img_w
        y1_relative_coord = 1.0 - (
            landmarks_coords["landmark_" + lms[0]][0] / raw_img_h
        )
        x2_relative_coord = landmarks_coords["landmark_" + lms[1]][1] / raw_img_w
        y2_relative_coord = 1.0 - (
            landmarks_coords["landmark_" + lms[1]][0] / raw_img_h
        )
        # Recalculate the distance based on the adjusted pixel size and resized image size
        distance = np.sqrt(
            (
                (x2_relative_coord - x1_relative_coord)
                * resized_img_w
                * adjusted_pixel_width
            )
            ** 2
            + (
                (y2_relative_coord - y1_relative_coord)
                * resized_img_h
                * adjusted_pixel_height
            )
            ** 2
        )
        # Prepare values to fill in the CoT template
        values_dict = {
            "metric_type": "distance",
            "<landmark 1>": p1_name,
            "<landmark 2>": p2_name,
            "<x1>": f"{x1_relative_coord:.3f}",
            "<y1>": f"{y1_relative_coord:.3f}",
            "<x2>": f"{x2_relative_coord:.3f}",
            "<y2>": f"{y2_relative_coord:.3f}",
            "<pixel_width>": f"{adjusted_pixel_width:.3f}",
            "<pixel_height>": f"{adjusted_pixel_height:.3f}",
            "<image_width>": f"{resized_img_w}",
            "<image_height>": f"{resized_img_h}",
            "<distance>": f"{distance:.3f}",
        }

    elif metric_type == "angle":
        # Gather values to fill in the CoT template
        line1_landmarks_coords = _get_landmarks_coords(
            doc, line1_lms
        )  # this coordinates are indices in array space
        line2_landmarks_coords = _get_landmarks_coords(
            doc, line2_lms
        )  # this coordinates are indices in array space
        # Convert to relative coordinates in image space
        x1_line1_relative_coord = (
            line1_landmarks_coords["landmark_" + line1_lms[0]][1] / raw_img_w
        )
        y1_line1_relative_coord = 1.0 - (
            line1_landmarks_coords["landmark_" + line1_lms[0]][0] / raw_img_h
        )
        x2_line1_relative_coord = (
            line1_landmarks_coords["landmark_" + line1_lms[1]][1] / raw_img_w
        )
        y2_line1_relative_coord = 1.0 - (
            line1_landmarks_coords["landmark_" + line1_lms[1]][0] / raw_img_h
        )
        x1_line2_relative_coord = (
            line2_landmarks_coords["landmark_" + line2_lms[0]][1] / raw_img_w
        )
        y1_line2_relative_coord = 1.0 - (
            line2_landmarks_coords["landmark_" + line2_lms[0]][0] / raw_img_h
        )
        x2_line2_relative_coord = (
            line2_landmarks_coords["landmark_" + line2_lms[1]][1] / raw_img_w
        )
        y2_line2_relative_coord = 1.0 - (
            line2_landmarks_coords["landmark_" + line2_lms[1]][0] / raw_img_h
        )
        # Recalculate the angle based on the adjusted pixel size and resized image size
        v1 = np.array(
            [
                (x2_line1_relative_coord - x1_line1_relative_coord)
                * resized_img_w
                * adjusted_pixel_width,
                (y2_line1_relative_coord - y1_line1_relative_coord)
                * resized_img_h
                * adjusted_pixel_height,
            ]
        )
        v2 = np.array(
            [
                (x2_line2_relative_coord - x1_line2_relative_coord)
                * resized_img_w
                * adjusted_pixel_width,
                (y2_line2_relative_coord - y1_line2_relative_coord)
                * resized_img_h
                * adjusted_pixel_height,
            ]
        )
        abs_cos_theta = np.abs(np.dot(v1, v2)) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )
        angle = np.arccos(abs_cos_theta)
        angle_degree = np.degrees(angle)
        # Prepare values to fill in the CoT template
        values_dict = {
            "metric_type": "angle",
            "<landmark 1>": line1_p1_name,
            "<landmark 2>": line1_p2_name,
            "<landmark 3>": line2_p1_name,
            "<landmark 4>": line2_p2_name,
            "<x1_line1>": f"{x1_line1_relative_coord:.3f}",
            "<y1_line1>": f"{y1_line1_relative_coord:.3f}",
            "<x2_line1>": f"{x2_line1_relative_coord:.3f}",
            "<y2_line1>": f"{y2_line1_relative_coord:.3f}",
            "<x1_line2>": f"{x1_line2_relative_coord:.3f}",
            "<y1_line2>": f"{y1_line2_relative_coord:.3f}",
            "<x2_line2>": f"{x2_line2_relative_coord:.3f}",
            "<y2_line2>": f"{y2_line2_relative_coord:.3f}",
            "<pixel_width>": f"{adjusted_pixel_width:.3f}",
            "<pixel_height>": f"{adjusted_pixel_height:.3f}",
            "<image_width>": f"{resized_img_w}",
            "<image_height>": f"{resized_img_h}",
            "<Ax>": f"{v1[0]:.3f}",
            "<Ay>": f"{v1[1]:.3f}",
            "<Bx>": f"{v2[0]:.3f}",
            "<By>": f"{v2[1]:.3f}",
            "<angle>": f"{angle:.3f}",
            "<angle_degree>": f"{angle_degree:.3f}",
        }
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")
    # ------------------------------------------------------------------

    return question, values_dict


def _doc_to_target_AngleDistanceTask(doc):
    """Get ground truth biometrics."""
    biometric_profile = doc["biometric_profile"]
    return biometric_profile["metric_value"]


def _doc_to_target_AngleDistanceTask_CoT(doc, values_dict):
    from medvision_bm.sft.sft_prompts import COT_TEMPLATE_ANGLE, COT_TEMPLATE_DISTANCE

    biometric_profile = doc["biometric_profile"]
    metric_type = biometric_profile["metric_type"]
    if metric_type == "angle":
        cot_template = COT_TEMPLATE_ANGLE
    elif metric_type == "distance":
        cot_template = COT_TEMPLATE_DISTANCE
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    # Prepare values to fill in the CoT template
    target_outputs_cot = fill_in_template(cot_template, values_dict)

    return target_outputs_cot


def img_proccessor_nii2png_save2disk(example, new_shape_hw=None):
    # Save tmp PNGs next to the source image inside a tmp_prepared_png folder
    img_path = example["image_file"]
    slice_dim = example["slice_dim"]
    slice_idx = example["slice_idx"]
    png_basename = Path(img_path).name.split(".", 1)[0]
    png_dir = os.path.join(os.path.dirname(img_path), "tmp_prepared_png")

    # Fast path: with an explicit resize the final size is known without touching
    # the NIfTI, so a previously generated PNG can be reused (PNG content depends
    # only on the source volume). Verify it fully decodes first — writes used to
    # be non-atomic, so a file may be a truncated leftover from a killed run.
    if new_shape_hw is not None:
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}_resized-wh-{new_shape_hw[1]}x{new_shape_hw[0]}.png"
        png_path = os.path.join(png_dir, png_filename)
        if os.path.exists(png_path):
            try:
                with Image.open(png_path) as _img:
                    _img.load()
                return [png_path]
            except Exception:
                pass  # corrupt/truncated: fall through and regenerate

    # Process image: read from nii.gz file and extract 2D slice
    pil_img = _doc_to_visual(example, new_shape_hw)[0]

    # NOTE: The size of Pillow image is given as a 2-tuple (width, height).
    imgsize_w, imgsize_h = pil_img.size
    if new_shape_hw is not None:
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}_resized-wh-{imgsize_w}x{imgsize_h}.png"
    else:
        png_filename = f"{png_basename}_dim{slice_dim}_slice{slice_idx}_original-wh-{imgsize_w}x{imgsize_h}.png"

    png_path = os.path.join(png_dir, png_filename)
    os.makedirs(png_dir, exist_ok=True)
    # Atomic write: existing files are trusted by the fast path above, so a
    # killed process must never leave a truncated PNG under the final name.
    tmp_png_path = f"{png_path}.tmp{os.getpid()}"
    pil_img.save(tmp_png_path, format="PNG")
    os.replace(tmp_png_path, png_path)
    return [png_path]


def img_proccessor_nii2png_save2dataset(example, new_shape_hw=None):
    # 1. Get the PIL Image object from your function
    image_obj = _doc_to_visual(example, new_shape_hw)[0]

    # 2. Save the image to a BytesIO buffer in PNG format
    img_byte_arr = io.BytesIO()
    image_obj.save(img_byte_arr, format="PNG")

    # 3. Store as a new Image opened from the in-memory bytes
    # This ensures the image data is fully loaded and "detached" from disk
    image_data = [Image.open(io.BytesIO(img_byte_arr.getvalue()))]
    return image_data


def _format_data_AngleDistanceTask(
    example,
    model_name,
    model_hf,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    target_str = str(_doc_to_target_AngleDistanceTask(example))
    prompt = _doc_to_text_AngleDistanceTask(example, model_name, model_hf, new_shape_hw)

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
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )

    return example


def _format_data_AngleDistanceTask_CoT(
    example,
    model_name,
    model_hf,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    prompt, values_dict = _doc_to_text_AngleDistanceTask_CoT(
        example, model_name, model_hf, new_shape_hw
    )
    target_str = _doc_to_target_AngleDistanceTask_CoT(example, values_dict)

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
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )
    return example


def _doc_to_text_TumorLesionTask(doc, model_name, model_hf, new_shape_hw=None):
    """Convert document to text."""
    from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
        _normalize_metric_unit,
        get_resized_img_shape,
    )
    from medvision_bm.sft.sft_prompts import FORMAT_PROMPT_TUMOR_LESION_SIZE

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
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
    pixel_size_hw, img_2d_raw = _load_resize_nifti_2d(
        img_path, slice_dim, slice_idx, new_shape_hw
    )  # explicit resizing
    img_shape = img_2d_raw.shape

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # Get resized image shape
    img_shape_resized, img_shape_content_hw = get_resized_img_shape(
        model_name, img_2d_raw, {"model_hf": model_hf}
    )  # implicit/dynamic resizing from VLM

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_resized
    resize_ratio_h = img_shape_content_hw[0] / original_height
    resize_ratio_w = img_shape_content_hw[1] / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

    if image_description != "" and image_description is not None:
        image_prompt = ": " + image_description
    else:
        image_prompt = ""

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image{image_prompt}, "
        f"estimate the major and minor axis lengths of the ellipse enclosing the {label_name}, in {metric_unit}.\n"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_TUMOR_LESION_SIZE}"
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


def _get_landmarks_coords(example, landmark_keys):
    """
    This function extracts the 2D coordinates of specified landmarks from the landmark file
    corresponding to the given slice dimension and slice index in the input case "example".

    This is based on the landmark file structure in the MedVision dataset:
        - HF: YongchengYAO/MedVision
        - commit: 6a774bf5b378788f1ca5447e4d593c431b81bb98

    CAUTION:
        - Changing the landmark file format in the dataset may break this function.

    Tips: To better understand the structure of landmarks in defferent tasks, check corresponding json.gz files in MedVision dataset.
        e.g.:
        - tumor/lesion size tasks: landmark_dict["slice_landmarks_x"][0]["landmarks"] is list of dict -- the length of list is the number of lesions in that 2D image slice
        - angle/distance tasks: landmark_dict["slice_landmarks_x"][0]["landmarks"] is dict

    Note for developers: read the NOTE comments in the code below for more details.
    """

    # Used in reasoning process reward
    landmark_data = _load_json(
        example["landmark_file"]
    )  # dict of keys: slice_landmarks_x, slice_landmarks_y, slice_landmarks_z
    slice_dim = example["slice_dim"]
    if slice_dim == 0:
        lm_key = "slice_landmarks_x"
    elif slice_dim == 1:
        lm_key = "slice_landmarks_y"
    elif slice_dim == 2:
        lm_key = "slice_landmarks_z"
    slice_idx = example["slice_idx"]
    lm_slice_ls = landmark_data[lm_key]  # list of dicts

    # Find the entry for the specified slice_idx
    matched_entries = [itm for itm in lm_slice_ls if itm.get("slice_idx") == slice_idx]

    # NOTE: Merge all landmarks from matched entries into a single dict
    # ------
    # e.g.,
    # matched_entries = [
    #     {
    #         "slice_idx": 128,
    #         "landmarks": {"P1": [...], "P2": [...]}
    #     },
    #     {
    #         "slice_idx": 128,
    #         "landmarks": {"P3": [...], "P4": [...]}
    #     }
    # ]
    # Result:
    # lm_slice = {
    #     "slice_idx": 128,
    #     "landmarks": {"P1": [...], "P2": [...], "P3": [...], "P4": [...]}
    # }
    # ------
    if matched_entries:
        lm_slice = {"slice_idx": slice_idx, "landmarks": {}}
        for entry in matched_entries:
            # ---
            # NOTE: For compatibility with landmark file formats from different datasets/tasks
            # In the current MedVision dataset format,
            # the only case where "landmarks" is a list is for tumor/lesion size tasks,
            # where there can be multiple lesions in the same 2D slice.
            # Since we filter out cases with multiple lesions in our study,
            # we directly extract the first element of the list here.
            # ---
            entry_landmarks = (
                entry.get("landmarks")[0]
                if isinstance(entry.get("landmarks"), list)
                else entry.get("landmarks")
            )
            lm_slice["landmarks"].update(entry_landmarks)
    else:
        raise ValueError(
            f"No landmark entry found for slice_dim: {slice_dim} and slice_idx: {slice_idx}"
        )

    landmark_coords = {}
    for p_name in landmark_keys:
        coor_3d = lm_slice["landmarks"][p_name]
        coor_2d = _extract_3dCoor_to_2dCoor(coor_3d, slice_dim)
        key = f"landmark_{p_name}"
        landmark_coords[key] = coor_2d

    return landmark_coords


def _doc_to_text_TumorLesionTask_CoT(doc, model_name, model_hf, new_shape_hw=None):
    """Convert document to text."""
    from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
        _normalize_metric_unit,
        get_resized_img_shape,
    )
    from medvision_bm.sft.sft_prompts import (
        COT_INSTRUCT_TL_NORM,
        FORMAT_PROMPT_TL_REASONING,
    )

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
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

    # [!] Read NIfTI image with explicit resizing to the specified new_shape_hw (if not None)
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    pixel_size_hw, img_explicit_resize_2d = _load_resize_nifti_2d(
        img_path, slice_dim, slice_idx, new_shape_hw
    )
    img_shape = img_explicit_resize_2d.shape

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # [!] Get resized image shape (implicit resizing from VLM)
    img_shape_implicit_resize, img_shape_content_hw = get_resized_img_shape(
        model_name, img_explicit_resize_2d, {"model_hf": model_hf}
    )

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_implicit_resize
    resize_ratio_h = img_shape_content_hw[0] / original_height
    resize_ratio_w = img_shape_content_hw[1] / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

    if image_description != "" and image_description is not None:
        image_prompt = ": " + image_description
    else:
        image_prompt = ""

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image{image_prompt}, "
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

    # ------------------------------------------------------------------
    # NOTE: CAVEAT!
    # !!! We need to convert the coordinates from the benchmark planner format to the output format. !!!
    #
    # Warning:
    # If you use this function, make sure you do not rotate the image in _doc_to_visual().
    #
    #              #---------------+   --
    #              |   * (P1)      |    |
    #              |               |    | -> image_size_height
    #              |               |    |
    #              &---------------+   --
    #
    # #: array space origin (upper-left corner)
    # &: image space origin (lower-left corner)
    # The point * can be written in array space as P1 and in image space as P1':
    #   - P1: (idx_dim0, idx_dim1)
    #   - P1': (x_1, y_1) = (idx_dim1, image_size_height - idx_dim0)
    # --------------------------------------
    # Gather values to fill in the CoT template

    # [!] Get the raw image shape (before explicit and implicit resizing)
    _, img_raw_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)
    raw_img_h, raw_img_w = img_raw_2d.shape

    # NOTE: landmarks_coords are calculated based on the original image size (raw_image_2d) before resizing
    landmarks_coords = _get_landmarks_coords(doc, ["P1", "P2", "P3", "P4"])

    # Caveat:
    # 1. x is the width direction, y is the height direction
    # 2. use relative coordinates
    # 3. (minor;optional) recalculate the major and minor axis lengths based on adjusted pixel size and resized image size; marginal error may exist compared to the original values due to rounding errors
    x1_major = landmarks_coords["landmark_P1"][1] / raw_img_w
    y1_major = 1 - landmarks_coords["landmark_P1"][0] / raw_img_h
    x2_major = landmarks_coords["landmark_P2"][1] / raw_img_w
    y2_major = 1 - landmarks_coords["landmark_P2"][0] / raw_img_h
    x1_minor = landmarks_coords["landmark_P3"][1] / raw_img_w
    y1_minor = 1 - landmarks_coords["landmark_P3"][0] / raw_img_h
    x2_minor = landmarks_coords["landmark_P4"][1] / raw_img_w
    y2_minor = 1 - landmarks_coords["landmark_P4"][0] / raw_img_h
    major_axis_length = math.sqrt(
        ((x2_major - x1_major) * resized_img_w * adjusted_pixel_width) ** 2
        + ((y2_major - y1_major) * resized_img_h * adjusted_pixel_height) ** 2
    )
    minor_axis_length = math.sqrt(
        ((x2_minor - x1_minor) * resized_img_w * adjusted_pixel_width) ** 2
        + ((y2_minor - y1_minor) * resized_img_h * adjusted_pixel_height) ** 2
    )

    # NOTE: keys should be in the "COT_TEMPLATE_TL_NORM" from medvision_bm.sft.sft_prompts
    values_dict = {
        "<label>": label_name,
        "<image_description>": image_description,
        "<image_width>": f"{resized_img_w}",
        "<image_height>": f"{resized_img_h}",
        "<pixel_width>": f"{adjusted_pixel_width:.3f}",
        "<pixel_height>": f"{adjusted_pixel_height:.3f}",
        "<metric_unit>": metric_unit,
        "<x1_major>": f"{x1_major:.3f}",
        "<y1_major>": f"{y1_major:.3f}",
        "<x2_major>": f"{x2_major:.3f}",
        "<y2_major>": f"{y2_major:.3f}",
        "<x1_minor>": f"{x1_minor:.3f}",
        "<y1_minor>": f"{y1_minor:.3f}",
        "<x2_minor>": f"{x2_minor:.3f}",
        "<y2_minor>": f"{y2_minor:.3f}",
        "<major_axis_length>": f"{major_axis_length:.3f}",
        "<minor_axis_length>": f"{minor_axis_length:.3f}",
    }
    # ------------------------------------------------------------------

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
    from medvision_bm.sft.sft_prompts import COT_TEMPLATE_TL_NORM

    # Prepare values to fill in the CoT template
    target_outputs_cot = fill_in_template(COT_TEMPLATE_TL_NORM, values_dict)

    return target_outputs_cot


def _format_data_TumorLesionTask(
    example,
    model_name,
    model_hf,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    target = _doc_to_target_TumorLesionTask(example)
    target_str = ", ".join([f"{value:.3f}" for value in target])
    prompt, _ = _doc_to_text_TumorLesionTask(
        example, model_name, model_hf, new_shape_hw
    )

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
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )

    return example


def _format_data_TumorLesionTask_CoT(
    example,
    model_name,
    model_hf,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    """
    Format data for TumorLesionTask with CoT reasoning.
    Compared to the non-CoT version, this function:
    1. Uses a different prompt template that includes reasoning steps.
    2. Returns a target string that includes reasoning steps.
    """
    prompt, values_dict = _doc_to_text_TumorLesionTask_CoT(
        example, model_name, model_hf, new_shape_hw
    )
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
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )

    return example


def _doc_to_text_DetectionTask(doc):
    """Convert document to text."""
    from medvision_bm.sft.sft_prompts import FORMAT_PROMPT_BOX_COORDINATES

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
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

    if image_description != "" and image_description is not None:
        image_prompt = ": " + image_description
    else:
        image_prompt = ""

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image{image_prompt}, "
        f"return the coordinates of the lower-left and upper-right corner of the bounding box for the {label_name}.\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_BOX_COORDINATES}"
    )
    return question


def _doc_to_text_DetectionTask_CoT(doc):
    """Convert document to text."""
    from medvision_bm.sft.sft_prompts import (
        COT_INSTRUCT_DETECTION,
        FORMAT_PROMPT_DETECTION_REASONING,
    )

    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
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

    if image_description != "" and image_description is not None:
        image_prompt = ": " + image_description
    else:
        image_prompt = ""

    # Question
    question = (
        f"Task:\n"
        f"Given the input medical image{image_prompt}, "
        f"return the coordinates of the lower-left and upper-right corner of the bounding box for the {label_name}.\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_DETECTION_REASONING}\n"
        f"Reasoning steps:\n"
        f"{COT_INSTRUCT_DETECTION}\n"
        f"Follow the reasoning steps to get the final answer in the required format."
    )

    # Prepare values_dict
    # NOTE: the keys must be in the COT_TEMPLATE_DETECTION from medvision_bm.sft.sft_prompts
    coor0_w, coor0_h, coor1_w, coor1_h = _doc_to_target_DetectionTask(doc)
    values_dict = {
        "<label_name>": label_name,
        "<coor0_w>": f"{coor0_w:.3f}",
        "<coor0_h>": f"{coor0_h:.3f}",
        "<coor1_w>": f"{coor1_w:.3f}",
        "<coor1_h>": f"{coor1_h:.3f}",
    }
    return question, values_dict


def _doc_to_target_DetectionTask(doc):
    """
     Get bounding box coordinates.

     Definition of the output (target) bounding box coordinates:
     1.  The origin of the coordinates is at the [lower-left corner] of the image.
     2.  The first two numbers are the coordinates of the [lower-left] corner and
         the last two numbers are the coordinates of the [upper-right] corner of the bounding box.
     3.  The coordinates are expected to be in the format of [coor0_dim1, coor0_dim0, coor1_dim1, coor1_dim0], where:
         - coor0: lower-left corner of the bounding box
         - coor1: upper-right corner of the bounding box
         - dim0: the first dimension of the image (height)
         - dim1: the second dimension of the image (width)

     Definition of bounding box coordinates in the benchmark planner:
     1. The origin of the coordinates is at the [top-left corner] of the image.
     2. The first two numbers are the coordinates of the [upper-left] corner and
        the last two numbers are the coordinates of the [lower-right] corner of the bounding box.

     That is,
         - in the benchmark planner, corrdinates are: [idx_dim0, idx_dim1]
         - target coordinates are in the format of [idx_width, idx_height] in image space

     NOTE: CAVEAT!
     !!! We need to convert the coordinates from the benchmark planner format to the output format. !!!

     Warning:
     If you use this function, make sure you do not rotate the image when extracting 2D slices from 3D NIfTI images,
     such as in _doc_to_visual().

     In summary, the conversion involves:
     Based on the upper-left and lower-right corner coordinates (P1 & P2) in the format of array indices [idx_dim0, idx_dim1] from the benchmark planner,
     we calculate the lower-left and upper-right corner coordinates (P1' & P2') in the format of image space indices [idx_width, idx_height] as follows:

         #-----------------------------+
         |   * (P1)         @ (P2')    |
         |                             |
         |                             |
         |                             |
         |                             |
         |                             |
         |   @ (P1')        * (P2)     |
         &-----------------------------+

         #---------(idx_dim1)----------+
         |                             |
         |                             |
         |                             |
      (idx_dim0)   array space         |
         |                             |
         |                             |
         |                             |
         +-----------------------------+

         +---------(idx_width)---------+
         |                             |
         |                             |
         |                             |
    (idx_height)   image space         |
         |                             |
         |                             |
         |                             |
         &-----------------------------+

     #: array space origin (upper-left corner)
     &: image space origin (lower-left corner)
     P1: the lower corner in array space (the min_coords in benchmark planner)
     P2: the upper corner in array space (the max_coords in benchmark planner)
     P1': the lower corner in image space
     P2': the upper corner in image space

     ------
     NOTE for developers and future versions:
     Rotating the image counter-clockwise by 90 degrees would avoid the need for coordinate conversion.
     ------
    """
    # Read NIfTI image
    img_size = doc["image_size_2d"]
    imgsize_h, imgsize_w = img_size
    # Convert the coordinates from the benchmark planner format to the output format.
    bm_coor0_h, bm_coor0_w = doc["bounding_boxes"]["min_coords"][0]
    bm_coor1_h, bm_coor1_w = doc["bounding_boxes"]["max_coords"][0]
    img_coor0_w = bm_coor0_w
    img_coor0_h = imgsize_h - bm_coor1_h
    img_coor1_w = bm_coor1_w
    img_coor1_h = imgsize_h - bm_coor0_h
    # Convert bounding box coordinates to relative coordinates
    coor0_h = img_coor0_h / imgsize_h
    coor0_w = img_coor0_w / imgsize_w
    coor1_h = img_coor1_h / imgsize_h
    coor1_w = img_coor1_w / imgsize_w
    # Return the relative coordinates in the image space (the origin is at the lower-left corner)
    return [coor0_w, coor0_h, coor1_w, coor1_h]


def _doc_to_target_DetectionTask_CoT(values_dict):
    from medvision_bm.sft.sft_prompts import COT_TEMPLATE_DETECTION

    # Prepare values to fill in the CoT template
    target_outputs_cot = fill_in_template(COT_TEMPLATE_DETECTION, values_dict)

    return target_outputs_cot


# NOTE: model_name is not used, but must be kept for consistent function signature -- check usage in prepare_dataset()
def _format_data_DetectionTask(
    example,
    model_name=None,
    model_hf=None,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    # Since reshaping does not affect relative coordinates, we do not pass new_shape_hw to _doc_to_text_DetectionTask()
    target_coords = _doc_to_target_DetectionTask(example)
    target_str = f"{target_coords[0]:.3f}, {target_coords[1]:.3f}, {target_coords[2]:.3f}, {target_coords[3]:.3f}"
    prompt = _doc_to_text_DetectionTask(example)

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
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )

    return example


# NOTE: model_name and model_hf is not used, but must be kept for consistent function signature -- check usage in prepare_dataset()
def _format_data_DetectionTask_CoT(
    example,
    model_name=None,
    model_hf=None,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    # Since reshaping does not affect relative coordinates, we do not pass new_shape_hw to _doc_to_text_DetectionTask()
    prompt, values_dict = _doc_to_text_DetectionTask_CoT(example)
    target_str = _doc_to_target_DetectionTask_CoT(values_dict)

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
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )

    # [Recommended] Save processed images to PNG files on disk
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )

    return example


def _load_single_dataset(
    dataset_hf_id,
    dataset_name,
    config,
    split,
    limit=None,
    download_mode="reuse_dataset_if_exists",
):
    """
    Load a single dataset configuration with improved error handling.

    Args:
        dataset_hf_id (str): Hugging Face dataset ID.
        dataset_name (str | None): Name to assign to the dataset (added as a column).
        config (str): Configuration name.
        split (str): Dataset split to load.
        limit (int | None): If specified, limit the number of samples to this number.
        download_mode (str): "reuse_dataset_if_exists" (default), "reuse_cache_if_exists", "force_redownload"

    Returns:
        Dataset: Loaded Hugging Face dataset.
    """
    try:
        print(
            f"\n[Info] Loading dataset:\nHF Dataset ID: {dataset_hf_id}\nConfiguration: {config}"
        )

        # Add timeout and retry logic for dataset loading
        max_retries = 5
        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    dataset_hf_id,
                    name=config,
                    trust_remote_code=True,
                    split=split,
                    streaming=False,
                    download_mode=download_mode,
                )
                if limit is not None and limit > 0 and len(ds) > limit:
                    ds = ds.select(range(limit))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    print(
                        f"[Warning] Attempt {attempt + 1} failed for {config}, retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    raise

        print(
            f"\n[Info] Successfully loaded {len(ds)} samples from config {config} (dataset: {dataset_name})"
        )
        return ds

    except Exception:
        print(
            f"[Error] Failed to load dataset:\nHF Dataset ID: {dataset_hf_id}\nConfiguration:{config}"
        )
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(
            f"[Error] Failed to load dataset:\nHF Dataset ID: {dataset_hf_id}\nConfiguration:{config}"
        )


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

    normalized = [
        ds.map(pad_dict_keys, desc="Normalizing dict columns") for ds in datasets_list
    ]
    return normalized


def safe_concatenate_datasets(datasets_list):
    datasets_list = safe_concat_align_top_keys(datasets_list, fill_value=None)
    datasets_list = safe_concat_align_dict_keys(
        datasets_list, dict_cols=None, fill_value=None
    )
    combined_dataset = concatenate_datasets(datasets_list)
    return combined_dataset


def group_train_test_split(
    dataset, group_column, test_size, seed=None, stratify_column=None
):
    """
    Splits a HF Dataset into train and validation sets ensuring samples with the
    same value in 'group_column' are in the same split.

    Args:
        dataset: The HF Dataset to split.
        group_column: The column name to group by (e.g., 'image_file').
        test_size: If float < 1.0, represents fraction of *samples* to aim for.
                   If int >= 1, represents exact number of *samples* to aim for.
        seed: Random seed for shuffling.
        stratify_column: Optional column name to stratify by (e.g., 'dataset_name').
            A stratum is the set of volumes belonging to one unique value of this
            column (e.g., all volumes from 'BraTS24' form one stratum, all volumes
            from 'AMOS22' form another). When provided, volumes are interleaved
            round-robin across strata — one volume per stratum per round — before
            the greedy allocation loop runs. This guarantees every stratum
            contributes at least one volume to val before any stratum gets a second,
            preventing a few large-volume datasets from monopolising the val quota.

    Returns:
        DatasetDict containing 'train' and 'validation'.
    """
    # 1. Group indices by the group_column (e.g., path to 3D volume)
    # This might load the column into memory, which is usually fine for string paths
    groups = dataset[group_column]
    group_to_indices = defaultdict(list)
    for idx, g_val in enumerate(groups):
        group_to_indices[g_val].append(idx)

    unique_groups = list(group_to_indices.keys())

    # 2. Shuffle groups (and optionally interleave across strata for dataset diversity)
    rng = np.random.default_rng(seed)

    if stratify_column is not None:
        # Map each volume (group) to its stratum value
        stratum_vals = dataset[stratify_column]
        group_to_stratum = {}
        for idx, g_val in enumerate(groups):
            if g_val not in group_to_stratum:
                group_to_stratum[g_val] = stratum_vals[idx]

        # Bucket volumes by stratum, shuffle within each bucket
        stratum_to_groups = defaultdict(list)
        for g in unique_groups:
            stratum_to_groups[group_to_stratum[g]].append(g)
        for sg in stratum_to_groups.values():
            rng.shuffle(sg)

        # Round-robin interleave: one volume per stratum per round
        # This guarantees every stratum contributes before any stratum gets a 2nd volume
        strata_iters = [iter(v) for v in stratum_to_groups.values()]
        unique_groups = []
        while strata_iters:
            next_iters = []
            for it in strata_iters:
                try:
                    unique_groups.append(next(it))
                    next_iters.append(it)
                except StopIteration:
                    pass
            strata_iters = next_iters
    else:
        rng.shuffle(unique_groups)

    # 3. Determine target sample count
    total_samples = len(dataset)
    if isinstance(test_size, float) and test_size < 1.0:
        target_test_samples = int(total_samples * test_size)
    else:
        target_test_samples = int(test_size)

    # 4. Allocate groups to validation until target count is reached
    val_indices = []
    current_val_samples = 0

    # Split index for groups
    split_idx = 0
    for i, g_val in enumerate(unique_groups):
        indices = group_to_indices[g_val]

        # If adding this group exceeds target significantly, we might skip (simple greedy here)
        # For now, we accumulate until we hit or exceed the target slightly to ensure adequate val size
        val_indices.extend(indices)
        current_val_samples += len(indices)

        if current_val_samples >= target_test_samples:
            split_idx = i + 1
            break

    # The rest go to train
    train_groups = unique_groups[split_idx:]
    train_indices = []
    for g_val in train_groups:
        train_indices.extend(group_to_indices[g_val])

    # 5. Create splits
    # optional: shuffle indices within the splits so they aren't ordered by volume
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return DatasetDict(
        {
            "train": dataset.select(train_indices),
            "validation": dataset.select(val_indices),
        }
    )


def load_split_limit_dataset(
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    num_workers_concat_datasets=4,
    tag_ds=None,
    download_mode="reuse_dataset_if_exists",
):
    """Load MedVision tasks, concatenate them, and split into train/validation.

    Reads the task list from ``tasks_list_json_path`` and loads each task's
    ``_Train`` split in parallel (falling back to single-threaded loading when
    any dataset was just downloaded, to avoid cache conflicts). The per-task
    datasets are concatenated in the deterministic JSON order (not arrival
    order) so the seeded shuffle and split downstream stay reproducible, then
    split into train and validation sets grouped by ``image_file`` (to prevent
    3D-volume leakage) and stratified by ``dataset_name``.

    Args:
        tasks_list_json_path (str): Path to the JSON file whose keys are the
            task names to load.
        limit_train_sample (int): Cap on training samples after concatenation.
            Use a value < 0 for no limit or > 0 for a fixed cap; 0 is rejected.
        limit_val_sample (int): Target size of the validation split; must be > 0.
        num_workers_concat_datasets (int): Requested worker processes for
            parallel loading; clamped to the CPU count and task count. Defaults
            to 4.
        tag_ds (str): Tag embedded in task names (``<dataset_name>_<tag_ds>``),
            used to recover the dataset name from each task. Required.
        download_mode (str): Passed through to the dataset loader. Defaults to
            ``"reuse_dataset_if_exists"``.

    Returns:
        DatasetDict: A dict with ``"train"`` and ``"validation"`` splits.

    Raises:
        AssertionError: If ``limit_val_sample`` is not > 0,
            ``limit_train_sample`` is 0, ``tag_ds`` is None, or
            ``MedVision_DATA_DIR`` is unset.
        RuntimeError: If any task fails to load.
    """
    # NOTE:
    # - limit_val_sample must be greater than 0 to ensure validation set is not empty.
    # - limit_train_sample can be <0 (no limit) or >0 (limited training set).
    assert limit_val_sample > 0, "\n [Error] limit_val_sample must be greater than 0."
    assert (
        limit_train_sample != 0
    ), "\n [Error] limit_train_sample cannot be 0. Use <0 for no limit or >0 for limited training set."

    # Early assertions
    assert (
        tag_ds is not None
    ), "\n [Error] tag_ds (i.e., the string in tasks names: <dataset_name>_<tag_ds>) must be provided."

    print(f"\n[Info] Starting dataset preparation from {tasks_list_json_path}")

    # Load tasks list from JSON file
    with open(tasks_list_json_path, "r") as f:
        tasks_dict = json.load(f)
    # NOTE: Detection task lists may carry eval-side task names ("_BoxCoordinate_",
    # e.g. the dataset-info AllSlices lists); the HF dataset only has "_BoxSize_"
    # configs, so rename before building config names.
    if tag_ds == "BoxSize":
        tasks_dict = {
            task.replace("_BoxCoordinate_", "_BoxSize_"): n
            for task, n in tasks_dict.items()
        }
    tasks = list(tasks_dict.keys())

    print(f"[Info] Found {len(tasks)} tasks to process")

    # Reduce parallelism to avoid memory issues - use fewer workers
    available_cpus = get_cgroup_limited_cpus()
    concat_workers = min(num_workers_concat_datasets, available_cpus, len(tasks))

    # NOTE: Force single-threaded loading for new datasets to avoid conflicts, otherwise errors may occur.
    data_dir = os.environ.get("MedVision_DATA_DIR")
    assert (
        data_dir is not None
    ), "\n [Error] MedVision_DATA_DIR environment variable must be set."
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

    task_to_ds = {}
    failed_tasks = []

    # Process datasets with controlled parallelism
    with ProcessPoolExecutor(max_workers=concat_workers) as executor:
        # Load training splits for all tasks in parallel
        # ------
        # NOTE: This is specific to the MedVision dataset and configs
        # For MedVision dataset:
        # - Config name for training set is in the format of "{task}_Train", while the test set is "{task}_Test"
        # - Dataset name can be extracted from task name (e.g., part before f"_{tag_ds}"): task.split(f"_{tag_ds}")[0]
        # ------

        # NOTE: Although we have arg "limit" in _load_single_dataset(), we do not use it here to limit samples per task.
        # Instead, we limit the total number of training samples after combining all datasets.
        future_to_task = {
            executor.submit(
                _load_single_dataset,
                "YongchengYAO/MedVision",
                task.split(f"_{tag_ds}")[0],
                task + "_Train",
                "train",
                download_mode=download_mode,
            ): task
            for task in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                ds = future.result(timeout=120)  # 2 minute timeout per task
                task_to_ds[task] = ds
                print(f"✓ Completed {task} ({len(task_to_ds)}/{len(tasks)})")

                # Monitor memory usage (cgroup-aware: psutil sees the HOST on pods)
                mem_used_gib, mem_limit_gib, mem_percent = get_cgroup_memory_percent()
                if mem_percent > 80:
                    print(
                        f"⚠️  High cgroup memory: {mem_used_gib:.1f}/{mem_limit_gib:.1f} GiB ({mem_percent:.0f}%)"
                    )

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

    # Combine all datasets.
    # NOTE: Reassemble in the deterministic `tasks` (JSON) order rather than the
    # as_completed() arrival order above, so the concatenated row layout is identical
    # across runs. This keeps the seeded shuffle/split downstream reproducible and
    # prevents crash+resume in the checkpointed parquet builder from re-sharding a
    # different ordering (duplicated/missing samples). All tasks are present here
    # because any failure would have raised above.
    print("\n[Info] Combining datasets...")
    datasets_list = [task_to_ds[task] for task in tasks]
    combined_dataset = concatenate_datasets(datasets_list)
    print(f"[Info] Combined dataset has {len(combined_dataset)} total samples")

    # Clear intermediate datasets to free memory
    del datasets_list, task_to_ds
    gc.collect()

    # Split the dataset into training and validation sets
    print(
        f"\n[Info] Splitting dataset into training and validation (target val size: {limit_val_sample}) sets"
    )

    # Split with group consideration (image_file) to prevent leakage
    print(
        f"\n[Info] Splitting dataset into training and validation (target val size: {limit_val_sample}) keeping 3D volumes grouped."
    )

    # NOTE: "image_file" is a column in the MedVision dataset representing the path to the 3D NIfTI image.
    # TODO: Make group_column configurable if needed.
    dataset = group_train_test_split(
        combined_dataset,
        group_column="image_file",
        test_size=limit_val_sample,
        seed=SEED,
        stratify_column="dataset_name",
    )

    # Limit the number of training and validation samples if specified
    if limit_train_sample > 0 and limit_train_sample < len(dataset["train"]):
        print(
            f"\n[Info][Warning] Limiting training samples to {limit_train_sample} (original: {len(dataset['train'])})"
        )
        dataset["train"] = (
            dataset["train"].shuffle(seed=SEED).select(range(limit_train_sample))
        )
    return dataset


def format_dataset(
    dataset,
    mapping_func,
    mapping_func_args,
    num_workers_format_dataset,
    writer_batch_size=1000,
):
    """Apply a formatting map function to a dataset with bounded parallelism.

    Runs ``dataset.map(mapping_func, ...)`` to convert raw MedVision rows into
    the chat ``messages`` format expected by the SFT trainer. The number of
    worker processes is capped at the cgroup-limited CPU count.

    Args:
        dataset: A HuggingFace ``Dataset`` or ``DatasetDict`` to format.
        mapping_func: The per-example formatting function passed to ``.map()``.
        mapping_func_args (dict): Keyword arguments forwarded to ``mapping_func``
            via ``fn_kwargs``.
        num_workers_format_dataset (int): Requested number of worker processes;
            clamped to the available CPU count.
        writer_batch_size (int): Number of rows buffered before writing to the
            Arrow cache. Defaults to 1000.

    Returns:
        The formatted dataset with the mapping function applied.
    """
    # Format the dataset with parallelism
    # Use conservative parallelism for formatting to avoid OOM
    available_cpus = get_cgroup_limited_cpus()
    format_workers = min(num_workers_format_dataset, available_cpus)
    print(
        f"\n[Info] Formatting dataset with {format_workers} workers (writer_batch_size={writer_batch_size})..."
    )

    def _map_single(ds):
        # Sort rows by source volume so each worker's contiguous shard hits the
        # per-process volume cache in _load_nifti_2d (one .nii.gz decompression
        # per volume instead of per row), then restore the original row order:
        # the seeded shuffles downstream permute positions, so the input order
        # must be unchanged for reproducibility.
        restore_order = "image_file" in ds.column_names and len(ds) > 1
        if restore_order:
            paths = ds["image_file"]
            order = sorted(range(len(paths)), key=paths.__getitem__)
            ds = ds.select(order)
        ds = ds.map(
            mapping_func,
            fn_kwargs=mapping_func_args,
            num_proc=format_workers,
            writer_batch_size=writer_batch_size,
            desc="Formatting dataset",
        )
        if restore_order:
            inverse_order = np.empty(len(order), dtype=np.int64)
            inverse_order[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
            ds = ds.select(inverse_order)
        return ds

    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict(
            {split: _map_single(ds) for split, ds in dataset.items()}
        )
    else:
        dataset = _map_single(dataset)
    return dataset


def clean_dataset(dataset, keys_to_keep):
    """Drop all columns from a dataset except a whitelist of keys.

    Uses ``remove_columns`` — a schema-only operation — so no pass over the
    data is made, keeping the cached rows small before training.

    Args:
        dataset: A HuggingFace ``Dataset`` or ``DatasetDict`` to prune.
        keys_to_keep (list[str]): Column names to retain; all other columns are
            removed.

    Returns:
        The dataset containing only the whitelisted columns.
    """

    def _drop_columns(ds):
        return ds.remove_columns(
            [col for col in ds.column_names if col not in keys_to_keep]
        )

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {split: _drop_columns(ds) for split, ds in dataset.items()}
        )
    return _drop_columns(dataset)


def format_clean_dataset(
    dataset,
    *,
    mapping_func,
    model_family_name,
    base_model_hf,
    num_workers_format_dataset=32,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    """Format a loaded+split MedVision dataset for SFT and prune it to the training columns.

    Second stage of :func:`prepare_dataset`, split out so callers can run
    :func:`load_split_limit_dataset` first, read the true split sizes (e.g. to
    name the prepared-dataset directory), and only then pay for formatting.
    Formatting is a row-preserving ``map``, so the sizes do not change here.

    Args:
        dataset (DatasetDict): Train/validation splits as returned by
            :func:`load_split_limit_dataset`.
        mapping_func, model_family_name, base_model_hf, num_workers_format_dataset,
        process_img, save_processed_img_to_disk, new_shape_hw: As in
            :func:`prepare_dataset`.

    Returns:
        DatasetDict: The formatted splits containing only the retained columns.
    """
    # Format dataset
    mapping_func_args = {
        "model_name": model_family_name,
        "model_hf": base_model_hf,
        "process_img": process_img,
        "save_processed_img_to_disk": save_processed_img_to_disk,
        "new_shape_hw": new_shape_hw,
    }
    dataset = format_dataset(
        dataset=dataset,
        mapping_func=mapping_func,
        mapping_func_args=mapping_func_args,
        num_workers_format_dataset=num_workers_format_dataset,
        writer_batch_size=50,
    )

    # Clean dataset to keep only necessary keys
    # "image_file" is the original NIfTI image path
    keys_to_keep = ["messages", "labels", "image_file", "slice_dim", "slice_idx"]
    if process_img:
        # "processed_images" is the embedded processed image tensor in the dataset (not recommended)
        keys_to_keep.append("processed_images")
    if save_processed_img_to_disk:
        # "image_file_png" is the path to the saved PNG image on disk
        keys_to_keep.append("image_file_png")
    dataset = clean_dataset(dataset, keys_to_keep)

    return dataset


def prepare_dataset(
    *,
    tasks_list_json_path,
    limit_train_sample,
    limit_val_sample,
    mapping_func,
    model_family_name,
    base_model_hf,
    num_workers_concat_datasets=4,
    num_workers_format_dataset=32,
    tag_ds=None,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
    download_mode="reuse_dataset_if_exists",
):
    """Load, format, and prune a MedVision dataset for SFT in one call.

    Combines :func:`load_split_limit_dataset`, :func:`format_dataset`, and
    :func:`clean_dataset`: it loads and splits the tasks, maps each example into
    the chat ``messages`` format via ``mapping_func``, then keeps only the
    columns needed for training.

    Args:
        tasks_list_json_path (str): Path to the JSON file listing the tasks.
        limit_train_sample (int): Training-sample cap (< 0 = no limit,
            > 0 = cap).
        limit_val_sample (int): Target validation-split size; must be > 0.
        mapping_func: Per-example formatting function applied during mapping.
        model_family_name (str): Model family name passed to ``mapping_func`` as
            ``model_name`` (used for image-resize logic).
        base_model_hf (str): HuggingFace model id passed to ``mapping_func`` as
            ``model_hf``.
        num_workers_concat_datasets (int): Worker processes for loading.
            Defaults to 4.
        num_workers_format_dataset (int): Worker processes for formatting.
            Defaults to 32.
        tag_ds (str): Tag embedded in task names; required by the loader.
        process_img (bool): If True, embed processed PNG images in the dataset
            (``processed_images`` column). Not recommended (large cache).
        save_processed_img_to_disk (bool): If True, write processed PNGs to disk
            and store their paths in ``image_file_png``.
        new_shape_hw (tuple[int, int] | None): Optional explicit (height, width)
            to resize slices to before formatting.
        download_mode (str): Passed through to the loader. Defaults to
            ``"reuse_dataset_if_exists"``.

    Returns:
        DatasetDict: Train/validation splits containing only the retained
        columns (``messages``, ``labels``, ``image_file``, ``slice_dim``,
        ``slice_idx``, plus ``processed_images`` and/or ``image_file_png`` when
        enabled).
    """
    # Load and split dataset
    dataset = load_split_limit_dataset(
        tasks_list_json_path=tasks_list_json_path,
        limit_train_sample=limit_train_sample,
        limit_val_sample=limit_val_sample,
        num_workers_concat_datasets=num_workers_concat_datasets,
        tag_ds=tag_ds,
        download_mode=download_mode,
    )

    # Format and prune
    return format_clean_dataset(
        dataset,
        mapping_func=mapping_func,
        model_family_name=model_family_name,
        base_model_hf=base_model_hf,
        num_workers_format_dataset=num_workers_format_dataset,
        process_img=process_img,
        save_processed_img_to_disk=save_processed_img_to_disk,
        new_shape_hw=new_shape_hw,
    )


def recompute_total_max_steps(trainer):
    """Recompute total planned update steps based on global dataset size, world size and desired epochs."""
    args = trainer.args
    grad_accum = args.gradient_accumulation_steps
    epoch = args.num_train_epochs
    per_device_bsz = args.per_device_train_batch_size

    # Prefer accelerate's world size; fallback to Trainer args/env
    state = PartialState()
    world_size = getattr(state, "num_processes", None) or getattr(
        args, "world_size", None
    )
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
            effective_bsz = max(1, per_device_bsz * world_size * grad_accum)
            if getattr(args, "dataloader_drop_last", False):
                steps_per_epoch = max(1, dataset_n // effective_bsz)
            else:
                steps_per_epoch = max(1, math.ceil(dataset_n / effective_bsz))
        except Exception:
            # Fallback if dataset is unsized (e.g., IterableDataset)
            train_dl = trainer.get_train_dataloader()
            steps_per_epoch = max(1, math.ceil(len(train_dl) / grad_accum))

        new_max_steps = steps_per_epoch * epoch

        # Main-process-only logs
        print(f"[Resume] world_size: {world_size}")
        print(f"[Resume] dataset size (global): {dataset_n}")
        print(f"[Resume] per_device_train_batch_size: {per_device_bsz}")
        print(f"[Resume] gradient_accumulation_steps: {grad_accum}")
        print(f"[Resume] num_train_epochs: {epoch}")
        print(f"[Resume] steps_per_epoch (computed): {steps_per_epoch}")
        print(f"[Resume] Recomputed new_max_steps (epochs based): {new_max_steps}")

    # Share the computed value to all processes so every worker uses the exact same max_steps.
    # This prevents mismatched training horizons, inconsistent checkpointing, or hangs in collective ops.
    new_max_steps = broadcast_int_from_main(new_max_steps)
    return new_max_steps


def _make_temperature_sampler_trainer(SFTTrainer):
    """Return a TemperatureSamplerSFTTrainer class bound to the given SFTTrainer base.

    Defined as a factory so the import of SFTTrainer (from trl) stays lazy and
    local to the caller, while the class itself is shared between prepare_trainer()
    and prepare_trainer_fullFT().

    Args:
        SFTTrainer: The base ``trl.SFTTrainer`` class to subclass.

    Returns:
        type: A ``TemperatureSamplerSFTTrainer`` subclass that overrides
        ``_get_train_sampler`` to draw examples with a
        :class:`~torch.utils.data.WeightedRandomSampler`.
    """

    # NOTE: We override only the train sampler behavior while keeping SFTTrainer unchanged.
    # This keeps compatibility with existing trainer setup/checkpoint logic.
    class TemperatureSamplerSFTTrainer(SFTTrainer):
        """SFTTrainer variant that uses temperature-based weighted sampling."""

        def __init__(self, *args, sample_weights, num_samples, **kwargs):
            super().__init__(*args, **kwargs)
            self._temperature_sample_weights = sample_weights
            self._temperature_num_samples = num_samples

        def _get_train_sampler(self, *args, **kwargs):
            # replacement=True is required so minority-task samples can be drawn more often
            # than their raw cardinality in one epoch.
            return WeightedRandomSampler(
                weights=self._temperature_sample_weights,
                num_samples=self._temperature_num_samples,
                replacement=True,
                generator=torch.Generator().manual_seed(SEED),
            )

    return TemperatureSamplerSFTTrainer


def _build_temperature_sampler_trainer(
    *,
    SFTTrainer,
    trainer_kwargs,
    data,
    temperature_sampler_T,
    temperature_sampler_task_column,
    temperature_sampler_num_samples,
):
    """Compute temperature-weighted sample probabilities and return a trainer.

    Shared by prepare_trainer() (LoRA) and prepare_trainer_fullFT() (full FT).
    Task-level sampling probability is set to ``p(task) ~ count(task)^(1/T)`` and
    each example is weighted by ``p(task) / count(task)`` so a
    :class:`~torch.utils.data.WeightedRandomSampler` reproduces those task
    proportions. Returns a plain SFTTrainer when only one task is present.

    Args:
        SFTTrainer: The base ``trl.SFTTrainer`` class to instantiate.
        trainer_kwargs (dict): Keyword arguments forwarded to the trainer
            constructor.
        data: DatasetDict with a ``"train"`` split.
        temperature_sampler_T (float): Sampling temperature; must be > 0. Larger
            values flatten the task distribution.
        temperature_sampler_task_column (str): Column of the train split holding
            each example's task label.
        temperature_sampler_num_samples (int | None): Number of draws per epoch;
            None or a value <= 0 keeps the training-set length.

    Returns:
        SFTTrainer: A temperature-sampling trainer, or a plain ``SFTTrainer``
        when only one task is present.

    Raises:
        ValueError: If ``temperature_sampler_T`` is not > 0, or the task column
            is missing from the train split.
    """
    if temperature_sampler_T <= 0:
        raise ValueError("temperature_sampler_T must be > 0.")

    train_dataset = data["train"]
    if temperature_sampler_task_column not in train_dataset.column_names:
        raise ValueError(
            f"Temperature sampler requires column '{temperature_sampler_task_column}' in train dataset. "
            "Regenerate prepared dataset with task labels or disable temperature sampler."
        )

    task_labels = train_dataset[temperature_sampler_task_column]
    task_counts = defaultdict(int)
    for task_label in task_labels:
        task_counts[str(task_label)] += 1

    if len(task_counts) <= 1:
        # With a single task there is nothing to rebalance; use default trainer path.
        safe_print(
            "[Info] Temperature sampler enabled but only one task found; falling back to standard sampling."
        )
        return SFTTrainer(**trainer_kwargs)

    count_tensor = torch.tensor(
        [float(c) for c in task_counts.values()],
        dtype=torch.float,
    )
    task_probs = count_tensor.pow(1.0 / float(temperature_sampler_T))
    task_probs = task_probs / task_probs.sum()

    # Per-sample weight for examples in task i:
    #   weight_i = p(task_i) / count(task_i)
    # This guarantees task-level sampling probability follows task_probs.
    weight_per_task = {
        task_name: float(task_probs[idx] / count_tensor[idx])
        for idx, task_name in enumerate(task_counts.keys())
    }
    sample_weights = torch.DoubleTensor(
        [weight_per_task[str(task_label)] for task_label in task_labels]
    )

    # Number of draws per epoch. Default (<=0) keeps the previous epoch length,
    # while still changing the task composition within each epoch.
    num_samples = (
        len(train_dataset)
        if temperature_sampler_num_samples is None
        or int(temperature_sampler_num_samples) <= 0
        else int(temperature_sampler_num_samples)
    )

    safe_print(
        f"[Info] Using temperature sampler (T={temperature_sampler_T}) with task counts: {dict(task_counts)}"
    )
    safe_print(
        f"[Info] Temperature-sampled per-task probabilities: "
        f"{ {k: round(float(task_probs[i]), 6) for i, k in enumerate(task_counts.keys())} }"
    )
    safe_print(f"[Info] Temperature sampler num_samples per epoch: {num_samples}")

    TemperatureSamplerSFTTrainer = _make_temperature_sampler_trainer(SFTTrainer)
    return TemperatureSamplerSFTTrainer(
        sample_weights=sample_weights,
        num_samples=num_samples,
        **trainer_kwargs,
    )


def prepare_trainer(
    *,
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
    enable_temperature_sampler=False,
    temperature_sampler_T=3.0,
    temperature_sampler_task_column="__task_name",
    temperature_sampler_num_samples=-1,
):
    """Build a QLoRA :class:`~trl.SFTTrainer` for MedVision SFT.

    Loads ``base_model_hf`` in 4-bit NF4 quantization, attaches a LoRA adapter
    (all-linear target modules, with ``lm_head`` and ``embed_tokens`` also
    trained), and wraps it in an ``SFTTrainer`` configured for BF16 training and
    Weights & Biases logging. When ``enable_temperature_sampler`` is set, a
    temperature-weighted multi-task sampler is used instead of uniform sampling.

    Args:
        run_name (str): Run name for logging / W&B.
        base_model_hf (str): HuggingFace id of the base image-text-to-text
            model.
        lora_checkpoint_dir (str): Output directory for adapter checkpoints.
        data: DatasetDict with ``"train"`` and ``"validation"`` splits.
        make_collate_fn: Factory called as ``make_collate_fn(processor)`` to
            build the data collator.
        per_device_train_batch_size (int): Per-device train batch size. Defaults
            to 14.
        per_device_eval_batch_size (int): Per-device eval batch size. Defaults
            to 14.
        gradient_accumulation_steps (int): Gradient accumulation steps. Defaults
            to 6.
        use_flash_attention_2 (bool): Use FlashAttention-2 when True, else eager
            attention. Defaults to True.
        num_train_epochs (int): Number of training epochs. Defaults to 1.
        save_steps (int): Steps between checkpoint saves. Defaults to 100.
        eval_steps (int): Steps between evaluations. Defaults to 50.
        logging_steps (int): Steps between log entries. Defaults to 50.
        save_total_limit (int): Max checkpoints to retain. Defaults to 10.
        dataloader_num_workers (int): DataLoader worker processes. Defaults to 8.
        gradient_checkpointing (bool): Enable gradient checkpointing. Defaults
            to False.
        dataloader_pin_memory (bool): Pin DataLoader memory. Defaults to True.
        push_LoRA (bool): Push the adapter to the Hub (private) when True.
            Defaults to False.
        enable_temperature_sampler (bool): Enable temperature-based multi-task
            sampling. Defaults to False.
        temperature_sampler_T (float): Sampling temperature. Defaults to 3.0.
        temperature_sampler_task_column (str): Column holding task labels.
            Defaults to ``"__task_name"``.
        temperature_sampler_num_samples (int): Draws per epoch; <= 0 keeps the
            training-set length. Defaults to -1.

    Returns:
        SFTTrainer: The configured trainer (a temperature-sampling subclass when
        enabled with more than one task).

    Raises:
        ValueError: If the GPU does not support bfloat16.
    """
    from peft import LoraConfig
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    # Check if GPU supports bfloat16
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError(
            "GPU does not support bfloat16, please use a GPU that supports bfloat16."
        )

    # Set the device string for multi-gpu training using accelerate's PartialState
    # ref: https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md#multi-gpu-training
    # MEDVISION_SFT_ATTN overrides the attention implementation (e.g. "sdpa") for
    # model families whose FA2 path is unvalidated on their transformers pin
    # (qwen3_5, gemma4); unset -> original FA2/eager behavior.
    attn_impl = os.environ.get("MEDVISION_SFT_ATTN") or (
        "flash_attention_2" if use_flash_attention_2 else "eager"
    )
    model_kwargs = dict(
        attn_implementation=attn_impl,
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
    model = AutoModelForImageTextToText.from_pretrained(base_model_hf, **model_kwargs)

    # Training never generates, so disable the KV/recurrent cache outright. The Trainer only
    # does this when gradient_checkpointing=True, but a live cache is harmful in training
    # regardless: it wastes memory, and under any activation-checkpoint RECOMPUTE a stateful
    # cache gets appended twice (observed on qwen3_5: exactly-2x K/V size mismatch in backward).
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    # Initialize processor
    processor = AutoProcessor.from_pretrained(base_model_hf)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    # PEFT configuration
    peft_config = LoraConfig(
        lora_alpha=32,  # scaling factor = lora_alpha / r, controls the strength of the LoRA update
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
        seed=SEED,
        data_seed=SEED,
        num_train_epochs=num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Enable gradient checkpointing to reduce memory usage
        gradient_checkpointing=gradient_checkpointing,
        # MEDVISION_SFT_OPTIM overrides the optimizer (e.g. "paged_adamw_8bit" for the
        # Gemma-family 27-31B QLoRA runs, whose fp32-trained modules_to_save embeddings
        # blow the fused-AdamW state past the GPU ceiling); unset -> fused AdamW.
        optim=os.environ.get("MEDVISION_SFT_OPTIM", "adamw_torch_fused"),
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
        report_to="wandb",  # Report metrics to Weights & Biases
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

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_config,
        processing_class=processor,
        data_collator=make_collate_fn(processor),
    )

    # Temperature sampler path (optional): rebalance multi-task sampling by sampling tasks
    # according to p(task) ~ count(task)^(1/T) instead of raw dataset proportion.
    if enable_temperature_sampler:
        return _build_temperature_sampler_trainer(
            SFTTrainer=SFTTrainer,
            trainer_kwargs=trainer_kwargs,
            data=data,
            temperature_sampler_T=temperature_sampler_T,
            temperature_sampler_task_column=temperature_sampler_task_column,
            temperature_sampler_num_samples=temperature_sampler_num_samples,
        )
    else:
        return SFTTrainer(**trainer_kwargs)


def prepare_trainer_fullFT(
    *,
    run_name,
    base_model_hf,
    model_weights_from=None,
    checkpoint_dir,
    data,
    make_collate_fn,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    use_flash_attention_2=True,
    num_train_epochs=1,
    save_steps=100,
    eval_steps=50,
    logging_steps=50,
    save_total_limit=5,
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    push_model=False,
    enable_temperature_sampler=False,
    temperature_sampler_T=3.0,
    temperature_sampler_task_column="__task_name",
    temperature_sampler_num_samples=-1,
):
    """Prepare an SFTTrainer for full parameter finetuning (no LoRA, no quantization).

    Loads the model in BF16 without any PEFT adapter. All parameters are trained.
    Use a lower learning rate and cosine scheduler compared to the LoRA variant.
    When ``enable_temperature_sampler`` is set, a temperature-weighted multi-task
    sampler is used instead of uniform sampling.

    Args:
        run_name (str): Run name for logging / W&B.
        base_model_hf (str): HuggingFace id of the base image-text-to-text
            model. Also the source of the processor.
        model_weights_from (str, optional): Local checkpoint directory to load
            the model weights from instead of ``base_model_hf`` (resume path).
            Pass the same directory to ``train_resume_from_checkpoint`` with
            ``weights_preloaded=True`` so the Trainer does not re-load it.
        checkpoint_dir (str): Output directory for checkpoints.
        data: DatasetDict with ``"train"`` and ``"validation"`` splits.
        make_collate_fn: Factory called as ``make_collate_fn(processor)`` to
            build the data collator.
        per_device_train_batch_size (int): Per-device train batch size. Defaults
            to 1.
        per_device_eval_batch_size (int): Per-device eval batch size. Defaults
            to 1.
        gradient_accumulation_steps (int): Gradient accumulation steps. Defaults
            to 16.
        use_flash_attention_2 (bool): Use FlashAttention-2 when True, else eager
            attention. Defaults to True.
        num_train_epochs (int): Number of training epochs. Defaults to 1.
        save_steps (int): Steps between checkpoint saves. Defaults to 100.
        eval_steps (int): Steps between evaluations. Defaults to 50.
        logging_steps (int): Steps between log entries. Defaults to 50.
        save_total_limit (int): Max checkpoints to retain. Defaults to 5.
        dataloader_num_workers (int): DataLoader worker processes. Defaults to 4.
        gradient_checkpointing (bool): Enable gradient checkpointing (on by
            default; required at 7B+ scale). Defaults to True.
        dataloader_pin_memory (bool): Pin DataLoader memory. Defaults to True.
        push_model (bool): Push the trained model to the Hub (private) when True.
            Defaults to False.
        enable_temperature_sampler (bool): Enable temperature-based multi-task
            sampling. Defaults to False.
        temperature_sampler_T (float): Sampling temperature. Defaults to 3.0.
        temperature_sampler_task_column (str): Column holding task labels.
            Defaults to ``"__task_name"``.
        temperature_sampler_num_samples (int): Draws per epoch; <= 0 keeps the
            training-set length. Defaults to -1.

    Returns:
        SFTTrainer: The configured trainer (a temperature-sampling subclass when
        enabled with more than one task).

    Raises:
        ValueError: If the GPU does not support bfloat16.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    # Check if GPU supports bfloat16
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError(
            "GPU does not support bfloat16, please use a GPU that supports bfloat16."
        )

    # Set the device string for multi-gpu training using accelerate's PartialState
    # ref: https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md#multi-gpu-training
    # MEDVISION_SFT_ATTN overrides the attention implementation (e.g. "sdpa") for
    # memory-constrained smoke tests; unset -> original FA2/eager behavior.
    attn_impl = os.environ.get("MEDVISION_SFT_ATTN") or (
        "flash_attention_2" if use_flash_attention_2 else "eager"
    )
    model_kwargs = dict(
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Under FSDP (accelerate launch --use_fsdp), do NOT load with device_map: dispatching the
    # full model onto each GPU before the FSDP wrap leaves every rank holding UNSHARDED weights
    # (observed: ~77GiB/GPU = full bf16 params + grad shard at 27-31B -> OOM in backward,
    # regardless of attention impl or checkpointing). Without device_map, transformers'
    # FSDP-aware loading (fsdp_cpu_ram_efficient_loading + sync_module_states) materializes
    # rank0 on CPU / other ranks on meta, and FSDP shards onto GPUs at wrap time.
    if os.environ.get("ACCELERATE_USE_FSDP", "").lower() != "true":
        model_kwargs["device_map"] = {"": PartialState().process_index}

    # Load the model in BF16 without quantization — all parameters will be trained.
    # On resume, model_weights_from points at the last checkpoint so the weights come in
    # through this same FSDP-aware path (rank0 on CPU, other ranks on meta, shard at wrap).
    # Trainer._load_from_checkpoint must then be skipped (train_resume_from_checkpoint
    # weights_preloaded=True): for a SHARDED weights-only checkpoint under FSDP it falls
    # into transformers' load_sharded_checkpoint, whose model.state_dict() all-gathers the
    # full unsharded model on EVERY rank (observed 76GiB/GPU at 31B -> OOM, 2026-07-08).
    if model_weights_from is not None:
        safe_print(f"[Resume] Loading model weights from checkpoint: {model_weights_from}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_weights_from or base_model_hf, **model_kwargs
    )

    # Training never generates, so disable the KV/recurrent cache outright. The Trainer only
    # does this when gradient_checkpointing=True, but a live cache is harmful in training
    # regardless: it wastes memory, and under any activation-checkpoint RECOMPUTE a stateful
    # cache gets appended twice (observed on qwen3_5: exactly-2x K/V size mismatch in backward).
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    # Initialize processor
    processor = AutoProcessor.from_pretrained(base_model_hf)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    args = SFTConfig(
        run_name=run_name,
        output_dir=checkpoint_dir,
        seed=SEED,
        data_seed=SEED,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Gradient checkpointing is on by default for full FT — required at 7B+ scale
        gradient_checkpointing=gradient_checkpointing,
        # Optimizer is overridable via env for memory-constrained smoke tests (e.g.
        # MEDVISION_SFT_OPTIM=adafactor to fit a 27-31B full-FT run on a 4xGPU / 400GB-RAM
        # pod with CPU offload). Defaults to the full-FT setting used by the real runs.
        optim=os.environ.get("MEDVISION_SFT_OPTIM", "adamw_torch_fused"),
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        # MEDVISION_SFT_SAVE_ONLY_MODEL=1 -> save model weights only, skip optimizer/scheduler
        # state. Needed with 8-bit optimizers under FSDP: FSDP's FULL_STATE_DICT optim_state_dict
        # all-gathers state shaped like each flat param, but bnb 8-bit state is quantized
        # (uint8 + per-block absmax) -> flat-numel mismatch RuntimeError in _convert_all_state_info.
        # Default off so the deliverable runs (adamw_torch_fused, fp32 state gathers fine) keep
        # resumable checkpoints.
        save_only_model=os.environ.get("MEDVISION_SFT_SAVE_ONLY_MODEL", "0") == "1",
        eval_strategy="steps",
        eval_steps=eval_steps,
        # MEDVISION_SFT_LR overrides the default full-FT learning rate (e.g. pure-bf16 runs
        # raise it to keep AdamW updates above the bf16 rounding floor). Default 2e-5 keeps
        # every existing pipeline bit-identical.
        learning_rate=float(os.environ.get("MEDVISION_SFT_LR", "2e-5")),
        # MEDVISION_SFT_PURE_BF16=1 disables accelerate mixed precision (bf16=False): the
        # model is already loaded in bf16 above, so params/grads/compute stay bf16-native
        # with NO fp32 master upcast and NO persistent bf16 _mp_shard. Needed for 27B full
        # FT on 4x80GB — the fp32-master recipe costs 27 (masters) + 13.5 (_mp_shard) +
        # 13.5 (fp32 grad shards) + 13.5 (8-bit optim) = 67.5GiB/GPU fixed, which cannot
        # fit; pure bf16 costs ~40.5GiB fixed. Also the only way to get bf16 grad shards:
        # keep_low_precision_grads + fp32 masters hard-fails (torch .grad setter requires
        # matching dtypes; see MEDVISION_SFT_BF16_GRADS note below). Default off.
        bf16=os.environ.get("MEDVISION_SFT_PURE_BF16", "0") != "1",
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=push_model,
        hub_private_repo=True,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # MEDVISION_SFT_USE_LIGER=1 patches the model with Liger kernels, most importantly the
        # fused linear cross-entropy: lm-head logits (seq x vocab; 262k vocab at Gemma3) are
        # never materialized, capping the seq-length-linear fp32 spike at the loss. Verified
        # applied on tf 4.54 + liger 0.8.0 (2026-07-03); note the 4x80GB 27B OOM's binding
        # term was the fp32 grad shards, not this spike — see MEDVISION_SFT_BF16_GRADS below.
        # Requires `pip install liger-kernel`; default off so validated pipelines are untouched.
        use_liger_kernel=os.environ.get("MEDVISION_SFT_USE_LIGER", "0") == "1",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=processor,
        data_collator=make_collate_fn(processor),
    )

    # Optional GPU-memory probe (MEDVISION_SFT_MEMPROBE=1): prints per-rank allocated/reserved
    # right after the FSDP wrap and after the first optimizer step. With FULL_SHARD working,
    # post-wrap allocated should be ~params/world_size (e.g. ~13.5GiB at 27B/4 ranks) — a
    # full-model figure (~54GiB) means sharding did not engage.
    if os.environ.get("MEDVISION_SFT_MEMPROBE") == "1":
        from transformers import TrainerCallback

        class _MemProbe(TrainerCallback):
            def _report(self, tag):
                if torch.cuda.is_available():
                    dev = torch.cuda.current_device()
                    alloc = torch.cuda.memory_allocated(dev) / 2**30
                    reserv = torch.cuda.memory_reserved(dev) / 2**30
                    # Device-wide truth (cudaMemGetInfo): observed 2026-07-02 that ~17-21GiB of
                    # device memory sits OUTSIDE this rank's torch allocator (sibling ranks'
                    # contexts / NCCL / VMM reservations NVML under-attributes). device_used -
                    # reserved quantifies that overhead; it is what shrinks the effective ceiling.
                    free_b, total_b = torch.cuda.mem_get_info(dev)
                    dev_used = (total_b - free_b) / 2**30
                    print(
                        f"[MEMPROBE] {tag}: device={dev} allocated={alloc:.1f}GiB reserved={reserv:.1f}GiB"
                        f" device_used={dev_used:.1f}GiB device_total={total_b / 2**30:.1f}GiB",
                        flush=True,
                    )
                    # Rank 0 also dumps per-process attribution to identify foreign users.
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        try:
                            import subprocess

                            out = subprocess.run(
                                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10,
                            ).stdout.strip()
                            print(f"[MEMPROBE] {tag}: compute-apps:\n{out}", flush=True)
                        except Exception as e:  # attribution is best-effort diagnostics only
                            print(f"[MEMPROBE] {tag}: compute-apps unavailable ({e})", flush=True)

            def on_train_begin(self, args, state, control, **kwargs):
                self._report("train_begin(post-FSDP-wrap)")
                # Config-effectiveness probe (added 2026-07-03 after 3 OOM runs whose memory
                # profile was insensitive to liger/bf16-grad knobs): print what is ACTUALLY
                # live inside the training loop, not what we asked for.
                try:
                    m = kwargs.get("model")
                    if m is not None:
                        fwd = getattr(m.forward, "__func__", m.forward)
                        print(
                            f"[MEMPROBE] forward_module={getattr(fwd, '__module__', '?')} "
                            f"grad_ckpt={getattr(m, 'is_gradient_checkpointing', '?')}",
                            flush=True,
                        )
                        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP

                        for sub in m.modules():
                            if isinstance(sub, _FSDP):
                                print(f"[MEMPROBE] fsdp.mixed_precision={sub.mixed_precision}", flush=True)
                                break
                        else:
                            print("[MEMPROBE] no FSDP submodule found in callback model", flush=True)
                except Exception as e:
                    print(f"[MEMPROBE] introspection failed: {e!r}", flush=True)

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    self._report("after_step_1")
                    # Optimizer-identity probe: accelerate has historically rebuilt bnb
                    # 8-bit optimizers as silent fp32 AdamW during FSDP prepare
                    # (huggingface/accelerate#1902) — +40GB/rank at 27B. Print the live
                    # class + first-param state dtypes; bnb 8-bit should show uint8 state.
                    try:
                        opt = kwargs.get("optimizer")
                        inner = getattr(opt, "optimizer", opt)
                        state_dtypes = {}
                        for param_state in inner.state.values():
                            for k, v in param_state.items():
                                if torch.is_tensor(v):
                                    state_dtypes[k] = str(v.dtype)
                            break
                        print(
                            f"[MEMPROBE] optimizer={type(inner).__name__} "
                            f"first-param state dtypes: {state_dtypes}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[MEMPROBE] optimizer introspection failed: {e!r}", flush=True)

        trainer_kwargs["callbacks"] = [_MemProbe()]

    # Temperature sampler path (optional): rebalance multi-task sampling by sampling tasks
    # according to p(task) ~ count(task)^(1/T) instead of raw dataset proportion.
    if enable_temperature_sampler:
        trainer = _build_temperature_sampler_trainer(
            SFTTrainer=SFTTrainer,
            trainer_kwargs=trainer_kwargs,
            data=data,
            temperature_sampler_T=temperature_sampler_T,
            temperature_sampler_task_column=temperature_sampler_task_column,
            temperature_sampler_num_samples=temperature_sampler_num_samples,
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    # MEDVISION_SFT_BF16_GRADS=1: store FSDP gradient shards in bf16 instead of fp32.
    # WARNING — PROVEN INCOMPATIBLE with accelerate mixed precision (2026-07-03 run):
    # accelerate's bf16 MP upcasts params to fp32 masters, and torch's .grad setter
    # requires grad dtype == param dtype, so FSDP's post-backward
    # `_use_sharded_grad_views` raises "attempting to assign a gradient with dtype
    # 'c10::BFloat16' to a tensor with dtype 'float'". bf16 grads require bf16 params —
    # use MEDVISION_SFT_PURE_BF16=1 (see SFTConfig above) instead, under which grads are
    # bf16-native and this knob is unnecessary. Kept only for non-upcast setups. Default off.
    if os.environ.get("MEDVISION_SFT_BF16_GRADS", "0") == "1":
        from torch.distributed.fsdp import MixedPrecision

        fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
        if fsdp_plugin is not None:
            fsdp_plugin.mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                keep_low_precision_grads=True,
            )
            safe_print("[Info] FSDP grad shards kept in bf16 (keep_low_precision_grads=True).")
        else:
            safe_print("[Warning] MEDVISION_SFT_BF16_GRADS=1 set but no FSDP plugin found; ignored.")

    # MEDVISION_SFT_SYNC_EACH_BATCH=1: disable Trainer's no_sync() during gradient
    # accumulation. ROOT CAUSE of the 4x identical 2026-07-03 step-0 OOMs (proven by the
    # rank2 CUDA allocator snapshot): HF Trainer wraps all non-final micro-steps in
    # accelerator.no_sync(), and FSDP's no_sync semantics accumulate FULL UNSHARDED
    # gradients on every rank — at 27B that is one 788MiB bf16 flat-grad per decoder layer,
    # ~48GB/rank on top of ~38GB of shards => OOM ~54 layers into the FIRST backward
    # (snapshot timeline: forward gathers freed cleanly 0<->1; backward net +1 per layer,
    # 55 live at OOM). Forcing sync each micro-batch reduce-scatters immediately, so grads
    # accumulate SHARDED (~13.5GB fp32 per rank at 27B/4). Cost: one reduce-scatter per
    # micro-batch instead of one per optimizer step (NVLink, negligible vs compute).
    # transformers 4.54 hardcodes the no_sync wrap (sync_each_batch is not consulted), so
    # neutralizing accelerator.no_sync is the surgical lever. Default off.
    if os.environ.get("MEDVISION_SFT_SYNC_EACH_BATCH", "0") == "1":
        import contextlib

        def _no_sync_disabled(model=None, **_kwargs):
            return contextlib.nullcontext()

        trainer.accelerator.no_sync = _no_sync_disabled
        safe_print(
            "[Info] Gradient sync forced every micro-batch (accelerator.no_sync disabled): "
            "FSDP grads accumulate sharded."
        )

    # MEDVISION_SFT_MEMSNAPSHOT=1: record the CUDA allocator history and, if training dies
    # with OOM, dump a per-rank snapshot (every live allocation + python stacks) next to the
    # checkpoints. Analyze offline with torch.cuda._memory_viz or by unpickling. This is the
    # ground-truth tool for peaks that resist config-level reasoning (see 2026-07-03 runs).
    if os.environ.get("MEDVISION_SFT_MEMSNAPSHOT") == "1":
        _orig_train = trainer.train

        def _train_with_oom_snapshot(*t_args, **t_kwargs):
            torch.cuda.memory._record_memory_history(max_entries=200_000)
            try:
                return _orig_train(*t_args, **t_kwargs)
            except torch.OutOfMemoryError:
                rank = os.environ.get("RANK", "0")
                snap_path = os.path.join(checkpoint_dir, f"oom_memsnap_rank{rank}.pickle")
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.cuda.memory._dump_snapshot(snap_path)
                    print(f"[MEMSNAP] OOM snapshot dumped to {snap_path}", flush=True)
                except Exception as e:
                    print(f"[MEMSNAP] snapshot dump failed: {e!r}", flush=True)
                raise

        trainer.train = _train_with_oom_snapshot

    return trainer


def merge_models(
    base_model_hf,
    lora_checkpoint_dir,
    merged_model_hf,
    merged_model_dir,
    push_to_hub,
):
    """Merge a LoRA adapter into its base model and optionally save/push it.

    Loads the base model on CPU in fp32 (so the sub-BF16 LoRA delta is
    representable), merges the adapter with ``safe_merge=True``, then optionally
    saves the merged model locally and/or pushes it to the Hugging Face Hub.
    This function is intended to be called **only on the main process**.

    Args:
        base_model_hf (str): HuggingFace id of the base model.
        lora_checkpoint_dir (str): Directory of the trained LoRA adapter (also
            the source of the processor).
        merged_model_hf (str): Target Hub repo id for the merged model. Required
            only when ``push_to_hub`` is True.
        merged_model_dir (str | None): Local directory to save the merged model
            to; skipped when None.
        push_to_hub (bool): If True, push the merged model and processor to the
            Hub as a private repo.

    Raises:
        ValueError: If ``push_to_hub`` is True but ``merged_model_hf`` is None.
    """
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("\n[Info] Starting model merge process (CPU-only)...")

    # 1) Load base model on CPU in fp32 so the LoRA delta (<<BF16 step size) is
    #    representable during merge. We cast back to bf16 before saving.
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_hf,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # 2) Load LoRA adapter and merge; safe_merge=True raises on NaN/inf
    peft_model = PeftModel.from_pretrained(model, lora_checkpoint_dir)
    peft_model = peft_model.to(torch.float32)
    merged_model = peft_model.merge_and_unload(safe_merge=True)

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
        print(f"[Info] Pushing merged model to Hugging Face Hub: {merged_model_hf}")
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


def train_resume_from_checkpoint(trainer, last_checkpoint, weights_preloaded=False):
    safe_print("[Resume] Requested resume_from_checkpoint=True")

    assert last_checkpoint is not None, f"No checkpoint found in {last_checkpoint}"
    safe_print(f"[Resume] Found checkpoint: {last_checkpoint}")

    # recompute_total_max_steps already broadcasts the integer so every process
    # receives the same `new_max_steps` value in its local variable.
    new_max_steps = recompute_total_max_steps(trainer)

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
            f"[Resume] Failed to read trainer_state.json ({e}); cannot resume training."
        )
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

        print(f"[Resume] Applied new_max_steps={new_max_steps} on all processes.")
        print(
            f"[Resume] Marked is_finished={trainer.state.is_finished} on all processes."
        )

    if weights_preloaded:
        # Weights were already loaded from last_checkpoint at model construction
        # (prepare_trainer_fullFT model_weights_from). Trainer._load_from_checkpoint would
        # re-load them via transformers' load_sharded_checkpoint, which under FSDP
        # all-gathers the full unsharded model on every rank and OOMs at 27B+ — skip it.
        # trainer_state.json restore, batch skipping, and the (fresh, save_only_model)
        # optimizer/scheduler handling in trainer.train() are unaffected.
        def _skip_load_from_checkpoint(resume_from_checkpoint, model=None):
            safe_print(
                "[Resume] Skipping Trainer._load_from_checkpoint: weights already "
                f"loaded from {resume_from_checkpoint} at model construction."
            )

        trainer._load_from_checkpoint = _skip_load_from_checkpoint

    safe_print("Resuming training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)


def parse_args_multiTask():
    """
    Parse command-line arguments for SFT on the MedVision dataset.
    """

    parser = argparse.ArgumentParser(description="SFT on the MedVision dataset")
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the run",
    )

    # -- Model arguments
    parser.add_argument(
        "--model_family_name",
        type=str,
        required=True,
        help="Model family name, used to identify the model groups that share the same image processor.",
    )
    parser.add_argument(
        "--base_model_hf",
        type=str,
        required=True,
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
    parser.add_argument(
        "--new_shape_hw",
        default=None,
        type=int,
        nargs=2,
        help="Target resize shape as (height, width). Example: --new_shape_hw 1080 1920. Result: args.new_shape_hw → [1080, 1920]",
    )
    parser.add_argument(
        "--ds_download_mode",
        type=str,
        default="reuse_dataset_if_exists",
        help="Dataset download mode: 'reuse_dataset_if_exists' (default), 'reuse_cache_if_exists', 'force_redownload'",
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
    parser.add_argument(
        "--enable_temperature_sampler",
        type=str2bool,
        default=False,
        # When enabled, prepare_trainer() switches to TemperatureSamplerSFTTrainer.
        help="Enable temperature-based weighted random sampling across tasks.",
    )
    parser.add_argument(
        "--temperature_sampler_T",
        type=float,
        default=3.0,
        # T=1 means proportional to counts; larger T flattens task probabilities.
        help="Temperature T for task sampling probabilities: p(task) ~ count^(1/T).",
    )
    parser.add_argument(
        "--temperature_sampler_task_column",
        type=str,
        default="__task_name",
        # This column is injected in train__SFT*.py when concatenating per-task datasets.
        help="Column name in prepared train dataset that stores task labels for weighted sampling.",
    )
    parser.add_argument(
        "--temperature_sampler_num_samples",
        type=int,
        default=-1,
        # <=0 uses len(train_dataset), matching default epoch length semantics.
        help="Number of drawn samples per epoch when temperature sampler is enabled. <=0 means len(train_dataset).",
    )
    args = parser.parse_args()
    return args


def check_model_supported(model_name):
    from lmms_eval.models import get_available_model_names

    supported_models = get_available_model_names()

    # Accept both "vllm_<name>" and "<name>" inputs.
    clean_models = []
    for supported_model in supported_models:
        if supported_model.startswith("vllm_"):
            # Use removeprefix if on Python 3.9+
            clean_model_name = supported_model.removeprefix("vllm_")
            clean_models.append(clean_model_name)
    supported_models.extend(clean_models)

    if model_name not in supported_models:
        raise ValueError(
            f"\n [Error] Model '{model_name}' is not supported. "
            f"Supported models are: {supported_models}"
        )


def parse_validate_args_multiTask():
    args = parse_args_multiTask()

    # Validate model family name
    check_model_supported(args.model_family_name)

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


def _get_sample_limit(kwargs, key, default=-1):
    """Fetch a sample-limit kwarg: missing/None -> default; 0 -> ValueError (ambiguous)."""
    v = kwargs.get(key)
    if v is None:
        return default
    if v == 0:
        raise ValueError(
            f"\n[Error] {key}=0 is ambiguous: use -1 (or leave it unset) for no limit; "
            "to skip a task, omit its --tasks_list_json_path_* instead."
        )
    return v


def parse_sample_limits(**kwargs):
    """
    Determine sample limits for each task with fallbacks.

    Logic:
        - Missing/None limit kwargs fall back to -1 (train: no limit, use the
          full dataset) or 100 (val fallbacks), mirroring the argparse defaults
        - If task-specific limit > 0: use it
        - Else: use per-task limit
        - An explicit limit of 0 raises ValueError (ambiguous): to skip a task,
          omit its tasks_list_json_path instead
        - If task JSON path is None: set limit to 0 (task not used)

    Returns:
        A tuple of sample limits:
        (train_limit_AD, val_limit_AD,
         train_limit_detect, val_limit_detect,
         train_limit_TL, val_limit_TL,
         train_limit_total)
    """

    # The total val limit is not returned (entry points read it from kwargs
    # directly), but 0 must be rejected at this same choke point.
    _get_sample_limit(kwargs, "val_sample_limit", default=100)

    # Determine sample limits for each task
    # Angle/distance task
    train_limit_AD = _get_sample_limit(kwargs, "train_sample_limit_task_AD")
    if train_limit_AD <= 0:
        train_limit_AD = _get_sample_limit(kwargs, "train_sample_limit_per_task")
    val_limit_AD = _get_sample_limit(kwargs, "val_sample_limit_task_AD")
    if val_limit_AD <= 0:
        val_limit_AD = _get_sample_limit(kwargs, "val_sample_limit_per_task", default=100)
    if kwargs.get("tasks_list_json_path_AD") is None:
        train_limit_AD = 0
        val_limit_AD = 0
    # Detection task
    train_limit_detect = _get_sample_limit(kwargs, "train_sample_limit_task_Detection")
    if train_limit_detect <= 0:
        train_limit_detect = _get_sample_limit(kwargs, "train_sample_limit_per_task")
    val_limit_detect = _get_sample_limit(kwargs, "val_sample_limit_task_Detection")
    if val_limit_detect <= 0:
        val_limit_detect = _get_sample_limit(kwargs, "val_sample_limit_per_task", default=100)
    if kwargs.get("tasks_list_json_path_detect") is None:
        train_limit_detect = 0
        val_limit_detect = 0
    # Tumor lesion size task
    train_limit_TL = _get_sample_limit(kwargs, "train_sample_limit_task_TL")
    if train_limit_TL <= 0:
        train_limit_TL = _get_sample_limit(kwargs, "train_sample_limit_per_task")
    val_limit_TL = _get_sample_limit(kwargs, "val_sample_limit_task_TL")
    if val_limit_TL <= 0:
        val_limit_TL = _get_sample_limit(kwargs, "val_sample_limit_per_task", default=100)
    if kwargs.get("tasks_list_json_path_TL") is None:
        train_limit_TL = 0
        val_limit_TL = 0
    # Total sample limit across all tasks
    train_limit_total = _get_sample_limit(kwargs, "train_sample_limit")

    return (
        train_limit_AD,
        val_limit_AD,
        train_limit_detect,
        val_limit_detect,
        train_limit_TL,
        val_limit_TL,
        train_limit_total,
    )


def _mask_turns(input_ids, labels, start_id, end_id, role_id, newline_id):
    """Turn-scan shared by the ChatML and Gemma completion-only maskers.

    For every assistant/model turn the header (``start_id`` immediately followed by
    ``role_id``) is masked, along with the role newline and every non-assistant turn;
    loss is kept on the response content up to and including the closing ``end_id``.
    Only ever writes -100, so -100s the caller already wrote (pad, image) survive inside
    the response.
    """
    seq_len = input_ids.shape[0]
    i = 0
    while i < seq_len:
        is_assistant_header = (
            input_ids[i].item() == start_id
            and i + 1 < seq_len
            and input_ids[i + 1].item() == role_id
        )
        if is_assistant_header:
            labels[i] = -100  # start-of-turn marker
            labels[i + 1] = -100  # role token
            j = i + 2
            if (
                newline_id is not None
                and j < seq_len
                and input_ids[j].item() == newline_id
            ):
                labels[j] = -100  # role-header newline
                j += 1
            # Train response content up to and including the closing end-of-turn.
            while j < seq_len and input_ids[j].item() != end_id:
                j += 1
            if j < seq_len:  # the closing end-of-turn stays in the loss
                j += 1
            i = j
        else:
            labels[i] = -100
            i += 1
    return labels


def mask_non_assistant_turns(input_ids, labels, tokenizer):
    """Mask everything except assistant response content + its closing ``<|im_end|>``.

    Completion-only masking: for every assistant turn the header
    ``<|im_start|>assistant\\n`` is masked (it is chat-template scaffolding the
    model never needs to generate), along with all system/user/tool turns; loss
    is computed only on the response tokens and the ``<|im_end|>`` that
    terminates the assistant turn. The trailing newline after ``<|im_end|>`` is
    also masked.

    ChatML only (Qwen). Gemma checkpoints must use ``mask_non_assistant_turns_gemma``.
    """
    newline_enc = tokenizer.encode("\n", add_special_tokens=False)
    return _mask_turns(
        input_ids,
        labels,
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("assistant"),
        newline_enc[0] if len(newline_enc) == 1 else None,
    )


# (start-of-turn, end-of-turn) marker pairs for the Gemma family. Gemma 3 / MedGemma use
# <start_of_turn>...<end_of_turn>; Gemma 4 renamed them to <|turn>...<turn|>. Each family's
# markers are absent from the other's vocabulary, so the pairs are mutually exclusive and the
# probe order is irrelevant.
_GEMMA_TURN_MARKERS = (
    ("<start_of_turn>", "<end_of_turn>"),  # Gemma 3 / MedGemma
    ("<|turn>", "<turn|>"),  # Gemma 4
)


def _resolve_special_token_id(tokenizer, token_str):
    """Vocab id of ``token_str``, or None if the tokenizer does not have it.

    ``convert_tokens_to_ids`` returns ``unk_token_id`` (3 on Gemma) -- not None -- for
    out-of-vocabulary tokens, which is exactly how another Gemma generation's markers would
    sneak through as a valid-looking id. Tokenizers with no unk token return None instead.
    """
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
        return None
    return token_id


def mask_non_assistant_turns_gemma(input_ids, labels, tokenizer):
    """Gemma-family counterpart of :func:`mask_non_assistant_turns`.

    Same completion-only policy, but the turn markers come from the Gemma vocabulary.
    Gemma 3 / MedGemma render ``<start_of_turn>model\\n ... <end_of_turn>``; Gemma 4 renders
    ``<|turn>model\\n ... <turn|>``. Both use the single token "model" as the assistant role.

    The families must not share a masker: on a Gemma tokenizer ``<|im_start|>`` resolves to
    ``unk_token_id``, no assistant header is ever matched, every label becomes -100 and the loss
    silently becomes NaN. Hence the probe below refuses rather than guesses.
    """
    for start_token, end_token in _GEMMA_TURN_MARKERS:
        start_id = _resolve_special_token_id(tokenizer, start_token)
        end_id = _resolve_special_token_id(tokenizer, end_token)
        if start_id is not None and end_id is not None:
            break
    else:
        raise ValueError(
            f"No Gemma turn markers in this tokenizer's vocabulary (tried {_GEMMA_TURN_MARKERS}). "
            "Refusing to mask: unrecognised markers would silently mask every token."
        )

    role_id = _resolve_special_token_id(tokenizer, "model")
    if role_id is None:
        raise ValueError(
            'Gemma assistant role token "model" is not in this tokenizer\'s vocabulary.'
        )

    newline_enc = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_enc[0] if len(newline_enc) == 1 else None

    labels = _mask_turns(input_ids, labels, start_id, end_id, role_id, newline_id)
    if not (labels != -100).any():
        raise RuntimeError(
            f"Completion-only masking left no tokens in the loss: no '{start_token}model' header "
            "was found. The chat template and _GEMMA_TURN_MARKERS have gone out of sync."
        )
    return labels


def _build_tooluse_messages_AD(prompt, values_dict):
    """Build 5-turn messages list for one AD tool-use training sample."""
    import json

    from medvision_bm.sft.sft_prompts_tooluse import (
        COT_INSTRUCT_ANGLE_TOOLUSE,
        COT_INSTRUCT_DISTANCE_TOOLUSE,
        COT_THINK_ANGLE_TOOLUSE,
        COT_THINK_DISTANCE_TOOLUSE,
        PYTHON_TEMPLATE_ANGLE,
        PYTHON_TEMPLATE_DISTANCE,
        TOOL_DEF,
    )

    metric_type = values_dict["metric_type"]

    if metric_type == "distance":
        code = PYTHON_TEMPLATE_DISTANCE.format(
            x1=values_dict["<x1>"],
            y1=values_dict["<y1>"],
            x2=values_dict["<x2>"],
            y2=values_dict["<y2>"],
            W=values_dict["<image_width>"],
            H=values_dict["<image_height>"],
            pw=values_dict["<pixel_width>"],
            ph=values_dict["<pixel_height>"],
        )
        think_text = fill_in_template(COT_THINK_DISTANCE_TOOLUSE, values_dict)
        user_instruct = COT_INSTRUCT_DISTANCE_TOOLUSE
    else:  # angle
        code = PYTHON_TEMPLATE_ANGLE.format(
            x1=values_dict["<x1_line1>"],
            y1=values_dict["<y1_line1>"],
            x2=values_dict["<x2_line1>"],
            y2=values_dict["<y2_line1>"],
            x3=values_dict["<x1_line2>"],
            y3=values_dict["<y1_line2>"],
            x4=values_dict["<x2_line2>"],
            y4=values_dict["<y2_line2>"],
            W=values_dict["<image_width>"],
            H=values_dict["<image_height>"],
            pw=values_dict["<pixel_width>"],
            ph=values_dict["<pixel_height>"],
        )
        think_text = fill_in_template(COT_THINK_ANGLE_TOOLUSE, values_dict)
        user_instruct = COT_INSTRUCT_ANGLE_TOOLUSE

    tool_result = safe_exec_python(code)
    tool_call_json = json.dumps({"name": "execute_python", "arguments": {"code": code}})
    assistant_turn3 = (
        f"<think> {think_text} </think><tool_call>{tool_call_json}</tool_call>"
    )
    assistant_turn5 = f"<answer> {tool_result} </answer>"

    prompt_base = prompt.rsplit("Report the reasoning process", 1)[0].rstrip()
    user_text = prompt_base + " " + user_instruct

    return [
        {"role": "system", "content": [{"type": "text", "text": json.dumps(TOOL_DEF)}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_turn3}]},
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": f"<tool_response>{tool_result}</tool_response>",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_turn5}]},
    ]


def _build_tooluse_messages_TL(prompt, values_dict):
    """Build 5-turn messages list for one TL tool-use training sample."""
    import json

    from medvision_bm.sft.sft_prompts_tooluse import (
        COT_INSTRUCT_TL_TOOLUSE,
        COT_THINK_TL_TOOLUSE,
        PYTHON_TEMPLATE_TL,
        TOOL_DEF,
    )

    code = PYTHON_TEMPLATE_TL.format(
        x1=values_dict["<x1_major>"],
        y1=values_dict["<y1_major>"],
        x2=values_dict["<x2_major>"],
        y2=values_dict["<y2_major>"],
        x3=values_dict["<x1_minor>"],
        y3=values_dict["<y1_minor>"],
        x4=values_dict["<x2_minor>"],
        y4=values_dict["<y2_minor>"],
        W=values_dict["<image_width>"],
        H=values_dict["<image_height>"],
        pw=values_dict["<pixel_width>"],
        ph=values_dict["<pixel_height>"],
    )
    think_text = fill_in_template(COT_THINK_TL_TOOLUSE, values_dict)
    tool_result = safe_exec_python(code)
    tool_call_json = json.dumps({"name": "execute_python", "arguments": {"code": code}})
    assistant_turn3 = (
        f"<think> {think_text} </think><tool_call>{tool_call_json}</tool_call>"
    )
    assistant_turn5 = f"<answer> {tool_result} </answer>"

    prompt_base = prompt.rsplit("Report the reasoning process", 1)[0].rstrip()
    user_text = prompt_base + " " + COT_INSTRUCT_TL_TOOLUSE

    return [
        {"role": "system", "content": [{"type": "text", "text": json.dumps(TOOL_DEF)}]},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_turn3}]},
        {
            "role": "tool",
            "content": [
                {
                    "type": "text",
                    "text": f"<tool_response>{tool_result}</tool_response>",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": assistant_turn5}]},
    ]


def _format_data_AngleDistanceTask_tooluse(
    example,
    model_name,
    model_hf,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    prompt, values_dict = _doc_to_text_AngleDistanceTask_CoT(
        example, model_name, model_hf, new_shape_hw
    )
    example["messages"] = _build_tooluse_messages_AD(prompt, values_dict)
    if process_img:
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )
    return example


def _format_data_TumorLesionTask_tooluse(
    example,
    model_name,
    model_hf,
    process_img=False,
    save_processed_img_to_disk=False,
    new_shape_hw=None,
):
    prompt, values_dict = _doc_to_text_TumorLesionTask_CoT(
        example, model_name, model_hf, new_shape_hw
    )
    example["messages"] = _build_tooluse_messages_TL(prompt, values_dict)
    if process_img:
        example["processed_images"] = img_proccessor_nii2png_save2dataset(
            example, new_shape_hw
        )
    if save_processed_img_to_disk:
        example["image_file_png"] = img_proccessor_nii2png_save2disk(
            example, new_shape_hw
        )
    return example
