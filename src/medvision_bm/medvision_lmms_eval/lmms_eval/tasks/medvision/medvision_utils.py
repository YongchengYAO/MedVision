import importlib
import os
import re
import sys

import nibabel as nib
import numpy as np
import PIL.Image
import torch
from medvision_ds.utils.doc_to_visual_utils import (
    add_bbox_overlay,
    add_landmarks_and_line_overlay,
    add_mask_overlay_contour,
)
from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoImageProcessor

from medvision_bm.sft.sft_prompts import (
    FORMAT_PROMPT_BIOMETRICS,
    FORMAT_PROMPT_BOX_COORDINATES,
    FORMAT_PROMPT_MASK_SIZE,
    FORMAT_PROMPT_TUMOR_LESION_SIZE,
)
from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE

# NOTE:
# For all tasks in the MedVision-Bench, we use these units for tasks:
#   - mask estimate: mm^2
#   - bounding box size estimate: mm
#   - angle estimate: degree
#   - distance estimate: mm


def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, normalize, and convert to PIL Image.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    return [pil_img]


def doc_to_visual_wBox(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, normalize, and convert to PIL Image with bounding box overlay.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d) 

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Read bbox info, considering possibel resizing
    # NOTE: bbox_*_coords is in (idx_dim0, idx_dim1) <=> (height, width) format
    bbox_min_coords = doc["bounding_boxes"]["min_coords"][0]
    bbox_max_coords = doc["bounding_boxes"]["max_coords"][0]
    # Considering possible resizing, adjust bbox coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        bbox_min_coords = (int(bbox_min_coords[0] * scale_h), int(bbox_min_coords[1] * scale_w))
        bbox_max_coords = (int(bbox_max_coords[0] * scale_h), int(bbox_max_coords[1] * scale_w))

    # Overlay the bounding box on the pil_img
    pil_img = add_bbox_overlay(pil_img, bbox_min_coords, bbox_max_coords)

    return [pil_img]


def doc_to_visual_wBox_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, create a black canvas, and convert to PIL Image with bounding box overlay.

    In short, the input image would be the contour of a box on a black canvas.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Create a 2D black canvas with the same shape as the image slice 
    black_canvas = np.zeros_like(img_2d, dtype=np.uint8)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(black_canvas, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Read bbox info, considering possibel resizing
    # NOTE: bbox_*_coords is in (idx_dim0, idx_dim1) <=> (height, width) format
    bbox_min_coords = doc["bounding_boxes"]["min_coords"][0]
    bbox_max_coords = doc["bounding_boxes"]["max_coords"][0]
    # Considering possible resizing, adjust bbox coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        bbox_min_coords = (int(bbox_min_coords[0] * scale_h), int(bbox_min_coords[1] * scale_w))
        bbox_max_coords = (int(bbox_max_coords[0] * scale_h), int(bbox_max_coords[1] * scale_w))

    # Overlay the bounding box on the pil_img
    pil_img = add_bbox_overlay(pil_img, bbox_min_coords, bbox_max_coords)

    return [pil_img]


def doc_to_visual_wMask(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, normalize, and convert to PIL Image with mask overlay.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    mask_path = doc["mask_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # [Mask] Load 2D slice from NIfTI file, with optional resizing
    if reshape_image_hw is not None:
        _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx)

    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Overlay the mask contour on the pil_img
    pil_img = add_mask_overlay_contour(pil_img, mask_2d)

    return [pil_img]


def doc_to_visual_wMask_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, create a black canvas, and convert to PIL Image with mask overlay.

    In short, the input image would be a mask contour on a black canvas.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    mask_path = doc["mask_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # [Mask] Load 2D slice from NIfTI file, with optional resizing
    if reshape_image_hw is not None:
        _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx)

    # Create a 2D black canvas with the same shape as the image slice 
    black_canvas = np.zeros_like(img_2d, dtype=np.uint8)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(black_canvas, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Overlay the mask contour on the pil_img
    pil_img = add_mask_overlay_contour(pil_img, mask_2d)

    return [pil_img]


def doc_to_visual_wVisualPrompt_TLTask(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, normalize, and convert to PIL Image with visual prompt overlay for tumor/lesion size estimation task.
    The visual prompt includes 2 lines representing the longest and its perpendicular distances of an ellipse enclosing the tumor/lesion.
    """
    from medvision_bm.sft.sft_utils import _get_landmarks_coords

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Get endpoint coordinates for the 2 lines, considering possible resizing
    # ------
    # Gather landmark coordinates in (idx_dim0, idx_dim1) <=> (height, width) format
    landmarks_coords = _get_landmarks_coords(doc, ["P1", "P2", "P3", "P4"])
    coords_p1 = landmarks_coords["landmark_P1"]
    coords_p2 = landmarks_coords["landmark_P2"]
    coords_p3 = landmarks_coords["landmark_P3"]
    coords_p4 = landmarks_coords["landmark_P4"]

    # Considering possible resizing, adjust landmark coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        coords_p1 = (int(coords_p1[0] * scale_h), int(coords_p1[1] * scale_w))
        coords_p2 = (int(coords_p2[0] * scale_h), int(coords_p2[1] * scale_w))
        coords_p3 = (int(coords_p3[0] * scale_h), int(coords_p3[1] * scale_w))
        coords_p4 = (int(coords_p4[0] * scale_h), int(coords_p4[1] * scale_w))
    # ------

    # Overlay the major and minor axes of the ellipse fitted to the tumor/lesion on the pil_img
    # Major axis line: green (#00FF00); Minor axis line: blue (#0000FF); Landmarks: red (#FF0000)
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p1, coords_p2, line_color="#F58D31", point_color="#FF0000")
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p3, coords_p4, line_color="#0000FF", point_color="#FF0000")

    return [pil_img]


def doc_to_visual_wVisualPrompt_TLTask_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, create a black canvas, and convert to PIL Image with visual prompt overlay for tumor/lesion size estimation task.
    The visual prompt includes 2 lines representing the longest, its perpendicular distances of an ellipse enclosing the tumor/lesion, 
    and the mask contour of the tumor/lesion.

    In short, the input image would be 4 landmark points, 2 lines, and a mask contour on a black canvas.
    """
    from medvision_bm.sft.sft_utils import _get_landmarks_coords

    # Read NIfTI image
    img_path = doc["image_file"]
    mask_path = doc["mask_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # [Mask] Load 2D slice from NIfTI file, with optional resizing
    if reshape_image_hw is not None:
        _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, mask_2d = _load_nifti_2d(mask_path, slice_dim, slice_idx)

    # Create a 2D black canvas with the same shape as the image slice 
    black_canvas = np.zeros_like(img_2d, dtype=np.uint8)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(black_canvas, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Get endpoint coordinates for the 2 lines, considering possible resizing
    # ------
    # Gather landmark coordinates in (idx_dim0, idx_dim1) <=> (height, width) format
    landmarks_coords = _get_landmarks_coords(doc, ["P1", "P2", "P3", "P4"])
    coords_p1 = landmarks_coords["landmark_P1"]
    coords_p2 = landmarks_coords["landmark_P2"]
    coords_p3 = landmarks_coords["landmark_P3"]
    coords_p4 = landmarks_coords["landmark_P4"]

    # Considering possible resizing, adjust landmark coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        coords_p1 = (int(coords_p1[0] * scale_h), int(coords_p1[1] * scale_w))
        coords_p2 = (int(coords_p2[0] * scale_h), int(coords_p2[1] * scale_w))
        coords_p3 = (int(coords_p3[0] * scale_h), int(coords_p3[1] * scale_w))
        coords_p4 = (int(coords_p4[0] * scale_h), int(coords_p4[1] * scale_w))
    # ------

    # Overlay the mask contour (in green) on the pil_img
    pil_img = add_mask_overlay_contour(pil_img, mask_2d)

    # Overlay the major and minor axes of the ellipse fitted to the tumor/lesion on the pil_img
    # Major axis line: green (#00FF00); Minor axis line: blue (#0000FF); Landmarks: red (#FF0000)
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p1, coords_p2, line_color="#F58D31", point_color="#FF0000")
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p3, coords_p4, line_color="#0000FF", point_color="#FF0000")

    return [pil_img]


def doc_to_visual_wVisualPrompt_distanceTask(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, normalize, and convert to PIL Image with visual prompt overlay for distance estimation task.
    The visual prompt includes one line representing the target distance.
    """
    from medvision_bm.sft.sft_utils import _get_landmarks_coords

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d) 

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Get endpoint coordinates for the target line, considering possible resizing
    # ------
    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry_module = importlib.import_module(f"medvision_ds.datasets.{dataset_module}.preprocess_biometry")

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry_module.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_type = biometric_profile["metric_type"]
    metric_map_name = biometric_profile["metric_map_name"]
    metric_key = biometric_profile["metric_key"]

    # Gather landmark coordinates in (idx_dim0, idx_dim1) <=> (height, width) format
    lines_map = task_info[metric_map_name]
    line_dict = lines_map[metric_key]
    lms = line_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
    landmarks_coords = _get_landmarks_coords(doc, lms)
    coords_p1 = landmarks_coords["landmark_" + lms[0]]
    coords_p2 = landmarks_coords["landmark_" + lms[1]]

    # Considering possible resizing, adjust landmark coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        coords_p1 = (int(coords_p1[0] * scale_h), int(coords_p1[1] * scale_w))
        coords_p2 = (int(coords_p2[0] * scale_h), int(coords_p2[1] * scale_w))
    # ------

    # Overlay the target line and its endpoints (landmarks) on the pil_img
    # Target line: green (#00FF00); Landmarks: red (#FF0000)
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p1, coords_p2, line_color="#00FF00", point_color="#FF0000")

    return [pil_img]


def doc_to_visual_wVisualPrompt_distanceTask_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, create a black canvas, and convert to PIL Image with visual prompt overlay for distance estimation task.
    The visual prompt includes one line representing the target distance.
    """
    from medvision_bm.sft.sft_utils import _get_landmarks_coords

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Create a 2D black canvas with the same shape as the image slice 
    black_canvas = np.zeros_like(img_2d, dtype=np.uint8)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(black_canvas, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Get endpoint coordinates for the target line, considering possible resizing
    # ------
    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry_module = importlib.import_module(f"medvision_ds.datasets.{dataset_module}.preprocess_biometry")

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry_module.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_type = biometric_profile["metric_type"]
    metric_map_name = biometric_profile["metric_map_name"]
    metric_key = biometric_profile["metric_key"]

    # Gather landmark coordinates in (idx_dim0, idx_dim1) <=> (height, width) format
    lines_map = task_info[metric_map_name]
    line_dict = lines_map[metric_key]
    lms = line_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
    landmarks_coords = _get_landmarks_coords(doc, lms)
    coords_p1 = landmarks_coords["landmark_" + lms[0]]
    coords_p2 = landmarks_coords["landmark_" + lms[1]]

    # Considering possible resizing, adjust landmark coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        coords_p1 = (int(coords_p1[0] * scale_h), int(coords_p1[1] * scale_w))
        coords_p2 = (int(coords_p2[0] * scale_h), int(coords_p2[1] * scale_w))
    # ------

    # Overlay the target line and its endpoints (landmarks) on the pil_img
    # Target line: green (#00FF00); Landmarks: red (#FF0000)
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p1, coords_p2, line_color="#00FF00", point_color="#FF0000")

    return [pil_img]


def doc_to_visual_wVisualPrompt_angleTask(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, normalize, and convert to PIL Image with visual prompt overlay for angle estimation task.
    The visual prompt includes 2 lines representing the two lines forming the target angle.
    """
    from medvision_bm.sft.sft_utils import _get_landmarks_coords

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Normalize the image to 0-255 range
    img_2d_normalized = normalize_img(doc, img_2d)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Get endpoint coordinates for the target line, considering possible resizing
    # ------
    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry_module = importlib.import_module(f"medvision_ds.datasets.{dataset_module}.preprocess_biometry")

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry_module.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_map_name = biometric_profile["metric_map_name"]
    metric_key = biometric_profile["metric_key"]

    # Get line 1 and line 2 coordinates in (idx_dim0, idx_dim1) <=> (height, width) format
    angles_map = task_info[metric_map_name]
    angle_dict = angles_map[metric_key]
    lines_map_name = angle_dict["element_map_name"]
    line_keys = angle_dict["element_keys"]
    lines_map = task_info[lines_map_name]
    line1_dict = lines_map[line_keys[0]]
    line1_lms = line1_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
    line2_dict = lines_map[line_keys[1]]
    line2_lms = line2_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
    line1_landmarks_coords = _get_landmarks_coords(doc, line1_lms)
    line2_landmarks_coords = _get_landmarks_coords(doc, line2_lms)

    # Mark line 1 endpoints as P1 and P2
    coords_p1 = line1_landmarks_coords["landmark_" + line1_lms[0]]
    coords_p2 = line1_landmarks_coords["landmark_" + line1_lms[1]]
    # Mark line 2 endpoints as P3 and P4
    coords_p3 = line2_landmarks_coords["landmark_" + line2_lms[0]]
    coords_p4 = line2_landmarks_coords["landmark_" + line2_lms[1]]

    # Considering possible resizing, adjust landmark coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        coords_p1 = (int(coords_p1[0] * scale_h), int(coords_p1[1] * scale_w))
        coords_p2 = (int(coords_p2[0] * scale_h), int(coords_p2[1] * scale_w))
        coords_p3 = (int(coords_p3[0] * scale_h), int(coords_p3[1] * scale_w))
        coords_p4 = (int(coords_p4[0] * scale_h), int(coords_p4[1] * scale_w))
    # ------

    # Overlay the 2 lines forming the target angle
    # Line 1: green (#00FF00); Line 2: blue (#0000FF); Landmarks: red (#FF0000)
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p1, coords_p2, line_color="#00FF00", point_color="#FF0000")
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p3, coords_p4, line_color="#0000FF", point_color="#FF0000")

    return [pil_img]


def doc_to_visual_wVisualPrompt_angleTask_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Read NIfTI image, create a black canvas, and convert to PIL Image with visual prompt overlay for angle estimation task.
    The visual prompt includes 2 lines representing the two lines forming the target angle.
    """
    from medvision_bm.sft.sft_utils import _get_landmarks_coords

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    image_size_2d = doc["image_size_2d"]

    # [Image] Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Create a 2D black canvas with the same shape as the image slice 
    black_canvas = np.zeros_like(img_2d, dtype=np.uint8)

    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(black_canvas, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")

    # Get endpoint coordinates for the target line, considering possible resizing
    # ------
    # Import the dataset-specific module from medvision_ds.datasets
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")
    preprocess_biometry_module = importlib.import_module(f"medvision_ds.datasets.{dataset_module}.preprocess_biometry")

    # Get task info
    taskID = doc["taskID"]
    bm_plan = preprocess_biometry_module.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_map_name = biometric_profile["metric_map_name"]
    metric_key = biometric_profile["metric_key"]

    # Get line 1 and line 2 coordinates in (idx_dim0, idx_dim1) <=> (height, width) format
    angles_map = task_info[metric_map_name]
    angle_dict = angles_map[metric_key]
    lines_map_name = angle_dict["element_map_name"]
    line_keys = angle_dict["element_keys"]
    lines_map = task_info[lines_map_name]
    line1_dict = lines_map[line_keys[0]]
    line1_lms = line1_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
    line2_dict = lines_map[line_keys[1]]
    line2_lms = line2_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
    line1_landmarks_coords = _get_landmarks_coords(doc, line1_lms)
    line2_landmarks_coords = _get_landmarks_coords(doc, line2_lms)

    # Mark line 1 endpoints as P1 and P2
    coords_p1 = line1_landmarks_coords["landmark_" + line1_lms[0]]
    coords_p2 = line1_landmarks_coords["landmark_" + line1_lms[1]]
    # Mark line 2 endpoints as P3 and P4
    coords_p3 = line2_landmarks_coords["landmark_" + line2_lms[0]]
    coords_p4 = line2_landmarks_coords["landmark_" + line2_lms[1]]

    # Considering possible resizing, adjust landmark coordinates
    if reshape_image_hw is not None:
        orig_h, orig_w = image_size_2d
        new_h, new_w = reshape_image_hw
        scale_h = new_h / orig_h
        scale_w = new_w / orig_w
        coords_p1 = (int(coords_p1[0] * scale_h), int(coords_p1[1] * scale_w))
        coords_p2 = (int(coords_p2[0] * scale_h), int(coords_p2[1] * scale_w))
        coords_p3 = (int(coords_p3[0] * scale_h), int(coords_p3[1] * scale_w))
        coords_p4 = (int(coords_p4[0] * scale_h), int(coords_p4[1] * scale_w))
    # ------

    # Overlay the 2 lines forming the target angle
    # Line 1: green (#00FF00); Line 2: blue (#0000FF); Landmarks: red (#FF0000)
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p1, coords_p2, line_color="#00FF00", point_color="#FF0000")
    pil_img = add_landmarks_and_line_overlay(pil_img, coords_p3, coords_p4, line_color="#0000FF", point_color="#FF0000")

    return [pil_img]


def create_doc_to_text_BoxCoordinate(preprocess_detection_module):
    def doc_to_text_BoxCoordinate(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_detection_module.benchmark_plan
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
        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, "
            f"return the coordinates of the lower-left and upper-right corners of the bounding box for the {label_name}.\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_BOX_COORDINATES}"
        )
        return question

    return doc_to_text_BoxCoordinate


def create_doc_to_text_BoxCoordinate_CoT(preprocess_detection_module):
    def doc_to_text_BoxCoordinate_CoT(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        from medvision_bm.sft.sft_prompts import COT_INSTRUCT_DETECTION

        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_detection_module.benchmark_plan
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
        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, "
            f"return the coordinates of the lower-left and upper-right corners of the bounding box for the {label_name}.\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_BOX_COORDINATES}\n"
            f"Reasoning steps:\n"
            f"{COT_INSTRUCT_DETECTION}\n"
            f"Follow the reasoning steps to get the final answer in the required format."
        )
        return question

    return doc_to_text_BoxCoordinate_CoT


def create_doc_to_text_BoxCoordinate_wBox(preprocess_detection_module):
    def doc_to_text_BoxCoordinate_wBox(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_detection_module.benchmark_plan
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
        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, and the highlighted bounding box enclosing the {label_name}, "
            f"return the coordinates of the lower-left and upper-right corners of the bounding box.\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_BOX_COORDINATES}"
        )
        return question

    return doc_to_text_BoxCoordinate_wBox


def doc_to_text_BoxCoordinate_wBox_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to text.
    This is only used for ablation study to evaluate the model's ability to understand the visual prompt (the highlighted bounding box) without the medical image.

    NOTE: 
    Keep the input argument format consistent with other doc_to_text functions for easier ablation study, 
    even though the image info in the document will not be used in this function.
    """
    question = (
        f"Task:\n"
        f"Return the coordinates of the lower-left and upper-right corners of the highlighted bounding box in the image.\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_BOX_COORDINATES}"
    )
    return question


def _process_img_qwen25vl(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor([img_PIL])
    image_grid_thw = processed_visual["image_grid_thw"][0]
    patch_size = img_processor.patch_size
    img_shape_resized_hw = (image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_qwen3vl(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor([img_PIL])
    image_grid_thw = processed_visual["image_grid_thw"][0]
    patch_size = img_processor.patch_size
    img_shape_resized_hw = (image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_lingshu(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor([img_PIL])
    image_grid_thw = processed_visual["image_grid_thw"][0]
    patch_size = img_processor.patch_size
    img_shape_resized_hw = (image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_medgemma(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor.preprocess(images=[img_PIL], return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_meddr(img_2d_raw, extra_kwargs):
    # NOTE: This is a workaround to import package from local folders
    dir_meddr = os.environ.get("MedDr_DIR")
    sys.path.append(dir_meddr)
    from src.dataset.transforms import build_transform
    from src.model.internvl_chat import InternVLChatModel

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    model = InternVLChatModel.from_pretrained(model_hf, low_cpu_mem_usage=True).eval()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    pad2square = model.config.pad2square
    image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)
    img_shape_resized_hw = image_processor(img_PIL).unsqueeze(0).shape
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_llavaonevision(img_2d_raw, extra_kwargs):
    from transformers.image_processing_utils import select_best_resolution

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)

    # NOTE:
    # Llave-OneVision dynamically resize the image to a shape that can fit in patches of size [384,384]
    # The image processor (LlavaOnevisionImageProcessor) first selects the best resolution for the input image,
    # then resizes the image to the selected resolution, and pads the image and extract patches.
    # Finally, a resized version of the image (with same size as a patch) is added to the list of patches.
    img_shape_resized_hw = select_best_resolution(img_PIL.size, img_processor.image_grid_pinpoints)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_llavamed(img_2d_raw, extra_kwargs):
    from llava.mm_utils import get_model_name_from_path, process_images
    from llava.model.builder import load_pretrained_model

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    model_name = get_model_name_from_path(model_hf)
    _, model, image_processor, _ = load_pretrained_model(model_hf, None, model_name)

    image_tensor = process_images([img_PIL], image_processor, model.config)[0]
    img_shape_resized_hw = image_tensor.shape[-2:]  # (height, width)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_llama_3_2_vision(img_2d_raw, extra_kwargs):
    from typing import Optional, Union

    from transformers.image_utils import (
        ChannelDimension,
        ImageInput,
        PILImageResampling,
        make_nested_list_of_images,
        to_numpy_array,
        validate_preprocess_arguments,
    )
    from transformers.models.mllama.image_processing_mllama import (
        MllamaImageProcessor,
        convert_to_rgb,
        to_channel_dimension_format,
    )
    from transformers.utils import TensorType

    class custom_MllamaImageProcessor(MllamaImageProcessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # adapted from MllamaImageProcessor.preprocess
        def cal_resized_image_shape(
            self,
            images: ImageInput,
            do_convert_rgb: Optional[bool] = None,
            do_resize: Optional[bool] = None,
            size: Optional[dict[str, int]] = None,
            resample: Optional[PILImageResampling] = None,
            do_rescale: Optional[bool] = None,
            rescale_factor: Optional[float] = None,
            do_normalize: Optional[bool] = None,
            image_mean: Optional[Union[float, list[float]]] = None,
            image_std: Optional[Union[float, list[float]]] = None,
            do_pad: Optional[bool] = None,
            max_image_tiles: Optional[int] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
        ):
            do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
            do_resize = do_resize if do_resize is not None else self.do_resize
            size = size if size is not None else self.size
            resample = resample if resample is not None else self.resample
            do_rescale = do_rescale if do_rescale is not None else self.do_rescale
            rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
            do_normalize = do_normalize if do_normalize is not None else self.do_normalize
            image_mean = image_mean if image_mean is not None else self.image_mean
            image_std = image_std if image_std is not None else self.image_std
            do_pad = do_pad if do_pad is not None else self.do_pad
            max_image_tiles = max_image_tiles if max_image_tiles is not None else self.max_image_tiles

            validate_preprocess_arguments(
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_resize=do_resize,
                size=size,
                resample=resample,
            )

            images = self.fetch_images(images)
            images_list = make_nested_list_of_images(images)

            if self.do_convert_rgb:
                images_list = [[convert_to_rgb(image) for image in images] for images in images_list]

            batch_resized_images_shape = []

            # iterate over batch samples
            for images in images_list:
                resized_images_shape = []

                # iterate over images in a batch sample
                for image in images:
                    # default PIL images to channels_last
                    if input_data_format is None and isinstance(image, PIL.Image.Image):
                        input_data_format = ChannelDimension.LAST

                    # convert to numpy array for processing
                    image = to_numpy_array(image)

                    # convert images to channels first format for faster processing
                    # LAST is slower for `pad` and not supported by `split_to_tiles`
                    data_format = ChannelDimension.FIRST
                    image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

                    # do_resize=False is not supported, validated
                    resized_image, _ = self.resize(
                        image=image,
                        size=size,
                        resample=resample,
                        max_image_tiles=max_image_tiles,
                        input_data_format=data_format,
                        data_format=data_format,
                    )
                    resized_images_shape.append(resized_image.shape)

                batch_resized_images_shape.append(resized_images_shape)
            return batch_resized_images_shape

    # NOTE:
    # Llama-3.2-Vision dynamically resize the image to a shape that can fit in patches of size [560, 560].
    # The image processor (MllamaImageProcessor) first selects the best resolution for the input image,
    # then resizes the image to the selected resolution, and pads the image and extract patches.

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]

    # Create custom image processor with the loaded config
    custom_img_processor = custom_MllamaImageProcessor.from_pretrained(model_hf)

    # Use the custom method to calculate resized image shape
    batch_resized_shapes = custom_img_processor.cal_resized_image_shape(images=[img_PIL])

    # Extract the shape for the single image (first batch, first image)
    # Get (height, width) from shape
    img_shape_resized_hw = batch_resized_shapes[0][0][-2:]

    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_internvl3(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor.preprocess(images=[img_PIL], return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_huatuogpt_vision(img_2d_raw, extra_kwargs):
    # NOTE: This is a workaround to import package from local folders
    dir_huatuogpt_vision = os.environ.get("HuatuoGPTVision_DIR")
    sys.path.append(dir_huatuogpt_vision)
    from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    model, _ = LlavaQwen2ForCausalLM.from_pretrained(model_hf, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.bfloat16)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(model_hf)
    vision_tower.to(dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor
    processed_visual = image_processor.preprocess(img_PIL, return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_healthgpt_L14(img_2d_raw, extra_kwargs):
    # NOTE: This is a workaround to import package from local folders
    dir_healthgpt = os.environ.get("HEALTHGPT_DIR")
    dir_demo = os.path.join(dir_healthgpt, "llava", "demo")
    sys.path.append(dir_healthgpt)
    sys.path.append(dir_demo)
    from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
    from llava.peft import LoraConfig, get_peft_model
    from utils import (
        com_vision_args,
        expand2square,
        find_all_linear_names,
        load_weights,
    )

    def prepare_model_healthgpt_L14(extra_kwargs):
        base_model_hf = extra_kwargs.get("base_model_hf", "microsoft/phi-4")
        vision_model_hf = extra_kwargs.get("vision_model_hf", "openai/clip-vit-large-patch14-336")
        dtype = extra_kwargs.get("dtype", "FP16")
        hlora_r = extra_kwargs.get("hlora_r", 32)
        hlora_alpha = extra_kwargs.get("hlora_alpha", 64)
        hlora_dropout = extra_kwargs.get("hlora_dropout", 0)
        hlora_nums = extra_kwargs.get("hlora_nums", 4)
        instruct_template = extra_kwargs.get("instruct_template", "phi4_instruct")

        hlora_weights_local = os.environ.get("HEALTHGPT-L14-HLORA-WEIGHTS-FILE")
        assert hlora_weights_local is not None and os.path.exists(hlora_weights_local), f"hlora_weights_local, {hlora_weights_local}, does not exist."

        if dtype == "BF16":
            model_dtype = torch.bfloat16
        elif dtype == "FP16":
            model_dtype = torch.float16
        elif dtype == "FP32":
            model_dtype = torch.float32

        load_config = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,  # Prioritize safetensors format if available
            "attn_implementation": "flash_attention_2",
            "torch_dtype": model_dtype,
        }

        model = LlavaPhiForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_hf, **load_config)

        lora_config = LoraConfig(
            r=hlora_r,
            lora_alpha=hlora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=hlora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            lora_nums=hlora_nums,
        )
        model = get_peft_model(model, lora_config)

        com_vision_args.model_name_or_path = base_model_hf
        com_vision_args.vision_tower = vision_model_hf
        com_vision_args.version = instruct_template

        model.get_model().initialize_vision_modules(model_args=com_vision_args)
        model.get_vision_tower().to(dtype=model_dtype)

        model = load_weights(model, hlora_weights_local)
        model.eval()
        return model

    def process_img_healthgpt(image, model):
        assert isinstance(image, Image.Image), "Input image must be a PIL Image."
        image = expand2square(image, tuple(int(x * 255) for x in model.get_vision_tower().image_processor.image_mean))
        processed_visual = model.get_vision_tower().image_processor.preprocess(image, return_tensors="pt")
        return processed_visual

    model = prepare_model_healthgpt_L14(extra_kwargs)
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    processed_visual = process_img_healthgpt(img_PIL, model)
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")

    return img_shape_resized_hw


def _process_img_gemma3(img_2d_raw, extra_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor.preprocess(images=[img_PIL], return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {pv_shape}")
    return img_shape_resized_hw


def get_resized_img_shape(model_name, img_2d_raw, extra_kwargs):
    # NOTE 1: usage in MedVision benchamrk 
    # The model_name is the same as the key in AVAILABLE_MODELS. If you add new models, the strings in the if conditions below should be consistent with the keys in AVAILABLE_MODELS.
    # NOTE 2: usage in SFT training 
    # When this function get_resized_img_shape() is not used in the MedVision benchmark, for example, if it is used for SFT model training,
    # the model_name could be different from AVAILABLE_MODELS. For example, we use the model name "vllm_qwen25vl" to refer to the 
    # vllm inference backend of Qwen2.5VL in the MedVision benchmark. While in SFT code (e.g., check the usage of model_family_name in medvision_bm.sft.sft_utils for more details), we can use "qwen25vl" as "model_family_name" 
    # NOTE 3: TODO/Maintainance: Supported model_name is hardcoded, could be improved 
    # For either case, the model_name should be consistent with the string used in the if conditions in this function to ensure the correct image processing method is called to get the resized image shape for pixel size adjustment. 

    assert model_name is not None, "[Error] model_name cannot be None. Please provide a valid model_name to get_resized_img_shape()."

    # Get reshaped image size so that we can adjust the pixel size dynamically
    if model_name in ["qwen3vl", "vllm_qwen3vl"]:
        img_shape_resized_hw = _process_img_qwen3vl(img_2d_raw, extra_kwargs) 
    elif model_name in ["vllm_qwen25vl", "qwen25vl"]:
        # NOTE: Qwen2.5-VL resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
        # Preprocessor config: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/blob/main/preprocessor_config.json
        # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
        img_shape_resized_hw = _process_img_qwen25vl(img_2d_raw, extra_kwargs)
    elif model_name == "lingshu":
        # NOTE: Lingshu resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
        # Preprocessor config: https://huggingface.co/lingshu-medical-mllm/Lingshu-32B/blob/main/preprocessor_config.json
        # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
        img_shape_resized_hw = _process_img_lingshu(img_2d_raw, extra_kwargs)
    elif model_name in ["vllm_llama_3_2_vision", "llama_3_2_vision"]:
        # NOTE: Llama-3.2-Vision dynamically resize the image to a shape that can fit in patches of size [560, 560].
        # Preprocessor config: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/preprocessor_config.json
        # Image processor - MllamaImageProcessor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py#L536
        img_shape_resized_hw = _process_img_llama_3_2_vision(img_2d_raw, extra_kwargs)
    elif model_name in ["vllm_llava_onevision", "llava_onevision"]:
        # NOTE: Llava-OneVision dynamically resize the image to a shape that can fit in patches of size [384,384]
        # NOTE: The current probing method only work for single image input, as padding is enabled for multiple image inputs
        # Preprocessor config: https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf/blob/main/preprocessor_config.json
        # Image processor - LlavaOnevisionImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/llava_onevision/image_processing_llava_onevision.py#L108
        img_shape_resized_hw = _process_img_llavaonevision(img_2d_raw, extra_kwargs)
    elif model_name in ["vllm_gemma3", "gemma3"]:
        # NOTE: HealthGPT resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
        # Preprocessor config: https://huggingface.co/google/gemma-3-27b-it/blob/main/preprocessor_config.json
        # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
        img_shape_resized_hw = [896, 896]
        # img_shape_resized_hw = _process_img_gemma3(img_2d_raw, extra_kwargs)  # for debugging only
    elif model_name == "medgemma":
        # NOTE: Medgemma resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
        # Preprocessor config: https://huggingface.co/google/medgemma-4b-it/blob/main/preprocessor_config.json
        # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
        img_shape_resized_hw = [896, 896]
        # img_shape_resized_hw = _process_img_medgemma(img_2d_raw, extra_kwargs) # for debugging only
    elif model_name == "meddr":
        # NOTE: MedDr resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
        # Check the fixed size in the model config: https://huggingface.co/Sunanhe/MedDr_0401/blob/main/config.json
        img_shape_resized_hw = [448, 448]
        # img_shape_resized_hw = _process_img_meddr(img_2d_raw, extra_kwargs) # for debugging only
    elif model_name == "llava_med":
        # NOTE: Llava-Med resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
        # Check the fixed size in the model config: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/main/config.json
        img_shape_resized_hw = [336, 336]
        # img_shape_resized_hw = _process_img_llavamed(img_2d_raw, extra_kwargs) # for debugging only
    elif model_name in ["vllm_internvl3", "internvl3"]:
        # NOTE: InternVL3 resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
        # Preprocessor config: https://huggingface.co/OpenGVLab/InternVL3-38B/blob/main/preprocessor_config.json
        # Image processor - CLIPImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/clip/image_processing_clip.py#L54
        img_shape_resized_hw = [448, 448]
        # img_shape_resized_hw = _process_img_internvl3(img_2d_raw, extra_kwargs)  # for debugging only
    elif model_name == "huatuogpt_vision":
        # NOTE: HuatuoGPT-Vision resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
        # The fixed size is configured in the "shortest_edge" in image processor: https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B-hf/blob/main/preprocessor_config.json
        # Image processor - CLIPImageProcessor:
        img_shape_resized_hw = [336, 336]
        # img_shape_resized_hw = _process_img_huatuogpt_vision(img_2d_raw, extra_kwargs)  # for debugging only
    elif model_name == "healthgpt_l14":
        # NOTE: HealthGPT resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
        img_shape_resized_hw = [336, 336]
        # img_shape_resized_hw = _process_img_healthgpt_L14(img_2d_raw, extra_kwargs)  # for debugging only
    else:
        raise ValueError(f"[Error] {model_name} is not recognised/supported.")
    return img_shape_resized_hw


def create_doc_to_text_TumorLesionSize(preprocess_biometry_module):
    def doc_to_text_TumorLesionSize(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
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

        # Load 2D slice from NIfTI file, with optional resizing
        # NOTE: reshape_image_hw is the shape of input image
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # Get biometrics profile for this case
        biometric_profile = doc["biometric_profile"]
        metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
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
        return question

    return doc_to_text_TumorLesionSize


def doc_to_text_TumorLesionSize_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to text.
    This is only used for ablation study to see how the model perform without the medical image input, 
    so the question is generated without the medical image description.

    NOTE: 
    Keep the input argument format consistent with other doc_to_text functions for easier ablation study, 
    even though the image info in the document will not be used in this function.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

    img_shape_hw = img_2d_raw.shape

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # -------------
    # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
    # -------------
    # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
    # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
    model_name = lmms_eval_specific_kwargs.get("model_name")
    img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape_hw
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_resized_hw
    resize_ratio_h = resized_img_h / original_height
    resize_ratio_w = resized_img_w / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
    # -------------

    # Question
    question = (
        f"Task:\n"
        f"Estimate the major and minor axis lengths of the ellipse enclosing the highlighted region, in {metric_unit}.\n"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_TUMOR_LESION_SIZE}"
    )
    return question


def create_doc_to_text_TumorLesionSize_wVisualPrompt(preprocess_biometry_module):
    def doc_to_text_TumorLesionSize_wVisualPrompt(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # Get biometrics profile for this case
        biometric_profile = doc["biometric_profile"]
        metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, and the two lines indicating the major and minor axes of the ellipse enclosing the {label_name}, "
            f"estimate the major and minor axis lengths in {metric_unit}.\n"
            f"Additional information:\n"
            f"{image_size_text}\n"
            f"{pixel_size_text}\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_TUMOR_LESION_SIZE}"
        )
        return question

    return doc_to_text_TumorLesionSize_wVisualPrompt


def doc_to_text_TumorLesionSize_wVisualPrompt_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to text.
    This is only used for ablation study to see how the model perform without the medical image input, 
    so the question is generated without the medical image description.

    NOTE: 
    Keep the input argument format consistent with other doc_to_text functions for easier ablation study, 
    even though the image info in the document will not be used in this function.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

    img_shape_hw = img_2d_raw.shape

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # -------------
    # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
    # -------------
    # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
    # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
    model_name = lmms_eval_specific_kwargs.get("model_name")
    img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape_hw
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_resized_hw
    resize_ratio_h = resized_img_h / original_height
    resize_ratio_w = resized_img_w / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
    # -------------

    # Question
    question = (
        f"Task:\n"
        f"Given the two lines indicating the major and minor axes of the ellipse enclosing the highlighted region, "
        f"estimate the major and minor axis lengths in {metric_unit}.\n"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_TUMOR_LESION_SIZE}"
    )
    return question


def create_doc_to_text_TumorLesionSize_CoT_woInstruct(preprocess_biometry_module):
    def doc_to_text_TumorLesionSize_CoT_woInstruct(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        from medvision_bm.sft.sft_prompts import FORMAT_PROMPT_TL_REASONING

        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # Get biometrics profile for this case
        biometric_profile = doc["biometric_profile"]
        metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, "
            f"estimate the major and minor axis lengths of the ellipse enclosing the {label_name}, in {metric_unit}.\n"
            f"Additional information:\n"
            f"{image_size_text}\n"
            f"{pixel_size_text}\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_TL_REASONING}\n"
        )
        return question

    return doc_to_text_TumorLesionSize_CoT_woInstruct


def create_doc_to_text_TumorLesionSize_CoT(preprocess_biometry_module):
    def doc_to_text_TumorLesionSize_CoT(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        from medvision_bm.sft.sft_prompts import (
            COT_INSTRUCT_TL_NORM,
            FORMAT_PROMPT_TL_REASONING,
        )

        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # Get biometrics profile for this case
        biometric_profile = doc["biometric_profile"]
        metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
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
        return question

    return doc_to_text_TumorLesionSize_CoT


def create_doc_to_text_MaskSize(preprocess_segmentation_module):
    def doc_to_text_MaskSize(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_segmentation_module.benchmark_plan
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        # NOTE: We hardcode the unit to millimeters here because there is no metric_unit field for mask size task in MedVision v1.0.0 (https://huggingface.co/datasets/YongchengYAO/MedVision)
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} millimeters (width) x {adjusted_pixel_height:.3f} millimeters (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, "
            f"estimate the physical size of the {label_name} in square millimeters.\n"
            f"Additional information:\n"
            f"{image_size_text}\n"
            f"{pixel_size_text}\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_MASK_SIZE}"
        )
        return question

    return doc_to_text_MaskSize


def create_doc_to_text_MaskSize_wMask(preprocess_segmentation_module):
    def doc_to_text_MaskSize_wMask(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        # Get task info
        taskID = doc["taskID"]
        bm_plan = preprocess_segmentation_module.benchmark_plan
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        # NOTE: We hardcode the unit to millimeters here because there is no metric_unit field for mask size task in MedVision v1.0.0 (https://huggingface.co/datasets/YongchengYAO/MedVision)
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} millimeters (width) x {adjusted_pixel_height:.3f} millimeters (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n"
            f"Given the input medical image{image_prompt}, and the segmentation mask of the {label_name}, "
            f"estimate the physical size of the mask in square millimeters.\n"
            f"Additional information:\n"
            f"{image_size_text}\n"
            f"{pixel_size_text}\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_MASK_SIZE}"
        )
        return question

    return doc_to_text_MaskSize_wMask


def doc_to_text_MaskSize_wMask_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to text.
    This is only used for ablation study to see how the model perform without the medical image input, 
    so the question is generated without the medical image description.

    NOTE: 
    Keep the input argument format consistent with other doc_to_text functions for easier ablation study, 
    even though the image info in the document will not be used in this function.
    """

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

    img_shape_hw = img_2d_raw.shape

    # -------------
    # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
    # -------------
    # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
    # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
    model_name = lmms_eval_specific_kwargs.get("model_name")
    img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape_hw
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_resized_hw
    resize_ratio_h = resized_img_h / original_height
    resize_ratio_w = resized_img_w / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    # NOTE: We hardcode the unit to millimeters here because there is no metric_unit field for mask size task in MedVision v1.0.0 (https://huggingface.co/datasets/YongchengYAO/MedVision)
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} millimeters (width) x {adjusted_pixel_height:.3f} millimeters (height)."
    # -------------

    # Question
    question = (
        f"Task:\n"
        f"Estimate the physical size of the highlighted region in square millimeters.\n"
        f"Additional information:\n"
        f"{image_size_text}\n"
        f"{pixel_size_text}\n"
        f"Format requirement:\n"
        f"{FORMAT_PROMPT_MASK_SIZE}"
    )
    return question


def _get_biometric_prompt_angle(biometrics_name, l1p1, l1p2, l2p1, l2p2, metric_unit):
    """Prepare prompt for angle estimate VQA. Inputs are names."""
    if biometrics_name is not None and biometrics_name != "":
        return f"estimate the angle of {biometrics_name} in {metric_unit}, which is the angle between 2 lines: (line 1) the line connecting {l1p1} and {l1p2}, (line 2) the line connecting {l2p1} and {l2p2}.\n"
    else:
        return f"estimate the angle between 2 lines in {metric_unit}: (line 1) the line connecting {l1p1} and {l1p2}, (line 2) the line connecting {l2p1} and {l2p2}.\n"


def _get_biometric_prompt_distance(biometrics_name, p1, p2, metric_unit):
    """Prepare prompt for distance estimate VQA. Inputs are names."""
    if biometrics_name is not None and biometrics_name != "":
        return f"estimate the distance of {biometrics_name} in {metric_unit}, which is the distance between 2 landmark points: (landmark 1) {p1}, (landmark 2) {p2}.\n"
    else:
        return f"estimate the distance between 2 landmark points in {metric_unit}: (landmark 1) {p1}, (landmark 2) {p2}.\n"


def create_doc_to_text_BiometricsFromLandmarks(preprocess_biometry_module):
    def doc_to_text_BiometricsFromLandmarks(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if metric_type == "distance":
            lines_map = task_info[metric_map_name]
            line_dict = lines_map[metric_key]
            lms_map_name = line_dict["element_map_name"]
            lms_map = task_info[lms_map_name]
            # list of 2 strings -- names of points (landmarks)
            lms = line_dict["element_keys"]
            p1_name = lms_map[lms[0]]
            p2_name = lms_map[lms[1]]
            biometrics_name = line_dict["name"]
            task_prompt = _get_biometric_prompt_distance(biometrics_name, p1_name, p2_name, metric_unit)
        if metric_type == "angle":
            angles_map = task_info[metric_map_name]
            angle_dict = angles_map[metric_key]
            lines_map_name = angle_dict["element_map_name"]
            # list of 2 strings -- names of lines
            line_keys = angle_dict["element_keys"]
            lines_map = task_info[lines_map_name]
            line1_dict = lines_map[line_keys[0]]
            # list of 2 strings -- names of points (landmarks)
            line1_lms = line1_dict["element_keys"]
            line1_lms_map_name = line1_dict["element_map_name"]
            line1_lms_map = task_info[line1_lms_map_name]
            line1_p1_name = line1_lms_map[line1_lms[0]]
            line1_p2_name = line1_lms_map[line1_lms[1]]
            line2_dict = lines_map[line_keys[1]]
            # list of 2 strings -- names of points (landmarks)
            line2_lms = line2_dict["element_keys"]
            line2_lms_map_name = line2_dict["element_map_name"]
            line2_lms_map = task_info[line2_lms_map_name]
            line2_p1_name = line2_lms_map[line2_lms[0]]
            line2_p2_name = line2_lms_map[line2_lms[1]]
            biometrics_name = angle_dict["name"]
            task_prompt = _get_biometric_prompt_angle(biometrics_name, line1_p1_name, line1_p2_name, line2_p1_name, line2_p2_name, metric_unit)

        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
        question = (
            f"Task:\n" 
            f"Given the input medical image{image_prompt}, {task_prompt}" 
            f"Additional information:\n" 
            f"{image_size_text}\n"
            f"{pixel_size_text}\n" 
            f"Format requirement:\n" 
            f"{FORMAT_PROMPT_BIOMETRICS}")
        return question

    return doc_to_text_BiometricsFromLandmarks


def create_doc_to_text_BiometricsFromLandmarks_wVisualPrompt(preprocess_biometry_module):
    def doc_to_text_BiometricsFromLandmarks_wVisualPrompt(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""

        if metric_type == "distance":
            lines_map = task_info[metric_map_name]
            line_dict = lines_map[metric_key]
            lms_map_name = line_dict["element_map_name"]
            lms_map = task_info[lms_map_name]
            # list of 2 strings -- names of points (landmarks)
            lms = line_dict["element_keys"]
            p1_name = lms_map[lms[0]]
            p2_name = lms_map[lms[1]]
            # Task description for distance measurement with visual prompt
            task_description = (
                f"Task:\n" 
                f"Given the input medical image{image_prompt}, and a line connecting {p1_name} and {p2_name}, " 
                f"estimate the physical distance of the line in {metric_unit}.\n"
                )
        if metric_type == "angle":
            angles_map = task_info[metric_map_name]
            angle_dict = angles_map[metric_key]
            lines_map_name = angle_dict["element_map_name"]
            # list of 2 strings -- names of lines
            line_keys = angle_dict["element_keys"]
            lines_map = task_info[lines_map_name]
            line1_dict = lines_map[line_keys[0]]
            # list of 2 strings -- names of points (landmarks)
            line1_lms = line1_dict["element_keys"]
            line1_lms_map_name = line1_dict["element_map_name"]
            line1_lms_map = task_info[line1_lms_map_name]
            line1_p1_name = line1_lms_map[line1_lms[0]]
            line1_p2_name = line1_lms_map[line1_lms[1]]
            line2_dict = lines_map[line_keys[1]]
            # list of 2 strings -- names of points (landmarks)
            line2_lms = line2_dict["element_keys"]
            line2_lms_map_name = line2_dict["element_map_name"]
            line2_lms_map = task_info[line2_lms_map_name]
            line2_p1_name = line2_lms_map[line2_lms[0]]
            line2_p2_name = line2_lms_map[line2_lms[1]]
            # Task description for angle measurement with visual prompt
            task_description = (
                f"Task:\n"
                f"Given the input medical image{image_prompt}, a line connecting {line1_p1_name} and {line1_p2_name}, and another line connecting {line2_p1_name} and {line2_p2_name}, "
                f"estimate the angle between the two lines in {metric_unit}.\n"
            )

        question = (
            f"{task_description}" 
            f"Additional information:\n" 
            f"{image_size_text}\n"
            f"{pixel_size_text}\n" 
            f"Format requirement:\n" 
            f"{FORMAT_PROMPT_BIOMETRICS}"
            )
        return question

    return doc_to_text_BiometricsFromLandmarks_wVisualPrompt


def doc_to_text_BiometricsFromLandmarks_wVisualPrompt_woMedImg(doc, lmms_eval_specific_kwargs=None):
    """
    Convert document to text.
    This is only used for ablation study to see how the model perform without the medical image input, 
    so the question is generated without the medical image description.

    NOTE: 
    Keep the input argument format consistent with other doc_to_text functions for easier ablation study, 
    even though the image info in the document will not be used in this function.
    """

    # Get biometrics profile for this case
    biometric_profile = doc["biometric_profile"]
    metric_type = biometric_profile["metric_type"]
    metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

    # Read NIfTI image
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
    else:
        pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

    img_shape_hw = img_2d_raw.shape

    # -------------
    # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
    # -------------
    # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model.
    # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
    model_name = lmms_eval_specific_kwargs.get("model_name")
    img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

    # Adjust pixel size based on the resize ratio
    original_height, original_width = img_shape_hw
    pixel_height, pixel_width = pixel_size_hw
    resized_img_h, resized_img_w = img_shape_resized_hw
    resize_ratio_h = resized_img_h / original_height
    resize_ratio_w = resized_img_w / original_width
    adjusted_pixel_height = pixel_height / resize_ratio_h
    adjusted_pixel_width = pixel_width / resize_ratio_w

    # Include image size information in the question text
    image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

    # Include pixel size information in question text
    pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
    # -------------

    # Question
    if metric_type == "distance":
        # Task description for distance measurement with visual prompt
        task_description = (
            f"Task:\n"
            f"Estimate the physical distance of the line in {metric_unit}.\n"
            )
    if metric_type == "angle":
        # Task description for angle measurement with visual prompt
        task_description = (
            f"Task:\n"
            f"Estimate the angle between the two lines in {metric_unit}.\n"
        )

    question = (
        f"{task_description}" 
        f"Additional information:\n" 
        f"{image_size_text}\n"
        f"{pixel_size_text}\n" 
        f"Format requirement:\n" 
        f"{FORMAT_PROMPT_BIOMETRICS}"
        )
    return question


def create_doc_to_text_BiometricsFromLandmarks_CoT_woInstruct(preprocess_biometry_module):
    def doc_to_text_BiometricsFromLandmarks_CoT_woInstruct(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        from medvision_bm.sft.sft_prompts import FORMAT_PROMPT_AD_REASONING

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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        if metric_type == "distance":
            # Task prompt
            lines_map = task_info[metric_map_name]
            line_dict = lines_map[metric_key]
            lms_map_name = line_dict["element_map_name"]
            lms_map = task_info[lms_map_name]
            # list of 2 strings -- names of points (landmarks)
            lms = line_dict["element_keys"]
            p1_name = lms_map[lms[0]]
            p2_name = lms_map[lms[1]]
            biometrics_name = line_dict["name"]
            task_prompt = _get_biometric_prompt_distance(biometrics_name, p1_name, p2_name, metric_unit)
        if metric_type == "angle":
            # Task prompt
            angles_map = task_info[metric_map_name]
            angle_dict = angles_map[metric_key]
            lines_map_name = angle_dict["element_map_name"]
            # list of 2 strings -- names of lines
            line_keys = angle_dict["element_keys"]
            lines_map = task_info[lines_map_name]
            line1_dict = lines_map[line_keys[0]]
            # list of 2 strings -- names of points (landmarks)
            line1_lms = line1_dict["element_keys"]
            line1_lms_map_name = line1_dict["element_map_name"]
            line1_lms_map = task_info[line1_lms_map_name]
            line1_p1_name = line1_lms_map[line1_lms[0]]
            line1_p2_name = line1_lms_map[line1_lms[1]]
            line2_dict = lines_map[line_keys[1]]
            # list of 2 strings -- names of points (landmarks)
            line2_lms = line2_dict["element_keys"]
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
            f"{FORMAT_PROMPT_AD_REASONING}\n"
        )

        return question

    return doc_to_text_BiometricsFromLandmarks_CoT_woInstruct


def create_doc_to_text_BiometricsFromLandmarks_CoT(preprocess_biometry_module):
    def doc_to_text_BiometricsFromLandmarks_CoT(doc, lmms_eval_specific_kwargs=None):
        """Convert document to text."""
        from medvision_bm.sft.sft_prompts import (
            COT_INSTRUCT_ANGLE,
            COT_INSTRUCT_DISTANCE,
            FORMAT_PROMPT_AD_REASONING,
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

        # Load 2D slice from NIfTI file, with optional resizing
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        # NOTE: img_shape_resized_hw is the shape of image after model-specific processing, which could be dynamic or fixed depending on the model. 
        # We will use img_shape_resized_hw to adjust the pixel size information in the prompt to make it consistent with the image size input to the model.
        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # Include image size information in the question text
        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."

        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # ------------- 

        # Question
        if metric_type == "distance":
            # CoT instruction - reasoning step description
            cot_instruction = COT_INSTRUCT_DISTANCE
            # Task prompt
            lines_map = task_info[metric_map_name]
            line_dict = lines_map[metric_key]
            lms_map_name = line_dict["element_map_name"]
            lms_map = task_info[lms_map_name]
            # list of 2 strings -- names of points (landmarks)
            lms = line_dict["element_keys"]
            p1_name = lms_map[lms[0]]
            p2_name = lms_map[lms[1]]
            biometrics_name = line_dict["name"]
            task_prompt = _get_biometric_prompt_distance(biometrics_name, p1_name, p2_name, metric_unit)
        if metric_type == "angle":
            # CoT instruction - reasoning step description
            cot_instruction = COT_INSTRUCT_ANGLE
            # Task prompt
            angles_map = task_info[metric_map_name]
            angle_dict = angles_map[metric_key]
            lines_map_name = angle_dict["element_map_name"]
            # list of 2 strings -- names of lines
            line_keys = angle_dict["element_keys"]
            lines_map = task_info[lines_map_name]
            line1_dict = lines_map[line_keys[0]]
            # list of 2 strings -- names of points (landmarks)
            line1_lms = line1_dict["element_keys"]
            line1_lms_map_name = line1_dict["element_map_name"]
            line1_lms_map = task_info[line1_lms_map_name]
            line1_p1_name = line1_lms_map[line1_lms[0]]
            line1_p2_name = line1_lms_map[line1_lms[1]]
            line2_dict = lines_map[line_keys[1]]
            # list of 2 strings -- names of points (landmarks)
            line2_lms = line2_dict["element_keys"]
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
            f"{FORMAT_PROMPT_AD_REASONING}\n"
            f"Reasoning steps:\n"
            f"{cot_instruction}\n"
            f"Follow the reasoning steps to get the final answer in the required format."
        )

        return question

    return doc_to_text_BiometricsFromLandmarks_CoT


def doc_to_target_BoxCoordinate(doc, lmms_eval_specific_kwargs=None):
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
     we calculate the lower-left and upper-right corners coordinates (P1' & P2') in the format of image space indices [idx_width, idx_height] as follows:

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
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # Load 2D slice from NIfTI file, with optional resizing
    reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
    if reshape_image_hw is not None:
        _, img_2d = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        img_size = img_2d.shape
    else:
        img_size = doc.get("image_size_2d", None)
        if img_size is None:
            _, img_size = _load_nifti_2d(img_path, slice_dim, slice_idx)

    # Convert the coordinates from the benchmark planner format to the output format.
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


def doc_to_target_TumorLesionSize(doc):
    """Get ground truth biometrics."""
    biometric_profile = doc["biometric_profile"]
    return [biometric_profile["metric_value_major_axis"][0], biometric_profile["metric_value_minor_axis"][0]]


def doc_to_target_MaskSize(doc):
    """Get segmentation mask (area) size."""
    return doc["ROI_area"]


def doc_to_target_BiometricsFromLandmarks(doc):
    """Get ground truth biometrics."""
    biometric_profile = doc["biometric_profile"]
    return biometric_profile["metric_value"]


def process_results_BoxCoordinate(doc, results):
    pred = results[0]
    # return 4 numbers separated by comma, or empty string
    pred = parser_last_k_nums(pred, 4)
    target_metric = np.array(doc_to_target_BoxCoordinate(doc))
    try:
        # Split the results string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if len(pred_metrics) != 4:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metric)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(absolute_error / (target_metric + 1e-15))
            success = True
    except Exception as e:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {"avgMAE": {"MAE": mean_absolute_error, "success": success}, "avgMRE": {"MRE": mean_relative_error, "success": success}, "SuccessRate": {"success": success}}


def process_results_TumorLesionSize(doc, results):
    """
    Process results for MaskSize task.
    """
    pred = results[0]
    # return 2 numbers separated by comma, or empty string
    pred = parser_last_k_nums(pred, 2)
    target_metric = np.array(doc_to_target_TumorLesionSize(doc))
    try:
        # Split the results string by comma and convert to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if len(pred_metrics) != 2:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metric)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(absolute_error / (target_metric + 1e-15))
            success = True
    except Exception as e:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {"avgMAE": {"MAE": mean_absolute_error, "success": success}, "avgMRE": {"MRE": mean_relative_error, "success": success}, "SuccessRate": {"success": success}}


def process_results_MaskSize(doc, results):
    """
    Process results for MaskSize task.
    """
    pred = results[0]
    pred = parser_last_k_nums(pred, 1)
    target_metric = np.array(doc_to_target_MaskSize(doc))
    try:
        # Convert the result string to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if pred_metrics < 0 or len(pred_metrics) != 1:
            absolute_error = np.nan
            relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metric)
            relative_error = absolute_error / (target_metric + 1e-15)
            success = True
    except Exception as e:
        absolute_error = np.nan
        relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {"MAE": {"AE": absolute_error, "success": success}, "MRE": {"RE": relative_error, "success": success}, "SuccessRate": {"success": success}}


def process_results_BiometricsFromLandmarks(doc, results):
    """
    Process results for Biometrics estimate task.
    """
    pred = results[0]
    pred = parser_last_k_nums(pred, 1)
    target_metric = np.array(doc_to_target_BiometricsFromLandmarks(doc))
    try:
        # Convert the result string to float32
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if pred_metrics < 0 or len(pred_metrics) != 1:
            absolute_error = np.nan
            relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metric)
            relative_error = absolute_error / (target_metric + 1e-15)
            success = True
    except Exception as e:
        absolute_error = np.nan
        relative_error = np.nan
        success = False

    # NOTE: The key name is important. It is referred in the "metric" field of the yaml file for this task.
    return {"MAE": {"AE": absolute_error, "success": success}, "MRE": {"RE": relative_error, "success": success}, "SuccessRate": {"success": success}}


def aggregate_results_MAE(results):
    """
    Args:
        results: a list of values returned by process_results_MaskSize()
    Returns:
        MAE: the average AE of the successful results
    """
    sum_AE = 0
    success_count = 0
    for result in results:
        if result["success"]:
            sum_AE += result["AE"]
            success_count += 1
    MAE = sum_AE / success_count if success_count > 0 else np.nan
    return MAE


def aggregate_results_MRE(results):
    """
    Args:
        results: a list of values returned by process_results_MaskSize()
    Returns:
        MRE: the average RE of the successful results
    """
    sum_RE = 0
    success_count = 0
    for result in results:
        if result["success"]:
            sum_RE += result["RE"]
            success_count += 1
    MRE = sum_RE / success_count if success_count > 0 else np.nan
    return MRE


def aggregate_results_avgMAE(results):
    """
    Args:
        results: a list of values returned by process_results_BoxSize()
    Returns:
        avgMAE: the average MAE of the successful results
    """
    sum_MAE = 0
    success_count = 0
    for result in results:
        if result["success"]:
            sum_MAE += result["MAE"]
            success_count += 1
    avgMAE = sum_MAE / success_count if success_count > 0 else np.nan
    return avgMAE


def aggregate_results_avgMRE(results):
    """
    Args:
        results: a list of values returned by process_results_BoxSize()
    Returns:
        avgMRE: the average MRE of the successful results
    """
    sum_MRE = 0
    success_count = 0
    for result in results:
        if result["success"]:
            sum_MRE += result["MRE"]
            success_count += 1
    avgMRE = sum_MRE / success_count if success_count > 0 else np.nan
    return avgMRE


def aggregate_results_SuccessRate(results):
    """
    Args:
        results: a list of values returned by process_results_BoxSize()
    Returns:
        success_rate: the percentage of successful results
    """
    success_count = 0
    for result in results:
        if result["success"]:
            success_count += 1
    success_rate = success_count / len(results) if len(results) > 0 else np.nan
    return success_rate


def _load_nifti_2d(nii_path, slice_dim, slice_idx, new_shape_hw=None):
    """Map function to load 2D slice from a 3D NIFTI images."""
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Image file {nii_path} does not exist.")
    img_nib = nib.load(nii_path)
    voxel_size = img_nib.header.get_zooms()
    image_3d = img_nib.get_fdata().astype("float32")
    if slice_dim == 0:
        image_2d = image_3d[slice_idx, :, :]
        pixel_size_hw = voxel_size[1:3]
    elif slice_dim == 1:
        image_2d = image_3d[:, slice_idx, :]
        pixel_size_hw = voxel_size[0:1] + voxel_size[2:3]
    elif slice_dim == 2:
        image_2d = image_3d[:, :, slice_idx]
        pixel_size_hw = voxel_size[0:2]
    else:
        raise ValueError("slice_dim must be 0, 1 or 2")

    # Reshape image and update pixel size if new_shape_hw is provided
    if new_shape_hw is not None:
        original_shape_hw = image_2d.shape
        # Calculate zoom factors for each dimension
        zoom_factors = (new_shape_hw[0] / original_shape_hw[0], new_shape_hw[1] / original_shape_hw[1])

        # Check if image is binary to use nearest neighbor interpolation
        is_binary = len(np.unique(image_2d)) <= 2
        order = 0 if is_binary else 1

        # order=1 for bilinear interpolation, order=0 for nearest neighbor
        image_2d = zoom(image_2d, zoom_factors, order=order)

        # Update pixel size based on zoom factors
        pixel_size_hw = (pixel_size_hw[0] / zoom_factors[0], pixel_size_hw[1] / zoom_factors[1])

    return (pixel_size_hw, image_2d)


def _normalize_metric_unit(metric_unit):
    """Return the long-form unit string for biometric_profile.metric_unit.
    Accepts str or single-element list.
    NIfTI voxel sizes are already in the same unit as metric_unit —
    no numeric conversion is needed here."""
    if isinstance(metric_unit, list):
        assert len(metric_unit) == 1, "metric_unit list must have one element"
        metric_unit = metric_unit[0]
    if metric_unit == "mm":
        return "millimeters"
    if metric_unit == "cm":
        return "centimeters"
    if metric_unit == "degree":
        return "degrees"
    raise ValueError(f"Unsupported metric_unit: {metric_unit}")


def parser_last_4_nums(text):
    # Find all numbers in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    # Return the last four numbers
    if len(numbers) < 4:
        return ""
    return ",".join(numbers[-4:])


def parser_last_2_nums(text):
    # Find all numbers in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    # Return the last two numbers
    if len(numbers) < 2:
        return ""
    return ",".join(numbers[-2:])


def parser_last_k_nums(text, k):
    # Find all numbers in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)

    # Return the last k numbers
    if len(numbers) < k:
        return ""
    return ",".join(numbers[-k:])


# ============================================================================
# Pixel-size-scaled (scaledPS) variants of TL and AD tasks.
#
# Per sample, a deterministic scaling factor is drawn from [1, 5] and applied
# to the pixel_size shown in the prompt. The ground-truth metric is recomputed
# under the scaled pixel_size so that a model that correctly reasons about
# pixel_size gets credit, while a model that ignores pixel_size (and produces
# an answer matching the unscaled metric) is penalized proportionally to S.
#
# - TL: uniform scaling (single S). Major/minor-axis GT scale by S.
# - AD: anisotropic (S_h, S_w). Distance/angle GT are recomputed from raw
#       landmark coordinates (retrieved via sft_utils._get_landmarks_coords)
#       in physical space (pixel_coord * pixel_size_hw * (S_h, S_w)).
#       Angle convention matches benchmark_planner._cal_angle: acute angle in
#       degrees via arccos(|v1·v2| / (|v1|·|v2|)).
# ============================================================================

_SCALED_PS_LOW = 0.5   # default; overridable via MEDVISION_SCALED_PS_LOW env var
_SCALED_PS_HIGH = 3.0  # default; overridable via MEDVISION_SCALED_PS_HIGH env var


def _get_pixel_size_scale_factor(doc, mode):
    """Return a deterministic pixel-size scale factor for a doc.

    mode="uniform"      -> returns float S
    mode="anisotropic"  -> returns tuple (S_h, S_w)

    The range [low, high] is read from MEDVISION_SCALED_PS_LOW / MEDVISION_SCALED_PS_HIGH
    env vars at call time (defaults 0.5 / 3.0), so callers can control it via the
    --scaled_ps_low / --scaled_ps_high CLI args of the eval scripts.

    The seed is a stable hash of fields that uniquely identify the sample,
    so the prompt (via doc_to_text) and the ground truth (via doc_to_target)
    always agree even though they are computed by independent calls.
    """
    import hashlib
    import os

    low = float(os.environ.get("MEDVISION_SCALED_PS_LOW", str(_SCALED_PS_LOW)))
    high = float(os.environ.get("MEDVISION_SCALED_PS_HIGH", str(_SCALED_PS_HIGH)))
    key_parts = [
        str(doc.get("image_file", "")),
        str(doc.get("slice_dim", "")),
        str(doc.get("slice_idx", "")),
        str(doc.get("taskID", "")),
        str(doc.get("label", "")),
    ]
    key = "|".join(key_parts).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)
    rng = np.random.RandomState(seed)
    if mode == "uniform":
        return float(rng.uniform(low, high))
    elif mode == "anisotropic":
        return float(rng.uniform(low, high)), float(rng.uniform(low, high))
    else:
        raise ValueError(f"Unknown scale mode: {mode}")


def create_doc_to_text_TumorLesionSize_CoT_scaledPS(preprocess_biometry_module):
    """Factory mirroring create_doc_to_text_TumorLesionSize_CoT; only the
    pixel_size shown in the prompt is scaled by a uniform factor S."""

    def doc_to_text_TumorLesionSize_CoT_scaledPS(doc, lmms_eval_specific_kwargs=None):
        from medvision_bm.sft.sft_prompts import (
            COT_INSTRUCT_TL_NORM,
            FORMAT_PROMPT_TL_REASONING,
        )

        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
        task_info = bm_plan["tasks"][int(taskID) - 1]

        label = str(doc["label"])
        labels_map = task_info["labels_map"]
        if label not in labels_map:
            raise ValueError(f"Label {label} not found in labels_map.")
        label_name = labels_map.get(label)

        image_description = task_info["image_description"]

        img_path = doc["image_file"]
        slice_dim = doc["slice_dim"]
        slice_idx = doc["slice_idx"]

        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        biometric_profile = doc["biometric_profile"]
        metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # scaledPS: multiply the displayed pixel size by a deterministic uniform factor S
        S = _get_pixel_size_scale_factor(doc, "uniform")
        adjusted_pixel_height = adjusted_pixel_height * S
        adjusted_pixel_width = adjusted_pixel_width * S

        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

        if image_description != "" and image_description is not None:
            image_prompt = ": " + image_description
        else:
            image_prompt = ""
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
        return question

    return doc_to_text_TumorLesionSize_CoT_scaledPS


def create_doc_to_target_TumorLesionSize_scaledPS():
    """Factory returning a target function that scales the stored major/minor
    axis GT by the uniform pixel-size scale factor."""

    def doc_to_target_TumorLesionSize_scaledPS(doc):
        S = _get_pixel_size_scale_factor(doc, "uniform")
        biometric_profile = doc["biometric_profile"]
        return [
            biometric_profile["metric_value_major_axis"][0] * S,
            biometric_profile["metric_value_minor_axis"][0] * S,
        ]

    return doc_to_target_TumorLesionSize_scaledPS


# Module-level scaledPS TL target (no factory state needed).
doc_to_target_TumorLesionSize_scaledPS = create_doc_to_target_TumorLesionSize_scaledPS()


def process_results_TumorLesionSize_scaledPS(doc, results):
    """Same shape as process_results_TumorLesionSize but compares to
    doc_to_target_TumorLesionSize_scaledPS."""
    pred = results[0]
    pred = parser_last_k_nums(pred, 2)
    target_metric = np.array(doc_to_target_TumorLesionSize_scaledPS(doc))
    try:
        prd_parts = pred.strip().split(",")
        pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
        if len(pred_metrics) != 2:
            mean_absolute_error = np.nan
            mean_relative_error = np.nan
            success = False
        else:
            absolute_error = np.abs(pred_metrics - target_metric)
            mean_absolute_error = np.mean(absolute_error)
            mean_relative_error = np.mean(absolute_error / (target_metric + 1e-15))
            success = True
    except Exception:
        mean_absolute_error = np.nan
        mean_relative_error = np.nan
        success = False

    return {
        "avgMAE": {"MAE": mean_absolute_error, "success": success},
        "avgMRE": {"MRE": mean_relative_error, "success": success},
        "SuccessRate": {"success": success},
    }


def create_doc_to_text_BiometricsFromLandmarks_CoT_scaledPS(preprocess_biometry_module):
    """Factory mirroring create_doc_to_text_BiometricsFromLandmarks_CoT; only
    the pixel_size shown in the prompt is scaled by an anisotropic pair
    (S_h, S_w). The image is NOT resized."""

    def doc_to_text_BiometricsFromLandmarks_CoT_scaledPS(doc, lmms_eval_specific_kwargs=None):
        from medvision_bm.sft.sft_prompts import (
            COT_INSTRUCT_ANGLE,
            COT_INSTRUCT_DISTANCE,
            FORMAT_PROMPT_AD_REASONING,
        )

        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
        task_info = bm_plan["tasks"][int(taskID) - 1]

        biometric_profile = doc["biometric_profile"]
        metric_type = biometric_profile["metric_type"]
        metric_map_name = biometric_profile["metric_map_name"]
        metric_key = biometric_profile["metric_key"]
        metric_unit = _normalize_metric_unit(biometric_profile["metric_unit"])

        image_description = task_info["image_description"]

        img_path = doc["image_file"]
        slice_dim = doc["slice_dim"]
        slice_idx = doc["slice_idx"]

        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        model_name = lmms_eval_specific_kwargs.get("model_name")
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resized_img_h, resized_img_w = img_shape_resized_hw
        resize_ratio_h = resized_img_h / original_height
        resize_ratio_w = resized_img_w / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w

        # scaledPS: anisotropic factors
        S_h, S_w = _get_pixel_size_scale_factor(doc, "anisotropic")
        adjusted_pixel_height = adjusted_pixel_height * S_h
        adjusted_pixel_width = adjusted_pixel_width * S_w

        image_size_text = f"The image size is {resized_img_w} pixels (width) x {resized_img_h} pixels (height)."
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."

        if metric_type == "distance":
            cot_instruction = COT_INSTRUCT_DISTANCE
            lines_map = task_info[metric_map_name]
            line_dict = lines_map[metric_key]
            lms_map_name = line_dict["element_map_name"]
            lms_map = task_info[lms_map_name]
            lms = line_dict["element_keys"]
            p1_name = lms_map[lms[0]]
            p2_name = lms_map[lms[1]]
            biometrics_name = line_dict["name"]
            task_prompt = _get_biometric_prompt_distance(biometrics_name, p1_name, p2_name, metric_unit)
        if metric_type == "angle":
            cot_instruction = COT_INSTRUCT_ANGLE
            angles_map = task_info[metric_map_name]
            angle_dict = angles_map[metric_key]
            lines_map_name = angle_dict["element_map_name"]
            line_keys = angle_dict["element_keys"]
            lines_map = task_info[lines_map_name]
            line1_dict = lines_map[line_keys[0]]
            line1_lms = line1_dict["element_keys"]
            line1_lms_map_name = line1_dict["element_map_name"]
            line1_lms_map = task_info[line1_lms_map_name]
            line1_p1_name = line1_lms_map[line1_lms[0]]
            line1_p2_name = line1_lms_map[line1_lms[1]]
            line2_dict = lines_map[line_keys[1]]
            line2_lms = line2_dict["element_keys"]
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
            f"{FORMAT_PROMPT_AD_REASONING}\n"
            f"Reasoning steps:\n"
            f"{cot_instruction}\n"
            f"Follow the reasoning steps to get the final answer in the required format."
        )
        return question

    return doc_to_text_BiometricsFromLandmarks_CoT_scaledPS


def create_doc_to_target_BiometricsFromLandmarks_scaledPS(preprocess_biometry_module):
    """Factory returning a target function that recomputes AD metric from raw
    landmark coords under anisotropic pixel-size scaling (S_h, S_w).

    - Loads original pixel_size via _load_nifti_2d (no reshape).
    - Loads raw landmark coords via sft_utils._get_landmarks_coords.
    - For distance: L2 norm of (dh*px_h*S_h, dw*px_w*S_w).
    - For angle: acute angle in degrees between physical-space line vectors,
      matching benchmark_planner._cal_angle.
    """

    def doc_to_target_BiometricsFromLandmarks_scaledPS(doc):
        from medvision_bm.sft.sft_utils import _get_landmarks_coords

        taskID = doc["taskID"]
        bm_plan = preprocess_biometry_module.benchmark_plan
        task_info = bm_plan["tasks"][int(taskID) - 1]

        biometric_profile = doc["biometric_profile"]
        metric_type = biometric_profile["metric_type"]
        metric_map_name = biometric_profile["metric_map_name"]
        metric_key = biometric_profile["metric_key"]

        # Original pixel_size in native NIfTI unit (matches metric_unit), no image reshape.
        pixel_size_hw, _ = _load_nifti_2d(doc["image_file"], doc["slice_dim"], doc["slice_idx"])
        px_h, px_w = float(pixel_size_hw[0]), float(pixel_size_hw[1])

        S_h, S_w = _get_pixel_size_scale_factor(doc, "anisotropic")

        if metric_type == "distance":
            line_dict = task_info[metric_map_name][metric_key]
            lms = line_dict["element_keys"]  # list of 2 landmark keys
            coords = _get_landmarks_coords(doc, lms)
            p1 = np.array(coords["landmark_" + lms[0]], dtype=np.float64)
            p2 = np.array(coords["landmark_" + lms[1]], dtype=np.float64)
            dh = p1[0] - p2[0]
            dw = p1[1] - p2[1]
            distance = float(np.sqrt((dh * px_h * S_h) ** 2 + (dw * px_w * S_w) ** 2))
            return distance

        if metric_type == "angle":
            angle_dict = task_info[metric_map_name][metric_key]
            lines_map_name = angle_dict["element_map_name"]
            line_keys = angle_dict["element_keys"]
            lines_map = task_info[lines_map_name]

            line1 = lines_map[line_keys[0]]
            line2 = lines_map[line_keys[1]]
            line1_lms = line1["element_keys"]
            line2_lms = line2["element_keys"]

            # Load all 4 landmarks in one call (_get_landmarks_coords accepts a list).
            all_lms = list(dict.fromkeys(line1_lms + line2_lms))
            coords = _get_landmarks_coords(doc, all_lms)

            def _vec_phys(lm_keys):
                p1 = np.array(coords["landmark_" + lm_keys[0]], dtype=np.float64)
                p2 = np.array(coords["landmark_" + lm_keys[1]], dtype=np.float64)
                dh = p2[0] - p1[0]
                dw = p2[1] - p1[1]
                return np.array([dh * px_h * S_h, dw * px_w * S_w], dtype=np.float64)

            v1 = _vec_phys(line1_lms)
            v2 = _vec_phys(line2_lms)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            cos_abs = np.abs(np.dot(v1, v2)) / (n1 * n2)
            # Clip for numerical safety (|dot|/... should be in [0,1])
            cos_abs = float(np.clip(cos_abs, 0.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_abs)))
            return angle_deg

        raise ValueError(f"Unsupported metric_type: {metric_type}")

    return doc_to_target_BiometricsFromLandmarks_scaledPS


def create_process_results_BiometricsFromLandmarks_scaledPS(preprocess_biometry_module):
    """Factory returning process_results that compares predictions to the
    factory-bound scaledPS target."""

    _target_fn = create_doc_to_target_BiometricsFromLandmarks_scaledPS(preprocess_biometry_module)

    def process_results_BiometricsFromLandmarks_scaledPS(doc, results):
        pred = results[0]
        pred = parser_last_k_nums(pred, 1)
        target_metric = np.array(_target_fn(doc))
        try:
            prd_parts = pred.strip().split(",")
            pred_metrics = np.array([np.float32(part.strip()) for part in prd_parts])
            if pred_metrics < 0 or len(pred_metrics) != 1:
                absolute_error = np.nan
                relative_error = np.nan
                success = False
            else:
                absolute_error = np.abs(pred_metrics - target_metric)
                relative_error = absolute_error / (target_metric + 1e-15)
                success = True
        except Exception:
            absolute_error = np.nan
            relative_error = np.nan
            success = False

        return {
            "MAE": {"AE": absolute_error, "success": success},
            "MRE": {"RE": relative_error, "success": success},
            "SuccessRate": {"success": success},
        }

    return process_results_BiometricsFromLandmarks_scaledPS
