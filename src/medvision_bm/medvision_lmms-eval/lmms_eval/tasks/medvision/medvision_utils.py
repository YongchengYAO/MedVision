import os
import sys
import re
import nibabel as nib
import numpy as np
import torch
import PIL.Image

from PIL import Image
from scipy.ndimage import zoom
from transformers import AutoImageProcessor


# NOTE:
# For all tasks in the MedVision-Bench, we use these units for tasks:
#   - mask estimate: mm^2
#   - bounding box size estimate: mm
#   - angle estimate: degree
#   - distance estimate: mm


# Prompts
FORMAT_PROMPT_BOX_COORDINATES = (
    "The answer should be four decimal numbers separated by commas without any units or additional text. "
    "The first two numbers are the coordinates of the lower-left corner and the last two numbers are the coordinates of the upper-right corner of the bounding box. "
    "Use relative coordinates in the image space, where the origin is at the lower-left corner of the image. "
    "Relative coordinates should be values between 0 and 1, representing the relative positions in the image."
)

FORMAT_PROMPT_MASK_SIZE = "The answer should be 1 decimal number without any units or additional text."

FORMAT_PROMPT_TUMOR_LESION_SIZE = "The answer should be two decimal numbers separated by a comma without any units or additional text. " "The first is the major axis length, and the second is the minor axis length."

FORMAT_PROMPT_BIOMETRICS = "The answer should be a single decimal number without any units or additional text."


# FIXME: to be updated
# Reasoning Prompts
REASONING_FORMAT = (
    "First thinks about the reasoning process in the mind and then provides the final answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. "
)

FORMAT_PROMPT_BOX_COORDINATES_REASONING = (
    REASONING_FORMAT + "The answer should be four decimal numbers separated by commas without any units or additional text. "
    "The first two numbers are the coordinates of the lower-left corner and the last two numbers are the coordinates of the upper-right corner of the bounding box. "
    "Use relative coordinates in the image space, where the origin is at the lower-left corner of the image. "
    "Relative coordinates should be values between 0 and 1, representing the relative positions in the image."
)

FORMAT_PROMPT_TUMOR_LESION_SIZE_REASONING = (
    REASONING_FORMAT + "The answer should be two decimal numbers separated by a comma without any units or additional text. " "The first is the major axis length, and the second is the minor axis length."
)


# FIXME: to be updated
# Reasoning instructions
COT_TUMOR_LESION_SIZE = (
    "Step 1: Identify the two axes of the ellipse enclosing the target region: "
    "(a) the major axis (the longest diameter) and "
    "(b) the minor axis (the shortest diameter). "
    "For each axis, determine its two endpoints and record their coordinates "
    "in the format (x, y) = (width index, height index). "
    "Thus, for the major axis, denote the endpoints as (x1_major, y1_major) and (x2_major, y2_major). "
    "For the minor axis, denote the endpoints as (x1_minor, y1_minor) and (x2_minor, y2_minor). "
    "Step 2: Given the pixel dimensions (pixel_width, pixel_height), calculate the physical length of each axis "
    "using the distance formula: "
    "axis_length = sqrt(((x2 - x1) * pixel_width)^2 + ((y2 - y1) * pixel_height)^2). "
    "Step 3: Apply this formula separately to the major axis endpoints and the minor axis endpoints "
    "to obtain the lengths of the major and minor axes of the ellipse."
)


def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Convert document to image with scale bar added."""
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
    if img_2d.max() > img_2d.min():
        img_2d_normalized = ((img_2d - img_2d.min()) / (img_2d.max() - img_2d.min()) * 255).astype(np.uint8)
    else:
        img_2d_normalized = np.zeros_like(img_2d, dtype=np.uint8)
    # Convert to PIL Image in grayscale mode
    pil_img = Image.fromarray(img_2d_normalized, mode="L")
    # Convert to RGB mode
    pil_img = pil_img.convert("RGB")
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
        format_prompt = FORMAT_PROMPT_BOX_COORDINATES
        question = f"Task:\n" f"Given the input medical image: {image_description}, " f"return the coordinates of the lower-left and upper-right corner of the bounding box for the {label_name}.\n" f"Format requirement:\n" f"{format_prompt}"
        return question

    return doc_to_text_BoxCoordinate


def _process_img_qwen25vl(img_2d_raw, lmms_eval_specific_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "Qwen/Qwen2.5-VL-32B-Instruct")
    img_processor = AutoImageProcessor.from_pretrained(sample_model_hf)
    processed_visual = img_processor([img_PIL])
    image_grid_thw = processed_visual["image_grid_thw"][0]
    patch_size = img_processor.patch_size
    img_shape_resized_hw = (image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_lingshu(img_2d_raw, lmms_eval_specific_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "lingshu-medical-mllm/Lingshu-32B")
    img_processor = AutoImageProcessor.from_pretrained(sample_model_hf)
    processed_visual = img_processor([img_PIL])
    image_grid_thw = processed_visual["image_grid_thw"][0]
    patch_size = img_processor.patch_size
    img_shape_resized_hw = (image_grid_thw[1] * patch_size, image_grid_thw[2] * patch_size)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_medgemma(img_2d_raw, lmms_eval_specific_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "google/medgemma-4b-it")
    img_processor = AutoImageProcessor.from_pretrained(sample_model_hf)
    processed_visual = img_processor.preprocess(images=[img_PIL], return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_meddr(img_2d_raw, lmms_eval_specific_kwargs):
    # NOTE: This is a workaround to import package from local folders
    dir_meddr = os.environ.get("MedDr_DIR")
    sys.path.append(dir_meddr)
    from src.model.internvl_chat import InternVLChatModel
    from src.dataset.transforms import build_transform

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "Sunanhe/MedDr_0401")
    model = InternVLChatModel.from_pretrained(sample_model_hf, low_cpu_mem_usage=True).eval()
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    pad2square = model.config.pad2square
    image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)
    img_shape_resized_hw = image_processor(img_PIL).unsqueeze(0).shape
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_llavaonevision(img_2d_raw, lmms_eval_specific_kwargs):
    from transformers.image_processing_utils import select_best_resolution

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "llava-hf/llava-onevision-qwen2-72b-ov-hf")
    img_processor = AutoImageProcessor.from_pretrained(sample_model_hf)

    # NOTE:
    # Llave-OneVision dynamically resize the image to a shape that can fit in patches of size [384,384]
    # The image processor (LlavaOnevisionImageProcessor) first selects the best resolution for the input image,
    # then resizes the image to the selected resolution, and pads the image and extract patches.
    # Finally, a resized version of the image (with same size as a patch) is added to the list of patches.
    img_shape_resized_hw = select_best_resolution(img_PIL.size, img_processor.image_grid_pinpoints)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_llavamed(img_2d_raw, lmms_eval_specific_kwargs):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "microsoft/llava-med-v1.5-mistral-7b")
    model_name = get_model_name_from_path(sample_model_hf)
    _, model, image_processor, _ = load_pretrained_model(sample_model_hf, None, model_name)

    image_tensor = process_images([img_PIL], image_processor, model.config)[0]
    img_shape_resized_hw = image_tensor.shape[-2:]  # (height, width)
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_llama_3_2_vision(img_2d_raw, lmms_eval_specific_kwargs):
    from typing import Optional, Union
    from transformers.models.mllama.image_processing_mllama import MllamaImageProcessor, to_channel_dimension_format, convert_to_rgb
    from transformers.image_utils import (
        ChannelDimension,
        ImageInput,
        PILImageResampling,
        make_nested_list_of_images,
        to_numpy_array,
        validate_preprocess_arguments,
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
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    # Create custom image processor with the loaded config
    custom_img_processor = custom_MllamaImageProcessor.from_pretrained(sample_model_hf)

    # Use the custom method to calculate resized image shape
    batch_resized_shapes = custom_img_processor.cal_resized_image_shape(images=[img_PIL])

    # Extract the shape for the single image (first batch, first image)
    img_shape_resized_hw = batch_resized_shapes[0][0][-2:]  # Get (height, width) from shape

    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_internvl3(img_2d_raw, lmms_eval_specific_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "OpenGVLab/InternVL3-38B")
    img_processor = AutoImageProcessor.from_pretrained(sample_model_hf)
    processed_visual = img_processor.preprocess(images=[img_PIL], return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_huatuogpt_vision(img_2d_raw, lmms_eval_specific_kwargs):
    # NOTE: This is a workaround to import package from local folders
    dir_huatuogpt_vision = os.environ.get("HuatuoGPTVision_DIR")
    sys.path.append(dir_huatuogpt_vision)
    from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM

    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "FreedomIntelligence/HuatuoGPT-Vision-34B")
    model, _ = LlavaQwen2ForCausalLM.from_pretrained(sample_model_hf, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.bfloat16)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(sample_model_hf)
    vision_tower.to(dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor
    processed_visual = image_processor.preprocess(img_PIL, return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")
    return img_shape_resized_hw


def _process_img_healthgpt_L14(img_2d_raw, lmms_eval_specific_kwargs):
    # NOTE: This is a workaround to import package from local folders
    dir_healthgpt = os.environ.get("HEALTHGPT_DIR")
    dir_demo = os.path.join(dir_healthgpt, "llava", "demo")
    sys.path.append(dir_healthgpt)
    sys.path.append(dir_demo)
    from llava.peft import LoraConfig, get_peft_model
    from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
    from utils import find_all_linear_names, load_weights, expand2square, com_vision_args

    def prepare_model_healthgpt_L14(lmms_eval_specific_kwargs):
        base_model_hf = lmms_eval_specific_kwargs.get("base_model_hf", "microsoft/phi-4")
        vision_model_hf = lmms_eval_specific_kwargs.get("vision_model_hf", "openai/clip-vit-large-patch14-336")
        dtype = lmms_eval_specific_kwargs.get("dtype", "FP16")
        hlora_r = lmms_eval_specific_kwargs.get("hlora_r", 32)
        hlora_alpha = lmms_eval_specific_kwargs.get("hlora_alpha", 64)
        hlora_dropout = lmms_eval_specific_kwargs.get("hlora_dropout", 0)
        hlora_nums = lmms_eval_specific_kwargs.get("hlora_nums", 4)
        instruct_template = lmms_eval_specific_kwargs.get("instruct_template", "phi4_instruct")

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

    model = prepare_model_healthgpt_L14(lmms_eval_specific_kwargs)
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    processed_visual = process_img_healthgpt(img_PIL, model)
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {img_shape_resized_hw}")

    return img_shape_resized_hw


def _process_img_gemma3(img_2d_raw, lmms_eval_specific_kwargs):
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    sample_model_hf = lmms_eval_specific_kwargs.get("sample_model_hf", "google/gemma-3-27b-it")
    img_processor = AutoImageProcessor.from_pretrained(sample_model_hf)
    processed_visual = img_processor.preprocess(images=[img_PIL], return_tensors="pt")
    pv_shape = processed_visual["pixel_values"].shape
    img_shape_resized_hw = (pv_shape[-2], pv_shape[-1])
    print(f"\nOriginal image size (HxW): {img_PIL.size[::-1]}; Resized image size (HxW): {pv_shape}")
    return img_shape_resized_hw


def get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs):
    # Supported models
    supported_models = ["qwen2_5_vl", "medgemma", "meddr", "llava_onevision", "llava_med", "llama_3_2_vision", "internvl3", "huatuogpt_vision", "healthgpt_l14", "gemma3", "lingshu"]
    if model_name not in supported_models:
        raise ValueError(f"Model {model_name} is not supported for tumor/lesion size estimation task.\n You need to add model-specific implementation in this function: doc_to_text_TumorLesionSize().")

    # Get reshaped image size so that we can adjust the pixel size dynamically
    if model_name == "qwen2_5_vl":
        # NOTE: Qwen2.5-VL resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
        # Preprocessor config: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/blob/main/preprocessor_config.json
        # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
        img_shape_resized_hw = _process_img_qwen25vl(img_2d_raw, lmms_eval_specific_kwargs)
    elif model_name == "lingshu":
        # NOTE: Lingshu resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
        # Preprocessor config: https://huggingface.co/lingshu-medical-mllm/Lingshu-32B/blob/main/preprocessor_config.json
        # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
        img_shape_resized_hw = _process_img_lingshu(img_2d_raw, lmms_eval_specific_kwargs)
    elif model_name == "llama_3_2_vision":
        # NOTE: Llama-3.2-Vision dynamically resize the image to a shape that can fit in patches of size [560, 560].
        # Preprocessor config: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/preprocessor_config.json
        # Image processor - MllamaImageProcessor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py#L536
        img_shape_resized_hw = _process_img_llama_3_2_vision(img_2d_raw, lmms_eval_specific_kwargs)
    elif model_name == "llava_onevision":
        # NOTE: Llava-OneVision dynamically resize the image to a shape that can fit in patches of size [384,384]
        # NOTE: The current probing method only work for single image input, as padding is enabled for multiple image inputs
        # Preprocessor config: https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf/blob/main/preprocessor_config.json
        # Image processor - LlavaOnevisionImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/llava_onevision/image_processing_llava_onevision.py#L108
        img_shape_resized_hw = _process_img_llavaonevision(img_2d_raw, lmms_eval_specific_kwargs)
    elif model_name == "gemma3":
        # NOTE: HealthGPT resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
        # Preprocessor config: https://huggingface.co/google/gemma-3-27b-it/blob/main/preprocessor_config.json
        # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
        img_shape_resized_hw = [896, 896]
        # img_shape_resized_hw = _process_img_gemma3(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
    elif model_name == "medgemma":
        # NOTE: Medgemma resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
        # Preprocessor config: https://huggingface.co/google/medgemma-4b-it/blob/main/preprocessor_config.json
        # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
        img_shape_resized_hw = [896, 896]
        # img_shape_resized_hw = _process_img_medgemma(img_2d_raw, lmms_eval_specific_kwargs) # for debugging only
    elif model_name == "meddr":
        # NOTE: MedDr resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
        # Check the fixed size in the model config: https://huggingface.co/Sunanhe/MedDr_0401/blob/main/config.json
        img_shape_resized_hw = [448, 448]
        # img_shape_resized_hw = _process_img_meddr(img_2d_raw, lmms_eval_specific_kwargs) # for debugging only
    elif model_name == "llava_med":
        # NOTE: Llava-Med resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
        # Check the fixed size in the model config: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/main/config.json
        img_shape_resized_hw = [336, 336]
        # img_shape_resized_hw = _process_img_llavamed(img_2d_raw, lmms_eval_specific_kwargs) # for debugging only
    elif model_name == "internvl3":
        # NOTE: InternVL3 resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
        # Preprocessor config: https://huggingface.co/OpenGVLab/InternVL3-38B/blob/main/preprocessor_config.json
        # Image processor - CLIPImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/clip/image_processing_clip.py#L54
        img_shape_resized_hw = [448, 448]
        # img_shape_resized_hw = _process_img_internvl3(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
    elif model_name == "huatuogpt_vision":
        # NOTE: HuatuoGPT-Vision resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
        # The fixed size is configured in the "shortest_edge" in image processor: https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B-hf/blob/main/preprocessor_config.json
        # Image processor - CLIPImageProcessor:
        img_shape_resized_hw = [336, 336]
        # img_shape_resized_hw = _process_img_huatuogpt_vision(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
    elif model_name == "healthgpt_l14":
        # NOTE: HealthGPT resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
        img_shape_resized_hw = [336, 336]
        # img_shape_resized_hw = _process_img_healthgpt_L14(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
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
        reshape_image_hw = lmms_eval_specific_kwargs.get("reshape_image_hw") if lmms_eval_specific_kwargs is not None else None
        if reshape_image_hw is not None:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw=reshape_image_hw)
        else:
            pixel_size_hw, img_2d_raw = _load_nifti_2d(img_path, slice_dim, slice_idx)

        img_shape_hw = img_2d_raw.shape

        # Get biometrics profile for this case
        biometric_profile = doc["biometric_profile"]
        metric_unit = biometric_profile["metric_unit"]
        if isinstance(metric_unit, list):
            assert len(metric_unit) == 1, "metric_unit list should have only one element."
            metric_unit = metric_unit[0]
        elif isinstance(metric_unit, str):
            if metric_unit == "mm":
                metric_unit = "millimeters"
            elif metric_unit == "cm":
                metric_unit = "centimeters"
        else:
            raise ValueError(f"Unsupported metric_unit type: {type(metric_unit)}")

        # -------------
        # NOTE: To get the reshaped image size and adjust pixel size information in the prompt, a model-specific processing is needed
        # -------------
        model_name = lmms_eval_specific_kwargs.get("model") if lmms_eval_specific_kwargs is not None else None
        assert model_name is not None, "Missing lmms_eval_specific_kwargs: 'model', check the base yaml file for this task."
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resize_ratio_h = img_shape_resized_hw[0] / original_height
        resize_ratio_w = img_shape_resized_hw[1] / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w
        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} {metric_unit} (width) x {adjusted_pixel_height:.3f} {metric_unit} (height)."
        # -------------

        # Question
        question = (
            f"Task:\n"
            f"Given the input medical image: {image_description}, "
            f"estimate the major and minor axis lengths of the ellipse enclosing the {label_name}, in {metric_unit}.\n"
            f"Additional information:\n"
            f"{pixel_size_text}\n"
            f"Format requirement:\n"
            f"{FORMAT_PROMPT_TUMOR_LESION_SIZE}"
        )
        return question

    return doc_to_text_TumorLesionSize


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
        model_name = lmms_eval_specific_kwargs.get("model") if lmms_eval_specific_kwargs is not None else None
        assert model_name is not None, "Missing lmms_eval_specific_kwargs: 'model', check the base yaml file for this task."
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resize_ratio_h = img_shape_resized_hw[0] / original_height
        resize_ratio_w = img_shape_resized_hw[1] / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w
        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} (width) x {adjusted_pixel_height:.3f} (height)."
        # -------------

        # Question
        question = (
            f"Task:\n" f"Given the input medical image: {image_description}, " f"estimate the physical size of the {label_name}.\n" f"Additional information:\n" f"{pixel_size_text}\n" f"Format requirement:\n" f"{FORMAT_PROMPT_MASK_SIZE}"
        )
        return question

    return doc_to_text_MaskSize


def _get_biometric_prompt_angle(biometrics_name, l1p1, l1p2, l2p1, l2p2, metric_unit):
    """Prepare prompt for angle estimate VQA. Inputs are names."""
    if biometrics_name is not None and biometrics_name != "":
        return f"estimate the angle of {biometrics_name} in {metric_unit}, " f"which is the angle between 2 lines: " f"(line 1) the line connecting {l1p1} and {l1p2}, " f"(line 2) the line connecting {l2p1} and {l2p2}.\n"
    else:
        return f"estimate the angle between 2 lines in {metric_unit}: " f"(line 1) the line connecting {l1p1} and {l1p2}, " f"(line 2) the line connecting {l2p1} and {l2p2}.\n"


def _get_biometric_prompt_distance(biometrics_name, p1, p2, metric_unit):
    """Prepare prompt for distance estimate VQA. Inputs are names."""
    metric_unit = metric_unit.strip().replace("mm", "millimeters")
    if biometrics_name is not None and biometrics_name != "":
        return f"estimate the distance of {biometrics_name} in {metric_unit}, " f"which is the distance between 2 landmark points: " f"(landmark 1) {p1}, " f"(landmark 2) {p2}.\n"
    else:
        return f"estimate the distance between 2 landmark points in {metric_unit}: " f"(landmark 1) {p1}, " f"(landmark 2) {p2}.\n"


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
        metric_unit = biometric_profile["metric_unit"]

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
        model_name = lmms_eval_specific_kwargs.get("model") if lmms_eval_specific_kwargs is not None else None
        assert model_name is not None, "Missing lmms_eval_specific_kwargs: 'model', check the base yaml file for this task."
        img_shape_resized_hw = get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs)

        # Adjust pixel size based on the resize ratio
        original_height, original_width = img_shape_hw
        pixel_height, pixel_width = pixel_size_hw
        resize_ratio_h = img_shape_resized_hw[0] / original_height
        resize_ratio_w = img_shape_resized_hw[1] / original_width
        adjusted_pixel_height = pixel_height / resize_ratio_h
        adjusted_pixel_width = pixel_width / resize_ratio_w
        # Include pixel size information in question text
        pixel_size_text = f"The pixel size for this image is {adjusted_pixel_width:.3f} mm (width) x {adjusted_pixel_height:.3f} mm (height)."
        # -------------

        # Question
        if metric_type == "distance":
            lines_map = task_info[metric_map_name]
            line_dict = lines_map[metric_key]
            lms_map_name = line_dict["element_map_name"]
            lms_map = task_info[lms_map_name]
            lms = line_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
            p1_name = lms_map[lms[0]]
            p2_name = lms_map[lms[1]]
            biometrics_name = line_dict["name"]
            task_prompt = _get_biometric_prompt_distance(biometrics_name, p1_name, p2_name, metric_unit)
        if metric_type == "angle":
            angles_map = task_info[metric_map_name]
            angle_dict = angles_map[metric_key]
            lines_map_name = angle_dict["element_map_name"]
            line_keys = angle_dict["element_keys"]  # list of 2 strings -- names of lines
            lines_map = task_info[lines_map_name]
            line1_dict = lines_map[line_keys[0]]
            line1_lms = line1_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
            line1_lms_map_name = line1_dict["element_map_name"]
            line1_lms_map = task_info[line1_lms_map_name]
            line1_p1_name = line1_lms_map[line1_lms[0]]
            line1_p2_name = line1_lms_map[line1_lms[1]]
            line2_dict = lines_map[line_keys[1]]
            line2_lms = line2_dict["element_keys"]  # list of 2 strings -- names of points (landmarks)
            line2_lms_map_name = line2_dict["element_map_name"]
            line2_lms_map = task_info[line2_lms_map_name]
            line2_p1_name = line2_lms_map[line2_lms[0]]
            line2_p2_name = line2_lms_map[line2_lms[1]]
            biometrics_name = angle_dict["name"]
            task_prompt = _get_biometric_prompt_angle(biometrics_name, line1_p1_name, line1_p2_name, line2_p1_name, line2_p2_name, metric_unit)

        question = f"Task:\n" f"Given the input medical image: {image_description}, " f"{task_prompt}" f"Additional information:\n" f"{pixel_size_text}\n" f"Format requirement:\n" f"{FORMAT_PROMPT_BIOMETRICS}"
        return question

    return doc_to_text_BiometricsFromLandmarks


def doc_to_target_BoxCoordinate(doc, lmms_eval_specific_kwargs=None):
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
    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]

    # FIXME: debug if this function accept lmms_eval_specific_kwargs
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
    pred = parser_last_k_nums(pred, 4)  # return 4 numbers separated by comma, or empty string
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
            mean_relative_error = np.mean(absolute_error / target_metric)
            success = True
    except:
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
    pred = parser_last_k_nums(pred, 2)  # return 2 numbers separated by comma, or empty string
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
            mean_relative_error = np.mean(absolute_error / target_metric)
            success = True
    except:
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
    except:
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
    except:
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
        if result["success"] == True:
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
        if result["success"] == True:
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
        if result["success"] == True:
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
        if result["success"] == True:
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
        if result["success"] == True:
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
        # Use scipy.ndimage.zoom for resizing
        image_2d = zoom(image_2d, zoom_factors, order=1)  # order=1 for bilinear interpolation

        # Update pixel size based on zoom factors
        pixel_size_hw = (pixel_size_hw[0] / zoom_factors[0], pixel_size_hw[1] / zoom_factors[1])

    return (pixel_size_hw, image_2d)


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
