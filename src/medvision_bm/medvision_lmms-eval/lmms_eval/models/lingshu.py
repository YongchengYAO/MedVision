import os
import base64
from io import BytesIO
import concurrent.futures
from functools import partial
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import eval_logger

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedType, wait_for_everyone
from accelerate import PartialState

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig


@register_model("lingshu")
class Lingshu(lmms):
    """
    Lingshu Model

    - HF: https://huggingface.co/lingshu-medical-mllm/Lingshu-32B
    """

    def __init__(
        self,
        model_hf: str = "lingshu-medical-mllm/Lingshu-32B",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attention_2: Optional[bool] = False,
        max_new_tokens: int = 300,
        num_workers: int = 8,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self._batch_size = int(batch_size)
        self.use_flash_attention_2 = use_flash_attention_2
        self.max_new_tokens = max_new_tokens
        self.num_workers = num_workers
        self.model_dtype = torch.bfloat16
        self.prepare_model(device, device_map)

    @property
    def model(self):
        # Returns model, unwrapping if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self._batch_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _get_base_device(self):
        # Compatible with DP & MP (device_map)
        return next(self._model.parameters()).device

    def prepare_model(self, device: Optional[str] = "cuda", device_map: Optional[str] = "auto"):
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_hf,
            torch_dtype=self.model_dtype,
            attn_implementation="flash_attention_2" if self.use_flash_attention_2 else "eager",
            device_map=device_map,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_hf)
        self._processor.tokenizer.padding_side = "left" 

        self._device = torch.device(device)
        self.device_map = device_map

        if device_map == "auto":
            eval_logger.info("Initialized with model parallelism (device_map='auto').")
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)

        self._rank = 0
        self._world_size = 1

    def __pil_img_to_base64(self, pil_img: Image.Image) -> str:
        base64_image = pil_img.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

    # Code adapted from https://huggingface.co/lingshu-medical-mllm/Lingshu-32B
    def infer(self, questions: Union[str, List[str]], pil_imgs: Union[Image.Image, List[Image.Image]]) -> Union[str, List[str]]:
        # Normalize inputs to lists
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(pil_imgs, Image.Image):
            pil_imgs = [pil_imgs]

        if len(questions) != len(pil_imgs):
            raise ValueError(f"Number of questions ({len(questions)}) must match number of images ({len(pil_imgs)})")

        if len(questions) == 0:
            return []

        # Prepare batch messages
        batch_messages = []
        for question, pil_img in zip(questions, pil_imgs):
            # NOTE: Lingshu is trained on Qwen2.5-VL, check accepted visual input format:
            #       - qwen2.5vl: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
            #       - qwen-vl-utils: https://pypi.org/project/qwen-vl-utils/0.0.11/

            # [option 1] Convert image to base64 string
            # base64_string = self.__pil_img_to_base64(pil_img)
            # messages = [{"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": question}]}]

            # [option 2] Use PIL image directly
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": question}]}]

            # NOTE: batch_messages must be a list of messages, where each messages is a list of dict
            batch_messages.append(messages)

        # Process messages for batch inference
        try:
            batch_texts = []
            all_image_inputs = []
            all_video_inputs = []

            for messages in batch_messages:
                text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                image_inputs, video_inputs = process_vision_info(messages)
                batch_texts.append(text)
                all_image_inputs.extend(image_inputs if image_inputs else [])
                all_video_inputs.extend(video_inputs if video_inputs else [])

            # Process entire batch with padding
            inputs = self._processor(
                text=batch_texts,
                images=all_image_inputs if all_image_inputs else None,
                videos=all_video_inputs if all_video_inputs else None,
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process batch messages: {e}")

        # Move inputs to device with proper dtype handling
        # Note: Cannot use inputs.to(device) - need per-tensor handling for different dtypes
        try:
            base_device = self._get_base_device()
            for key in inputs:
                if inputs[key] is not None:
                    if key == "attention_mask":
                        inputs[key] = inputs[key].to(base_device)  # Keep original dtype
                    elif key == "pixel_values":
                        inputs[key] = inputs[key].to(base_device, dtype=self.model_dtype)
                    else:
                        inputs[key] = inputs[key].to(base_device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to move inputs to device: {e}")

        # Generate with error handling
        try:
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("Out of memory during generation. Try reducing batch size.")
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

        # Extract new tokens only
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]

        # Batch decode responses
        try:
            responses = self._processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception as e:
            raise RuntimeError(f"Failed to decode responses: {e}")

        return responses

    def process_single_request(self, request: Instance) -> str:
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

        # Extract and validate visual input
        visual_doc = self.task_dict[task][split][doc_id]
        visuals = doc_to_visual(visual_doc)

        # Ensure single Image.Image input
        if isinstance(visuals, list):
            if len(visuals) == 1 and isinstance(visuals[0], Image.Image):
                visual = visuals[0]
            else:
                flattened = self.flatten([visuals]) if not isinstance(visuals[0], Image.Image) else visuals
                if len(flattened) == 1 and isinstance(flattened[0], Image.Image):
                    visual = flattened[0]
                else:
                    raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")
        elif isinstance(visuals, Image.Image):
            visual = visuals
        else:
            raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")

        return contexts, visual

    def process_batch_parallel(self, batch_requests):
        batch_contexts = [None] * len(batch_requests)
        batch_visuals = [None] * len(batch_requests)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_requests), self.num_workers)) as executor:
            process_func = partial(self.process_single_request)
            results = list(executor.map(process_func, batch_requests))

        for idx, (contexts, visual) in enumerate(results):
            batch_contexts[idx] = contexts
            batch_visuals[idx] = visual

        return batch_contexts, batch_visuals

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # Process requests in batches
        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i : i + self.batch_size]
            batch_size = len(batch_requests)

            # Use parallel processing for batch preparation
            batch_contexts, batch_visuals = self.process_batch_parallel(batch_requests)

            # Get batch model outputs
            batch_responses = self.infer(questions=batch_contexts, pil_imgs=batch_visuals)

            # Ensure batch_responses is a list
            if isinstance(batch_responses, str):
                batch_responses = [batch_responses]

            res.extend(batch_responses)
            pbar.update(batch_size)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MedGemma")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for MedGemma")
