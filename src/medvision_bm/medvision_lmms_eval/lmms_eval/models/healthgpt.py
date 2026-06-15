import os
import sys
from typing import List, Optional, Tuple

import torch
import transformers
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers.utils import logging

logging.set_verbosity_info()

import tokenizers
from accelerate import Accelerator, DistributedType
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.device_utils import setup_device_with_accelerate

# NOTE: This is a workaround to import package from local folders
dir_healthgpt = os.environ.get("HEALTHGPT_DIR")
dir_demo = os.path.join(dir_healthgpt, "llava", "demo")
sys.path.append(dir_healthgpt)
sys.path.append(dir_demo)
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM
from llava.peft import LoraConfig, get_peft_model
from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
from utils import (
    add_special_tokens_and_resize_model,
    com_vision_args,
    expand2square,
    find_all_linear_names,
    load_weights,
)


@register_model("healthgpt")
class HealthGPT(lmms):
    """
    HealthGPT Model

    Github:
        - https://github.com/DCDmllm/HealthGPT

    HealthGPT-L14:
        Base Model: phi-4
            - https://huggingface.co/microsoft/phi-4
        Vision Transformer for image processing: clip-vit-large-patch14-336
            - https://huggingface.co/openai/clip-vit-large-patch14-336
        HLORA Weights:
            - https://huggingface.co/lintw/HealthGPT-L14/tree/main
        dtype: bfloat16

    HealthGPT-XL32:
        Base Model: Qwen2.5-32B-Instruct
            - https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
        Vision Transformer for image processing: clip-vit-large-patch14-336
            - https://huggingface.co/openai/clip-vit-large-patch14-336
        HLORA Weights:
            - https://huggingface.co/lintw/HealthGPT-XL32/tree/main
        dtype: float16

    dtype is variant-specific (L14 bfloat16, XL32 float16) and is set via the
    `model_dtype` arg. (https://github.com/ZJU4HealthCare/HealthGPT)
    """

    _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    def __init__(
        self,
        base_model_hf: str = "microsoft/phi-4",
        vision_model_hf: str = "openai/clip-vit-large-patch14-336",
        hlora_weights_local: str = None,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        hlora_r: int = 32,
        hlora_alpha: int = 64,
        hlora_dropout: float = 0.0,
        hlora_nums: int = 4,
        vq_idx_nums: int = 8192,
        instruct_template: str = "phi4_instruct",
        fusion_layer_path: str = None,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = None,
        num_beams: int = 1,
        max_new_tokens: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_model_hf = base_model_hf
        self.vision_model_hf = vision_model_hf
        self.hlora_weights_local = hlora_weights_local
        if model_dtype not in self._DTYPE_MAP:
            raise ValueError(f"Unsupported model_dtype: {model_dtype}. Choose from {list(self._DTYPE_MAP)}.")
        self.model_dtype = self._DTYPE_MAP[model_dtype]
        self.attn_implementation = attn_implementation
        self.hlora_r = hlora_r
        self.hlora_alpha = hlora_alpha
        self.hlora_dropout = hlora_dropout
        self.hlora_nums = hlora_nums
        self.vq_idx_nums = vq_idx_nums
        self.instruct_template = instruct_template
        self.fusion_layer_path = fusion_layer_path
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.prepare_model()

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def prepare_model(self):
        # Set up accelerator and device assignment using standard practice
        self.accelerator = Accelerator()
        self._device, self.device_map, self._rank, self._world_size = setup_device_with_accelerate(self.accelerator)

        # Optimize model loading with better device_map and config
        load_config = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,  # Prioritize safetensors format if available
            "attn_implementation": self.attn_implementation,
            "torch_dtype": self.model_dtype,
        }
        # Always load to the specific device to avoid cross-device issues
        load_config["device_map"] = {"": self.device}

        # Add loading progress information
        eval_logger.info(f"Loading base model from {self.base_model_hf}...")

        self._model = LlavaPhiForCausalLM.from_pretrained(pretrained_model_name_or_path=self.base_model_hf, **load_config)

        lora_config = LoraConfig(
            r=self.hlora_r,
            lora_alpha=self.hlora_alpha,
            target_modules=find_all_linear_names(self._model),
            lora_dropout=self.hlora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            lora_nums=self.hlora_nums,
        )
        self._model = get_peft_model(self._model, lora_config)

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model_hf,
            padding_side="right",
            use_fast=False,
        )
        num_new_tokens = add_special_tokens_and_resize_model(self._tokenizer, self._model, self.vq_idx_nums)
        print(f"Number of new tokens added for unified task: {num_new_tokens}")

        com_vision_args.model_name_or_path = self.base_model_hf
        com_vision_args.vision_tower = self.vision_model_hf
        com_vision_args.version = self.instruct_template

        self._model.get_model().initialize_vision_modules(model_args=com_vision_args)
        self._model.get_vision_tower().to(dtype=self.model_dtype)

        self._model = load_weights(self._model, self.hlora_weights_local)
        self._model.eval()

        # Set up model on the target device
        self._model.to(self.model_dtype).to(self.device)
        if self.accelerator.num_processes > 1:
            assert self.accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self._model = self.accelerator.prepare(self._model)
            else:
                self._model = self.accelerator.prepare_model(self._model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {self.accelerator.num_processes} devices with data parallelism")
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Image inputs
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if len(visuals) == 1 and isinstance(visuals[0], Image.Image):
                visual = visuals[0]
            else:
                raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")

            # Get model outputs
            response = self.infer(question=contexts, pil_img=visual, max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens))
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for BiomedGPT")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for BiomedGPT")

    # Modified from the source:
    # https://github.com/DCDmllm/HealthGPT/blob/main/llava/demo/com_infer_phi4.py
    # @commit: ce1589daac41fa81c4868e7958ca9b9ff332a85d
    def infer(
        self,
        question: str = None,
        pil_img: str = None,
        max_new_tokens: int = None,
    ):
        _max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        if pil_img is not None:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        else:
            qs = question
        conv = conversation_lib.conv_templates[self.instruct_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device).unsqueeze_(0)
        if pil_img is not None:
            image = pil_img.convert("RGB")
            image = expand2square(image, tuple(int(x * 255) for x in self._model.get_vision_tower().image_processor.image_mean))
            image_tensor = self._model.get_vision_tower().image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze_(0)
        with torch.inference_mode():
            output_ids = self._model.base_model.model.generate(
                input_ids,
                images=image_tensor.to(dtype=self.model_dtype, device=self.device, non_blocking=True) if pil_img is not None else None,
                image_sizes=image.size if pil_img is not None else None,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=_max_new_tokens,
                use_cache=True,
            )
        response = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
