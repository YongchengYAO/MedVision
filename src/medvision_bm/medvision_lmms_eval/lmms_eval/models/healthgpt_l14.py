import os, sys
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger as eval_logger
from typing import List, Optional, Tuple

import transformers
from transformers.utils import logging

logging.set_verbosity_info()

import tokenizers
from accelerate import Accelerator, DistributedType

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# NOTE: This is a workaround to import package from local folders
dir_healthgpt = os.environ.get("HEALTHGPT_DIR")
dir_demo = os.path.join(dir_healthgpt, "llava", "demo")
sys.path.append(dir_healthgpt)
sys.path.append(dir_demo)
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib
from llava.peft import LoraConfig, get_peft_model
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_phi3 import LlavaPhiForCausalLM, LlavaPhiConfig
from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
from utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, expand2square, com_vision_args


@register_model("healthgpt_l14")
class HealthGPT_L14(lmms):
    """
    HealthGPT Model

    Github:
        - https://github.com/DCDmllm/HealthGPT

    (Default)
    HealthGPT-XL32:
        Base Model: Qwen2.5-32B-Instruct
            - https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
        Vision Transformer for image processing: clip-vit-large-patch14-336
            - https://huggingface.co/openai/clip-vit-large-patch14-336
        HLORA Weights:
            - https://huggingface.co/lintw/HealthGPT-XL32/tree/main

    """

    def __init__(
        self,
        base_model_hf: str = "microsoft/phi-4",
        vision_model_hf: str = "openai/clip-vit-large-patch14-336",
        hlora_weights_local: str = None,
        dtype: str = "FP16",
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
        max_new_tokens: int = 1024,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_model_hf = base_model_hf
        self.vision_model_hf = vision_model_hf
        self.hlora_weights_local = hlora_weights_local
        self.dtype = dtype
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
        self.prepare_model(device, device_map)

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

    def prepare_model(self, device, device_map):
        # Set up accelerator
        self.accelerator = Accelerator()
        if self.accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            self.device_map = f"cuda:{self.accelerator.local_process_index}"
        elif self.accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
            self.device_map = f"cuda:{self.accelerator.local_process_index}"

        assert self.dtype in ["FP16", "FP32", "BF16"], ValueError(f"Unsupported dtype: {self.dtype}, should be one of [FP16, FP32, BF16]")
        if self.dtype == "BF16":
            model_dtype = torch.bfloat16
        elif self.dtype == "FP16":
            model_dtype = torch.float16
        elif self.dtype == "FP32":
            model_dtype = torch.float32

        # Optimize model loading with better device_map and config
        load_config = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,  # Prioritize safetensors format if available
            "attn_implementation": self.attn_implementation,
            "torch_dtype": model_dtype,
        }
        if self.device_map == "auto":
            load_config["device_map"] = "auto"
        elif self.accelerator.num_processes > 1:
            # For distributed training, use local device
            load_config["device_map"] = {"": self.device}
        else:
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
        self._model.get_vision_tower().to(dtype=model_dtype)

        self._model = load_weights(self._model, self.hlora_weights_local)
        self._model.eval()

        # Set up model
        if self.device_map == "auto":
            self._model.to(model_dtype).to("cuda")
        else:
            self._model.to(model_dtype).to(self.device)
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
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

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
            response = self.infer(question=contexts, pil_img=visual)
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
    ):
        model_dtype = torch.float32 if self.dtype == "FP32" else (torch.float16 if self.dtype == "FP16" else torch.bfloat16)

        if pil_img:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        else:
            qs = question
        conv = conversation_lib.conv_templates[self.instruct_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").cuda().unsqueeze_(0)
        if pil_img:
            image = pil_img.convert("RGB")
            image = expand2square(image, tuple(int(x * 255) for x in self._model.get_vision_tower().image_processor.image_mean))
            image_tensor = self._model.get_vision_tower().image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze_(0)
        with torch.inference_mode():
            output_ids = self._model.base_model.model.generate(
                input_ids,
                images=image_tensor.to(dtype=model_dtype, device="cuda", non_blocking=True) if pil_img else None,
                image_sizes=image.size if pil_img else None,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        response = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response
