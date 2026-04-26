import os
import sys
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import LlamaTokenizer, logging

logging.set_verbosity_error()

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.device_utils import setup_device_with_accelerate

# NOTE: This is a workaround for the issue with the import of the MedDr module
dir_meddr = os.environ.get("MedDr_DIR")
sys.path.append(dir_meddr)
from src.dataset.transforms import build_transform
from src.model.internvl_chat import InternVLChatModel

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


@register_model("meddr")
class MedDr(lmms):
    """
    LLaVA-Med Model

    dtype: BF16 (https://huggingface.co/Sunanhe/MedDr_0401/blob/main/config.json)
    """

    def __init__(
        self,
        model_hf: str = "Sunanhe/MedDr_0401",
        attn_implementation: str = "flash_attention_2",
        max_new_tokens: Optional[int] = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.model_dtype = torch.bfloat16 # use model dtype (https://huggingface.co/Sunanhe/MedDr_0401/blob/main/config.json)
        self.attn_implementation = attn_implementation
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

        # Add loading progress information
        eval_logger.info(f"Loading base model from {self.model_hf}...")

        # Optimize model loading with better device_map and config
        load_config = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,  # Prioritize safetensors format if available
            "attn_implementation": self.attn_implementation,
            "torch_dtype": self.model_dtype,
        }
        # Always load to the specific device to avoid cross-device issues
        load_config["device_map"] = {"":  self.device}

        # Load model
        self._tokenizer = LlamaTokenizer.from_pretrained(self.model_hf, **load_config)
        self._model = InternVLChatModel.from_pretrained(self.model_hf, low_cpu_mem_usage=True, torch_dtype=self.model_dtype).eval()
        image_size = self._model.config.force_image_size or self._model.config.vision_config.image_size
        pad2square = self._model.config.pad2square
        img_context_token_id = self._tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._model.img_context_token_id = img_context_token_id
        self._image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)

        # Set up model and move to device
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
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = self.max_new_tokens

            response = self.eval_model(question=contexts, pil_img=visual, max_new_tokens=gen_kwargs.get("max_new_tokens"))
            res.append(response)
            torch.cuda.empty_cache()
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for BiomedGPT")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for BiomedGPT")

    # Modified from the source:
    # https://github.com/sunanhe/MedDr/blob/main/demo.py
    def eval_model(self, question: str, pil_img: Image.Image, max_new_tokens: int = 4096) -> str:
        generation_config = dict(
            num_beams=1,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        image = self._image_processor(pil_img).unsqueeze(0).to(self._device).to(self.model_dtype)
        with torch.no_grad():
            response = self._model.chat(tokenizer=self._tokenizer, pixel_values=image, question=question, generation_config=generation_config, print_out=False)
        return response
