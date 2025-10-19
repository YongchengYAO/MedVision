import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger as eval_logger
from typing import List, Optional, Tuple

from accelerate import Accelerator, DistributedType

from transformers import LlamaTokenizer, logging
logging.set_verbosity_error()

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# NOTE: This is a workaround for the issue with the import of the MedDr module
dir_meddr = os.environ.get("MedDr_DIR")
sys.path.append(dir_meddr)
from src.model.internvl_chat import InternVLChatModel
from src.dataset.transforms import build_transform

IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'


@register_model("meddr")
class MedDr(lmms):
    """
    LLaVA-Med Model
    """

    def __init__(
        self,
        model_path: str = "Sunanhe/MedDr_0401",
        dtype: str = "FP16",
        attn_implementation: str = "flash_attention_2",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.dtype = dtype
        self.attn_implementation = attn_implementation
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

        self.model_dtype = torch.float32 if self.dtype == "FP32" else (torch.float16 if self.dtype == "FP16" else torch.bfloat16)

        # Add loading progress information
        eval_logger.info(f"Loading base model from {self.model_path}...")

        # Optimize model loading with better device_map and config
        load_config = {
            "low_cpu_mem_usage": True,
            "use_safetensors": True,  # Prioritize safetensors format if available
            "attn_implementation": self.attn_implementation,
            "torch_dtype": self.model_dtype,
        }
        if self.device_map == "auto":
            load_config["device_map"] = "auto"
        elif self.accelerator.num_processes > 1:
            # For distributed training, use local device
            load_config["device_map"] = {"": self.device}
        else:
            load_config["device_map"] = {"": self.device}

        # Load model
        self._tokenizer = LlamaTokenizer.from_pretrained(self.model_path, **load_config)
        self._model = InternVLChatModel.from_pretrained(self.model_path, low_cpu_mem_usage=True, torch_dtype=self.model_dtype).eval()
        image_size = self._model.config.force_image_size or self._model.config.vision_config.image_size
        pad2square = self._model.config.pad2square
        img_context_token_id = self._tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._model.img_context_token_id = img_context_token_id
        self._image_processor = build_transform(is_train=False, input_size=image_size, pad2square=pad2square)

        # Set up model
        if self.device_map == "auto":
            self._model.to(self.model_dtype).to("cuda")
        else:
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
            response = self.eval_model(question=contexts, pil_img=visual)
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for BiomedGPT")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for BiomedGPT")

    # Modified from the source:
    # https://github.com/sunanhe/MedDr/blob/main/demo.py
    def eval_model(self, question: str, pil_img: Image.Image) -> str:
        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        image = self._image_processor(pil_img).unsqueeze(0).to(self._device).to(self.model_dtype)
        with torch.no_grad():
            response = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=image,
                question=question,
                generation_config=generation_config,
                print_out=False
            )
        return response
