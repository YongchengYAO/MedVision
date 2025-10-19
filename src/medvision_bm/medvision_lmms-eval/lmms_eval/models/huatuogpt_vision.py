import os
import sys
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple

from accelerate import Accelerator

from transformers import logging

logging.set_verbosity_error()

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# NOTE: This is a workaround for the issue with the import of HuatuoGPT-Vision modules
dir_huatuogpt_vision = os.environ.get("HuatuoGPTVision_DIR")
sys.path.append(dir_huatuogpt_vision)
from cli import HuatuoChatbot


@register_model("huatuogpt_vision")
class HuatuoGPT_Vision(lmms):
    """
    HuatuoGPT-Vision Model
    """

    def __init__(
        self,
        model_path: str = "FreedomIntelligence/HuatuoGPT-Vision-34B",
        dtype: str = "FP16",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.dtype = dtype
        self.prepare_model(device, device_map)

    @property
    def tokenizer(self):
        return self.huatuo_chatbot.model.tokenizer

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

        # Load model
        self.huatuo_chatbot = HuatuoChatbot(self.model_path, device=self._device)

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

            # Get model outputs
            response = self.huatuo_chatbot.inference(text=contexts, images=visuals)
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for BiomedGPT")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for BiomedGPT")
