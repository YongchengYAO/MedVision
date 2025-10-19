import os
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger as eval_logger
from typing import List, Optional, Tuple

from transformers import set_seed, logging
logging.set_verbosity_error()

from accelerate import Accelerator, DistributedType

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images


@register_model("llava_med")
class LLaVA_Med(lmms):
    """
    LLaVA-Med Model
    """

    def __init__(
        self,
        model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
        model_base: str = None,
        conv_mode: str = "mistral_instruct",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        dtype: str = "FP16",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.dtype = dtype
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

        model_dtype = torch.float32 if self.dtype == "FP32" else (torch.float16 if self.dtype == "FP16" else torch.bfloat16)

        # Add loading progress information
        eval_logger.info(f"Loading base model from {self.model_path}...")

        # Load model
        set_seed(0)
        disable_torch_init()
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        self._tokenizer, self._model, self._image_processor, self.context_len = load_pretrained_model(model_path, self.model_base, model_name)

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
            self.accelerator = self.accelerator
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
                raise ValueError("i 1 image input and it should be of Image.Image type.")

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
    # https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/model_vqa.py
    # @commit: 821e1c7
    def eval_model(self, question: str, pil_img: Image.Image) -> str:
        if self._model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates[self.conv_mode].copy()

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        image_tensor = process_images([pil_img], self._image_processor, self._model.config)[0]
        if self.dtype == "FP32":
            image_tensor = image_tensor.unsqueeze(0).cuda()
        elif self.dtype == "FP16":
            image_tensor = image_tensor.unsqueeze(0).half().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
