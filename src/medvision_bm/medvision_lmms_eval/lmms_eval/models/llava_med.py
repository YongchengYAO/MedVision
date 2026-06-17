import json
import os
from typing import List, Optional, Tuple

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import logging, set_seed

logging.set_verbosity_error()

from accelerate import Accelerator, DistributedType
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from medvision_bm.utils.configs import SEED
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.device_utils import setup_device_with_accelerate


@register_model("llava_med")
class LLaVA_Med(lmms):
    """
    LLaVA-Med Model

    dtype: BF16 (https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/main/config.json)

    Generation note: greedy decoding (temperature=0) on the mistral_instruct
    template can collapse to an immediate EOS (empty output) on longer prompts;
    eval_model() sets min_new_tokens to work around this. See the comment there
    and https://github.com/haotian-liu/LLaVA/issues/1363.
    """

    def __init__(
        self,
        model_hf: str = "microsoft/llava-med-v1.5-mistral-7b",
        model_base: str = None,
        conv_mode: str = "mistral_instruct",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        max_new_tokens: int = 4096,
        stop_strings: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.stop_strings: List[str] = json.loads(stop_strings) if stop_strings else []
        self.model_dtype = torch.bfloat16 # use model dtype (https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/main/config.json)
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

        # Load model
        set_seed(SEED)
        disable_torch_init()
        model_path = os.path.expanduser(self.model_hf)
        model_name = get_model_name_from_path(model_path)
        self._tokenizer, self._model, self._image_processor, self.context_len = load_pretrained_model(model_path, self.model_base, model_name)

        # Set up model — device placement is handled by load_pretrained_model;
        # move to the correct dtype and device explicitly
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
            # resume: skip already-finished samples (greedy decoding only)
            _greedy = not (gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0))
            _key = self._resp_cache_key(doc_id, task, split, contexts)
            if _greedy:
                _cached = self.resp_cache_get(task, _key)
                if _cached is not None:
                    res.append(_cached)
                    pbar.update(1)
                    continue

            # Image inputs
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if len(visuals) == 1 and isinstance(visuals[0], Image.Image):
                visual = visuals[0]
            else:
                raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")

            # Get model outputs
            response = self.eval_model(question=contexts, pil_img=visual)
            if _greedy:
                self.resp_cache_put(task, _key, response)
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

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        image_tensor = process_images([pil_img], self._image_processor, self._model.config)[0]
        # The model expects images in BF16 on the same device, so convert and move the image tensor accordingly
        image_tensor = image_tensor.unsqueeze(0).to(torch.bfloat16).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str] + self.stop_strings
        stopping_criteria = KeywordsStoppingCriteria(keywords, self._tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                # --- Empty-response workaround: force a minimum generation length ---
                # LLaVA-Med uses the `mistral_instruct` conversation template, which is
                # prone to degenerate/empty *greedy* generation: on longer prompts (e.g.
                # the CoT measurement tasks) greedy decoding (temperature=0) emits an
                # immediate EOS as its first real step. The raw output is just
                # `<s> ▁ </s>`, which decodes to "" after skip_special_tokens + .strip(),
                # so the sample is scored as a failure even though the model is capable —
                # suppressing EOS for the first few steps yields a correct, parseable CoT
                # answer. We deliberately KEEP deterministic greedy decoding for benchmark
                # reproducibility (the benchmark standardizes temperature=0 across all
                # models) and instead force a minimum of 16 new tokens to escape the
                # premature-EOS basin. Every legitimate AD/TL/detection answer is far
                # longer than 16 tokens, so this never truncates a real response.
                # Refs:
                #  - LLaVA issue #1363 (mistral_instruct empty output):
                #    https://github.com/haotian-liu/LLaVA/issues/1363
                #  - Reference inference (third_party/LLaVA-Med/llava/eval/model_vqa.py)
                #    uses temperature=0.2 + do_sample=True; the benchmark uses greedy.
                min_new_tokens=16,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        for stop_str in self.stop_strings:
            idx = outputs.find(stop_str)
            if idx != -1:
                outputs = outputs[:idx + len(stop_str)]
                break
        return outputs
