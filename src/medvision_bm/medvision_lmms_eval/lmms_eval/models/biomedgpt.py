from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator, DistributedType
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.device_utils import setup_device_with_accelerate
from loguru import logger as eval_logger
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import OFAModel, OFATokenizer


@register_model("biomedgpt")
class BiomedGPT(lmms):
    """
    BiomedGPT Model
    "https://huggingface.co/PanaceaAI"

    There are two versions of the model:
    1. `PanaceaAI/BiomedGPT-Base-Pretrained`: The base model, which is a general-purpose model.
    2. `PanaceaAI/instruct-biomedgpt-base`: The instruction-tuned version of the base model.

    # NOTE: USE this model with caution, as discussed here: https://github.com/taokz/BiomedGPT/issues/39#issuecomment-2374711794
    """

    def __init__(
        self,
        pretrained: str = "PanaceaAI/BiomedGPT-Base-Pretrained",
        **kwargs,
    ) -> None:
        super().__init__()

        # Set up accelerator and device assignment using standard practice
        accelerator = Accelerator()
        self._device, self.device_map, self._rank, self._world_size = setup_device_with_accelerate(accelerator)

        # Set up model and tokenizer
        self._tokenizer = OFATokenizer.from_pretrained(pretrained)

        # NOTE: USE this model with caution, as discussed here: https://github.com/taokz/BiomedGPT/issues/39#issuecomment-2374711794
        self._model = OFAModel.from_pretrained(pretrained, ignore_mismatched_sizes=True)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)

        # Set up image processor
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        self.patch_resize_transform = transforms.Compose([lambda image: image.convert("RGB"), transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

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
                patch_img = self.patch_resize_transform(visual).unsqueeze(0)
            else:
                raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")

            # Text inputs
            text_tokens = self._tokenizer([contexts], return_tensors="pt").input_ids

            # Prepare text and image tensors, move to device
            text_tokens = text_tokens.to(self.device)
            patch_img = patch_img.to(self.device)

            # Get model outputs
            # https://colab.research.google.com/drive/1AMG-OwmDpnu24a9ZvCNvZi3BZwb3nSfS?usp=sharing#scrollTo=WgLUTdMIuUb_
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            gen = self._model.generate(text_tokens, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3, max_length=gen_kwargs["max_new_tokens"])
            response = self._tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
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
