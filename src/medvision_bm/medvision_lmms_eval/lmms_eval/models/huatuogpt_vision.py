import json
import os
import sys
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList, logging

logging.set_verbosity_error()

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.device_utils import setup_device_with_accelerate

# NOTE: This is a workaround for the issue with the import of HuatuoGPT-Vision modules
dir_huatuogpt_vision = os.environ.get("HuatuoGPTVision_DIR")
sys.path.append(dir_huatuogpt_vision)
from cli import HuatuoChatbot


class _KeywordsStoppingCriteria(StoppingCriteria):
    """Stop generation once any keyword string appears in the newly generated text.

    Only the tokens generated *after* the prompt are decoded and checked, so a
    keyword that also occurs in the prompt (e.g. the ``</answer>`` tag the model
    is instructed to emit) does not trigger an immediate stop. A single instance
    is shared across all samples via the chatbot's ``gen_kwargs``, so ``reset()``
    must be called before every generation.
    """

    def __init__(self, keywords: List[str], tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self._start_len = None

    def reset(self):
        self._start_len = None

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        # On the first decoding step, everything seen so far is the prompt plus
        # the single token just generated; remember where generation starts.
        if self._start_len is None:
            self._start_len = max(input_ids.shape[1] - 1, 0)
        for seq in input_ids:
            text = self.tokenizer.decode(
                seq[self._start_len:], skip_special_tokens=True
            )
            if not any(kw in text for kw in self.keywords):
                return False
        return True


@register_model("huatuogpt_vision")
class HuatuoGPT_Vision(lmms):
    """
    HuatuoGPT-Vision Model
    """

    def __init__(
        self,
        model_hf: str = "FreedomIntelligence/HuatuoGPT-Vision-34B",
        stop_strings: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.stop_strings: List[str] = json.loads(stop_strings) if stop_strings else []
        self.prepare_model()
        self._setup_stopping_criteria()

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

    def prepare_model(self):
        # Set up accelerator and device assignment using standard practice
        self.accelerator = Accelerator()
        self._device, self.device_map, self._rank, self._world_size = setup_device_with_accelerate(self.accelerator)
        # Load model
        self.huatuo_chatbot = HuatuoChatbot(self.model_hf, device=self._device)

    def _setup_stopping_criteria(self):
        # Wire optional stop strings into the chatbot's generation kwargs so that
        # `self.model.generate(..., **gen_kwargs)` halts at the first match.
        self._stopping_criteria = None
        if self.stop_strings:
            self._stopping_criteria = _KeywordsStoppingCriteria(
                self.stop_strings, self.huatuo_chatbot.tokenizer
            )
            self.huatuo_chatbot.gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [self._stopping_criteria]
            )

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

            # Get model outputs
            if self._stopping_criteria is not None:
                self._stopping_criteria.reset()
            response = self.huatuo_chatbot.inference(text=contexts, images=visuals)
            if _greedy:
                self.resp_cache_put(task, _key, response)
            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for HuatuoGPT-Vision")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for HuatuoGPT-Vision")
