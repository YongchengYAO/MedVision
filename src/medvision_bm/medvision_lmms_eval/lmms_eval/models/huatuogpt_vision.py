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
        max_new_tokens: int = 4096,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.stop_strings: List[str] = json.loads(stop_strings) if stop_strings else []
        # Same model-arg pattern as llava_med: the task YAMLs declare no
        # generation_kwargs, so the decode budget must come from the wrapper or
        # it silently falls through to HuatuoChatbot's internal 512 -- which
        # starved every CoT response (64% of this model's non-answers end
        # mid-word). Only the budget is corrected by default; the upstream
        # decoding method (do_sample=True, temperature=0.2) is the model's
        # intended recipe and is kept.
        self.max_new_tokens = max_new_tokens
        # The decoding-mode switch lives HERE, on model_args, and not in the
        # per-request gen_kwargs on purpose: lmms-eval injects
        # {"do_sample": False} into every task that declares no
        # generation_kwargs, so a request-level switch cannot tell an explicit
        # greedy ask from that injected default. model_args carry no injected
        # defaults -- absent means "upstream recipe", present means the
        # operator chose. E.g. --model_args model_hf=...,do_sample=False for
        # greedy, or do_sample=True,temperature=0.7 for hotter sampling.
        self.do_sample = do_sample
        self.temperature = temperature
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
        # Set the decode budget while (by default) KEEPING the upstream decoding
        # method (do_sample=True, temperature=0.2). The 512-token budget is the
        # defect; the sampling recipe is the model's intended usage. Keys are
        # updated in place so the stopping criteria added afterwards share the
        # same dict. Without this, HuatuoChatbot.inference() generates under the
        # third-party 512 budget no matter what the task or launcher declare.
        self.huatuo_chatbot.gen_kwargs["max_new_tokens"] = self.max_new_tokens
        self._apply_decoding_mode()

    def _apply_decoding_mode(self):
        """Resolve the operator's do_sample/temperature model-args, if given.

        Both absent (the default) leaves the upstream recipe untouched. The
        resolution follows the llava_med convention -- ``do_sample`` derived
        from ``temperature`` when only one is given, and an explicit
        ``temperature <= 0`` always means greedy, matching how the resume-cache
        check reads temperature 0 elsewhere in this file.
        """
        if self.do_sample is None and self.temperature is None:
            return
        gk = self.huatuo_chatbot.gen_kwargs
        do_sample = self.do_sample
        if self.temperature is not None and float(self.temperature) <= 0:
            do_sample = False
        elif do_sample is None:
            do_sample = float(self.temperature) > 0
        if do_sample:
            gk["do_sample"] = True
            if self.temperature is not None:
                gk["temperature"] = float(self.temperature)
        else:
            # Drop temperature entirely rather than passing 0: HF generate
            # warns on temperature with do_sample=False.
            gk["do_sample"] = False
            gk.pop("temperature", None)

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

    # Generation kwargs forwarded from the task YAML into the chatbot. "until" is
    # deliberately NOT forwarded: stop strings are wired explicitly via
    # _setup_stopping_criteria, and the YAML's inherited until=["\n\n"] would
    # truncate every CoT response at its first blank line.
    _FORWARDED_GEN_KEYS = ("max_new_tokens", "min_new_tokens", "top_p",
                           "num_beams", "repetition_penalty")

    def _apply_gen_kwargs(self, gen_kwargs):
        """Push the task's generation kwargs into the chatbot's gen_kwargs.

        HuatuoChatbot.inference() reads only its OWN gen_kwargs dict, whose
        upstream default budget is max_new_tokens=512. Before this hook the
        per-request kwargs unpacked in generate_until never reached generate(),
        so every response was cut at 512 tokens regardless of what the task
        declared.

        ``do_sample``/``temperature`` are deliberately NOT forwarded: lmms-eval
        injects ``{"do_sample": False}`` as the default for every task that
        declares no generation_kwargs, so honouring it here would silently flip
        the model from its intended sampling recipe (do_sample=True,
        temperature=0.2) to greedy on every run -- there is no way to tell that
        injected default apart from a task that genuinely asks for greedy.

        Keys are updated in place, never by replacing the dict: the stopping
        criteria installed by _setup_stopping_criteria live in the same dict and
        must survive.
        """
        gk = self.huatuo_chatbot.gen_kwargs
        for k in self._FORWARDED_GEN_KEYS:
            if k in gen_kwargs:
                gk[k] = gen_kwargs[k]

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
            self._apply_gen_kwargs(gen_kwargs)
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
