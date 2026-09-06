#!/usr/bin/env python3
"""scaffold_new_model.py -- generate the skeleton files for a new MedVision VLM wrapper.

Purpose
    Adding a VLM to the MedVision benchmark means touching the same set of files every
    time. This generator writes those files as TODO-marked skeletons into a scratch
    directory, mirroring the repository layout so each file can be reviewed and then
    copied to its real destination:

        lmms_eval/models/<key>.py                     model class + @register_model
        patches/AVAILABLE_MODELS.patch.txt            entry for lmms_eval/models/__init__.py
        patches/medvision_utils_dispatch.patch.txt    _process_img_<key>() + get_resized_img_shape branch
        src/medvision_bm/benchmark/eval__<key>.py     evaluation entry point
        script/benchmark-detect/eval__<Display>__detect.sh
        script/benchmark-TL/eval__<Display>__TL.sh
        script/benchmark-AD/eval__<Display>__AD.sh
        unit-test/<key>-image-resize/test_<key>_resize.py
        requirements/requirements_eval_<key>.txt

    The templates are distilled from the repository's own `vllm_qwen25vl.py` (vLLM),
    `medgemma.py` (HF + accelerate), `claude.py` (API), `eval__gemini.py` /
    `eval__qwen2_5_vl.py` and the `script/benchmark-*/eval__*.sh` launchers, trimmed to the
    load-bearing parts. Nothing is installed, downloaded or executed.

Prerequisites
    Python >= 3.8, standard library only. No MedVision install is required.

Usage
    scaffold_new_model.py --key vllm_mymodel --class-name VLLM_MyModel --kind vllm \
        --hf-id Org/MyModel-7B --out-dir ./scaffold --dry-run
    scaffold_new_model.py --key mymodel_api --class-name MyModelAPI --kind api \
        --hf-id vendor-model-1 --display-name MyModel-1 --out-dir ./scaffold

    --kind vllm : vLLM backend (tensor parallel, `--model_hf_id`, batch_size_per_gpu)
    --kind hf   : HuggingFace backend driven by `accelerate launch` (data parallel)
    --kind api  : hosted API model (caps table + client-side pre-resize, no GPU)

    By default the generator REFUSES to write inside a MedVision checkout (it is a scratch
    scaffold, not an in-place edit); pass --allow-checkout to override.

Exit codes
    0 = files written (or listed with --dry-run); 1 = write/IO error;
    2 = invalid arguments or refused destination.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CLASS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

TODO = "TODO"


# =========================================================================== templates
MODEL_VLLM = '''"""{TODO}: MedVision vLLM wrapper for @DISPLAY@ (model key "@KEY@").

Distilled from the repository's lmms_eval/models/vllm_qwen25vl.py. Keep the class name and
the @register_model key identical to the AVAILABLE_MODELS entry.
"""

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional, Tuple

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from vllm import LLM, SamplingParams


@register_model("@KEY@")
class @CLASS@(lmms):
    """@DISPLAY@ served by vLLM.

    {TODO}: document the checkpoint family, the vLLM version that registers this
    architecture, and any hf_overrides / dtype the model needs.
    """

    def __init__(
        self,
        model_hf: str = "@HFID@",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        # NOTE: an EXPLICIT output-token budget is mandatory. Without it the upstream
        # framework default (512) silently truncates chain-of-thought answers.
        max_new_tokens: int = 4096,
        threads: int = 16,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        stop_strings: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(model_hf, str) and model_hf != "", "model_hf must be an HF id or local path"
        self.model_hf = model_hf
        self.max_new_tokens = max_new_tokens
        self.threads = threads
        self.chat_template = chat_template
        self.stop_strings: List[str] = json.loads(stop_strings) if stop_strings else []

        # JSON-looking model args (e.g. hf_overrides={"architectures": [...]}) arrive as strings
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like argument '{key}': {value}")

        # MedVision-only kwargs must not reach vLLM
        kwargs.pop("reshape_image_hw", None)

        self.client = LLM(
            model=self.model_hf,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.batch_size_per_gpu = int(batch_size)

    def encode_image(self, image) -> str:
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.copy()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def flatten(self, nested):
        return [item for sub in nested for item in sub]

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), desc="Model Responding")

        # Resume: greedy responses are cached per sample by the base class
        pending = []
        for gi, req in enumerate(requests):
            _, gen_kwargs, _, doc_id, task, split = req.arguments
            if gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0):
                pending.append((gi, req))
                continue
            cached = self.resp_cache_get(task, self._resp_cache_key(doc_id, task, split, req.arguments[0]))
            if cached is not None:
                res[gi] = cached
                pbar.update(1)
            else:
                pending.append((gi, req))

        bs = self.batch_size_per_gpu
        for batch in [pending[i : i + bs] for i in range(0, len(pending), bs)]:
            batched_messages = []
            params = {}
            for _, req in batch:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments
                gen_kwargs.setdefault("max_new_tokens", self.max_new_tokens)
                gen_kwargs.setdefault("temperature", 0)
                gen_kwargs.setdefault("top_p", 0.95)
                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                }
                if self.stop_strings:
                    params["stop"] = list(dict.fromkeys(self.stop_strings))
                    params["include_stop_str_in_output"] = True

                visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
                with ThreadPoolExecutor(max_workers=self.threads) as ex:
                    imgs = [f.result() for f in [ex.submit(self.encode_image, v) for v in visuals]]

                # {TODO}: adapt the message layout if @DISPLAY@ needs a system prompt or a
                # different image/text ordering.
                messages = [{"role": "user", "content": [{"type": "text", "text": contexts}]}]
                for img in imgs:
                    messages[-1]["content"].append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    )
                batched_messages.append(messages)

            sampling_params = SamplingParams(**params)
            if self.chat_template is not None:
                tmpl = open(self.chat_template).read() if os.path.isfile(self.chat_template) else self.chat_template
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=tmpl)
            else:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages)

            for (gi, req), out in zip(batch, response):
                text = out.outputs[0].text
                _, gen_kwargs, _, doc_id, task, split = req.arguments
                if not (gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0)):
                    self.resp_cache_put(task, self._resp_cache_key(doc_id, task, split, req.arguments[0]), text)
                res[gi] = text
            pbar.update(len(batch))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for @CLASS@")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for @CLASS@")
'''

MODEL_HF = '''"""{TODO}: MedVision HuggingFace wrapper for @DISPLAY@ (model key "@KEY@").

Distilled from the repository's HF wrappers (lmms_eval/models/medgemma.py): the model is
replicated per process by `accelerate launch --num_processes=<visible GPUs>` (data
parallel), so the wrapper must expose rank / world_size from the Accelerator.
"""

from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator, DistributedType
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


@register_model("@KEY@")
class @CLASS@(lmms):
    """@DISPLAY@ loaded with transformers and driven by accelerate (data parallel)."""

    def __init__(
        self,
        model_hf: str = "@HFID@",
        batch_size: int = 1,
        # NOTE: an EXPLICIT output-token budget is mandatory (see the token-budget rule);
        # a wrapper that omits it inherits a third-party 512-token default silently.
        max_new_tokens: int = 4096,
        dtype: str = "bfloat16",
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        kwargs.pop("reshape_image_hw", None)  # MedVision-only kwarg
        self.model_hf = model_hf
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens

        accelerator = Accelerator()
        self.accelerator = accelerator
        device_map = {"": accelerator.local_process_index} if accelerator.num_processes > 1 else "auto"

        # {TODO}: replace with the correct AutoModel class / loader for @DISPLAY@.
        self._processor = AutoProcessor.from_pretrained(model_hf, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_hf,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU]
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        self._device = accelerator.device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, nested):
        return [item for sub in nested for item in sub]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            greedy = not (gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0)) if gen_kwargs else True
            key = self._resp_cache_key(doc_id, task, split, contexts)
            if greedy:
                cached = self.resp_cache_get(task, key)
                if cached is not None:
                    res.append(cached)
                    pbar.update(1)
                    continue

            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            if len(visuals) != 1 or not isinstance(visuals[0], Image.Image):
                raise ValueError("MedVision tasks provide exactly one PIL image per sample")

            # Per-request budget wins over the constructor default
            max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens)) if gen_kwargs else self.max_new_tokens

            # {TODO}: build the chat messages @DISPLAY@ expects and decode only the completion.
            inputs = self._processor(text=contexts, images=visuals[0], return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = self._processor.batch_decode(out[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)[0]

            if greedy:
                self.resp_cache_put(task, key, text)
            res.append(text)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for @CLASS@")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for @CLASS@")
'''

MODEL_API = '''"""{TODO}: MedVision API wrapper for @DISPLAY@ (model key "@KEY@").

Distilled from the repository's lmms_eval/models/claude.py. Two rules are load-bearing:

1. SUPPORTED_MODEL_CAPS is the SINGLE source of truth for this provider's image caps and
   resize rule. The task layer (medvision_utils.get_resized_img_shape) imports
   @KEY@_resized_hw() from this file, so the size stated in the T/L and A/D prompts can
   never drift from the image actually sent. An unverified model code must RAISE.
2. The image is pre-resized client-side so the provider's own resize AND padding are
   no-ops (perceived canvas == sent image == stated size).
"""

import base64
import io
import json
import math
import os
from typing import List, Optional, Tuple, Union

import backoff
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# {TODO}: replace with the real per-model caps read from the provider's official vision
# documentation, one entry per model code. DO NOT add a generic default: an unverified
# model would emit a wrong pixel size and silently corrupt every measurement.
# ---------------------------------------------------------------------------
SUPPORTED_MODEL_CAPS = {
    "@HFID@": (1568, 1568),  # (long_edge_cap_px, max_image_tokens)
}

# {TODO}: confirm the provider's patch/grid size. 28 is right for the 14x14-patch +
# 2x2-merge families; OpenAI-style patch models use 32.
_GRID = 28
# {TODO}: pixels per image token used by the provider's token formula.
_PX_PER_TOKEN = 750.0


def _normalize_model_code(model_code: str) -> str:
    """Strip an OpenRouter-style "<vendor>/" prefix so both id forms match one entry."""
    return model_code.split("/")[-1]


def @KEY@_image_caps(model_code: str) -> Tuple[int, int]:
    """Return (long_edge_cap_px, max_image_tokens); raise for an unverified model code."""
    normalized = _normalize_model_code(model_code)
    if normalized not in SUPPORTED_MODEL_CAPS:
        raise ValueError(
            f"[@KEY@] Unsupported model code {model_code!r} (normalized {normalized!r}). "
            f"Its image-resolution caps are not verified, so the image size / pixel size stated "
            f"in MedVision T/L and A/D prompts could be wrong. Look up the per-model limits in "
            f"the provider's vision documentation and add an entry to SUPPORTED_MODEL_CAPS in "
            f"lmms_eval/models/@KEY@.py (the single source of truth)."
        )
    return SUPPORTED_MODEL_CAPS[normalized]


def _floor_to_grid(x: float) -> int:
    return max(_GRID, (int(x) // _GRID) * _GRID)


def @KEY@_resized_hw(img_h: int, img_w: int, model_code: str) -> Tuple[int, int]:
    """Shape to send so the provider's resize and pad are both no-ops.

    {TODO}: verify this formula against the provider's documented pipeline before use.
    Never upscales. Raises for unsupported models (via @KEY@_image_caps).
    """
    long_edge_cap, max_img_tokens = @KEY@_image_caps(model_code)
    scale = min(
        1.0,
        long_edge_cap / max(img_h, img_w),
        math.sqrt(max_img_tokens * _PX_PER_TOKEN / (img_h * img_w)),
    )
    return _floor_to_grid(img_h * scale), _floor_to_grid(img_w * scale)


def _giveup_on_bad_request(e: Exception) -> bool:
    # 400s are deterministic; retrying only wastes credit.
    return getattr(e, "status_code", None) == 400


@register_model("@KEY@")
class @CLASS@(lmms):
    """@DISPLAY@ via its native API or an OpenAI-compatible gateway."""

    def __init__(
        self,
        model: str = "@HFID@",
        provider: str = "native",
        # NOTE: API wrappers use a LARGER explicit budget than local models because
        # reasoning tokens share it.
        max_tokens: Optional[int] = 16000,
        stop_strings: Optional[Union[List[str], str]] = None,
        **kwargs,  # absorbs model_hf / reshape_image_hw, consumed task-side by the evaluator
    ) -> None:
        super().__init__()
        self.model_code = model
        # Fail fast, before any task runs, if the caps are unverified.
        @KEY@_image_caps(model)
        if provider not in ["native", "openrouter"]:
            raise ValueError(f"Unsupported provider: {provider}. Use 'native' or 'openrouter'.")
        self.provider = provider
        self.max_tokens = int(max_tokens)
        self.stop_strings: List[str] = json.loads(stop_strings) if isinstance(stop_strings, str) else (stop_strings or [])
        self.prepare_model()

    def prepare_model(self):
        # NOTE: .strip() is required -- pod/k8s-injected secrets carry a trailing newline,
        # which is an illegal HTTP header value.
        import openai

        if self.provider == "openrouter":
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            )
        else:
            # {TODO}: swap in the vendor SDK and its API-key env var.
            self.client = openai.OpenAI(api_key=os.environ["@ENVKEY@"].strip())

    def flatten(self, nested):
        return [item for sub in nested for item in sub]

    def _encode_image(self, visual: Image.Image) -> str:
        """Pre-resize onto the provider grid, then base64-encode as lossless PNG."""
        img_w, img_h = visual.size
        new_h, new_w = @KEY@_resized_hw(img_h, img_w, self.model_code)
        if (new_h, new_w) != (img_h, img_w):
            visual = visual.resize((new_w, new_h), Image.LANCZOS)
        long_edge_cap, max_img_tokens = @KEY@_image_caps(self.model_code)
        assert new_h % _GRID == 0 and new_w % _GRID == 0
        assert max(new_h, new_w) <= long_edge_cap and (new_h * new_w) / _PX_PER_TOKEN <= max_img_tokens + 1
        buffer = io.BytesIO()
        visual.convert("RGB").save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    @backoff.on_exception(backoff.expo, Exception, max_tries=10, base=1.0,
                          jitter=backoff.random_jitter, giveup=_giveup_on_bad_request)
    def _generate_with_retry(self, image_b64: str, contexts: str, max_tokens: int) -> str:
        # {TODO}: adapt to the vendor request schema (reasoning/thinking flags, sampling
        # parameters some models reject, response field names).
        request_kwargs = dict(
            model=self.model_code,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": contexts},
                    ],
                }
            ],
        )
        if self.stop_strings:
            request_kwargs["stop"] = self.stop_strings
        response = self.client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            greedy = not (gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0)) if gen_kwargs else True
            key = self._resp_cache_key(doc_id, task, split, contexts)
            if greedy:
                cached = self.resp_cache_get(task, key)
                if cached is not None:
                    res.append(cached)
                    pbar.update(1)
                    continue

            visuals = self.flatten([doc_to_visual(self.task_dict[task][split][doc_id])])
            if len(visuals) != 1 or not isinstance(visuals[0], Image.Image):
                raise ValueError("We only support 1 image input and it should be of Image.Image type.")
            image_b64 = self._encode_image(visuals[0])

            # A per-task max_new_tokens (task YAML) takes precedence over the model default
            max_tokens = int(gen_kwargs.get("max_new_tokens", self.max_tokens)) if gen_kwargs else self.max_tokens
            resp = self._generate_with_retry(image_b64, contexts, max_tokens)
            if greedy:
                self.resp_cache_put(task, key, resp)
            res.append(resp)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for @CLASS@")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented for @CLASS@")
'''

SNIPPET_AVAILABLE = '''# Add this entry to AVAILABLE_MODELS in lmms_eval/models/__init__.py.
# The dict key must equal the @register_model("...") string in lmms_eval/models/@KEY@.py
# AND the value passed as `--model` by src/medvision_bm/benchmark/eval__@KEY@.py.
# get_model() resolves the value as lmms_eval.models.<key>.<ClassName>, so key and module
# file name must match too.

AVAILABLE_MODELS = {
    # ... existing entries ...
    # @DISPLAY@
    "@KEY@": "@CLASS@",
}
'''

SNIPPET_DISPATCH_API = '''# One edit in lmms_eval/tasks/medvision/medvision_utils.py.
#
# WHY: Tumor/Lesion-size and Angle/Distance prompts state the image size and pixel size the
# model must use for its pixel->mm arithmetic. Those numbers must describe the canvas the
# vision encoder ACTUALLY perceives. Without a branch here the run dies with:
#   ValueError: [Error] @KEY@ is not recognised/supported.
#
# For an API model the rule and the cap table live ONLY in lmms_eval/models/@KEY@.py; this
# branch imports them. Do NOT re-implement the formula here -- one source of truth means the
# prompt-side size and the image actually sent can never drift. The import is FUNCTION-LOCAL
# so the SFT path (which calls get_resized_img_shape but never with an API model name) does
# not load the model layer or the vendor SDK.
#
# get_resized_img_shape() returns (perceived_canvas_hw, content_hw); an API model that is
# pre-resized to a fixed point of the provider's pipeline has canvas == content, so only
# img_shape_resized_hw is set.


# ---- the branch inside get_resized_img_shape() ----------------------------------------
#      Insert BEFORE the final `else: raise ValueError(...)`.
    elif model_name == "@KEY@":
        from lmms_eval.models.@KEY@ import @KEY@_resized_hw

        img_h, img_w = img_2d_raw.shape[:2]
        # model_hf carries the RAW provider model code; normalization (stripping an
        # OpenRouter "<vendor>/" prefix) lives in exactly one place, the model file.
        model_code = (extra_kwargs or {}).get("model_hf") or ""
        img_shape_resized_hw = @KEY@_resized_hw(img_h, img_w, model_code)
        print(f"\\nOriginal image size (HxW): {(img_h, img_w)}; Resized image size (HxW): {tuple(img_shape_resized_hw)}")
'''

SNIPPET_DISPATCH = '''# Two edits in lmms_eval/tasks/medvision/medvision_utils.py.
#
# WHY: Tumor/Lesion-size and Angle/Distance prompts state the image size and pixel size the
# model must use for its pixel->mm arithmetic. Those numbers must describe the canvas the
# vision encoder ACTUALLY perceives after the model's internal resize. Without a branch here
# the run dies with: ValueError: [Error] @KEY@ is not recognised/supported.
#
# get_resized_img_shape() returns a PAIR:
#   (perceived_canvas_hw, content_hw)
# * perceived_canvas_hw -> the "The image size is W x H" sentence in the prompt;
# * content_hw          -> the resize ratio used to rescale the pixel size per axis.
# They are identical unless the model letterboxes/pads (then content_hw is the PRE-PAD
# content size, so the short axis does not get an inflated pixel size).


# ---- edit 1: the probe (place it next to the other _process_img_* helpers) -------------
def _process_img_@KEY@(img_2d_raw, extra_kwargs):
    # {TODO}: keep this only if @DISPLAY@ resizes dynamically. For a FIXED-size processor
    # delete this function and hard-code the size in the branch below (cheaper: no
    # processor download per sample).
    img_PIL = Image.fromarray(img_2d_raw).convert("RGB")
    model_hf = extra_kwargs["model_hf"]  # injected by evaluator.py from --model_args model_hf=...
    img_processor = AutoImageProcessor.from_pretrained(model_hf)
    processed_visual = img_processor([img_PIL])
    # {TODO}: read the resized H/W from whatever this processor returns, e.g.
    #   grid = processed_visual["image_grid_thw"][0]
    #   img_shape_resized_hw = (int(grid[1]) * img_processor.patch_size,
    #                           int(grid[2]) * img_processor.patch_size)
    # or pixel_values.shape[-2:] for processors that emit a spatial tensor.
    raise NotImplementedError("{TODO}: derive the resized (H, W) for @DISPLAY@")


# ---- edit 2: the branch inside get_resized_img_shape() --------------------------------
#      Insert BEFORE the final `else: raise ValueError(...)`.
#      List every alias: the AVAILABLE_MODELS key AND the SFT `model_family_name`
#      string, if this family is also fine-tuned.
    elif model_name in ["@KEY@"]:
        # {TODO}: cite the checkpoint's preprocessor_config.json / image processor here.
        img_shape_resized_hw = _process_img_@KEY@(img_2d_raw, extra_kwargs)
        # Fixed-size alternative:  img_shape_resized_hw = [896, 896]
        # Letterboxing alternative (canvas != content):
        #     img_shape_resized_hw = [336, 336]
        #     img_shape_content_hw = _padsquare_clip_content_hw(img_2d_raw, 336)
'''

EVAL_ENTRY = '''"""{TODO}: MedVision evaluation entry point for @DISPLAY@.

Copy to src/medvision_bm/benchmark/eval__@KEY@.py and run as
    python -m medvision_bm.benchmark.eval__@KEY@ ...
Distilled from the repository's eval__qwen2_5_vl.py (vLLM), eval__medgemma.py (HF) and
eval__gemini.py (API).
"""

import argparse
import json
import os
import subprocess

from medvision_bm.benchmark.eval_utils import parse_sample_indices
from medvision_bm.utils import (
    ensure_hf_hub_installed,
    install_medvision_ds,
    install_vendored_lmms_eval,
    load_tasks,
    load_tasks_status,
    setup_env_hf_medvision_ds,
    update_task_status,
)
@EXTRA_IMPORTS@

@API_KEY_TABLE@
def run_evaluation_for_task(
    lmmseval_module: str,
    model_args: str,
    task: str,
    batch_size: int,
    sample_limit: int,
    output_path: str,
    sample_indices: list = None,
    log_sys_prompt: bool = False,
):
    print(f"\\nRunning task: {task}\\n")
    cmd = [
        "python3",
        "-m",
@LAUNCH_PREFIX@        "lmms_eval",
        "--model",
        lmmseval_module,
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--batch_size",
        f"{batch_size}",
        "--limit",
        f"{sample_limit}",
        "--log_samples",
        "--output_path",
        output_path,
        "--verbosity=INFO",
    ]
    if sample_indices is not None:
        cmd += ["--sample_indices", json.dumps(sample_indices)]
    if log_sys_prompt:
        cmd += ["--log_sys_prompt"]
    result = subprocess.run(cmd, check=False)
    print(f"Command executed with return code: {result.returncode}")
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedVision benchmarking for @DISPLAY@.")
    # model-specific
    parser.add_argument("--model_hf_id", default="@HFID@", type=str,
                        help="HF model id / local path (also sent as model_args model_hf=...).")
    parser.add_argument("--model_name", required=True, type=str,
                        help="Display name; becomes the results sub-directory and the completed-tasks key.")
@KIND_ARGS@    parser.add_argument("--reshape_image_hw", default=None, type=str,
                        help="Reshape images to this height and width (format: H,W) before the model sees them.")
    # resource-specific
    parser.add_argument("--batch_size@BS_SUFFIX@", default=1, type=int, help="Batch size@BS_HELP@.")
    # task list, data, output, status
    parser.add_argument("--tasks_list_json_path", required=True, type=str, help="Path to the task-list JSON.")
    parser.add_argument("--results_dir", required=True, type=str, help="Path to the results directory.")
    parser.add_argument("--task_status_json_path", required=True, type=str, help="Path to the completed-tasks JSON.")
    parser.add_argument("--data_dir", required=True, type=str, help="Path to the MedVision data directory.")
    # evaluation-specific
    parser.add_argument("--sample_limit", default=1000, type=int, help="Max samples per task (API runs use 100).")
    parser.add_argument("--sample_indices", default=None, type=str, metavar="[start:stop]|[start,stop,step]",
                        help="Evaluate a slice of samples instead of the first --sample_limit.")
    parser.add_argument("--log-sys-prompt", action="store_true", default=False,
                        help="Log the system prompt (if any) in the per-sample JSONL output.")
    parser.add_argument("--stop_strings", nargs="*", default=None, metavar="STRING",
                        help="Stop sequences passed to the model for all tasks.")
    # debugging and control
    parser.add_argument("--skip_env_setup", action="store_true", help="Skip environment setup steps.")
    parser.add_argument("--skip_update_status", action="store_true", help="Do not record completed tasks.")
    parser.add_argument("--env_setup_only", action="store_true", help="Only perform environment setup and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
@API_KEY_CHECK@
    # NOTE: DO NOT change the order of these calls -- the vendored lmms_eval install and the
    # medvision_ds install both move pinned packages, and a later step must win.
    # ------
    setup_env_hf_medvision_ds(data_dir)
    if not args.skip_env_setup:
        # {TODO}: confirm the huggingface_hub version this model's stack needs.
        ensure_hf_hub_installed(hf_hub_version="0.36.0")
        install_vendored_lmms_eval(proj_dependency="@KEY@")
        install_medvision_ds(data_dir)
@EXTRA_INSTALL@        if args.env_setup_only:
            print("\\nEnvironment setup completed as per argument --env_setup_only. Exiting now.\\n")
            return
    else:
        print("\\n[Warning] Skipping environment setup as per argument --skip_env_setup.\\n")
    # ------

    tasks = load_tasks(args.tasks_list_json_path)
    for task in tasks:
        if task in load_tasks_status(args.task_status_json_path, args.model_name):
            print(f"Task {task} already completed. Skipping...")
            continue

        # model_hf is what the evaluator injects into lmms_eval_specific_kwargs, so the
        # task layer can compute the perceived image size for this exact checkpoint.
        model_args = @MODEL_ARGS@

        if args.reshape_image_hw is not None:
            s = args.reshape_image_hw.strip()
            s = ",".join(s.split()) if (" " in s) and ("," not in s) else s
            if not (s.startswith("[") or s.startswith("(")) and "," in s:
                s = f"[{s}]"
            model_args += f",reshape_image_hw={s}"
        if args.stop_strings:
            model_args += f",stop_strings={json.dumps(args.stop_strings, separators=(',', ':'))}"

        rc = run_evaluation_for_task(
            lmmseval_module="@KEY@",
            model_args=model_args,
            task=task,
            batch_size=@BS_EXPR@,
            sample_limit=args.sample_limit,
            output_path=os.path.join(args.results_dir, args.model_name),
            sample_indices=parse_sample_indices(args.sample_indices) if args.sample_indices else None,
            log_sys_prompt=args.log_sys_prompt,
        )
        if rc == 0:
            if not args.skip_update_status:
                update_task_status(args.task_status_json_path, args.model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
'''

LAUNCHER = '''# {TODO}: MedVision @TASKFAM@ launcher for @DISPLAY@.
# Copy to script/benchmark-@TASKFAM@/eval__@DISPLAY@__@TASKFAM@.sh in a checkout and edit
# benchmark_dir before running. Requires @HWNOTE@.

ENV_NAME="eval-@KEY@"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"

# Set paths and configs
benchmark_dir="{TODO}/path/to/MedVision"        # {TODO}: your checkout
data_dir="${benchmark_dir}/Data"
model_hf_id="@HFID@"
model_name="@DISPLAY@"                          # results sub-directory name
@RES_VARS@
# Other configs (safe to leave as is)
task_tag="MedVision-@TASKTAG@"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-@TASKTAG@.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=@SAMPLELIMIT@
@API_KEY_BLOCK@
set -euo pipefail

# Install medvision_bm. Build the wheel on node-local disk, not on a shared network tree:
# setuptools caches created build dirs process-globally, and a transiently vanishing
# subdir on a shared filesystem makes a later copy fail with "No such file or directory".
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \\
    pyproject.toml MANIFEST.in LICENSE src \\
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"

# Dataset annotation version to evaluate against
export MedVision_PLANNER_VERSION='1.0.0'
@ACK_RELEASE@
# Output token budget -- ALWAYS set one explicitly
@TOKEN_VAR@=@TOKEN_VAL@

# (Method 1) Manually install requirements before running the eval script (more robust).
# These three lines are LOAD-BEARING and must stay in this order.
# ---
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps @KEY@
pip install -r "${benchmark_dir}/requirements/requirements_eval_@KEY@.txt" --no-deps

python -m medvision_bm.benchmark.eval__@KEY@ \\
    --skip_env_setup \\
    --model_hf_id $model_hf_id \\
    --model_name $model_name \\
    --results_dir $result_dir \\
    --data_dir $data_dir \\
    --tasks_list_json_path $tasks_list_json_path \\
    --task_status_json_path $task_status_json_path \\
@RUN_FLAGS@    --sample_limit $sample_limit
# ---

# # (Method 2) Let the eval script install requirements (simpler, version-conflict prone).
# # KEEP THIS BLOCK IN SYNC WITH METHOD 1 -- both are edited together in the repository.
# # Debugging flags: --env_setup_only / --skip_env_setup / --skip_update_status
# python -m medvision_bm.benchmark.eval__@KEY@ \\
# --model_hf_id $model_hf_id \\
# --model_name $model_name \\
# --results_dir $result_dir \\
# --data_dir $data_dir \\
# --tasks_list_json_path $tasks_list_json_path \\
# --task_status_json_path $task_status_json_path \\
@RUN_FLAGS_COMMENTED@# --sample_limit $sample_limit \\

conda deactivate
# conda remove -n $ENV_NAME --all -y
'''

TEST_API = '''"""Offline tests for the @DISPLAY@ image-resize rule (model key "@KEY@").

The rule and the cap table live in ONE place -- lmms_eval/models/@KEY@.py -- and the task
layer (medvision_utils.get_resized_img_shape) imports @KEY@_resized_hw() from there, so
there is no second copy to keep in sync. These tests import the shipped functions directly.

NOTE: importing lmms_eval.models.@KEY@ transitively pulls `transformers` (via lmms_eval.api),
so run this inside the model's evaluation environment, not a bare interpreter. No network
and no API key are required.

Run: python test_@KEY@_resize.py     (or: pytest test_@KEY@_resize.py)
"""

import os
import sys

# {TODO}: point this at the vendored lmms_eval package directory of your checkout
# (<repo>/src/medvision_bm/medvision_lmms_eval), as it is on sys.path at eval time.
sys.path.insert(0, os.environ.get("MEDVISION_LMMS_EVAL_DIR", ""))

from lmms_eval.models.@KEY@ import (  # noqa: E402
    SUPPORTED_MODEL_CAPS,
    @KEY@_image_caps,
    @KEY@_resized_hw,
)

GRID = 28  # {TODO}: match _GRID in the model file
MODEL = "@HFID@"


def _on_grid(h, w):
    return h % GRID == 0 and w % GRID == 0


def test_caps_table_is_explicit():
    # An enumerated table, never a generic default.
    assert SUPPORTED_MODEL_CAPS
    for caps in SUPPORTED_MODEL_CAPS.values():
        assert isinstance(caps, tuple) and len(caps) == 2


def test_unsupported_model_raises():
    # An unverified model must fail loudly instead of silently using wrong caps.
    for bad in ["totally-unknown-model", "vendor/unknown-9.9", ""]:
        for fn in (@KEY@_image_caps, lambda m: @KEY@_resized_hw(512, 512, m)):
            try:
                fn(bad)
                raise AssertionError(f"{bad!r} should have raised")
            except ValueError:
                pass


def test_outputs_are_on_grid_and_within_caps():
    long_edge_cap, max_tokens = @KEY@_image_caps(MODEL)
    for h, w in [(512, 512), (4000, 3000), (3000, 800), (2000, 2000), (333, 777), (1, 1)]:
        nh, nw = @KEY@_resized_hw(h, w, MODEL)
        assert _on_grid(nh, nw), f"{h}x{w} -> {nh}x{nw} is not on the {GRID}px grid"
        assert max(nh, nw) <= long_edge_cap
        assert (nh * nw) / 750.0 <= max_tokens + 1


def test_never_upscales():
    for h, w in [(GRID, GRID), (100, 50), (1000, 1000)]:
        nh, nw = @KEY@_resized_hw(h, w, MODEL)
        assert nh <= h and nw <= w


def test_aspect_ratio_roughly_preserved():
    h, w = @KEY@_resized_hw(4000, 3000, MODEL)
    assert abs(w / h - 3000 / 4000) < 0.02


# {TODO}: add a live token-count probe (a separate, credential-gated script) confirming that
# an on-grid image incurs NO extra image tokens from server-side padding.

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
'''

TEST_LOCAL = '''"""Perceived-image-size tests for @DISPLAY@ (model key "@KEY@").

MedVision states the image size and pixel size in Tumor/Lesion-size and Angle/Distance
prompts. They must describe the canvas the vision encoder actually perceives, so the
dispatch branch in medvision_utils.get_resized_img_shape() must agree with what the real
image processor does.

NOTE: this downloads the image processor of "@HFID@" from the Hub the first time it runs
(set HF_HOME / HF_TOKEN accordingly) and needs the model's evaluation environment
(transformers pin). It is offline-safe once the processor is cached.

Run: python test_@KEY@_resize.py
"""

import os
import sys

import numpy as np

# {TODO}: point this at the vendored lmms_eval package directory of your checkout
# (<repo>/src/medvision_bm/medvision_lmms_eval), as it is on sys.path at eval time.
sys.path.insert(0, os.environ.get("MEDVISION_LMMS_EVAL_DIR", ""))

from lmms_eval.tasks.medvision.medvision_utils import get_resized_img_shape  # noqa: E402

MODEL_KEY = "@KEY@"
MODEL_HF = "@HFID@"
EXTRA = {"model_hf": MODEL_HF}


def _fake_slice(h, w):
    return (np.random.default_rng(0).random((h, w)) * 255).astype("uint8")


def test_dispatch_branch_exists():
    # Without a branch, get_resized_img_shape raises
    # "ValueError: [Error] @KEY@ is not recognised/supported."
    canvas, content = get_resized_img_shape(MODEL_KEY, _fake_slice(512, 512), EXTRA)
    assert len(canvas) == 2 and len(content) == 2


def test_shapes_are_positive_ints():
    for h, w in [(512, 512), (256, 512), (1935, 2400)]:
        canvas, content = get_resized_img_shape(MODEL_KEY, _fake_slice(h, w), EXTRA)
        for value in list(canvas) + list(content):
            assert int(value) == value and value > 0, f"{h}x{w} -> {canvas}/{content}"


def test_non_square_input_keeps_per_axis_ratio():
    # content_hw drives the per-axis pixel-size rescale. For a NON-padding model it equals
    # the canvas; for a letterboxing model it must be the PRE-PAD content size, so the two
    # axis ratios stay equal.
    h, w = 256, 512
    canvas, content = get_resized_img_shape(MODEL_KEY, _fake_slice(h, w), EXTRA)
    ratio_h, ratio_w = content[0] / h, content[1] / w
    # {TODO}: assert the behaviour @DISPLAY@ actually has -- equal ratios for a
    # pad-to-square model, or ratio_h != ratio_w for a stretch-to-fixed-canvas model.
    print(f"{h}x{w} -> canvas={tuple(canvas)} content={tuple(content)} "
          f"ratio_h={ratio_h:.4f} ratio_w={ratio_w:.4f}")


# {TODO}: assert the exact expected canvas for at least one input size, derived from the
# checkpoint's preprocessor_config.json (fixed size) or from a direct AutoImageProcessor
# call (dynamic size). A test that only checks "it does not raise" would not have caught
# the class of bug this file exists to prevent.

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
'''

REQUIREMENTS = '''# {TODO}: requirements_eval_@KEY@.txt -- a FROZEN, fully pinned environment for @DISPLAY@.
#
# How the repository produces these files: build the environment once by hand (conda env +
# medvision_bm + `install_vendored_lmms_eval --lmms_eval_opt_deps @KEY@` + the backend), then
# `pip freeze > requirements/requirements_eval_@KEY@.txt`. Launchers install it with
# `pip install -r ... --no-deps`, so EVERY transitive dependency must be listed and pinned.
#
# Pins that matter across the whole benchmark:
#   huggingface-hub==0.36.0   # the pin medvision_bm validates against
#   transformers==<x.y.z>     # NEVER leave unpinned: a newer transformers imports
#                             # `is_offline_mode` from huggingface_hub, which 0.36.0
#                             # removed -> ImportError at lmms_eval import time
#   datasets==3.6.0
#
# Also declare the runtime extras in medvision_lmms_eval/pyproject.toml under
# [project.optional-dependencies] as `@KEY@ = [...]`, so
# `install_vendored_lmms_eval --lmms_eval_opt_deps @KEY@` installs them.

huggingface-hub==0.36.0
datasets==3.6.0
transformers==0.0.0            # {TODO}: replace with the validated version
@BACKEND_REQ@# {TODO}: append the full `pip freeze` output of the working environment.
'''


# =========================================================================== generation
def derive_display(key: str) -> str:
    return "".join(p.capitalize() for p in key.replace("vllm_", "").split("_"))


def render(template: str, mapping: Dict[str, str]) -> str:
    out = template.replace("{TODO}", TODO)
    for placeholder, value in mapping.items():
        out = out.replace(placeholder, value)
    return out


def build_files(key: str, class_name: str, kind: str, hf_id: str, display: str) -> Dict[str, str]:
    base = {
        "@KEY@": key,
        "@CLASS@": class_name,
        "@HFID@": hf_id,
        "@DISPLAY@": display,
        "@ENVKEY@": key.upper() + "_API_KEY",
    }
    files: Dict[str, str] = {}

    model_template = {"vllm": MODEL_VLLM, "hf": MODEL_HF, "api": MODEL_API}[kind]
    files[os.path.join("lmms_eval", "models", f"{key}.py")] = render(model_template, base)
    files[os.path.join("patches", "AVAILABLE_MODELS.patch.txt")] = render(SNIPPET_AVAILABLE, base)
    files[os.path.join("patches", "medvision_utils_dispatch.patch.txt")] = render(
        SNIPPET_DISPATCH_API if kind == "api" else SNIPPET_DISPATCH, base
    )

    # ---- eval entry point -------------------------------------------------------------
    if kind == "api":
        eval_map = dict(base)
        eval_map["@EXTRA_IMPORTS@"] = ""
        eval_map["@API_KEY_TABLE@"] = (
            "# provider -> accepted API key env vars (first non-empty wins)\n"
            "API_KEY_ENV_VARS = {\n"
            f'    "native": ["{key.upper()}_API_KEY"],\n'
            '    "openrouter": ["OPENROUTER_API_KEY"],\n'
            "}\n\n\n"
        )
        eval_map["@LAUNCH_PREFIX@"] = ""
        eval_map["@KIND_ARGS@"] = (
            '    parser.add_argument("--api_provider", default="native", choices=["native", "openrouter"],\n'
            '                        type=str, help="API provider used to reach the model.")\n'
            '    parser.add_argument("--max_tokens", default=16000, type=int,\n'
            '                        help="Default max output tokens per request (reasoning shares this budget). '
            'A per-task max_new_tokens from the task YAML takes precedence.")\n'
        )
        eval_map["@API_KEY_CHECK@"] = (
            "\n    # Fail fast if the provider's API key is missing. Values are stripped downstream:\n"
            "    # pod-injected secrets carry a trailing newline, which is an illegal HTTP header value.\n"
            "    key_vars = API_KEY_ENV_VARS[args.api_provider]\n"
            "    if not any(os.environ.get(v, \"\").strip() for v in key_vars):\n"
            "        raise EnvironmentError(f\"None of {key_vars} is set.\")\n"
        )
        eval_map["@EXTRA_INSTALL@"] = ""
        eval_map["@MODEL_ARGS@"] = (
            "(\n"
            '            f"model={args.model_hf_id},"\n'
            '            f"provider={args.api_provider},"\n'
            '            f"model_hf={args.model_hf_id},"\n'
            '            f"max_tokens={args.max_tokens}"\n'
            "        )"
        )
        eval_map["@BS_SUFFIX@"] = ""
        eval_map["@BS_HELP@"] = " (API models run 1 request at a time)"
        eval_map["@BS_EXPR@"] = "args.batch_size"
    elif kind == "vllm":
        eval_map = dict(base)
        eval_map["@EXTRA_IMPORTS@"] = (
            "from medvision_bm.utils import (  # vLLM-only helpers\n"
            "    install_torch_cu124,\n"
            "    install_vllm,\n"
            "    set_cuda_num_processes,\n"
            "    setup_env_vllm,\n"
            ")\n"
        )
        eval_map["@API_KEY_TABLE@"] = "\n"
        eval_map["@LAUNCH_PREFIX@"] = ""
        eval_map["@KIND_ARGS@"] = (
            '    parser.add_argument("--gpu_memory_utilization", default=0.8, type=float,\n'
            '                        help="Fraction of each GPU reserved by vLLM.")\n'
            '    parser.add_argument("--max_new_tokens", default=4096, type=int,\n'
            '                        help="Output token budget. ALWAYS set this explicitly.")\n'
            '    parser.add_argument("--dtype", default="bfloat16", type=str, help="vLLM dtype.")\n'
        )
        eval_map["@API_KEY_CHECK@"] = (
            "\n    # Tensor-parallel size = number of visible GPUs\n"
            "    num_processes = set_cuda_num_processes()\n"
        )
        eval_map["@EXTRA_INSTALL@"] = (
            "        install_torch_cu124()\n"
            "        # {TODO}: pin the vLLM version that registers this architecture.\n"
            '        install_vllm(data_dir, version="0.10.0")\n'
            "        # {TODO}: re-install the transformers/accelerate pins LAST so they win resolution.\n"
        ).replace("{TODO}", TODO)
        eval_map["@MODEL_ARGS@"] = (
            "(\n"
            '            f"model_hf={args.model_hf_id},"\n'
            '            f"gpu_memory_utilization={args.gpu_memory_utilization},"\n'
            '            f"tensor_parallel_size={num_processes},"\n'
            '            f"max_num_seqs={args.batch_size_per_gpu},"\n'
            '            f"max_new_tokens={args.max_new_tokens},"\n'
            '            f"dtype={args.dtype}"\n'
            "        )"
        )
        eval_map["@BS_SUFFIX@"] = "_per_gpu"
        eval_map["@BS_HELP@"] = " per GPU (the main vLLM throughput lever)"
        eval_map["@BS_EXPR@"] = "args.batch_size_per_gpu"
    else:  # hf
        eval_map = dict(base)
        eval_map["@EXTRA_IMPORTS@"] = "from medvision_bm.utils import set_cuda_num_processes\n"
        eval_map["@API_KEY_TABLE@"] = "\n"
        eval_map["@LAUNCH_PREFIX@"] = (
            '        "accelerate.commands.launch",\n'
            '        f"--num_processes={num_processes}",\n'
            '        "--main_process_port=29501",\n'
            '        "-m",\n'
        )
        eval_map["@KIND_ARGS@"] = (
            '    parser.add_argument("--max_new_tokens", default=4096, type=int,\n'
            '                        help="Output token budget. ALWAYS set this explicitly.")\n'
        )
        eval_map["@API_KEY_CHECK@"] = (
            "\n    # One process per visible GPU (data parallel)\n"
            "    num_processes = set_cuda_num_processes()\n"
        )
        eval_map["@EXTRA_INSTALL@"] = ""
        eval_map["@MODEL_ARGS@"] = (
            "(\n"
            '            f"model_hf={args.model_hf_id},"\n'
            '            f"batch_size={args.batch_size_per_gpu},"\n'
            '            f"max_new_tokens={args.max_new_tokens}"\n'
            "        )"
        )
        eval_map["@BS_SUFFIX@"] = "_per_gpu"
        eval_map["@BS_HELP@"] = " per GPU (total batch = this x num_processes)"
        eval_map["@BS_EXPR@"] = "args.batch_size_per_gpu * num_processes"

    if kind == "hf":
        # num_processes must be visible inside run_evaluation_for_task
        eval_src = render(EVAL_ENTRY, eval_map).replace(
            "def run_evaluation_for_task(\n    lmmseval_module: str,",
            "def run_evaluation_for_task(\n    num_processes: int,\n    lmmseval_module: str,",
        ).replace(
            '            lmmseval_module="', '            num_processes=num_processes,\n            lmmseval_module="'
        )
    else:
        eval_src = render(EVAL_ENTRY, eval_map)
    files[os.path.join("src", "medvision_bm", "benchmark", f"eval__{key}.py")] = eval_src

    # ---- launchers --------------------------------------------------------------------
    for fam, tag, limit in (("detect", "detect-CoT", 1000), ("TL", "TL-CoT", 1000), ("AD", "AD-CoT", 1000)):
        lm = dict(base)
        lm["@TASKFAM@"] = fam
        lm["@TASKTAG@"] = tag
        lm["@SAMPLELIMIT@"] = str(100 if kind == "api" else limit)
        lm["@ACK_RELEASE@"] = (
            "export MedVision_ACK_RELEASE='1.4.0'   # required when pinning an older planner version\n"
            if fam == "TL" else ""
        )
        if kind == "api":
            lm["@HWNOTE@"] = "an API key (no GPU)"
            lm["@RES_VARS@"] = 'batch_size=1\nreshape_image_hw="512x512"     # {TODO}: keep or drop\n'.replace("{TODO}", TODO)
            lm["@API_KEY_BLOCK@"] = (
                "\n# API key check + sanitisation (pod-injected env vars can carry a trailing newline,\n"
                "# which breaks HTTP auth headers)\n"
                f'api_key_var="{key.upper()}_API_KEY"\n'
                'if [ -z "${!api_key_var:-}" ]; then\n'
                '    echo "[Error] ${api_key_var} is not set." >&2\n'
                "    exit 1\n"
                "fi\n"
                'export "${api_key_var}"="$(printf \'%s\' "${!api_key_var}" | tr -d \'\\n\')"\n'
            )
            lm["@TOKEN_VAR@"] = "max_tokens"
            lm["@TOKEN_VAL@"] = "16000"
            lm["@RUN_FLAGS@"] = "    --batch_size $batch_size \\\n    --max_tokens $max_tokens \\\n    --reshape_image_hw $reshape_image_hw \\\n"
            lm["@RUN_FLAGS_COMMENTED@"] = "# --batch_size $batch_size \\\n# --max_tokens $max_tokens \\\n# --reshape_image_hw $reshape_image_hw \\\n"
        elif kind == "vllm":
            lm["@HWNOTE@"] = "CUDA GPUs (tensor parallel = visible GPUs)"
            lm["@RES_VARS@"] = "batch_size_per_gpu=10\ngpu_memory_utilization=0.9\n"
            lm["@API_KEY_BLOCK@"] = ""
            lm["@TOKEN_VAR@"] = "max_new_tokens"
            lm["@TOKEN_VAL@"] = "4096"
            lm["@RUN_FLAGS@"] = (
                "    --batch_size_per_gpu $batch_size_per_gpu \\\n"
                "    --gpu_memory_utilization $gpu_memory_utilization \\\n"
                "    --max_new_tokens $max_new_tokens \\\n"
            )
            lm["@RUN_FLAGS_COMMENTED@"] = (
                "# --batch_size_per_gpu $batch_size_per_gpu \\\n"
                "# --gpu_memory_utilization $gpu_memory_utilization \\\n"
                "# --max_new_tokens $max_new_tokens \\\n"
            )
        else:
            lm["@HWNOTE@"] = "CUDA GPUs (one accelerate process per GPU)"
            lm["@RES_VARS@"] = "batch_size_per_gpu=4\n"
            lm["@API_KEY_BLOCK@"] = ""
            lm["@TOKEN_VAR@"] = "max_new_tokens"
            lm["@TOKEN_VAL@"] = "4096"
            lm["@RUN_FLAGS@"] = "    --batch_size_per_gpu $batch_size_per_gpu \\\n    --max_new_tokens $max_new_tokens \\\n"
            lm["@RUN_FLAGS_COMMENTED@"] = "# --batch_size_per_gpu $batch_size_per_gpu \\\n# --max_new_tokens $max_new_tokens \\\n"
        files[os.path.join("script", f"benchmark-{fam}", f"eval__{display}__{fam}.sh")] = render(LAUNCHER, lm)

    # ---- unit test + requirements -----------------------------------------------------
    files[os.path.join("unit-test", f"{key}-image-resize", f"test_{key}_resize.py")] = render(
        TEST_API if kind == "api" else TEST_LOCAL, base
    )
    req_map = dict(base)
    req_map["@BACKEND_REQ@"] = {
        "vllm": "vllm==0.0.0                    # " + TODO + ": the version that registers this architecture\n",
        "hf": "accelerate==1.9.0\n",
        "api": "# vendor SDK, e.g. openai==<pinned>\n",
    }[kind]
    files[os.path.join("requirements", f"requirements_eval_{key}.txt")] = render(REQUIREMENTS, req_map)
    return files


def looks_like_checkout(path: str) -> Optional[str]:
    cur = os.path.abspath(path)
    while True:
        if os.path.isdir(os.path.join(cur, "src", "medvision_bm")) or os.path.isdir(
            os.path.join(cur, "medvision_lmms_eval")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="lmms_eval model key, e.g. vllm_mymodel (lowercase, snake_case).")
    ap.add_argument("--class-name", required=True, help="Python class name, e.g. VLLM_MyModel.")
    ap.add_argument("--kind", required=True, choices=["vllm", "hf", "api"], help="Backend family.")
    ap.add_argument("--hf-id", required=True, help="HF model id / local path, or the API model code for --kind api.")
    ap.add_argument("--out-dir", required=True, help="Scratch directory to write the scaffold into.")
    ap.add_argument("--display-name", default=None, help="Launcher/results display name (default: derived from --key).")
    ap.add_argument("--dry-run", action="store_true", help="List the files that would be written and exit.")
    ap.add_argument("--allow-checkout", action="store_true",
                    help="Permit writing inside a MedVision checkout (refused by default).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files in --out-dir.")
    args = ap.parse_args(argv)

    if not KEY_RE.match(args.key):
        print(f"ERROR: --key {args.key!r} must be lowercase snake_case (it becomes a module name).", file=sys.stderr)
        return 2
    if not CLASS_RE.match(args.class_name):
        print(f"ERROR: --class-name {args.class_name!r} is not a valid Python identifier.", file=sys.stderr)
        return 2
    display = args.display_name or derive_display(args.key)
    if "/" in display or " " in display:
        print(f"ERROR: --display-name {display!r} must not contain spaces or '/' (it becomes a file name).",
              file=sys.stderr)
        return 2

    out_dir = os.path.abspath(args.out_dir)
    checkout = looks_like_checkout(out_dir)
    if checkout and not args.allow_checkout:
        print(f"ERROR: {out_dir} is inside what looks like a MedVision checkout ({checkout}).\n"
              f"       This generator writes SKELETONS for review, not in-place edits.\n"
              f"       Choose a scratch directory, or pass --allow-checkout if you really mean it.",
              file=sys.stderr)
        return 2

    files = build_files(args.key, args.class_name, args.kind, args.hf_id, display)

    if args.dry_run:
        print(f"[dry-run] would write {len(files)} files under {out_dir}:")
        for rel in sorted(files):
            print(f"  {rel}  ({len(files[rel].splitlines())} lines)")
        print(f"\nmodel key      : {args.key}\nclass          : {args.class_name}\n"
              f"backend        : {args.kind}\nHF id / code   : {args.hf_id}\ndisplay name   : {display}")
        return 0

    existing = [rel for rel in files if os.path.exists(os.path.join(out_dir, rel))]
    if existing and not args.force:
        print("ERROR: refusing to overwrite existing files (pass --force):", file=sys.stderr)
        for rel in existing:
            print(f"  {rel}", file=sys.stderr)
        return 2

    try:
        for rel, content in sorted(files.items()):
            dest = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write(content)
            print(f"wrote {dest}")
    except OSError as exc:
        print(f"ERROR: could not write the scaffold: {exc}", file=sys.stderr)
        return 1

    print(
        "\nNext steps (see references/add-a-model.md for the full checklist):\n"
        f"  1. Fill in every {TODO} marker, starting with lmms_eval/models/{args.key}.py.\n"
        f"  2. Copy the two patches/ snippets into lmms_eval/models/__init__.py and\n"
        "     lmms_eval/tasks/medvision/medvision_utils.py by hand.\n"
        f"  3. Declare the `{args.key}` extra in medvision_lmms_eval/pyproject.toml\n"
        "     ([project.optional-dependencies]) and freeze the requirements file.\n"
        f"  4. Verify wiring: list_registered_models.py --expect {args.key}\n"
        "  5. Run the resize test, then a 2-sample evaluation before any full run."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
