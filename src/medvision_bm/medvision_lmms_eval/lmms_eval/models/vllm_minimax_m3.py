import base64
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
from decord import VideoReader, cpu
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from medvision_bm.utils.configs import SEED

NUM_SECONDS_TO_SLEEP = 5

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


@register_model("vllm_minimax_m3")
class VLLM_MiniMaxM3(lmms):
    """
    vLLM wrapper for MiniMax-M3 (vision-language) evaluation.

    MiniMax-M3 (https://huggingface.co/MiniMaxAI/MiniMax-M3) is a native multimodal model:
    a 428B-parameter sparse MoE (~23B activated) text backbone with a CLIP-style vision tower,
    using MiniMax Sparse Attention (MSA) for long context. Its HF architecture is
    `MiniMaxM3SparseForConditionalGeneration` (model_type `minimax_m3_vl`).

    IMPORTANT -- vLLM support:
        The checkpoint ships only `configuration_*.py` (no HF modeling file), so the forward pass
        must come from a NATIVE vLLM/SGLang implementation of `minimax_m3_vl`, not from HF remote
        code. This wrapper therefore requires a vLLM build that registers the MiniMax-M3 VL
        architecture. If `LLM(...)` raises an "unknown/unsupported architecture" error, upgrade
        vLLM to a release that lands MiniMax-M3-VL support (see eval__minimax_m3.py install step).

    Sampling defaults mirror the checkpoint's generation_config.json (do_sample=true,
    temperature=1.0, top_p=0.95) plus the model card's top_k=40. MiniMax-M3 is a reasoning model
    (emits <think>/<mm:think> spans); greedy decoding (temperature=0) is NOT recommended -- a task's
    generation_kwargs still override these per-task. Generation uses a fixed seed
    (medvision_bm.utils.configs.SEED) so sampling-based runs stay reproducible.

    Args mirror the Qwen3-VL vLLM wrapper; see that class for the full argument reference. Only the
    model-specific defaults (model_hf, temperature/top_k) differ here.
    """

    def __init__(
        self,
        model_hf: str = "MiniMaxAI/MiniMax-M3",
        lora_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 1,
        max_frame_num: int = 32,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        threads: int = 16,  # Threads to use for decoding visuals
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        stop_strings: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.max_frame_num = max_frame_num
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.threads = threads
        self.chat_template = chat_template
        self.lora_path = lora_path
        self.stop_strings: List[str] = json.loads(stop_strings) if stop_strings else []
        self.system_prompt: Optional[str] = json.loads(system_prompt)[0] if system_prompt else None

        # Convert any string arguments that start with { and end with } to dictionaries
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        # Remove MedVision-specific kwargs that should not be forwarded to vLLM
        kwargs.pop("reshape_image_hw", None)

        # Set up vllm client
        lora_kwargs = {}
        if self.lora_path is not None:
            adapter_config_path = os.path.join(self.lora_path, "adapter_config.json")
            if os.path.isfile(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                lora_rank = adapter_config.get("r", None)
                if lora_rank is None:
                    raise ValueError(f"LoRA rank 'r' not found in adapter_config.json at {adapter_config_path}. Please ensure the file contains the necessary configuration for LoRA.")
            else:
                raise FileNotFoundError(f"adapter_config.json not found at {self.lora_path}. Please ensure the file exists and contains the necessary configuration for LoRA.")
            lora_kwargs["enable_lora"] = True
            lora_kwargs["max_lora_rank"] = lora_rank

        self.client = LLM(
            model=self.model_hf,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **lora_kwargs,
            **kwargs,
        )

        # Set padding side
        tokenizer = self.client.get_tokenizer()
        tokenizer.padding_side = "left"
        self.client.set_tokenizer(tokenizer)

        self.batch_size_per_gpu = int(batch_size)

    def _shutdown_engine(self):
        # Deterministically tear down the vLLM engine while the interpreter is still healthy.
        # See the Qwen3-VL wrapper for the full rationale (vLLM V1 EngineCore monitor race).
        client = getattr(self, "client", None)
        if client is None:
            return
        self.client = None
        try:
            del client
            gc.collect()
        except Exception:
            pass

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)

        # Always show progress - vLLM runs as single process with internal GPU distribution
        pbar = tqdm(total=len(requests), desc="Model Responding")

        # Resolve already-cached responses up front so an interrupted run resumes;
        # only the uncached requests are sent to the model. Non-greedy sampling is
        # never cached (identical args would collide on the same key).
        pending = []  # list of (global_index, request)
        for gi, req in enumerate(requests):
            _, gen_kwargs, _, doc_id, task, _ = req.arguments
            if gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0):
                pending.append((gi, req))
                continue
            cached = self.resp_cache_get(task, self._resp_cache_key(doc_id, task, req.arguments[5], req.arguments[0]))
            if cached is not None:
                res[gi] = cached
                pbar.update(1)
            else:
                pending.append((gi, req))

        batch_size = self.batch_size_per_gpu
        batched_requests = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx][1].arguments

                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = self.max_new_tokens
                # MiniMax-M3 is a reasoning model: greedy decoding (temperature=0) is not recommended.
                # Fall back to the configured sampling params (mirror generation_config.json + card);
                # a task's generation_kwargs still override these per-task.
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = self.temperature
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = self.top_p
                if "top_k" not in gen_kwargs:
                    gen_kwargs["top_k"] = self.top_k

                # String-level stop sequences are applied ONLY when explicitly provided via
                # --stop_strings. The task config's `until` (lmms-eval defaults it to the fewshot
                # delimiter "\n\n") is NOT forwarded as a decoding stop -- the fewshot delimiter must
                # not double as a generation terminator, or it truncates multi-paragraph CoT after
                # the first blank line. Default stopping relies on the model's EOS token.
                stop = list(dict.fromkeys(self.stop_strings))

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                    "top_k": gen_kwargs["top_k"],
                    # Fixed seed for reproducible sampling-based eval.
                    "seed": SEED,
                }
                if stop:
                    params["stop"] = stop
                if self.stop_strings:
                    params["include_stop_str_in_output"] = True

                # params is collected per-request; after the loop, SamplingParams
                # is built once from the last request's params for the batch call.

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []  # multiple images or frames for video
                    all_tasks = []
                    with ThreadPoolExecutor(max_workers=self.threads) as executor:
                        for visual in visuals:
                            if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                                all_tasks.append(executor.submit(self.encode_video, visual))
                            elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                                all_tasks.append(executor.submit(self.encode_image, visual))
                            elif isinstance(visual, Image.Image):
                                all_tasks.append(executor.submit(self.encode_image, visual))

                        for future in all_tasks:
                            imgs.append(future.result())

                if self.system_prompt:
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                        {"role": "user", "content": []},
                    ]
                else:
                    messages = [{"role": "user", "content": []}]
                # Images first, then text (vLLM expands the model's image placeholder via the chat template)
                for img in imgs:
                    messages[-1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                messages[-1]["content"].append({"type": "text", "text": contexts})

                batched_messages.append(messages)

            sampling_params = SamplingParams(**params)

            lora_request = None
            if self.lora_path is not None:
                lora_request = LoRARequest("adapter", 1, self.lora_path)

            if self.chat_template is not None:
                if os.path.isfile(self.chat_template):
                    with open(self.chat_template, "r") as f:
                        chat_template = f.read()
                else:
                    chat_template = self.chat_template
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=chat_template, lora_request=lora_request)
            else:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, lora_request=lora_request)

            # For reasoning models, vLLM may split output into reasoning_content (inside <think>...)
            # and text (after </think>). Combine both for robust downstream <answer> parsing.
            def _get_full_text(output) -> str:
                text = output.text
                reasoning = getattr(output, "reasoning_content", None) or ""
                if reasoning:
                    return reasoning + ("\n" + text if text else "")
                return text

            response_text = [_get_full_text(o.outputs[0]) for o in response]

            assert len(response_text) == len(batch_requests)
            for (gi, req), text in zip(batch_requests, response_text):
                _, gen_kwargs, _, doc_id, task, split = req.arguments
                if not (gen_kwargs.get("do_sample", False) or gen_kwargs.get("temperature", 0)):
                    self.resp_cache_put(task, self._resp_cache_key(doc_id, task, split, req.arguments[0]), text)
                res[gi] = text
            pbar.update(len(batch_requests))

        pbar.close()

        # Each MedVision task runs as its own `lmms_eval` subprocess that exits right after this
        # call, so this is the engine's last use. Tear it down now (while the interpreter is
        # healthy) to avoid vLLM's spurious "EngineCore died unexpectedly" teardown race.
        self._shutdown_engine()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented yet.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented yet.")
