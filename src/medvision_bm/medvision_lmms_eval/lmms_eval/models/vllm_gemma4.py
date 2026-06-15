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

NUM_SECONDS_TO_SLEEP = 5

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


@register_model("vllm_gemma4")
class VLLM_Gemma4(lmms):
    """
    VLLM model wrapper for large multimodal models evaluation.

    This class provides a wrapper around the VLLM library to run inference on
    vision-language models. It supports both image and video inputs with automatic
    encoding and batched processing.

    Supported models: https://docs.vllm.ai/en/latest/models/supported_models.html

    Supported media formats:
        - Images: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        - Videos: .mp4, .avi, .mov, .flv, .wmv

    Image preprocessing (MedVision pixel-size rule):
        Images are sent raw (base64 PNG); vLLM applies the HF Gemma4ImageProcessor internally.
        Gemma 4 vision is variable-resolution: one scale factor (upscaling allowed) fits the
        image into a budget of max_patches = max_soft_tokens * pooling_kernel_size^2
        (= 280 * 9 = 2520) patches of 16x16 px, then each side is independently floored to a
        multiple of pooling_kernel_size * patch_size (= 3 * 16 = 48), so the aspect ratio is
        only approximately preserved (handled by per-axis pixel-size adjustment in the prompt).
        The processor outputs a flattened,
        sequence-padded patch list, NOT a spatial image grid:
            pixel_values:       [batch, max_patches, patch_size^2 * 3] = [batch, 2520, 768]
                                (the last two dims are config constants, not the resized H, W)
            image_position_ids: [batch, max_patches, 2] -> (x=col, y=row); padding = -1
        The TL/AD prompts must state the post-resize image and pixel size, which
        _process_img_gemma4() in lmms_eval/tasks/medvision/medvision_utils.py recovers by
        probing the same processor and reading the valid (non -1) patch-grid extent.

    Chat template:
        The chat template is used to format the conversation for the model. It can be
        provided as a file path or as a template string directly.
        - Chat template intro: https://huggingface.co/docs/transformers/en/chat_templating
        - VLLM chat method: https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat

    Args:
        model_hf (str): HuggingFace model identifier or path to the model.
            Default: "google/gemma-4-31B-it"
        tensor_parallel_size (int): Number of GPUs to use for tensor parallelism.
            Default: 1
        gpu_memory_utilization (float): Fraction of GPU memory to use for model weights.
            Should be between 0.0 and 1.0. Default: 0.8
        batch_size (int): Number of requests to process in parallel per GPU.
            Default: 1
        max_frame_num (int): Maximum number of frames to extract from videos.
            Frames are sampled uniformly across the video duration. Default: 32
        min_new_tokens (int): Minimum number of tokens to generate before EOS or
            stop strings are honored (vLLM `min_tokens`). Default 0 (off). Optional
            lever for the rare immediate-EOS/empty-output case; leave at 0 when
            using a "</answer>" stop string (see note at the params dict).
        threads (int): Number of threads to use for parallel visual encoding.
            Default: 16
        trust_remote_code (bool, optional): Whether to trust remote code when loading
            the model. Default: True
        chat_template (str, optional): Path to chat template file or template string.
            If None, uses the model's default template. Default: None
        enable_thinking (bool): Enable Gemma 4 reasoning ("thinking") mode, passed through
            to the model's chat template via chat_template_kwargs. Default: True
        **kwargs: Additional arguments passed to the VLLM LLM constructor.
            - NOTE: model specific arguments can be passed here without the need to add more arguments to this class (see example below)
            - String arguments that look like JSON dictionaries will be automatically parsed.


    Python Example 1: (example of passing model specific arguments)
    # ---------------------
    import subprocess
    cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            "vllm",
            "--model_args",
            "model_hf=meta-llama/Llama-4-Scout-17B-16E-Instruct,"
            "tensor_parallel_size=4,"
            "dtype=bfloat16,"
            "max_model_len=10240,"
            "gpu_memory_utilization=0.9,"
            'override_generation_config={"attn_temperature_tuning": true},' # example of passing model specific arguments, JSON string will be parsed automatically
            "enforce_eager=True,"
            "kv_cache_dtype=fp8",
            "--tasks",
            task, # change this to your task
            "--batch_size",
            "1",
            "--limit",
            "10",
            "--log_samples",
            "--output_path",
            "logs",
        ]
    cmd_result = subprocess.run(cmd, check=False)
    # ---------------------


    # NOTE: No need to pass the chat template file if it is already defined in the model tokenizer.
    # The chat method automatically applies the model's chat template to format the prompt
    # - vllm chat method: https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat

    """

    def __init__(
        self,
        model_hf: str = "google/gemma-4-31B-it",
        lora_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        max_frame_num: int = 32,
        max_new_tokens: int = 4096,
        min_new_tokens: int = 0,
        threads: int = 16,  # Threads to use for decoding visuals
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        enable_thinking: bool = True,
        stop_strings: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.max_frame_num = max_frame_num
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.threads = threads
        self.chat_template = chat_template
        self.enable_thinking = enable_thinking
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

        self.batch_size_per_gpu = int(batch_size)

    def _shutdown_engine(self):
        # Deterministically tear down the vLLM engine while the interpreter is still healthy.
        #
        # In vLLM 0.19 the V1 client runs a daemon thread (MPClientEngineMonitor) that watches
        # the EngineCore subprocess. At interpreter exit the subprocess can be reaped before the
        # client's own shutdown runs, so the monitor sees a live client + dead core and logs
        # "EngineCore ... died unexpectedly" (core_client.py: monitor returns early only if
        # `_finalizer.alive` is False or `resources.engine_dead` is True). The cure is to trigger
        # the client's finalizer FIRST, here, not at GC teardown: dropping the last reference and
        # forcing collection runs MPClient.shutdown() -> _finalizer.detach(), which flips
        # `_finalizer.alive` to False so the monitor stays silent. (Older vLLM 0.10 used elsewhere
        # in this repo has no such monitor, which is why those wrappers need no teardown code.)
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

                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0

                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95

                until = gen_kwargs.get("until") or []
                if isinstance(until, str):
                    until = [until]
                until_list = [s for s in until if s is not None]
                # Only when the user supplies explicit stop strings via --stop_strings (e.g.
                # "</answer>") do we drop the newline/whitespace-only entries that lmms-eval
                # auto-injects into `until` (the fewshot delimiter "\n\n"; see api/task.py).
                # Gemma 4's CoT puts blank lines between <step-k> blocks, so a "\n\n" stop halts
                # generation mid-reasoning, before <answer> is produced. Gating on self.stop_strings
                # keeps the default path unchanged -- models with no explicit terminator still
                # benefit from the "\n\n" runaway-stop -- while an explicit stop string then defines
                # exactly where generation should end. (Mirrors vllm_qwen3vl.py.)
                if self.stop_strings:
                    until_list = [s for s in until_list if s.strip() != ""]
                stop = list(dict.fromkeys(until_list + self.stop_strings))

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    # Optional minimum generation length (vLLM `min_tokens`); default 0 = off,
                    # tunable via --min_new_tokens to escape a rare immediate-EOS (empty output).
                    # NOTE: min_tokens suppresses BOTH EOS *and* stop strings until the floor, so
                    # keep it 0 (or well below a real answer's length) when relying on a
                    # "</answer>" stop, otherwise it forces tokens past the answer terminator.
                    "min_tokens": self.min_new_tokens,
                    "top_p": gen_kwargs["top_p"],
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
                # When there is no image token in the context, append the image to the text
                messages[-1]["content"].append({"type": "text", "text": contexts})
                for img in imgs:
                    messages[-1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

                batched_messages.append(messages)

            sampling_params = SamplingParams(**params)

            # NOTE:
            # The chat method automatically applies the model's chat template to format the prompt
            # - vllm chat method: https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat
            # The logic here is similar to the vllm implementation as shown here (https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat)
            # - vllm implementation: https://github.com/vllm-project/vllm/blob/d97841078b6e0dde8da36d5a2b8e8857a2c37944/vllm/entrypoints/chat_utils.py#L829
            lora_request = None
            if self.lora_path is not None:
                lora_request = LoRARequest("adapter", 1, self.lora_path)

            chat_template_kwargs = {"enable_thinking": self.enable_thinking}
            if self.chat_template is not None:
                if os.path.isfile(self.chat_template):
                    with open(self.chat_template, "r") as f:
                        chat_template = f.read()
                else:
                    chat_template = self.chat_template
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=chat_template, chat_template_kwargs=chat_template_kwargs, lora_request=lora_request)
            else:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template_kwargs=chat_template_kwargs, lora_request=lora_request)

            # NOTE: For thinking models, CompletionOutput may split output into:
            #   - reasoning_content: content inside the model's thinking block (may be empty if thinking disabled)
            #   - text: content AFTER the thinking block (may be empty if all content is in reasoning_content)
            # We combine both to get the full response for robust downstream parsing.
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
        # healthy) to avoid vLLM 0.19's spurious "EngineCore died unexpectedly" teardown race.
        self._shutdown_engine()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented yet.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("generate_until_multi_round is not implemented yet.")
