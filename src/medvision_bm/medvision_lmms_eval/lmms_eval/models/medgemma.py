import concurrent.futures
import json
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin, PartialState
from accelerate.utils import DistributedType, wait_for_everyone
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import eval_logger
from PIL import Image
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline


@register_model("medgemma")
class MedGemma(lmms):
    """
    MedGemma Model

    - HF: https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4

    dtype: BF16 (https://huggingface.co/google/medgemma-27b-it/blob/main/config.json)
    """

    def __init__(
        self,
        model_hf: str = "google/medgemma-4b-it",
        distributed_type: str = "multi-gpu",  # "fsdp" or "multi-gpu"
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attention_2: Optional[bool] = False,
        use_pipeline: bool = True,
        max_new_tokens: int = 4096,
        num_workers: int = 8,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        stop_strings: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_hf = model_hf
        self.distributed_type = distributed_type
        self.batch_size_per_gpu = int(batch_size)
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_pipeline = use_pipeline
        self.max_new_tokens = max_new_tokens
        self.num_workers = num_workers
        self.stop_strings: List[str] = json.loads(stop_strings) if stop_strings else []
        # Default preserves the original "You are an expert radiologist." behaviour.
        self.system_prompt: Optional[str] = json.loads(system_prompt)[0] if system_prompt else "You are an expert radiologist."

        # NOTE: google/medgemma-4b-it only supports bfloat16, setting data type to others will cause empty output
        self.model_dtype = torch.bfloat16

        if self.use_pipeline:
            self.prepare_pipeline(device, device_map)
        else:
            self.prepare_model(device, device_map)

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _get_model_kwargs(self):
        # Set the device string for multi-gpu training using accelerate's PartialState
        model_kwargs = dict(
            torch_dtype=self.model_dtype,
            device_map=self.device_map,
            attn_implementation="flash_attention_2" if self.use_flash_attention_2 else "eager",
        )
        return model_kwargs

    def _setup_distributed_inference(self, device: Optional[str] = "cuda", device_map: Optional[str] = "auto"):
        # Set up accelerator
        if self.distributed_type == "fsdp":
            fsdp_plugin = FullyShardedDataParallelPlugin(
                state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
            )
            self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        else:
            if self.distributed_type != "multi-gpu":
                eval_logger.warning(f"Unknown distributed_type '{self.distributed_type}', falling back to multi-gpu.")
            self.accelerator = Accelerator()

        if self.accelerator.num_processes > 1:
            # Pin the entire model to this process's GPU
            self.device_map = {"":  self.accelerator.local_process_index}
        else:
            # Single-process: respect user's device_map (e.g., "auto", "cpu", etc.)
            self.device_map = device_map

        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes
        self._device = self.accelerator.device

    def prepare_model(self, device: Optional[str] = "cuda", device_map: Optional[str] = "auto"):
        # Set up distributed training if necessary
        self._setup_distributed_inference(device=device, device_map=device_map)

        # Load model
        model_kwargs = self._get_model_kwargs()
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_hf,
            **model_kwargs,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_hf)

        if self.accelerator.num_processes > 1:
            assert self.accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            eval_logger.info(f"Distributed type: {self.accelerator.distributed_type}")

            # Prepare model and processor for distributed training
            # NOTE: modified from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/f64dfa5fd063e989a0a665d2fd0615df23888c83/lmms_eval/models/internvl2.py#L223
            if self.accelerator.distributed_type == DistributedType.FSDP:
                self._model, self._processor = self.accelerator.prepare(self._model, self._processor)
            elif self.accelerator.distributed_type == DistributedType.MULTI_GPU:
                self._processor = self.accelerator.prepare(self._processor)
                self._model = self.accelerator.prepare_model(self._model, evaluation_mode=True)

            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {self.accelerator.num_processes} devices with data parallelism")
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)

    def prepare_pipeline(self, device: Optional[str] = "cuda", device_map: Optional[str] = "auto"):
        """
        By setting device_map to "auto", the pipeline will let Accelerate choose the device.
        """
        # Set up distributed training if necessary
        self._setup_distributed_inference(device=device, device_map=device_map)

        # Load pipeline
        model_kwargs = self._get_model_kwargs()
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_hf,
            model_kwargs=model_kwargs,
        )
        self.pipe.model.generation_config.do_sample = False

    def _apply_stop_strings(self, text: str) -> str:
        if not self.stop_strings:
            return text
        earliest_pos, earliest_stop = len(text), None
        for stop in self.stop_strings:
            pos = text.find(stop)
            if pos != -1 and pos < earliest_pos:
                earliest_pos, earliest_stop = pos, stop
        if earliest_stop is not None:
            return text[: earliest_pos + len(earliest_stop)]
        return text

    # Code adapted from https://huggingface.co/google/medgemma-4b-it
    def infer(self, questions: Union[str, List[str]], pil_imgs: Union[Image.Image, List[Image.Image]], max_new_tokens: int = None) -> Union[str, List[str]]:
        _max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        # Handle single input case
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(pil_imgs, Image.Image):
            pil_imgs = [pil_imgs]

        batch_messages = []
        for question, pil_img in zip(questions, pil_imgs):
            user_msg = {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": pil_img}]}
            if self.system_prompt:
                messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}, user_msg]
            else:
                messages = [user_msg]
            batch_messages.append(messages)

        if self.use_pipeline:
            outputs = self.pipe(text=batch_messages, max_new_tokens=_max_new_tokens, batch_size=self.batch_size)
            responses = [output[0]["generated_text"][-1]["content"] for output in outputs]
        else:
            responses = []
            for messages in batch_messages:
                inputs = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self._model.device, dtype=self.model_dtype)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    generation = self._model.generate(**inputs, max_new_tokens=_max_new_tokens, do_sample=False)
                    generation = generation[0][input_len:]
                resp = self._processor.decode(generation, skip_special_tokens=True)
                responses.append(resp)

        return responses

    def process_single_request(self, request: Instance) -> str:
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

        # Image inputs - optimized visual processing
        visual_doc = self.task_dict[task][split][doc_id]
        visuals = doc_to_visual(visual_doc)

        # Handle visual processing more efficiently
        if isinstance(visuals, list):
            if len(visuals) == 1 and isinstance(visuals[0], Image.Image):
                visual = visuals[0]
            else:
                # Only flatten if necessary
                flattened = self.flatten([visuals]) if not isinstance(visuals[0], Image.Image) else visuals
                if len(flattened) == 1 and isinstance(flattened[0], Image.Image):
                    visual = flattened[0]
                else:
                    raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")
        elif isinstance(visuals, Image.Image):
            visual = visuals
        else:
            raise ValueError("The model only supports 1 image input and it should be of Image.Image type.")

        return contexts, visual

    def process_batch_parallel(self, batch_requests):
        batch_contexts = [None] * len(batch_requests)
        batch_visuals = [None] * len(batch_requests)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_requests), self.num_workers)) as executor:
            process_func = partial(self.process_single_request)
            results = list(executor.map(process_func, batch_requests))

        for idx, (contexts, visual) in enumerate(results):
            batch_contexts[idx] = contexts
            batch_visuals[idx] = visual

        return batch_contexts, batch_visuals

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # Process requests in batches
        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i : i + self.batch_size]
            batch_size = len(batch_requests)

            # Use parallel processing for batch preparation
            batch_contexts, batch_visuals = self.process_batch_parallel(batch_requests)

            # Get batch model outputs
            _gen_kwargs = batch_requests[0].args[1]
            batch_responses = self.infer(questions=batch_contexts, pil_imgs=batch_visuals, max_new_tokens=_gen_kwargs.get("max_new_tokens", self.max_new_tokens))

            # Ensure batch_responses is a list
            if isinstance(batch_responses, str):
                batch_responses = [batch_responses]

            batch_responses = [self._apply_stop_strings(r) for r in batch_responses]
            res.extend(batch_responses)
            pbar.update(batch_size)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MedGemma")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for MedGemma")
