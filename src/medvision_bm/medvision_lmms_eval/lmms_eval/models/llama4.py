import warnings
import torch
import PIL
import numpy
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from loguru import logger as eval_logger
from typing import List, Tuple, Optional, Union
from accelerate import Accelerator, DistributedType
from transformers import AutoProcessor, Llama4ForConditionalGeneration, BitsAndBytesConfig
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


# suppress dynamo errors to fall back to eager if FakeTensor device‐prop mismatch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore")


@register_model("llama4")
class Llama4(lmms):
    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "bf16",
        attn_implementation: Optional[str] = "flex_attention",
        device_map: Optional[Union[str, dict]] = "auto",
        batch_size: int = 1,
        quant_int4: bool = True,
        **kwargs,
    ) -> None:
        """
        A plain LLaMA-4 model for text-only inference, compatible with lmms-eval.
        """
        super().__init__()

        # Device setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load processor
        self.processor = AutoProcessor.from_pretrained(pretrained)

        # Load model
        if quant_int4:
            # 4‑bit quantization settings
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            # Load the quantized model
            self._model = Llama4ForConditionalGeneration.from_pretrained(
                pretrained,
                attn_implementation=attn_implementation,
                device_map=device_map,
                quantization_config=bnb_config,
            )
        else:
            if dtype == "bf16":
                __dtype = torch.bfloat16
            elif dtype == "fp16":
                __dtype = torch.float16
            elif dtype == "fp32":
                __dtype = torch.float32
            elif dtype == "int8":
                __dtype = torch.int8
            elif dtype == "int4":
                __dtype = torch.int4
            self._model = Llama4ForConditionalGeneration.from_pretrained(
                pretrained,
                attn_implementation=attn_implementation,
                device_map=device_map,
                torch_dtype=__dtype,
            )

        # Only move model to device if not using a device_map
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self._batch_size = batch_size

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

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

            # Prepare messages
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            messages = [{"role": "user", "content": []}]
            images = []
            for visual in visuals:
                if isinstance(visual, PIL.Image.Image):
                    images.append(visual)
                elif isinstance(visual, torch.Tensor) or isinstance(visual, numpy.ndarray):
                    images.append(to_pil_image(visual))
                else:
                    eval_logger.error(f"Unsupported visual type: {type(visual)}. Converting to PIL image if possible.")
            for _ in range(len(images)):
                messages[-1]["content"].append({"type": "image"})
            messages[-1]["content"].append({"type": "text", "text": contexts})

            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Get the model's response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                )
                response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])[0]
                pbar.update(1)
        pbar.close()
        return response

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Error: generate_until_multi_round not implemented for Llama4")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Error: loglikelihood not implemented for Llama4")
