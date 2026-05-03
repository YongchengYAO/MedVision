import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from lmms_eval.api.registry import register_model
from tqdm import tqdm
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from medvision_bm.sft.sft_prompts_tooluse import (
    COT_INSTRUCT_ANGLE_TOOLUSE,
    COT_INSTRUCT_DISTANCE_TOOLUSE,
    COT_INSTRUCT_TL_TOOLUSE,
    TOOL_DEF,
)
from medvision_bm.utils.tool_execution import safe_exec_python

from .vllm_qwen25vl import VLLM_Qwen25VL


@register_model("vllm_qwen25vl_tooluse")
class VLLM_Qwen25VL_ToolUse(VLLM_Qwen25VL):
    """Two-phase tool-use inference for Qwen2.5-VL SFT models.

    Phase 1: generate until </tool_call> (stop string kept in output).
    Phase 2: append tool response turn and generate <answer>...</answer>.
    """

    def _pick_instruct(self, doc: dict) -> str:
        if doc.get("taskType") == "Tumor-Lesion-Size":
            return COT_INSTRUCT_TL_TOOLUSE
        if doc.get("biometric_profile", {}).get("metric_type") == "angle":
            return COT_INSTRUCT_ANGLE_TOOLUSE
        return COT_INSTRUCT_DISTANCE_TOOLUSE

    @staticmethod
    def _transform_prompt(contexts: str, instruct: str) -> str:
        SENTINEL = "Report the reasoning process"
        if SENTINEL not in contexts:
            raise ValueError(f"_transform_prompt: sentinel {SENTINEL!r} not found in prompt")
        base = contexts.rsplit(SENTINEL, 1)[0].rstrip()
        return base + " " + instruct

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(1))
            return (parsed.get("arguments") or {}).get("code")
        except json.JSONDecodeError:
            return None

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), desc="Model Responding (tool-use)")
        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

        for batch_requests in batched_requests:
            # Build Phase 1 messages
            phase1_messages = []
            for req in batch_requests:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments
                doc = self.task_dict[task][split][doc_id]

                instruct = self._pick_instruct(doc)
                user_text = self._transform_prompt(contexts, instruct)

                visuals = [doc_to_visual(doc)]
                if None in visuals:
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []
                    all_tasks = []
                    with ThreadPoolExecutor(max_workers=self.threads) as executor:
                        for visual in visuals:
                            if isinstance(visual, str) and any(
                                visual.endswith(ext)
                                for ext in (".mp4", ".avi", ".mov", ".flv", ".wmv")
                            ):
                                all_tasks.append(executor.submit(self.encode_video, visual))
                            else:
                                all_tasks.append(executor.submit(self.encode_image, visual))
                        for future in all_tasks:
                            imgs.append(future.result())

                # Images first, then text — matches training data format
                user_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in imgs
                ]
                user_content.append({"type": "text", "text": user_text})

                phase1_messages.append([
                    {"role": "system", "content": [{"type": "text", "text": json.dumps(TOOL_DEF)}]},
                    {"role": "user", "content": user_content},
                ])

            lora_request = LoRARequest("adapter", 1, self.lora_path) if self.lora_path else None

            # Phase 1: generate <think>…</think><tool_call>…</tool_call>
            params1 = SamplingParams(
                stop=["</tool_call>"],
                max_tokens=512,
                include_stop_str_in_output=True,
                temperature=0,
            )
            resp1 = self.client.chat(
                sampling_params=params1,
                messages=phase1_messages,
                lora_request=lora_request,
            )

            # Between phases: extract and execute code, extend messages with tool turn
            phase1_texts = []
            phase2_messages = []
            for i, out1 in enumerate(resp1):
                p1_text = out1.outputs[0].text
                phase1_texts.append(p1_text)
                code = self._extract_code(p1_text)
                tool_result = safe_exec_python(code) if code else "ERROR: no tool call found"

                phase2_messages.append(phase1_messages[i] + [
                    {"role": "assistant", "content": [{"type": "text", "text": p1_text}]},
                    {
                        "role": "tool",
                        "content": [{"type": "text", "text": f"<tool_response>{tool_result}</tool_response>"}],
                    },
                ])

            # Phase 2: generate <answer>…</answer>
            params2 = SamplingParams(
                stop=["<|im_end|>"],
                max_tokens=64,
                temperature=0,
            )
            resp2 = self.client.chat(
                sampling_params=params2,
                messages=phase2_messages,
                lora_request=lora_request,
            )

            response_text = [p1 + o.outputs[0].text for p1, o in zip(phase1_texts, resp2)]
            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res
