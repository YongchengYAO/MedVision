"""
Two-phase vLLM inference for tool-use SFT eval.

Phase 1: generate up to </tool_call> stop token
Between phases: extract code from <tool_call> JSON, run safe_exec_python, build continuation
Phase 2: generate <answer>...</answer> from continuation
"""

import argparse
import json
import os
import re
import subprocess
import sys

from medvision_bm.sft.sft_prompts_tooluse import TOOL_DEF
from medvision_bm.utils import (
    install_vllm,
    load_tasks,
    load_tasks_status,
    setup_env_hf_medvision_ds,
    update_task_status,
)
from medvision_bm.utils.tool_execution import safe_exec_python


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool-use SFT two-phase vLLM inference."
    )
    parser.add_argument("--model_hf_id", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tasks_list_json_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--task_status_json_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample_limit", default=100, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float)
    parser.add_argument("--max_tokens_phase1", default=512, type=int)
    parser.add_argument("--max_tokens_phase2", default=64, type=int)
    parser.add_argument("--skip_env_setup", action="store_true")
    parser.add_argument("--skip_update_status", action="store_true")
    return parser.parse_args()


def install_transformers_for_qwen25vl(
    transformers_version="4.54.1", accelerate_version="1.9.0"
):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"transformers=={transformers_version}",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"accelerate=={accelerate_version}"],
        check=True,
    )


def extract_code(text):
    """Parse <tool_call>JSON</tool_call> and return the code field, or None."""
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj.get("arguments", {}).get("code")
    except json.JSONDecodeError:
        return None


def build_continuation(tool_result):
    return (
        f"</tool_call><|im_end|>\n"
        f"<|im_start|>tool\n"
        f"<tool_response>\n{tool_result}\n</tool_response>\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    args = parse_args()
    setup_env_hf_medvision_ds(args.data_dir)

    if not args.skip_env_setup:
        install_vllm(args.data_dir, version="0.10.0")
        install_transformers_for_qwen25vl()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_hf_id,
        enable_lora=(args.lora_path is not None),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    processor = AutoProcessor.from_pretrained(args.model_hf_id, trust_remote_code=True)

    tasks = load_tasks(args.tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(args.task_status_json_path, args.model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        from medvision_bm.medvision_lmms_eval.lmms_eval.tasks import get_task_dict

        task_dict = get_task_dict([task])
        task_obj = task_dict[task]
        docs = list(task_obj.test_docs())[: args.sample_limit]

        prompts = []
        for doc in docs:
            from medvision_bm.sft.sft_prompts_tooluse import (
                COT_INSTRUCT_ANGLE_TOOLUSE,
                COT_INSTRUCT_DISTANCE_TOOLUSE,
                COT_INSTRUCT_TL_TOOLUSE,
            )
            from medvision_bm.sft.sft_utils import (
                _doc_to_text_AngleDistanceTask_CoT,
                _doc_to_text_TumorLesionTask_CoT,
            )

            task_type = "AD" if "biometric_profile" in doc else "TL"
            if task_type == "AD":
                prompt_text, values_dict = _doc_to_text_AngleDistanceTask_CoT(
                    doc,
                    model_name=args.model_hf_id,
                    model_hf=args.model_hf_id,
                )
                metric_type = values_dict.get("metric_type", "distance")
                instruct = (
                    COT_INSTRUCT_DISTANCE_TOOLUSE
                    if metric_type == "distance"
                    else COT_INSTRUCT_ANGLE_TOOLUSE
                )
            else:
                prompt_text, values_dict = _doc_to_text_TumorLesionTask_CoT(
                    doc,
                    model_name=args.model_hf_id,
                    model_hf=args.model_hf_id,
                )
                instruct = COT_INSTRUCT_TL_TOOLUSE

            prompt_base = prompt_text.rsplit("Report the reasoning process", 1)[
                0
            ].rstrip()
            user_text = prompt_base + " " + instruct

            messages = [
                {"role": "system", "content": json.dumps(TOOL_DEF)},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                messages,
                tools=[TOOL_DEF],
                add_generation_prompt=True,
                tokenize=False,
            ).strip()
            prompts.append(prompt)

        # Phase 1: generate until </tool_call>
        params1 = SamplingParams(
            stop=["</tool_call>"], max_tokens=args.max_tokens_phase1
        )
        outputs1 = llm.generate(prompts, params1)

        # Between phases: execute code, build continuation
        new_prompts = []
        for out in outputs1:
            gen = out.outputs[0].text
            code = extract_code(gen)
            result = safe_exec_python(code) if code else "ERROR: no tool call found"
            new_prompts.append(out.prompt + gen + build_continuation(result))

        # Phase 2: generate <answer>...</answer>
        params2 = SamplingParams(stop=["<|im_end|>"], max_tokens=args.max_tokens_phase2)
        outputs2 = llm.generate(new_prompts, params2)

        # Write JSONL output compatible with parse_outputs.py.
        # parse_outputs.py reads data["resps"][0][0] as the model response text and
        # data["target"] as the ground-truth string.  We store phase2_output (which
        # contains the <answer>...</answer> tag) in resps[0][0] so that
        # extract_last_k_nums_within_answer_tag can parse it without any changes to
        # parse_outputs.py.  model_output and phase1_output are kept as extra fields
        # for debugging but are not consumed by the downstream parser.
        from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
            doc_to_target_BiometricsFromLandmarks,
            doc_to_target_TumorLesionSize,
        )

        out_dir = os.path.join(args.results_dir, args.model_name)
        os.makedirs(out_dir, exist_ok=True)
        # Filename must match the pattern `{task_id}_samples_{suffix}.jsonl` expected
        # by parse_outputs.py's _extract_task_id regex: ([^/\\]+)_samples_
        out_path = os.path.join(out_dir, f"{task}_samples_0.jsonl")
        # Write companion results JSON with minimal structure required by parse_outputs.py
        results_json_path = os.path.join(out_dir, f"{task}_results.json")
        with open(results_json_path, "w") as rf:
            json.dump({"results": {task: {}}}, rf)
        with open(out_path, "w") as f:
            for doc, out1, out2 in zip(docs, outputs1, outputs2):
                task_type = "AD" if "biometric_profile" in doc else "TL"
                if task_type == "AD":
                    target = doc_to_target_BiometricsFromLandmarks(doc)
                else:
                    target = doc_to_target_TumorLesionSize(doc)
                phase2_text = out2.outputs[0].text
                record = {
                    "doc_id": doc.get("doc_id", ""),
                    "doc": dict(doc),
                    "target": str(target),
                    # resps format expected by parse_outputs.py: [[response_string]]
                    "resps": [[phase2_text]],
                    # extra fields for debugging / traceability
                    "model_output": out1.outputs[0].text
                    + build_continuation("")
                    + phase2_text,
                    "phase1_output": out1.outputs[0].text,
                    "phase2_output": phase2_text,
                }
                f.write(json.dumps(record) + "\n")
        print(
            f"Wrote {len(docs)} samples to {out_path} (results JSON: {results_json_path})"
        )

        if not args.skip_update_status:
            update_task_status(args.task_status_json_path, args.model_name, task)


if __name__ == "__main__":
    main()
