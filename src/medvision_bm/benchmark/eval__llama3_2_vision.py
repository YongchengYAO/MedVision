import argparse
import json
import os
import subprocess

from medvision_bm.benchmark.eval_utils import parse_sample_indices
from medvision_bm.utils import (
    ensure_hf_hub_installed,
    install_medvision_ds,
    install_torch_cu124,
    install_vendored_lmms_eval,
    install_vllm,
    load_tasks,
    load_tasks_status,
    set_cuda_num_processes,
    setup_env_hf_medvision_ds,
    setup_env_vllm,
    update_task_status,
)


def run_evaluation_for_task_vllm_proxy(
    lmmseval_module: str,
    model_args: str,
    task: str,
    batch_size: int,
    sample_limit: int,
    output_path: str,
    sample_indices: list = None,
):
    print(f"\nRunning task: {task}\n")
    subprocess.run("conda env list", check=True, shell=True)
    cmd = [
        "python3",
        "-m",
        "lmms_eval",
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
    ]
    if sample_indices is not None:
        cmd += ["--sample_indices", json.dumps(sample_indices)]
    cmd_result = subprocess.run(cmd, check=False)
    print(f"Command executed with return code: {cmd_result.returncode}")
    return cmd_result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedVision benchmarking.")
    # model-specific arguments
    parser.add_argument(
        "--model_hf_id",
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        type=str,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--model_name",
        default="Llama-3.2-11B-Vision-Instruct",
        type=str,
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        type=str,
        help="Hugging Face path to LoRA adapter (if using LoRA). If not using LoRA, set to empty string or leave unset.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        type=str,
        help="Data type for model weights (e.g., float32, float16, bfloat16). Default is 'auto', which uses the model config or falls back to fp16 in vllm for fp16/32 models.",
    )
    parser.add_argument(
        "--reshape_image_hw",
        default=None,
        type=str,
        help="Reshape images to this height and width (format: H,W) before feeding into the model. Default is None.",
    )
    # set max_new_tokens
    parser.add_argument(
        "--max_new_tokens",
        default=4096,
        type=int,
        help="Maximum number of new tokens to generate per sample.",
    )
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=None,
        metavar="STRING",
        help=(
            "Stop strings for generation (e.g. '</answer>'). "
            "Generation halts at the first match. "
            "Passed to the model as stop sequences for all tasks."
        ),
    )
    # resource-specific arguments
    parser.add_argument(
        "--batch_size_per_gpu",
        default=10,
        type=int,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        default=0.99,
        type=float,
        help="GPU memory utilization fraction, used in vllm",
    )
    # task-specific arguments
    parser.add_argument(
        "--tasks_list_json_path",
        type=str,
        help="Path to the tasks list JSON file.",
    )
    # data, output and status paths
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--task_status_json_path",
        type=str,
        help="Path to the task status JSON file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the MedVision data directory.",
    )
    # evaluation-specific arguments
    parser.add_argument(
        "--sample_limit",
        default=1000,
        type=int,
        help="Maximum number of samples to evaluate per task.",
    )
    parser.add_argument(
        "--sample_indices",
        default=None,
        type=str,
        metavar="[start:stop]|[start,stop,step]",
        help=(
            "Select a subset of samples by index for partial inference. "
            "Accepted formats: [start:stop] (range) or [start,stop,step] (range with step). "
            "When set, overrides --sample_limit for sample selection."
        ),
    )
    # debugging and control arguments
    parser.add_argument(
        "--skip_env_setup",
        action="store_true",
        help="Skip environment setup steps.",
    )
    parser.add_argument(
        "--skip_update_status",
        action="store_true",
        help="Skip updating task status after completion -- useful for debugging.",
    )
    parser.add_argument(
        "--env_setup_only",
        action="store_true",
        help="Only perform environment setup and exit.",
    )
    parser.add_argument(
        "--scaled_ps_low",
        default=0.5,
        type=float,
        help="Lower bound of the pixel-size scaling factor range for -scaledPS task variants.",
    )
    parser.add_argument(
        "--scaled_ps_high",
        default=3.0,
        type=float,
        help="Upper bound of the pixel-size scaling factor range for -scaledPS task variants.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["MEDVISION_SCALED_PS_LOW"] = str(args.scaled_ps_low)
    os.environ["MEDVISION_SCALED_PS_HIGH"] = str(args.scaled_ps_high)

    # Configuration
    model_hf = args.model_hf_id
    model_name = args.model_name
    lora_path = args.lora_path
    dtype = args.dtype
    tasks_list_json_path = args.tasks_list_json_path
    result_dir = args.results_dir
    task_status_json_path = args.task_status_json_path
    data_dir = args.data_dir
    gpu_memory_utilization = args.gpu_memory_utilization
    sample_limit = args.sample_limit
    max_new_tokens = args.max_new_tokens

    num_processes = set_cuda_num_processes()

    # NOTE: DO NOT change the order of these calls
    # ------
    setup_env_hf_medvision_ds(data_dir)
    if not args.skip_env_setup:
        # NOTE: Install huggingface-hub, required version may vary for different models, check requirements 
        ensure_hf_hub_installed(hf_hub_version="0.35.3")
        install_vendored_lmms_eval()
        install_medvision_ds(data_dir)
        install_torch_cu124()
        # NOTE: vllm version may need to be adjusted based on compatibility of model and transformers version
        install_vllm(data_dir, version="0.10.2")
        if args.env_setup_only:
            print(
                "\nEnvironment setup completed as per argument --env_setup_only. Exiting now.\n"
            )
            return
    else:
        print(
            "\n[Warning] Skipping environment setup as per argument --skip_env_setup. This should only be used for debugging.\n"
        )
        setup_env_vllm(data_dir)
    # ------

    tasks = load_tasks(tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(task_status_json_path, model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        batch_size = args.batch_size_per_gpu * num_processes

        vllm_model_args = (
            f"model_hf={model_hf},"
            + (f"lora_path={lora_path}," if lora_path is not None else "")
            + f"gpu_memory_utilization={gpu_memory_utilization},"
            f"tensor_parallel_size={num_processes},"
            f"max_num_seqs={batch_size},"  # maximum batch size
            f"max_new_tokens={max_new_tokens},"
            f"dtype={dtype}"

        )

        # add reshape_image_hw to modle args if specified, with normalization to ensure correct parsing
        if args.reshape_image_hw is not None:
            raw = args.reshape_image_hw
            if isinstance(raw, str):
                s = raw.strip()
                s = ",".join(s.split()) if (" " in s) and ("," not in s) else s
                if not (s.startswith("[") or s.startswith("(")) and "," in s:
                    s = f"[{s}]"
            else:
                s = raw
            vllm_model_args += f",reshape_image_hw={s}"

        if args.stop_strings:
            vllm_model_args += f",stop_strings={json.dumps(args.stop_strings, separators=(',', ':'))}"

        parsed_sample_indices = None
        if args.sample_indices is not None:
            parsed_sample_indices = parse_sample_indices(args.sample_indices)

        rc = run_evaluation_for_task_vllm_proxy(
            lmmseval_module="vllm_llama_3_2_vision",
            model_args=vllm_model_args,
            task=task,
            batch_size=batch_size,
            sample_limit=sample_limit,
            output_path=os.path.join(result_dir, model_name),
            sample_indices=parsed_sample_indices,
        )

        if rc == 0:
            if not args.skip_update_status:
                update_task_status(task_status_json_path, model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
