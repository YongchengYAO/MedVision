import argparse
import json
import os
import subprocess
import sys

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


def install_transformers_for_minimax_m3(transformers_version="4.57.1"):
    # NOTE: Reinstall transformers to the version required by MiniMax-M3, overwriting any
    # incompatible version pulled in by other deps (e.g. vLLM). MiniMax-M3 was converted with
    # transformers 4.52.4 and its remote image_processor.py relies on the fast image/video
    # processor API (transformers.image_processing_utils_fast / video_processing_utils), so any
    # transformers >= 4.52 with that API works; we pin the repo-standard 4.57.1 here.
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


def run_evaluation_for_task_vllm_proxy(
    lmmseval_module: str,
    model_args: str,
    task: str,
    batch_size: int,
    sample_limit: int,
    output_path: str,
    sample_indices: list = None,
    log_sys_prompt: bool = False,
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
        "--verbosity=DEBUG",
    ]
    if sample_indices is not None:
        cmd += ["--sample_indices", json.dumps(sample_indices)]
    if log_sys_prompt:
        cmd += ["--log_sys_prompt"]
    cmd_result = subprocess.run(cmd, check=False)
    print(f"Command executed with return code: {cmd_result.returncode}")
    return cmd_result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedVision benchmarking.")
    # model-specific arguments
    parser.add_argument(
        "--lmmseval_module",
        default="vllm_minimax_m3",
        type=str,
        help="lmms-eval model module name.",
    )
    parser.add_argument(
        "--model_hf_id",
        default="MiniMaxAI/MiniMax-M3",
        type=str,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        type=str,
        help="Hugging Face path to LoRA adapter (if using LoRA). If not using LoRA, set to empty string or leave unset.",
    )
    parser.add_argument(
        "--model_name",
        default="MiniMax-M3",
        type=str,
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        type=str,
        help="Data type for model weights (e.g., float32, float16, bfloat16). Default is 'auto', which uses the model config (bfloat16 for MiniMax-M3).",
    )
    # set reshape_image_hw
    parser.add_argument(
        "--reshape_image_hw",
        default=None,
        type=str,
        help="Reshape images to this height and width (format: H,W) before feeding into the model. Default is None, which means no reshaping will be applied.",
    )
    # set max_new_tokens
    parser.add_argument(
        "--max_new_tokens",
        default=4096,
        type=int,
        help="Maximum number of new tokens to generate per sample.",
    )
    # Sampling parameters.
    # NOTE: MiniMax-M3 is a reasoning model; greedy decoding (temperature=0) is not recommended.
    # Defaults below mirror the checkpoint's generation_config.json (do_sample=true, temperature=1.0,
    # top_p=0.95) plus the model card's top_k=40. Generation uses a fixed seed internally.
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Sampling temperature. Do NOT set to 0 for this reasoning model.",
    )
    parser.add_argument(
        "--top_p",
        default=0.95,
        type=float,
        help="Nucleus (top-p) sampling probability.",
    )
    parser.add_argument(
        "--top_k",
        default=40,
        type=int,
        help="Top-k sampling cutoff.",
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
        default=20,
        type=int,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        default=0.99,
        type=float,
        help="GPU memory utilization fraction, used in vllm",
    )
    parser.add_argument(
        "--vllm_version",
        default="0.11.0",
        type=str,
        help=(
            "vLLM version to install. IMPORTANT: MiniMax-M3 VL (architecture "
            "MiniMaxM3SparseForConditionalGeneration, model_type minimax_m3_vl) needs a vLLM "
            "build that natively registers this architecture -- the checkpoint ships no HF "
            "modeling file. The default here is a starting point; if vLLM raises an "
            "'unsupported architecture' error, set this to the release that lands MiniMax-M3-VL "
            "support."
        ),
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
    parser.add_argument(
        "--log-sys-prompt",
        action="store_true",
        default=False,
        help="If set, log the system prompt (if any) in the per-sample JSONL output files.",
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
    dtype = args.dtype
    lora_path = args.lora_path
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
        install_vendored_lmms_eval(proj_dependency="minimax_m3")
        install_medvision_ds(data_dir)
        install_torch_cu124()

        # NOTE: MiniMax-M3 VL needs a vLLM build that natively registers the minimax_m3_vl
        # architecture (the checkpoint ships no HF modeling file). Set --vllm_version to that
        # release if the default does not support it.
        install_vllm(data_dir, version=args.vllm_version)

        # NOTE: Reinstall transformers to overwrite potentially incompatible versions
        install_transformers_for_minimax_m3(transformers_version="4.57.1")

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
            f"temperature={args.temperature},"
            f"top_p={args.top_p},"
            f"top_k={args.top_k},"
            f"dtype={dtype}"
            + (
                f",stop_strings={json.dumps(args.stop_strings, separators=(',', ':'))}"
                if args.stop_strings
                else ""
            )
        )

        # add reshape_image_hw to model args if specified, with normalization to ensure correct parsing
        if args.reshape_image_hw is not None:
            raw = args.reshape_image_hw
            if isinstance(raw, str):
                s = raw.strip()
                # replace whitespace with comma
                s = ",".join(s.split()) if (" " in s) and ("," not in s) else s
                if not (s.startswith("[") or s.startswith("(")) and "," in s:
                    s = f"[{s}]"
            else:
                s = raw
            vllm_model_args += f",reshape_image_hw={s}"

        parsed_sample_indices = None
        if args.sample_indices is not None:
            parsed_sample_indices = parse_sample_indices(args.sample_indices)

        rc = run_evaluation_for_task_vllm_proxy(
            lmmseval_module=args.lmmseval_module,
            model_args=vllm_model_args,
            task=task,
            batch_size=batch_size,
            sample_limit=sample_limit,
            output_path=os.path.join(result_dir, model_name),
            sample_indices=parsed_sample_indices,
            log_sys_prompt=args.log_sys_prompt,
        )

        if rc == 0:
            if not args.skip_update_status:
                update_task_status(task_status_json_path, model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
