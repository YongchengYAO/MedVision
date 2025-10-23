import argparse
import os
import subprocess

from medvision_bm.utils import (ensure_hf_hub_installed, install_medvision_ds,
                                install_torch_cu124,
                                install_vendored_lmms_eval, install_vllm,
                                load_tasks, load_tasks_status,
                                set_cuda_num_processes,
                                setup_env_hf_medvision_ds, setup_env_vllm,
                                update_task_status)


def run_evaluation_for_task_vllm_proxy(
    lmmseval_module: str,
    model_args: str,
    task: str,
    batch_size: int,
    sample_limit: int,
    output_path: str,
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
    cmd_result = subprocess.run(cmd, check=False)
    print(f"Command executed with return code: {cmd_result.returncode}")
    return cmd_result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedVision benchmarking.")
    # model-specific arguments
    parser.add_argument(
        "--model_hf_id",
        default="OpenGVLab/InternVL3-38B",
        type=str,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--model_name",
        default="InternVL3-38B",
        type=str,
        help="Name of the model to evaluate.",
    )
    # resource-specific arguments
    parser.add_argument(
        "--minimum_gpu",
        default=2,
        type=int,
        help="Minimum number of GPUs to use.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=1,
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    model_hf = args.model_hf_id
    model_name = args.model_name
    tasks_list_json_path = args.tasks_list_json_path
    result_dir = args.results_dir
    task_status_json_path = args.task_status_json_path
    data_dir = args.data_dir
    gpu_memory_utilization = args.gpu_memory_utilization
    sample_limit = args.sample_limit

    num_processes = set_cuda_num_processes(minimum_gpu=args.minimum_gpu)

    # NOTE: DO NOT change the order of these calls
    # ------
    setup_env_hf_medvision_ds(data_dir)
    if not args.skip_env_setup:
        ensure_hf_hub_installed()
        install_vendored_lmms_eval()
        install_medvision_ds(data_dir)
        install_torch_cu124()
        # NOTE: vllm version may need to be adjusted based on compatibility of model and transformers version
        install_vllm(data_dir, version="0.10.0")
    else:
        print(
            f"\n[Warning] Skipping environment setup as per argument --skip_env_setup. This should only be used for debugging.\n"
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
            f"model_version={model_hf},"
            f"gpu_memory_utilization={gpu_memory_utilization},"
            f"tensor_parallel_size={num_processes},"
            f"max_num_seqs={batch_size},"  # maximum batch size
            # "max_model_len=1024," # (optional) maximum input length
            # for InternVL3, https://github.com/vllm-project/vllm/blob/f3137cdd81cae3a48282c22130fbcadcfc64ea95/examples/offline_inference/vision_language.py#L393
            'limit_mm_per_prompt={"image": 1},'
            "dtype=bfloat16"  # https://huggingface.co/OpenGVLab/InternVL3-38B
        )

        rc = run_evaluation_for_task_vllm_proxy(
            lmmseval_module="vllm_internvl3",
            model_args=vllm_model_args,
            task=task,
            batch_size=batch_size,
            sample_limit=sample_limit,
            output_path=os.path.join(result_dir, model_name),
        )

        if rc == 0 and not args.skip_update_status:
            update_task_status(task_status_json_path, model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
