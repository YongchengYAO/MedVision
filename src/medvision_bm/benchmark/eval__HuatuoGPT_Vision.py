import os
import subprocess
import argparse
import sys
from medvision_bm.utils import (
    load_tasks,
    load_tasks_status,
    update_task_status,
    set_cuda_num_processes,
    setup_env_var,
    ensure_hf_hub_installed,
    install_vendored_lmms_eval,
    install_medvision_ds,
    install_flash_attention_torch_and_deps_py311_v2,
    clone_github_repo_with_fallback,
)


def install_huatuogpt_vision_dependencies_post(dir_third_party: str):
    # ------------------------------
    # NOTE: This is specific to the HuatuoGPT-Vision model
    # NOTE: Put this section after lmms-eval installation to avoid conflicts
    # ------------------------------
    # Install HuatuoGPT-Vision codebase
    os.makedirs(dir_third_party, exist_ok=True)
    folder_name = "HuatuoGPT-Vision"
    dir_huatuogpt_vision = os.path.join(dir_third_party, folder_name)

    if not os.path.exists(dir_huatuogpt_vision):
        # NOTE: Fix codebase to a specific commit
        github_commit = (
            "e1a52dcf6c0417f4b6ac1d378b01147280192fca"  # Commits on Apr 23, 2025
        )
        repo_url = "https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git"

        clone_github_repo_with_fallback(
            repo_url=repo_url,
            target_dir=dir_huatuogpt_vision,
            commit_hash=github_commit,
            parent_dir=dir_third_party,
        )

    # NOTE: This is a workaround for the issue with the import of the modules
    os.environ["HuatuoGPTVision_DIR"] = dir_huatuogpt_vision

    # NOTE: Workaround of some issues in huatuogpt-vision codebase
    subprocess.run("pip install transformers==4.40.0", check=True, shell=True)
    # ------------------------------


def run_evaluation_for_task(
    num_processes: int,
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
        "accelerate.commands.launch",
        f"--num_processes={num_processes}",
        "--main_process_port=29501",
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
        default="FreedomIntelligence/HuatuoGPT-Vision-34B",
        type=str,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--model_name",
        default="HuatuoGPT-Vision-34B",
        type=str,
        help="Name of the model to evaluate.",
    )
    # resource-specific arguments
    parser.add_argument(
        "--minimum_gpu",
        default=1,
        type=int,
        help="Minimum number of GPUs to use.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=4,
        type=int,
        help="Batch size per GPU.",
    )
    # task-specific arguments
    parser.add_argument(
        "--tasks_list_json_path",
        type=str,
        help="Path to the tasks list JSON file.",
    )
    # data, output and status paths
    parser.add_argument(
        "--dir_third_party",
        type=str,
        help="Path to the third-party directory.",
    )
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
    dir_third_party = args.dir_third_party
    task_status_json_path = args.task_status_json_path
    data_dir = args.data_dir
    sample_limit = args.sample_limit

    num_processes = set_cuda_num_processes(minimum_gpu=args.minimum_gpu)

    # NOTE: DO NOT change the order of these calls
    # ------
    setup_env_var(data_dir)
    if not args.skip_env_setup:
        ensure_hf_hub_installed()
        install_vendored_lmms_eval(proj_dependency="huatuogpt_vision")
        install_medvision_ds(data_dir)
        install_flash_attention_torch_and_deps_py311_v2()
        install_huatuogpt_vision_dependencies_post(dir_third_party)
    else:
        print(
            f"\n[Warning] Skipping environment setup as per argument --skip_env_setup. This should only be used for debugging.\n"
        )
    # ------

    tasks = load_tasks(tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(task_status_json_path, model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        batch_size = args.batch_size_per_gpu * num_processes
        model_args = f"model_path={model_hf}," "device_map=auto"

        rc = run_evaluation_for_task(
            num_processes=num_processes,
            lmmseval_module="huatuogpt_vision",
            model_args=model_args,
            task=task,
            batch_size=batch_size,
            sample_limit=sample_limit,
            output_path=result_dir,
        )

        if rc == 0 and not args.skip_update_status:
            update_task_status(task_status_json_path, model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
