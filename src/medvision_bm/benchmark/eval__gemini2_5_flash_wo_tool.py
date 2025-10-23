import argparse
import os
import subprocess

from medvision_bm.utils import (ensure_hf_hub_installed, install_medvision_ds,
                                install_vendored_lmms_eval, load_tasks,
                                load_tasks_status, setup_env_hf_medvision_ds,
                                update_task_status)


def run_evaluation_for_task_API_models(
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
        "--verbosity=DEBUG",
    ]
    cmd_result = subprocess.run(cmd, check=False)
    print(f"Command executed with return code: {cmd_result.returncode}")
    return cmd_result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Run MedVision benchmarking.")
    # model-specific arguments
    parser.add_argument(
        "--model_name",
        default="gemini-2.5-flash-woTool",
        type=str,
        help="Name of the model to evaluate.",
    )
    # resource-specific arguments
    parser.add_argument(
        "--batch_size",
        default=1,
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
    model_name = args.model_name
    tasks_list_json_path = args.tasks_list_json_path
    result_dir = args.results_dir
    task_status_json_path = args.task_status_json_path
    data_dir = args.data_dir
    sample_limit = args.sample_limit

    # NOTE: DO NOT change the order of these calls
    # ------
    setup_env_hf_medvision_ds(data_dir)
    if not args.skip_env_setup:
        ensure_hf_hub_installed()
        install_vendored_lmms_eval(proj_dependency="gemini")
        install_medvision_ds(data_dir)
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

        # model configuration for Gemini 2.5 without tool use
        model_args = (
            "model=gemini-2.5-flash,"
            "thinkingBudget=-1,"
            "use_tool=False,"
            "json_output=True,"
            "ignore_thoughts=False"
        )

        rc = run_evaluation_for_task_API_models(
            lmmseval_module="gemini__2_5",
            model_args=model_args,
            task=task,
            batch_size=args.batch_size,
            sample_limit=sample_limit,
            output_path=os.path.join(result_dir, model_name),
        )

        if rc == 0 and not args.skip_update_status:
            update_task_status(task_status_json_path, model_name, task)
        else:
            print(f"Warning: Task {task} failed (return code {rc})")


if __name__ == "__main__":
    main()
