import os
import subprocess
import argparse
from huggingface_hub import snapshot_download
from medvision_bm.utils import (
    load_tasks,
    load_tasks_status,
    update_task_status,
    set_cuda_num_processes,
    setup_env_hf_medvision_ds,
    ensure_hf_hub_installed,
    install_vendored_lmms_eval,
    install_medvision_ds,
    install_flash_attention_torch_and_deps_py311_v2,
)


def install_healthgpt_dependencies_post(dir_third_party: str, model_name: str):
    # ------------------------------
    # NOTE: This is specific to the HealthGPT model
    # NOTE: Put this section after lmms-eval installation to avoid conflicts
    # ------------------------------
    os.makedirs(dir_third_party, exist_ok=True)
    folder_name = "HealthGPT"
    dir_healthgpt = os.path.join(dir_third_party, folder_name)

    # NOTE: This is a workaround for the issue with the import of the HealthGPT module
    os.environ["HEALTHGPT_DIR"] = dir_healthgpt

    # Install codebase
    if not os.path.exists(dir_healthgpt):
        # NOTE: Fix codebase to a specific commit
        github_commit = (
            "c044a13254b76c5eec8c2e6c55e3324318c27940"  # Commits on Apr 23, 2025
        )
        try:
            # Clone the repository
            subprocess.run(
                f"git clone https://github.com/DCDmllm/HealthGPT.git {folder_name}",
                cwd=dir_third_party,
                check=True,
                shell=True,
            )
            # Checkout specific commit
            subprocess.run(
                f"git checkout {github_commit}",
                cwd=dir_healthgpt,
                check=True,
                shell=True,
            )
        except Exception:
            raise RuntimeError(
                f"Failed to clone HealthGPT repository at commit {github_commit}."
            )

    # Workaround for pydantic and deepspeed conflicts
    subprocess.run("pip install pydantic==1.10.24", check=True, shell=True)
    
    # Install requirements
    subprocess.run(
        "pip install -r requirements.txt", cwd=dir_healthgpt, check=True, shell=True
    )

    # Download model weights
    assert model_name in [
        "HealthGPT-XL32",
        "HealthGPT-L14",
    ], f"Unsupported model name: {model_name}"
    if model_name == "HealthGPT-XL32":
        hlora_path_hf = "lintw/HealthGPT-XL32"
        hlora_filename = "com_hlora_weights_QWEN_32B.bin"
        snapshot_download(
            repo_id=hlora_path_hf,
            allow_patterns=hlora_filename,
            local_dir=dir_healthgpt,
        )
        base_model_hf = "Qwen/Qwen2.5-32B-Instruct"  # HF identifier
        vision_model_hf = "openai/clip-vit-large-patch14-336"  # HF identifier
        hlora_weights_local = os.path.join(dir_healthgpt, hlora_filename)
        instruct_template = "phi4_instruct"
        hlora_r = 32
        hlora_alpha = 64
        hlora_nums = 4
        vq_idx_nums = 8192
        # NOTE: !!! Important for debugging resized image size in medvision_utils._process_img_healthgpt_XL32
        os.environ["HEALTHGPT-XL32-HLORA-WEIGHTS-FILE"] = hlora_weights_local
        return {
            "base_model_hf": base_model_hf,
            "vision_model_hf": vision_model_hf,
            "hlora_weights_local": hlora_weights_local,
            "instruct_template": instruct_template,
            "hlora_r": hlora_r,
            "hlora_alpha": hlora_alpha,
            "hlora_nums": hlora_nums,
            "vq_idx_nums": vq_idx_nums,
        }
    elif model_name == "HealthGPT-L14":
        hlora_path_hf = "lintw/HealthGPT-L14"
        hlora_filename = "com_hlora_weights_phi4.bin"
        snapshot_download(
            repo_id=hlora_path_hf,
            allow_patterns=hlora_filename,
            local_dir=dir_healthgpt,
        )
        base_model_hf = "microsoft/phi-4"  # HF identifier
        vision_model_hf = "openai/clip-vit-large-patch14-336"  # HF identifier
        hlora_weights_local = os.path.join(dir_healthgpt, hlora_filename)
        instruct_template = "phi4_instruct"
        hlora_r = 32
        hlora_alpha = 64
        hlora_nums = 4
        vq_idx_nums = 8192
        # NOTE: !!! Important for debugging resized image size in medvision_utils._process_img_healthgpt_L14
        os.environ["HEALTHGPT-L14-HLORA-WEIGHTS-FILE"] = hlora_weights_local
        return {
            "base_model_hf": base_model_hf,
            "vision_model_hf": vision_model_hf,
            "hlora_weights_local": hlora_weights_local,
            "instruct_template": instruct_template,
            "hlora_r": hlora_r,
            "hlora_alpha": hlora_alpha,
            "hlora_nums": hlora_nums,
            "vq_idx_nums": vq_idx_nums,
        }
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
        "--model_name",
        default="HealthGPT-L14",
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
        default=20,
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
    setup_env_hf_medvision_ds(data_dir)
    if not args.skip_env_setup:
        ensure_hf_hub_installed()
        install_vendored_lmms_eval()
        install_medvision_ds(data_dir)
    else:
        print(
            f"\n[Warning] Skipping environment setup as per argument --skip_env_setup. This should only be used for debugging.\n"
        )
    model_configs = install_healthgpt_dependencies_post(dir_third_party, model_name)
    install_flash_attention_torch_and_deps_py311_v2()
    # ------

    tasks = load_tasks(tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(task_status_json_path, model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        base_model_hf = model_configs.get("base_model_hf")
        vision_model_hf = model_configs.get("vision_model_hf")
        hlora_weights_local = model_configs.get("hlora_weights_local")
        instruct_template = model_configs.get("instruct_template")
        hlora_r = model_configs.get("hlora_r")
        hlora_alpha = model_configs.get("hlora_alpha")
        hlora_nums = model_configs.get("hlora_nums")
        vq_idx_nums = model_configs.get("vq_idx_nums")
        model_args = (
            f"base_model_hf={base_model_hf},"
            f"vision_model_hf={vision_model_hf},"
            f"hlora_weights_local={hlora_weights_local},"
            f"instruct_template={instruct_template},"
            f"hlora_r={hlora_r},"
            f"hlora_alpha={hlora_alpha},"
            f"hlora_nums={hlora_nums},"
            f"vq_idx_nums={vq_idx_nums},"
            "device_map=auto,"
            "dtype=FP16"  # ["FP16", "FP32", "BF16"]
        )

        batch_size = args.batch_size_per_gpu * num_processes

        if model_name == "HealthGPT-L14":
            module = "healthgpt_l14"
        elif model_name == "HealthGPT-XL32":
            module = "healthgpt_xl32"

        rc = run_evaluation_for_task(
            num_processes=num_processes,
            lmmseval_module=module,
            model_args=model_args,
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
