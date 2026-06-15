import argparse
import json
import os
import subprocess

from huggingface_hub import snapshot_download

from medvision_bm.benchmark.eval_utils import parse_sample_indices

from medvision_bm.utils import (
    ensure_hf_hub_installed,
    install_flash_attention_torch_and_deps_py311_v2,
    install_medvision_ds,
    install_vendored_lmms_eval,
    load_tasks,
    load_tasks_status,
    set_cuda_num_processes,
    setup_env_hf_medvision_ds,
    update_task_status,
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
            "model_dtype": "float16",
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
            "model_dtype": "bfloat16",
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
    sample_indices: list = None,
    log_sys_prompt: bool = False,
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
        "--model_name",
        default="HealthGPT-L14",
        type=str,
        help="Name of the model to evaluate.",
    )
    # output length arguments
    parser.add_argument(
        "--max_new_tokens",
        default=4096,
        type=int,
        help="Maximum number of new tokens to generate.",
    )
    # resource-specific arguments
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
        "--reshape_image_hw",
        default=None,
        type=str,
        help="Reshape images to this height and width (format: H,W) before feeding into the model. Default is None.",
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
    model_name = args.model_name
    tasks_list_json_path = args.tasks_list_json_path
    result_dir = args.results_dir
    dir_third_party = args.dir_third_party
    task_status_json_path = args.task_status_json_path
    data_dir = args.data_dir
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
        if args.env_setup_only:
            print(
                "\nEnvironment setup completed as per argument --env_setup_only. Exiting now.\n"
            )
            return
    else:
        print(
            "\n[Warning] Skipping environment setup as per argument --skip_env_setup. This should only be used for debugging.\n"
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
        model_dtype = model_configs.get("model_dtype")
        model_args = (
            f"base_model_hf={base_model_hf},"
            f"vision_model_hf={vision_model_hf},"
            f"hlora_weights_local={hlora_weights_local},"
            f"instruct_template={instruct_template},"
            f"hlora_r={hlora_r},"
            f"hlora_alpha={hlora_alpha},"
            f"hlora_nums={hlora_nums},"
            f"vq_idx_nums={vq_idx_nums},"
            f"model_dtype={model_dtype},"
            f"max_new_tokens={max_new_tokens}"
        )

        # add reshape_image_hw to model args if specified, with normalization to ensure correct parsing
        if args.reshape_image_hw is not None:
            raw = args.reshape_image_hw
            if isinstance(raw, str):
                s = raw.strip()
                s = ",".join(s.split()) if (" " in s) and ("," not in s) else s
                if not (s.startswith("[") or s.startswith("(")) and "," in s:
                    s = f"[{s}]"
            else:
                s = raw
            model_args += f",reshape_image_hw={s}"

        batch_size = args.batch_size_per_gpu * num_processes

        module = "healthgpt"

        parsed_sample_indices = None
        if args.sample_indices is not None:
            parsed_sample_indices = parse_sample_indices(args.sample_indices)

        rc = run_evaluation_for_task(
            num_processes=num_processes,
            lmmseval_module=module,
            model_args=model_args,
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
