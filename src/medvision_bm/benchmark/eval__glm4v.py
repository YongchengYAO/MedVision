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


def install_transformers_for_glm4v(transformers_version="5.12.1"):
    # NOTE: Reinstall transformers to a version compatible with BOTH GLM-4.6V and vLLM.
    # GLM-4.6V REQUIRES transformers>=5.2.0: its preprocessor_config.json declares the
    # Glm46VImageProcessor / Glm46VProcessor classes, which exist ONLY in transformers 5.2.0+
    # (4.57.x ships just the older Glm4vImageProcessor and fails with "Unrecognized image
    # processor"). That in turn needs a vLLM that accepts transformers 5.x: vLLM 0.19.x allows
    # transformers>=5.6 (its requires_dist excludes only 5.0-5.5.0, no upper bound) and ships the
    # ALLOWED_LAYER_TYPES fallback so `import vllm` works on transformers 5.x. The version must be in
    # BOTH ranges (>=5.2.0 for GLM, >=5.6 for vLLM 0.19.x) -> 5.12.1 (latest) works. Note: vLLM 0.12.0
    # CANNOT run GLM-4.6V (it pins transformers<5, which lacks Glm46VImageProcessor). Run LAST so
    # transformers wins the dependency resolution.
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
        "--verbosity=INFO",
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
        default="vllm_glm4v",
        type=str,
        help="lmms-eval model module name.",
    )
    parser.add_argument(
        "--model_hf_id",
        default="zai-org/GLM-4.6V",
        type=str,
        help="Hugging Face model ID. Use 'zai-org/GLM-4.6V' (MoE) or 'zai-org/GLM-4.6V-Flash' (dense).",
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        type=str,
        help="Hugging Face path to LoRA adapter (if using LoRA). If not using LoRA, set to empty string or leave unset.",
    )
    parser.add_argument(
        "--model_name",
        default="GLM-4.6V",
        type=str,
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        type=str,
        help="Data type for model weights (e.g., float32, float16, bfloat16). Default is 'auto', which uses the model config or falls back to fp16 in vllm for fp16/32 models.",
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
    # NOTE: GLM-4.6V is a hybrid-reasoning model; greedy decoding (temperature=0) makes it stop
    # early inside <think> before emitting <answer>. Defaults below mirror the model's
    # generation_config.json (temperature=0.8, top_p=0.6, top_k=2). repetition_penalty=1.1 is the
    # model-card recommendation (not in generation_config). Generation uses a fixed seed
    # internally for reproducibility.
    parser.add_argument(
        "--temperature",
        default=0.8,
        type=float,
        help="Sampling temperature. Do NOT set to 0 for this reasoning model.",
    )
    parser.add_argument(
        "--top_p",
        default=0.6,
        type=float,
        help="Nucleus (top-p) sampling probability.",
    )
    parser.add_argument(
        "--top_k",
        default=2,
        type=int,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--repetition_penalty",
        default=1.1,
        type=float,
        help="Repetition penalty (GLM-4.6V model-card recommendation).",
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
        install_vendored_lmms_eval(proj_dependency="glm4v")
        install_medvision_ds(data_dir)
        install_torch_cu124()

        # NOTE: vLLM 0.19.x is required for GLM-4.6V. vLLM 0.12.0 pins transformers<5 and CANNOT load
        # GLM-4.6V's Glm46VImageProcessor (transformers 5.2.0+ only); vLLM 0.19.x allows transformers
        # >=5.6 (excludes only 5.0-5.5.0) and has the ALLOWED_LAYER_TYPES fallback. Supports glm4v/glm4v_moe.
        install_vllm(data_dir, version="0.19.1")

        # NOTE: Reinstall transformers LAST so it overrides any version pulled in by vLLM/lmms-eval.
        # Must be >=5.2.0 for GLM-4.6V's Glm46V processor AND >=5.6 for vLLM 0.19.x (which excludes
        # transformers 5.0-5.5.0); 5.12.1 satisfies both (see install_transformers_for_glm4v).
        install_transformers_for_glm4v(transformers_version="5.12.1")

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

    # GLM-4.6V runs on vLLM 0.19.1 (V1 engine), which executes the model in separate worker
    # processes. Two interacting issues on multi-GPU setups must BOTH be worked around, else engine
    # init fails with "Engine core initialization failed":
    #   1. setup_env_vllm() forces VLLM_WORKER_MULTIPROC_METHOD=spawn. With vLLM 0.19.1 + GLM-4.6V the
    #      *spawned* EngineCore dies silently during bootstrap (empty "Failed core proc(s): {}", no
    #      child traceback). Using "fork" avoids that.
    #   2. But lmms_eval's __main__ builds an accelerate Accelerator() before the model is created,
    #      which initializes CUDA in the launcher process -- and a forked vLLM worker then cannot
    #      re-initialize CUDA ("Cannot re-initialize CUDA in forked subprocess"). Forcing accelerate
    #      onto CPU stops that launcher-side CUDA init; vLLM still uses the GPUs because it reads
    #      CUDA_VISIBLE_DEVICES directly, independent of accelerate.
    # With both set, GLM-4.6V loads and runs on 1 or 2 GPUs. The lmms_eval subprocess launched below
    # inherits these env vars.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    os.environ["ACCELERATE_USE_CPU"] = "true"

    # NOTE: Opt out of the dataset loader's load-time reinstall of medvision_ds, which would
    # break this stack. At dataset-load time the loader runs `pip install .` on the medvision_ds
    # source unless MedVision_FORCE_INSTALL_CODE is explicitly "false" -- it defaults to TRUE
    # when unset, and setup_env_hf_medvision_ds() sets it to "true". That reinstall applies
    # medvision_ds's own huggingface_hub==0.36.0 pin, silently downgrading huggingface_hub
    # mid-run (pip output is swallowed by capture_output=True). The running process keeps its
    # already-imported transformers, but every process started afterwards -- a spawned vLLM
    # worker, or the next task's lmms_eval -- dies with "ImportError: cannot import name
    # 'is_offline_mode' from 'huggingface_hub'": GLM-4.6V needs transformers 5.x, which needs
    # huggingface_hub>=1.5.0. Nothing is lost: the latest medvision_ds is installed on every
    # run by install_medvision_ds() above (or by the launcher's own install step), and crucially
    # BEFORE the requirements re-pin huggingface_hub. Must be set AFTER the block above, since
    # install_medvision_ds() re-sets this flag to "true" on its way out.
    os.environ["MedVision_FORCE_INSTALL_CODE"] = "false"

    tasks = load_tasks(tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(task_status_json_path, model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        batch_size = args.batch_size_per_gpu * num_processes

        # NOTE: Unlike Qwen3-VL, GLM-4.6V needs NO hf_overrides -- vLLM supports glm4v / glm4v_moe
        # natively (`vllm serve zai-org/GLM-4.6V` runs bare), so no config patching is required.
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
            f"repetition_penalty={args.repetition_penalty},"
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
