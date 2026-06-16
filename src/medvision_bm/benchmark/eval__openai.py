import argparse
import json
import os
import subprocess

from medvision_bm.benchmark.eval_utils import parse_sample_indices
from medvision_bm.utils import (
    ensure_hf_hub_installed,
    install_medvision_ds,
    install_vendored_lmms_eval,
    load_tasks,
    load_tasks_status,
    setup_env_hf_medvision_ds,
    update_task_status,
)

API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def run_evaluation_for_task_API_models(
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
        "--openai_model_code",
        required=True,
        type=str,
        help=(
            "OpenAI model code. For --api_provider openai, use an OpenAI model ID "
            "(e.g. gpt-5.5), see https://developers.openai.com/api/docs/models. "
            "For --api_provider openrouter, use an OpenRouter model ID (e.g. openai/gpt-5.5), "
            "see https://openrouter.ai/models"
        ),
    )
    parser.add_argument(
        "--api_provider",
        default="openai",
        choices=["openai", "openrouter"],
        type=str,
        help="API provider to access the OpenAI model: 'openai' (direct) or 'openrouter'.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        type=str,
        help=(
            "Reasoning effort for reasoning models (e.g. low / medium / high). "
            "If not set, the parameter is omitted entirely and the provider default applies."
        ),
    )
    parser.add_argument(
        "--max_tokens",
        default=16000,
        type=int,
        help="Default max output tokens per request. A per-task max_new_tokens from the task YAML takes precedence.",
    )
    parser.add_argument(
        "--reshape_image_hw",
        default=None,
        type=str,
        help="Reshape images to this height and width (format: H,W) before feeding into the model. Default is None.",
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
        required=True,
        type=str,
        help="Path to the tasks list JSON file.",
    )
    # data, output and status paths
    parser.add_argument(
        "--results_dir",
        required=True,
        type=str,
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--task_status_json_path",
        required=True,
        type=str,
        help="Path to the task status JSON file.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
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
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=None,
        metavar="STRING",
        help=(
            "Stop strings for generation (e.g. '</answer>'). "
            "Generation halts at the first match. "
            "Passed to the model as stop sequences for all tasks. "
            "NOTE: OpenAI reasoning models (gpt-5.x / o-series) may reject the stop "
            "parameter; leave unset for those models."
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

    # Fail fast if the provider's API key is missing
    api_key_env_var = API_KEY_ENV_VARS[args.api_provider]
    if not os.environ.get(api_key_env_var, "").strip():
        raise EnvironmentError(
            f"{api_key_env_var} is not set. Export it before running the OpenAI evaluation with --api_provider {args.api_provider}."
        )

    # Configuration
    openai_model_code = args.openai_model_code
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
        # NOTE: Install huggingface-hub, required version may vary for different models, check requirements
        ensure_hf_hub_installed(hf_hub_version="0.36.0")
        install_vendored_lmms_eval(proj_dependency="openai")
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
    # ------

    tasks = load_tasks(tasks_list_json_path)

    for task in tasks:
        completed_tasks = load_tasks_status(task_status_json_path, model_name)
        if task in completed_tasks:
            print(f"Task {task} already completed. Skipping...")
            continue

        # model configuration for OpenAI
        # NOTE: model_hf carries the raw model code; the evaluator injects it into
        # lmms_eval_specific_kwargs, and openai.openai_resized_hw() normalizes it
        # ("openai/" prefix and date snapshot suffix stripped) for cap lookup --
        # normalization lives in exactly one place (openai._normalize_model_code).
        model_args = (
            f"model={openai_model_code},"
            f"provider={args.api_provider},"
            f"model_hf={openai_model_code},"
            f"max_tokens={args.max_tokens}"
        )

        # only forwarded when set: the model file omits the reasoning parameter entirely
        # when unset (provider default applies)
        if args.reasoning_effort is not None:
            model_args += f",reasoning_effort={args.reasoning_effort}"

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

        if args.stop_strings:
            model_args += (
                f",stop_strings={json.dumps(args.stop_strings, separators=(',', ':'))}"
            )

        parsed_sample_indices = None
        if args.sample_indices is not None:
            parsed_sample_indices = parse_sample_indices(args.sample_indices)

        rc = run_evaluation_for_task_API_models(
            lmmseval_module="openai",
            model_args=model_args,
            task=task,
            batch_size=args.batch_size,
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
