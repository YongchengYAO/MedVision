import argparse
import os
from pathlib import Path

from medvision_bm.utils import (
    install_cuda_toolkit,
    install_medvision_ds,
    install_vendored_lmms_eval,
    install_vllm,
    run_pip_install,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Python packages from a requirements file."
    )
    parser.add_argument(
        "-r",
        "--requirement",
        type=str,
        required=True,
        help="Path to the requirements.txt file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory to store downloaded datasets and source code.",
    )
    parser.add_argument(
        "--lmms_eval_opt_deps",
        type=str,
        help="Optional dependencies for lmms_eval installation.",
    )
    parser.add_argument(
        "--cuda_version",
        type=str,
        default="12.4",
        help="CUDA toolkit version to install (default: 12.4).",
    )
    parser.add_argument(
        "--vllm_version",
        type=str,
        default="0.10.0",
        help="vLLM version to install (default: 0.10.0).",
    )
    parser.add_argument(
        "--skip_cuda_toolkit",
        action="store_true",
        help="Skip installing the CUDA toolkit (for API-only models with no local GPU inference).",
    )
    parser.add_argument(
        "--skip_vllm",
        action="store_true",
        help="Skip installing vLLM (for models that don't use it).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting environment setup...")

    # Install the vendored lmms_eval package
    print("\n[Info] Installing vendored lmms_eval package...")
    opt_deps = args.lmms_eval_opt_deps
    if opt_deps is not None:
        install_vendored_lmms_eval(proj_dependency=opt_deps)
    else:
        install_vendored_lmms_eval()

    # Install dataset codebase: medvision_ds
    print("\n[Info] Installing medvision_ds package...")
    os.makedirs(args.data_dir, exist_ok=True)
    install_medvision_ds(args.data_dir)

    # Install CUDA
    if not args.skip_cuda_toolkit:
        install_cuda_toolkit(version=args.cuda_version)

    # Install vLLM
    # Some model evaluation may not need vLLM, but we install it here for completeness
    if not args.skip_vllm:
        install_vllm(data_dir=args.data_dir, version=args.vllm_version)

    # Install packages from the specified requirements file
    print(f"\n[Info] Installing packages from: {args.requirement}")
    req_path = Path(args.requirement).expanduser().resolve()
    run_pip_install(req_path)


if __name__ == "__main__":
    main()
