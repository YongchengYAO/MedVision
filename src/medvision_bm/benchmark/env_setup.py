import argparse
import os
import subprocess
import sys
from pathlib import Path

from medvision_bm.utils import install_medvision_ds, install_vendored_lmms_eval, run_pip_install


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Starting environment setup...")

    # Install packages from the specified requirements file
    print(f"\n[Info] Installing packages from: {args.requirement}")
    req_path = Path(args.requirement).expanduser().resolve()
    run_pip_install(req_path)

    # Install the vendored lmms_eval package
    print(f"\n[Info] Installing vendored lmms_eval package...")
    opt_deps = args.lmms_eval_opt_deps
    if opt_deps is not None:
        install_vendored_lmms_eval(proj_dependency=opt_deps)
    else:
        install_vendored_lmms_eval()

    # Install dataset codebase: medvision_ds
    print(f"\n[Info] Installing medvision_ds package...")
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    install_medvision_ds(data_dir)


if __name__ == "__main__":
    main()
