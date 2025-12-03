import argparse
import subprocess
import sys

from medvision_bm.utils import (
    install_flash_attention_torch_and_deps_py311_v2,
    install_medvision_ds,
    run_pip_install,
)


def install_basic_packages():
    """Install basic required packages."""
    print("Installing basic packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "datasets==3.6.0",
            "numpy==1.26.4",
            "protobuf==3.20",
            "wandb==0.21.4",
            "trl==0.19.1",
            "huggingface_hub==0.36.0",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            "bitsandbytes",
            "peft",
            "hf_xet",
            "tensorboard",
            "nibabel",
            "scipy",
            "Pillow",
            "accelerate",
        ],
        check=True,
    )


def install_transformers(version="4.54.0"):
    subprocess.run(
        f"pip install transformers=={version}", shell=True, check=True)


def install_sft_dependencies(data_dir=None, requirement=None):
    """Install all dependencies in the correct order."""
    if requirement is not None:
        print(f"Installing packages from requirements file: {requirement}")
        run_pip_install(requirements_path=requirement)
        install_medvision_ds(data_dir)
    else:
        print("Installing packages individually...")
        install_basic_packages()
        install_flash_attention_torch_and_deps_py311_v2()
        install_transformers(version="4.54.0")
        install_medvision_ds(data_dir)


def parser_args():
    parser = argparse.ArgumentParser(
        description="Install all dependencies for SFT on MedVision datasets.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory path"
    )
    parser.add_argument(
        "-r",
        "--requirement",
        type=str,
        help="Path to the requirements.txt file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()

    install_sft_dependencies(data_dir=args.data_dir,
                             requirement=args.requirement)
