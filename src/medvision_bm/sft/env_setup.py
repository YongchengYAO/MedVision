import argparse
import subprocess
import sys

from medvision_bm.utils import (
    install_flash_attention_torch_and_deps_py311_v2, install_medvision_ds)


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
            "huggingface_hub",
            "accelerate",
        ],
        check=True,
    )


def install_transformers(version="4.54.0"):
    subprocess.run(
        f"pip install transformers=={version}", shell=True, check=True)


def install_sft_dependencies(data_dir=None):
    """Install all dependencies in the correct order."""
    install_basic_packages()
    install_flash_attention_torch_and_deps_py311_v2()
    install_transformers(version="4.54.0")
    install_medvision_ds(data_dir)


def parser_args():
    parser = argparse.ArgumentParser(
        description="Install all dependencies for SFT on MedVision datasets.")
    parser.add_argument("--data_dir", type=str,
                        default=None, help="Data directory path")
    args = parser.parse_args()

    assert args.data_dir is not None, "--data_dir is required"
    return args


if __name__ == "__main__":
    args = parser_args()

    install_sft_dependencies(data_dir=args.data_dir)
