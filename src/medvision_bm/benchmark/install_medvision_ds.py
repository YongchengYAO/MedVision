import argparse
import os

from medvision_bm.utils import install_medvision_ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Python packages from a requirements file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory to store downloaded datasets and source code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Install dataset codebase: medvision_ds
    print(f"\n[Info] Installing medvision_ds package...")
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    install_medvision_ds(data_dir)


if __name__ == "__main__":
    main()
