"""Console entry point for the MedVision benchmark.

Provides short aliases for the longer ``python -m medvision_bm.benchmark.*``
commands, e.g. ``mvbm install mvds -d <data_dir>``.
"""

import argparse
import os


def _install_mvds(args: argparse.Namespace) -> None:
    # Imported lazily: medvision_bm.utils pulls in torch and datasets, which
    # should not be required just to print `mvbm --help`.
    from medvision_bm.utils import install_medvision_ds

    print("\n[Info] Installing medvision_ds package...")
    os.makedirs(args.data_dir, exist_ok=True)
    install_medvision_ds(args.data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mvbm", description="Shortcut commands for the MedVision benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_install = subparsers.add_parser(
        "install", help="Install MedVision components."
    )
    subparsers_install = parser_install.add_subparsers(dest="target", required=True)

    parser_mvds = subparsers_install.add_parser(
        "mvds",
        help="Install the medvision_ds dataset codebase "
        "(alias of `python -m medvision_bm.benchmark.install_medvision_ds`).",
    )
    parser_mvds.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory to store downloaded datasets and source code.",
    )
    parser_mvds.set_defaults(func=_install_mvds)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
