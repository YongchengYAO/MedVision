import argparse

from medvision_bm.utils import install_vendored_lmms_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the vendored lmms_eval package."
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

    # Install the vendored lmms_eval package
    print("\n[Info] Installing vendored lmms_eval package...")
    opt_deps = args.lmms_eval_opt_deps
    if opt_deps is not None:
        install_vendored_lmms_eval(proj_dependency=opt_deps)
    else:
        install_vendored_lmms_eval()


if __name__ == "__main__":
    main()
