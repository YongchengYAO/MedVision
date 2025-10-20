import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Python packages from a requirements file."
    )
    parser.add_argument(
        "-r",
        "--requirement",
        required=True,
        help="Path to the requirements.txt file.",
    )
    return parser.parse_args()


def run_pip_install(requirements_path: Path) -> int:
    if not requirements_path.exists() or not requirements_path.is_file():
        print(f"Error: Requirements file not found: {requirements_path}", file=sys.stderr)
        return 2

    # Use the current interpreter to run pip to avoid PATH/env mismatches.
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ]

    # Optionally disable pip's version check to reduce noise and speed up.
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    print(f"Installing packages from: {requirements_path}")
    try:
        proc = subprocess.run(cmd, env=env)
        return proc.returncode
    except KeyboardInterrupt:
        print("Installation interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error running pip: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    args = parse_args()
    req_path = Path(args.requirement).expanduser().resolve()
    exit_code = run_pip_install(req_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
