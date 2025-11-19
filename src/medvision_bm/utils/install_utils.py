import os
import shlex
import subprocess
import sys
from pathlib import Path
from importlib.resources import files  # Python 3.9+

from huggingface_hub import snapshot_download


def run_pip_install(requirements_path):
    # Normalize input to a Path so both str and Path work.
    req_path = Path(requirements_path).expanduser().resolve(strict=False)

    if not req_path.exists() or not req_path.is_file():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_path}")

    # Use the current interpreter to run pip to avoid PATH/env mismatches.
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        "-r",
        str(req_path),
    ]

    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    print(f"Installing packages from: {req_path}")
    subprocess.run(cmd, env=env, check=True)


def ensure_hf_hub_installed():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
    except ImportError:
        subprocess.run(
            "pip install huggingface_hub[cli]", check=True, shell=True)


def _install_lmms_eval(
    lmms_eval_dir,
    editable_install=False,
    proj_dependency=None,
):
    # compose extras text like .[extra]
    extras_txt = f"[{proj_dependency}]" if proj_dependency else ""

    tmp_build_lock_file = os.path.join(lmms_eval_dir, ".build.lock")
    build_dir = os.path.join(lmms_eval_dir, "build")
    dist_dir = os.path.join(lmms_eval_dir, "dist")
    egg_info_dir = os.path.join(lmms_eval_dir, "lmms_eval.egg-info")
    wheel_dir = os.path.join(lmms_eval_dir, "wheels")
    os.makedirs(wheel_dir, exist_ok=True)

    # Common pip flags
    base_pip_flags = "--no-cache-dir --force-reinstall"

    # Case A: Editable install (always from source; extras allowed)
    if editable_install:
        # Example: pip install -e .[qwen2_5_vl]
        cmd = (
            f"flock -w 600 {shlex.quote(tmp_build_lock_file)} bash -lc '"
            f"python -m pip install {base_pip_flags} -e .{extras_txt}"
            f"'"
        )
        subprocess.run(cmd, check=True, shell=True, cwd=lmms_eval_dir)
        return

    # Case B: Non-editable, NO extras → build wheel once, then install wheel
    if proj_dependency is None:
        cmd = (
            f"flock -w 600 {shlex.quote(tmp_build_lock_file)} bash -lc '"
            f"rm -rf {shlex.quote(build_dir)} {shlex.quote(dist_dir)} {shlex.quote(egg_info_dir)} && "
            f"python -m pip install --upgrade build && "
            f"python -m build --wheel --outdir {shlex.quote(wheel_dir)} {shlex.quote(lmms_eval_dir)} && "
            f"latest_wheel=$(ls -t {shlex.quote(wheel_dir)}/lmms_eval-*.whl | head -n1) && "
            f'pip install {base_pip_flags} "$latest_wheel"'
            f"'"
        )
        subprocess.run(cmd, check=True, shell=True, cwd=lmms_eval_dir)
        return

    # Case C: Non-editable WITH extras → install from source with extras
    # (extras on a wheel path is not supported)
    cmd = (
        f"flock -w 600 {shlex.quote(tmp_build_lock_file)} bash -lc '"
        f"python -m pip install {base_pip_flags} .{extras_txt}"
        f"'"
    )
    subprocess.run(cmd, check=True, shell=True, cwd=lmms_eval_dir)


def install_lmms_eval(
    benchmark_dir,
    lmms_eval_folder,
    editable_install=False,
    proj_dependency=None,
):
    lmms_eval_dir = os.path.join(benchmark_dir, lmms_eval_folder)
    _install_lmms_eval(
        lmms_eval_dir=lmms_eval_dir,
        editable_install=editable_install,
        proj_dependency=proj_dependency,
    )


def install_vendored_lmms_eval(
    editable_install=True,
    proj_dependency=None,
):
    """
    Install the vendored lmms-eval package that ships inside medvision_bm.
    """
    # Locate the vendored lmms-eval package, check [tool.setuptools.package-data] in pyproject.toml
    lmms_eval_dir = str(files("medvision_bm").joinpath("medvision_lmms-eval"))
    # NOTE: Must install the vendored lmms-eval in editable mode, otherwise tasks files won't be found.
    # TODO: Check: Why editable install causes issues in some cases?
    _install_lmms_eval(
        lmms_eval_dir=lmms_eval_dir,
        editable_install=editable_install,
        proj_dependency=proj_dependency,
    )


def setup_env_hf(data_dir):
    # Safeguard data_dir: you can use relative path with this function
    data_dir = os.path.abspath(data_dir) 

    # Set Hugging Face dataset and cache directories
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        data_dir, ".cache", "huggingface", "datasets"
    )
    os.environ["HF_HOME"] = os.path.join(data_dir, ".cache", "huggingface")


def setup_env_medvision_ds(
    data_dir,
    force_install_code=True,
    force_download_data=False,
):
    # Safeguard data_dir: you can use relative path with this function
    data_dir = os.path.abspath(data_dir)

    # Set dataset directory
    os.makedirs(data_dir, exist_ok=True)
    os.environ["MedVision_DATA_DIR"] = data_dir

    # Force install dataset codebase, default to "False"
    if force_install_code:
        os.environ["MedVision_FORCE_INSTALL_CODE"] = "true"

    # Force download dataset, default to "False"
    if force_download_data:
        os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "true"


def setup_env_hf_medvision_ds(data_dir,
                              force_install_code=True,
                              force_download_data=False,
                              ):
    # Set environment variables for medvision_ds
    setup_env_medvision_ds(
        data_dir=data_dir,
        force_install_code=force_install_code,
        force_download_data=force_download_data,
    )

    # Set environment variables for Hugging Face
    setup_env_hf(data_dir)


def install_medvision_ds(
    data_dir,
    local_dir=None,
):
    if local_dir is None:
        # Safeguard data_dir: you can use relative path with this function
        data_dir = os.path.abspath(data_dir)
        
        os.makedirs(data_dir, exist_ok=True)
        snapshot_download(
            repo_id="YongchengYAO/MedVision",
            allow_patterns="src/*",
            repo_type="dataset",
            local_dir=data_dir,
        )
        dir_bmvqa = os.path.abspath(os.path.join(data_dir, "src"))
    else:
        dir_bmvqa = os.path.abspath(os.path.join(local_dir, "src"))

    tmp_build_lock_file = os.path.join(dir_bmvqa, ".build.lock")
    build_dir = os.path.join(dir_bmvqa, "build")
    dist_dir = os.path.join(dir_bmvqa, "dist")
    egg_info_dir = os.path.join(dir_bmvqa, "medvision_ds.egg-info")
    wheel_dir = os.path.join(dir_bmvqa, "wheels")
    os.makedirs(wheel_dir, exist_ok=True)
    cmd_w_flock = (
        f"flock -w 600 {shlex.quote(tmp_build_lock_file)} bash -lc '"
        f"rm -rf {shlex.quote(build_dir)} {shlex.quote(dist_dir)} {shlex.quote(egg_info_dir)} && "
        f"python -m pip install --upgrade build && "
        f"python -m build --wheel --outdir {shlex.quote(wheel_dir)} {shlex.quote(dir_bmvqa)} && "
        f"latest_wheel=$(ls -t {shlex.quote(wheel_dir)}/medvision_ds-*.whl | head -n1) && "
        f'pip install --no-cache-dir --force-reinstall "$latest_wheel"\''
    )
    
    # Try with flock, fallback to without flock if it fails
    try:
        subprocess.run(cmd_w_flock, check=True, shell=True)
    except subprocess.CalledProcessError:
        print("Warning: flock failed, attempting installation without file lock...")
        cmd_no_flock = (
            f"bash -lc '"
            f"rm -rf {shlex.quote(build_dir)} {shlex.quote(dist_dir)} {shlex.quote(egg_info_dir)} && "
            f"python -m pip install --upgrade build && "
            f"python -m build --wheel --outdir {shlex.quote(wheel_dir)} {shlex.quote(dir_bmvqa)} && "
            f"latest_wheel=$(ls -t {shlex.quote(wheel_dir)}/medvision_ds-*.whl | head -n1) && "
            f'pip install --no-cache-dir --force-reinstall "$latest_wheel"\''
        )
        subprocess.run(cmd_no_flock, check=True, shell=True)

    # Set environment variables for medvision_ds
    setup_env_hf_medvision_ds(data_dir=data_dir)


def pip_install_medvision_ds():
    try:
        print(f'\n[Info] Installing medvision_ds from Hugging Face Datasets repo: pip install "git+https://huggingface.co/datasets/YongchengYAO/MedVision.git#subdirectory=src"')
        subprocess.run(
            'pip install "git+https://huggingface.co/datasets/YongchengYAO/MedVision.git#subdirectory=src"',
            check=True,
            shell=True,
        )
        print("Successfully installed medvision_ds.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing medvision_ds: {e}", file=sys.stderr)


def pip_install_medvision_bm():
    try:
        print(f'\n[Info] Installing medvision_bm from GitHub repo: pip install "git+https://github.com/YongchengYAO/MedVision.git"')
        subprocess.run(
            'pip install "git+https://github.com/YongchengYAO/MedVision.git"',
            check=True,
            shell=True,
        )
        print("Successfully installed medvision_bm.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing medvision_bm: {e}", file=sys.stderr)


def setup_env_cuda():
    print("Setting up CUDA environment...")
    cuda_home = os.environ.get("CONDA_PREFIX", "")
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cuda_home}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )


def install_cuda_toolkit(version="12.4"):
    """Install CUDA toolkit using conda."""
    print("Installing CUDA toolkit...")
    subprocess.run(
        ["conda", "install", "-c", "nvidia", f"cuda-toolkit={version}", "-y"],
        check=True,
    )
    setup_env_cuda()


def install_torch_cu121():
    """Install PyTorch with CUDA support."""
    print("Installing PyTorch...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.5.0+cu121",
            "torchvision==0.20.0+cu121",
            "torchaudio==2.5.0+cu121",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "--force-reinstall",
        ],
        check=True,
    )
    setup_env_cuda()


def install_torch_cu124():
    """Install PyTorch with CUDA support."""
    print("Installing PyTorch...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.6.0+cu124",
            "torchvision==0.21.0+cu124",
            "torchaudio==2.6.0+cu124",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "--force-reinstall",
        ],
        check=True,
    )
    setup_env_cuda()


def install_flash_attention_torch_and_deps_py39():
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA 12.4...")
    subprocess.run(
        "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 "
        "--index-url https://download.pytorch.org/whl/cu124 --force-reinstall",
        check=True,
        shell=True,
    )

    # Install CUDA
    print("Installing CUDA toolkit and components...")
    subprocess.run(
        "conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y",
        check=True,
        shell=True,
    )
    subprocess.run(
        "conda install nvidia/label/cuda-12.4.0::cuda-nvcc -y", check=True, shell=True
    )
    subprocess.run(
        "conda install cudnn -y",
        check=True,
        shell=True,
    )
    subprocess.run(
        "pip install --upgrade nvidia-cuda-cupti-cu12==12.4.* "
        "nvidia-cuda-nvrtc-cu12==12.4.* "
        "nvidia-cuda-runtime-cu12==12.4.*",
        check=True,
        shell=True,
    )
    setup_env_cuda()

    # Install Flash Attention
    print("Installing Flash Attention...")
    subprocess.run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl",
        check=True,
        shell=True,
    )

    # Install numpy version 1.26.4
    print("Installing numpy...")
    subprocess.run("pip install numpy==1.26.4", check=True, shell=True)

    # Install protobuf version 3.20
    print("Installing protobuf 3.20.x")
    subprocess.run("pip install protobuf==3.20", check=True, shell=True)


def install_flash_attention_torch_and_deps_py39_v2():
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA 12.4...")
    install_torch_cu124()

    # Install Flash Attention
    print("Installing Flash Attention...")
    subprocess.run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl",
        check=True,
        shell=True,
    )

    # Install numpy version 1.26.4
    print("Installing numpy...")
    subprocess.run("pip install numpy==1.26.4", check=True, shell=True)

    # Install protobuf version 3.20
    print("Installing protobuf 3.20.x")
    subprocess.run("pip install protobuf==3.20", check=True, shell=True)


def install_flash_attention_torch_and_deps_py310():
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA 12.4...")
    subprocess.run(
        "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 "
        "--index-url https://download.pytorch.org/whl/cu124 --force-reinstall",
        check=True,
        shell=True,
    )

    # Install CUDA
    print("Installing CUDA toolkit and components...")
    subprocess.run(
        "conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y",
        check=True,
        shell=True,
    )
    subprocess.run(
        "conda install nvidia/label/cuda-12.4.0::cuda-nvcc -y", check=True, shell=True
    )
    subprocess.run(
        "conda install cudnn -y",
        check=True,
        shell=True,
    )
    subprocess.run(
        "pip install --upgrade nvidia-cuda-cupti-cu12==12.4.* "
        "nvidia-cuda-nvrtc-cu12==12.4.* "
        "nvidia-cuda-runtime-cu12==12.4.*",
        check=True,
        shell=True,
    )
    setup_env_cuda()

    # Install Flash Attention
    print("Installing Flash Attention...")
    subprocess.run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        check=True,
        shell=True,
    )

    # Install numpy version 1.26.4
    print("Installing numpy...")
    subprocess.run("pip install numpy==1.26.4", check=True, shell=True)

    # Install protobuf version 3.20
    print("Installing protobuf 3.20.x")
    subprocess.run("pip install protobuf==3.20", check=True, shell=True)


def install_flash_attention_torch_and_deps_py310_v2():
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA 12.4...")
    install_torch_cu124()

    # Install Flash Attention
    print("Installing Flash Attention...")
    subprocess.run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
        check=True,
        shell=True,
    )

    # Install numpy version 1.26.4
    print("Installing numpy...")
    subprocess.run("pip install numpy==1.26.4", check=True, shell=True)

    # Install protobuf version 3.20
    print("Installing protobuf 3.20.x")
    subprocess.run("pip install protobuf==3.20", check=True, shell=True)


def install_flash_attention_torch_and_deps_py311():
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA 12.4...")
    subprocess.run(
        "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 "
        "--index-url https://download.pytorch.org/whl/cu124 --force-reinstall",
        check=True,
        shell=True,
    )

    # Install CUDA
    print("Installing CUDA toolkit and components...")
    subprocess.run(
        "conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y",
        check=True,
        shell=True,
    )
    subprocess.run(
        "conda install nvidia/label/cuda-12.4.0::cuda-nvcc -y", check=True, shell=True
    )
    subprocess.run(
        "conda install cudnn -y",
        check=True,
        shell=True,
    )
    subprocess.run(
        "pip install --upgrade nvidia-cuda-cupti-cu12==12.4.* "
        "nvidia-cuda-nvrtc-cu12==12.4.* "
        "nvidia-cuda-runtime-cu12==12.4.*",
        check=True,
        shell=True,
    )
    setup_env_cuda()

    # Install Flash Attention
    print("Installing Flash Attention...")
    subprocess.run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        check=True,
        shell=True,
    )

    # Install numpy version 1.26.4
    print("Installing numpy...")
    subprocess.run("pip install numpy==1.26.4", check=True, shell=True)

    # Install protobuf version 3.20
    print("Installing protobuf 3.20.x")
    subprocess.run("pip install protobuf==3.20", check=True, shell=True)


def install_flash_attention_torch_and_deps_py311_v2():
    # Install PyTorch with CUDA support
    print("Installing PyTorch with CUDA 12.4...")
    install_torch_cu124()

    # Install Flash Attention
    print("Installing Flash Attention...")
    subprocess.run(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        check=True,
        shell=True,
    )

    # Install numpy version 1.26.4
    print("Installing numpy...")
    subprocess.run("pip install numpy==1.26.4", check=True, shell=True)

    # Install protobuf version 3.20
    print("Installing protobuf 3.20.x")
    subprocess.run("pip install protobuf==3.20", check=True, shell=True)


def setup_env_vllm(data_dir):
    # Safeguard data_dir: you can use relative path with this function
    data_dir = os.path.abspath(data_dir)

    # Ensure proper process spawning
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Set the cache directory for vllm
    os.environ["XDG_CACHE_HOME"] = os.path.join(data_dir, ".cache", "vllm")


def install_vllm(data_dir, version="0.10.0"):
    # Install and setup vllm
    try:
        subprocess.run("pip install blobfile", check=True, shell=True)
        subprocess.run(
            f"pip install vllm=={version}",
            check=True,
            shell=True,
        )
        print("Successfully installed vllm")

    except Exception as e:
        raise RuntimeError(f"Error installing vllm: {e}")
    setup_env_vllm(data_dir)
