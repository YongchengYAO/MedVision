ENV_NAME="eval-minimax-m3-int4"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    # Python 3.12 to match the patched vLLM fork's precompiled wheels (see vLLM fork build below).
    conda create -n "${ENV_NAME}" python==3.12 -y
fi
conda activate "${ENV_NAME}"

# (CUDA 13 forward-compatibility is handled by a verified, root-free block just
#  before Step 3 below -- see "CUDA 13 forward-compatibility shim".)

# =============================================================================
# HARDWARE -- MiniMax-M3-AWQ-INT4 on 4x H100 80GB (TP=4, no CPU offload)
# -----------------------------------------------------------------------------
# MiniMax-M3 is a 428B sparse-MoE VLM; ALL experts stay resident (the ~23B
# "activated" figure is compute-per-token, NOT memory). AWQ-INT4 weights are
# ~214 GB. This node has 4 visible GPUs, so vLLM auto-derives
# tensor_parallel_size=4 and shards the weights to ~54 GB per GPU -- well within
# each H100's 80 GB, leaving ~21 GB/GPU for KV cache at gpu_memory_utilization
# below. The 316 GB aggregate VRAM holds the whole model, so NO CPU offload is
# needed (cpu_offload_gb=0).
#   *** DO NOT raise cpu_offload_gb here. vLLM applies it PER GPU worker, so with
#   TP=4 a value of N means 4*N GiB of host RAM. This pod's cgroup is capped at
#   400 GiB (memory.max), NOT the node's ~1.9 TB; the old cpu_offload_gb=120
#   demanded 480 GiB and was silently OOM-killed (SIGKILL, no traceback) during
#   model load. Offload is also much slower (weights stream GPU<->CPU every
#   forward pass) and unnecessary now that the model fits in VRAM. ***
# vLLM auto-detects the quant from config; the minimax_m3_vl compressed-tensors
# kernels still require the patched fork (built below). The unsloth GGUF quants
# are llama.cpp-only and CANNOT be used here.
# See docs/Model-Hardware-Requirements.md.
# =============================================================================

# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
# 4-bit AWQ-INT4 checkpoint -- ~214GB weights, sharded across 4x H100 80GB via TP=4 (see HARDWARE above).
model_hf_id="cyankiwi/MiniMax-M3-AWQ-INT4"
model_name="MiniMax-M3-INT4"
batch_size_per_gpu=1
# 0.90 (not 0.95): a dead vLLM engine leaks ~4 GiB VRAM, so when the driver auto-retries after a worker
# crash, 0.95 (needs 75.23/79.19 GiB free) fails the startup memory check on the ~75.18 GiB left and the
# run wedges in a restart loop. 0.90 (needs 71.3 GiB) recovers despite the leak, and the extra ~4 GiB of
# activation headroom hedges against the rare mid-generation hard worker abort. KV cache (~7 GiB) is still
# ample for max_model_len=32768 (needs ~1.4 GiB/full-length request).
gpu_memory_utilization=0.90
# vLLM CPU offload, GiB PER GPU worker. 0 = keep all weights in VRAM. With TP=4 the 214GB model fits
# (~54GB/GPU), so no offload is needed; a nonzero value here is multiplied by 4 and OOM-kills the pod
# (400 GiB cgroup cap). See the *** warning in the HARDWARE block above before changing.
cpu_offload_gb=0

# vLLM -- PATCHED FORK (not a pip release).
# WHY a fork: mainline vLLM already runs the minimax_m3_vl architecture (it landed upstream; for now it
# ships via nightly / the official Docker image -- the "0.24.0+" the vLLM recipe cites is NOT on PyPI
# yet, newest pip is 0.23.0), but ONLY for the unquantized BF16 / MXFP8 checkpoints (that is why the
# MXFP8 script needs no fork). Mainline also has a
# generic compressed-tensors loader and AWQ kernels -- but the per-layer wiring that maps THIS 428B
# sparse-MoE arch's attention/MoE-expert Linear weights onto the AWQ-INT4 kernels (and keeps the vision
# tower / router / norms in higher precision) was never merged upstream. So mainline can load a
# MiniMax-M3, just not an AWQ-INT4 one. That model-specific glue is a Python-only patch (no CUDA
# recompilation, it rides an upstream precompiled base) living in toncao/vllm @ branch
# minimax-m3-compressed-tensors. install_vllm() (pip install vllm==X) CANNOT provide it, so this script:
#   1) runs the standard MedVision setup with --env_setup_only (torch, lmms_eval, medvision_ds,
#      transformers + a throwaway pip vLLM),
#   2) replaces vLLM with the patched fork (built below),
#   3) runs the eval with --skip_env_setup.
# If a future pip vLLM merges AWQ-INT4 minimax_m3_vl support, drop steps 1-3's fork dance and just run
# this like the MXFP8 script (single full-setup invocation with --vllm_version pointing at that release).
# block_size=128 (mandatory for MiniMax Sparse Attention) is still set by the wrapper (vllm_minimax_m3.py).
# throwaway pip vLLM for step 1 ONLY -- must be an installable PyPI version (newest is 0.23.0; the
# vLLM recipe's "0.24.0+" is NOT on PyPI yet). Step 2 uninstalls it and swaps in the fork, so whether
# this version itself supports MiniMax-M3 is irrelevant -- it just has to install so --env_setup_only
# completes (installing it at a non-existent version is what crashed the first run).
vllm_version="0.23.0"
vllm_fork_dir="${data_dir}/.vllm-minimax-m3-int4-fork"
vllm_fork_base_commit="a7fdfeef72323eb3db6f0620e4ea200290d0ca5a"

# Sampling config.
# MiniMax-M3 is a reasoning model: do NOT use greedy decoding (temperature=0). Defaults mirror the
# checkpoint's generation_config.json (temperature=1.0, top_p=0.95) plus the model card's top_k=40.
# (Generation uses a fixed internal seed from medvision_bm.utils.configs.SEED, so runs are reproducible.)
temperature=1.0
top_p=0.95
top_k=40

# Max new tokens.
# MiniMax-M3 is a verbose reasoning model: its <mm:think> chain alone can exceed the driver's 4096-token
# default, truncating most responses before <answer>/</answer> is ever emitted (the stop string then
# never fires). 16384 fits well under max_model_len=32768 (prompt + image tokens + output) and clears
# the truncation. Raise only if responses still truncate; going past ~24k would also need max_model_len up.
max_new_tokens=16384

# Stop string.
# REQUIRED for reasoning models: lmms-eval auto-injects the fewshot delimiter "\n\n" as a stop string
# for generate_until tasks, which would halt generation between CoT steps before <answer> is emitted.
# Passing an explicit --stop_strings gives a clean terminator AND signals the wrapper to drop the
# auto-injected "\n\n" stop (see vllm_minimax_m3.py).
stop_string='</answer>'

# Other configs (safe to leave as is)
task_tag="MedVision-AD-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-AD-CoT.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=1000

# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
# tree). setuptools build_py caches created dirs in a process-global memo, and on
# CephFS a build subdir can transiently vanish (async delete/recreate lag or an
# unguarded concurrent writer), after which the cache refuses to recreate it and a
# later file copy dies with: could not create '...': No such file or directory.
# A private local build dir is immune; only the shared-env install needs the lock.
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"

# Use MedVision dataset v1.0.0
export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.1.1'

# Eval args shared by the setup-only and run passes (defined once to avoid drift).
common_args=(
    --lmmseval_module vllm_minimax_m3
    --model_hf_id "$model_hf_id"
    --model_name "$model_name"
    --vllm_version "$vllm_version"
    --results_dir "$result_dir"
    --data_dir "$data_dir"
    --tasks_list_json_path "$tasks_list_json_path"
    --task_status_json_path "$task_status_json_path"
    --batch_size_per_gpu "$batch_size_per_gpu"
    --gpu_memory_utilization "$gpu_memory_utilization"
    --cpu_offload_gb "$cpu_offload_gb"
    --sample_limit "$sample_limit"
    --max_new_tokens "$max_new_tokens"
    --temperature "$temperature"
    --top_p "$top_p"
    --top_k "$top_k"
    --stop_strings "$stop_string"
)

# --- Step 1: standard env setup (installs torch, lmms_eval, medvision_ds, transformers + a throwaway
#             pip vLLM that step 2 replaces). Exits after setup. ---
python -m medvision_bm.benchmark.eval__minimax_m3 "${common_args[@]}" --env_setup_only

# --- Step 2: replace pip vLLM with the patched fork (minimax_m3_vl compressed-tensors) ---
if [ ! -d "${vllm_fork_dir}" ]; then
    git clone https://github.com/toncao/vllm.git "${vllm_fork_dir}"
    git -C "${vllm_fork_dir}" remote add upstream https://github.com/vllm-project/vllm.git
    git -C "${vllm_fork_dir}" fetch upstream "${vllm_fork_base_commit}"
    git -C "${vllm_fork_dir}" checkout minimax-m3-compressed-tensors
fi
pip uninstall -y vllm || true
# VLLM_USE_PRECOMPILED=1 downloads the precompiled CUDA bits for the pinned upstream base commit, so
# this is a Python-only (no-nvcc) install. If it fails on a torch ABI mismatch, rebuild the env via the
# fork's documented uv path instead:
#   uv venv --python 3.12 && source .venv/bin/activate
#   VLLM_USE_PRECOMPILED=1 uv pip install -e "${vllm_fork_dir}" --torch-backend=auto
VLLM_USE_PRECOMPILED=1 pip install -e "${vllm_fork_dir}"

# Resolve the torch/NCCL stack conflict that --env_setup_only leaves behind.
# install_torch_cu124 (step 1) installs a CUDA-12 torch and pulls `nvidia-nccl-cu12`; the fork's torch is
# a newer CUDA-13 build linking `nvidia-nccl-cu13`. The two NCCL wheels have different package names, so
# repeated installs leave a mix: torch then either loads the stale cu12 lib (older, missing
# ncclCommWindowDeregister -> "undefined symbol") or finds no libnccl.so.2 at all. Make it deterministic:
# remove BOTH and reinstall a single clean cu13 NCCL (the one the fork's torch links). NCCL keeps ABI
# compatibility across cu13 minors, so the latest cu13 wheel resolves the symbol for the fork's torch.
pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13 || true
pip install --no-deps --force-reinstall nvidia-nccl-cu13

# Do NOT re-pin transformers to 4.57.1 here: the fork's vLLM (recent main) requires transformers>=5.5.3
# and installs 5.12.1, and MiniMax-M3 image processing no longer needs the 4.57.x processor path (the
# perceived-size probe in medvision_utils._process_img_minimax_m3 calls the processor's own smart_resize
# directly, so it is transformers-version-agnostic). Forcing 4.57.1 here breaks the fork's vLLM import.

# --- CUDA 13 forward-compatibility shim -------------------------------------
# The fork's torch is 2.11.0+cu130 (CUDA 13.0), but this node's NVIDIA *kernel*
# driver is 570.x, which maxes out at CUDA 12.8. cu130 torch refuses to init
# against it ("The NVIDIA driver on your system is too old (found version
# 12080)") -- the crash that killed the previous run at engine startup. The
# kernel driver is host-level and CANNOT be upgraded from inside a pod.
# NVIDIA Forward Compatibility (datacenter GPUs only -- the H200 qualifies)
# fixes this by supplying a newer *userspace* libcuda that exposes CUDA-13 entry
# points and translates them down to the older kernel module. We extract the
# cuda-compat-13-0 deb (ships the 580.x userspace driver) into a local dir --
# no root/apt, and Ubuntu 20.04 has no native cuda-compat-13 so we pull the
# ubuntu2204 deb (the libcuda .so is glibc-old and distro-agnostic) -- then put
# it FIRST on LD_LIBRARY_PATH so the loader resolves the 580 libcuda by SONAME
# instead of the host's 570 one. Verified on this node: compat libcuda ->
# cuInit SUCCESS, cuDriverGetVersion 13000. No-op if the host driver is >=580.
compat_root="${data_dir}/.cuda-compat-13-0"
compat_dir="${compat_root}/extracted/usr/local/cuda-13.0/compat"
compat_deb_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.167.08-1ubuntu1_amd64.deb"
driver_major="$(sed -nE 's/.*Kernel Module +([0-9]+)\..*/\1/p' /proc/driver/nvidia/version 2>/dev/null || echo 0)"
if [ "${driver_major:-0}" -ge 580 ]; then
    echo "Host NVIDIA driver ${driver_major}.x supports CUDA 13 natively; skipping forward-compat shim."
else
    echo "Host NVIDIA driver ${driver_major}.x caps below CUDA 13; enabling cu13 forward-compat shim."
    if [ ! -e "${compat_dir}/libcuda.so.1" ]; then
        mkdir -p "${compat_root}/extracted"
        curl -fsSL --retry 3 -o "${compat_root}/cuda-compat-13-0.deb" "${compat_deb_url}"
        dpkg-deb -x "${compat_root}/cuda-compat-13-0.deb" "${compat_root}/extracted"
    fi
    export LD_LIBRARY_PATH="${compat_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Prepended forward-compat libcuda to LD_LIBRARY_PATH: ${compat_dir}"
fi

# Defragment the CUDA allocator. The 214GB AWQ-INT4 model shards to ~61 GiB/GPU at TP=4, leaving only
# ~18 GiB on each 80GB H100 for everything else. vLLM's post-load profiling pass (it profiles a max-size
# multimodal item before sizing the KV cache) briefly spikes activations and OOMs by <1 GiB while ~2 GiB
# sits "reserved but unallocated" (fragmentation). expandable_segments lets PyTorch reuse that slack
# instead of failing the allocation -- the fix recommended in the torch.OutOfMemoryError message itself.
# Inherited by the vLLM worker subprocesses (where the allocation actually happens).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Align the CUDA *compiler* toolchain so flashinfer's JIT-compiled sampling kernels build. NVIDIA's pip
# CUDA wheels split one toolkit across independently-versioned packages, and the fork install above pulls
# the newest nvcc/nvvm/crt (13.2/13.3) while torch pins the runtime headers (cudart, CUDA_VERSION) to 13.0.
# A flashinfer JIT compile touches all four: a mismatch fails at three successive layers -- CCCL's
# "compiler and toolkit headers are incompatible" #error (nvcc vs cudart), then ptxas "Unsupported
# .version" (cicc PTX ISA vs ptxas), then "'__cudaLaunch' was not declared" (cudafe++ stub vs crt header).
# cudart is the anchor (13.0, torch-pinned), so force nvcc + nvvm(cicc) + crt to the matching 13.0.88.
# --no-deps so pip touches nothing else (notably not torch). Verified: flashinfer renorm.cu then compiles.
pip install --no-deps "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88"

# Point CUDA_HOME at the CUDA toolkit so vLLM can JIT-compile the MiniMax-M3 fork's custom CUDA kernels
# at engine startup. The fork triggers a runtime nvcc compile; with CUDA_HOME unset, vLLM falls back to
# /usr/local/cuda (absent in this pod) and the worker dies with "Could not find nvcc". There is no system
# toolkit here, but the cu13 pip wheel (nvidia-cuda-nvcc-cu13) ships a complete one -- nvcc, headers,
# nvvm, libs -- inside site-packages. Derive its root from the active env and expose it. (Distinct from
# the LD_LIBRARY_PATH compat shim above, which provides only libcuda.so, not a compiler.)
cuda_home="$(python -c 'import nvidia.cu13 as c; print(list(c.__path__)[0])')"
if [ -x "${cuda_home}/bin/nvcc" ]; then
    export CUDA_HOME="${cuda_home}"
    export CUDA_PATH="${cuda_home}"
    export PATH="${cuda_home}/bin:${PATH}"
    echo "Set CUDA_HOME for nvcc JIT compile: ${cuda_home}"
else
    echo "WARNING: nvcc not found under ${cuda_home}; engine may fail to compile custom kernels."
fi

# Make -lcudart resolvable at the flashinfer JIT *link* step. After the compile toolchain is aligned,
# flashinfer links its sampling.so with `-L${cuda_home}/lib64 ... -lcudart -lcuda`, but the cu13 wheel
# puts libs in `lib` (not lib64) and ships only the versioned `libcudart.so.13` -- so `ld -lcudart` fails
# with "cannot find -lcudart". (-lcuda is fine: the host /usr/lib libcuda.so is on ld's default path.)
# Provide an unversioned libcudart.so symlink in a side dir and put it on LIBRARY_PATH (which ld searches
# for -l libs at link time). Avoids mutating site-packages, which pip could clobber.
cuda_link_dir="${data_dir}/.cuda-link"
if [ -e "${cuda_home}/lib/libcudart.so.13" ]; then
    mkdir -p "${cuda_link_dir}"
    ln -sf "${cuda_home}/lib/libcudart.so.13" "${cuda_link_dir}/libcudart.so"
    export LIBRARY_PATH="${cuda_link_dir}:${cuda_home}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    echo "Added libcudart.so to LIBRARY_PATH for flashinfer link: ${cuda_link_dir}"
fi

# --- Step 3: run the eval against the now-patched vLLM ---
python -m medvision_bm.benchmark.eval__minimax_m3 "${common_args[@]}" --skip_env_setup

conda deactivate
# conda remove -n $ENV_NAME --all -y
