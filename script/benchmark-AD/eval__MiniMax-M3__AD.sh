ENV_NAME="eval-minimax-m3"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    # Python 3.12 to match the fully tested MiniMax-M3-INT4 env (same torch 2.11/cu13 stack).
    conda create -n "${ENV_NAME}" python==3.12 -y
fi
conda activate "${ENV_NAME}"

# (CUDA 13 forward-compatibility is handled by a verified, root-free block just
#  before Step 3 below -- see "CUDA 13 forward-compatibility shim".)

# =============================================================================
# HARDWARE -- MiniMax-M3-MXFP8 needs >= 8x 80GB GPUs (TP=8). 4x 80GB CANNOT run it.
# -----------------------------------------------------------------------------
# MiniMax-M3 is a 428B sparse-MoE VLM; ALL experts stay resident (the ~23B
# "activated" figure is compute-per-token, NOT memory). Weights per precision:
#     BF16   ~856 GB  -> does not fit any node here (needs ~8x H200)
#     MXFP8  ~428 GB  -> ~54 GB/GPU at TP=8 -> FITS 8x 80GB (4x H200 141GB: tight)
#     INT4   ~214 GB  -> ~54 GB/GPU at TP=4 -> the 4x 80GB option (separate script)
# This script targets the OFFICIAL MXFP8 checkpoint (MiniMaxAI/MiniMax-M3-MXFP8,
# NVIDIA-quantized from the FP16 weights; vLLM auto-detects the quant from its
# config). On a 4x 80GB node (320 GB aggregate) the ~428 GB of weights CANNOT
# fit and the engine OOMs during load -- on those nodes run the fully tested
# eval__MiniMax-M3-INT4__AD.sh (AWQ-INT4) instead. Do NOT try to close the gap
# with cpu_offload_gb: vLLM applies it PER GPU worker (so TP=4 means 4x the
# value in host RAM) and this pod's cgroup is capped at 400 GiB -- the loader
# gets SIGKILLed with no traceback (see the warning in the INT4 script). The
# guard below fails fast instead of wedging a node.
# The eval driver derives tensor_parallel_size from the visible GPU count, so
# CUDA_VISIBLE_DEVICES is deliberately unset below: expose the whole node and
# TP spans all of its GPUs (8x 80GB -> TP=8). NOTE: the unsloth GGUF quants are
# llama.cpp-only and CANNOT be used here.
# See docs/Model-Hardware-Requirements.md.
# =============================================================================

# The driver sets tensor_parallel_size from CUDA_VISIBLE_DEVICES when present; a stale 4-GPU
# value inherited from another workflow's shell/pod spec would silently force TP=4 (~107 GB/GPU
# -> OOM) even on an 8-GPU node that passes the VRAM check below, since nvidia-smi ignores the
# variable. This script wants the whole node, so drop any inherited restriction.
unset CUDA_VISIBLE_DEVICES

# Fail fast if this node cannot hold the MXFP8 weights (~428 GB + KV/overhead).
# Pass: 8x 80GB (~637 GiB), 4x H200 141GB (~562 GiB). Fail: 4x 80GB (~319 GiB).
total_vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print int(s)}')"
num_gpus="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
required_vram_mib=$((470 * 1024))
if [ "${total_vram_mib:-0}" -lt "${required_vram_mib}" ]; then
    echo "[Error] MiniMax-M3-MXFP8 needs >= 470 GiB aggregate VRAM (e.g. 8x 80GB GPUs);" >&2
    echo "        this node has ${num_gpus} GPU(s) totaling $((${total_vram_mib:-0} / 1024)) GiB." >&2
    echo "        On 4x 80GB nodes use eval__MiniMax-M3-INT4__AD.sh instead." >&2
    exit 1
fi
echo "VRAM check passed: ${num_gpus} GPU(s), $((total_vram_mib / 1024)) GiB aggregate."

# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
# Official MXFP8 checkpoint -- ~428GB weights, sharded to ~54GB/GPU across 8x 80GB via TP=8 (see HARDWARE above).
model_hf_id="MiniMaxAI/MiniMax-M3-MXFP8"
model_name="MiniMax-M3"
batch_size_per_gpu=1
# 0.90 (not 0.95): a dead vLLM engine leaks ~4 GiB VRAM, so when the driver auto-retries after a worker
# crash, 0.95 fails the startup memory check on what is left and the run wedges in a restart loop. 0.90
# recovers despite the leak, and the extra headroom hedges against the rare mid-generation hard worker
# abort. Verified on the INT4 run; KV cache is still ample for max_model_len=32768.
gpu_memory_utilization=0.90
# vLLM CPU offload, GiB PER GPU worker. 0 = keep all weights in VRAM. At TP=8 the 428GB model fits
# (~54GB/GPU), so no offload is needed; a nonzero value here is multiplied by the worker count and
# OOM-kills the pod (400 GiB cgroup cap). See the warning in the HARDWARE block before changing.
cpu_offload_gb=0

# vLLM version. MiniMax-M3 VL ships no HF modeling file, so vLLM must NATIVELY register the
# minimax_m3_vl architecture (MiniMaxM3SparseForConditionalGeneration). Native support for the
# unquantized BF16 / MXFP8 checkpoints landed in vLLM 0.24.0, which is now ON PyPI (the INT4
# script's "0.24.0 is not on PyPI yet" note predates the release). No patched fork is needed for
# MXFP8 -- the toncao/vllm fork exists solely for the AWQ-INT4 per-layer wiring. block_size=128
# (mandatory for MiniMax Sparse Attention) is set by the wrapper (vllm_minimax_m3.py).
vllm_version="0.24.0"

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

# WHY the setup/run split (mirrors the tested INT4 script, minus its fork dance): the driver's env
# setup re-pins transformers to 4.57.1 AFTER installing vLLM (install_transformers_for_minimax_m3),
# but vLLM 0.24.0 requires transformers>=5.5.3 and fails to import on 4.57.1. Setup also leaves a
# mixed cu12/cu13 NCCL stack behind (see Step 2). Both must be repaired between setup and run, so:
#   1) --env_setup_only  2) repair the env  3) --skip_env_setup.

# --- Step 1: standard env setup (installs torch, lmms_eval, medvision_ds, vLLM 0.24.0,
#             then re-pins transformers to 4.57.1 -- repaired in Step 2). Exits after setup. ---
python -m medvision_bm.benchmark.eval__minimax_m3 "${common_args[@]}" --env_setup_only

# --- Step 2: repair the env that setup leaves behind ---

# Restore transformers for vLLM 0.24.0 (requires >=5.5.3; setup re-pinned 4.57.1). MiniMax-M3 image
# processing no longer needs the 4.57.x processor path (the perceived-size probe in
# medvision_utils._process_img_minimax_m3 calls the processor's own smart_resize directly, so it is
# transformers-version-agnostic). 5.12.1 is the version verified against the same-vintage vLLM in
# the tested INT4 run.
pip install "transformers==5.12.1"

# Resolve the torch/NCCL stack conflict that setup leaves behind.
# install_torch_cu124 installs a CUDA-12 torch and pulls `nvidia-nccl-cu12`; vLLM 0.24.0 then upgrades
# torch to 2.11.0 -- a CUDA-13 build linking `nvidia-nccl-cu13`. The two NCCL wheels have different
# package names but ship the same libnccl.so.2 path, so repeated installs leave a mix: torch then either
# loads the stale cu12 lib (older, missing ncclCommWindowDeregister -> "undefined symbol") or finds no
# libnccl.so.2 at all. Make it deterministic: remove BOTH and reinstall a single clean cu13 NCCL (the
# one torch 2.11.0 links). NCCL keeps ABI compatibility across cu13 minors, so the latest cu13 wheel
# resolves the symbol. Verified on the INT4 run (same torch).
pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13 || true
pip install --no-deps --force-reinstall nvidia-nccl-cu13

# --- CUDA 13 forward-compatibility shim -------------------------------------
# vLLM 0.24.0's torch is 2.11.0+cu130 (CUDA 13.0), but this cluster's NVIDIA
# *kernel* driver is 570.x, which maxes out at CUDA 12.8. cu130 torch refuses
# to init against it ("The NVIDIA driver on your system is too old (found
# version 12080)"). The kernel driver is host-level and CANNOT be upgraded from
# inside a pod. NVIDIA Forward Compatibility (datacenter GPUs only) fixes this
# by supplying a newer *userspace* libcuda that exposes CUDA-13 entry points
# and translates them down to the older kernel module. We extract the
# cuda-compat-13-0 deb (ships the 580.x userspace driver) into a local dir --
# no root/apt, and Ubuntu 20.04 has no native cuda-compat-13 so we pull the
# ubuntu2204 deb (the libcuda .so is glibc-old and distro-agnostic) -- then put
# it FIRST on LD_LIBRARY_PATH so the loader resolves the 580 libcuda by SONAME
# instead of the host's 570 one. Verified on the INT4 run: compat libcuda ->
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

# Defragment the CUDA allocator. vLLM's post-load profiling pass (it profiles a max-size multimodal
# item before sizing the KV cache) briefly spikes activations and can OOM by <1 GiB while ~2 GiB sits
# "reserved but unallocated" (fragmentation) -- observed on the tested INT4 run. expandable_segments
# lets PyTorch reuse that slack instead of failing the allocation. Inherited by the vLLM worker
# subprocesses (where the allocation actually happens).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Align the CUDA *compiler* toolchain so flashinfer's JIT-compiled sampling kernels build (vLLM 0.24.0
# hard-depends on flashinfer-python). NVIDIA's pip CUDA wheels split one toolkit across independently-
# versioned packages, and the vLLM install above pulls the newest nvcc/nvvm/crt while torch pins the
# runtime headers (cudart, CUDA_VERSION) to 13.0. A flashinfer JIT compile touches all four: a mismatch
# fails at three successive layers -- CCCL's "compiler and toolkit headers are incompatible" #error
# (nvcc vs cudart), then ptxas "Unsupported .version" (cicc PTX ISA vs ptxas), then "'__cudaLaunch' was
# not declared" (cudafe++ stub vs crt header). cudart is the anchor (13.0, torch-pinned), so force
# nvcc + nvvm(cicc) + crt to the matching 13.0.88. --no-deps so pip touches nothing else (notably not
# torch). Verified on the INT4 run: flashinfer renorm.cu then compiles.
pip install --no-deps "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88"

# Point CUDA_HOME at the CUDA toolkit so vLLM can JIT-compile MiniMax-M3's custom CUDA kernels at
# engine startup. With CUDA_HOME unset, vLLM falls back to /usr/local/cuda (absent in this pod) and the
# worker dies with "Could not find nvcc". There is no system toolkit here, but the cu13 pip wheels
# installed above ship a complete one -- nvcc, headers, nvvm, libs -- inside site-packages. Derive its
# root from the active env and expose it. (Distinct from the LD_LIBRARY_PATH compat shim above, which
# provides only libcuda.so, not a compiler.)
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

# --- Step 3: run the eval against the repaired env ---
python -m medvision_bm.benchmark.eval__minimax_m3 "${common_args[@]}" --skip_env_setup

conda deactivate
# conda remove -n $ENV_NAME --all -y
