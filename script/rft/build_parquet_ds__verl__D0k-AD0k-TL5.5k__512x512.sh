# Build verl parquet dataset for RFT: T/L task only (5.5K train / 50 val), 512×512.

ENV_NAME="rft-verl-ds"

# Fail early on errors, treat unset vars as errors, and propagate failures in pipelines
set -euo pipefail

# Initialize conda for this shell before using `conda activate`
eval "$(conda shell.bash hook)"

# Use classic conda solver for deterministic non-interactive installs
conda config --set solver classic
conda --version 2>&1 | grep -qF "26.1.1" || conda install -y conda=26.1.1

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"

# Set paths
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
export MedVision_DATA_DIR=${data_dir}

# NOTE: The built dataset must be used only with the specified model_family_name or models sharing the same image processor.
# Supported model_family_name values: see get_resized_img_shape() in medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py
model_family_name="qwen25vl"
model_hf="Qwen/Qwen2.5-VL-7B-Instruct" # used to load the image processor
num_workers_concat_datasets=2
num_workers_format_dataset=32

# Data configs
# NOTE: At least one task JSON path must be provided; set multiple for multi-task training.
# NOTE: Allow sampling with replacement if limit exceeds dataset size.
# NOTE: train_sample_limit is a post-concatenation global cap applied after per-task limits;
#       keep it equal to the sum of per-task limits to avoid silent truncation.
tasks_list_json_path_TL="${benchmark_dir}/tasks_list/tasks_MedVision-TL__train_SFT.json" # Total samples: 5551
train_sample_limit=5500
val_sample_limit=50
train_sample_limit_task_TL=5500
val_sample_limit_task_TL=50
new_shape_hw=(512 512) # (height, width) passed to --new_shape_hw

# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
# tree) to avoid a CephFS metadata race in setuptools build_py (a build subdir can
# transiently vanish and the mkpath cache refuses to recreate it); always rebuilds
# to pick up local source changes. Only the shared-env install needs the lock.
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
export benchmark_dir wheelhouse lockfile
bash -c '
    set -euo pipefail
    build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
    trap "rm -rf \"${build_tmp}\"" EXIT
    tar -cf - -C "${benchmark_dir}" --exclude="*.egg-info" --exclude=__pycache__ \
        pyproject.toml MANIFEST.in LICENSE src \
      | tar -xf - -C "${build_tmp}"
    python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
    built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
    cp -f "${built_wheel}" "${wheelhouse}/"
    flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"
'

# Setup environment for SFT since we import SFT-related modules
# NOTE: update "--requirement" and "--lmms_eval_opt_deps" arguments based on the model_family_name
python -m medvision_bm.sft.env_setup --data_dir "${data_dir}" --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt" --lmms_eval_opt_deps qwen2_5_vl

# Build Verl datasets
python -m medvision_bm.rft.verl.build_parquet_ds \
    --model_family_name "${model_family_name}" \
    --model_hf "${model_hf}" \
    --data_dir "${data_dir}" \
    --num_workers_concat_datasets "${num_workers_concat_datasets}" \
    --num_workers_format_dataset "${num_workers_format_dataset}" \
    --tasks_list_json_path_TL "${tasks_list_json_path_TL}" \
    --train_sample_limit "${train_sample_limit}" \
    --val_sample_limit "${val_sample_limit}" \
    --train_sample_limit_task_TL "${train_sample_limit_task_TL}" \
    --val_sample_limit_task_TL "${val_sample_limit_task_TL}" \
    --new_shape_hw "${new_shape_hw[0]}" "${new_shape_hw[1]}"

conda deactivate
