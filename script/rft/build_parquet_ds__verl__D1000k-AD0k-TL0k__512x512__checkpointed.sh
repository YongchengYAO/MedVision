# Build verl parquet dataset for RFT: Detection task only, large scale (1M train / 500 val), 512×512.
# Uses the checkpointed builder to avoid OOM when processing large datasets.

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
model_hf="Qwen/Qwen2.5-VL-7B-Instruct"  # used to load the image processor
num_workers_concat_datasets=16
num_workers_format_dataset=256

# Data configs
# NOTE: At least one task JSON path must be provided; set multiple for multi-task training.
# NOTE: Allow sampling with replacement if limit exceeds dataset size.
# NOTE: train_sample_limit is a post-concatenation global cap applied after per-task limits;
#       keep it equal to the sum of per-task limits to avoid silent truncation.
tasks_list_json_path_detect="${benchmark_dir}/tasks_list/tasks_MedVision-detect__train_SFT.json"  # Total samples: 2695205
train_sample_limit=1000000
val_sample_limit=500
train_sample_limit_task_Detection=1000000
val_sample_limit_task_Detection=500
new_shape_hw=(512 512)  # (height, width) passed to --new_shape_hw


# Install medvision_bm (locked shared build); always rebuilds to pick up local source changes
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
export benchmark_dir wheelhouse
flock "${lockfile}" bash -c '
    set -euo pipefail
    rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
    python -m pip wheel "${benchmark_dir}" -w "${wheelhouse}" --no-deps
    latest_wheel="$(ls -t "${wheelhouse}"/medvision_bm-*.whl | head -n1)"
    python -m pip install --force-reinstall "${latest_wheel}"
'

# Setup environment for SFT since we import SFT-related modules
# NOTE: update "--requirement" and "--lmms_eval_opt_deps" arguments based on the model_family_name
python -m medvision_bm.sft.env_setup --data_dir "${data_dir}" --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt" --lmms_eval_opt_deps qwen2_5_vl


# Build Verl datasets
# PYTHONFAULTHANDLER dumps a traceback on fatal signals (e.g. SIGSEGV).
# HF_DATASETS_VERBOSITY=warning surfaces HuggingFace-level errors before they become silent hangs.
export PYTHONFAULTHANDLER=1
export HF_DATASETS_VERBOSITY=warning
python -m medvision_bm.rft.verl.build_parquet_ds__checkpointed \
--model_family_name "${model_family_name}" \
--model_hf "${model_hf}" \
--data_dir "${data_dir}" \
--num_workers_concat_datasets "${num_workers_concat_datasets}" \
--num_workers_format_dataset "${num_workers_format_dataset}" \
--tasks_list_json_path_detect "${tasks_list_json_path_detect}" \
--train_sample_limit "${train_sample_limit}" \
--val_sample_limit "${val_sample_limit}" \
--train_sample_limit_task_Detection "${train_sample_limit_task_Detection}" \
--val_sample_limit_task_Detection "${val_sample_limit_task_Detection}" \
--new_shape_hw "${new_shape_hw[0]}" "${new_shape_hw[1]}"

conda deactivate
