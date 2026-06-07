ENV_NAME="rft-verl-ds"

# Fail early on errors, treat unset vars as errors, and propagate failures in pipelines
set -euo pipefail

# Initialize conda for this shell before using `conda activate`
eval "$(conda shell.bash hook)"

# TODO: debug - use classic solver and install specific conda non-interactively
conda config --set solver classic
conda install -y conda=26.1.1

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

# Important NOTE:
# ---
# The model image processor is used during dataset preparation to ensure the images are processed in a way that is compatible with the model.
# !!! The built dataset should be used only with the specified model_family_name or models with the same image processor.
# ---
# Supported model_family_name: check get_resized_img_shape() in medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py
model_family_name="qwen25vl"
model_hf="Qwen/Qwen2.5-VL-7B-Instruct" # Used to load the image processor
num_workers_concat_datasets=16
num_workers_format_dataset=256

# Data configs
# ----------------------------------------------------------------------------------
# ------
# NOTE: At least one of the following 3 task JSON paths must be provided. Set multiple task JSON paths for multi-task training
# ------
# tasks_list_json_path_AD="${benchmark_dir}/tasks_list/tasks_MedVision-AD__train_SFT.json" # Total samples: 5545
tasks_list_json_path_detect="${benchmark_dir}/tasks_list/tasks_MedVision-detect__train_SFT.json" # Total samples: 2695205
# tasks_list_json_path_TL="${benchmark_dir}/tasks_list/tasks_MedVision-TL__train_SFT.json" # Total samples: 5551

# ------
# NOTE: Allow sampling with replacement if limit exceeds dataset size 
# ------
# [Required] Sample limits in total
train_sample_limit=1000000
val_sample_limit=500

# # [Option 1] For approximately balanced sampling across 3 tasks
# train_sample_limit_per_task=333333 
# val_sample_limit_per_task=166

# # [Option 2] For task-specific sampling across 3 tasks
# train_sample_limit_task_AD=5500
# val_sample_limit_task_AD=45
train_sample_limit_task_Detection=1000000
val_sample_limit_task_Detection=500
# train_sample_limit_task_TL=5500
# val_sample_limit_task_TL=50

# (Optional) Resize shape for images during dataset preparation
new_shape_hw=(512 512)  # explicitly reshape images to size (height, width)
# ----------------------------------------------------------------------------------


# Install medvision_bm (locked shared build)
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
flock "${lockfile}" bash -c '
    set -euo pipefail
    benchmark_dir="'"${benchmark_dir}"'"
    wheelhouse="'"${wheelhouse}"'"
    rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
    python -m pip wheel "${benchmark_dir}" -w "${wheelhouse}" --no-deps
    latest_wheel="$(ls -t "${wheelhouse}"/medvision_bm-*.whl | head -n1)"
    python -m pip install --force-reinstall "${latest_wheel}"
'

# Setup environment for SFT since we import SFT-related modules
# NOTE: update "--requirement" and "--lmms_eval_opt_deps" arguments based on the model_family_name
python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt" --lmms_eval_opt_deps qwen2_5_vl


# Build Verl datasets
# ------
# Add optional argument below:
# To resize all images to a new shape during dataset preparation:
# --new_shape_hw ${new_shape_hw[0]} ${new_shape_hw[1]} \
# ------
# Improve crash visibility: PYTHONFAULTHANDLER dumps a traceback on fatal signals (e.g. SIGSEGV).
# HF_DATASETS_VERBOSITY=warning surfaces HuggingFace-level errors before they become silent hangs.
export PYTHONFAULTHANDLER=1
export HF_DATASETS_VERBOSITY=warning
python -m medvision_bm.rft.verl.build_parquet_ds__checkpointed \
--model_family_name ${model_family_name} \
--model_hf ${model_hf} \
--data_dir ${data_dir} \
--num_workers_concat_datasets ${num_workers_concat_datasets} \
--num_workers_format_dataset ${num_workers_format_dataset} \
--tasks_list_json_path_detect ${tasks_list_json_path_detect} \
--train_sample_limit ${train_sample_limit} \
--val_sample_limit ${val_sample_limit} \
--train_sample_limit_task_Detection ${train_sample_limit_task_Detection} \
--val_sample_limit_task_Detection ${val_sample_limit_task_Detection} \
--new_shape_hw ${new_shape_hw[0]} ${new_shape_hw[1]} \

conda deactivate
# conda remove -n $ENV_NAME --all -y
