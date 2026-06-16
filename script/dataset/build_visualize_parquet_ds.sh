#! /bin/bash
ENV_NAME="medvision-prep-ds"

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
dir_medvision="/root/Documents/MedVision"
export MedVision_DATA_DIR="${dir_medvision}/Data"
dir_parquet="${MedVision_DATA_DIR}/raw_parquet"
dir_figure="${dir_medvision}/Figures"

# Install medvision_bm (locked shared build)
set -euo pipefail
lockfile="${dir_medvision}/.medvision_build.lock"
wheelhouse="${dir_medvision}/.wheelhouse"
mkdir -p "${wheelhouse}"
flock "${lockfile}" bash -c '
    set -euo pipefail
    dir_medvision="'"${dir_medvision}"'"
    wheelhouse="'"${wheelhouse}"'"
    rm -rf "${dir_medvision}/build" "${dir_medvision}/src/medvision_bm.egg-info"
    python -m pip wheel "${dir_medvision}" -w "${wheelhouse}" --no-deps
    latest_wheel="$(ls -t "${wheelhouse}"/medvision_bm-*.whl | head -n1)"
    python -m pip install --force-reinstall "${latest_wheel}"
'

# Install medvision_ds and vendored lmms_eval
# NOTE: visualization requires medvision_ds and the vendored lmms_eval
python -m medvision_bm.benchmark.install_medvision_ds --data_dir ${MedVision_DATA_DIR}
python -m medvision_bm.benchmark.install_vendored_lmms_eval
# Force reinstall some packages (temporary solution)
pip install transformers==4.57.1

# NOTE: Check medvision_bm/dataset/build_parquet_ds.py for setting sample size limit for each task
# ---
# Building parquet datasets
# Detecction
python -m medvision_bm.dataset.build_parquet_ds \
    --parquet_ds_dir ${dir_parquet}/medvision_Detection \
    --tasks_list_json_path_detect ${dir_medvision}/tasks_list/tasks_MedVision-detect__train_SFT.json \
    --num_workers_concat_datasets 1

# AD
python -m medvision_bm.dataset.build_parquet_ds \
    --parquet_ds_dir ${dir_parquet}/medvision_AD \
    --tasks_list_json_path_AD ${dir_medvision}/tasks_list/tasks_MedVision-AD__train_SFT.json \
    --num_workers_concat_datasets 1

# TL
python -m medvision_bm.dataset.build_parquet_ds \
    --parquet_ds_dir ${dir_parquet}/medvision_TL \
    --tasks_list_json_path_TL ${dir_medvision}/tasks_list/tasks_MedVision-TL__train_SFT.json \
    --num_workers_concat_datasets 1
# ---

# Visualization
# ---
# Detection
python -m medvision_bm.dataset.visualize_samples \
    --parquet_ds_path ${dir_parquet}/medvision_Detection/test.parquet \
    --fig_dir ${dir_figure}/Fig-Detection \
    --num_samples 100 \
    --task_type Detection

# Angle & Distance
python -m medvision_bm.dataset.visualize_samples \
    --parquet_ds_path ${dir_parquet}/medvision_AD/test.parquet \
    --fig_dir ${dir_figure}/Fig-AD-Angle \
    --num_samples 100 \
    --task_type Angle

python -m medvision_bm.dataset.visualize_samples \
    --parquet_ds_path ${dir_parquet}/medvision_AD/test.parquet \
    --fig_dir ${dir_figure}/Fig-AD-Distance \
    --num_samples 100 \
    --task_type Distance

# Tumor/Lesion size
python -m medvision_bm.dataset.visualize_samples \
    --parquet_ds_path ${dir_parquet}/medvision_TL/test.parquet \
    --fig_dir ${dir_figure}/Fig-TL \
    --num_samples 100 \
    --task_type TL
# ---
