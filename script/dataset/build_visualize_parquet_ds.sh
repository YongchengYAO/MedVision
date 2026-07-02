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

# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
# tree). setuptools build_py caches created dirs in a process-global memo, and on
# CephFS a build subdir can transiently vanish (async delete/recreate lag or an
# unguarded concurrent writer), after which the cache refuses to recreate it and a
# later file copy dies with: could not create '...': No such file or directory.
# A private local build dir is immune; only the shared-env install needs the lock.
set -euo pipefail
lockfile="${dir_medvision}/.medvision_build.lock"
wheelhouse="${dir_medvision}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${dir_medvision}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"

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
