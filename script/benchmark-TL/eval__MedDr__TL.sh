ENV_NAME="eval-meddr"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.9 -y
fi
conda activate "${ENV_NAME}"

# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
dir_third_party="${benchmark_dir}/third_party"
model_hf_id="Sunanhe/MedDr_0401"
model_name="MedDr"
batch_size_per_gpu=2

# Other configs (safe to leave as is)
task_tag="MedVision-TL-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL-CoT.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=1000

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

# Use MedVision dataset v1.0.0
export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.1.1'

# (Method 1) Manually install requirements before running the eval script (more robust)
# ---
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval
pip install -r "${benchmark_dir}/requirements/requirements_eval_meddr.txt" --no-deps

# Important: Fix module import failure in distributed subprocess
export PYTHONPATH="${dir_third_party}/MedDr:$PYTHONPATH"

python -m medvision_bm.benchmark.eval__meddr \
    --skip_env_setup \
    --model_hf_id $model_hf_id \
    --model_name $model_name \
    --results_dir $result_dir \
    --dir_third_party $dir_third_party \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size_per_gpu $batch_size_per_gpu \
    --sample_limit $sample_limit
# ---

# # (Method 2) Automatically install requirements in the eval script (simpler, but may incur package version conflicts or bugs introduced by new versions of packages)
# # Add these arguments for debugging:
# # --env_setup_only \
# # --skip_env_setup \
# # --skip_update_status \
# python -m  medvision_bm.benchmark.eval__meddr \
# --model_hf_id $model_hf_id \
# --model_name $model_name \
# --results_dir $result_dir \
# --dir_third_party $dir_third_party \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size_per_gpu $batch_size_per_gpu \
# --sample_limit $sample_limit \

conda deactivate
# conda remove -n $ENV_NAME --all -y
