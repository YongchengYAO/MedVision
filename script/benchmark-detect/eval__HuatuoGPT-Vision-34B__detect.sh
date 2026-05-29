ENV_NAME="eval-huatuogpt-vision"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"

# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
dir_third_party="${benchmark_dir}/third_party"
model_hf_id="FreedomIntelligence/HuatuoGPT-Vision-34B"
model_name="HuatuoGPT-Vision-34B"
batch_size_per_gpu=2

# Other configs (safe to leave as is)
task_tag="MedVision-detect"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-detect.json"
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

# Important: Fix module import failure in distributed subprocess
export PYTHONPATH="${dir_third_party}/HuatuoGPT-Vision:$PYTHONPATH"

# Run
# Add these arguments for debugging:
# --env_setup_only \
# --skip_env_setup \
# --skip_update_status \

export MedVision_PLANNER_VERSION='1.0.0'

python -m  medvision_bm.benchmark.eval__huatuogpt_vision \
--model_hf_id $model_hf_id \
--model_name $model_name \
--results_dir $result_dir \
--dir_third_party $dir_third_party \
--data_dir $data_dir \
--tasks_list_json_path $tasks_list_json_path \
--task_status_json_path $task_status_json_path \
--batch_size_per_gpu $batch_size_per_gpu \
--sample_limit $sample_limit \

conda deactivate
# conda remove -n $ENV_NAME --all -y
