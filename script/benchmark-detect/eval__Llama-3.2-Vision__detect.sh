ENV_NAME="eval-llama3-vision"

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
model_hf_id="meta-llama/Llama-3.2-11B-Vision-Instruct"
model_name="Llama-3.2-11B-Vision-Instruct"
batch_size_per_gpu=4
gpu_memory_utilization=0.9

# Other configs (safe to leave as is)
task_tag="MedVision-detect-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-detect-CoT.json"
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
# - use requirements_*.txt
# - install medvision_ds and vendored_lmms_eval, since they are not included in the requirements file
# - use '--skip_env_setup' in the eval__medgemma script to skip re-installing packages
# ---
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval
pip install -r "${benchmark_dir}/requirements/requirements_eval_llama3_vision.txt" --no-deps

python -m medvision_bm.benchmark.eval__llama3_2_vision \
    --skip_env_setup \
    --model_hf_id $model_hf_id \
    --model_name $model_name \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size_per_gpu $batch_size_per_gpu \
    --gpu_memory_utilization $gpu_memory_utilization \
    --sample_limit $sample_limit
# ---

# # (Method 2) Automatically install requirements in the eval script (simpler, but may incur package version conflicts or bugs introduced by new versions of packages)
# # Add these arguments for debugging:
# # --env_setup_only \
# # --skip_env_setup \
# # --skip_update_status \
# python -m  medvision_bm.benchmark.eval__llama3_2_vision \
# --model_hf_id $model_hf_id \
# --model_name $model_name \
# --results_dir $result_dir \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size_per_gpu $batch_size_per_gpu \
# --gpu_memory_utilization $gpu_memory_utilization \
# --sample_limit $sample_limit \

conda deactivate
# conda remove -n $ENV_NAME --all -y
