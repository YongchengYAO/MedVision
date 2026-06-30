# Pixel-size-scaled (-scaledPS) benchmark variant of
# script/benchmark-AD/eval__MedVision-V0-7B__AD.sh.
#
# Identical to the regular AD eval except it targets the scaledPS task list and
# passes the pixel-size scaling range: the prompt's reported pixel_size is
# multiplied by a per-sample factor while the image pixels are unchanged, testing
# whether the model reasons from the reported pixel_size. See commit 2bfec27 for
# the scaledPS benchmark design.

ENV_NAME="eval-medvision-v0"

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
model_hf_id="YongchengYAO/MedVision-V0-7B"
model_name="MedVision-V0-7B"
batch_size_per_gpu=10
gpu_memory_utilization=0.9
reshape_image_hw="512x512"

# Pixel-size scaling factor range for the -scaledPS variant (default [0.5, 3.0];
# the *_largeS stress runs used [10, 20]).
scaled_ps_low=0.5
scaled_ps_high=3.0

# Other configs (safe to leave as is)
task_tag="MedVision-AD-CoT-scaledPS"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-AD-CoT-scaledPS.json"
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
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps medvision_v0
pip install -r "${benchmark_dir}/requirements/requirements_eval_medvision-v0.txt" --no-deps

python -m medvision_bm.benchmark.eval__medvision-model-rft \
    --skip_env_setup \
    --model_hf_id $model_hf_id \
    --model_name $model_name \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size_per_gpu $batch_size_per_gpu \
    --gpu_memory_utilization $gpu_memory_utilization \
    --sample_limit $sample_limit \
    --reshape_image_hw $reshape_image_hw \
    --scaled_ps_low $scaled_ps_low \
    --scaled_ps_high $scaled_ps_high \
    --use_system_prompt
# ---

# # (Method 2) Automatically install requirements in the eval script (simpler, but may incur package version conflicts or bugs introduced by new versions of packages)
# # Add these arguments for debugging:
# # --env_setup_only \
# # --skip_env_setup \
# # --skip_update_status \
# python -m medvision_bm.benchmark.eval__medvision-model-rft \
# --model_hf_id $model_hf_id \
# --model_name $model_name \
# --results_dir $result_dir \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size_per_gpu $batch_size_per_gpu \
# --gpu_memory_utilization $gpu_memory_utilization \
# --sample_limit $sample_limit \
# --reshape_image_hw $reshape_image_hw \
# --scaled_ps_low $scaled_ps_low \
# --scaled_ps_high $scaled_ps_high \
# --use_system_prompt

conda deactivate
# conda remove -n $ENV_NAME --all -y
