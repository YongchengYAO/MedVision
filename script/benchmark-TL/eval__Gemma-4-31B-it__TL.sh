ENV_NAME="eval-gemma4"

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
model_hf_id="google/gemma-4-31B-it"
model_name="gemma-4-31B-it"
batch_size_per_gpu=10
gpu_memory_utilization=0.95
max_model_len=8192 # cap Gemma 4's 256K context so KV cache fits on a single 80GB GPU

# Stop string.
# REQUIRED: lmms-eval auto-injects the fewshot delimiter "\n\n" as a stop string for
# generate_until tasks. The CoT prompt puts blank lines between <step-k> blocks, so that
# "\n\n" stop halts generation mid-reasoning, before <answer> is produced. An explicit
# --stop_strings "</answer>" gives a clean terminator AND signals the wrapper to drop the
# auto-injected "\n\n" stop (see vllm_gemma4.py). NOTE: Gemma 4's native thinking mode is
# disabled (--no-enable_thinking) -- with thinking on it ignores the <think>/<answer> format
# and degenerates into repetition (validated: 5/5 parseable answers vs 0/5 with thinking on).
stop_string='</answer>'

# Other configs (safe to leave as is)
task_tag="MedVision-TL-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL-CoT.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=1000

# Install medvision_bm (locked shared build) from the worktree benchmark_dir
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
# python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
# python -m medvision_bm.benchmark.install_vendored_lmms_eval
# pip install -r "${benchmark_dir}/requirements/requirements_eval_gemma4.txt" --no-deps
#
# python -m medvision_bm.benchmark.eval__gemma4 \
# --skip_env_setup \
# --model_hf_id $model_hf_id \
# --model_name $model_name \
# --results_dir $result_dir \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size_per_gpu $batch_size_per_gpu \
# --gpu_memory_utilization $gpu_memory_utilization \
# --sample_limit $sample_limit \
# --max_model_len $max_model_len \
# --no-enable_thinking \
# --stop_strings "$stop_string"
# ---

# (Method 2) Automatically install requirements via the eval script's built-in setup pipeline
# Add these arguments for debugging:
# --env_setup_only \
# --skip_env_setup \
# --skip_update_status \
# ---
python -m medvision_bm.benchmark.eval__gemma4 \
    --model_hf_id $model_hf_id \
    --model_name $model_name \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size_per_gpu $batch_size_per_gpu \
    --gpu_memory_utilization $gpu_memory_utilization \
    --sample_limit $sample_limit \
    --max_model_len $max_model_len \
    --no-enable_thinking \
    --stop_strings "$stop_string"
# ---

conda deactivate
# conda remove -n $ENV_NAME --all -y
