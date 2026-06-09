ENV_NAME="eval-qwen3vl"

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
model_hf_id="Qwen/Qwen3-VL-32B-Thinking"
model_name="Qwen3-VL-32B-Thinking"
batch_size_per_gpu=2
gpu_memory_utilization=0.95

# Sampling config.
# IMPORTANT: Qwen3-VL "Thinking" models are validated for SAMPLING, not greedy decoding.
# The values below mirror the model's own generation_config.json. To experiment, change them
# here; do NOT set temperature=0 for Thinking models. (Generation uses a fixed internal seed
# from medvision_bm.utils.configs.SEED, so sampling runs remain reproducible.)
temperature=0.8
top_p=0.95
top_k=20

# Stop string.
# REQUIRED for Qwen3-VL "Thinking": lmms-eval auto-injects the fewshot delimiter "\n\n" as a
# stop string for generate_until tasks. Thinking models put blank lines between CoT steps, so
# that "\n\n" stop halts generation right after <step-1-answer>, before <answer> is produced.
# Passing an explicit --stop_strings (a) gives a clean terminator at the end of the answer and
# (b) signals the wrapper to drop the auto-injected "\n\n" stop (see vllm_qwen3vl.py). 
stop_string='</answer>'

# Other configs (safe to leave as is)
task_tag="MedVision-AD-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-AD-CoT.json"
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

# (Method 1) Manually install requirements before running the eval script (more robust)
# ---
# python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
# python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps qwen3_vl
# pip install -r "${benchmark_dir}/requirements/requirements_eval_qwen3vl.txt" --no-deps
#
# python -m medvision_bm.benchmark.eval__qwen3_vl \
# --skip_env_setup \
# --lmmseval_module vllm_qwen3vl \
# --model_hf_id $model_hf_id \
# --model_name $model_name \
# --results_dir $result_dir \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size_per_gpu $batch_size_per_gpu \
# --gpu_memory_utilization $gpu_memory_utilization \
# --sample_limit $sample_limit \
# --temperature $temperature \
# --top_p $top_p \
# --top_k $top_k \
# --stop_strings "$stop_string"
# ---

# (Method 2) Automatically install requirements via the eval script's built-in setup pipeline
# Add these arguments for debugging:
# --env_setup_only \
# --skip_env_setup \
# --skip_update_status \
# ---
python -m medvision_bm.benchmark.eval__qwen3_vl \
--lmmseval_module vllm_qwen3vl \
--model_hf_id $model_hf_id \
--model_name $model_name \
--results_dir $result_dir \
--data_dir $data_dir \
--tasks_list_json_path $tasks_list_json_path \
--task_status_json_path $task_status_json_path \
--batch_size_per_gpu $batch_size_per_gpu \
--gpu_memory_utilization $gpu_memory_utilization \
--sample_limit $sample_limit \
--temperature $temperature \
--top_p $top_p \
--top_k $top_k \
--stop_strings "$stop_string"
# ---

conda deactivate
# conda remove -n $ENV_NAME --all -y
