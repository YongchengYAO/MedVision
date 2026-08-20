ENV_NAME="eval-glm4v"

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
model_hf_id="zai-org/GLM-4.6V"
model_name="GLM-4.6V"
batch_size_per_gpu=1
gpu_memory_utilization=0.95

# Sampling config.
# IMPORTANT: GLM-4.6V is a hybrid-reasoning model validated for SAMPLING, not greedy decoding.
# The values below mirror the model's own generation_config.json (temperature=0.8, top_p=0.6,
# top_k=2); repetition_penalty=1.1 is the model-card recommendation. To experiment, change them
# here; do NOT set temperature=0 (greedy makes it stop early inside <think>, before <answer>).
# (Generation uses a fixed internal seed from medvision_bm.utils.configs.SEED, so sampling runs
# remain reproducible.)
temperature=0.8
top_p=0.6
top_k=2
repetition_penalty=1.1

# Stop string.
# REQUIRED for GLM-4.6V (reasoning): lmms-eval auto-injects the fewshot delimiter "\n\n" as a
# stop string for generate_until tasks. Reasoning models put blank lines between CoT steps, so
# that "\n\n" stop halts generation right after <step-1-answer>, before <answer> is produced.
# Passing an explicit --stop_strings (a) gives a clean terminator at the end of the answer and
# (b) signals the wrapper to drop the auto-injected "\n\n" stop (see vllm_glm4v.py).
stop_string='</answer>'

# Other configs (safe to leave as is)
task_tag="MedVision-TL-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL-CoT.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=1000

# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
# tree). setuptools build_py caches created dirs in a process-global memo, and on
# CephFS a build subdir can transiently vanish (async delete/recreate lag or an
# unguarded concurrent writer), after which the cache refuses to recreate it and a
# later file copy dies with: could not create '...': No such file or directory.
# A private local build dir is immune; only the shared-env install needs the lock.
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"

# Use MedVision dataset v1.0.0
export MedVision_PLANNER_VERSION='1.0.0'
export MedVision_ACK_RELEASE='1.4.0'

# Set output token limit (default to 4096)
max_new_tokens=4096

# (Method 1) Manually install requirements before running the eval script (more robust)
# ---
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps glm4v
pip install -r "${benchmark_dir}/requirements/requirements_eval_glm4v.txt" --no-deps

python -m medvision_bm.benchmark.eval__glm4v \
    --skip_env_setup \
    --lmmseval_module vllm_glm4v \
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
    --repetition_penalty $repetition_penalty \
    --max_new_tokens $max_new_tokens \
    --stop_strings "$stop_string"
# ---

# (Method 2) Automatically install requirements via the eval script's built-in setup pipeline
# Add these arguments for debugging:
# --env_setup_only \
# --skip_env_setup \
# --skip_update_status \
# ---
# python -m medvision_bm.benchmark.eval__glm4v \
# --lmmseval_module vllm_glm4v \
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
# --repetition_penalty $repetition_penalty \
# --max_new_tokens $max_new_tokens \
# --stop_strings "$stop_string"
# ---

conda deactivate
# conda remove -n $ENV_NAME --all -y
