ENV_NAME="eval-claude"

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
model_name="Claude-Fable-5"
batch_size=1

# API provider and model code
# - anthropic (direct): https://platform.claude.com/docs/en/about-claude/models/overview
# - openrouter: set api_provider="openrouter" and use an OpenRouter model ID,
#   e.g. anthropic_model_code="anthropic/claude-fable-5" (https://openrouter.ai/models);
#   requires OPENROUTER_API_KEY instead of ANTHROPIC_API_KEY.
api_provider="anthropic"
anthropic_model_code="claude-fable-5"

# API key check + sanitization (pod-injected env vars can carry a trailing newline,
# which breaks HTTP auth headers)
if [ "${api_provider}" = "anthropic" ]; then
    api_key_var="ANTHROPIC_API_KEY"
else
    api_key_var="OPENROUTER_API_KEY"
fi
if [ -z "${!api_key_var:-}" ]; then
    echo "[Error] ${api_key_var} is not set." >&2
    exit 1
fi
export "${api_key_var}"="$(printf '%s' "${!api_key_var}" | tr -d '\n')"

# Other configs (safe to leave as is)
task_tag="MedVision-detect-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-detect-CoT.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=100
reshape_image_hw="512x512"

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

# (Method 1) Manually install requirements before running the eval script (more robust)
# ---
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps claude
pip install -r "${benchmark_dir}/requirements/requirements_eval_claude.txt" --no-deps

python -m medvision_bm.benchmark.eval__claude \
--skip_env_setup \
--api_provider $api_provider \
--anthropic_model_code $anthropic_model_code \
--model_name $model_name \
--results_dir $result_dir \
--data_dir $data_dir \
--tasks_list_json_path $tasks_list_json_path \
--task_status_json_path $task_status_json_path \
--batch_size $batch_size \
--sample_limit $sample_limit \
--reshape_image_hw $reshape_image_hw \
# ---

# # (Method 2) Automatically install requirements in the eval script (simpler, but may incur package version conflicts or bugs introduced by new versions of packages)
# # Add these arguments for debugging:
# # --env_setup_only \
# # --skip_env_setup \
# # --skip_update_status \
# python -m medvision_bm.benchmark.eval__claude \
# --api_provider $api_provider \
# --anthropic_model_code $anthropic_model_code \
# --model_name $model_name \
# --results_dir $result_dir \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size $batch_size \
# --sample_limit $sample_limit \
# --reshape_image_hw $reshape_image_hw \

conda deactivate
# conda remove -n $ENV_NAME --all -y
