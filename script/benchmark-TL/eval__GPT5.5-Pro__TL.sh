ENV_NAME="eval-openai"

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
benchmark_dir="/mnt/vincent-pvc-rwm/Github/MedVision/"
#benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
model_name="GPT-5.5-Pro"
batch_size=1
reasoning_effort="low"
max_tokens=4096

# API provider and model code
# - openai (direct): https://developers.openai.com/api/docs/models
# - openrouter: use an OpenRouter model ID (https://openrouter.ai/models);
#   requires OPENROUTER_API_KEY instead of OPENAI_API_KEY.
# ---
#api_provider="openai"
#openai_model_code="gpt-5.5-pro"
api_provider="openrouter"
openai_model_code="openai/gpt-5.5-pro"
# ---

# API key check + sanitization (pod-injected env vars can carry a trailing newline,
# which breaks HTTP auth headers)
if [ "${api_provider}" = "openai" ]; then
    api_key_var="OPENAI_API_KEY"
else
    api_key_var="OPENROUTER_API_KEY"
fi
if [ -z "${!api_key_var:-}" ]; then
    echo "[Error] ${api_key_var} is not set." >&2
    exit 1
fi
export "${api_key_var}"="$(printf '%s' "${!api_key_var}" | tr -d '\n')"

# Other configs (safe to leave as is)
task_tag="MedVision-TL-CoT"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL-CoT.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=100
reshape_image_hw="512x512"

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
export MedVision_ACK_RELEASE='1.1.1'

# (Method 1) Manually install requirements before running the eval script (more robust)
# ---
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps openai
pip install -r "${benchmark_dir}/requirements/requirements_eval_gpt.txt" --no-deps

python -m medvision_bm.benchmark.eval__openai \
    --skip_env_setup \
    --api_provider $api_provider \
    --openai_model_code $openai_model_code \
    --model_name $model_name \
    --reasoning_effort $reasoning_effort \
    --max_tokens $max_tokens \
    --results_dir $result_dir \
    --data_dir $data_dir \
    --tasks_list_json_path $tasks_list_json_path \
    --task_status_json_path $task_status_json_path \
    --batch_size $batch_size \
    --sample_limit $sample_limit \
    --reshape_image_hw $reshape_image_hw
# ---

# # (Method 2) Automatically install requirements in the eval script (simpler, but may incur package version conflicts or bugs introduced by new versions of packages)
# # Add these arguments for debugging:
# # --env_setup_only \
# # --skip_env_setup \
# # --skip_update_status \
#python -m medvision_bm.benchmark.eval__openai \
# --api_provider $api_provider \
# --openai_model_code $openai_model_code \
# --model_name $model_name \
# --reasoning_effort $reasoning_effort \
# --max_tokens $max_tokens \
# --results_dir $result_dir \
# --data_dir $data_dir \
# --tasks_list_json_path $tasks_list_json_path \
# --task_status_json_path $task_status_json_path \
# --batch_size $batch_size \
# --sample_limit $sample_limit \
# --reshape_image_hw $reshape_image_hw \

conda deactivate
## conda remove -n $ENV_NAME --all -y
