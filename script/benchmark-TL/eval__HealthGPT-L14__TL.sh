ENV_NAME="eval-healthgpt"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"
pip install huggingface-hub

# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir="${benchmark_dir}/Data"
dir_third_party="${benchmark_dir}/third_party"
model_name="HealthGPT-L14"
batch_size_per_gpu=20

# Other configs (safe to leave as is)
task_tag="MedVision-TL"
result_dir="${benchmark_dir}/Results/${task_tag}"
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL.json"
task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"
sample_limit=1000

# Install medvision_bm
rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
pip install "${benchmark_dir}"

# Run
# Add these arguments for debugging:
# --env_setup_only \
# --skip_env_setup \
# --skip_update_status \
CUDA_VISIBLE_DEVICES=0 \
python -m  medvision_bm.benchmark.eval__healthgpt \
--model_name $model_name \
--results_dir $result_dir \
--dir_third_party $dir_third_party \
--data_dir $data_dir \
--tasks_list_json_path $tasks_list_json_path \
--task_status_json_path $task_status_json_path \
--batch_size_per_gpu $batch_size_per_gpu \
--sample_limit $sample_limit \
2>&1 | tee eval__HealthGPT-L14__TL.log

conda deactivate
# conda remove -n $ENV_NAME --all -y
