ENV_NAME="sft-medgemma"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${ENV_NAME}" ]; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    conda create -n "${ENV_NAME}" python==3.11 -y
fi
conda activate "${ENV_NAME}"
conda install -c nvidia cuda-toolkit=12.4 -y

# Set paths
benchmark_dir="/mnt/vincent-pvc-rwm/Github/MedVision"
train_sft_dir="${benchmark_dir}/SFT"
data_dir="${benchmark_dir}/Data"

# Data configs
tasks_list_json_path="${benchmark_dir}/tasks_list/tasks_MedVision-TL__train_SFT.json"

# Model configs
base_model_hf="google/medgemma-27b-it"
run_name="MedVision__SFT__medgemma-27b-it__TL"
lora_checkpoint_dir="${train_sft_dir}/${run_name}/checkpoints/${run_name}" # Put ${run_name} at the end for distinct HF repo names when pushing LoRA checkpoints
merged_model_hf="MedVision__SFT-m__medgemma-27b-it__TL"
merged_model_dir="${train_sft_dir}/${run_name}/merged_model"

# Training configs
epoch=30
save_steps=100
eval_steps=50
logging_steps=50
save_total_limit=10 # Maximum number of checkpoints to save
use_flash_attention_2=true
num_workers_concat_datasets=4
num_workers_format_dataset=32
dataloader_num_workers=8
train_sample_limit=10000
val_sample_limit=100
dataloader_pin_memory=true
use_flash_attention_2=true

# Resumed training configs
resume_from_checkpoint=true # Enable resuming from the last checkpoint

# Resource-constrained training configs
gradient_checkpointing=true # Enable gradient checkpointing to save memory
per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=3 # Control effective batch size: effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

# Merge and push configs
push_LoRA=false # Push LoRA checkpoint to HF Hub after each save
push_merged_model=true # Push merged model to HF Hub after training
merge_only=false # [No training] Merge the last checkpoint and push to HF Hub
merge_model=true # [With training] Merge after training and push to HF Hub

# Set wandb configs for logging
wandb_resume="allow" # Wandb resume mode (e.g., 'allow', 'must', 'never')
wandb_dir="${train_sft_dir}/${run_name}"
wandb_project="MedVision-SFT"
wandb_run_name=${run_name}

# # NOTE: For continuing an existing run, set the wandb_run_id to the ID of the existing run.
# wandb_run_id="run-001" # run ID must be unique in the wandb_project

# Install medvision_bm
rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
pip install "${benchmark_dir}"

# Setup training env
python -m medvision_bm.sft.env_setup --data_dir ${data_dir}

# # [Alternative] Setup training env: use a specific requirements file
# python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_medgemma.txt"

# # [Debugging] Disable WANDB online logging
# export WANDB_MODE=offline      # or HF_DISABLE_WANDB=1
# export WANDB_CORE_DEBUG=true
# export WANDB_DEBUG=true

# # Run
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --num_processes=2 --main_process_port=29502 --mixed_precision=bf16 \
-m  medvision_bm.sft.train__SFT__medgemma__TL \
--run_name ${run_name} \
--base_model_hf ${base_model_hf} \
--lora_checkpoint_dir ${lora_checkpoint_dir} \
--merged_model_hf ${merged_model_hf} \
--merged_model_dir ${merged_model_dir} \
--wandb_resume ${wandb_resume} \
--wandb_dir ${wandb_dir} \
--wandb_project ${wandb_project} \
--wandb_run_name ${wandb_run_name} \
--wandb_run_id ${wandb_run_id} \
--data_dir ${data_dir} \
--tasks_list_json_path ${tasks_list_json_path} \
--epoch ${epoch} \
--save_steps ${save_steps} \
--eval_steps ${eval_steps} \
--logging_steps ${logging_steps} \
--save_total_limit ${save_total_limit} \
--per_device_train_batch_size ${per_device_train_batch_size} \
--per_device_eval_batch_size ${per_device_eval_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--use_flash_attention_2 ${use_flash_attention_2} \
--num_workers_concat_datasets ${num_workers_concat_datasets} \
--num_workers_format_dataset ${num_workers_format_dataset} \
--dataloader_num_workers ${dataloader_num_workers} \
--train_sample_limit ${train_sample_limit} \
--val_sample_limit ${val_sample_limit} \
--push_LoRA ${push_LoRA} \
--push_merged_model ${push_merged_model} \
--merge_model ${merge_model} \
--merge_only ${merge_only} \
--resume_from_checkpoint ${resume_from_checkpoint} \
--gradient_checkpointing ${gradient_checkpointing} \
--dataloader_pin_memory ${dataloader_pin_memory} \
2>&1 | tee train__SFT__medgemma-27b-it__TL.log

conda deactivate
# conda remove -n $ENV_NAME --all -y
