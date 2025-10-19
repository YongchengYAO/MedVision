# Debugging env setup



## 📊 Benchmark

- **[Debug]** Use the dependencies list in `requirements` for debugging packages conflict

  1. Install the benchmark codebase `medvision_bm`

  2. Modify dependencies list, such as `requirements/requirements_eval_qwen25vl.txt`

  3. Setup env

     📝 Match `--lmms_eval_opt_deps` with model:
     - Choose from [`meddr`, `lingshu`, `huatuogpt_vision`, `llava_med`, `qwen2_5_vl`, `gemini`] – defined [here](https://github.com/YongchengYAO/MedVision/blob/master/src/medvision_bm/medvision_lmms-eval/pyproject.toml)
     - If model is not one of these, ignore `--lmms_eval_opt_deps` 

     ```bash
     # NOTE: replace <local-data-folder>
     python -m medvision_bm.benchmark.env_setup -r requirements/requirements_eval_qwen25vl.txt --lmms_eval_opt_deps qwen2_5_vl --data_dir <local-data-folder>
     ```

  4. Skip env setup with `--skip_env_setup` in benchmarking scripts `script/medvision-*/eval__*`

      ```bash
      # Run
      # Add these arguments for debugging:
      # --skip_env_setup \
      # --skip_update_status \
      CUDA_VISIBLE_DEVICES=0 \
      python -m  medvision_bm.benchmark.eval__qwen2_5_vl \
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
      2>&1 | tee eval__Qwen2.5-VL__AD.log
      ```

<br/>

## 🎯 Training: SFT

- **[Debug]** Use the dependencies list in `requirements` for debugging packages conflict

  1. Modify dependencies list, such as `requirements/requirements_sft_qwen25vl.txt`

  2. Use the alternative setup command in `script/medvision-*/train__SFT__*` 
      ```bash
      # [Alternative] Setup training env: use a specific requirements file
      python -m medvision_bm.sft.env_setup --data_dir ${data_dir} --requirement "${benchmark_dir}/requirements/requirements_sft_qwen25vl.txt"
      ```
      

<br/>

