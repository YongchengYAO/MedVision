![MedVision](fig/medvision-logo.png)

# About

This is the official codebase `medvision_bm` of the **MedVision** project. 

🌏 [**Project**](https://medvision-vlm.github.io) 🧑🏻‍💻 [**Code**](https://github.com/YongchengYAO/MedVision) 🩻 [**Dataset**](https://huggingface.co/datasets/YongchengYAO/MedVision) 🐳 [**Docker**](https://hub.docker.com/r/vincentycyao/medvision/tags) 🤗 [**SFT Models**](https://huggingface.co/collections/YongchengYAO/medvision-sft-models) 📖 [**arXiv**](https://arxiv.org/abs/2511.18676)

```
@misc{yao2025medvisiondatasetbenchmarkquantitative,
      title={MedVision: Dataset and Benchmark for Quantitative Medical Image Analysis}, 
      author={Yongcheng Yao and Yongshuo Zong and Raman Dutt and Yongxin Yang and Sotirios A Tsaftaris and Timothy Hospedales},
      year={2025},
      eprint={2511.18676},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18676}, 
}
```

<br/>

# 🔥 News

- [Oct 8, 2025] 🚀 Release **MedVision** dataset v1.0.0

<br/>

# 📜 TODO

- [x] Add preprint and project page
- [x] Release training code 
- [x] Release docker images
- [ ] New tasks guide
- [ ] New models guide

<br/>

# 🛠️ Install `medvision_bm`

```bash
pip install "git+https://github.com/YongchengYAO/MedVision.git"
```

<br/>

# 🔧 Install `medvision_ds` (Optional)

`medvision_ds` is the dataset codebase. It can be installed from `medvision_bm`:

```bash
# Replace <local-data-folder>
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <local-data-folder>  
```

<br/>

# 🐳 Use Docker

📝 Docker images are built from these [dockerfiles](https://github.com/YongchengYAO/MedVision/tree/master/dockerfile)

1. Choose the docker image for a specific model: https://hub.docker.com/r/vincentycyao/medvision/tags
2. Map local volumes and GPUs, use docker image `vincentycyao/medvision:<tag>`

```bash
# NOTE: replace </path/to/working/folder>, <tag>
docker run -it --rm \
	--gpus '"device=0,1"' \
	-v </path/to/working/folder>:/root/Documents/MedVision \
	vincentycyao/medvision:<tag> \
	bash
```

```bash
# In the container
git clone https://github.com/YongchengYAO/MedVision.git /root/Documents/MedVision
```

Next:

- Check the conda env name and activate: `conda env list`, `conda activate <env>`

- Skip environment setup:

  - Benchmarking: use `--skip_env_setup` for scripts in /root/Documents/MedVision/script/benchmark-*

  - SFT: disable this line in /root/Documents/MedVision/script/sft-*

    ```bash
    python -m medvision_bm.sft.env_setup --data_dir ${data_dir}
    ```

[File structure](https://github.com/YongchengYAO/MedVision/tree/master/docs/file-structure.md): imaging data, benchmark results, and model checkpoints are automatically saved

<br/>

# 📊 Benchmark

- **[Usage]** 

  1. The scripts in `script/benchmark-*/eval__*` should be sufficient for dependencies installation, data processing, and benchmarking
  2. After evaluating all models in step 1, parse model outputs and calculate metrics (e.g., MRE, MAE, IoU, Success Rate):

  ```bash
  # CLI command: 
  # python -m medvision_bm.benchmark.parse_outputs
  #
  # args:
  # --task_type: ["AD", "TL", "Detection"]
  # --task_dir: task folder
  # --model_dir: model folder
  # --limit: limit sample size in the parsed files
  # --skip_existing: (store_ture arg) skip parsed files
  
  # example 1: parse all models for the T/L task 
  python -m medvision_bm.benchmark.parse_outputs --task_type TL --task_dir Results/MedVision-TL
  
  # example 2: parse one model for the detection task and skip existing parsed files
  python -m medvision_bm.benchmark.parse_outputs --task_type Detection --model_dir Results/MedVision-detect/Qwen2.5-VL-32B-Instruct --skip_existing
  ```

  3. Summarize model performance for each task

  ```bash
  # CLI command: 
  # python -m medvision_bm.benchmark.summarize_AD_task 
  # python -m medvision_bm.benchmark.summarize_detection_task
  # python -m medvision_bm.benchmark.summarize_TL_task
  # python -m medvision_bm.benchmark.analyze_detection_task_boxsize
  # python -m medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random
  #
  # args:
  # --task_dir: task folder
  # --model_dir: model folder
  # --skip_model_wo_parsed_files: skip model directories that don't have a 'parsed' folder
  
  # example 1: summarize all models for the A/D task
  python -m medvision_bm.benchmark.summarize_AD_task --task_dir Results/MedVision-AD
  
  # example 2: summarize one model for the detection task
  python -m medvision_bm.benchmark.summarize_detection_task --model_dir Results/MedVision-detect/Qwen2.5-VL-32B-Instruct
  
  # example 3: analyze how target size affect detection performance
  python -m medvision_bm.benchmark.analyze_detection_task_boxsize --task_dir Results/MedVision-detect
  
  # example 4: compare detection performance with randow guessing
  python -m medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random --task_dir Results/MedVision-detect
  ```

  File structure after these steps:

  ```
  ├── MedVision
  	├── completed_tasks 
  		├── completed_tasks_MedVision-AD.json         # <== tasks status tracker
  		├── ...
  	├── Results                                     # <== benchmark results
  		├── MedVision-AD
  			├── ...
  			├── summary_AD_task.txt                     # <== [step 3] summary
  		├── MedVision-detect
  			├── Qwen2.5-VL-32B-Instruct
  				├── parsed                               
  					├── *.jsonl                             # <== [step 2] parsed model outputs
  					├── *.json                              # <== [step 2] parsed summary file
  					├── summary_*                           # <== [step 3] mean metrics, values
  				├── *.jsonl                               # <== [step 1] model outputs
  				├── *.json                                # <== [step 1] summary file
  			├── ...
  			├── summary_detection_task.txt              # <== [step 3] summary
  		├── MedVision-TL
  			├── ...
  			├── summary_TL_task.txt                     # <== [step 3] summary
  ```

- **[Debug]** [here](https://github.com/YongchengYAO/MedVision/tree/master/docs/debug_env_setup.md)


<br/>

# 🎯 Training: SFT

- **[Usage]** The scripts in `script/sft-*/train__SFT__*` should be sufficient for dependencies installation, data processing, and training.
- **[Debug]** [here](https://github.com/YongchengYAO/MedVision/tree/master/docs/debug_env_setup.md)

- **[Tutorial]** [Supervised Fine-Tuning (SFT) for VLMs on Medical Image Data](https://github.com/YongchengYAO/MedVision/tree/master/docs/SFT_Tutorial.md), [discussion channel](https://discuss.huggingface.co/t/supervised-fine-tuning-sft-for-vlms-on-medical-image-data/170805#p-246436-supervised-fine-tuning-sft-for-vlms-on-medical-image-data-1)

- **[SFT Model Checkpoints]** [details](https://github.com/YongchengYAO/MedVision/tree/master/docs/SFT_model_checkpoints.md)

<br/>

# 💿 Data Downloading (Optional)

Something about the **MedVision** dataset:

- Concepts
  - `MedVision`: the collection of public imaging data and our annotations
  - `dataset`: name of the public datasets, such `BraTS24`, `MSD`, `OAIZIB-CM`
  - `data-config`: name of predefined subsets
    - naming convention: `{dataset}_{annotation-type}_{task-ID}_{slice}_{split}`
      - `dataset`: [details](https://huggingface.co/datasets/YongchengYAO/MedVision#datasets)
      - `annotation-type`: 
        - `BoxSize`: detection annotations (bounding box)
        - `TumorLesionSize`: tumor/lesion size annotations
        - `BiometricsFromLandmarks`: angle/distance annotations
      - `task-ID`: `Task[xx]`
        - For datasets with multiple image-mask pairs, we defined tasks in `medvision_ds/datasets/*/preprocess_*.py`
        - source: [medvision_ds](https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/src)
        - e.g., detection tasks for the `BraTS24` dataset is defined in the `benchmark_plan` in `medvision_ds/datasets/BraTS24/preprocess_detection.py`
      - `slice`: [`Sagittal`, `Coronal`, `Axial`]
      - `split`: [`Train`, `Test`]
  
- Any combination of [`data-config` x `split`] will incur the downloading and processing of the whole `dataset`

Since it takes some time for data downloading and processing, you can just download datasets from tasks list (example [here](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list)) or configs list (example [here](https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/info)) in advance.

⚠️ You need to set API token for these datasets (see [detailed instructions](https://huggingface.co/datasets/YongchengYAO/MedVision#datasets)): FeTA24, SKM-TEA, and ToothFairy2

```bash
# CLI command:
# python -m medvision_bm.benchmark.download_datasets
#
# arg:
# --data_dir: (required) data folder
# --tasks_json: task json file
# --configs_csv: config json file
# --force_download_data: (store_true arg) force redownload raw imaging data
# ⚠️ `--force_download_data` is for debugging only, it will repeatedly download data for tasks/configs of the same dataset

# NOTE: replace <task-list-json>, <data-folder>
python -m medvision_bm.benchmark.download_datasets --tasks_json <task-list-json> --data_dir <data-folder>
```
or
```bash
# NOTE: replace <config-list-csv>, <data-folder>
python -m medvision_bm.benchmark.download_datasets --configs_csv <config-list-csv> --data_dir <data-folder>
```

<br/>
