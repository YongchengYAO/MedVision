![MedVision](fig/medvision-logo.png)

# About

This is the official codebase `medvision_bm` of the **MedVision** project. 

🌏 **Project**: [to be updated]

🧑🏻‍💻 **Code**: https://github.com/YongchengYAO/MedVision

🩻 **Huggingface Dataset**: [https://huggingface.co/datasets/YongchengYAO/MedVision](https://huggingface.co/datasets/YongchengYAO/MedVision)

🐳 **Docker**: https://hub.docker.com/r/vincentycyao/medvision/tags

<br/>

# 🔥 News

- [Oct 8, 2025] 🚀 Release **MedVision** dataset v1.0.0

<br/>

# 📜 TODO

- [ ] Add preprint and project page
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

# 🐳 Use Docker

📝 Docker images are built from these [dockerfiles](https://github.com/YongchengYAO/MedVision/tree/master/dockerfile)

1. Choose the docker image for a specific model: https://hub.docker.com/r/vincentycyao/medvision/tags
2. Map local volumes and GPUs

```bash
# NOTE: replace </path/to/local/Data>, </path/to/local/Results>, </path/to/local/SFT> </path/to/completed_tasks>, <tag>
docker run -it --rm \
	--gpus '"device=0,1"' \
	-v </path/to/local/Data>:/root/Documents/MedVision/Data \
	-v </path/to/local/Results>:/root/Documents/MedVision/Results \
	-v </path/to/local/SFT>:/root/Documents/MedVision/SFT \
	-v </path/to/completed_tasks>:/root/Documents/MedVision/completed_tasks \
	vincentycyao/medvision:<tag> \
	bash
```

```bash
# In the container
cd /root/Documents/MedVision
git pull
# Run the scripts in script/
```

[File structure](https://github.com/YongchengYAO/MedVision/tree/master/docs/file-structure.md): imaging data, benchmark results, and model checkpoints are automatically saved

<br/>

# 📊 Benchmark

- **[Usage]** 

  1. The scripts in `script/benchmark-*/eval__*` should be sufficient for dependencies installation, data processing, and benchmarking
  2. After evaluating all models in step 1, parse model outputs and calculate metrics (e.g., MRE, MAE, IoU, Success Rate):

  ```bash
  # args:
  # --task_type: ["AD", "TL", "Detection"]
  # --task_dir: task folder
  # --model_dir: model folder
  # --limit: limit sample size in the parsed files
  # --skip_existing: (store_ture arg) skip parsed files
  
  # example 1: parse all models for the T/L task 
  python -m medvision_bm.benchmark.parse_outputs --task_type TL --task_dir Results/MedVision-TL
  
  # example 2: parse one model for the detection task and skip existing parsed files
  python -m medvision_bm.benchmark.parse_outputs --task_type Detection --task_dir Results/MedVision-detect/Qwen2.5-VL-32B-Instruct --skip_existing
  ```

  File structure after 2 steps:

  ```
  ├── MedVision
  	├── completed_tasks 
  		├── completed_tasks_MedVision-AD.json           # <== tasks status tracker
  		├── ...
  	├── Results                                         # <== benchmark results
  		├── MedVision-AD
  		├── MedVision-detect
  			├── Qwen2.5-VL-32B-Instruct
  				├── parsed                                   # folder for parsed files
  				├── *.jsonl                                  # <== model outputs
  				├── *.json                                   # <== summary file
  		├── MedVision-TL
  ```

  

- **[Debug]** [here](https://github.com/YongchengYAO/MedVision/tree/master/docs/debug_env_setup.md)


<br/>

# 🎯 Training: SFT

- **[Usage]** The scripts in `script/sft-*/train__SFT__*` should be sufficient for dependencies installation, data processing, and training.
- **[Debug]** [here](https://github.com/YongchengYAO/MedVision/tree/master/docs/debug_env_setup.md)

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

```bash
# NOTE: replace <task-list-json>
python -m medvision_bm.benchmark.download_datasets --tasks_json <task-list-json>
```
or
```bash
# NOTE: replace <config-list-csv>
python -m medvision_bm.benchmark.download_datasets --configs_csv <config-list-csv>
```

<br/>