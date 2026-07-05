<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fig/medvision-logo-dark.svg">
    <img src="fig/medvision-logo.svg" alt="MedVision Logo" />
  </picture><br>

  # Dataset and Benchmark for *Quantitative* Medical Image Analysis

  | 🌏 [**Project**](https://medvision-vlm.github.io) | 🧑🏻‍💻 [**GitHub**](https://github.com/YongchengYAO/MedVision) | 📦 [**PyPI**](https://pypi.org/project/medvision-bm/) | 📚 [**Docs**](https://medvision.readthedocs.io/en/latest/index.html) | 🩻 [**Dataset**](https://huggingface.co/datasets/YongchengYAO/MedVision) | 🐳 [**Docker**](https://hub.docker.com/r/vincentycyao/medvision/tags) | 🤗 [**Models**](https://huggingface.co/collections/YongchengYAO/medvision-v0) | 🚀 [**Demo**](https://huggingface.co/spaces/YongchengYAO/MedVision-V0-demo) | 📖 [**arXiv**](https://arxiv.org/abs/2511.18676) |

  🔎 Benchmarking VLMs for medical vision tasks: detection and measurement 📏

  💿 30.8M annotated samples | multi-modality | multi-anatomy | 3D/2D medical image 💿

  📏 Annotation: segmentation mask | landmark coordinate | bounding box | tumor/lesion size | distance | angle 📏

  🎯 Post-training: SFT, RFT (RL), CoT, LoRA | Framework: TRL, verl 🎯

</div>


```
@misc{yao2026medvisionbenchmarkingquantitativemedical,
      title={MedVision: Benchmarking Quantitative Medical Image Analysis}, 
      author={Yongcheng Yao and Yongshuo Zong and Raman Dutt and Yongxin Yang and Sotirios A Tsaftaris and Timothy Hospedales},
      year={2026},
      eprint={2511.18676},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18676}, 
}
```

<br/>

# 🏆 Benchmarked Models

MedVision benchmarks **19 vision–language models** — open-weight general-purpose and medical VLMs plus proprietary API models — on detection, tumor/lesion size, and angle/distance measurement. The live **open leaderboard** (per-task score tables + a **frontier API-model pilot study**) lives on the [**project page**](https://medvision-vlm.github.io).

#### Ours

<p align="left">
<b><a href="https://huggingface.co/YongchengYAO/MedVision-V0-7B">MedVision-V0</a></b>
</p>

#### General-purpose VLMs (open-weight)

<p align="left">
<img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/qwen-color.png" height="20" alt="Qwen"/> <a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">Qwen2.5-VL</a>, <a href="https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking">Qwen3-VL-Thinking</a> &nbsp;·&nbsp; <img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/internlm-color.png" height="20" alt="InternVL"/> <a href="https://huggingface.co/OpenGVLab/InternVL3-38B">InternVL3</a> &nbsp;·&nbsp; <img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/gemma-color.png" height="20" alt="Gemma"/> <a href="https://huggingface.co/google/gemma-3-27b-it">Gemma-3</a>, <a href="https://huggingface.co/google/gemma-4-31B-it">Gemma-4</a> &nbsp;·&nbsp; <img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/meta-color.png" height="20" alt="Meta"/> <a href="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct">Llama-3.2-Vision</a> &nbsp;·&nbsp; <a href="https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf">LLaVA-OneVision</a> &nbsp;·&nbsp; <img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/glmv-color.png" height="20" alt="GLM"/> <a href="https://huggingface.co/zai-org/GLM-4.6V">GLM-4.6V</a>, <a href="https://huggingface.co/zai-org/GLM-4.6V-Flash">GLM-4.6V-Flash</a>
</p>

#### Medical VLMs (open-weight)

<p align="left">
<img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/gemma-color.png" height="20" alt="Gemma"/> <a href="https://huggingface.co/google/medgemma-4b-it">MedGemma</a> &nbsp;·&nbsp; <a href="https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b">LLaVA-Med</a>, <a href="https://huggingface.co/lingshu-medical-mllm/Lingshu-32B">Lingshu</a>, <a href="https://huggingface.co/Sunanhe/MedDr_0401">MedDr</a>, <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B">HuatuoGPT-Vision</a>, <a href="https://huggingface.co/lintw/HealthGPT-L14">HealthGPT-L14</a>
</p>

#### Proprietary / API

<p align="left">
<img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/claude-color.png" height="20" alt="Claude"/> Claude-Fable-5 &nbsp;·&nbsp; <picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/dark/openai.png"><img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/openai.png" height="20" alt="OpenAI"/></picture> GPT-5.5-Pro &nbsp;·&nbsp; <img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/gemini-color.png" height="20" alt="Gemini"/> Gemini-3.1-Pro &nbsp;·&nbsp; <img src="https://cdn.jsdelivr.net/npm/@lobehub/icons-static-png@1.91.0/light/kimi-color.png" height="20" alt="Kimi"/> Kimi-K2.6
</p>

> The project-page leaderboard currently publishes full score tables for the 12 off-the-shelf VLMs + MedVision-V0, plus a Claude-Fable-5 / Gemini-3.1-Pro API pilot on tumor/lesion size. Newer entries (Qwen3-VL-Thinking, Gemma-4, GLM-4.6V/-Flash, GPT-5.5-Pro, Kimi-K2.6) have eval scripts wired up and are being rolled into the leaderboard.

<br/>

# 🔥 News

- [Jul 4, 2026] Released the benchmarking/fine-tuning codebase `medvision_bm` v1.1.1 — [release notes](https://github.com/YongchengYAO/MedVision/blob/master/docs/codebase-release/release-v1.1.1.md)
  - 📚 New [documentation site](https://medvision.readthedocs.io/en/latest/index.html) on Read the Docs: installation, dataset, benchmarking, and fine-tuning guides plus the full CLI and Python API reference.
- [Jun 29, 2026] 🚀 Released the **MedVision** dataset (`medvision_ds`) v1.1.1 — [release notes](https://github.com/YongchengYAO/MedVision/tree/master/docs/dataset-release/release-v1.1.1.md)
  - **Highlight**: corrected T/L ellipse fit — fixes a transposed in-plane voxel-spacing bug (wrong axis lengths and major/minor labelling on anisotropic slices, e.g. sagittal/coronal); ~22% fewer T/L samples on anisotropic data, isotropic data (e.g., axial slices) essentially unchanged
  - **Backward compatibility**: The codebase `medvision_ds` will be automatically updated to the latest (v1.1.1). `MedVision_PLANNER_VERSION='latest'` now resolves to `'1.1.1'`; pin `'1.1.0'` or `'1.0.0'` for earlier annotations.
  - ⚠️ New env var `MedVision_ACK_RELEASE`: required **only** when you pin an older version (`MedVision_PLANNER_VERSION` below the latest) — **set it to the latest version (`1.1.1`) to acknowledge you have read this release note and unblock loading legacy data**. 
  - Always set `MedVision_FORCE_INSTALL_CODE='True'` to receive notification of future releases. See [Environment Variables](https://huggingface.co/datasets/YongchengYAO/MedVision#environment-variables).
- [Jun 9, 2026] Released [MedVision-V0](https://huggingface.co/collections/YongchengYAO/medvision-v0), [RFT code](https://github.com/YongchengYAO/verl/tree/medvision-rl), [preprint v2](https://arxiv.org/abs/2511.18676), [project page](https://medvision-vlm.github.io/) with interactive case viewer. 

<details>
<summary>Older news (Click to expand)</summary>
- [May 15, 2026] Released the benchmarking/fine-tuning codebase `medvision_bm` v1.1.0 — [release notes](https://github.com/YongchengYAO/MedVision/blob/master/docs/codebase-release/release-v1.1.0.md)

- [May 14, 2026] Released the **MedVision** dataset (`medvision_ds`) v1.1.0 — [release notes](https://github.com/YongchengYAO/MedVision/tree/master/docs/dataset-release/release-v1.1.0.md)
  - **Highlight**: new T/L sample filtering (with ambiguous cases removed), more T/L samples with a single small target (cluster size > 20)
  - **Backward compatibility**: The codebase `medvision_ds` will be automatically updated to the latest (v1.1.0). `MedVision_PLANNER_VERSION` is required (v1.1.0+) to specify the annotation data version. Setting `MedVision_PLANNER_VERSION='1.0.0'` will fall back to **MedVision** dataset v1.0.0.
  - 🧪 Test backward compatibility: 
  ```bash
  python unit-test/medvision-ds-planner-version/test_planner_switch_medvision_ds_v1.1.0.py --data_dir <local-data-folder>
  ```
- [Dec 10, 2025] Added preprint, training code, docker images, released models, new tasks/models guide
- [Oct 8, 2025] Released **MedVision** dataset v1.0.0

</details>

<br/>

# 🌟 Quick Start

> 📚 **Read the Docs:** [Installation](https://medvision.readthedocs.io/en/latest/getting-started/installation.html) · [Quickstart walkthrough](https://medvision.readthedocs.io/en/latest/getting-started/quickstart.html)

**Option 1 — run the full pipeline (benchmarking, SFT/RFT).** Clone the repo and install from the local copy. Use this when you rely on the repo's folder structure (e.g. `script/`, `tasks_list/`, `Results/`), since the scripts and configs live there.

```bash
git clone https://github.com/YongchengYAO/MedVision.git MedVision
cd MedVision
pip install .
pip show medvision_bm
```

**Option 2 — import the package in your own project.** Install `medvision_bm` from PyPI. Use this when you only want to `import` its modules/functions (e.g. `from medvision_bm.utils import parse_utils`) and do **not** need the repo's folder structure.

Stable release (PyPI):

```bash
pip install medvision-bm
pip show medvision_bm
```

Or the nightly build (latest commit on GitHub master):

```bash
pip install "git+https://github.com/YongchengYAO/MedVision.git"
pip show medvision_bm
```

<br/>

# 🐳 Use Docker

> 📚 **Read the Docs:** [Installation → Docker](https://medvision.readthedocs.io/en/latest/getting-started/installation.html#docker)

Docker images are built from these [dockerfiles](https://github.com/YongchengYAO/MedVision/tree/master/dockerfile)

1. Choose the docker image for a specific model: https://hub.docker.com/r/vincentycyao/medvision/tags

   ```bash
   docker pull vincentycyao/medvision:<tag>
   ```

2. Map local volumes and GPUs, then use the docker image `vincentycyao/medvision:<tag>`

   ```bash
   # NOTE: replace </path/to/working/folder>, <tag>
   docker run -it --rm \
       --gpus all \
       -v </path/to/working/folder>:/root/Documents/MedVision \
       vincentycyao/medvision:<tag> \
       bash
   ```

   ```bash
   # In the container
   git clone https://github.com/YongchengYAO/MedVision.git /root/Documents/MedVision
   cd /root/Documents/MedVision

   # Check existing Conda env and activate 
   conda env list
   conda activate <env-name>

   # Install the latest medvision_bm
   pip install .
   pip show medvision_bm

   # Install the latest medvision_ds
   python -m medvision_bm.benchmark.install_medvision_ds --data_dir ./Data
   pip show medvision_ds
   ```
> [!TIP]
> Treat the `MedVision` folder as the working directory for benchmarking and fine-tuning.
>
> [File structure](https://github.com/YongchengYAO/MedVision/tree/master/docs/file-structure.md): imaging data, benchmark results, and model checkpoints are automatically saved

<br/>

# 💿 Data

> 📚 **Read the Docs:** [Dataset concepts](https://medvision.readthedocs.io/en/latest/dataset/concepts.html) · [Loading data](https://medvision.readthedocs.io/en/latest/dataset/loading.html)

- **Dataset.** For the full description of the MedVision dataset (source datasets, modalities, anatomies, annotation types, and returned fields), see the [Hugging Face dataset repo](https://huggingface.co/datasets/YongchengYAO/MedVision).

- **Benchmark subtasks ↔ dataset subsets.** Each subtask in this benchmark links to a subset of the MedVision dataset. The per-subtask sample sizes are listed for each dataset version:
  - [`all_tasks__ds_v1.0.0`](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list/all_tasks__ds_v1.0.0)
  - [`all_tasks__ds_v1.1.0`](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list/all_tasks__ds_v1.1.0)
  - [`all_tasks__ds_v1.1.1`](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list/all_tasks__ds_v1.1.1)

- **Pixel size (physical spacing) distribution.** Because the quantitative tasks require pixel→mm arithmetic, the distribution of pixel sizes across subtasks is provided in [`pixel_sizes__ds_v1.0.0`](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list/pixel_sizes__ds_v1.0.0).

<br/>

# 📊 Benchmark

> 📚 **Read the Docs:** [Pipeline overview](https://medvision.readthedocs.io/en/latest/benchmarking/overview.html) · [Running evaluations](https://medvision.readthedocs.io/en/latest/benchmarking/running-evaluations.html) · [Parsing & summarizing](https://medvision.readthedocs.io/en/latest/benchmarking/parsing-and-summarizing.html) · [CLI reference](https://medvision.readthedocs.io/en/latest/reference/cli.html)

### Benchmark Setting
- Proprietary/API models are evaluated in a **pilot study** that caps each subtask at **100 samples**; all other (open-weight) models use a limit of **1000 samples** per subtask. 
- The subtasks are defined in [`tasks_list/`](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list): 
  - A/D → [`tasks_MedVision-AD-CoT.json`](https://github.com/YongchengYAO/MedVision/blob/master/tasks_list/tasks_MedVision-AD-CoT.json)
  - T/L → [`tasks_MedVision-TL-CoT.json`](https://github.com/YongchengYAO/MedVision/blob/master/tasks_list/tasks_MedVision-TL-CoT.json)
  - Detection → [`tasks_MedVision-detect-CoT.json`](https://github.com/YongchengYAO/MedVision/blob/master/tasks_list/tasks_MedVision-detect-CoT.json).

### Steps
1. The scripts in [`script/benchmark-*/`](https://github.com/YongchengYAO/MedVision/tree/master/script/) should be sufficient for dependency installation, data processing, and benchmarking

     > Set these variables:
     > - `benchmark_dir`: the working directory
     > - `model_hf_id`: Hugging Face ID (`<user>/<model>`) of the tested model
     > - `model_name`: user-defined identifier for the tested model, used as folder name in `Results/MedVision-*/`
     > - resource-constrained configs, such as `batch_size_per_gpu`

     > **Crash-safe resume.** 
     > 
     > During evaluation each finished output is written immediately to `Results/MedVision-*/<model_name>/response_cache/<task>_rank<N>.jsonl`, so re-running an interrupted eval skips already-completed samples instead of regenerating them — only the in-flight sample is lost. The cache key includes a hash of the prompt, so editing a prompt/config automatically invalidates stale entries (no need to clear the folder). Set the environment variable `MEDVISION_RESP_CACHE=0` to disable this layer entirely and reproduce the original (no-cache) behavior.

2. After evaluating all models in step 1, parse model outputs and calculate metrics (e.g., MRE, MAE, nMAE, IoU, F1, Precision, Recall, Success Rate). Base command:

     > Command:
     > python -m medvision_bm.benchmark.parse_outputs
     > 
     > Arguments:
     > - `--task_type`: one of `["AD", "TL", "Detection"]`
     > - `--task_dir`: task folder
     > - `--model_dir`: model folder
     > - `--limit`: limit sample size in the parsed files
     > - `--skip_existing`: (store_true) skip parsed files
     > - `--processes`, `-p`: number of processes
     > - `--rm_old`: remove existing `parsed` folder for each model

     Example 1 — parse all models for the T/L task:

     ```bash
     python -m medvision_bm.benchmark.parse_outputs \
     --task_type TL \
     --task_dir Results/MedVision-TL \
     -p 32
     ```

     Example 2 — parse all models for the A/D task (remove existing `parsed` folder):

     ```bash
     python -m medvision_bm.benchmark.parse_outputs \
     --task_type AD \
     --task_dir Results/MedVision-AD \
     -p 32 \
     --rm_old
     ```

     Example 3 — parse one model for the detection task and skip existing parsed files:

     ```bash
     python -m medvision_bm.benchmark.parse_outputs \
     --task_type Detection \
     --model_dir Results/MedVision-detect/Qwen2.5-VL-32B-Instruct \
     --skip_existing \
     -p 32
     ```

3. Summarize model performance for each task
  
      > If `medvision_ds` is missing, install with:
      >
      > python -m medvision_bm.benchmark.install_medvision_ds --data_dir <local-data-folder>

      > Command:
      >
      > python -m medvision_bm.benchmark.summarize_{AD,TL,detection}_task
      > 
      > Arguments:
      > - `--task_dir`: task folder
      > - `--model_dir`: model folder
      > - `--limit`: limit sample size in the parsed files
      > - `--skip_model_wo_parsed_files`: skip model directories that don't have a `parsed` folder
      > - `--processes`, `-p`: number of processes
      > - `--removed_samples_dir`: (TL task only) root directory with per-dataset removed_samples JSON files, used to filter ambiguous cases

      Example 1 — summarize all models for the A/D task:

      ```bash
      python -m medvision_bm.benchmark.summarize_AD_task \
      --task_dir Results/MedVision-AD \
      -p 32
      ```

      Example 2 — summarize all models for the T/L task:

      ```bash
      python -m medvision_bm.benchmark.summarize_TL_task \
      --task_dir Results/MedVision-TL \
      --removed_samples_dir <local-data-folder>/Datasets \
      -p 32

      ```

      Example 3 — summarize one model for the detection task:

      ```bash
      python -m medvision_bm.benchmark.summarize_detection_task \
      --model_dir Results/MedVision-detect/Qwen2.5-VL-32B-Instruct \
      -p 32
      ```

- **[File structure]** after steps 1-3

  ```text
  ├── MedVision
  │   ├── completed_tasks 
  │   │   ├── completed_tasks_MedVision-AD.json       # <== tasks status tracker
  │   │   ├── ...
  │   ├── Results                                     # <== benchmark results
  │   │   ├── MedVision-AD
  │   │   │   ├── ...
  │   │   │   ├── summary_AD_task.txt                 # <== [step 3] summary
  │   │   ├── MedVision-detect
  │   │   │   ├── Qwen2.5-VL-32B-Instruct
  │   │   │   │   ├── parsed                               
  │   │   │   │   │   ├── *.jsonl                     # <== [step 2] parsed model outputs
  │   │   │   │   │   ├── *.json                      # <== [step 2] parsed summary file
  │   │   │   │   │   ├── summary_*                   # <== [step 3] mean metrics, values
  │   │   │   │   ├── response_cache                  # <== [step 1] per-sample resume cache (auto; MEDVISION_RESP_CACHE=0 to disable)
  │   │   │   │   │   ├── *_rank*.jsonl               #        one line per finished sample, written as produced
  │   │   │   │   ├── *.jsonl                         # <== [step 1] model outputs
  │   │   │   │   ├── *.json                          # <== [step 1] summary file
  │   │   │   ├── ...
  │   │   │   ├── summary_detection_task.txt          # <== [step 3] summary
  │   │   ├── MedVision-TL
  │   │   │   ├── ...
  │   │   │   ├── summary_TL_task.txt                 # <== [step 3] summary
  ```


- **[Analysis & Visualization]** (optional) Scripts in [`script/visualization`](https://github.com/YongchengYAO/MedVision/tree/master/script/visualization):
  - **Radar charts** (`viz_radar.sh`, `viz_radar_batch.sh`): cross-model comparison across metrics.
  - **Detection label × box-size** (`viz_detection_sampleSize_per_label_x_boxSize.sh`): detection metrics and sample distribution per label × box-to-image ratio group.
  - **A/D landmark overlays** (`viz_ad_landmarks.sh`): per-sample GT vs. predicted landmarks and lines.
  - **A/D response panels** (`viz_ad_responses.sh`): per-sample prompt/response/GT panels.
  - **T/L axis overlays** (`viz_tl_axes.sh`): per-sample predicted vs. GT axes with mask contour.
  - **T/L response panels** (`viz_tl_responses.sh`): per-sample prompt/response/GT panels.
  - **Detection box overlays** (`viz_detection_boxes.sh`): per-sample GT vs. predicted bounding boxes.
  - **Detection response panels** (`viz_detection_responses.sh`): per-sample prompt/response/GT panels.
  - **Comparison grids** (`viz_compile_grid_batch.sh`): tile per-sample overlays across models.

- **[Analysis]** (optional) Scripts in [`script/analyze`](https://github.com/YongchengYAO/MedVision/tree/master/script/analyze):
  - **Process accuracy** (`process-accuracy/analyze_process_accuracy_TL.py`, `process-accuracy/analyze_process_accuracy_AD.py`): step-by-step CoT accuracy for T/L (4 steps: major/minor axis endpoint norm-L2 → axis length MRE) and A/D (3 steps: landmark coordinate norm-L2 → scalar MRE), evaluated against ground truth.
  - **Equation accuracy** (`equation-accuracy/analyze_equation_accuracy_TL.py`, `equation-accuracy/analyze_equation_accuracy_AD.py`): arithmetic correctness independent of ground truth — extracts the equation the model wrote, evaluates it in Python, and computes MRE between that result and the model's own reported answer.
  - **Detection × target size** (`detection--target-size/run_analysis.sh`): detection metrics (F1, IoU, etc.) stratified by box-to-image ratio, revealing performance trends across small, medium, and large targets.

- **[Troubleshooting]** [here](https://github.com/YongchengYAO/MedVision/tree/master/docs/debug_env_setup.md)

<br/>


# 🎯 Training: SFT

> 📚 **Read the Docs:** [Supervised fine-tuning (SFT)](https://medvision.readthedocs.io/en/latest/fine-tuning/sft.html)

- **[Script]** [`script/sft/train*.sh`](https://github.com/YongchengYAO/MedVision/tree/master/script/sft) handles dependency installation, data processing, and training.

  > Set these variables in the script:
  >
  > - `benchmark_dir`: the working directory
  > - `base_model_hf`: Hugging Face ID (`<user>/<model>`) of the base model, or the path to a local model folder.
  > - `run_name`: an identifier for the current training
  > - `merged_model_hf`: Hugging Face model name (`<model>`) of the merged model
  > - resource-constrained configs, such as
  >   - `per_device_train_batch_size`
  >   - `gradient_accumulation_steps`
  >   - `CUDA_VISIBLE_DEVICES=0,1,2,3` and `--num_processes=4`

- **[Blog]** [Supervised Fine-Tuning (SFT) for VLMs on Medical Image Data](https://huggingface.co/blog/YongchengYAO/medvision-sft-guide)

<br/>

# 🎯 Training: RFT

> 📚 **Read the Docs:** [Reinforcement fine-tuning (RFT)](https://medvision.readthedocs.io/en/latest/fine-tuning/rft.html)

RL fine-tuning uses the verl framework. MedVision provides **parquet dataset builders** that turn the MedVision tasks into verl-ready parquet datasets.

- **[Data Processing]** Build the verl parquet dataset with the scripts in [`script/rft`](https://github.com/YongchengYAO/MedVision/tree/master/script/rft), which call:
  - `medvision_bm.rft.verl.build_parquet_ds`: normal parquet dataset builder
  - `medvision_bm.rft.verl.build_parquet_ds__checkpointed`: checkpointed builder to avoid OOM, recommended for large datasets (e.g. ~1M detection samples)

  Available scripts:
  - `build_parquet_ds__verl__D0k-AD5.5k-TL0k__512x512.sh`: A/D task only (5.5K train / 45 val)
  - `build_parquet_ds__verl__D0k-AD0k-TL5.5k__512x512.sh`: T/L task only (5.5K train / 50 val)
  - `build_parquet_ds__verl__D110k-AD0k-TL0k__512x512.sh`: Detection task only (110K train / 105 val)
  - `build_parquet_ds__verl__D110k-AD5.5k-TL5.5k__512x512.sh`: all 3 tasks combined (121K train / 200 val)
  - `build_parquet_ds__verl__D1000k-AD0k-TL0k__512x512__checkpointed.sh`: Detection task only, large scale (1M train / 500 val); uses the checkpointed builder

- **[RFT]** RL fine-tuning in [https://github.com/YongchengYAO/verl/tree/medvision-rl](https://github.com/YongchengYAO/verl/tree/medvision-rl)

- **[Evaluation]** Evaluate the trained model with `eval__MedVision-V0-7B__detect.sh` (in `script/benchmark-*/`).


<br/>

# 📚 New Tasks/Models Guide

> 📚 **Read the Docs:** [Adding a new model](https://medvision.readthedocs.io/en/latest/extending/add-a-model.html) · [Adding a new task](https://medvision.readthedocs.io/en/latest/extending/add-a-task.html)

[New tasks guide](https://github.com/YongchengYAO/MedVision/blob/master/docs/New-Tasks-Guide.md) | [New models guide](https://github.com/YongchengYAO/MedVision/blob/master/docs/New-Models-Guide.md) 

## 🖼️ Model Image Processing

For the quantitative tasks (TL/AD), the image size and pixel size stated in each prompt must match the resolution the model's vision encoder actually perceives after its internal resize. [Model image processing](https://github.com/YongchengYAO/MedVision/blob/master/docs/Model-Image-Processing.md) documents the per-model strategy (fixed perceived size, dynamic processor probe, or API resize formula), with code references, validation status, and known caveats for every supported model.

<br/>

# 📖 Essential Dataset Concept

> 📚 **Read the Docs:** [Dataset concepts](https://medvision.readthedocs.io/en/latest/dataset/concepts.html)

We cover some essential concepts that help you use the MedVision dataset with ease.

## Concepts: Dataset & Data Configuration

- `MedVision`: the collection of public imaging data and our annotations
- `dataset`: name of the public datasets, such as `BraTS24`, `MSD`, `OAIZIB-CM`
- `data-config`: name of predefined subsets
  - naming convention: `{dataset}_{annotation-type}_{task-ID}_{slice}_{split}`
    - `dataset`: [details](https://huggingface.co/datasets/YongchengYAO/MedVision#datasets)
    - `annotation-type`: 
      - `BoxSize`: detection annotations (bounding box)
      - `TumorLesionSize`: tumor/lesion size annotations
      - `BiometricsFromLandmarks`: angle/distance annotations
      - `MaskSize`: area / mask-size annotations
    - `task-ID`: `Task[xx]` (Note: this is a local ID in the dataset, not a global ID in MedVision.)
      - For datasets with multiple image-mask pairs, we defined tasks in `medvision_ds/datasets/*/preprocess_*.py`
      - source: [medvision_ds](https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/src)
      - e.g., detection tasks for the `BraTS24` dataset are defined in the `benchmark_plan` in `medvision_ds/datasets/BraTS24/preprocess_detection.py`
    - `slice`: [`Sagittal`, `Coronal`, `Axial`]
    - `split`: [`Train`, `Test`]

<br/>

## What's returned from MedVision Dataset?

We only share the annotations (https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/Datasets). The data loading script [`MedVision.py`](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/MedVision.py) will handle raw image downloading and processing. The returned fields in each sample are defined as follows.

In `MedVision.py`, the class `MedVision(GeneratorBasedBuilder)` defines the feature dict and the method `_generate_examples()` builds the dataset.


<details>
<summary>Code block in `MedVision(GeneratorBasedBuilder)` (Click to expand)</summary>

  ``` python
  """
  MedVision dataset.

  NOTE: To update the features returned by the load_dataset() method, the followings should be updated:
          - the feature dict in this class 
          - the dict yielded by the _generate_examples() method 
  """

  # The feature dict for the task:
  # - Mask-Size
  features_dict_MaskSize = {
      "dataset_name": Value("string"),
      "taskID": Value("string"),
      "taskType": Value("string"),
      "image_file": Value("string"),
      "mask_file": Value("string"),
      "slice_dim": Value("uint8"),
      "slice_idx": Value("uint16"),
      "label": Value("uint16"),
      "image_size_2d": Sequence(Value("uint16"), length=2),
      "pixel_size": Sequence(Value("float16"), length=2),
      "image_size_3d": Sequence(Value("uint16"), length=3),
      "voxel_size": Sequence(Value("float16"), length=3),
      "pixel_count": Value("uint32"),
      "ROI_area": Value("float16"),
  }

  # The feature dict for the task:
  # - Box-Size
  features_dict_BoxSize = {
      "dataset_name": Value("string"),
      "taskID": Value("string"),
      "taskType": Value("string"),
      "image_file": Value("string"),
      "mask_file": Value("string"),
      "slice_dim": Value("uint8"),
      "slice_idx": Value("uint16"),
      "label": Value("uint16"),
      "image_size_2d": Sequence(Value("uint16"), length=2),
      "pixel_size": Sequence(Value("float16"), length=2),
      "image_size_3d": Sequence(Value("uint16"), length=3),
      "voxel_size": Sequence(Value("float16"), length=3),
      "bounding_boxes": Sequence(
          {
              "min_coords": Sequence(Value("uint16"), length=2),
              "max_coords": Sequence(Value("uint16"), length=2),
              "center_coords": Sequence(Value("uint16"), length=2),
              "dimensions": Sequence(Value("uint16"), length=2),
              "sizes": Sequence(Value("float16"), length=2),
          },
      ),
  }

  features_dict_BiometricsFromLandmarks = {
      "dataset_name": Value("string"),
      "taskID": Value("string"),
      "taskType": Value("string"),
      "image_file": Value("string"),
      "landmark_file": Value("string"),
      "slice_dim": Value("uint8"),
      "slice_idx": Value("uint16"),
      "image_size_2d": Sequence(Value("uint16"), length=2),
      "pixel_size": Sequence(Value("float16"), length=2),
      "image_size_3d": Sequence(Value("uint16"), length=3),
      "voxel_size": Sequence(Value("float16"), length=3),
      "biometric_profile": {
          "metric_type": Value("string"),
          "metric_map_name": Value("string"),
          "metric_key": Value("string"),
          "metric_value": Value("float16"),
          "metric_unit": Value("string"),
          "slice_dim": Value("uint8"),
      },
  }

  features_dict_TumorLesionSize = {
      "dataset_name": Value("string"),
      "taskID": Value("string"),
      "taskType": Value("string"),
      "image_file": Value("string"),
      "landmark_file": Value("string"),
      "mask_file": Value("string"),
      "slice_dim": Value("uint8"),
      "slice_idx": Value("uint16"),
      "label": Value("uint16"),
      "image_size_2d": Sequence(Value("uint16"), length=2),
      "pixel_size": Sequence(Value("float16"), length=2),
      "image_size_3d": Sequence(Value("uint16"), length=3),
      "voxel_size": Sequence(Value("float16"), length=3),
      "biometric_profile": Sequence(
          {
              "metric_type": Value("string"),
              "metric_map_name": Value("string"),
              "metric_key_major_axis": Value("string"),
              "metric_value_major_axis": Value("float16"),
              "metric_key_minor_axis": Value("string"),
              "metric_value_minor_axis": Value("float16"),
              "metric_unit": Value("string"),
          },
      ),
  }
  ```

</details>


<details>
<summary>Code block in `_generate_examples` (Click to expand)</summary>

  ```python
  # Task type: Mask-Size
  if taskType == "Mask-Size":
      flatten_slice_profiles = (
          MedVision_BenchmarkPlannerSegmentation.flatten_slice_profiles_2d
      )
      if imageSliceType.lower() == "sagittal":
          slice_dim = 0
      elif imageSliceType.lower() == "coronal":
          slice_dim = 1
      elif imageSliceType.lower() == "axial":
          slice_dim = 2
      slice_profile_flattened = flatten_slice_profiles(biometricData, slice_dim)
      for idx, case in enumerate(slice_profile_flattened):
          # Skip cases with a mask size smaller than 200 pixels
          if case["pixel_count"] < 200:
              continue
          else:
              yield idx, {
                  "dataset_name": dataset_name,
                  "taskID": taskID,
                  "taskType": taskType,
                  "image_file": os.path.join(dataset_dir, case["image_file"]),
                  "mask_file": os.path.join(dataset_dir, case["mask_file"]),
                  "slice_dim": case["slice_dim"],
                  "slice_idx": case["slice_idx"],
                  "label": case["label"],
                  "image_size_2d": case["image_size_2d"],
                  "pixel_size": case["pixel_size"],
                  "image_size_3d": case["image_size_3d"],
                  "voxel_size": case["voxel_size"],
                  "pixel_count": case["pixel_count"],
                  "ROI_area": case["ROI_area"],
              }

  # Task type: Box-Size
  if taskType == "Box-Size":
      if imageType.lower() == "2d":
          flatten_slice_profiles = (
              MedVision_BenchmarkPlannerDetection.flatten_slice_profiles_2d
          )
          if imageSliceType.lower() == "sagittal":
              slice_dim = 0
          elif imageSliceType.lower() == "coronal":
              slice_dim = 1
          elif imageSliceType.lower() == "axial":
              slice_dim = 2
          slice_profile_flattened = flatten_slice_profiles(
              biometricData, slice_dim
          )
          for idx, case in enumerate(slice_profile_flattened):
              # Skip cases with multiple bounding boxes in the same slice
              if len(case["bounding_boxes"]) > 1:
                  continue
              # Skip cases with a bounding box size smaller than 10 pixels in any dimension
              elif (
                  case["bounding_boxes"][0]["dimensions"][0] < 10
                  or case["bounding_boxes"][0]["dimensions"][1] < 10
              ):
                  continue
              else:
                  yield idx, {
                      "dataset_name": dataset_name,
                      "taskID": taskID,
                      "taskType": taskType,
                      "image_file": os.path.join(dataset_dir, case["image_file"]),
                      "mask_file": os.path.join(dataset_dir, case["mask_file"]),
                      "slice_dim": case["slice_dim"],
                      "slice_idx": case["slice_idx"],
                      "label": case["label"],
                      "image_size_2d": case["image_size_2d"],
                      "pixel_size": case["pixel_size"],
                      "image_size_3d": case["image_size_3d"],
                      "voxel_size": case["voxel_size"],
                      "bounding_boxes": case["bounding_boxes"],
                  }

  # Task type: Biometrics-From-Landmarks
  if taskType == "Biometrics-From-Landmarks":
      if imageType.lower() == "2d":
          flatten_slice_profiles = (
              MedVision_BenchmarkPlannerBiometry.flatten_slice_profiles_2d
          )
          if imageSliceType.lower() == "sagittal":
              slice_dim = 0
          elif imageSliceType.lower() == "coronal":
              slice_dim = 1
          elif imageSliceType.lower() == "axial":
              slice_dim = 2
          slice_profile_flattened = flatten_slice_profiles(
              biometricData, slice_dim
          )
          for idx, case in enumerate(slice_profile_flattened):
              yield idx, {
                  "dataset_name": dataset_name,
                  "taskID": taskID,
                  "taskType": taskType,
                  "image_file": os.path.join(dataset_dir, case["image_file"]),
                  "landmark_file": os.path.join(
                      dataset_dir, case["landmark_file"]
                  ),
                  "slice_dim": case["slice_dim"],
                  "slice_idx": case["slice_idx"],
                  "image_size_2d": case["image_size_2d"],
                  "pixel_size": case["pixel_size"],
                  "image_size_3d": case["image_size_3d"],
                  "voxel_size": case["voxel_size"],
                  "biometric_profile": case["biometric_profile"],
              }

  # Task type: Biometrics-From-Landmarks-Distance
  if taskType == "Biometrics-From-Landmarks-Distance":
      if imageType.lower() == "2d":
          flatten_slice_profiles = (
              MedVision_BenchmarkPlannerBiometry.flatten_slice_profiles_2d
          )
          if imageSliceType.lower() == "sagittal":
              slice_dim = 0
          elif imageSliceType.lower() == "coronal":
              slice_dim = 1
          elif imageSliceType.lower() == "axial":
              slice_dim = 2
          slice_profile_flattened = flatten_slice_profiles(
              biometricData, slice_dim
          )
          for idx, case in enumerate(slice_profile_flattened):
              if case["biometric_profile"]["metric_type"] == "distance":
                  yield idx, {
                      "dataset_name": dataset_name,
                      "taskID": taskID,
                      "taskType": taskType,
                      "image_file": os.path.join(dataset_dir, case["image_file"]),
                      "landmark_file": os.path.join(
                          dataset_dir, case["landmark_file"]
                      ),
                      "slice_dim": case["slice_dim"],
                      "slice_idx": case["slice_idx"],
                      "image_size_2d": case["image_size_2d"],
                      "pixel_size": case["pixel_size"],
                      "image_size_3d": case["image_size_3d"],
                      "voxel_size": case["voxel_size"],
                      "biometric_profile": case["biometric_profile"],
                  }

  # Task type: Biometrics-From-Landmarks-Angle
  if taskType == "Biometrics-From-Landmarks-Angle":
      if imageType.lower() == "2d":
          flatten_slice_profiles = (
              MedVision_BenchmarkPlannerBiometry.flatten_slice_profiles_2d
          )
          if imageSliceType.lower() == "sagittal":
              slice_dim = 0
          elif imageSliceType.lower() == "coronal":
              slice_dim = 1
          elif imageSliceType.lower() == "axial":
              slice_dim = 2
          slice_profile_flattened = flatten_slice_profiles(
              biometricData, slice_dim
          )
          for idx, case in enumerate(slice_profile_flattened):
              if case["biometric_profile"]["metric_type"] == "angle":
                  yield idx, {
                      "dataset_name": dataset_name,
                      "taskID": taskID,
                      "taskType": taskType,
                      "image_file": os.path.join(dataset_dir, case["image_file"]),
                      "landmark_file": os.path.join(
                          dataset_dir, case["landmark_file"]
                      ),
                      "slice_dim": case["slice_dim"],
                      "slice_idx": case["slice_idx"],
                      "image_size_2d": case["image_size_2d"],
                      "pixel_size": case["pixel_size"],
                      "image_size_3d": case["image_size_3d"],
                      "voxel_size": case["voxel_size"],
                      "biometric_profile": case["biometric_profile"],
                  }

  # Task type: Tumor-Lesion-Size
  if taskType == "Tumor-Lesion-Size":
      if imageType.lower() == "2d":
          # Get the target label for the task
          target_label = benchmark_plan["tasks"][int(taskID) - 1]["target_label"]

          flatten_slice_profiles = (
              MedVision_BenchmarkPlannerBiometry_fromSeg.flatten_slice_profiles_2d
          )
          if imageSliceType.lower() == "sagittal":
              slice_dim = 0
          elif imageSliceType.lower() == "coronal":
              slice_dim = 1
          elif imageSliceType.lower() == "axial":
              slice_dim = 2
          slice_profile_flattened = flatten_slice_profiles(
              biometricData, slice_dim
          )
          for idx, case in enumerate(slice_profile_flattened):
              n_total_clusters = case["n_total_clusters"]
              if n_total_clusters is not None:
                  # New JSON (v1.1.0+): filter on raw cluster count
                  if n_total_clusters > 1:
                      continue
              else:
                  # Old JSON (v1.0.0): fall back to above-threshold cluster count
                  if len(case["biometric_profile"]) > 1:
                      continue
              yield idx, {
                  "dataset_name": dataset_name,
                  "taskID": taskID,
                  "taskType": taskType,
                  "image_file": os.path.join(dataset_dir, case["image_file"]),
                  "mask_file": os.path.join(dataset_dir, case["mask_file"]),
                  "landmark_file": os.path.join(
                      dataset_dir, case["landmark_file"]
                  ),
                  "slice_dim": case["slice_dim"],
                  "slice_idx": case["slice_idx"],
                  "label": target_label,
                  "image_size_2d": case["image_size_2d"],
                  "pixel_size": case["pixel_size"],
                  "image_size_3d": case["image_size_3d"],
                  "voxel_size": case["voxel_size"],
                  "biometric_profile": case["biometric_profile"],
              }

  ```
</details>

<br/>

## Dataset Building Workflow

### Workflow

<img src="fig/medvision-dataset-flow.svg" alt="MedVision Dataset Building Workflow" /><br>
</br>

There are a few ways to control the dataset loading and building behavior:

- **Rebuild Dataset (Arrow files)**: Use the `download_mode` argument in `load_dataset()` ([docs](https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/builder_classes#datasets.DownloadMode)).
  - **[1]** Set `download_mode="force_redownload"` to ignore the cached Arrow files and trigger the data loading script `MedVision.py` to rebuild the dataset.
- **Redownload Raw Data**:
  - **[2]** `MedVision_FORCE_DOWNLOAD_DATA`: Set this environment variable to `True` to force re-downloading raw images and annotations.
  - **[3]** `.downloaded_datasets.json`: This tracker file records downloaded status. Removing a dataset's entry here will trigger a re-download of the raw data for that dataset.
  
> ⚠️ 
> **How to properly update/redownload raw data?**
>
> If you need to update raw data (images, masks, landmarks) using [2] or [3], you **MUST ALSO** use [1] (`download_mode="force_redownload"`).
>
> Why? Because if Hugging Face finds a valid cached dataset (Arrow files), it will load it directly and **skip running the script entirely**. Without running the script, the environment variable [2] or tracker file [3] will never be checked.
>
> **Summary:**
> - Update Arrow/Fields only: Use [1].
> - Update Raw Data: Use [1] **AND** ([2] or [3]).
>
> 🔥 We will maintain a [change log](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/changelog.md) for essential updates.

<br/>

### Examples

<details>
<summary> Running this for the first time will download the raw data and build the dataset </summary>

```python
import os
from datasets import load_dataset

# Set data folder
wd = os.path.join(os.getcwd(), "Data-testing")
os.makedirs(wd, exist_ok=True)
os.environ["MedVision_DATA_DIR"] = wd

# Pick a dataset config name and split
config = "OAIZIB-CM_BoxSize_Task01_Axial_Test"
split_name = "test" # use "test" for testing set config; use "train" for training set config 

# Get dataset
ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        trust_remote_code=True,
        split=split_name,
    )
```
</details>

<details>
<summary> Running the same script again will use the cached dataset </summary>

```python
import os
from datasets import load_dataset

# Set data folder
wd = os.path.join(os.getcwd(), "Data-testing")
os.makedirs(wd, exist_ok=True)
os.environ["MedVision_DATA_DIR"] = wd

# Pick a dataset config name and split
config = "OAIZIB-CM_BoxSize_Task01_Axial_Test"
split_name = "test" # use "test" for testing set config; use "train" for training set config 

# Get dataset
ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        trust_remote_code=True,
        split=split_name,
    )
```
</details>

<details>
<summary> Adding `download_mode="force_redownload"` will skip raw data downloading and rebuild the dataset </summary>

```python
import os
from datasets import load_dataset

# Set data folder
wd = os.path.join(os.getcwd(), "Data-testing")
os.makedirs(wd, exist_ok=True)
os.environ["MedVision_DATA_DIR"] = wd

# Pick a dataset config name and split
config = "OAIZIB-CM_BoxSize_Task01_Axial_Test"
split_name = "test" # use "test" for testing set config; use "train" for training set config 

# Get dataset
ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        trust_remote_code=True,
        split=split_name,
        download_mode="force_redownload",
    )
```
</details>

<details>
<summary> Adding `download_mode="force_redownload"` and `os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "True"` will redownload raw data and rebuild the dataset </summary>

```python
import os
from datasets import load_dataset

# Set data folder
wd = os.path.join(os.getcwd(), "Data-testing")
os.makedirs(wd, exist_ok=True)
os.environ["MedVision_DATA_DIR"] = wd

# Pick a dataset config name and split
config = "OAIZIB-CM_BoxSize_Task01_Axial_Test"
split_name = "test" # use "test" for testing set config; use "train" for training set config 

# Force redownload
os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "True"

# Get dataset
ds = load_dataset(
        "YongchengYAO/MedVision",
        name=config,
        trust_remote_code=True,
        split=split_name,
        download_mode="force_redownload",
    )
```
</details>

<br/>

## Download Mode in MedVision Dataset

<details>
<summary> (Advanced) Understand how the customized dataset loading script `MedVision.py` changes the behavior of `download_mode` in `load_dataset()` </summary>

- `download_mode` can be one of these: `"reuse_dataset_if_exists"` (default), `"reuse_cache_if_exists"`, `"force_redownload"`

- Default behavior of `download_mode` in `load_dataset()`:
    |                                   | Downloads | Dataset |
    | :-------------------------------- | :-------- | :------ |
    | reuse_dataset_if_exists (default) | Reuse     | Reuse   |
    | reuse_cache_if_exists             | Reuse     | Fresh   |
    | force_redownload                  | Fresh     | Fresh   |

- `download_mode` in MedVision dataset:
    |                                                        | Downloads | Dataset |
    | :----------------------------------------------------- | :-------- | :------ |
    | reuse_dataset_if_exists (default)                      | Reuse     | Reuse   |
    | reuse_cache_if_exists                                  | Reuse     | Fresh   |
    | force_redownload (MedVision_FORCE_DOWNLOAD_DATA=False) | Reuse     | Fresh   |
    | force_redownload (MedVision_FORCE_DOWNLOAD_DATA=True)  | Fresh     | Fresh   |
</details>

🔥 Summary: [Understanding the download mode of MedVision dataset](https://github.com/YongchengYAO/MedVision/issues/11)

<br/>

# 💿 Data Downloading (Optional)

> 📚 **Read the Docs:** [Loading data → batch download](https://medvision.readthedocs.io/en/latest/dataset/loading.html) · [CLI reference](https://medvision.readthedocs.io/en/latest/reference/cli.html)

Since data downloading and processing take time, you can download datasets from the [tasks list](https://github.com/YongchengYAO/MedVision/tree/master/tasks_list) or [configs list](https://github.com/YongchengYAO/MedVision/tree/master/docs/dataset-configs) in advance.


> [!NOTE]
> ⚠️ You need to set an API token for these datasets (see [detailed instructions](https://huggingface.co/datasets/YongchengYAO/MedVision#datasets)): FeTA24, SKM-TEA, and ToothFairy2

> Command: 
> 
> `python -m medvision_bm.benchmark.download_datasets`
> 
> Arguments:
> - `--data_dir`: (required) data folder
> - `--tasks_json`: task json file
> - `--configs_csv`: config csv file
> - `--force_download_data`: (store_true) force redownload raw imaging data
> - ⚠️ for debugging only; it will repeatedly download data for tasks/configs of the same dataset

Download from a task-list JSON (replace `<task-list-json>`, `<data-folder>`):

```bash
python -m medvision_bm.benchmark.download_datasets \
--tasks_json <task-list-json> \
--data_dir <data-folder>
```

Or from a configs CSV (replace `<config-list-csv>`, `<data-folder>`):

```bash
python -m medvision_bm.benchmark.download_datasets \
--configs_csv <config-list-csv> \
--data_dir <data-folder>
```

<br/>

# 📜 License

MedVision is released under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Users are permitted to utilize, adapt, and build upon this dataset for both academic and commercial purposes, provided that appropriate credit is given. MedVision is a meta-dataset built upon various publicly available source datasets. While the annotations provided by MedVision are covered by the CC-BY 4.0 license, any downstream application must continue to comply with the specific usage terms and licensing requirements stipulated by the curators of the original raw imaging data. It is the responsibility of the user to ensure that their application of this data aligns with the license agreements of all constituent source datasets.

<br/>

# 🩵 Acknowledgement

This work was supported by the United Kingdom Research and Innovation (grant EP/S02431X/1), UKRI Centre for Doctoral Training in Biomedical AI at the University of Edinburgh, School of Informatics.


MedVision is based on some open-source projects:
- [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): VLM evaluation framework
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): LLM evaluation framework
- [vllm-project/vllm](https://github.com/vllm-project/vllm): LLM/VLM inference
- [volcengine/verl](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning for LLMs

<br/>
