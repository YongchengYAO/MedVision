# Workflows: build parquet -> train in the verl fork -> evaluate

Three stages. Only stage A (and inspection) runs on a CPU host; stages B and C need GPUs.

## A. Build a verl-ready parquet dataset (CPU, RAM- and disk-heavy, downloads on first use)

### A0. Prerequisites (do once)

1. `medvision_bm` installed **with the SFT extras of the target family** -- the builders import
   `medvision_bm.sft.sft_utils` and load `--model_hf`'s image processor through transformers. The repository's
   launchers do this with `python -m medvision_bm.sft.env_setup --data_dir <data_dir> --requirement
   <requirements_sft_<family>.txt> --lmms_eval_opt_deps <extra>` (for `qwen25vl`: `requirements_sft_qwen25vl.txt`,
   extra `qwen2_5_vl`). Pins and install order: `../../environment-setup/SKILL.md`, `../../sft/SKILL.md`.
2. `medvision_ds` installed into `<data_dir>/src` (`mvbm install mvds -d <data_dir>`).
3. `export MedVision_DATA_DIR=<data_dir>` (the loader asserts it; `--data_dir` alone is not enough) and
   `export MedVision_PLANNER_VERSION=<version>` (`1.0.0` reproduces the paper's data; older-than-latest pins also need
   `MedVision_ACK_RELEASE`, see `../../dataset-and-tasks/SKILL.md`).
4. Task lists in the **SFT namespace** (`tasks_MedVision-{AD,TL,detect}__train_SFT.json`: keys like
   `<dataset>_BoxSize_Task01_Axial`, `_TumorLesionSize_`, `_BiometricsFromLandmarks_`; the loader appends `_Train`).
   Pool sizes recorded in the launchers: A/D 5 545, T/L 5 551, detection 2 695 205 rows.

### A1. Smoke build (minutes, small RAM)

```
MedVision_PLANNER_VERSION=1.0.0 bash scripts/build_parquet_ds.sh --data-dir <data_dir> \
    --tasks-ad <tasks_dir>/tasks_MedVision-AD__train_SFT.json \
    --tasks-tl <tasks_dir>/tasks_MedVision-TL__train_SFT.json \
    --tasks-detect <tasks_dir>/tasks_MedVision-detect__train_SFT.json \
    --dry-run                      # prints the python -m command + the output dir; drop --dry-run to build
```

Defaults: 100 train / 10 val per task, `--new-shape-hw 512 512`, `qwen25vl` / `Qwen/Qwen2.5-VL-7B-Instruct`,
2 concat workers, 8 format workers. Output: `<data_dir>/verl_datasets/qwen25vl/ds__AD100_D100_TL100_all300__resized-hw-512x512/`.
Validate:

```
python scripts/inspect_parquet_ds.py --path <data_dir>/verl_datasets/qwen25vl/ds__AD100_D100_TL100_all300__resized-hw-512x512
```

Expect 7 columns, `per ability` = `medvision-detection` 100 / `medvision-tl` 100 / `medvision-angle` +
`medvision-distance` = 100, images decoded as PNG 512x512, and the system prompt starting "A conversation between a
User and an Assistant".

### A2. Paper-scale datasets (what the repository launchers build)

| Launcher (repository `script/rft/`) | Tasks | Per-task train / val | Global `--train_sample_limit` / `--val_sample_limit` | Workers concat / format | Builder | Output dir name |
| --- | --- | --- | --- | --- | --- | --- |
| `...D0k-AD5.5k-TL0k__512x512.sh` | A/D | 5500 / 45 | 5500 / 45 | 2 / 32 | normal | `ds__AD5500_D0_TL0_all5500__resized-hw-512x512` |
| `...D0k-AD0k-TL5.5k__512x512.sh` | T/L | 5500 / 50 | 5500 / 50 | 2 / 32 | normal | `ds__AD0_D0_TL5500_all5500__resized-hw-512x512` |
| `...D110k-AD0k-TL0k__512x512.sh` | detection | 110000 / 105 | 110000 / 105 | 2 / 32 | normal | `ds__AD0_D110000_TL0_all110000__resized-hw-512x512` |
| `...D110k-AD5.5k-TL5.5k__512x512.sh` | all three | AD 5500/45, Det 110000/105, TL 5500/50 | 121000 / 200 | 2 / 32 | normal | `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512` |
| `...D1000k-AD0k-TL0k__512x512__checkpointed.sh` | detection | 1000000 / 500 | 1000000 / 500 | 16 / 256 | **checkpointed**, `PYTHONFAULTHANDLER=1`, `HF_DATASETS_VERBOSITY=warning` | `ds__AD0_D1000000_TL0_all1000000__resized-hw-512x512` |

All use `model_family_name=qwen25vl`, `model_hf=Qwen/Qwen2.5-VL-7B-Instruct`, `--new_shape_hw 512 512`, and set the
global cap **equal to the sum of the per-task caps** (a smaller global cap silently truncates; a larger one samples with
replacement). Equivalent wrapper call for the 121K set:

```
MedVision_PLANNER_VERSION=1.0.0 bash scripts/build_parquet_ds.sh --data-dir <data_dir> \
    --tasks-ad <...>/tasks_MedVision-AD__train_SFT.json     --train-limit-ad 5500     --val-limit-ad 45 \
    --tasks-detect <...>/tasks_MedVision-detect__train_SFT.json --train-limit-detect 110000 --val-limit-detect 105 \
    --tasks-tl <...>/tasks_MedVision-TL__train_SFT.json     --train-limit-tl 5500     --val-limit-tl 50 \
    --workers-concat 2 --workers-format 32
```

### A3. Large builds: checkpointed + sharded (>~100K rows, or any box with <= 64 GB RAM)

```
MedVision_PLANNER_VERSION=1.0.0 bash scripts/build_parquet_ds.sh --data-dir <data_dir> \
    --tasks-detect <...>/tasks_MedVision-detect__train_SFT.json --train-limit-detect 1000000 --val-limit-detect 500 \
    --checkpointed --shard-size 50000 --workers-concat 16 --workers-format 64
```

- RAM: the map phase buffers `workers x 50` PIL images (~0.75 MB each at 512x512) plus one shard in Arrow
  (~100 KB/row -> ~5 GB at 50 000). Source budget: 10-15 GB peak per shard at 64 workers. On a 32 GB box use
  `--workers-format 16 --shard-size 20000` (about 0.6 GB buffers + 2 GB shard) and check the cgroup limit, not `free -g`.
- Resume: rerun the identical command; `checkpoint.json` skips finished shards, adopts orphan shard files, then
  formats validation and stream-merges `shards/` into `train_verl.parquet`. Changing any limit or the resize changes the
  directory name and starts a fresh build.
- Disk: ~100 GB for 1M rows at 512x512, **twice** while `shards/` and the merged file coexist (shards are kept).
- The fork's stage-3 and multitask recipes read `shards/train_shard_*.parquet` directly when present, so the merge is
  optional for training.

### A4. Model-family rule

A parquet is valid **only for models sharing the image processor of `--model_family_name` / `--model_hf`**: the T/L
and A/D prompts state the image size and pixel size **as perceived after that family's resize**. Detection prompts have
no pixel size. To RFT another family (e.g. `gemma4`), rebuild with `--model-family gemma4 --model-hf <gemma4 id>`; the
output lands under `verl_datasets/gemma4/`. Supported keys: `cli-reference.md` §1.

## B. Train with GRPO in the verl fork (GPU; 4x H200 for the paper's settings)

Not part of `medvision_bm`; see `rft-recipes.md` for the exact recipes, rewards and hyper-parameters.

1. Clone the fork and check out `medvision-rl`; create its env with `bash setup_conda_verl.sh` (pinned stack; no
   system CUDA needed per the fork README).
2. Sequential MedVision-V0 pipeline (each stage's `merged_hf_model` feeds the next):
   ```
   DATASET_ROOT=<data_dir>/verl_datasets/qwen25vl/ds__AD5500_D0_TL0_all5500__resized-hw-512x512 \
   BASE_MODEL_PATH=<full-SFT-CoT-checkpoint> DRY_RUN=1 bash examples/grpo_trainer/train__rft-sequential__1-AD.sh   # preview
   # then DRY_RUN=0; stage 2 with ds__AD0_D0_TL5500_all5500__... and BASE_MODEL_PATH=<stage1>/global_step_N/actor/merged_hf_model
   # stage 3 with ds__AD0_D1000000_TL0_all1000000__... (shards/ accepted); MedVision-V0 = its global_step_250
   ```
3. Multi-task alternative: `train__rft-multitask.sh` on `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512`
   (T=8 mixing + curriculum; `+data.curriculum.enable=False` disables the curriculum); additive-reward ablation:
   `train__rft-multitask__additive-reward.sh`. Its default `BASE_MODEL_HF` is a private repo -- set `BASE_MODEL_PATH`
   or `BASE_MODEL_HF=<your SFT checkpoint or YongchengYAO/MedVision-V0-7B>`.
4. Give the base model as `BASE_MODEL_PATH` (local dir) or `BASE_MODEL_HF` (downloaded by the recipe first); never
   pass a Hub id straight to verl's `actor_rollout_ref.model.path`.
5. The recipe merges the last checkpoint: `python -m verl.model_merger merge --backend fsdp --local_dir
   <ckpt>/actor --target_dir <ckpt>/actor/merged_hf_model`. If you trained with LoRA, strip wrapper prefixes with
   `python -m medvision_bm.rft.verl.patch_layer_name --model_dir <merged_dir>`.

## C. Evaluate the RFT model on the benchmark (GPU + vLLM)

```
export MedVision_PLANNER_VERSION='1.0.0'         # the paper's dataset version
python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps medvision_v0
pip install -r <repo>/requirements/requirements_eval_medvision-v0.txt --no-deps   # pins vllm 0.10.0, torch 2.7.1, transformers 4.54.1 -- read ../../environment-setup first
python -m medvision_bm.benchmark.eval__medvision-model-rft \
    --skip_env_setup \
    --model_hf_id <merged_hf_model dir or YongchengYAO/MedVision-V0-7B> \
    --model_name MedVision-V0-7B \
    --results_dir <results_dir>/MedVision-detect-CoT \
    --data_dir <data_dir> \
    --tasks_list_json_path <tasks_dir>/tasks_MedVision-detect-CoT.json \
    --task_status_json_path <completed_tasks_dir>/completed_tasks_MedVision-detect-CoT.json \
    --batch_size_per_gpu 10 --gpu_memory_utilization 0.9 --max_new_tokens 4096 --sample_limit 1000 \
    --reshape_image_hw 512x512 \
    --use_system_prompt
```

Repeat with `tasks_MedVision-TL-CoT.json` / `tasks_MedVision-AD-CoT.json` (results dir + status file per task tag).
`--use_system_prompt` (injects the training `SYSTEM_PROMPT`) and `--reshape_image_hw 512x512` (the training resize)
are what make this evaluation match the RFT distribution; omit `--skip_env_setup` only if you want the script to install
its own pinned stack. Then parse and summarise as usual (`../../results-parsing-and-metrics/SKILL.md`):
`python -m medvision_bm.benchmark.parse_outputs ...` and `summarize_{detection,TL,AD}_task`.

## D. Where to stop

- No GPU: stop after A/inspection; describe B and C, never launch them.
- No network / no dataset download permission: the first build needs the HF datasets; `--dry-run` still works.
- Private Hub repos (`YongchengYAO/MedVision-V0-7B-dev0630`) need credentials you may not have; use the public
  `YongchengYAO/MedVision-V0-7B` or a local checkpoint.
