# Reinforcement fine-tuning (RFT)

RFT closes the loop after [supervised fine-tuning](sft.md): instead of imitating chain-of-thought traces, the model is optimised directly against a reward computed from the geometry of its predictions. MedVision runs this with GRPO (group-relative policy optimisation).

## Who does what

The work is split across two repositories, and it helps to keep them straight:

- **This package (`medvision_bm`) builds the data.** It converts MedVision tasks into verl-ready Parquet datasets: rendered images, chat-formatted prompts, ground-truth answers, and the per-sample metadata the reward functions need.
- **The [verl fork](https://github.com/YongchengYAO/verl/tree/medvision-rl) runs the training.** The GRPO loop, the rollout/actor machinery, and the reward functions themselves all live on the `medvision-rl` branch. `medvision_bm` never launches training — it only produces the Parquet files that the fork consumes.

:::{note}
The reward functions on the fork (under `verl/utils/reward_score/medvision_rewards/`) are the consumers of the metadata built here. When this page mentions a field like `landmark_1_wh` or a `data_source` such as `medvision-detection`, that is the contract between the two repos.
:::

## Building the Parquet dataset

The entry point is a plain argparse module you invoke with `python -m`:

```bash
python -m medvision_bm.rft.verl.build_parquet_ds \
    --model_family_name qwen25vl \
    --model_hf "Qwen/Qwen2.5-VL-7B-Instruct" \
    --data_dir "$MedVision_DATA_DIR" \
    --tasks_list_json_path_AD     tasks_list/tasks_MedVision-AD__train_SFT.json \
    --tasks_list_json_path_detect tasks_list/tasks_MedVision-detect__train_SFT.json \
    --tasks_list_json_path_TL     tasks_list/tasks_MedVision-TL__train_SFT.json \
    --train_sample_limit_task_AD        5500 --val_sample_limit_task_AD        45 \
    --train_sample_limit_task_Detection 110000 --val_sample_limit_task_Detection 105 \
    --train_sample_limit_task_TL        5500 --val_sample_limit_task_TL        50 \
    --train_sample_limit 121000 --val_sample_limit 200 \
    --new_shape_hw 512 512
```

The `script/rft/` directory ships ready-to-run wrappers around this call — each one pins a conda env, rebuilds the wheel, runs `medvision_bm.sft.env_setup`, and then invokes the builder with a fixed task mix:

| Script | Task mix (train / val) |
| --- | --- |
| `build_parquet_ds__verl__D0k-AD5.5k-TL0k__512x512.sh` | A/D only (5.5K / 45) |
| `build_parquet_ds__verl__D0k-AD0k-TL5.5k__512x512.sh` | T/L only (5.5K / 50) |
| `build_parquet_ds__verl__D110k-AD0k-TL0k__512x512.sh` | Detection only (110K / 105) |
| `build_parquet_ds__verl__D110k-AD5.5k-TL5.5k__512x512.sh` | all three combined (121K / 200) |
| `build_parquet_ds__verl__D1000k-AD0k-TL0k__512x512__checkpointed.sh` | Detection at scale (1M / 500) |

### Builder variants

Four modules share a common CLI core — the `__checkpointed` variants add `--shard_size`, and the
`_with_testset` variants add `--test_sample_limit*` and `--train_sample_limit_per_subset`. Pick by
dataset size and intent:

- `medvision_bm.rft.verl.build_parquet_ds` — the default. Loads, formats, and writes each task in one pass.
- `medvision_bm.rft.verl.build_parquet_ds__checkpointed` — writes intermediate shards so a run can survive an out-of-memory kill and resume. Use it for the very large detection splits (the ~1M-sample script above relies on it).
- `medvision_bm.rft.verl.build_parquet_ds_with_testset` (and its `__checkpointed` twin) — also carves out a held-out test split. GRPO training itself only reads `train` and `validation`; the test split exists for debugging and offline inspection.

### Key flags

| Flag | Meaning |
| --- | --- |
| `--data_dir` | Root data folder (mirror of `MedVision_DATA_DIR`); output lands under `<data_dir>/verl_datasets/<model_family_name>/`. |
| `--model_family_name` | Image-processor group (e.g. `qwen25vl`). Determines how images are resized. |
| `--model_hf` | Hugging Face id used to load that processor. |
| `--tasks_list_json_path_AD` / `_detect` / `_TL` | One task-list JSON per task type. Supply one, two, or all three; missing ones are skipped. |
| `--train_sample_limit_task_AD` / `_Detection` / `_TL` (+ `val_` twins) | Per-task caps applied while each task is loaded (`-1` = no limit). |
| `--train_sample_limit` / `--val_sample_limit` | Global caps applied after the tasks are concatenated. Keep these equal to the sum of the per-task limits: a smaller value silently truncates, and a **larger** one silently oversamples **with replacement** (seeded bootstrap from `SEED`). |
| `--shard_size` | Checkpointed builders only (default `50000`). Training rows per shard — the main lever on peak RAM. |
| `--new_shape_hw H W` | Resize target as height then width (e.g. `512 512`); omit to keep native resolution. |
| `--without_cot_instruction` | Emit the lite (no reasoning-format) prompt. **Deprecated** — the intended pipeline is SFT-CoT followed by RFT, so keeping the CoT instruction avoids a train-time distribution shift. |

The builder writes `train_verl.parquet` and `validation_verl.parquet`. Each row carries the fields verl expects — `prompt`, `images`, `ground_truth`, `data_source`, `ability`, `reward_model`, and `extra_info`.

:::{warning}
A built dataset is tied to the `model_family_name` it was made for: the image resize ratio and final pixel dimensions come from that model's processor. Reusing a Parquet file with a different model family will feed it mispreprocessed images and mismatched prompts. The output directory name encodes the model and sample limits so caches can't collide.
:::

## The reasoning-format prompt

Every sample's system message tells the model to separate its reasoning from its final answer. Two variants live in [`rft_prompts.py`](../reference/api/rft_prompts.md):

- **`SYSTEM_PROMPT`** (the CoT builders) asks for reasoning inside `<think>...</think>` and the result inside `<answer>...</answer>`, and additionally requires each intermediate step to be wrapped as `<step-k-reasoning>` / `<step-k-answer>` pairs. That per-step structure is what makes a process reward possible — the grader can read off each intermediate landmark.
- **`SYSTEM_PROMPT_LITE`** *(deprecated)* — the `--without_cot_instruction` path; keeps only the outer `<think>` / `<answer>` split, with no per-step tags.

So a CoT rollout looks like:

```text
<think>
  <step-1-reasoning> ... </step-1-reasoning>
  <step-1-answer> ... </step-1-answer>
  ...
</think>
<answer> 12.480, 7.930 </answer>
```

## How the reward is composed

The reward lives entirely on the verl fork (`verl/utils/reward_score/medvision_rewards/`); here it is enough to know its shape and what data feeds it. Each rollout is scored by up to three components in [0, 1]. The two accuracy components map an error *e* to a reward with ρ(*e*) = exp(−*e*), so an exact prediction earns 1 and an unparseable value earns 0:

1. **Format reward** — the soft variant (default) combines a reasoning-structure score for the `<think>` / `<step-k-*>` blocks (weight 0.8) with a binary check of the final `<answer>` (weight 0.2); the binary variant uses the answer check alone. Detection, which has no CoT steps, always uses the binary check.
2. **Process reward** (T/L and A/D only) — the mean over the CoT steps of ρ(step error). Localization steps are scored by the displacement of their worst-localized point in relative coordinates (divided by √2), measurement steps by their relative error. The CoT builders populate `extra_info` with the true intermediate values so this can be scored: landmark pairs (`landmark_P1_wh`…`landmark_P4_wh`) for T/L, and the line endpoints or points for A/D.
3. **Answer reward** — ρ of the mean relative error of the final values (T/L, A/D), or of the overlap error (1 − CIoU)/2 of the predicted box (detection).

For T/L and A/D the components are combined **multiplicatively** by default, `r = r_format + r_process * r_answer`, so the final-answer credit is conditional on an accurate reasoning chain; detection has no process reward and reduces to `r = r_format + r_answer`. The paper's reward-design ablation uses the additive alternative `r = r_format + r_process + r_answer`. All of this is one entry point on the fork, `medvision_general.compute_score`, configured through `custom_reward_function.reward_kwargs` (`format_reward`, `composition`); the fork's [`REWARDS.md`](https://github.com/YongchengYAO/verl/blob/medvision-rl/REWARDS.md) documents the options and the CLI overrides.

Each row also stamps a `data_source` / `ability` that routes it to the right reward function: `medvision-tl` for tumour/lesion size, `medvision-ad` (further split into `medvision-angle` / `medvision-distance` by metric type) for angle-distance, and `medvision-detection` for detection.

## Multi-task RFT: task mixing and curriculum learning

MedVision-V0 is trained with three sequential RFT stages (A/D → T/L → detection). The fork also implements the paper's single-stage **multi-task RFT** over the 121K mixture, with two additions:

- **Temperature-scaled task mixing** (`verl/utils/dataset/temperature_sampler.py`): a task with *N* samples is drawn with probability proportional to *N*^(1/*T*), sampling with replacement; *T* = 8 gives roughly 42 % / 29 % / 29 % of draws to detection / T/L / A/D instead of the natural 91 % / 4.5 % / 4.5 %.
- **Epoch-level curriculum learning** (`verl/utils/dataset/curriculum.py`): samples the policy reliably solves — EMA answer error below the benchmark's own threshold (MRE < 0.1; overlap error < 0.25, i.e. IoU > 0.5, for detection) — are moved to a per-task easy pool, with a retention mix-in, rotating audits, hysteresis-guarded demotion and a per-task floor. The algorithm, its configuration, and the mapping from the paper's symbols to config options are documented in [`CURRICULUM_FILTERING.md`](https://github.com/YongchengYAO/verl/blob/medvision-rl/CURRICULUM_FILTERING.md).

## Recipes on the fork

| Recipe (`examples/grpo_trainer/`) | Setting in the paper |
| --- | --- |
| `train__rft-sequential__1-AD.sh`, `…__2-TL.sh`, `…__3-detection.sh` (run in order, each from the previous stage's checkpoint) | MedVision-V0: sequential A/D → T/L → detection RFT |
| `train__rft-multitask.sh` | multi-task RFT ablation (task mixing + curriculum) |
| `train__rft-multitask__additive-reward.sh` | additive-reward ablation |

Each recipe takes the dataset directory from `DATASET_ROOT` and the base model from either `BASE_MODEL_PATH` (a local checkpoint) or `BASE_MODEL_HF` (a Hub id, downloaded locally before training). For the exact field contracts and the formatter functions that emit them, see the API reference for [`rft.verl.verl_utils`](../reference/api/verl_utils.md).

## After training

Point the standard benchmark pipeline at the fine-tuned checkpoint, exactly as you would for any other model — see [Running evaluations](../benchmarking/running-evaluations.md). The `script/benchmark-*/` drivers include an `eval__MedVision-V0-7B__detect.sh` example for scoring an RFT'd detection model.

A verl checkpoint saved through PEFT/LoRA still carries wrapper prefixes in its safetensors keys. Strip them first:

```bash
python -m medvision_bm.rft.verl.patch_layer_name --model_dir <ckpt> [--push_to_hub --repo_id <id>]
```
