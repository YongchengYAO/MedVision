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

Three modules share the same CLI surface; pick by dataset size and intent:

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
| `--train_sample_limit` / `--val_sample_limit` | Global caps applied after the tasks are concatenated. Keep these equal to the sum of the per-task limits so nothing is silently truncated. |
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

The reward lives entirely on the verl fork; here it is enough to know its shape and what data feeds it. Conceptually it sums three parts:

1. **Format reward** — did the rollout produce a well-formed `<think>`/`<answer>` (and, for CoT, `<step-k-*>`) structure that can be parsed at all.
2. **Process reward** — how close each intermediate step is to ground truth. The CoT builders populate `extra_info` with the true intermediate coordinates so this can be scored: landmark pairs (`landmark_P1_wh`…`landmark_P4_wh`) for T/L, the two points or two lines for A/D, and the box corners for detection.
3. **Answer reward** — the final metric against `ground_truth`.

Each row also stamps a `data_source` / `ability` that routes it to the right reward function: `medvision-tl` for tumour/lesion size, `medvision-ad` (further split into `medvision-angle` / `medvision-distance` by metric type) for angle-distance, and `medvision-detection` for detection.

For the exact field contracts and the formatter functions that emit them, see the API reference for [`rft.verl.verl_utils`](../reference/api/verl_utils.md). For the GRPO configuration and the reward implementations, follow the [verl fork](https://github.com/YongchengYAO/verl/tree/medvision-rl).

## After training

Point the standard benchmark pipeline at the fine-tuned checkpoint, exactly as you would for any other model — see [Running evaluations](../benchmarking/running-evaluations.md). The `script/benchmark-*/` drivers include an `eval__MedVision-V0-7B__detect.sh` example for scoring an RFT'd detection model.
