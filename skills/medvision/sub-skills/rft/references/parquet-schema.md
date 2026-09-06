# verl parquet schema written by the MedVision RFT builders

Verified from `medvision_bm.rft.verl.verl_utils` (the six `_format_data_*_verl` formatters and
`prepare_dataset_for_verl*`) and by writing a mixed-task fixture with `datasets` 3.6.0 /
`pyarrow` and reading the schema back with `scripts/inspect_parquet_ds.py`.

## 1. Files and directory naming

Every builder writes under `--data_dir`:

```
<data_dir>/verl_datasets/<model_family_name>/
  ds__AD<a>_D<d>_TL<t>_all<total>[_wo-CoT-Instruct]__resized-hw-<H>x<W>   # or __original when --new_shape_hw is omitted
    train_verl.parquet
    validation_verl.parquet
    test_verl.parquet                 # only build_parquet_ds_with_testset*  (not consumed by the RFT recipes)
    shards/train_shard_NNNN.parquet   # only *__checkpointed builders (kept after the merge)
    checkpoint.json                   # only *__checkpointed builders
```

- `<a>`, `<d>`, `<t>` are the resolved **per-task train limits** (`parse_sample_limits`): the task-specific
  flag if > 0, else `--train_sample_limit_per_task`, and **0 when the task JSON is not given**; `-1` means "whole pool"
  and is printed literally (`ds__AD0_D0_TL-1_all-1__...`).
- `<total>` is `--train_sample_limit` (global post-concatenation cap, `-1` = none).
- `_wo-CoT-Instruct` appears only with `--without_cot_instruction`.
- Recipes of the verl fork expect these exact names, e.g. `ds__AD5500_D110000_TL5500_all121000__resized-hw-512x512`
  (see `rft-recipes.md`).
- There is **no output-directory flag**; `--prepared_ds_dir` is parsed but never read by any builder (verified from source).
- `checkpoint.json` keys: `completed_train_shards` (list of ints), `val_done`, `merged`, `n_shards`, `total_train`
  (+ `test_done` in the with-testset checkpointed variant).

## 2. Columns (all splits, all builders)

Only these seven columns survive `clean_dataset` (`keys_to_keep`); everything else from the MedVision
rows (image paths, masks, metadata) is dropped.

| Column | Arrow type (as written) | Content |
| --- | --- | --- |
| `prompt` | `list<struct<role: string, content: list<struct<type: string, text: string>>>>` | 2 chat messages: `system` (one text part = the RFT system prompt) and `user` (`{"type":"image"}` placeholder followed by one text part = the task prompt) |
| `images` | `list<struct<bytes: binary, path: string>>` | exactly **one** PNG per row, embedded as bytes (`path` is null); the HF `datasets` `Image` feature encoding |
| `ground_truth` | `string` | comma+space-joined values formatted `%.3f` (see §4) |
| `data_source` | `string` | `medvision-tl` / `medvision-ad` / `medvision-detection` |
| `ability` | `string` | `medvision-tl` / `medvision-angle` / `medvision-distance` / `medvision-detection` (the reward and the samplers route on this column) |
| `reward_model` | `struct<style: string, ground_truth: string>` | always `{"style": "rule", "ground_truth": <same string as ground_truth>}` |
| `extra_info` | `struct<...>` | ground-truth intermediate values for the process reward (see §3); **union of all tasks' fields** when tasks are mixed |

Arrow also stores a `huggingface` schema-metadata blob so `datasets.load_dataset("parquet", ...)` restores the
`Image` feature and decodes `images` back to PIL objects.

### Mixed tasks and `extra_info`

`build_parquet_ds.py` formats each task separately and then `concatenate_datasets([...])`. With `datasets` 3.6.0 the
differing `extra_info` structs are **aligned into one struct containing the union of all field names**; a row carries
`null` for the fields of the other tasks (verified with a T/L + A/D fixture). The checkpointed builders format mixed
shards through a `_task_tag` dispatch and reach the same layout. Consumers must therefore treat missing keys / `None`
values in `extra_info` as "not applicable", which is what the fork's reward `compute_score` does (it dispatches on
`ability`, injected into `extra_info` by the fork's reward managers).

## 3. Per-task values

Coordinates are **relative image positions** `[w, h]` in `[0, 1]` (width fraction first, then height fraction), taken
from the same CoT text formatters that generate the SFT prompts, so the origin convention matches the prompt text
(detection: origin at the image's lower-left corner; lower-left corner listed before upper-right).

| Task (`tag_ds` used to load) | `data_source` | `ability` | `ground_truth` | `extra_info` (CoT builders, default) | `extra_info` (`--without_cot_instruction`) |
| --- | --- | --- | --- | --- | --- |
| T/L size (`TumorLesionSize`) | `medvision-tl` | `medvision-tl` | `"<major>, <minor>"` in real-world units | `landmark_P1_wh`, `landmark_P2_wh` (major-axis endpoints), `landmark_P3_wh`, `landmark_P4_wh` (minor-axis endpoints) | `{"placeholder": true}` |
| A/D distance (`BiometricsFromLandmarks`) | `medvision-ad` | `medvision-distance` | `"<distance>"` | `metric_type: "distance"`, `landmark_1_wh`, `landmark_2_wh` | `{"metric_type": "distance"}` |
| A/D angle (`BiometricsFromLandmarks`) | `medvision-ad` | `medvision-angle` | `"<angle>"` | `metric_type: "angle"`, `line_1_point_1_wh`, `line_1_point_2_wh`, `line_2_point_1_wh`, `line_2_point_2_wh` | `{"metric_type": "angle"}` |
| Detection (`BoxSize`) | `medvision-detection` | `medvision-detection` | `"<x0>, <y0>, <x1>, <y1>"` relative, lower-left then upper-right | `lowerleft_corner_wh`, `upperright_corner_wh` (from the CoT formatter's values) | same two keys, built from the target itself |

- Every `*_wh` value is a 2-element `list<double>`.
- `ability` for A/D is derived from the sample's `metric_type`; any other value raises `ValueError("Unsupported metric_type")`.
- The lite (`--without_cot_instruction`) T/L and A/D rows carry **no landmark ground truth**, so the fork's process
  reward cannot be computed for them; the repository source marks the flag as deprecated because the paper keeps the
  CoT instruction in RFT to avoid a train/SFT prompt-distribution shift.

## 4. `ground_truth` formatting

`", ".join(f"{v:.3f}" for v in target)` -- three decimals, comma+space separated, no units, no brackets. Examples:
`"24.130, 11.070"` (T/L), `"52.100"` (A/D), `"0.100, 0.400, 0.600, 0.900"` (detection). `reward_model.ground_truth`
is the identical string.

## 5. `prompt` content

System message text (CoT builders) -- `SYSTEM_PROMPT` in `medvision_bm.rft.verl.rft_prompts` (787 characters):

```
A conversation between a User and an Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks through the reasoning process internally, then provides the User with the answer. The reasoning process and the final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. For example: <think> reasoning process here </think> <answer> answer here </answer>. Within the <think> </think> tags, report the reasoning process for each step inside <step-k-reasoning> </step-k-reasoning> tags, followed by the intermediate results in <step-k-answer> </step-k-answer> tags. For example: <think> <step-1-reasoning> reasoning for step 1 </step-1-reasoning> <step-1-answer> intermediate result from step 1 </step-1-answer> </think>.
```

`SYSTEM_PROMPT_LITE` (424 characters, `--without_cot_instruction`) is the first three sentences only (no
`<step-k-*>` instructions). The same `SYSTEM_PROMPT` is what `eval__medvision-model-rft --use_system_prompt` injects at
benchmark time, which is why that flag is mandatory when evaluating an RFT model.

User message text = the task prompt from the shared SFT/benchmark formatters (`Task:` / `Additional information:` /
`Format requirement:` / `Reasoning steps:` blocks). For T/L and A/D the `Additional information:` block states the
**image size and pixel size as perceived by `model_family_name`'s image processor** (via `get_resized_img_shape`),
after the `--new_shape_hw` resize. Detection prompts contain no pixel size and are family-independent.

## 6. `images`

`img_proccessor_nii2png_save2dataset` loads the NIfTI slice, resizes it to `--new_shape_hw` (H, W) when given,
PNG-encodes it in memory and stores a one-element list. Rows are written with `writer_batch_size=50` so each formatting
worker buffers at most 50 images (about 0.75 MB each at 512x512 RGB). Expect roughly 100 KB per row on disk at 512x512
(the checkpointed builder's own budget: 50 000 rows = about 5 GB per shard). The verl fork's `MedVisionDataset` reads
this column via `data.image_key=images` and does **no further resize**.

## 7. Quick validation

```
python scripts/inspect_parquet_ds.py --path <dataset_dir>            # schema, counts per data_source/ability, first row
python scripts/inspect_parquet_ds.py --path <dataset_dir> --json     # machine-readable; exit 2 if a verl column is missing
```
