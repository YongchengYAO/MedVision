# Benchmark pipeline overview

## What the benchmark actually measures

MedVision is not a visual question-answering benchmark in the usual sense. Every subtask asks a vision–language model to read a medical image and return a **number with physical meaning** — a bounding box in pixel coordinates, a tumour or lesion axis length in millimetres, or an anatomical angle in degrees. The score therefore reflects one thing: how close the model's reported quantity is to a ground-truth measurement derived from expert segmentations and landmarks.

Three task families are graded, each with its own annotation type and its own notion of "correct":

| Task | Annotation type | What the model outputs | Core metrics |
| --- | --- | --- | --- |
| Detection | `BoxCoordinate` | 4 box corner values | IoU, Precision, Recall, F1, SuccessRate |
| Tumour/Lesion size (TL) | `TumorLesionSize` | major + minor axis (mm) | MAE, MRE, nMAE, SuccessRate |
| Angle/Distance (AD) | `BiometricsFromLandmarks` | one scalar (degrees or mm) | MAE, MRE, nMAE (distances only), SuccessRate |

Because the answer is a physical quantity, a model can only score well if it (a) understands the image, (b) reasons about the geometry, and (c) converts pixels to physical units correctly.

## The three stages

Scoring a model is a fixed three-stage pipeline. Each stage is a `python -m` entry point, and later stages read the artefacts the earlier ones drop into `Results/<task_tag>/<model>/`.

**1. Evaluate.** The launcher scripts under `script/benchmark-*/` (`eval__*.sh`) invoke `python -m medvision_bm.benchmark.eval__<model>`, which runs inference over the sampled subtasks and writes one raw JSONL of prompts and responses per subtask. You typically only edit a few variables in the shell script (working directory, the model's Hugging Face id, a folder-name label, and batch size), then run it — see [Running evaluations](running-evaluations.md).

**2. Parse.** Raw model text is turned into structured predictions and per-sample metrics by `python -m medvision_bm.benchmark.parse_outputs --task_type {AD|TL|Detection} --task_dir <Results/...> -p <workers>`. This extracts the numeric answer from the `<answer></answer>` tags, compares it to ground truth, and stores the result in a `parsed/` subfolder next to the raw outputs.

**3. Summarize.** Per-model, per-subtask metrics are aggregated into leaderboard-style tables by the task-specific summarizer: `python -m medvision_bm.benchmark.summarize_AD_task`, `summarize_TL_task`, or `summarize_detection_task`, again with `--task_dir <Results/...> -p <workers>`. Add `--skip_model_wo_parsed_files` to ignore model folders that have not been through stage 2. Parsing and summarizing are covered in [Parsing and summarizing](parsing-and-summarizing.md).

:::{tip}
Stage 1 keeps a **crash-safe, per-sample resume cache**. As each sample finishes, its output is appended to `Results/<task_tag>/<model>/response_cache/<task>_rank<N>.jsonl`, so re-launching an interrupted run resumes where it stopped instead of recomputing finished samples — at most the single in-flight sample is redone. [The cache key hashes the prompt, so changing a prompt or config auto-invalidates stale entries.]{.mv-accent} Set `MEDVISION_RESP_CACHE=0` (not recommended) to switch the cache off and reproduce the plain no-cache behaviour.
:::

## Benchmark setting: sample caps

The two model classes run at different scales. Open-weight models are evaluated on up to **1000 samples per subtask**. Proprietary API models run as a smaller **pilot study capped at 100 samples per subtask**, keeping API cost bounded while still spanning every task family. The subtasks themselves are fixed by the task-list JSONs under `tasks_list/`, so the same anatomy/modality mix is scored for every model within a class.

## Metrics reference

All per-sample metrics are computed at parse time; the summarizers then aggregate them. A prediction is a **success** only if its text parses into exactly the expected number of values for the task — 4 for Detection, 2 for TL, 1 for AD.

[SuccessRate]{.mv-accent}. The fraction of samples whose prediction was parseable and had the right shape. This is the model's basic instruction-following / output-format score, independent of numeric accuracy, and every failed parse lowers it.

**Detection** — [IoU, Precision, Recall, F1]{.mv-accent}. These overlap metrics compare the predicted box to the ground-truth box. A prediction that fails to parse into 4 numbers is **scored as 0**, not dropped: the denominator is always the full sample count, so a model cannot inflate its overlap scores by emitting unparseable answers on hard cases. (The detection `avgMAE` on the raw corner values is also computed as a secondary diagnostic; there a failed parse is `NaN`-excluded, exactly the way the measurement tasks handle MAE below. Overlap is the primary signal.) The summarizer also emits a COCO-style hit-rate grid,
[Acc@IoU>=k]{.mv-accent} for k = 0.50, 0.55, … 0.95 and the swept mean
[Acc@IoU[0.50:0.95]]{.mv-accent} — the fraction of samples clearing each IoU threshold, again over the
full sample count, so a failed parse (IoU 0) counts as a miss at every threshold.

**TL / AD** — [MAE, MRE, and nMAE]{.mv-accent}. For the measurement tasks the error is reported in physical units:

- [MAE]{.mv-accent} (mean absolute error) is the average magnitude of `|prediction − ground truth|` — millimetres for TL axes and AD distances, degrees for AD angles.
- [MRE]{.mv-accent} (mean relative error) normalizes that error by the ground-truth magnitude, giving a dimensionless, scale-free measure of accuracy.
- TL additionally reports [nMAE]{.mv-accent} — the MAE divided by the image's diagonal length in physical space, `sqrt((H·pixel_size_h)² + (W·pixel_size_w)²)`. This expresses the error as a fraction of the whole slice's physical extent, making it comparable across scans of different sizes. nMAE is defined only for length targets (TL axes, and AD *distances*); angles have no nMAE, since dividing a degree error by a length is meaningless.

The failure-handling convention differs from Detection **by design** — and the two rules are consistent once you separate the metric types. *Higher-is-better overlap* metrics (IoU, Precision, Recall, F1) score a parse failure as **0**, because a non-answer genuinely deserves the worst-possible overlap and belongs in the average. *Error* metrics (MAE, MRE) must not: a failure has no sensible error value — 0 would be a *perfect* score and ∞ would swamp the mean — so an unparseable prediction is recorded as **NaN** and **excluded** from the averaged error. The miss is still penalized elsewhere: it lowers SuccessRate, and the threshold counts (`MRE<k`, `IoU>k`) use the full sample count as their denominator, so failures depress them for every task. In short, MAE/MRE answer "how accurate is the model *when it answers*", while SuccessRate answers "how often does it answer at all".

:::{note}
In the code and in each model's `<task_id>_results.json`, MAE, MRE, and nMAE may appear under the keys `avgMAE`, `avgMRE`, and `avgNMAE`. These are exactly the quantities this documentation and the paper call **MAE**, **MRE**, and **nMAE**.
:::

:::{warning}
For AD, samples whose ground-truth angle or distance is essentially zero are dropped before aggregation. A near-zero denominator would send MRE toward infinity and dominate the average. The cutoff is `AD_NEAR_ZERO_GT_THRESHOLD = 0.1` in `medvision_bm.utils.configs`; ground-truth values below it are filtered out of the AD metrics entirely.
:::
