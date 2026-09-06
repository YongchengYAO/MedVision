# Detection x target size

**Question it answers:** does detection accuracy depend on how big the target is? A single mean IoU hides the fact that
a model may localise large organs well and miss small lesions entirely. This analysis re-bins every already-scored
detection sample by **box-to-image area ratio** and reports the metric curve across those bins, next to a random-box
baseline that shows what each bin gives away for free.

CPU only, no re-inference. The slow part is the random baseline (`RANDOM_BOX_SIMULATIONS = 100` simulated boxes per
ground-truth box).

Bundled entry point: `scripts/detection_target_size.sh` (metrics + figure) and `scripts/config-detect-boxImgRatio.yaml`
(the model list for the figure).

---

## 1. What is computed

### Binning

`box_img_ratio` is read from the parsed record when present, otherwise recomputed from the target box (relative
coordinates: `|x2-x1| * |y2-y1|`). Samples fall into **5%-wide bins**:

```
Box/Image < 5%   |  5% <= Box/Image < 10%  |  ...  |  85% <= Box/Image < 90%  |  Box/Image >= 90%
```

19 bins in total (18 explicit thresholds plus the `>= 90%` catch-all). The bin edges are a fixed table in
`medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random`; they are not configurable from the CLI.

### Per-bin metrics

Per bin the module re-aggregates the per-sample metrics that `parse_outputs` already wrote into
`*_BoxCoordinate_*.jsonl` (it recomputes nothing from raw responses): `avgMAE`, `IoU`, `F1`, `Precision`, `Recall`,
`SuccessRate`, `num_samples`. Failure handling is the benchmark's: MAE is NaN and excluded, while IoU/F1/Precision/Recall
count a failure as 0 - see `../../results-parsing-and-metrics/references/metrics.md` for the denominators.

`MINIMUM_GROUP_SIZE = 50` (in `medvision_bm.utils.configs`) does **not** apply here: it is used by
`summarize_detection_task` / `summarize_TL_task` to drop small anatomy groups from *their* averages. Box-ratio bins are
reported at whatever size they have, so read `num_samples` per bin before believing a spiky curve. If a bin looks
implausible, check its count first.

### Random-box baseline

In `--task-dir` mode the analyzer also writes a `random_detection/` folder **inside the task directory**. It reads the
ground truth from the first model folder that actually has the requested parsed subfolder (any model works - only GT is
used), draws `RANDOM_BOX_SIMULATIONS = 100` random boxes per GT box, and scores them with the same metric code. The RNG
is seeded from `medvision_bm.utils.configs.SEED = 1024`, and files are read in a timestamp-stripped sorted order so the
draw does not depend on which reference model was picked.

The baseline is what makes the curve interpretable: in the large-box bins a random box already overlaps the target, so
a high IoU there is not evidence of localisation. `summarize_detection_task` deliberately excludes the
`random_detection` folder from the ordinary leaderboard.

---

## 2. The two analyzer modules

| Module | Writes | Use it for |
|---|---|---|
| `medvision_bm.benchmark.analyze_detection_task_boxsize_vs_random` | `summary_values_per_boxImgRatio_detect_Task.json` + `summary_metrics_per_boxImgRatio_detect_Task.json` into each parsed folder, **plus** `random_detection/` in the task dir | the metric-vs-ratio curve and the baseline (what `scripts/detection_target_size.sh` runs) |
| `medvision_bm.benchmark.analyze_detection_task_boxsize` | the same two JSONs **plus** `summary_metrics_boxImgRatio_x_label_detect_Task.csv` (anatomy-grouped) and `summary_metrics_boxImgRatio_x_fineLabel_detect_Task.csv` (raw labels) | the label x box-size breakdown; no random baseline |

Both take `--task_dir` or `--model_dir`, `--parsed_dirname`, `--limit`, `--skip_model_wo_parsed_files` and
`--processes/-p`; `..._vs_random` additionally has `--ref_model_dir` + `--out_dir` to regenerate only the baseline.
Exact help text: `cli-reference.md`.

Outputs always land **inside the chosen `--parsed_dirname` folder**, so analysing `llm-parsed_<judge>/` never
overwrites the published `parsed/` summaries.

---

## 3. The two figures

| Figure module | Input it reads | Output |
|---|---|---|
| `medvision_bm.benchmark.viz_detection_performance_per_boxImgRatio` | `summary_metrics_per_boxImgRatio_detect_Task.json` per model | `metrics_boxImgRatio-dotline.pdf` - F1 / Precision / Recall vs box-to-image ratio, one line per model |
| `medvision_bm.benchmark.viz_detection_sampleSize_per_label_x_boxSize` | the two `..._x_{label,fineLabel}_detect_Task.csv` files | `fig_detection__metrics-boxSize__labelLevel.pdf` (`--label_level`, default) or `...__anatomyLevel.pdf` (`--anatomy_level`) - metrics and sample distribution per label x box-size group |

The two consume **different** artifacts, so run `analyze_detection_task_boxsize` first if you want the label-level
figure. The label figure suffixes its filename with `__<parsed_dirname>` for a non-default source; the ratio figure does
not, so give it a distinct `--out_dir` per source.

Config differences worth remembering:

- The **ratio** figure resolves `<in_dir>/<config key>/<metrics JSON>`, so the key must already include the parsed
  folder. `scripts/detection_target_size.sh` handles this: it rewrites every key except `random_detection` to
  `<folder>/<parsed_dirname>` into a temporary config before plotting.
- The **label** figure resolves `<in_dir>/<config key>/<parsed_dirname>/<csv>`, so its config keys stay plain folder
  names and the source is given with `--parsed_dirname`. It fails loudly when the CSVs are missing rather than
  silently plotting fewer models.

Both are catalogued with the other figure entry points in `../../../references/visualization-catalog.md`.

---

## 4. `config-detect-boxImgRatio.yaml`

```yaml
model_display_name:
  "<results folder name under the Detection task dir>": "<legend label>"
  ...
  "random_detection": "Random"
```

Order = legend order. Keep `random_detection` if you want the baseline curve. The bundled file is a **template listing
the repository's paper roster**; folder names are run-specific (bugfix suffixes, token-budget suffixes), so edit them to
match your own results tree. A key with no matching folder simply contributes no line to the ratio plot, which is why a
stale config shows up as a missing model rather than an error - check the legend against the folders you expect.

---

## 5. Recipe

```bash
# whole task tree: per-model ratio metrics + random baseline + figure
bash scripts/detection_target_size.sh \
    --task-dir ${benchmark_dir}/Results/<Detection task> \
    --parsed-dirname parsed \
    --config scripts/config-detect-boxImgRatio.yaml \
    --out-dir ${benchmark_dir}/Figures/boxImgRatio \
    --skip-model-wo-parsed-files -p 8

# one model, no baseline, no figure
bash scripts/detection_target_size.sh \
    --model-dir ${benchmark_dir}/Results/<Detection task>/<model> --skip-viz

# label x box-size CSVs, then the label-level figure
python -m medvision_bm.benchmark.analyze_detection_task_boxsize \
    --task_dir ${benchmark_dir}/Results/<Detection task> --parsed_dirname parsed \
    --skip_model_wo_parsed_files -p 8
python -m medvision_bm.benchmark.viz_detection_sampleSize_per_label_x_boxSize \
    --config <your config>.yaml --in_dir ${benchmark_dir}/Results/<Detection task> \
    --parsed_dirname parsed --out_dir ${benchmark_dir}/Figures/boxImgRatio --label_level
```

Add `--dry-run` to the wrapper to print both commands without running them. Use `--repo-root <checkout>` if
`medvision_bm` is not pip-installed for the interpreter you pass with `--python`.

## 6. Reading the result

- Compare each model's curve **against the random baseline in the same bin**, never against its own overall mean.
- A curve that rises monotonically with box size and converges on the baseline in the large bins means the model is
  mostly exploiting box size, not localising.
- Check `num_samples` per bin in `summary_metrics_per_boxImgRatio_detect_Task.json` before quoting the extremes; the
  smallest and largest bins are usually thin.
- `SuccessRate` per bin separates "answered badly" from "did not answer in the required format"; a low `F1` with a low
  `SuccessRate` is a formatting problem first (see `../../llm-judge-parsing/SKILL.md` for the format-robust re-parse).
