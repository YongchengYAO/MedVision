# Analysis workflows

Every workflow here is **post-hoc**: it re-reads records the benchmark already produced and never re-runs a model.
All of them are CPU-only, network-free, and take seconds to minutes per model.

Placeholders used below: `${benchmark_dir}` = the folder holding `Results/`; `<data_dir>` = the folder holding the
downloaded datasets (`<data_dir>/Datasets/<dataset>/...`); `<task tag>` = one results subfolder such as an A/D, T/L or
Detection task directory; `<judge>` = an LLM-judge model tag.

**Shared preconditions.**

1. Step 2 of the benchmark (`parse_outputs`) has run, so each model folder has `parsed/` (or `llm-parsed_<judge>/` after
   the judge pass). Producing those is `../../results-parsing-and-metrics/SKILL.md` and `../../llm-judge-parsing/SKILL.md`.
2. `medvision_bm` is importable by the interpreter you use - except for CDA, which imports nothing from it.
3. Every analysis **writes beside its inputs**. Copy the tree first if the originals must stay untouched, and never run
   these against a results tree you cannot regenerate without checking what the outputs will overwrite.

Verify the environment first with `python ../../../scripts/check_medvision_env.py`.

---

## Which analysis answers which question

| The user asks | Run |
|---|---|
| "Would this error change the treatment decision?" / "kappa" / "clinical" / "stage" | [CDA](#1-clinical-decision-agreement-cda) |
| "Where in the chain of thought does it break?" / "are the landmarks or the arithmetic wrong?" | [process accuracy](#2-process-accuracy) |
| "Can it do the maths it wrote down?" / "is the formula right but the number wrong?" | [equation accuracy](#3-equation-accuracy) |
| "Is it worse on small targets?" / "how much of the IoU is just box size?" | [detection x target size](#4-detection-x-target-size) |
| "Does it actually use the reported pixel size?" | [scaledPS](#5-scaledps-ablation-reference-only) - the analysis half only |

---

## 1. Clinical Decision Agreement (CDA)

**Purpose.** Map each measurement (prediction and ground truth) through a published clinical cutoff table into a
category, and score agreement with Cohen's / quadratic-weighted kappa. Method, cutoff tables, kappa variants, the
volume-level bootstrap and every output field: `cda.md`.

**Inputs.** An A/D task directory and/or a T/L task directory, each with model folders containing `parsed/` (or an
`llm-parsed*` folder). Optionally `<data_dir>/Datasets` for the T/L removed-samples filter. Only three datasets carry a
proxy: Ceph-Biometrics-400 (SNA, SNB) and KiTS23 + KiPA22 (renal AJCC T category).

**Command.**

```bash
bash scripts/run_cda.sh \
  --ad-task-dir ${benchmark_dir}/Results/<AD task tag> --ad-config scripts/cda/config-AD-CoT.yaml \
  --tl-task-dir ${benchmark_dir}/Results/<TL task tag> --tl-config scripts/cda/config-TL-CoT.yaml \
  --removed-samples-dir <data_dir>/Datasets \
  --repo-root ${benchmark_dir} --out ${benchmark_dir}/CDA_REPORT.md
```

Add `--dry-run` first: it prints the five underlying commands and exits without touching anything. Before the first real
run, edit the two config YAMLs so their folder names match your results tree - a listed folder that is absent is a hard
`FileNotFoundError` (deliberately, so a stale config or an A/D-vs-T/L config mix-up fails loudly). Drop both
`--config` flags to analyse every subfolder with plain folder names instead.

**Outputs.** Per model, inside the parsed folder: `summary_metrics_CDA_Task*.json` and `summary_values_CDA_Task*.json`.
Per task: `summary_CDA_task*.txt` (leaderboards + confusion matrices) and `summary_CDA_uncertainty*.json` (CIs and
p-values). Plus the single Markdown report at `--out`.

**Interpretation.** Read the report's provenance table first (which parsed source, which sample set). Then, within one
proxy: `kappa` is the headline, `Acc` vs `AccCov` is the coverage gap, `Nparsed` vs `Ntotal` catches a model that
answered a handful of prompts, and the CI/`p` come from resampling **volumes**. Never compare kappa across proxies.
The `.txt` reports carry the per-model detail the Markdown leaves out.

### Variants

```bash
# score the LLM-judge re-parse instead (marker _llm-parsed-<judge> on task-level files)
bash scripts/run_cda.sh --ad-task-dir ... --tl-task-dir ... --ad-config ... --tl-config ... \
     --parsed-dirname llm-parsed_<judge> --out ${benchmark_dir}/CDA_REPORT_llm-parsed-<judge>.md

# one task only: agreement + uncertainty run, the Markdown report is skipped
bash scripts/run_cda.sh --tl-task-dir ${benchmark_dir}/Results/<TL task tag> --tl-config scripts/cda/config-TL-CoT.yaml

# fewer resamples while iterating (the default is 4000)
bash scripts/run_cda.sh ... --n-boot 500
```

`--parsed-dirname` must be identical across all three steps; the wrapper guarantees that. The prefix (`parsed` vs
`llm-parsed*`) also picks the prediction field, so a source and its field can never be mixed up.

---

## 2. Process accuracy

**Purpose.** Break one final measurement into its CoT steps and score each against ground truth - T/L in 4 steps
(major/minor axis endpoints by normalised L2, then major/minor axis length by MRE), A/D in 3 steps (two landmark
coordinates by normalised L2, then the scalar by MRE). Step definitions and output fields:
`process-and-equation-accuracy.md`.

**Inputs.** A T/L or A/D task/model directory with `parsed/`; **plus** the dataset landmark JSONs at the absolute
`doc["landmark_file"]` paths recorded at evaluation time; **plus** `medvision_ds` (mandatory for A/D, needed for the
per-label aggregation in T/L).

**Command.**

```bash
python scripts/analyze_process_accuracy_TL.py \
    --task_dir ${benchmark_dir}/Results/<TL task tag> \
    --removed_samples_dir <data_dir>/Datasets          # optional; matches the benchmark's filtered sample set

python scripts/analyze_process_accuracy_AD.py \
    --task_dir ${benchmark_dir}/Results/<AD task tag>

# a single model
python scripts/analyze_process_accuracy_TL.py --model_dir ${benchmark_dir}/Results/<TL task tag>/<model>
```

**Outputs.** `<stem>_proc_acc[_filtered].jsonl` beside each input; `summary_proc_acc_{TL,AD}_metrics*.json` in the
parsed folder; `summary_proc_acc_{TL,AD}_model*.txt` in the model folder; `summary_proc_acc_{TL,AD}_task*.txt` in the
task folder.

**Interpretation.** Compare the step columns for one model:

- large `step1/step2 normL2`, small `step3/step4 MRE` -> the model cannot find the structure but converts consistently;
  the final number is right for the wrong reason or wrong because of localisation.
- small `normL2`, large `MRE` -> localisation is fine and the failure is in the length computation or the pixel-size
  conversion; follow up with equation accuracy to see whether the arithmetic or the formula is at fault.
- low `success_rate` -> many responses did not emit the step tags at all; the step means then describe a self-selected
  subset. Check it before quoting a step mean.
- A/D: `n_ignored` counts near-zero-GT samples excluded from the step-3 average (`AD_NEAR_ZERO_GT_THRESHOLD = 0.1`).

---

## 3. Equation accuracy

**Purpose.** Arithmetic correctness with no ground truth involved: extract the equation the model wrote, evaluate it in
Python, and report `MRE(model's own answer, python evaluation)`. Details: `process-and-equation-accuracy.md`.

**Inputs.** A task/model directory with `parsed/`, or explicit JSONL paths. No images, no landmarks, no ground truth.
`medvision_ds` is needed only for the T/L per-label aggregation; A/D needs it not at all.

**Command.**

```bash
python scripts/analyze_equation_accuracy_TL.py --task_dir ${benchmark_dir}/Results/<TL task tag>
python scripts/analyze_equation_accuracy_AD.py --task_dir ${benchmark_dir}/Results/<AD task tag>

# per-sample records only, no aggregation and no medvision_ds - works on any folder
python scripts/analyze_equation_accuracy_TL.py \
    --jsonl "${benchmark_dir}/Results/<TL task tag>/<model>/llm-parsed_<judge>/*.jsonl"
```

**Outputs.** `<stem>_eq_acc[_filtered].jsonl` beside each input, plus the `summary_eq_acc_*` JSON/TXT files (aggregating
modes only).

**Interpretation.**

- `equation_MRE` near 0 across the board = the model evaluates what it writes; any measurement error is upstream
  (localisation or the formula), so read process accuracy next.
- `equation_MRE` large while process-accuracy steps 1/2 are good = a pure arithmetic failure.
- **A high `fail=` count matters as much as the mean.** Samples with no parseable equation contribute nothing to the
  mean, so a model that rarely writes a formula can post an excellent `equation_MRE` on a handful of samples. Always
  quote `n_valid` next to the mean.
- `step{k}_eval_error` rows used a function outside the evaluator's whitelist and are excluded, not scored 0.

---

## 4. Detection x target size

**Purpose.** Stratify detection metrics by box-to-image area ratio (5% bins) and compare against a random-box baseline.
Bin table, metric semantics, baseline construction and the two figures: `detection-target-size.md`.

**Inputs.** A Detection task/model directory whose parsed folder holds `*_BoxCoordinate_*.jsonl`; `medvision_bm` (and
`medvision_ds` — **both** analyzer variants resolve label names from the segmentation benchmark plans); `PyYAML`,
`matplotlib`, `pandas` for the figure.

**Command.**

```bash
bash scripts/detection_target_size.sh \
    --task-dir ${benchmark_dir}/Results/<Detection task tag> \
    --parsed-dirname parsed \
    --config scripts/config-detect-boxImgRatio.yaml \
    --out-dir ${benchmark_dir}/Figures/boxImgRatio \
    --skip-model-wo-parsed-files -p 8
```

Edit `scripts/config-detect-boxImgRatio.yaml` first so the folder names match your tree. `--dry-run` prints the two
commands; `--skip-viz` runs the metrics only; `--repo-root <checkout>` puts `<checkout>/src` on `PYTHONPATH` when
`medvision_bm` is not installed.

**Outputs.** `summary_metrics_per_boxImgRatio_detect_Task.json` + `summary_values_per_boxImgRatio_detect_Task.json` in
each parsed folder, `random_detection/` in the task folder (task-dir mode only), and
`<out-dir>/metrics_boxImgRatio-dotline.pdf`.

**Interpretation.** Compare each model against the random baseline **within a bin**, check `num_samples` per bin before
trusting an extreme point, and treat a curve that converges on the baseline in the large bins as evidence the model is
exploiting box size rather than localising.

---

## 5. scaledPS ablation (reference only)

**Purpose.** Test whether a model actually reasons from the *reported* pixel size: the prompt's `pixel_size` is
multiplied by a per-sample factor while the image pixels are unchanged, so a model that reads the number must scale its
answer.

**This sub-skill owns only the analysis half.** The repository's `script/analyze/scaled-pixel-size/` holds two
GPU evaluation launchers (a scaledPS variant of the T/L and A/D MedVision-V0 evals, pointed at the scaledPS task lists)
plus a wrapper that installs the eval stack and then calls the process-accuracy analyzers on the resulting
`...-scaledPS` result folders. Running the evaluation is `../../benchmark-evaluation/SKILL.md` (requires GPU).

Once those results exist, the analysis is the ordinary process-accuracy command pointed at the scaledPS results folder:

```bash
python scripts/analyze_process_accuracy_TL.py --model_dir ${benchmark_dir}/Results/<TL scaledPS task tag>/<model>
python scripts/analyze_process_accuracy_AD.py --model_dir ${benchmark_dir}/Results/<AD scaledPS task tag>/<model>
```

The analyzers detect a scaledPS file **by its name** (the substring `scaledPS`) and then score against the *scaled*
ground truth in the record's `target` field rather than the unscaled `biometric_profile` value, rebuilding a matching
scaled physical diagonal for the nMAE denominator. That denominator needs `_compute_physical_diagonal` from the vendored
eval utilities: run the analysis in an environment that has the full eval stack (`torch`, `transformers`, `nibabel`,
`medvision_ds`), or every scaledPS `nMAE` will be NaN while the MREs stay correct. The script says so once, loudly, on
stderr.

**Interpretation.** A model that ignores the reported pixel size keeps its unscaled answer, so its scaledPS step-3/4
MRE degrades roughly in proportion to the scale factor; a model that uses it holds its MRE near the unscaled run.

---

## Cross-cutting recipes

### Analysing the LLM-judge re-parse instead of the regex parse

| Analysis | How |
|---|---|
| CDA | `--parsed-dirname llm-parsed_<judge>` (prefix also selects `LLM_filtered_resps`) |
| detection x target size | `--parsed-dirname llm-parsed_<judge>` (outputs land in that folder) |
| process / equation accuracy | `--jsonl "<model>/llm-parsed_<judge>/*.jsonl"` - these two hardcode `parsed/` in `--task_dir`/`--model_dir` mode |

Process and equation accuracy read `resps` (the raw response), which the judge pass leaves untouched, so their
per-sample numbers are identical between `parsed/` and `llm-parsed_<judge>/` for the same sample. Only CDA and the
detection analyzers actually consume the parser's prediction field.

### Keeping a comparison honest

- Fix one parsed source, one sample-set choice (filtered or not) and one model roster for the whole comparison, and say
  which in any write-up. Filtered and unfiltered outputs sit side by side under different names precisely because they
  are not interchangeable.
- Do not mix models analysed with `--limit` and models analysed without it.
- Re-check `n`/`Nparsed`/`n_valid` on every row before quoting a mean; each of these analyses excludes samples for its
  own reasons.

### Dry-running before touching a results tree

Both wrappers support `--dry-run`, which prints the exact commands and exits 0 without reading or writing anything. Use
it to confirm which folders would be written into before a first run against a real tree.
