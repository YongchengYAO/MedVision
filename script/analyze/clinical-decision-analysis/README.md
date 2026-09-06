# Clinical Decision Agreement (CDA)

**Does the model's measurement error change the clinical decision?**

MedVision normally scores a model on how far its measurement is from the truth
(MAE, MRE, IoU). That tells you the size of the error but not whether the error
*matters*. A 3 mm miss on a 90 mm tumor is noise; the same 3 mm miss on a tumor
sitting at 40 mm moves it from stage T1a to T1b and changes the treatment plan.

CDA re-scores existing benchmark outputs through that lens. It takes each
measurement, applies a published clinical cutoff table to turn it into a
category (a stage, a skeletal class), and asks whether the model's number lands
in the **same category** as the ground-truth number. The headline statistic is
Cohen's kappa — agreement above what you would get by chance.

Nothing is re-run: CDA only reads the parsed outputs already in
`Results/<task>/<model>/parsed/*.jsonl`. No GPU, no inference, seconds per model.

## Quick start

From the repo root:

```bash
REMOVED_SAMPLES_DIR=$PWD/Data/Datasets bash script/analyze/clinical-decision-analysis/run_CDA_analysis.sh
```

That runs everything over the canonical result directories and writes the
reports listed at the end of its output. The headline result is `CDA_REPORT.md`
in this folder — every leaderboard in one Markdown file. To run a single piece,
see [Running it yourself](#running-it-yourself); to score the LLM-judge re-parse
instead, see [Scoring the LLM-judge re-parse](#scoring-the-llm-judge-re-parse).

`REMOVED_SAMPLES_DIR` makes CDA score the **same sample set as the T/L
benchmark** by dropping multi-cluster slices — see
[Matching the benchmark's sample set](#matching-the-benchmarks-sample-set).
Omit it and you get the unfiltered numbers instead, under filenames without the
`_filtered` marker; the two are not interchangeable, so pick one and stay with
it. Note the variable must point at a real directory: a path that does not exist
silently yields *unfiltered* numbers in `_filtered`-named files.

It affects the **T/L task only**. The exclusion list marks slices whose mask has
more than one connected component, and A/D measurements come from landmarks, not
masks — so there is no such slice, and A/D outputs never carry a `_filtered`
marker.

### Scoring the LLM-judge re-parse instead

Each model folder can hold several sets of parsed results: `parsed/` from the
regex parser, plus one `llm-parsed_<judge>/` per LLM-judge re-parse. Pick one
with `CDA_PARSED_DIR`:

```bash
REMOVED_SAMPLES_DIR=$PWD/Data/Datasets CDA_PARSED_DIR=llm-parsed_gemma-4-31b \
  bash script/analyze/clinical-decision-analysis/run_CDA_analysis.sh
```

Any folder starting with `llm-parsed` is accepted — you do not need to register a
new judge anywhere. The prefix is what matters, because it is what tells CDA the
prediction lives in `LLM_filtered_resps` rather than `filtered_resps`. A name
matching no known prefix is rejected outright rather than guessed at.

Outputs are marked with the source, so runs never overwrite each other:
`CDA_REPORT_llm-parsed-gemma-4-31b.md` beside `CDA_REPORT.md`, and matching
`_llm-parsed-gemma-4-31b` markers on the `Results/` reports. Every report names
its source in the provenance table.

This reads `Results/` and `Data/`, both of which are gitignored. Cloning the
repo is not enough to reproduce the numbers — you need those trees too.

## What gets measured

| Proxy | Data | Measurement becomes | Statistic |
|---|---|---|---|
| **SNA maxillary position** | Ceph-Biometrics-400 | retrusive / normal / protrusive, around Steiner's 82°±2 | weighted κ |
| **SNB mandibular position** | Ceph-Biometrics-400 | retrusive / normal / protrusive, around Steiner's 80°±2 | weighted κ |
| **AJCC renal T category** | KiTS23, KiPA22 | T1a / T1b / T2a / T2b at 4, 7, 10 cm | weighted κ |

Sources: Steiner CC, *Am J Orthod* 1953; AJCC Cancer Staging Manual, 8th ed.,
2017. The cutoff numbers live in `cda_config.py` — that file is the authority,
this table is a summary of it.

> **Angles are folded into [0°, 90°].** The benchmark defines its angle target as
> `arccos(|A·B| / (‖A‖‖B‖))`, and that absolute value folds every angle into
> [0°, 90°], so an SNA or SNB above 90° is reflected back below it (a true SNA of
> 94.4° is stored as 85.6°), which can move a subject across a band edge. See
> DESIGN.md for the full consequence.

## How agreement is measured

Category of the *prediction* vs category of the *ground-truth measurement*. This
isolates the model: both sides go through the same cutoff table, so any
disagreement is caused by measurement error alone.

The analysis is followed by an uncertainty pass giving bootstrap 95% CIs and a
one-sided p-value for κ > 0. Both come from the same resampling pass, which
draws whole imaging volumes rather than individual slices — for a tumour proxy
the slices of one tumour are not independent observations.

## Reading the output

Start with **`CDA_REPORT.md`** in this folder: all the leaderboards, with
confidence intervals, in one Markdown file. Every CDA output is generated, not
checked in — the reports here and the rest of the `Results/` tree are all
gitignored, so run the pipeline to produce them:

| File | What it holds |
|---|---|
| `CDA_REPORT.md` (this folder) | **the final report** — every leaderboard, plus provenance |
| `CDA_REPORT_<source>.md` (this folder) | the same, one per LLM-judge re-parse |
| `<task_dir>/summary_CDA_task_canonical.txt` | per-proxy leaderboards + per-model detail |
| `<task_dir>/summary_CDA_uncertainty.json` | CIs and p-values |
| `<model>/parsed/summary_metrics_CDA_Task.json` | that model's numbers, machine-readable |
| `<model>/parsed/summary_values_CDA_Task.json` | one record per sample, for re-analysis |

The `.txt` reports carry per-model detail the final report leaves out — the per-proxy
overall block and the per-dataset breakdown; go to them when a leaderboard row raises a
question. Full confusion matrices are only in `summary_metrics_CDA_Task.json`.

A filtered run (`REMOVED_SAMPLES_DIR`) writes a `_filtered` twin of each T/L file
rather than overwriting it, so both sets can sit side by side. The marker goes
before `_canonical` — e.g. `summary_CDA_task_filtered_canonical.txt`. Check which
one you are reading: the two contain different numbers. A/D files have no twin,
and `CDA_REPORT.md` names the sample set each task contributed in its provenance
table.

Columns you will see:

- **Acc** — fraction of parsed samples landing in the right category.
- **Kappa** / **wKappa** — chance-corrected agreement. 0 means "no better than
  guessing with the same marginals"; 1 is perfect. `wKappa` (ordinal proxies)
  penalises a T1a-vs-T2b confusion more than a T1a-vs-T1b one.
- **Flip** — 1 − Acc, the decision-flip rate.
- **AccCov** — accuracy with unparseable predictions counted as wrong. Compare
  it against **Acc** to see how much of a good score is really coverage.
- **Nparsed / Ntotal** — a model that answers 14 of 120 prompts can post a
  flattering **Acc**; `Nparsed` is how you catch that.
- **n / vols** (uncertainty report) — records scored, and the number of
  independent imaging volumes they came from. For tumour proxies one volume
  contributes many slices, so `vols` is the honest sample size.

### Before you quote a number

- **κ is not comparable across proxies.** Both sides of the comparison are
  derived from the same continuous measurement, so agreement depends mostly on
  how close the cohort's values sit to a cutoff. A proxy whose cutoff falls in a
  dense part of the distribution will score lower for the same measurement
  accuracy. Compare models within a proxy, not proxies against each other.
- **Check `Nparsed` and the majority class.** The renal categories are far from
  uniform, so a constant answer already scores a substantial accuracy on its
  own. κ corrects for that; raw accuracy does not.
- **Small n.** Rows with a handful of parsed predictions still print a κ, a CI
  and a p-value — arithmetically valid, practically meaningless. The uncertainty
  report marks them `low_n` (fewer than 10 scored records) and tags the console
  line `<- low n`; the text report does not, so read `Nparsed` there.

## Running it yourself

Each script runs directly — no `-m`, no package install. Run them **from the
repo root**: the `$PWD` in `REMOVED_SAMPLES_DIR` resolves against it.
(`run_CDA_analysis.sh` itself locates the repo from its own path, so only the
arguments you pass it are CWD-sensitive.)

```bash
CDA=script/analyze/clinical-decision-analysis

# Agreement — one invocation per task directory, each with its own config
python $CDA/summarize_CDA_task.py --task_dir Results/MedVision-AD-v2-CoT \
    --config_yaml $CDA/config-AD-CoT.yaml --skip_model_wo_parsed_files
python $CDA/summarize_CDA_task.py --task_dir Results/MedVision-TL-v2-CoT \
    --config_yaml $CDA/config-TL-CoT.yaml --skip_model_wo_parsed_files

# Uncertainty — run after the step above; it reads what that step persists
python $CDA/cda_uncertainty.py --task_dir Results/MedVision-AD-v2-CoT \
    --config_yaml $CDA/config-AD-CoT.yaml
python $CDA/cda_uncertainty.py --task_dir Results/MedVision-TL-v2-CoT \
    --config_yaml $CDA/config-TL-CoT.yaml

# Final report — run last; renders CDA_REPORT.md from what the steps above wrote
python $CDA/build_CDA_report.py \
    --ad_task_dir Results/MedVision-AD-v2-CoT --ad_config_yaml $CDA/config-AD-CoT.yaml \
    --tl_task_dir Results/MedVision-TL-v2-CoT --tl_config_yaml $CDA/config-TL-CoT.yaml \
    --out $CDA/CDA_REPORT.md
```

Running the steps by hand with `--removed_samples_dir`? Pass it to the **T/L**
call only, then add `--filtered` to the T/L `cda_uncertainty.py` call and to
`build_CDA_report.py`, so they read the `_filtered` inputs that step just wrote.
Neither does any filtering itself — they only need the marker.

To score an LLM-judge re-parse, add `--parsed_dirname llm-parsed_<judge>` to
**all three** scripts and give `build_CDA_report.py` a different `--out`. The flag
must match across them: each script reads what the previous one wrote. It also
selects the row field holding the prediction (`LLM_filtered_resps` rather than
`filtered_resps`), which is why a name matching no known prefix is rejected. A
name whose prefix is valid but whose folder is missing gets caught later, by
`build_CDA_report.py` refusing to render a report in which nothing was found.

**Pass the config that matches the task directory.** A model's results folder can
differ between the AD and TL directories — one task gets re-run under a bugfix
and picks up a new `_bugfix-<sha>` suffix while the other does not, so there is
one config per task. Naming a folder that is not on disk is a hard error, which
means both a stale config *and* a mixed-up pair fail loudly rather than quietly
dropping a model.

### Matching the benchmark's sample set

The T/L benchmark drops slices whose mask has more than one connected component
(its canonical report is `summary_TL_task_filtered.txt`). CDA does not, by
default — so it scores 1,064 renal samples where the benchmark scores 1,025. To
score the same set:

```bash
REMOVED_SAMPLES_DIR=$PWD/Data/Datasets bash $CDA/run_CDA_analysis.sh
```

Every T/L output filename then gains a `_filtered` marker, so filtered and
unfiltered runs sit side by side rather than overwriting each other. The runner
does not apply it to the A/D task: with no mask there is no multi-cluster slice,
so it removed nothing there while still writing a full set of `_filtered` files
identical to the unfiltered ones.

Drop `--config_yaml` to process every model subdirectory with plain folder names
instead of the curated set the configs name. `summarize_CDA_task.py` also takes
`--model_dir` for a single model; `cda_uncertainty.py` does not — it always works
over a task directory.

## Files

| File | Role |
|---|---|
| `config-AD-CoT.yaml` | the evaluated models and their folders in an A/D results dir |
| `config-TL-CoT.yaml` | the same models and their folders in a T/L results dir |
| `cda_config.py` | clinical cutoff tables |
| `cda_stats.py` | categorisation, kappas, config loading |
| `summarize_CDA_task.py` | per-sample categorisation and per-proxy agreement |
| `cda_uncertainty.py` | clustered bootstrap CIs + p-values |
| `build_CDA_report.py` | renders `CDA_REPORT.md` from what the two above persist |
| `run_CDA_analysis.sh` | runs all of the above in order |
| `CDA_REPORT[_<source>].md` | **generated** — the final report, one per parsed source |
| `DESIGN.md` | implementation details, invariants, known limitations |

The folder is self-contained: it does not import `medvision_bm`. `numpy` is the
only hard dependency; `PyYAML` is imported lazily and needed only for
`--config_yaml`. `sklearn` is not used — the kappas are reimplemented and
verified against hand-computed references.

## Further reading

`docs/clinical-decision-agreement.md` has the full write-up, citations and
results. `DESIGN.md` in this folder documents the implementation for anyone
(human or agent) modifying these scripts.
