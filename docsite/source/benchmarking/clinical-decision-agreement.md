# Clinical Decision Agreement (CDA)

**Does the model's measurement error change the clinical decision?**

The benchmark metrics (see the [overview](overview.md)) report how far a model's
measurement is from the truth — MAE, MRE, IoU. That tells you the *size* of the
error, not whether it *matters*. A 3 mm miss on a 90 mm tumour is noise; the
same 3 mm miss on a tumour sitting at 40 mm moves it from stage T1a to T1b and
changes the treatment plan.

The CDA suite (`script/analyze/clinical-decision-analysis/`) re-scores existing
parsed benchmark outputs through that lens. Each measurement is pushed through a
published clinical cutoff table to become a category (a tumour stage, a skeletal
class), and the question becomes whether the model's number lands in the
[same category]{.mv-accent} as the ground truth. The headline statistic is
Cohen's kappa — agreement above what chance alone would produce.

Nothing is re-run: CDA only reads the `parsed/*.jsonl` records that
[parsing](parsing-and-summarizing.md) already produced. No GPU, no inference, no
`medvision_bm` import — seconds per model.

## The clinical proxies

| Proxy | Data | Measurement becomes | Statistic |
| --- | --- | --- | --- |
| **SNA maxillary position** | Ceph-Biometrics-400 | retrusive / normal / protrusive, around Steiner's 82°±2 | weighted κ |
| **SNB mandibular position** | Ceph-Biometrics-400 | retrusive / normal / protrusive, around Steiner's 80°±2 | weighted κ |
| **AJCC renal T category** | KiTS23, KiPA22 | T1a / T1b / T2a / T2b at 4, 7, 10 cm | weighted κ |

Cutoffs come from Steiner (*Am J Orthod*, 1953) and the AJCC Cancer Staging
Manual, 8th ed. (2017). The authoritative numbers live in `cda_config.py`; the
table above is a summary.

:::{warning}
**Angles are folded into [0°, 90°].** The benchmark defines its angle target as
`arccos(|A·B| / (‖A‖‖B‖))`, and that absolute value folds every angle into
[0°, 90°], so an SNA or SNB above 90° is reflected back below it (a true SNA of
94.4° is stored as 85.6°). That can move a subject across a band edge.
:::

## How agreement is measured

Category of the *prediction* vs category of the *ground-truth measurement*. Both
sides go through the same cutoff table, so any disagreement is
[caused by measurement error alone]{.mv-accent}.

The analysis is followed by an uncertainty pass: bootstrap 95% confidence
intervals and a one-sided p-value for κ > 0. The bootstrap resamples
[whole imaging volumes, not slices]{.mv-accent} — the 1,064 renal records come
from 121 volumes, and slices of one tumour are not independent observations. An
i.i.d. per-slice bootstrap gave intervals five times too narrow.

## Quick start

From the repo root:

```bash
REMOVED_SAMPLES_DIR=$PWD/Data/Datasets bash script/analyze/clinical-decision-analysis/run_CDA_analysis.sh
```

That runs the analysis, the uncertainty pass and the report over the canonical
result directories. The headline output is `CDA_REPORT.md` in the script's
folder — every leaderboard, with confidence intervals, in one Markdown file.
All CDA outputs are generated, never checked in; the pipeline reads `Results/`
and `Data/`, both gitignored, so a bare clone cannot reproduce the numbers.

`REMOVED_SAMPLES_DIR` makes CDA drop the same multi-cluster T/L slices the
published benchmark drops (renal 1,064 → 1,025 samples; see the
[T/L exclusion](parsing-and-summarizing.md#tl-only-excluding-multi-cluster-samples)),
and marks every T/L output filename with `_filtered` so filtered and unfiltered
runs sit side by side. It affects the T/L task only — A/D measures landmarks,
not masks, so there is no multi-cluster slice to drop.

:::{warning}
`REMOVED_SAMPLES_DIR` must point at a real directory. A path that does not
exist silently yields *unfiltered* numbers inside `_filtered`-named files.
:::

### Scoring the LLM-judge re-parse

Each model folder can hold several parsed sets: `parsed/` from the regex parser
plus one `llm-parsed*/` folder per
[LLM-judge re-parse](llm-judge-parsing.md). Select one with `CDA_PARSED_DIR`:

```bash
REMOVED_SAMPLES_DIR=$PWD/Data/Datasets CDA_PARSED_DIR=llm-parsed_gemma-4-31b \
  bash script/analyze/clinical-decision-analysis/run_CDA_analysis.sh
```

The `llm-parsed` prefix is what tells CDA the prediction lives in
`LLM_filtered_resps` rather than `filtered_resps`; a name matching no known
prefix is rejected outright rather than guessed at. Outputs carry the source in
their filenames (`CDA_REPORT_llm-parsed-gemma-4-31b.md` beside `CDA_REPORT.md`),
so runs never overwrite each other.

The individual scripts (`summarize_CDA_task.py`, `cda_uncertainty.py`,
`build_CDA_report.py`) can also be run one at a time — see
[the suite's README](https://github.com/YongchengYAO/MedVision/blob/master/script/analyze/clinical-decision-analysis/README.md)
for the per-script invocations and the flag pairings they require.

## Reading the output

Start with `CDA_REPORT.md`. The per-task `.txt` reports underneath it carry the
per-model detail (full confusion matrices, per-dataset breakdowns) the final
report leaves out. Columns you will see:

- **Acc** — fraction of parsed samples landing in the right category.
- **Kappa / wKappa** — chance-corrected agreement: 0 is "no better than guessing
  with the same marginals", 1 is perfect. `wKappa` (ordinal proxies) penalises a
  T1a-vs-T2b confusion more than a T1a-vs-T1b one.
- **Flip** — 1 − Acc, the decision-flip rate.
- **AccCov** — accuracy with unparseable predictions counted as wrong; compare
  against **Acc** to see how much of a good score is really coverage.
- **Nparsed / Ntotal** — a model that answers 14 of 120 prompts can post a
  flattering **Acc**; `Nparsed` is how you catch that.
- **n / vols** (uncertainty report) — records scored, and the independent
  imaging volumes they came from. For tumour proxies `vols` is the honest
  sample size.

Before quoting a number:

- **κ is not comparable across proxies.** Agreement depends mostly on
  how close the cohort's values sit to a cutoff, so a proxy whose cutoff falls
  in a dense part of the distribution scores lower for the same measurement
  accuracy. Compare models within a proxy, never proxies against each other.
- **Check `Nparsed` and the majority class.** The renal categories are far from
  uniform, so a constant answer already scores a substantial accuracy; κ
  corrects for that, raw accuracy does not.
- **Small n.** Rows with under 10 scored records still print a κ, a CI and a
  p-value — arithmetically valid, practically meaningless. The uncertainty
  report marks them `low_n`; the text report does not, so read
  `Nparsed` there.
- **Renal staging is per-slice, not per-tumour** — each scored record is
  one 2D slice's measurement pushed through the staging table.

## Further reading

- [`script/analyze/clinical-decision-analysis/README.md`](https://github.com/YongchengYAO/MedVision/blob/master/script/analyze/clinical-decision-analysis/README.md)
  — full user guide: per-script invocations, filtered runs, output file map.
- [`script/analyze/clinical-decision-analysis/DESIGN.md`](https://github.com/YongchengYAO/MedVision/blob/master/script/analyze/clinical-decision-analysis/DESIGN.md)
  — implementation notes, statistical choices (clustered bootstrap, the absent
  permutation test), invariants and known limitations.
