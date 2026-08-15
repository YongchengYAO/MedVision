# CDA — implementation notes

Reference for anyone modifying these scripts. User-facing description is in
`README.md`; the paper write-up is `docs/clinical-decision-agreement.md`.

Everything here is post-hoc re-scoring of existing benchmark outputs. No
inference, no GPU, no `medvision_bm` import. If a change to this folder requires
either, the change is in the wrong place.

## Module map

```
cda_config.py   clinical cutoff tables + output filenames + parsed-source table
                + seed.  Imports nothing local.
cda_stats.py    categorize, kappas, config loading, model-dir resolution,
                removed-samples filtering.  Imports nothing local.
    ^                    ^                       ^
summarize_CDA_task.py    analyze_CDA_renal_truelabel.py    cda_uncertainty.py
    (Track 1)                  (Track 2)                    (reads both tracks' output)
                    build_CDA_report.py   renders CDA_REPORT.md from what all
                                          three persist; recomputes nothing.
run_CDA_analysis.sh  drives all four, in order, for both task dirs.
```

Dependency direction is strictly one way. The four top-level scripts never
import each other; anything they both need belongs in `cda_stats.py`. That rule
is what keeps the two tracks from drifting apart — they previously carried
near-identical copies of the model-directory resolution logic.

## Input contract

Reads `<model_dir>/<parsed_dirname>/*.jsonl`, skipping any file whose basename
contains `_proc_acc` or `_eq_acc` (those are a different schema and will not
parse). Fields used per row:

| Field | Use |
|---|---|
| `doc.biometric_profile.metric_key` | selects the cephalometric proxy |
| `doc.image_file` | case id (Track 2) and resampling cluster (uncertainty) |
| `doc.slice_dim`, `doc.slice_idx` | passed through to the values JSON; part of the removed-samples key |
| `doc.taskID` | removed-samples key only (note the spelling — see below) |
| `target` | ground-truth measurement |
| *(prediction field)* | model prediction — **the field name depends on the source**, see below |

### Parsed sources: the folder fixes the field

A model folder can hold more than one set of parsed results, selected with
`--parsed_dirname` (env `CDA_PARSED_DIR` in the runner). Sources are matched by
**prefix**, and the prefix determines the prediction field:

| Folder prefix | Written by | Prediction field |
|---|---|---|
| `parsed` (default) | the regex parser | `filtered_resps` |
| `llm-parsed…` | any LLM-judge re-parse | `LLM_filtered_resps` |

Prefix, not exact name, because the judge writes **one folder per judge model and
per debug limit** — `llm-parsed_gemma-4-31b`, `llm-parsed_gemma-4-31b-limit100`,
and one more for every future judge. That set grows without bound and cannot be
enumerated; the prefix is what is stable, because the prefix is what determines
the row schema. Longest prefix wins.

The rows in that table are **prefixes, not folder names**: point
`--parsed_dirname` at a folder that exists. The only judged outputs on disk are
gemma-4-31b's (`llm-parsed_gemma-4-31b`, plus its `-limit100` debug twin), named
in `cda_config.CDA_LLM_PARSED_DIRNAME` for docs and examples only — nothing
resolves against that constant, so a different judge needs no change here, only a
different `--parsed_dirname`.

The row schema is otherwise identical and the filenames match exactly, which is
the trap: **a source cannot be a plain path swap.** Reading an LLM-judge folder
while still looking for `filtered_resps` finds the key absent on every row,
scores every sample a parse failure, and produces a complete-looking report in
which every model shows `n_parsed = 0`. So `CDA_PARSED_SOURCE_PREFIXES` in
`cda_config.py` maps prefix → field and `parsed_source_field()` is the only way
to get the field. Adding a source family is one line there. **Do not add a flag
that sets the field independently — the pairing is the safety property.**

Because names are matched by prefix, argparse `choices` no longer applies:
`validate_parsed_dirname` is the `type` hook that rejects an unmatched name, and
the runner calls the same function before doing any work. A name with a *valid
prefix* but no folder on disk cannot be caught that way, so `build_CDA_report.py`
raises when neither task directory yielded any metrics — see below.

Where the outputs go:

- **Per-model** JSONs are written back into the source folder, so they need no
  marker: `llm-parsed_gemma-4-31b/summary_metrics_CDA_Task.json` is already
  unambiguous.
- **Task-level** reports sit above the model folders and would collide, so they
  gain a source marker from `cda_config.source_suffix()` — `llm-parsed_gemma-4-31b`
  → `_llm-parsed-gemma-4-31b`, and the default source takes **no** marker so
  existing filenames are unchanged. The marker leads the suffix chain (below).
  It keeps the folder name legible rather than squashing it to alphanumerics:
  with one folder per judge, `_llm-parsed-gemma-4-31b` and
  `_llm-parsed-gemma-4-31b-limit100` have to be distinguishable at a glance.
  `_` becomes `-` so the underscore keeps its role as the suffix separator.
- `run_CDA_analysis.sh` asks `cda_config` for the marker instead of
  reimplementing it in shell. The "two sources never overwrite each other"
  guarantee holds only while the two agree, so there must be one rule.

A model missing the selected folder is reported in the report's NOT REPORTED
block, not dropped silently — a partially-covered source must never look
complete. Keep the two configs pointed at folders that exist for the sources you
intend to score: an earlier mismatch (the TL config naming a HuatuoGPT run the
judge pass had not been run against) showed up exactly there, as 17 of 18 models.

Dataset name comes from the **filename**, via `re.search(r"samples_([^_]+)_", …)`.
This assumes dataset names contain no underscore, which holds for all current
CDA datasets (`Ceph-Biometrics-400`, `KiTS23`, `KiPA22`). Adding an
underscore-bearing dataset to a proxy requires changing that regex.

File kind also comes from the filename: `TumorLesionSize` → `tl`,
`BiometricsFromLandmarks_Angle` → `angle`. Distance files match neither and are
skipped — there is no distance proxy.

**Directory listings are always sorted.** Both analysis scripts use
`cda_stats.sorted_glob`, and `get_subfolders` sorts its `os.scandir` result.
Plain `glob`/`scandir` return raw `readdir` order, which is stable only while a
directory is untouched — it changes when the tree is rewritten, copied, or moved
to another filesystem. That order propagates further than it looks: file order
sets the order records land in the values JSON, which sets *first-appearance*
cluster order in `cda_uncertainty._cluster_index`, which the seeded
`rng.integers(0, n_clusters, n_clusters)` then indexes into. A permuted listing
therefore draws a different set of volumes — same seed, different confidence
interval. Verified: forcing `glob.glob` to return reversed order leaves the
metrics and values JSONs byte-identical.

The one-time cost of adopting this: `by_dataset` keys reordered (KiTS23/KiPA22 →
alphabetical) and renal CI bounds moved by at most 0.0099. No point estimate
changed — κ is an order-independent sum — and no CI crossed zero.

### Value extraction

Ground truth: `ast.literal_eval(target)`. For T/L this is a `[major, minor]`
pair and element 0 is taken. **Verified: element 0 is the major axis in 673/673
KiTS23 samples**, so the convention is safe. For angles the target is a bare
scalar, handled by an `isinstance` branch — do not assume it is a list.

Prediction: split `filtered_resps[0]` on commas and require **exactly** the
expected count (2 for T/L, 1 for angles), matching the benchmark's own success
rule. Wrong count → prediction is `None` → counted as a parse failure, not as a
disagreement, and surfaced separately as `n_parsed` vs `n_total`.

T/L prediction uses `vals[0]`, not `max(vals)`. AJCC says "greatest dimension",
which argues for `max`, but `vals[0]` is the axis the MAE/MRE parser scores, so
CDA stays measuring the same quantity as the rest of the benchmark. Measured
cost of this choice: predictions are minor-axis-first in ~1–3% of samples and
`max()` would change the assigned T category for **≤ 3 of 673** samples. If you
switch to `max`, switch it in both `summarize_CDA_task._gt_pred_values` and
`analyze_CDA_renal_truelabel._parse_major`, and say so in the report.

### Removed-samples filtering (matching the T/L benchmark)

The T/L benchmark excludes slices whose mask has more than one connected
component — which is why its canonical report is `summary_TL_task_filtered.txt`.
Without the same exclusion CDA scores **1,064 renal samples where the benchmark
scores 1,025**, so the two are not like-for-like.

`--removed_samples_dir` (with `--removed_samples_filename`, default
`multi_cluster_samples_v1.0.0_to_v1.1.0.json`) applies it, mirroring
`summarize_TL_task.py`. `run_CDA_analysis.sh` passes it through from the
repo-wide `REMOVED_SAMPLES_DIR` env var, which is unset by default:

```bash
REMOVED_SAMPLES_DIR=$PWD/Data/Datasets bash run_CDA_analysis.sh
```

Three things to know:

- The exclusion key is
  `(image_file relative to the dataset folder, slice_dim, slice_idx, task_id)`.
  **The task id is `doc["taskID"]`** — not `doc["task_ID"]`, which is the
  spelling used *inside* the removed-samples JSON. Mixing the two silently
  matches nothing and the filter becomes a no-op.
- Every output filename gains a `_filtered` marker, ordered
  `_filtered` → `_canonical` → `_limitN`, so a filtered and an unfiltered run
  cannot overwrite each other. `cda_uncertainty.py` takes `--filtered` to read
  and write those names; it does no filtering itself.
- **Filtering is a T/L-only concept, and `run_CDA_analysis.sh` passes it to the
  T/L calls only.** The exclusion list marks slices whose *mask* has more than one
  connected component; A/D measurements come from landmark coordinates, so there
  is no mask and no such slice. A dataset with no removed-samples file is simply
  unfiltered, so a filtered A/D run was numerically a **verified no-op** — but it
  still wrote a complete `_filtered` set (2 task-level + 36 per-model files)
  byte-identical to its unfiltered twin. Those duplicates were deleted and the
  A/D calls no longer take the flag. Passing `--removed_samples_dir` against an
  A/D directory by hand will recreate them.

Measured effect: renal Track 1 1,064 → 1,025 slices (KiTS23 673→640, KiPA22
391→385); Track 2 joined cases 95 → 92, organ-confined 70 → 68. Track 2 is the
more exposed of the two because it takes a per-case **max** over slices, so one
excluded slice can set a whole case's `gt_max` and shift its stage.

## Proxy routing

`_proxies_for_sample(file_kind, dataset_name, metric_key)`:

- `angle` → looks `metric_key` up in `CDA_CEPH_ANGLE_PROXIES`; unknown keys map
  to no proxy. Note this keys on `metric_key` alone, not on dataset — safe only
  while no other dataset reuses those keys.
- `tl` → the renal T-stage proxy iff `dataset_name in CDA_RENAL_TL_DATASETS`.

A sample can hit several proxies; each contributes one record per proxy. Results
aggregate twice: per proxy (`overall`) and per proxy × dataset, keyed
`"<proxy> @ <dataset>"` (`by_dataset`).

## Categorisation semantics

`cda_stats.categorize(value, cutoffs, labels, right_closed)`.

`right_closed` is **either one bool or a list with one entry per cutoff**:

- `True` — a value exactly on the cutoff falls in the **lower** category
  (advance only when `v > c`). Correct for one-sided rules: AJCC "≤ 4 cm is
  T1a", ANB "≤ 4° is Class I".
- `False` — advance when `v >= c`; the value falls in the **upper** category.

The per-cutoff list exists because a **two-sided band cannot be expressed with a
single flag**. Steiner's SNA norm is 82°±2, i.e. normal is the closed interval
`[80, 84]`. That needs `[False, True]`: the lower edge opens upward so 80.0 is
normal, the upper edge stays closed so 84.0 is also normal. With a single
`True`, 80.0 was categorised as *retrusive*.

This is not hypothetical. In the 360 Ceph-Biometrics-400 ground-truth angles,
**2 sit exactly on a lower band edge** (SNA 80.0, SNB 78.0) and were
mis-assigned; across 50 model folders, 10 parsed predictions land exactly on a
cutoff (models like round numbers). KiTS23 also reports rounded pathologic sizes
such as 4.0 cm, landing straight on the AJCC boundary.

**When adding a proxy, state the boundary direction per published rule.** It is
a property of the rule, not a house convention.

## Statistics

All in `cda_stats.py`, numpy only. Both kappas are verified against
hand-computed references (2×2 plain κ = 0.400; 3×3 quadratic-weighted κ =
0.800).

- `_confusion_matrix` drops any pair whose GT or predicted label is not in
  `labels`. Currently unreachable — `categorize` only ever emits labels from the
  same list — but it means `accuracy` (computed over all zipped pairs) and
  `cohen_kappa` (computed over the matrix) would silently diverge if a caller
  ever passed a narrower label set.
- **Label order is load-bearing** for `weighted_kappa`: it defines the ordinal
  distance. Order must come from the config's `labels` list, never from sorting
  or first appearance. Scrambling a 3-class order changes κ from 0.800 to 0.544.
  Quadratic weights are scale-invariant, so passing a longer label list (e.g.
  the 6-class T scale where only 4 classes are occupied) gives the same value.
- Degenerate cases follow sklearn: κ is `nan` when both raters collapse onto the
  *same* single category (`pe >= 1`), and `0.0` when only the prediction
  collapses.
- `cal_clinical_agreement` reports `n_parsed` and `n_total` alongside
  `accuracy_coverage_adjusted` / `flip_rate_coverage_adjusted`, which count
  parse failures as disagreement. Plain `accuracy` and the kappas are over
  parsed samples only. Both are reported because neither alone is honest: plain
  accuracy rewards a model that declines to answer, coverage-adjusted accuracy
  conflates instruction-following with measurement skill.
- `ordinal: True` on a proxy adds `weighted_kappa` to the metrics and makes
  `cda_uncertainty` report the weighted statistic for it.

## Model selection: one config per task

There are **two** configs, `config-AD-CoT.yaml` and `config-TL-CoT.yaml`, each a
flat `model_display_name` map of results folder → display name, same schema as
the `script/visualization` configs. Pass the one matching the task directory.

Two files rather than one, because **a model's canonical results folder can
differ between tasks**. When one task is re-run under a bugfix it gains a
`_bugfix-<sha>` suffix and the other does not. Concretely, HealthGPT is
`HealthGPT-L14_bugfix-2eb7706` under AD and `HealthGPT-L14_bugfix-0a4c5e2` under
TL — neither folder exists in the other task's directory.

The cost of the split is that display names and ordering are duplicated across
the two files and can drift. **Keep them identical; only folder names may
differ.** That is the one thing to check when editing either file.

`load_model_display_map(config_yaml)` reads the map.
`resolve_model_dirs(task_dir, display_map)` turns it into ordered
`(model_dir, display)` pairs and **raises `FileNotFoundError` if any listed
folder is absent**. That hard failure is deliberate and does double duty: it
catches a stale config, and it catches passing the AD config against a TL
directory (or vice versa), since the mismatched HealthGPT folder will not exist.
The original bug was a config naming five superseded folders that all still
existed on disk, so the numbers were silently stale rather than missing.

Note what this check *cannot* catch: a folder that exists but is superseded. No
filesystem check can. The folder names must be kept in sync with
`script/visualization/config-AD-CoT.yaml` and `config-TL-CoT.yaml`, which are
the repo's authority for which run is canonical for which task.

`_process_task_directory` drives both computation and reporting from that one
resolved list, so the set of models whose metrics are recomputed is exactly the
set that appears in the report.

## Track 2 specifics

- Case id is `os.path.basename(image_file).split(".")[0]` — `case_00149.nii.gz`
  → `case_00149`, which is the KiTS23 clinical table key.
- Per-case aggregation takes the **max** major axis over the case's annotated
  slices, as a stand-in for AJCC's 3D greatest dimension. Note the asymmetry
  this creates: `pred_max` is a max over only the slices that *parsed*, so a
  model with many parse failures maxes over fewer slices and is biased toward
  smaller sizes (lower stage).
- Clinical table is fetched from the public KiTS23 repo and cached at
  `Data/Datasets/KiTS23/kits23_clinical.json`, **relative to the working
  directory** — hence "run from the repo root". `--no_download` forces the cache.
- `pathology_t_stage` values map through `KITS23_TSTAGE_MAP`; `"na"` and unknown
  values become `None` and those cases are dropped from the join.
- Three comparison rows per stratum: the model, the 2D-slice GT **reference**,
  and the pathologic-3D-size **ceiling**. Only the last is a genuine ceiling for
  size-only rules. The 2D-slice row is a reference: max-over-slices
  under-estimates the 3D greatest dimension, so a model that over-measures can
  beat it. Together they separate "loss from invasion-based staging" and "loss
  from 2D slicing" from "loss from the model".
- Strata: full 6-class (exposes that no size rule can ever emit pT3/pT4) and
  organ-confined pT1–pT2 (where size is the staging axis and the comparison is
  fair).

## Uncertainty: the resampling unit is the volume

`cda_uncertainty.py` re-reads the per-sample categorisations the two tracks
persist, so its CI and p-value describe exactly the number they annotate. It
must run **after** both; with no input it raises rather than writing an
empty-but-successful report.

**Both procedures resample whole imaging volumes, not records.** Track 1 scores
one record per annotated 2D slice, and a tumour contributes many slices —
measured: **1,064 renal records from 121 volumes**, mean 8.8, max 64 from a
single volume. Resampling records i.i.d. treats those 64 slices as 64
independent facts. Measured effect on the renal proxy: the i.i.d. CI is
**5.0× too narrow**, `[-0.067, -0.012]` (excludes zero) versus the clustered
`[-0.215, +0.056]` (includes zero) — enough to flip a null result into an
apparently significant one.

The cluster id is `image_file` for Track 1 and `case_id` for Track 2, which is
already one record per case. Cephalometric proxies are one record per subject,
so clustering is a provable no-op there — every A/D row satisfies
`n == n_clusters`, whatever that model's parse coverage happens to be.

The p-value comes from **inverting that same bootstrap distribution**
(`p = (#{replicate <= 0} + 1) / (n_valid + 1)`, one-sided for κ > 0), so one
resampling pass produces both numbers and they cannot disagree about whether
zero is plausible.

A label-permutation test was used here and was removed as **unfixable under
clustering**. The exchangeable unit is the volume, but volumes have unequal
sizes, so permuted prediction blocks cannot be re-paired against the references
without splitting clusters across reference boundaries — which strips
between-cluster variance out of the null. Measured on a clustered null with
reference and prediction independent by construction (60 volumes, ~9 records
each, 120 trials): the block-permutation test rejected at **0.100** against a
nominal 0.05, the bootstrap inversion at **0.050**. Do not reintroduce a
permutation test here without re-running that calibration check. Note the nulls
differ — a permutation test addresses the sharp null of no association, this
addresses whether the parameter is zero.

`n_clusters` is reported next to `n` in both the JSON and the console table, and
is counted from `_cluster_index` (what the resampler actually uses) rather than
from distinct ids, so records with no `image_file` show up as the singleton
clusters they are. If `n_clusters < n`, `n` overstates the information content.

## The final report

Every other output lands in the gitignored `Results/` tree, scattered across one
directory per model. `build_CDA_report.py` gathers the leaderboards into
`CDA_REPORT.md`, which lives beside the code rather than under `Results/`.

It is generated, not checked in: `.gitignore` excludes
`CDA_REPORT*.md` — a glob rather than a list, because there is one report per
parsed source and the judge pass adds one per judge model. A clone gets the
analysis and reproduces the numbers by running it. Keep it that way — the report
is derived from `Results/` and `Data/`, neither of which is in the repo, so a
committed copy could silently disagree with the trees it claims to summarise.

Four rules keep it trustworthy:

- **It is a renderer, not an analysis.** Nothing is recomputed; every cell is one
  field of one JSON. If a number looks wrong, the bug is upstream. Do not add a
  computation here — add it to the track that owns the statistic and re-render.
- **It reads JSON, never the `.txt` reports.** Those truncate for display (model
  names to 26 chars, proxy labels to 56) and would round-trip lossily.
- **No timestamp.** An unchanged analysis regenerates a byte-identical file, so
  `git diff` shows real numeric movement instead of a churning date line.
  Verified across a full rerun. Do not add a generation date.
- **Filtered-artifact resolution is per task and is disclosed.** `--filtered`
  reads `_filtered` artifacts where a task publishes them; A/D publishes none, so
  it falls back to unfiltered, and the provenance table states which sample set
  each task contributed. The fallback must never be silent.

Two presentation choices worth keeping: the decision-flip column is dropped
(it is exactly `1 − Acc`, so it carries no information), and the Track-2
reference rows collapse to two lines when they are identical across models —
which they are, because they depend only on the joined case set, not on the
model. Emitting one identical row per model reads as N measurements of the
ceiling when there is only one. The per-model table returns automatically if a
model ever joins a different case set.

Verification when changing the renderer: the Markdown must agree with the `.txt`
leaderboards, which are produced by a different code path from the same JSONs.
Last checked — Track 1: 360 cells over 4 proxies, 0 mismatches; Track 2: 90
cells over 18 models, 0 mismatches.

## Output schemas

`<model>/parsed/summary_metrics_CDA_Task.json`
```
{ "overall":    { "<proxy>": {name, ordinal, accuracy, cohen_kappa, [weighted_kappa],
                              confusion{gt}{pred}, flip_rate, n_parsed, n_total,
                              accuracy_coverage_adjusted, flip_rate_coverage_adjusted} },
  "by_dataset": { "<proxy> @ <dataset>": { …same… } } }
```

`<model>/parsed/summary_values_CDA_Task.json` — a flat list, one record per
sample × proxy: `proxy, dataset, metric_key, gt_value, pred_value, gt_category,
pred_category, image_file, slice_dim, slice_idx`. This is the re-analysis
surface; `cda_uncertainty.py` consumes it and `image_file` is what makes cluster
resampling possible. Keep that field.

`<task_dir>/summary_CDA_uncertainty[_truelabel].json` — run metadata
(`n_boot`, `seed`, `ci_method`, `p_method`, `cluster_note`) plus `rows` of
`{model, proxy, statistic, n, n_clusters, point_estimate, ci_lower, ci_upper,
p_kappa_gt_zero, low_n, n_boot_valid}`. `low_n` is `n < MIN_INFORMATIVE_N` (10).

`script/analyze/clinical-decision-analysis/CDA_REPORT[_<source>].md` — the
final Markdown report: provenance (including which parsed source the numbers
came from), a legend, one Track-1 table per proxy, the Track-2 leaderboard and
its GT reference rows. Rendered by `build_CDA_report.py`; gitignored, and
overwritten on every run of its own source.

Task-level output filenames carry `_<source>` (non-default parsed source, e.g.
`_llm-parsed-gemma-4-31b`) then
`_filtered` (removed-samples run) then `_canonical` (a config was given) then
`_limit<N>` (a debug run), in that order, so no two runs of different scope can
overwrite each other. **A/D artifacts never carry `_filtered`** — see the
removed-samples section. Per-model outputs carry `_filtered`/`_limit<N>` only:
the source is already encoded in the folder they live in.

## Known limitations

Real and unfixed. Anything quoting CDA numbers should carry these.

1. **The benchmark angle is folded into [0°, 90°]** — not merely unsigned. The
   prompt defines the target as `arccos(|A·B| / (‖A‖‖B‖))`; the absolute value
   makes the stored value `min(θ, 180° − θ)`. Verified: the maximum stored value
   across all 8 angle keys is exactly 90.0000, with none above. Consequences:
   - Class III (signed ANB < 0) is unrecoverable, so the ANB proxy is binary.
     Roughly a third of the cohort is truly Class III, and the folding is a
     *reflection*, not a coarsening — some Class III subjects land in Class II,
     the opposite of the truth.
   - SNA/SNB above 90° reflect back below it (a true SNA of 94.4° stores as
     85.6°), which can move a subject across a band edge.
   - **Signed ANB is NOT recoverable as SNA − SNB from the shipped values.** On
     the stored (folded) angles `||SNA − SNB| − ANB|` reaches **8.74°** and
     exceeds 0.01° for 83 of 120 subjects; the identity holds only on unfolded
     angles. An earlier version of this file claimed the opposite — that was
     wrong, and any doc repeating it should be corrected.

   Restoring a three-class ANB therefore means recomputing angles from the
   landmark coordinates — a **new derived measurement**, not a re-categorisation
   of an existing one. That changes CDA's contract ("re-scores existing parsed
   outputs") and the paper's headline ANB number, so it is a proxy redefinition
   for the maintainer to approve, not a bug fix.
2. **Per-slice T staging.** Track 1 assigns an AJCC T category to each 2D
   slice's in-plane major axis, but AJCC is defined on the tumour's greatest
   dimension. **26.6% of renal slices carry a T category different from their
   own tumour's max-over-slices category.** Track 2 aggregates per case and does
   not have this problem; the Track-1 renal number is a measurement-sensitivity
   statistic, not a staging accuracy.
3. **No majority-class baseline.** Always answering "T1a" scores ≈0.557
   accuracy on the organ-confined stratum. Accuracy columns carry no such
   reference row; κ does correct for it, which is why κ is the headline.
4. **Small-n rows are flagged, not suppressed.** A row with `n_parsed = 1` still
   emits κ, a zero-width 95% CI and `p = 1.0` — arithmetically correct,
   informationally empty. `cda_uncertainty.py` sets `low_n` (`n <
   MIN_INFORMATIVE_N`, 10) and marks the console line, deliberately rather than
   suppressing: κ *is* well defined there, and a threshold inside the stats layer
   would impose an arbitrary policy on every caller. The Track-1 text report
   carries no such flag, so check `Nparsed` when reading it.
5. **One RNG seed** (`CDA_SEED = 1024`) is shared across every model and proxy,
   so equal-n rows are evaluated on identical resample draws. Fine for
   reproducibility, but the rows are not independent replicates.
6. **κ is not comparable across proxies.** Both raters are deterministic
   functions of the same measurement, so κ depends on where the cutoff falls in
   the cohort's distribution. It is a model × cohort property.
7. **`gt2Dslice_vs_pathologic` is a reference, not a bound.** It is one
   particular estimator (max-over-slices of the GT major axis), not a maximum
   over estimators, and a model that over-measures can beat it because
   max-over-slices systematically *under*-estimates the 3D greatest dimension.
   The report prose, README and this file now all call it a reference; only
   `pathologic3Dsize_vs_pathologic` is a genuine ceiling for size-only rules.
   The **JSON key is still `gt2Dslice_vs_pathologic`** — deliberately unchanged,
   since it is neutral and downstream files already read it.
8. **Track 2's per-case prediction is maxed over parsed slices only**, while GT
   is maxed over all slices, and a case counts as measured if *any* single slice
   parsed. A model that parses few slices maxes over fewer values and is biased
   toward smaller sizes and lower stages. `AccCov` in the leaderboard is the
   guard against reading such a row as skill.

## Invariants

- Never import `medvision_bm` here, and keep `sklearn` out. `numpy` is the only
  hard dependency; `PyYAML` stays lazily imported inside the loader.
- Shared logic goes in `cda_stats.py`. The three scripts must not import each
  other, and must not grow private copies of resolution or categorisation.
- `CDA_SEED` comes from `cda_config.py`. Do not hardcode a seed.
- Cutoff tables live only in `cda_config.py`. No literal 4.0 / 40.0 / 70.0 /
  100.0 in the analysis scripts.
- A config-listed folder missing from disk must stay a hard error.
- `image_file` must stay in the values JSON — cluster resampling depends on it.
- Never call bare `glob.glob` or `os.scandir` for inputs. Use
  `cda_stats.sorted_glob` / `get_subfolders`: unsorted listings silently change
  the bootstrap draw across machines and filesystem moves.
- Never hardcode `"parsed"` as a path segment. Take `parsed_dirname` and get the
  prediction field from `parsed_source_field()` — the folder and the field must
  travel together, or a wrong-source run reports `n_parsed = 0` instead of
  failing.
- `build_CDA_report.py` renders only. No statistic may be computed there, and it
  must emit no timestamp — an unchanged analysis has to regenerate byte-identical
  output.
- Removed-samples filtering goes to the T/L calls only. An A/D `_filtered` file
  is by definition a duplicate.
- Changing a cutoff, a boundary direction, an `ordinal` flag or the resampling
  unit changes published numbers. Re-run `run_CDA_analysis.sh` (which re-renders
  `CDA_REPORT.md`) and update `docs/clinical-decision-agreement.md` in the same
  change.
