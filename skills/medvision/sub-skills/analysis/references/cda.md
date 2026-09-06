# Clinical Decision Agreement (CDA)

**Question it answers:** does the model's measurement error change the *clinical decision*? MAE/MRE/IoU tell you the
size of an error, not whether it matters. A 3 mm miss on a 90 mm tumour is noise; the same 3 mm miss on a 40 mm tumour
crosses an AJCC stage boundary.

CDA is post-hoc re-scoring only. It reads parsed benchmark records, pushes both the prediction and the ground truth
through the **same published cutoff table**, and measures agreement between the two resulting categories. No inference,
no GPU, no `medvision_bm` import - `numpy` is the only hard dependency (`PyYAML` is imported lazily, and only when a
`--config_yaml` is given). Seconds per model.

Bundled implementation: `scripts/cda/` (`cda_config.py`, `cda_stats.py`, `summarize_CDA_task.py`,
`cda_uncertainty.py`, `build_CDA_report.py`, plus the two config templates), driven by `scripts/run_cda.sh`.

---

## 1. Method

Both sides of the comparison are deterministic functions of a measurement, so any disagreement is caused by measurement
error alone:

```
gt_category   = categorize(ground-truth measurement, cutoffs, labels, right_closed)
pred_category = categorize(model measurement,        cutoffs, labels, right_closed)
agreement     = accuracy, Cohen's kappa, and (ordinal proxies) quadratic-weighted kappa
```

A sample can contribute to several proxies; each contributes one record per proxy. Results aggregate twice: per proxy
(`overall`) and per proxy x dataset, keyed `"<proxy> @ <dataset>"` (`by_dataset`).

### The proxies shipped in `cda_config.py`

| Proxy key (config) | Proxy name | Data | Categories | Cutoffs | Boundary rule |
|---|---|---|---|---|---|
| `A-L_1_2-L_2_5` | SNA maxillary position | Ceph-Biometrics-400 (angle files) | retrusive / normal / protrusive maxilla | 80.0, 84.0 deg | `right_closed = [False, True]` |
| `A-L_1_2-L_2_6` | SNB mandibular position | Ceph-Biometrics-400 (angle files) | retrusive / normal / protrusive mandible | 78.0, 82.0 deg | `right_closed = [False, True]` |
| `CDA_RENAL_TSTAGE` | AJCC renal T category (greatest dimension) | KiTS23, KiPA22 (T/L files) | T1a / T1b / T2a / T2b | 40.0, 70.0, 100.0 mm | `right_closed = True` |

**Where the cutoffs come from.** The angle bands are Steiner's cephalometric norms (SNA 82 deg +/- 2, SNB 80 deg +/- 2;
Steiner CC, *Am J Orthod* 1953). The renal thresholds are the AJCC Cancer Staging Manual, 8th ed. (2017): T1a <= 4 cm,
T1b > 4-7 cm, T2a > 7-10 cm, T2b > 10 cm, organ-confined. `cda_config.py` is the authority; the table above summarises
it. Angle proxies are keyed on `biometric_profile.metric_key` alone, so they are correct only while no other dataset
reuses those keys.

**Boundary direction is per published rule, not a house convention.** `right_closed = True` means a value exactly on a
cutoff falls in the *lower* category (advance only when `v > c`), which is what "<= 4 cm is T1a" requires. A two-sided
band cannot be expressed with a single flag: Steiner's normal band is the closed interval `[80, 84]`, so its lower edge
must open upward and its upper edge stay closed - hence the per-cutoff list `[False, True]`. This is not hypothetical:
the source notes that 2 Ceph-Biometrics-400 ground-truth angles sit exactly on a lower band edge (SNA 80.0, SNB 78.0),
and models emit round numbers on cutoffs too.

### Angles are folded into [0, 90] degrees

The benchmark defines its angle target as `arccos(|A.B| / (||A|| ||B||))`. The absolute value makes the stored value
`min(theta, 180 - theta)`, so an SNA above 90 deg is reflected back below it (a true 94.4 deg is stored as 85.6 deg),
which can move a subject across a band edge. `cda_config.py` records that the maximum stored value across all 8 angle
keys is exactly 90.0000 with none above. Recovering an unfolded angle would mean recomputing it from landmark
coordinates - a *new derived measurement*, outside CDA's "re-score existing parsed outputs" contract. Carry this
limitation with any SNA/SNB number.

### Kappa variants

Implemented in `cda_stats.py` with numpy only (no scikit-learn); the source states both are verified against
hand-computed references (2x2 plain kappa = 0.400; 3x3 quadratic-weighted kappa = 0.800).

- **`cohen_kappa`** - chance-corrected agreement at the observed marginals. 0 = no better than guessing.
- **`weighted_kappa`** - quadratic weights, added only for proxies with `ordinal: True` (all three shipped proxies).
  It penalises a T1a-vs-T2b confusion more than T1a-vs-T1b. **Label order is load-bearing**: it defines the ordinal
  distance and must come from the config's `labels` list, never from sorting or first appearance (the source records
  that scrambling a 3-class order moves kappa from 0.800 to 0.544). Quadratic weights are scale-invariant, so a longer
  label list with unoccupied classes gives the same value.
- **Degenerate cases follow scikit-learn's convention**: kappa is `nan` when both raters collapse onto the *same*
  single category (`pe >= 1`), and `0.0` when only the prediction collapses.

### Coverage vs accuracy

A prediction must parse into exactly the expected number of values (1 for an angle, 2 for T/L, matching the benchmark's
own success rule); anything else is `None` and counted as a **parse failure**, not a disagreement.

- `accuracy`, `cohen_kappa`, `weighted_kappa` are computed over **parsed** samples only.
- `accuracy_coverage_adjusted` / `flip_rate_coverage_adjusted` count parse failures as wrong.
- `n_parsed` vs `n_total` is how you catch a model that answers 14 of 120 prompts and posts a flattering accuracy.

`flip_rate` is exactly `1 - accuracy` (the Markdown report omits it for that reason).

### The T/L value that is categorised

Ground truth is `ast.literal_eval(target)`; for T/L that is a `[major, minor]` pair and **element 0** is taken. The
prediction likewise uses `vals[0]`, not `max(vals)`, so CDA measures the same quantity the MAE/MRE parser scores. AJCC
says "greatest dimension", which would argue for `max`; the source records the measured cost of the choice as at most
3 of 673 samples changing T category.

---

## 2. Volume-level bootstrap (`cda_uncertainty.py`)

The uncertainty pass re-reads the per-sample categorisations that step 1 persisted, so its interval describes exactly
the number it annotates. It must run **after** step 1; with no input it raises rather than writing an empty report.

- **The resampling unit is the imaging volume, not the record.** The cluster id is `image_file`. A T/L proxy scores one
  record per annotated 2D slice and one tumour contributes many slices, so i.i.d. resampling would treat them as
  independent facts. The source reports a measured renal case of 1,064 records from 121 volumes (mean 8.8, max 64) where
  the i.i.d. interval was 5x too narrow and excluded zero while the clustered one included it.
- Cephalometric proxies are one record per subject, so clustering is a provable no-op there (`n == n_clusters`).
- **The p-value is inverted from the same bootstrap distribution**: `p = (#{replicate <= 0} + 1) / (n_valid + 1)`,
  one-sided for kappa > 0. One resampling pass produces both numbers, so they cannot disagree about whether zero is
  plausible. A label-permutation test was tried and removed as unfixable under clustering.
- Defaults: `--n_boot 4000`; seed = `CDA_SEED = 1024` in `cda_config.py` (mirrors `medvision_bm.utils.configs.SEED`,
  duplicated so the folder stays self-contained). One seed is shared across every model and proxy, so equal-n rows are
  evaluated on identical draws - reproducible, but the rows are not independent replicates.
- `low_n` marks rows with fewer than `MIN_INFORMATIVE_N` = 10 scored records. They are flagged, never suppressed; the
  `.txt` report carries no such flag, so read `Nparsed` there.
- **Directory listings must stay sorted.** The modules use `cda_stats.sorted_glob` / `get_subfolders` because file order
  sets record order, which sets first-appearance cluster order, which the seeded RNG indexes into - a permuted listing
  would draw different volumes under the same seed. Point estimates are order-independent; intervals are not.

---

## 3. Configuration

### `cda_config.py` - what it holds

| Symbol | Meaning |
|---|---|
| `CDA_CEPH_ANGLE_PROXIES` | `metric_key -> {name, cutoffs, labels, ordinal, right_closed}` for SNA and SNB |
| `CDA_RENAL_TSTAGE`, `CDA_RENAL_TL_DATASETS` | the AJCC proxy spec and the datasets it applies to (`("KiTS23", "KiPA22")`) |
| `CDA_PARSED_SOURCE_PREFIXES` | `{"llm-parsed": "LLM_filtered_resps", "parsed": "filtered_resps"}` - folder prefix to prediction field |
| `CDA_DEFAULT_PARSED_DIRNAME` | `"parsed"` |
| `CDA_LLM_PARSED_DIRNAME` | the judge folder named for documentation and CLI examples only; nothing resolves against it |
| `parsed_source_field`, `validate_parsed_dirname`, `source_suffix` | the only supported way to map a folder to its field and to its output marker |
| `CDA_SEED` | `1024` |
| `SUMMARY_FILENAME_CDA_METRICS`, `SUMMARY_FILENAME_CDA_VALUES` | `summary_metrics_CDA_Task.json`, `summary_values_CDA_Task.json` |

The parsed source is matched by **prefix, longest first**, and a name matching none is rejected outright. The prefix is
what determines the row schema, and the judge writes one folder per judge model and per debug limit
(`llm-parsed_<judge>`, `llm-parsed_<judge>-limit100`, ...), so the set of names cannot be enumerated. Selecting a source
therefore also selects its field - reading a judge folder while looking for `filtered_resps` would find the key absent
on every row and produce a complete-looking report where every model shows `n_parsed = 0`.

### Model config YAML (`scripts/cda/config-AD-CoT.yaml`, `config-TL-CoT.yaml`)

One top-level key:

```yaml
model_display_name:
  "<results folder name under the task dir>": "<display name in reports>"
```

- Order = report order. **Every listed folder must exist under `--task_dir` or the scripts raise `FileNotFoundError`.**
  That hard failure is deliberate: it catches a stale config *and* catches passing the A/D config against a T/L
  directory. It cannot catch a folder that exists but is superseded.
- **There is one config per task** because a model's canonical folder can differ between tasks (a task re-run under a
  bugfix gains a `_bugfix-<sha>` suffix while the other task does not). Keep display names identical between the two
  files; only folder names may differ.
- Drop `--config_yaml` to analyse every subfolder of the task dir with plain folder names; the task-level report then
  carries no `_canonical` marker.
- The bundled files are **templates listing the repository's paper roster**. Folder names are run-specific - edit them
  to match your own results tree.

### Input contract

Reads `<model_dir>/<parsed_dirname>/*.jsonl`, skipping any basename containing `_proc_acc` or `_eq_acc` (different
schema). Fields used per row:

| Field | Use |
|---|---|
| `doc.biometric_profile.metric_key` | selects the cephalometric proxy |
| `doc.image_file` | resampling cluster id (uncertainty) |
| `doc.slice_dim`, `doc.slice_idx` | passed into the values JSON; part of the removed-samples key |
| `doc.taskID` | removed-samples key only - **note the spelling**, the JSON file itself uses `task_ID` |
| `target` | ground-truth measurement |
| `filtered_resps` **or** `LLM_filtered_resps` | prediction - chosen by the `--parsed_dirname` prefix |

Dataset name comes from the **filename** via `re.search(r"samples_([^_]+)_", ...)`, which assumes dataset names contain
no underscore (true for `Ceph-Biometrics-400`, `KiTS23`, `KiPA22`). File kind also comes from the filename:
`TumorLesionSize` -> `tl`, `BiometricsFromLandmarks_Angle` -> `angle`; distance files match neither and are skipped
(there is no distance proxy).

### Matching the T/L benchmark's sample set

`summarize_TL_task` drops slices whose mask has more than one connected component. CDA does not by default, so the two
score different sets. Pass `--removed-samples-dir <data_dir>/Datasets` (wrapper) or `--removed_samples_dir` (script) to
apply the same exclusion; every T/L output filename then gains a `_filtered` marker so both runs can sit side by side.
Pick one convention and stay with it - the numbers are not interchangeable.

Filtering is **T/L-only**: A/D measurements come from landmarks, so there is no multi-cluster slice and no
`_filtered` twin. `cda_uncertainty.py --filtered` and `build_CDA_report.py --filtered` do no filtering themselves; they
only need to know which filenames to read, and the report discloses per task which sample set it used.

---

## 4. Output files

Per-model, written **into the parsed source folder** (no source marker needed - the folder already identifies it):

| File | Content |
|---|---|
| `summary_metrics_CDA_Task[_filtered][_limit<N>].json` | `{"overall": {proxy: {...}}, "by_dataset": {"<proxy> @ <dataset>": {...}}}` where each entry holds `name, ordinal, accuracy, cohen_kappa, [weighted_kappa], confusion{gt}{pred}, flip_rate, n_parsed, n_total, accuracy_coverage_adjusted, flip_rate_coverage_adjusted` |
| `summary_values_CDA_Task[_filtered][_limit<N>].json` | flat list, one record per sample x proxy: `proxy, dataset, metric_key, gt_value, pred_value, gt_category, pred_category, image_file, slice_dim, slice_idx`. This is the re-analysis surface and what `cda_uncertainty.py` consumes; `image_file` is what makes cluster resampling possible |

Task-level, written in the task directory:

| File | Content |
|---|---|
| `summary_CDA_task[_<source>][_filtered][_canonical][_limit<N>].txt` | per-proxy leaderboards plus per-model detail (full confusion matrices, per-dataset breakdowns) |
| `summary_CDA_uncertainty[_<source>][_filtered].json` | run metadata (`n_boot`, `seed`, `ci_method`, `p_method`, `cluster_note`) plus `rows` of `{model, proxy, statistic, n, n_clusters, point_estimate, ci_lower, ci_upper, p_kappa_gt_zero, low_n, n_boot_valid}` |

Report, written wherever `--out` points:

`CDA_REPORT[_<source>].md` - **a renderer, not an analysis**. It reads the JSONs (never the `.txt` files, which
truncate for display), recomputes nothing, and emits no timestamp so an unchanged analysis regenerates byte-identically.

Suffix order is fixed: `_<source>` then `_filtered` then `_canonical` then `_limit<N>`. Two runs of different scope
therefore cannot overwrite each other. A/D artifacts never carry `_filtered`.

### Report structure

The repository ships one rendered example (`CDA_REPORT_llm-parsed-gemma-4-31b.md` in its CDA folder) whose sections
show the layout the renderer produces:

1. Title + "generated, do not edit by hand" note.
2. **Provenance** table: parsed source and the field it implies, each task directory with filtered/unfiltered status,
   each config file with its model count, and the uncertainty settings (resamples, seed, "resampling whole imaging
   volumes").
3. **Reading these tables**: the column legend, plus the two caveats that low `Nparsed` is flagged and that kappa is not
   comparable across proxies.
4. **Decision agreement**: one `###` section per proxy, each with its data source line and a table of
   `Model | Acc | AccCov | kappa | wkappa | 95% CI | p | Nparsed | Ntotal`, in config order.
5. **Regenerating this report**: the exact command.

A model missing the selected parsed folder appears in a **Not reported** block ("N of M configured models") rather than
being silently dropped, and if *no* model yielded metrics in either task the renderer raises `FileNotFoundError`
instead of emitting a complete-looking but empty report.

### Before you quote a number

- **Kappa is not comparable across proxies.** Both raters derive from the same continuous measurement, so agreement
  depends mostly on where the cutoff falls in the cohort's distribution. Compare models within a proxy.
- **Check `Nparsed` and the majority class.** Renal categories are far from uniform; a constant answer already scores a
  substantial accuracy. Kappa corrects for that, raw accuracy does not, and there is no majority-class baseline row.
- **The renal proxy stages a slice, not a tumour.** It assigns an AJCC T category to each 2D slice's in-plane major
  axis; the source records that 26.6% of renal slices carry a category different from their own tumour's
  max-over-slices category. Treat it as a measurement-sensitivity statistic, not staging accuracy.
- **Small-n rows are flagged, not suppressed.** A row with `n_parsed = 1` still prints a kappa, a zero-width CI and
  `p = 1.0`.

---

## 5. Running it

See `workflows.md` for the full recipes. The short form, all three steps in order:

```bash
bash scripts/run_cda.sh \
  --ad-task-dir ${benchmark_dir}/Results/<AD task> --ad-config scripts/cda/config-AD-CoT.yaml \
  --tl-task-dir ${benchmark_dir}/Results/<TL task> --tl-config scripts/cda/config-TL-CoT.yaml \
  --removed-samples-dir <data_dir>/Datasets \
  --out ${benchmark_dir}/CDA_REPORT.md --repo-root ${benchmark_dir}
```

Add `--dry-run` first to see the five commands without executing them. The step order is not optional: uncertainty reads
what the agreement step persisted, and the report reads what both persisted.
