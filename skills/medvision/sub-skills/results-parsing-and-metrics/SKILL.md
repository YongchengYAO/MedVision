---
name: results-parsing-and-metrics
description: "Turns raw MedVision evaluation JSONL files into per-sample metrics and per-model / cross-model summaries (parse_outputs, summarize_AD_task, summarize_TL_task, summarize_detection_task) and explains every metric exactly: SuccessRate, MAE, MRE, nMAE, IoU/F1/Precision/Recall, MRE<k / IoU>k / Acc@IoU thresholds, NaN-vs-0 failure handling, anatomy/modality/box-size grouping, the A/D near-zero-GT filter, T/L removed-samples filtering, and re-summarizing LLM-judge output via --parsed_dirname / --resps_key. Use when a user asks to parse results, compute or compare benchmark metrics, read summary_*.json or summary_*_task.txt files, debug a suspicious number, or dedupe result JSONLs."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision results: parsing, summarizing, and metric interpretation

Use this sub-skill after evaluation has produced `Results/<task_tag>/<model_name>/<ts>_samples_<config>.jsonl` files. It covers
benchmark steps 2 (parse) and 3 (summarize) of the `medvision_bm` pipeline for the three task families - Detection (bounding boxes),
Tumor/Lesion size (T/L, mm) and Angle/Distance (A/D, degrees or mm) - and the exact meaning of every number those steps write.
Everything here runs on CPU without network; both steps import `torch`, `transformers` and `medvision_ds` (the vendored eval
utilities), so those must be installed - CPU builds are enough.

## Quick start

```bash
# step 2: extract the <answer> numbers and score every sample (k = 1 A/D, 2 T/L, 4 Detection)
python -m medvision_bm.benchmark.parse_outputs --task_type TL --task_dir Results/MedVision-TL -p 8
# step 3: per-model summary_metrics_*.json / summary_values_*.json + task-level summary_TL_task.txt
python -m medvision_bm.benchmark.summarize_TL_task --task_dir Results/MedVision-TL -p 8 \
    --removed_samples_dir <data_dir>/Datasets --skip_model_wo_parsed_files
# same two steps in one call (--dry-run prints the commands first)
scripts/parse_and_summarize.sh --task-type TL --task-dir Results/MedVision-TL -p 8 --removed-samples-dir <data_dir>/Datasets
# look at what was written
python scripts/inspect_summary.py --path Results/MedVision-TL/<model_name>/parsed
```

Replace `TL`/`summarize_TL_task` by `AD`/`summarize_AD_task` or `Detection`/`summarize_detection_task`. `--removed_samples_dir` exists
for T/L only.

## Route by task

| Need | Read / run |
|---|---|
| Run or script the parse -> summarize steps, recompute on `llm-parsed_<judge>/`, pilot `--limit` comparisons, dedupe, multiprocessing | `references/workflows.md` |
| Exact flags and help text of `parse_outputs`, the three summarizers, `remove_duplicate_samples`, the box-size helpers | `references/cli-reference.md` |
| What a metric means, its formula, denominator, failure handling, units, grouping keys, the near-zero and removed-sample filters, the scaledPS note | `references/metrics.md` |
| File names, suffix rules (`_filtered`, `_limit<N>`, `__<parsed_dirname>`), parsed-record keys, summary JSON shapes | `references/output-files.md` |
| Call the scoring/grouping functions from Python (`cal_metrics`, `cal_IoU`, `group_by_*`, summarizer helpers, eval-time `process_results_*`) | `references/api-reference.md` |
| An error message, an all-NaN column, a number that "looks wrong" | `references/troubleshooting.md` |
| Prove the failure semantics on the installed package (exit 1 if they changed) | `python scripts/metrics_demo.py` |
| Pretty tables of any `summary_*.json`, `<ts>_results.json`, or a parsed folder inventory (duplicates, success counts) | `python scripts/inspect_summary.py --path <file-or-dir>` |
| Run step 2 + 3 with consistent flags | `scripts/parse_and_summarize.sh --help` |

## Facts to keep straight (all verified against the package)

- **Answer scope.** Only numbers inside the first `<answer>...</answer>` block count; the last k of them are the prediction. No block,
  or fewer than k numbers, is a *failure* (`SuccessRate.success = False`) even when the response is correct elsewhere. The LLM-judge
  pass (`../llm-judge-parsing/SKILL.md`) exists to separate "cannot measure" from "did not follow the format".
- **Failure values.** A/D and T/L: MAE and MRE are NaN and excluded from `avgMAE`/`avgMRE`. Detection: MAE is NaN but
  IoU, F1, Precision, Recall are **0**, so the reported `IoU`/`F1`/`Precision`/`Recall` means are over ALL samples
  (`IoU_reported = IoU_over_successes * SuccessRate`).
- **Threshold keys divide by the total sample count.** `MRE<k`, `MAE<k`, `IoU>k` (code uses `>=`), `Acc@IoU>=tau` and
  `Acc@IoU[0.50:0.95]` mix instruction-following with accuracy; `MRE<1.0` equals `SuccessRate` because the last bucket is `[0.9, inf)`;
  `Acc@IoU>=0.50` equals `IoU>0.5`.
- **nMAE** = MAE / physical diagonal of the slice (`sqrt((H*px_h)^2 + (W*px_w)^2)`, mm from the NIfTI header). T/L always;
  A/D only for distances (angles: NaN). Needs the image files at the recorded `doc.image_file` paths at parse time.
- **A/D near-zero ground truth** (`AD_NEAR_ZERO_GT_THRESHOLD = 0.1`) is dropped by `summarize_AD_task` before counting;
  `parse_outputs` does not apply it.
- **Grouping.** Detection: `"<anatomy group> @ <MR|CT|US|XR|PET> (<S|C|A>)"` via `label_map_regroup`; T/L: renamed label via
  `label_map_rename`; A/D: `"<dataset>_<metric_type>_<metric_key>"`. Detection report rows split into `anatomy` vs `T/L`
  (`TUMOR_LESION_GROUP_KEYS = ["tumor", "lesion", "metastatic"]`), dropping `EXCLUDED_KEYS = ["miscellaneous", "others"]` and groups
  with fewer than `MINIMUM_GROUP_SIZE = 50` samples. Cross-group rows are sample-weighted means.
- **Units.** T/L in mm, A/D in mm (distance) or degrees (angle), detection boxes as relative `[x_min, y_min, x_max, y_max]` in [0, 1]
  with the origin at the lower-left corner of the image.
- **Judge output.** `llm-parsed_<judge>/` records carry `LLM_filtered_resps`, never `filtered_resps`; summarize them with
  `--parsed_dirname llm-parsed_<judge> --resps_key LLM_filtered_resps` (the summarizer aborts otherwise). Outputs get a
  `__<parsed_dirname>` qualifier so published reports are untouched; no re-parse is needed.
- **Roster hygiene.** Result trees hold many more model folders than a study reports; use `--models <names>` and
  `--skip_model_wo_parsed_files`, and keep `--limit`/`--removed_samples_dir`/`--parsed_dirname` identical across the models of one report.

## Bundled scripts

- `scripts/parse_and_summarize.sh` - runs `parse_outputs` then the matching summarizer; maps `--task-type` to k and module, refuses
  `--removed-samples-dir` outside T/L, skips the parse step for non-default `--parsed-dirname`, `--dry-run` prints both commands.
- `scripts/metrics_demo.py` - synthetic detection/T/L/A/D fixture through the real `cal_metrics` and summarizer counters; prints the
  NaN-vs-0 behaviour, the total-count denominators and the near-zero filter; non-zero exit if semantics drift.
- `scripts/inspect_summary.py` - read-only tables for every summary file type, sample-weighted overall row, parsed-folder inventory
  with duplicate `doc_id` detection.

## Boundaries

- Running evaluations (GPU/API), launcher anatomy, token budgets, resume caches: `../benchmark-evaluation/SKILL.md`.
- Judge environment, roster YAML, `run_llm_parsing.sh` steps: `../llm-judge-parsing/SKILL.md`; this sub-skill only consumes its
  `llm-parsed_<judge>/` folders.
- Clinical Decision Agreement, process/equation accuracy, detection-by-target-size interpretation: `../analysis/SKILL.md`
  (the box-size analyzers are only pointed at here).
- Figures (radars, overlays, box-size plots): `../../references/visualization-catalog.md`.
- Installing `medvision_bm`/`medvision_ds` and version pins: `../environment-setup/SKILL.md`; task names and dataset configs:
  `../dataset-and-tasks/SKILL.md`; shared vocabulary: `../../references/concepts-and-glossary.md`; cross-cutting failures:
  `../../references/troubleshooting.md`.

## Safe operating rules

- Steps 2 and 3 only write inside `<model_name>/parsed/` (or the chosen `--parsed_dirname`) and the task folder; `--rm_old` deletes
  `parsed/` first - never point it at a tree whose parsed files you cannot regenerate.
- Treat `Results/` as data: do not hand-edit JSONLs; use `remove_duplicate_samples` into a new `--out_dir`.
- Do not `pip install` into an evaluation environment to satisfy an import; the pinned model stacks break easily (see
  `../environment-setup/SKILL.md`).
