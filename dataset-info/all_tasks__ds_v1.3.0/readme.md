Each JSON file maps a subtask name to its sample size (`{"task name": sample_size, ...}`), covering every subtask the MedVision benchmark can load at this dataset version.

Counts are produced by streaming each config from the Hugging Face MedVision dataset, so they are what an evaluation run actually sees — the per-sample quality/size filters are already applied (single-instance). A subtask that resolves to zero samples is omitted rather than listed as `0`. Regenerate with [`script/misc/regen_all_tasks_v1.2.0-v1.4.0.sh`](../../script/misc/regen_all_tasks_v1.2.0-v1.4.0.sh).

**A subtask is listed only if this pin can load it.** Which annotations a version may serve is decided by the loader's own `_ANNOTATION_INDEX` (what is published per dataset and plan kind) and `_PAUSED_ANNOTATIONS` (what is withheld), so these lists can be *narrower* than the aggregate counts in `dataset-info/datasets_summary_v<version>/`, which are computed from the local plans and model neither gate.

## What is in `all_tasks__ds_v1.3.0/`

- **Coverage**: Detection 29 datasets / 378 subtasks (24,749,253 samples) · T/L 12 datasets / 224 subtasks (47,725) · A/D 2 datasets / 10 subtasks (7,925).
- **vs v1.2.0**: adds MSWAL, plus MAMA-MIA and PI-CAI, which become loadable here via their `1.2.1` republish. No annotation logic changed.
- Detection and T/L both reconcile exactly against `datasets_summary_v1.3.0/`.

In every version below, **A/D covers only Ceph-Biometrics-400 and FeTA24** (5 subtasks per split): the AFIDs, PDDCA and VerSe biometry plans are listed in `_PAUSED_ANNOTATIONS` and cannot be loaded at any pin — the 2,064-annotation difference against the summaries. **Detection is byte-identical across releases** for a given dataset set — no detection plan has ever been regenerated — so only the datasets a release *adds* move that total.
