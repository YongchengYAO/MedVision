Each JSON file maps a subtask name to its sample size (`{"task name": sample_size, ...}`), covering every subtask the MedVision benchmark can load at this dataset version.

Counts are produced by streaming each config from the Hugging Face MedVision dataset, so they are what an evaluation run actually sees — the per-sample quality/size filters are already applied (single-instance). A subtask that resolves to zero samples is omitted rather than listed as `0`. Regenerate with [`script/misc/regen_all_tasks_v1.2.0-v1.4.0.sh`](../../script/misc/regen_all_tasks_v1.2.0-v1.4.0.sh).

**A subtask is listed only if this pin can load it.** Which annotations a version may serve is decided by the loader's own `_ANNOTATION_INDEX` (what is published per dataset and plan kind) and `_PAUSED_ANNOTATIONS` (what is withheld), so these lists can be *narrower* than the aggregate counts in `dataset-info/datasets_summary_v<version>/`, which are computed from the local plans and model neither gate.

## What is in `all_tasks__ds_v1.2.0/`

- **Coverage**: Detection 26 datasets / 360 subtasks (24,615,925 samples) · T/L 9 datasets / 182 subtasks (35,838) · A/D 2 datasets / 10 subtasks (7,925).
- **vs v1.1.1**: adds DEEP-PSMA, LIDC-IDRI, LNQ2023, PDDCA, TopCoW24 and VerSe. T/L annotations themselves are unchanged from v1.1.1 for the datasets both releases contain.
- ⚠️ **MAMA-MIA and PI-CAI are absent.** Their v1.2.0 annotations were withdrawn (missing RAS+ reorientation) and republished as `1.2.1`, so a `1.2.0` pin cannot load them. `datasets_summary_v1.2.0/` still counts them — 73,222 Detection and 3,722 T/L annotations — which is exactly why that summary is larger than these lists.

In every version below, **A/D covers only Ceph-Biometrics-400 and FeTA24** (5 subtasks per split): the AFIDs, PDDCA and VerSe biometry plans are listed in `_PAUSED_ANNOTATIONS` and cannot be loaded at any pin — the 2,064-annotation difference against the summaries. **Detection is byte-identical across releases** for a given dataset set — no detection plan has ever been regenerated — so only the datasets a release *adds* move that total.
