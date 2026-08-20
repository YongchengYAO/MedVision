Each JSON file maps a subtask name to its sample size (`{"task name": sample_size, ...}`), covering every subtask the MedVision benchmark can load at this dataset version.

Counts are produced by streaming each config from the Hugging Face MedVision dataset, so they are what an evaluation run actually sees — the per-sample quality/size filters are already applied (single-instance). A subtask that resolves to zero samples is omitted rather than listed as `0`. Regenerate with [`script/misc/regen_all_tasks_v1.2.0-v1.4.0.sh`](../../script/misc/regen_all_tasks_v1.2.0-v1.4.0.sh).

**A subtask is listed only if this pin can load it.** Which annotations a version may serve is decided by the loader's own `_ANNOTATION_INDEX` (what is published per dataset and plan kind) and `_PAUSED_ANNOTATIONS` (what is withheld), so these lists can be *narrower* than the aggregate counts in `dataset-info/datasets_summary_v<version>/`, which are computed from the local plans and model neither gate.

## What is in `all_tasks__ds_v1.4.0/`

- **Coverage**: Detection 29 datasets / 378 subtasks (24,749,253 samples) · T/L 12 datasets / 228 subtasks (966,189) · A/D 2 datasets / 10 subtasks (7,925). The 228 T/L subtasks are the full grid: 38 T/L tasks x 3 planes x 2 splits.
- **vs v1.3.0**: T/L only, and the change is large — **47,725 → 966,189 samples (20x)** across all 12 T/L datasets. Clusters are now selected by a physical size floor in millimetres rather than a raw pixel count, a gate that silently discarded rotated ellipses is removed, and the ellipse fit is guarded against degenerate results. See the [v1.4.0 release note](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/release-v1.4.0.md).
- ⚠️ **The train/test split moved on six datasets** (HNTSMRG24, KiPA22, KiTS23, MSD, autoPET-III, BraTS24): case counts per split are unchanged, but which cases land on each side differs. Do not compare a v1.4.0 test-split metric against a pre-1.4.0 one on those six.
- Detection and T/L both reconcile exactly against `datasets_summary_v1.4.0/`.

In every version below, **A/D covers only Ceph-Biometrics-400 and FeTA24** (5 subtasks per split): the AFIDs, PDDCA and VerSe biometry plans are listed in `_PAUSED_ANNOTATIONS` and cannot be loaded at any pin — the 2,064-annotation difference against the summaries. **Detection is byte-identical across releases** for a given dataset set — no detection plan has ever been regenerated — so only the datasets a release *adds* move that total.
