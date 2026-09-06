# Maintainer Workflows: dataset-info Catalogues

Reference-only. These tools stream configs from Hugging Face or read the whole local `Datasets/` tree, so they need network and/or a complete data root and can run for hours. All CLIs below were checked with `--help`; the shell recipes are described from the repository's `script/misc/` launchers, which are not bundled because they hard-code a checkout layout.

## `dataset-info/` contents

| Path | Produced by | Content |
| --- | --- | --- |
| `dataset-configs/<version>/ConfigurationsList_{All,Test,Train}.csv` | dataset release | every config name for that release (no header) |
| `all_tasks__ds_v<version>/tasks_MedVision-*__{Test,Train}.json` | `configs_to_tasks` (<= 1.1.1) / `regen_all_tasks.py` (1.2.0+) | task names with HF-exact sample counts per plane and split |
| `datasets_summary_v<version>/` | `summarize_datasets` | `dataset_files.jsonl`, `dataset_summary_{filtered,raw}.json`, `dataset_summary.csv`, `dataset_label_stats.csv`, figures |
| `image_sizes__ds_v1.0.0/`, `pixel_sizes__ds_v1.0.0/` | `configs_to_image_sizes` / `configs_to_pixel_sizes` | per-task `image_size_2d` / `pixel_size` distributions (+ `__summary.json`) |
| `datasets_info.json`, `datasets_info_link_audit.json` | `compile_dataset_info.py` | per-dataset metadata (website, data links, licence, paper, HF mirrors, notes) for the web explorer |

Version pins used when the shipped files were generated: `all_tasks__ds_v1.0.0` with `MedVision_PLANNER_VERSION=1.0.0` and `MedVision_ACK_RELEASE=1.1.1` (the ACK value must now be `1.4.0` or the dataset's newest version).

## `python -m medvision_bm.utils.configs_to_tasks` (config CSV -> task JSON)

```text
--data_dir DIR --configs_csv CSV --out OUT.json [--families BoxSize,MaskSize,TumorLesionSize,BiometricsFromLandmarks]
[--planes Axial,Coronal,Sagittal] [--split {train,test,all}] [--cot] [--limit N] [--no-count] [--no-streaming]
```

Selects CSV rows by `parts[1]` (family), `parts[-2]` (plane), `parts[-1]` (split); converts each config with `config_to_task` (strip split, `BoxSize -> BoxCoordinate`, `-CoT` if `--cot`); counts by streaming `load_dataset(..., streaming=True)` after dropping non-scalar columns (avoids an Arrow cast failure on `bounding_boxes`, whose data carries an extra `mask_image_ratio` field the schema omits). `--no-count` writes `0` counts (naming-only run); `--no-streaming` uses `len(load_dataset(...))` and materialises the Arrow cache. Version comes from `MedVision_PLANNER_VERSION`, not a flag. Recipe: 14 invocations, one per `{family} x {plane} x {split}` with `--cot` only for the test split.

## `python -m medvision_bm.utils.configs_to_{image,pixel}_sizes`

Same parser (`size_dist_utils.build_parser`) and streaming loop; bucket `image_size_2d` as `"HxW"` or `pixel_size` as `"h.hhhxw.www"` mm; write the distribution JSON plus a `__summary.json` (square vs non-square / isotropic vs anisotropic counts, min/max/weighted-median). The repository launchers mirror the `configs_to_tasks` recipe with `imagesizes_`/`pixelsizes_` filename stems. T/L distributions differ by version (pin as needed).

## `python -m medvision_bm.utils.summarize_datasets` (local plans -> summary)

```text
--data_dir DIR [--datasets A,B] [--out_dir DIR] [--plan_version X.Y.Z] [--no_detection] [--viz] [--viz_only] [--reuse_from DIR]
```

Reads `benchmark_plan_*.json.gz` only (no HF). Segmentation plan = case inventory and ROI areas; biometry plan = T/L and A/D measurements; detection plans are read only for BoxSize benchmark counts (`--no_detection` skips them: ~8.5 min instead of ~34 min and far less memory, but BoxSize counts are then absent). `--plan_version` resolves every kind with the `resolve_plan_path` ceiling, so an older pin reproduces that release's summary and skips datasets that did not exist yet. `--reuse_from <older summary dir>` recomputes only biometry for a new version. `--viz` renders `dataset_summary.pdf/.svg`, two-ring donuts and a word cloud (the word cloud needs `pip install wordcloud`, not a declared dependency; it is skipped with a warning otherwise). `labels_map` is taken live from `medvision_ds` when importable (may import `nibabel`), else from the plan. Repository launcher variables: `DATA_DIR`, `PLAN_VERSION` (default `1.2.0` there), `OUT_DIR`, `REUSE_FROM`.

## `regen_all_tasks.py` (repository `script/misc/`, not bundled)

```text
--version X.Y.Z (required) [--data_dir DIR] [--medvision_py PATH] [--dataset_path YongchengYAO/MedVision|<local checkout>]
[--cache dataset-info/.all_tasks_counts_cache.json] [--out_dir DIR] [--no-count]
```

Regenerates `all_tasks__ds_v<version>/` for 1.2.0+: parses `_ANNOTATION_INDEX`, `_PAUSED_ANNOTATIONS` and the `MedVisionConfig(name=...)` list **from a local copy of `MedVision.py`** (so it needs a checkout of the dataset repo, not just the installed package), resolves each config's annotation version at the pin, and counts by streaming, caching counts keyed by `config|resolved version` so detection plans (identical in every release) are streamed once. Sets `MedVision_PLANNER_VERSION=<version>` and `MedVision_ACK_RELEASE=<newest indexed version>` itself; defaults `MedVision_FORCE_INSTALL_CODE=false`. Zero-count subtasks are dropped. The companion shell loop runs `1.2.0 1.3.0 1.4.0` oldest-first with `PYTHONPATH=<repo>/src` so a site-packages copy of `medvision_bm` cannot shadow the checkout, and takes `DATA_DIR`, `DATASET_PATH`.

## `compile_dataset_info.py` (repository `script/misc/`, not bundled)

```text
[--out_dir dataset-info] [--audit_links] [--medvision_ds_src <path to medvision_ds source tree>]
```

Imports every dataset's `preprocess_{segmentation,detection,biometry}` module from `medvision_ds`, asserts the `dataset_info` dicts agree, normalises licence spellings, derives HF redistribution links from the download scripts (`download_fast.py`, then `download.py`; private mirrors and non-redistributions removed), attaches hand-written access notes for FeTA24/SKM-TEA/ToothFairy2, and writes `datasets_info.json` (+ link audit with `--audit_links`). **Guard**: after inserting `--medvision_ds_src` at the front of `sys.path` it checks `medvision_ds.__file__.startswith(args.medvision_ds_src)` and aborts with `refusing to compile from a possibly stale installed copy` when the import still resolved to a site-packages copy. Always point `--medvision_ds_src` at the source tree you are editing.

## Practical notes

- Streaming counts for the large detection datasets take hours; reuse caches (`.all_tasks_counts_cache.json`) and pin `MedVision_FORCE_INSTALL_CODE=false`.
- Catalogue lists can be narrower than the summaries: the summaries model neither the paused table nor zero-sample subtasks.
- Detection counts in `all_tasks__ds_v*` reconcile exactly with `datasets_summary_v*`; A/D differs by the paused AFIDs/PDDCA/VerSe plans (2,064 annotations).
