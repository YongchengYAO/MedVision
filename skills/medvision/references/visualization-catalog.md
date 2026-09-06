# Visualization Catalog (reference only)

## Purpose

Read this when a user asks for one of MedVision's figures (radars, per-sample
overlays, leaderboard timeline, label clouds, GT annotation panels, webpage
data exports). These scripts live in the repository's `script/visualization/`
directory and are **not bundled** in this skill: they assume the repository
layout (`Results/`, `Data/`, `Figures/`, roster YAMLs next to the scripts) and
many are specific to the paper or the project webpage. Use this catalog to pick
the right entry point and to know its inputs; then read the script's `--help`
in a checkout before running it.

All per-sample figures share one convention: the slice is rotated 90° CCW for
display and coordinates are transformed accordingly; the panel aspect ratio
follows the physical extent (pixel size × array size). The full specification is
distilled in `../sub-skills/biomedparse-ablation/references/visualization-convention.md`.

## Model-comparison figures (read `parsed/` or `llm-parsed_<judge>/` summaries)

| Entry point (`script/visualization/`) | What it draws | Main inputs |
| --- | --- | --- |
| `viz_radar.sh` → `viz_radar.py` | radar chart comparing models across metrics for one task | task `Results/` dir, roster YAML (`config-{detect,TL,AD}-CoT.yaml`), metric list |
| `viz_radar_batch.sh`, `viz_radar_batch_leaderboard.sh` | all radars for Detection, A/D, T/L | same, batched; leaderboard variant uses the published roster |
| `viz_radar_grid.sh` → `viz_radar_grid.py` | per-model grid: one row per model, six radars (Detection R/P/F1, A/D Angle/Distance MRE, T/L MRE) with per-sample violin + box overlays | summaries + `summary_values_*` |
| `viz_benchmark_leaderboard_timeline.sh` → `.py` | accuracy vs. model release date, one panel per task (Detection IoU; T/L, Distance, Angle as 1/MRE) | summaries + release-date table |
| `viz_detection_sampleSize_per_label_x_boxSize.sh` | detection metrics and sample distribution per label × box-to-image-ratio group | `config-detect-sampleSize-per-label-boxSize.yaml`, box-size analysis outputs |
| `viz_OOD_label.sh` → `viz_OOD_label.py` | 2×2 "target @ modality" label clouds, in-distribution vs target-OOD rosters, Detection and T/L | task lists |
| `viz_compile_grid_batch.sh` → `viz_compile_grid.py` | tiles pre-generated per-sample overlays across models into comparison grids | per-sample figure folders |

## Per-sample figures (need `Data/` images and `parsed/` records)

| Entry point | What it draws |
| --- | --- |
| `viz_detection_boxes.sh` → `viz_detection_boxes.py` | image + GT box (green) + predicted box (orange) |
| `viz_detection_responses.sh` → `.py` | prompt / response / GT panel per detection sample |
| `viz_tl_axes.sh` → `viz_tl_axes.py` | image + mask contour + GT axes (dashed) + predicted axes, L-shaped scale bar |
| `viz_tl_responses.sh` → `.py` | prompt / response / GT panel per T/L sample |
| `viz_ad_landmarks.sh` → `viz_ad_landmarks.py` | image + GT and predicted landmarks and lines |
| `viz_ad_responses.sh` → `.py` | prompt / response / GT panel per A/D sample |
| `viz_ellipse_fit_comparison.sh` → `.py` | image-space vs real-space ellipse fit on anisotropic coronal/sagittal slices (with and without aspect correction) |
| `run-viz-samples.sh`, `run-viz-resp.sh` | thin batch drivers for the T/L sample/response figures |

## Ground-truth-only figures (need `Data/`, no model outputs)

| Entry point | What it draws |
| --- | --- |
| `viz_gt_annotations.sh` → `viz_gt_annotations.py` | one compiled figure with a labelled row block per task, rendered from the on-disk benchmark plans |
| `viz_planeOOD_samples.sh` → `viz_planeOOD_samples.py` | the same volume and target in the axial (in-distribution) plane and both OOD planes; Detection and T/L only |

## Webpage data exports (project page `medvision-vlm.github.io`)

`export_annotation_preview.sh`, `export_benchmark_leaderboard_timeline_data.sh`,
`export_demo_gallery.sh`, `export_detection_performance_per_boxImgRatio_data.sh`,
`export_detection_sampleSize_per_label_x_boxSize_data.sh`, `export_pilot_cases.sh`,
`export_webpage_cases.sh`, `export_violin_data.sh`, and the Python
`export_explorer_data.py`, `export_radar_data.py`, `export_violin_data.py`,
`export_webpage_cases.py` write JavaScript/JSON data files for the interactive
webpage. They take a `--page_dir` (webpage checkout) and read `Results/` and
`dataset-info/`; they are maintainer tooling and out of scope for this skill.

## Utility

`figure_concat.sh` → `figure_concat.py`: combine PNG/PDF panels into one m×n
composite (needs `pypdf`).

## Roster YAMLs

`config-{detect,TL,AD}-CoT.yaml` (main roster), `config-*-planeOOD.yaml`,
`config-*-taskOOD.yaml`, `config-*--SFT-RFT.yaml` (ablation rosters),
`config-TL-pilot-CoT.yaml` (API pilot) map `Results/` model folder names to
display names. The LLM-judge pipeline uses the same YAML shape; see
`../sub-skills/llm-judge-parsing/references/recipes.md`.

## Figure-file conventions

Figures are saved with `dpi=300` and clamped below the arXiv 34-megapixel cap;
line art goes to PDF, image panels to PNG. Colours for GT/prediction/axes are
constants in `medvision_bm.utils.configs` (`C_GT_BOX`, `C_PRED_BOX`, `C_GT_MAJOR`, …).
