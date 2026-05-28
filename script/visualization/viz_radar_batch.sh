#!/bin/bash
# Plot all radar figures for Detection, AD, and TL tasks using viz_radar.py.
#
# Usage:
#   bash viz_radar_batch.sh
#
# Override defaults via environment variables:
#   RESULTS_DIR=<path>    Root results directory (default: <MEDVISION_DIR>/Results)
#   FIG_DIR=<path>        Output directory for figures (default: <MEDVISION_DIR>/Figures)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-$MEDVISION_DIR/Results}"
FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"

# Verbose model names for violin overlays
VERBOSE_DETECT="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250_CoT"
VERBOSE_AD_TL="MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250"

# ── Detection ─────────────────────────────────────────────────────────────────

# Detection, all models, 3 metrics
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type Detection \
    --config_yaml "$SCRIPT_DIR/config-detect-CoT.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-detect-v2" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_detection-v2.png \
    --metrics_list Recall Precision F1 \
    --show_label_name \
    --label_col 2 \
    --legend_col 2

# Detection, SFT-RFT, 3 metrics, violin overlay
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type Detection \
    --config_yaml "$SCRIPT_DIR/config-detect-CoT--SFT-RFT.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-detect-v2" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_detection--SFT-RFT-v2.png \
    --metrics_list Recall Precision F1 \
    --verbose_model "$VERBOSE_DETECT" \
    --show_label_name \
    --label_col 2 \
    --legend_col 1

# Detection, plane-OOD
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type Detection \
    --config_yaml "$SCRIPT_DIR/config-detect-CoT-planeOOD.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-detect-v2-CoT-planeOOD" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_detection-planeOOD.png \
    --metrics_list Recall Precision F1 IoU \
    --show_label_name \
    --label_col 3 \
    --legend_col 2

# Detection, task-OOD
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type Detection \
    --config_yaml "$SCRIPT_DIR/config-detect-CoT-taskOOD.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-detect-v2-CoT-taskOOD" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_detection-taskOOD.png \
    --metrics_list Recall Precision F1 IoU \
    --show_label_name \
    --label_col 3 \
    --legend_col 2

# ── AD ────────────────────────────────────────────────────────────────────────

# AD-CoT, all models
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type AD \
    --config_yaml "$SCRIPT_DIR/config-AD-CoT.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-AD-v2-CoT" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_AD-CoT.png \
    --metrics_list avgMRE \
    --show_label_name \
    --label_col 2 \
    --legend_col 2

# AD-CoT, SFT-RFT, violin overlay
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type AD \
    --config_yaml "$SCRIPT_DIR/config-AD-CoT--SFT-RFT.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-AD-v2-CoT" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_AD-CoT--SFT-RFT.png \
    --metrics_list avgMRE \
    --verbose_model "$VERBOSE_AD_TL" \
    --show_label_name \
    --label_col 2 \
    --legend_col 2

# ── TL ────────────────────────────────────────────────────────────────────────

# TL-CoT, all models
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type TL \
    --config_yaml "$SCRIPT_DIR/config-TL-CoT.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-TL-v2-CoT" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_TL-CoT.png \
    --metrics_list avgMRE \
    --show_label_name \
    --label_col 1 \
    --legend_col 1

# TL-CoT, SFT-RFT, violin overlay
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type TL \
    --config_yaml "$SCRIPT_DIR/config-TL-CoT--SFT-RFT.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-TL-v2-CoT" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_TL-CoT--SFT-RFT.png \
    --metrics_list avgMRE \
    --verbose_model "$VERBOSE_AD_TL" \
    --show_label_name \
    --label_col 1 \
    --legend_col 1

# TL-CoT, plane-OOD, violin overlay
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type TL \
    --config_yaml "$SCRIPT_DIR/config-TL-CoT-planeOOD.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-TL-v2-CoT-planeOOD" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_TL-CoT-planeOOD.png \
    --metrics_list avgMRE \
    --verbose_model "$VERBOSE_AD_TL" \
    --show_label_name \
    --label_col 1 \
    --legend_col 1

# TL-CoT, task-OOD, violin overlay
python "$SCRIPT_DIR/viz_radar.py" \
    --task_type TL \
    --config_yaml "$SCRIPT_DIR/config-TL-CoT-taskOOD.yaml" \
    --task_dir "$RESULTS_DIR/MedVision-TL-v2-CoT-taskOOD" \
    --fig_dir "$FIG_DIR" \
    --fig_name fig_TL-CoT-taskOOD.png \
    --metrics_list avgMRE \
    --verbose_model "$VERBOSE_AD_TL" \
    --show_label_name \
    --label_col 1 \
    --legend_col 1
