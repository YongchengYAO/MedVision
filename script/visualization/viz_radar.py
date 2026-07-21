"""
Plot radar charts comparing model performance across multiple metrics.

Reads per-task summary JSON files from each model's parsed/ directory and
renders one polar subplot per metric, with each model drawn as a separate
trace. Models and their display names are loaded from a YAML config file.

Optionally overlays per-sample violin + box plots on radar spokes for
selected models (requires *_samples_*.jsonl in each model's parsed/ dir).

Supports AD, TL, and Detection task types.

Usage:
    python viz_radar.py \
        --task_type [AD|TL|Detection] \
        --config_yaml <path/to/config.yaml> \
        --task_dir <path/to/task_dir> \
        --fig_dir <output_dir> \
        --fig_name <output.pdf> \
        [--metrics_list METRIC1 METRIC2 ...] \
        [--verbose_model MODEL1 [MODEL2 ...]] \
        [--show_scatter] \
        [--show_label_name] \
        [--radar_cell_inches N] \
        [--label_col N] \
        [--legend_col N] \
        [--save_as_png] [--save_as_pdf]

Output formats:
    - No flags (what viz_radar.sh and viz_radar_batch.sh do) → ["pdf"] —
      exactly what the line-plots→PDF rule wants.
    - --save_as_png → ["png"] only — an explicit override for when you need
      a raster copy.
    - --save_as_png --save_as_pdf → both files written, one per format, via
      the extension swap in plot_metrics_multi_model.

Config YAML format:
    model_display_name:
      <model_dir_name>: "<Display Name>"
      ...
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for compatibility
from math import ceil, pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory

from medvision_bm.utils.configs import (
    C_ANATOMY_LABEL,
    C_TUMOR_LESION_LABEL,
    EXCLUDED_KEYS,
    MINIMUM_GROUP_SIZE,
    radar_model_colors,
)
from medvision_bm.utils.plot_utils import save_fig_capped

# Model colours, matched to the box-to-image-ratio figure
# (medvision_bm.benchmark.viz_detection_performance_per_boxImgRatio, produced by
# script/analyze/detection--target-size/run_analysis.sh) so the two read as one family:
# tab10 in order, then each reused hue DARKENED to 80% on the wrap.
# `configs.radar_model_colors` is byte-identical to `plt.cm.tab10.colors`, so it is used here to
# keep the palette config-driven rather than reaching into a matplotlib colormap.
#
# NOT configs.extend_palette: that applies its own alternating lighter/darker HLS shift, which
# would diverge from the reference for every model past the 10th.
_MODEL_COLORS = radar_model_colors
_WRAP_DARKEN = 0.8

# Legend text. One size for every task; a multi-column legend is then scaled to the label block's
# width by plot_label_mapping (see there). The band keeps that fit from running away when a config
# has very few / very many models -- it is a guard, not the usual outcome.
_LEGEND_FONTSIZE = 16
_LEGEND_FONTSIZE_RANGE = (7.0, 22.0)

# Angle box plot. The 45° labels are 3 lines deep, and each extra line offsets diagonally into the
# next box's slot, so the per-model width is what keeps them from colliding — raise it, not the
# rotation, if a longer model name ever overlaps.
_BOX_FACE_ALPHA = 0.7
_BOX_LABEL_ROTATION = 45
_BOX_INCHES_PER_MODEL = 1.05


def model_palette(n):
    """``n`` model colours: ``_MODEL_COLORS`` in order, reused hues darkened by ``_WRAP_DARKEN``.

    Public because ``export_radar_data.py`` imports it, so the PDF radar and the interactive web
    radar stay on one palette definition.

    Mirrors the reference figure's rule exactly (viz_detection_performance_per_boxImgRatio, the
    ``base_colors``/``darker_color`` block). Note the darkening does NOT compound: it is applied to
    every index past the first cycle, so >= 2 full wraps (21+ models) would repeat a colour. With
    18 models the list wraps once and every colour is distinct.
    """
    from matplotlib.colors import to_hex, to_rgb

    out = []
    for i in range(n):
        base = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        if i < len(_MODEL_COLORS):
            out.append(base)
        else:
            out.append(to_hex(tuple(c * _WRAP_DARKEN for c in to_rgb(base))))
    return out


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Configuration dictionary containing model list and optional
        model display-name mappings.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_metrics(json_path, metrics_list, task_type="Detection"):
    """Load and process metrics from a single model's JSON file.

    Args:
        json_path: Path to JSON file containing model metrics
        metrics_list: List of metric names to extract from the metrics dict

    Returns:
        pd.DataFrame: DataFrame with Target, selected metrics, and num_samples columns
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    first_target = next(iter(data))
    print(f"\n[Info] Loaded metrics from: {json_path}")
    print(
        f"[Info] Available metrics:\n{list(data[first_target].keys())}\n[Info] You can pass these metric names to --metrics_list argument to visualize them."
    )

    targets = []
    metrics_data = {metric: [] for metric in metrics_list}
    num_samples = []

    for target, metrics in data.items():
        if task_type in ("Detection", "TL") and any(
            k in target.lower() for k in EXCLUDED_KEYS
        ):
            continue
        for metric in metrics_list:
            if metric not in metrics:
                raise KeyError(
                    f"Metric '{metric}' not found in metrics dict for target '{target}'. Available metrics: {list(metrics.keys())}"
                )
        targets.append(target)
        for metric in metrics_list:
            metrics_data[metric].append(metrics[metric])
        num_samples.append(metrics["num_samples"])

    df_dict = {"Target": targets}
    df_dict.update(metrics_data)
    df_dict["num_samples"] = num_samples

    df = pd.DataFrame(df_dict)
    return df.sort_values(
        metrics_list[0] if metrics_list else "Target", ascending=False
    )


def is_purple_label(label_name):
    """Determine if a label should be colored purple (tumor/lesion-related).

    Args:
        label_name: Label name to check

    Returns:
        bool: True if label contains tumor/lesion-related terms
    """
    text = label_name.lower()
    return any(
        term in text
        for term in [
            "tumor",
            "cancer",
            "cyst",
            "stroke",
            "lesion",
            "resection cavity",
            "edema",
            "metastatic",
            "vestibular schwannoma",
        ]
    )


def _reconstruct_detect_label(doc):
    """Reconstruct the Detection summary-JSON label key from a parsed JSONL sample's doc.

    Detection samples carry ``doc["label"]`` (integer index) directly — unlike TL
    where the target label is looked up from the benchmark plan.

    Returns the label string, or None if it cannot be reconstructed.
    """
    from medvision_bm.utils.configs import label_map_regroup
    from medvision_bm.utils.parse_utils import (
        get_labelsMap_imgModality_from_seg_benchmark_plan,
    )

    dataset_name = doc.get("dataset_name")
    task_id_raw = doc.get("taskID")
    slice_dim = doc.get("slice_dim")
    label_idx = doc.get("label")
    if (
        dataset_name is None
        or task_id_raw is None
        or slice_dim is None
        or label_idx is None
    ):
        return None

    task_id = int(task_id_raw)
    try:
        result = get_labelsMap_imgModality_from_seg_benchmark_plan(
            dataset_name, task_id
        )
    except (ValueError, Exception):
        return None
    if not result or isinstance(result, dict):
        return None
    labels_map, img_modality = result
    if not labels_map:
        return None
    label_name = labels_map.get(str(label_idx))
    if label_name is None:
        return None

    # Detection groups fine-grained labels into parent classes (e.g. "vestibular
    # schwannoma" → "Head-Neck Tumor/Lesion") matching the summary JSON keys.
    parent_class = label_map_regroup.get(label_name)
    if parent_class is None:
        return None

    modality_map = {
        "MRI": "MR",
        "CT": "CT",
        "ultrasound": "US",
        "X-ray": "XR",
        "PET": "PET",
    }
    modality = modality_map.get(img_modality, img_modality)
    slicetype_map = {0: "S", 1: "C", 2: "A"}
    slicetype = slicetype_map.get(slice_dim)
    if slicetype is None:
        return None

    return f"{parent_class} @ {modality} ({slicetype})"


def _reconstruct_tl_label(doc):
    """Reconstruct the TL summary-JSON label key from a parsed JSONL sample's doc.

    Mirrors the logic in ``group_by_label_modality_slice`` in parse_utils.py:
        ``label_map_rename[label_name] + " @ " + modality + " (" + slicetype + ")"``

    Returns the label string, or None if the label cannot be reconstructed.
    """
    from medvision_bm.utils.configs import label_map_rename
    from medvision_bm.utils.parse_utils import (
        get_labelsMap_imgModality_from_biometry_benchmark_plan,
        get_targetLabel_imgModality_from_biometry_benchmark_plan,
    )

    dataset_name = doc.get("dataset_name")
    task_id_raw = doc.get("taskID")
    slice_dim = doc.get("slice_dim")
    if dataset_name is None or task_id_raw is None or slice_dim is None:
        return None

    task_id = int(task_id_raw)
    label_idx, _ = get_targetLabel_imgModality_from_biometry_benchmark_plan(
        dataset_name, task_id
    )
    labels_map, img_modality = get_labelsMap_imgModality_from_biometry_benchmark_plan(
        dataset_name, task_id
    )
    label_name = labels_map.get(str(label_idx))
    if label_name is None:
        return None

    new_label = label_map_rename.get(label_name)
    if new_label is None:
        return None

    modality_map = {
        "MRI": "MR",
        "CT": "CT",
        "ultrasound": "US",
        "X-ray": "XR",
        "PET": "PET",
    }
    modality = modality_map.get(img_modality, img_modality)
    slicetype_map = {0: "S", 1: "C", 2: "A"}
    slicetype = slicetype_map.get(slice_dim)
    if slicetype is None:
        return None

    return f"{new_label} @ {modality} ({slicetype})"


def load_per_sample_values(parsed_dir, metric_name, common_labels, task_type="AD"):
    """Load per-sample metric values for a single model, used to draw violin plots.

    Reads all ``*_samples_*.jsonl`` files in *parsed_dir*.  Each JSONL line is
    one inference sample.  The function reconstructs the label from the sample's
    ``doc`` metadata and extracts the scalar metric value.

    Supports AD, TL, and Detection task samples.

    Args:
        parsed_dir: Path to the model's ``parsed/`` directory.
        metric_name: Name of the metric to extract (e.g. ``"avgMRE"``).
        common_labels: Iterable of label strings to keep; others are ignored.
        task_type: One of ``"AD"``, ``"TL"``, or ``"Detection"`` (default ``"AD"``).

    Returns:
        dict mapping label → list of per-sample float values.
    """
    label_values = {label: [] for label in common_labels}
    common_set = set(common_labels)

    for jsonl_path in glob.glob(os.path.join(parsed_dir, "*_samples_*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                doc = sample.get("doc", {})

                if task_type == "TL":
                    label = _reconstruct_tl_label(doc)
                    if label is None:
                        continue
                elif task_type == "Detection":
                    label = _reconstruct_detect_label(doc)
                    if label is None:
                        continue
                else:
                    # AD task: label = <dataset>_<metric_type>_<metric_key>
                    bp = doc.get("biometric_profile")
                    if bp is None or "metric_key" not in bp:
                        continue
                    label = (
                        f"{doc['dataset_name']}_{bp['metric_type']}_{bp['metric_key']}"
                    )

                if label not in common_set:
                    continue

                # Detection JSONL stores IoU under "avgIoU" while the summary JSON
                # and --metrics_list use the shorter "IoU" name.
                _DETECT_KEY_ALIASES = {"IoU": "avgIoU"}
                jsonl_key = (
                    _DETECT_KEY_ALIASES.get(metric_name, metric_name)
                    if task_type == "Detection"
                    else metric_name
                )
                raw = sample.get(jsonl_key)
                if raw is None:
                    continue
                if isinstance(raw, dict):
                    # AD/TL dicts carry an explicit "success" flag; Detection dicts
                    # don't (failures are encoded as 0.0 rather than a flag).
                    if "success" in raw and not raw["success"]:
                        continue
                    value = next(
                        (
                            v
                            for k, v in raw.items()
                            if k != "success" and isinstance(v, (int, float))
                        ),
                        None,
                    )
                elif isinstance(raw, (int, float)):
                    value = float(raw)
                else:
                    continue

                if value is not None:
                    label_values[label].append(float(value))

    return label_values


def plot_violin_on_spoke(
    ax, angle, values, should_invert, color, N, alpha=0.35, show_scatter=False
):
    """Overlay a violin + box plot + jittered scatter along a single radar spoke.

    Rendering is composed of three layers (back to front):

    1. **Violin** — KDE-based filled polygon centred on *angle*, capped at
       30 % of the inter-spoke angular gap so violins never overlap.
    2. **Box plot** — IQR box and 1.5×IQR whiskers drawn over the violin.
    3. **Jittered scatter** — outliers always shown; inliers only when
       *show_scatter* is True.  Angular jitter is bounded by the local KDE
       density so dots stay inside the violin contour.

    Args:
        ax: Polar ``Axes`` object.
        angle: Spoke angle in radians.
        values: Raw (pre-transform) per-sample metric values.
        should_invert: If True, apply ``1 - clamp(v,0,1)`` (for MRE/MAE).
        color: Base colour for violin fill, box, and scatter dots.
        N: Total number of spokes; used to set the max angular half-width.
        alpha: Opacity of the violin fill polygon.
    """
    from scipy.stats import gaussian_kde

    if len(values) < 3:
        return

    if should_invert:
        plot_vals = np.array([1.0 - min(max(v, 0.0), 1.0) for v in values])
    else:
        plot_vals = np.array([min(max(v, 0.0), 1.0) for v in values])

    try:
        kde = gaussian_kde(plot_vals)
    except Exception:
        return

    r_grid = np.linspace(0.0, 1.0, 200)
    density = kde(r_grid)

    max_half_width = (2 * pi / N) * 0.3
    max_d = density.max()
    if max_d == 0:
        return
    half_widths = (density / max_d) * max_half_width

    thetas = np.concatenate([angle - half_widths, (angle + half_widths)[::-1]])
    rs = np.concatenate([r_grid, r_grid[::-1]])
    ax.fill(thetas, rs, color=color, alpha=alpha, zorder=3, linewidth=0)

    q1, median, q3 = np.percentile(plot_vals, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    whisker_lo = plot_vals[plot_vals >= lo_fence].min()
    whisker_hi = plot_vals[plot_vals <= hi_fence].max()
    box_hw = max_half_width * 0.40

    box_thetas = [
        angle - box_hw,
        angle + box_hw,
        angle + box_hw,
        angle - box_hw,
        angle - box_hw,
    ]
    box_rs = [q1, q1, q3, q3, q1]
    ax.fill(box_thetas, box_rs, color="white", alpha=0.35, zorder=4)
    ax.plot(box_thetas, box_rs, color=color, linewidth=1.5, zorder=5)

    ax.plot([angle, angle], [whisker_lo, q1], color=color, linewidth=1.5, zorder=5)
    ax.plot([angle, angle], [q3, whisker_hi], color=color, linewidth=1.5, zorder=5)
    ax.plot(
        [angle - box_hw * 0.6, angle + box_hw * 0.6],
        [whisker_lo, whisker_lo],
        color=color,
        linewidth=1.5,
        zorder=5,
    )
    ax.plot(
        [angle - box_hw * 0.6, angle + box_hw * 0.6],
        [whisker_hi, whisker_hi],
        color=color,
        linewidth=1.5,
        zorder=5,
    )
    ax.plot(
        [angle - box_hw, angle + box_hw],
        [median, median],
        color=color,
        linewidth=2.5,
        zorder=6,
    )

    rng = np.random.default_rng(seed=42)
    for val in plot_vals:
        is_outlier = val < whisker_lo or val > whisker_hi
        if not is_outlier and not show_scatter:
            continue
        idx = int(np.clip(round(val * (len(r_grid) - 1)), 0, len(r_grid) - 1))
        local_hw = half_widths[idx]
        jitter = rng.uniform(-local_hw, local_hw)
        ax.plot(
            angle + jitter,
            val,
            "o",
            color=color,
            markersize=6 if is_outlier else 4.5,
            alpha=0.35,
            zorder=8 if is_outlier else 7,
            markeredgewidth=0,
        )


def plot_radar_chart(
    data_dict,
    metric_name,
    label_numbers,
    label_names,
    ax,
    max_value,
    model_name_display_map,
    verbose_samples_by_model=None,
    verbose_model_colors=None,
    show_scatter=False,
):
    """Plot radar chart for a single metric across all models.

    Args:
        data_dict: Dictionary mapping model names to metric values.
        metric_name: Name of the metric being plotted.
        label_numbers: List of label numbers for axes.
        label_names: List of label names for coloring.
        ax: Matplotlib polar axes object.
        max_value: Maximum value for scaling y-axis.
        model_name_display_map: Dictionary mapping model names to display names.
        verbose_samples_by_model: Optional dict mapping model name →
            {label → list of per-sample values} for violin overlays.
        verbose_model_colors: Optional dict mapping model name → colour.
    """
    N = len(label_numbers)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # One colour per model, in config order (see model_palette: tab10, reused hues darkened).
    model_colors = model_palette(len(data_dict))

    # MRE and MAE are "lower is better": invert so the outer ring = best performance.
    # Inversion: plot_value = 1 - clamp(metric_value, 0, 1)
    should_invert = "MRE" in metric_name.upper() or "MAE" in metric_name.upper()

    for i, (model_name, values) in enumerate(data_dict.items()):
        if should_invert:
            capped_values = [min(max(v, 0), 1) for v in values]
            plot_values = [1 - v for v in capped_values]
        else:
            plot_values = values
        values_plot = plot_values + plot_values[:1]

        display_name = model_name_display_map.get(model_name, model_name)
        ax.plot(
            angles,
            values_plot,
            linestyle="-",
            linewidth=3,
            alpha=0.9,
            label=display_name,
            color=model_colors[i],
        )

    if verbose_samples_by_model is not None:
        for verbose_model_name, verbose_samples in verbose_samples_by_model.items():
            for spoke_idx, label_name in enumerate(label_names):
                samples = verbose_samples.get(label_name, [])
                if not samples:
                    continue
                plot_violin_on_spoke(
                    ax,
                    angle=angles[spoke_idx],
                    values=samples,
                    should_invert=should_invert,
                    color=(verbose_model_colors or {}).get(verbose_model_name, "gray"),
                    N=N,
                    show_scatter=show_scatter,
                )

    ax.set_xticks(angles[:-1])
    colored_labels = [
        (num, C_TUMOR_LESION_LABEL, "bold")
        if is_purple_label(name)
        else (num, C_ANATOMY_LABEL, "normal")
        for num, name in zip(label_numbers, label_names)
    ]
    ax.set_xticklabels([str(num) for num in label_numbers], fontsize=16)
    for tick_label, (num, color, fontweight) in zip(
        ax.get_xticklabels(), colored_labels
    ):
        tick_label.set_color(color)
        tick_label.set_fontweight(fontweight)

    # The negative lower bound pushes the center off-screen so the innermost
    # visible ring sits at y=0, giving the radar a cleaner look.
    y_limit_max = 1.1
    ax.set_ylim(-0.3, y_limit_max)
    y_ticks = np.arange(0, y_limit_max, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([])

    for y_val in y_ticks:
        if y_val == 0:
            if should_invert:
                label_text = "≥1"
            else:
                continue
        else:
            # For inverted metrics reverse the transform to show original values.
            label_text = f"{(1 - y_val):.1f}" if should_invert else f"{y_val:.1f}"
        ax.text(
            0,
            y_val,
            label_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=16,
            fontweight="bold",
            zorder=10,
        )

    display_metric_name = (
        metric_name[3:] if metric_name.lower().startswith("avg") else metric_name
    )
    arrow = " ↓" if should_invert else " ↑"
    ax.grid(True)
    return display_metric_name + arrow


# Map verbose dataset names to short display abbreviations.
_DATASET_ABBR = {
    "Ceph-Biometrics-400": "Ceph",
    "FeTA24": "FeTA24",
}

# Regex patterns for AD label names.
# Distance:  {dataset}_distance_L-{p1}-{p2}
_RE_DISTANCE = re.compile(r"^(.+)_distance_L-(\d+)-(\d+)$")
# Angle:     {dataset}_angle_A-L_{p1}_{p2}-L_{p3}_{p4}
_RE_ANGLE = re.compile(r"^(.+)_angle_A-L_(\d+)_(\d+)-L_(\d+)_(\d+)$")


def split_ad_labels(labels):
    """Partition AD labels into ``[("Angle", [...]), ("Distance", [...])]``, dropping empty groups.

    Angle and distance are measured in different units (degree vs mm) and come from different
    dataset sets, so they get their own radar row rather than sharing one set of spokes.
    Labels matching neither pattern are appended to a trailing "Other" group so nothing is
    silently dropped.
    """
    angle = [x for x in labels if _RE_ANGLE.match(x)]
    distance = [x for x in labels if _RE_DISTANCE.match(x)]
    other = [x for x in labels if not _RE_ANGLE.match(x) and not _RE_DISTANCE.match(x)]
    return [(n, ls) for n, ls in (("Angle", angle), ("Distance", distance), ("Other", other)) if ls]


def _pooled_success_rate(parsed_dir, labels):
    """Sample-weighted success rate over ``labels`` from a model's AD summary JSON.

    ``SR = Σ(SuccessRate_t · num_samples_t) / Σ num_samples_t`` — i.e. total successful parses over
    total samples across the given targets. Returns ``None`` if the summary is missing or covers
    none of the targets (so the caller can label it n/a rather than 0%).
    """
    from medvision_bm.utils.configs import SUMMARY_FILENAME_AD_METRICS

    path = os.path.join(parsed_dir, SUMMARY_FILENAME_AD_METRICS)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        summary = json.load(f)
    label_set = set(labels)
    succ = total = 0
    for target, m in summary.items():
        if target not in label_set:
            continue
        n = m.get("num_samples", 0)
        succ += m.get("SuccessRate", 0.0) * n
        total += n
    return (succ / total) if total else None


# Combining Low Line: appended after each character it underlines that character, so the rule is
# part of the glyph run and therefore rotates and line-wraps with the text for free.
#
# Deliberately NOT mathtext ``\underline``: that is only available in matplotlib >= 3.11 and raises
# "Unknown symbol: \underline" on 3.10.x, which is what ships in this project's environment. Plain
# text also renders hyphens correctly, whereas mathtext turns "-" into a wider minus glyph.
_COMBINING_LOW_LINE = "̲"


def _underline(text):
    """``text`` with a combining low line after every character (including spaces, so the rule
    reads as one continuous underline rather than per-word dashes)."""
    return "".join(ch + _COMBINING_LOW_LINE for ch in text)


def _boxplot_model_label(name, sr):
    """Stacked x-tick label: model name / parameter count / ``SR=NN%``.

    The trailing parenthetical of a display name is the parameter count ("MedVision-V0 (7B)"), so
    it goes on its own line to keep the rotated labels short. Both identity lines are underlined
    when SR < 50%; the SR line itself stays plain.
    """
    m = re.match(r"^(.*?)\s*(\([^()]*\))$", name)
    name_part, param_part = (m.group(1), m.group(2)) if m else (name, None)
    mark = _underline if (sr is not None and sr < 0.5) else (lambda s: s)

    lines = [mark(name_part)]
    if param_part:
        lines.append(mark(param_part))
    lines.append("SR=n/a" if sr is None else f"SR={sr * 100:.0f}%")
    return "\n".join(lines)


def plot_angle_mae_boxplot(
    task_dir, models, model_display_name, angle_labels, out_stem, formats=("pdf",)
):
    """Save a box plot of per-sample angle MAE (degrees), one box per model.

    Complements the AD radar, which shows avgMRE per target: this shows the *distribution* of
    absolute error pooled across all angle targets, so heavy tails and outliers are visible.
    Reads ``avgMAE`` (a plain float per sample) via :func:`load_per_sample_values`; samples whose
    parse failed carry ``success: false`` and are skipped there.

    Boxes are ordered by ascending MAE (best model leftmost, MAE = the mean of the box's own
    samples).

    COLOUR IS BOUND TO THE MODEL BEFORE ORDERING: each model takes ``palette[i]`` at its index *i*
    in ``models`` — the same config order, palette and ``model_palette`` call the radar uses — and
    that colour travels with the model through the MAE sort. So a model has the SAME colour in the
    radar and here, even though the two figures order models differently. Do not re-derive the
    colour from a box's position; that would repaint models by rank and break the correspondence.

    Under each box: the model's success rate (SR, pooled over angle targets from the summary JSON);
    model names with SR < 50% are underlined, since a low-SR box summarises only the minority of
    samples that parsed.
    """
    # palette indexed by config order == the radar's mapping (see plot_radar_chart / the verbose
    # colour block in plot_metrics_multi_model, which both use model_palette over the same list)
    palette = model_palette(len(models))
    recs = []
    for i, model in enumerate(models):
        parsed_dir = os.path.join(task_dir, model, "parsed")
        if not os.path.isdir(parsed_dir):
            continue
        by_label = load_per_sample_values(
            parsed_dir, "avgMAE", angle_labels, task_type="AD"
        )
        values = [v for vs in by_label.values() for v in vs]
        if not values:
            print(f"[Warning] No angle avgMAE samples for {model}; omitted from the box plot.")
            continue
        recs.append(
            {
                "values": values,
                "mae": sum(values) / len(values),        # the model's MAE = mean of its samples
                "sr": _pooled_success_rate(parsed_dir, angle_labels),
                "name": model_display_name.get(model, model),
                "color": palette[i],                     # bound BEFORE the sort, by config index
            }
        )

    if not recs:
        print("[Warning] No angle samples found for any model; box plot skipped.")
        return None

    recs.sort(key=lambda r: r["mae"])                     # ascending MAE; colours travel with recs

    fig, ax = plt.subplots(figsize=(max(10, _BOX_INCHES_PER_MODEL * len(recs) + 4), 7))
    bp = ax.boxplot(
        [r["values"] for r in recs],
        patch_artist=True,
        showfliers=False,   # angle AE has a long tail; fliers would compress every box
        showmeans=True,     # boxes are ORDERED by mean (= MAE); mark it so the sort key is visible
        meanprops=dict(marker="D", markerfacecolor="#111111",
                       markeredgecolor="none", markersize=4),
        widths=0.62,
        medianprops=dict(color="#222222", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
    )
    # Alpha on the FACE only (not patch.set_alpha, which would fade the outline too), so the box
    # reads lighter while its edge, median and whiskers stay crisp.
    for patch, r in zip(bp["boxes"], recs):
        patch.set_facecolor(to_rgba(r["color"], _BOX_FACE_ALPHA))
        patch.set_edgecolor("#555555")

    ax.set_xticks(range(1, len(recs) + 1))
    # Angled rather than vertical. A 3-line label at 45° offsets each extra line perpendicular to
    # the text, so it drifts diagonally into its neighbour's slot -- the reason _BOX_INCHES_PER_MODEL
    # buys enough horizontal room per box for the block to clear the next one.
    ax.set_xticklabels(
        [_boxplot_model_label(r["name"], r["sr"]) for r in recs],
        rotation=_BOX_LABEL_ROTATION,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    ax.set_ylabel("MAE (degree)", fontsize=14)
    ax.grid(axis="y", color="#CCCCCC", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for fmt in formats:
        save_fig_capped(f"{out_stem}.{fmt}", bbox_inches="tight", transparent=True)
    print(f"Angle MAE box plot saved to: {out_stem}.{formats[0]}")
    plt.close(fig)
    return f"{out_stem}.{formats[0]}"


def abbreviate_label_name(name):
    """Abbreviate label names for display.

    AD labels are reformatted as:
      - distance: ``<dataset>: d(P1,P2)``
      - angle:    ``<dataset>: a({P1,P2},{P3,P4})``

    Other labels (Detection / TL) receive lighter cleanup only.

    Args:
        name: Original label name

    Returns:
        str: Abbreviated label name
    """
    m = _RE_DISTANCE.match(name)
    if m:
        dataset, p1, p2 = m.group(1), m.group(2), m.group(3)
        return f"{_DATASET_ABBR.get(dataset, dataset)}: d(P{p1},P{p2})"

    m = _RE_ANGLE.match(name)
    if m:
        dataset, p1, p2, p3, p4 = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4),
            m.group(5),
        )
        return (
            f"{_DATASET_ABBR.get(dataset, dataset)}: a({{P{p1},P{p2}}},{{P{p3},P{p4}}})"
        )

    name = name.replace("Tumor/Lesion", "T/L").replace("Lesion/Tumor", "T/L")
    return name.strip()


def _compute_label_panel_layout(label_mapping, num_columns=2, panel_width_inches=4):
    """Compute column x-positions and axes width fraction for the label panel.

    Returns:
        col_x_positions : list of float — x anchor (axes coords) for each column.
        width_ratio     : float in (0, 1] — fraction of axes width occupied by content.
    """
    labels = list(label_mapping.items())
    labels_per_column = (len(labels) + num_columns - 1) // num_columns
    columns = [
        labels[i * labels_per_column : (i + 1) * labels_per_column]
        for i in range(num_columns)
    ]

    subplot_width_pts = panel_width_inches * 72
    char_width = 10 / subplot_width_pts
    x_margin = 0.03
    col_gap = 0.08

    def _max_chars(col):
        return max(
            (len(f"{num}: {abbreviate_label_name(name)}") for num, name in col),
            default=0,
        )

    if num_columns == 1 or not columns[0]:
        col0_chars = _max_chars(columns[0]) if columns else 0
        right_edge = x_margin + col0_chars * char_width + x_margin
        col_x_positions = [x_margin]
    else:
        col_x_positions = []
        current_x = x_margin
        for col in columns:
            col_x_positions.append(current_x)
            current_x += _max_chars(col) * char_width + col_gap
        last_col_chars = _max_chars(columns[-1]) if columns else 0
        right_edge = col_x_positions[-1] + last_col_chars * char_width + x_margin

    width_ratio = min(max(right_edge, 0.25), 1.0)
    return col_x_positions, width_ratio


def plot_label_mapping(
    label_mapping,
    ax,
    task_type="AD",
    panel_width_inches=4,
    panel_height_inches=8.0,
    legend_handles=None,
    legend_labels=None,
    label_col=None,
    legend_col=None,
):
    """Plot label number-to-name mapping and optional model legend.

    Args:
        label_mapping: Dictionary mapping label numbers to label names.
        ax: Matplotlib axes object.
        task_type: Task type string; "TL" uses default font instead of monospace.
        panel_width_inches: Physical width in inches for character-width estimation.
        panel_height_inches: Physical height in inches for line-height estimation.
        legend_handles: Line handles from axes[0].get_legend_handles_labels(), or None.
        legend_labels: Display names matching legend_handles, or None.
        label_col: Number of columns for the label list. None = auto.
        legend_col: Number of columns for the model legend. None = auto. When the legend ends up
            with more than one column its font is scaled so the legend's rendered width matches
            the label block's, keeping the panel's two stacked elements the same width.
    """
    ax.axis("off")

    labels = list(label_mapping.items())
    y_pos = 0.95
    line_height = max(1.5 * 16.0 / (panel_height_inches * 72), 0.030)
    num_columns = label_col if label_col is not None else 1

    col_x_positions, _ = _compute_label_panel_layout(
        label_mapping, num_columns, panel_width_inches=panel_width_inches
    )

    labels_per_column = (len(labels) + num_columns - 1) // num_columns
    columns = [
        labels[i * labels_per_column : (i + 1) * labels_per_column]
        for i in range(num_columns)
    ]

    # TL label names use the default font; all others use monospace.
    label_fontfamily = "monospace" if task_type != "TL" else None
    label_texts = []

    for col_idx, col_labels in enumerate(columns):
        x_pos = (
            col_x_positions[col_idx]
            if col_idx < len(col_x_positions)
            else col_idx * (1.0 / num_columns)
        )
        for i, (num, name) in enumerate(col_labels):
            abbreviated_name = abbreviate_label_name(name)
            if task_type == "TL" or is_purple_label(name):
                color, fontweight = C_TUMOR_LESION_LABEL, "bold"
            else:
                color, fontweight = C_ANATOMY_LABEL, "normal"
            text_kwargs = dict(
                transform=ax.transAxes,
                fontsize=16,
                verticalalignment="top",
                color=color,
                fontweight=fontweight,
            )
            if label_fontfamily is not None:
                text_kwargs["fontfamily"] = label_fontfamily
            label_texts.append(
                ax.text(
                    x_pos,
                    y_pos - i * line_height,
                    f"{num}: {abbreviated_name}",
                    **text_kwargs,
                )
            )

    max_lines = max(len(col) for col in columns)
    next_y = y_pos - max_lines * line_height

    legend_refit = None
    if legend_handles and legend_labels:
        n_legend_cols = (
            legend_col
            if legend_col is not None
            else (2 if len(legend_handles) > 4 else 1)
        )

        def _draw_legend(fontsize):
            return ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                bbox_to_anchor=(col_x_positions[0], next_y - line_height * 0.5),
                fontsize=fontsize,
                ncol=n_legend_cols,
                frameon=True,
            )

        # One base size for every task (no per-task special case).
        state = {"legend": _draw_legend(_LEGEND_FONTSIZE)}

        if n_legend_cols > 1 and label_texts:

            def legend_refit():
                """Scale the legend's font until its width matches the label block's.

                Every part of a legend's width — text, handle, padding, column spacing — is a
                multiple of its font size, so width is essentially LINEAR in font size and one
                correction lands within a few percent. This measures the CURRENT size rather than
                the base, so calling it again simply corrects whatever error is left: the caller
                re-runs it after ``tight_layout``, which is what makes the fit exact for
                multi-column label blocks (their inter-column gap is in axes fractions, so it
                rescales when the layout changes the axes' pixel width).
                """
                fig = ax.figure
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                extents = [t.get_window_extent(renderer) for t in label_texts]
                label_px = max(e.x1 for e in extents) - min(e.x0 for e in extents)
                legend_px = state["legend"].get_window_extent(renderer).width
                if label_px <= 0 or legend_px <= 0:
                    return
                current = state["legend"].get_texts()[0].get_fontsize()
                lo, hi = _LEGEND_FONTSIZE_RANGE
                fitted = min(max(current * label_px / legend_px, lo), hi)
                if abs(fitted - current) > 0.1:
                    state["legend"].remove()
                    state["legend"] = _draw_legend(fitted)

            legend_refit()  # first pass, so the pre-layout figure is already close

    has_modality = any("@" in name for name in label_mapping.values())
    title = "Label @ Modality" if has_modality else "Label"
    return title, col_x_positions[0], legend_refit


def plot_metrics_multi_model(
    task_type,
    fig_dir,
    task_dir,
    fig_name,
    config,
    minimum_group_size,
    metrics_list,
    verbose_model=None,
    show_scatter=False,
    show_label_name=False,
    radar_cell_inches=8,
    label_col=None,
    legend_col=None,
    formats=("pdf",),
):
    """Plot radar charts for metrics across all models.

    Args:
        task_type: Type of the task to process: ['AD', 'TL', 'Detection'].
        fig_dir: Directory to save output figure.
        task_dir: Directory containing model folders.
        fig_name: Output figure filename.
        config: Configuration dictionary from YAML file.
        minimum_group_size: Minimum number of samples per label to include.
        metrics_list: List of metric names to extract and plot.
        verbose_model: Optional model name or list of model names whose
            per-sample distributions are overlaid as violin plots on each spoke.
        show_label_name: When True, add a label number-to-name mapping panel.
    """
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    model_display_name = config.get("model_display_name", {})
    models = list(model_display_name.keys())
    if not models:
        raise ValueError(
            "Config must define model_display_name with at least one model key."
        )

    if task_type == "Detection":
        from medvision_bm.utils.configs import SUMMARY_FILENAME_DETECT_METRICS

        json_filename = SUMMARY_FILENAME_DETECT_METRICS
    elif task_type == "AD":
        from medvision_bm.utils.configs import SUMMARY_FILENAME_AD_METRICS

        json_filename = SUMMARY_FILENAME_AD_METRICS
    elif task_type == "TL":
        from medvision_bm.utils.configs import SUMMARY_FILENAME_TL_METRICS

        json_filename = SUMMARY_FILENAME_TL_METRICS.replace(".json", "_filtered.json")
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    model_data = {}
    for model in models:
        parsed_json_dir = os.path.join(task_dir, model, "parsed")
        if not os.path.isdir(parsed_json_dir):
            raise ValueError(f"Parsed directory not found: {parsed_json_dir}")
        json_file = os.path.join(parsed_json_dir, json_filename)
        if not os.path.exists(json_file):
            raise ValueError(f"JSON file not found: {json_file}")
        df = load_model_metrics(json_file, metrics_list, task_type=task_type)
        if task_type == "Detection":
            filtered_df = df[df["num_samples"] >= minimum_group_size]
            if len(filtered_df) == 0:
                print(
                    f"[Warning] Model {model} skipped: no labels with ≥ {minimum_group_size} samples."
                )
                continue
            model_data[model] = filtered_df
        else:
            model_data[model] = df

    if len(model_data) == 0:
        print(
            f"No models found with valid data (minimum sample size: {minimum_group_size})"
        )
        return

    all_labels = [set(df["Target"]) for df in model_data.values()]
    common_labels = sorted(list(set.intersection(*all_labels))) if all_labels else []

    if len(common_labels) == 0:
        print("No common labels found across all models")
        return

    # Row groups. AD splits into an Angle row and a Distance row (different units, different
    # datasets); every other task keeps its single unnamed group, i.e. the original layout.
    label_groups = (
        split_ad_labels(common_labels) if task_type == "AD" else [(None, common_labels)]
    )
    # Spoke numbering restarts at 1 within each row, so each row gets its own mapping panel.
    group_mappings = [
        {i + 1: label for i, label in enumerate(labels)} for _, labels in label_groups
    ]

    label_mapping = group_mappings[0]
    label_numbers = list(label_mapping.keys())

    if verbose_model is None:
        verbose_models = []
    elif isinstance(verbose_model, str):
        verbose_models = [verbose_model]
    else:
        verbose_models = list(verbose_model)

    for verbose_model_name in verbose_models:
        if verbose_model_name not in models:
            raise ValueError(
                f"--verbose_model '{verbose_model_name}' is not in the model_display_name keys in the config."
            )

    verbose_samples_by_metric = {}
    if verbose_models:
        for metric in metrics_list:
            verbose_samples_by_metric[metric] = {}
            print(
                f"\n[Info] Loading per-sample data for verbose models on metric: {metric}"
            )
            for verbose_model_name in verbose_models:
                verbose_parsed_dir = os.path.join(
                    task_dir, verbose_model_name, "parsed"
                )
                verbose_samples_by_metric[metric][verbose_model_name] = (
                    load_per_sample_values(
                        verbose_parsed_dir, metric, common_labels, task_type=task_type
                    )
                )
                n_loaded = sum(
                    len(v)
                    for v in verbose_samples_by_metric[metric][
                        verbose_model_name
                    ].values()
                )
                print(
                    f"  {verbose_model_name} / {metric}: {n_loaded} samples across {len(common_labels)} labels"
                )

    num_metrics = len(metrics_list)
    RADAR_INCHES = radar_cell_inches
    LABEL_ROW_HEIGHT = RADAR_INCHES * 0.75
    num_metric_cols = int(np.ceil(np.sqrt(num_metrics)))
    num_metric_rows = int(np.ceil(num_metrics / num_metric_cols))

    num_groups = len(label_groups)

    if show_label_name and num_metrics == 3 and num_groups == 1:
        # 2×2 grid; label occupies the vacant 4th slot [1, 1].
        gs_rows, gs_cols = 2, 2
        width_ratios = height_ratios = None
        fig_width = RADAR_INCHES * 2
        fig_height = RADAR_INCHES * 2
        label_panel_width = RADAR_INCHES
        label_panel_height = RADAR_INCHES
    elif show_label_name and num_metrics == 4 and num_groups == 1:
        # 2×2 radar subplots + full-width label row at the bottom.
        gs_rows, gs_cols = 3, 2
        width_ratios = None
        height_ratios = [RADAR_INCHES, RADAR_INCHES, LABEL_ROW_HEIGHT]
        fig_width = RADAR_INCHES * 2
        fig_height = RADAR_INCHES * 2 + LABEL_ROW_HEIGHT
        label_panel_width = RADAR_INCHES * 2
        label_panel_height = LABEL_ROW_HEIGHT
    else:
        # Square-ish radar grid + optional right-column label panel.
        if show_label_name:
            _num_label_cols = label_col if label_col is not None else 1
            _max = RADAR_INCHES * num_metric_cols
            # size the shared label column for the WIDEST group, so no row's panel is clipped
            _wr = max(
                _compute_label_panel_layout(m, _num_label_cols, _max)[1]
                for m in group_mappings
            )
            LABEL_INCHES = _max * _wr
        else:
            LABEL_INCHES = 0
        gs_rows = num_metric_rows * num_groups
        gs_cols = num_metric_cols + (1 if show_label_name else 0)
        width_ratios = [RADAR_INCHES] * num_metric_cols + (
            [LABEL_INCHES] if show_label_name else []
        )
        height_ratios = None
        fig_width = RADAR_INCHES * num_metric_cols + LABEL_INCHES
        fig_height = RADAR_INCHES * num_metric_rows * num_groups
        label_panel_width = LABEL_INCHES
        label_panel_height = RADAR_INCHES * num_metric_rows

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        gs_rows,
        gs_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )

    # One row-block of radars per label group (AD: Angle then Distance; otherwise a single block).
    group_axes, mapping_axes = [], []
    for g in range(num_groups):
        row_base = g * num_metric_rows
        group_axes.append(
            [
                fig.add_subplot(
                    gs[row_base + idx // num_metric_cols, idx % num_metric_cols],
                    projection="polar",
                )
                for idx in range(num_metrics)
            ]
        )
        if not show_label_name:
            mapping_axes.append(None)
        elif num_groups == 1 and num_metrics == 3:
            mapping_axes.append(fig.add_subplot(gs[1, 1]))
        elif num_groups == 1 and num_metrics == 4:
            mapping_axes.append(fig.add_subplot(gs[2, :]))
        elif num_groups == 1:
            mapping_axes.append(fig.add_subplot(gs[:, num_metric_cols]))
        else:
            mapping_axes.append(
                fig.add_subplot(
                    gs[row_base : row_base + num_metric_rows, num_metric_cols]
                )
            )

    axes = group_axes[0]
    mapping_ax = mapping_axes[0]

    # Pre-compute verbose model colours to match the plotted line colours.
    verbose_model_colors = {}
    if verbose_models:
        # Must match plot_radar_chart's assignment exactly (same palette, same
        # model order) so the violin overlays share their line's colour.
        model_colors = model_palette(len(model_data))
        for idx, model_name in enumerate(model_data.keys()):
            if model_name in verbose_models:
                verbose_model_colors[model_name] = model_colors[idx]
        for verbose_model_name in verbose_models:
            verbose_model_colors.setdefault(verbose_model_name, "gray")

    # axis_titles: ax → (x_axes, title_text, ha)
    axis_titles = {}

    for g, (group_name, group_labels) in enumerate(label_groups):
        g_numbers = list(group_mappings[g].keys())
        for idx, metric in enumerate(metrics_list):
            metric_data = {}
            max_value = 0
            for model_name, df in model_data.items():
                df_filtered = (
                    df[df["Target"].isin(group_labels)]
                    .set_index("Target")
                    .reindex(group_labels)
                )
                values = df_filtered[metric].tolist()
                metric_data[model_name] = values
                max_value = max(max_value, max(values))

            _title = plot_radar_chart(
                metric_data,
                metric,
                g_numbers,
                group_labels,
                group_axes[g][idx],
                max_value,
                model_display_name,
                verbose_samples_by_model=verbose_samples_by_metric.get(metric),
                verbose_model_colors=verbose_model_colors,
                show_scatter=show_scatter,
            )
            if group_name is not None:
                _title = f"{group_name} — {_title}"
            axis_titles[group_axes[g][idx]] = (0.5, _title, "center")

    handles, leg_labels = group_axes[0][0].get_legend_handles_labels()
    legend_refits = []
    if show_label_name:
        for g in range(num_groups):
            # The model legend is shared across rows, so draw it once, in the LAST row's panel
            # (passing None suppresses it in plot_label_mapping).
            is_last = g == num_groups - 1
            _label_title, _label_x, _legend_refit = plot_label_mapping(
                group_mappings[g],
                mapping_axes[g],
                task_type,
                panel_width_inches=label_panel_width,
                panel_height_inches=label_panel_height,
                legend_handles=handles if is_last else None,
                legend_labels=leg_labels if is_last else None,
                label_col=label_col,
                legend_col=legend_col,
            )
            axis_titles[mapping_axes[g]] = (_label_x, _label_title, "left")
            if _legend_refit is not None:
                legend_refits.append(_legend_refit)
    else:
        _legend_ncol = (
            legend_col if legend_col is not None else (2 if len(handles) > 4 else 1)
        )
        fig.legend(
            handles,
            leg_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            fontsize=16,
            ncol=_legend_ncol,
        )

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=4.0)

    # tight_layout changes the panel's pixel width, which moves the label block's inter-column gaps
    # (they are in axes fractions). Re-run the fit on the final geometry so the legend really does
    # match the label block; each pass corrects the remaining error, so this converges.
    for _refit in legend_refits:
        _refit()

    all_axes = [ax for g_axes in group_axes for ax in g_axes] + (
        [a for a in mapping_axes if a is not None] if show_label_name else []
    )

    # Measure true content top per row for pixel-accurate title placement.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_top_px = {}
    for ax in all_axes:
        row_key = ax.get_subplotspec().rowspan.start
        bbox = ax.get_tightbbox(renderer)
        if bbox is not None:
            row_top_px[row_key] = max(row_top_px.get(row_key, 0.0), bbox.y1)

    # Place titles with a blended transform: x in axes coords, y in figure coords.
    title_gap_px = 10
    fig_h_px = fig.bbox.height
    for ax in all_axes:
        if ax not in axis_titles:
            continue
        x_axes, title_text, ha = axis_titles[ax]
        row_key = ax.get_subplotspec().rowspan.start
        target_y_fig = (row_top_px.get(row_key, 0.0) + title_gap_px) / fig_h_px
        trans = blended_transform_factory(ax.transAxes, fig.transFigure)
        ax.text(
            x_axes,
            target_y_fig,
            title_text,
            transform=trans,
            fontsize=24,
            fontweight="bold",
            horizontalalignment=ha,
            verticalalignment="bottom",
            clip_on=False,
        )

    stem = os.path.splitext(os.path.join(fig_dir, fig_name))[0]
    for fmt in formats:
        save_fig_capped(f"{stem}.{fmt}", bbox_inches="tight", transparent=True)
    output_file = f"{stem}.{formats[0]}"
    print(f"Figure saved to: {output_file}")
    plt.close()

    # AD only: an extra box plot of the per-sample angle absolute error, which the radar (a single
    # avgMRE per target) cannot show. Saved alongside the radar figure.
    angle_labels = next(
        (labels for name, labels in label_groups if name == "Angle"), []
    )
    if angle_labels:
        plot_angle_mae_boxplot(
            task_dir,
            list(model_data.keys()),
            model_display_name,
            angle_labels,
            f"{stem}_angle-MAE-box",
            formats=formats,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot radar charts for metrics across multiple models"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Task type: AD, TL, or Detection.",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to YAML config file (model_display_name dict).",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Directory to save the output figure.",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        required=True,
        help="Directory containing model folders (each with a parsed/ subdirectory).",
    )
    parser.add_argument(
        "--fig_name",
        type=str,
        required=True,
        help="Output figure filename (e.g., radar_detection.pdf).",
    )
    parser.add_argument(
        "--metrics_list",
        type=str,
        nargs="+",
        default=["Precision", "F1"],
        help="Metric names to plot (e.g., Precision F1 Recall).",
    )
    parser.add_argument(
        "--verbose_model",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Model name(s) whose per-sample distributions are overlaid as violin "
            "plots on each radar spoke. Must match keys in model_display_name. "
            "Requires *_samples_*.jsonl files in each model's parsed/ directory."
        ),
    )
    parser.add_argument(
        "--show_scatter",
        action="store_true",
        default=False,
        help="Overlay jittered scatter on violin (only with --verbose_model).",
    )
    parser.add_argument(
        "--show_label_name",
        action="store_true",
        default=False,
        help="Add a label number-to-name mapping panel alongside the radar plots.",
    )
    parser.add_argument(
        "--radar_cell_inches",
        type=float,
        default=8,
        help="Width in inches for each radar subplot cell (default: 8).",
    )
    parser.add_argument(
        "--label_col",
        type=int,
        default=None,
        help="Columns in the label panel (default: auto).",
    )
    parser.add_argument(
        "--legend_col",
        type=int,
        default=None,
        help="Columns in the model legend (default: auto).",
    )
    parser.add_argument(
        "--save_as_png", action="store_true", help="Save figures as PNG."
    )
    parser.add_argument(
        "--save_as_pdf", action="store_true", help="Save figures as PDF."
    )

    args = parser.parse_args()
    assert args.task_type in [
        "AD",
        "TL",
        "Detection",
    ], f"Invalid task_type '{args.task_type}'. Must be one of: AD, TL, Detection."

    formats = [
        f for f, on in (("png", args.save_as_png), ("pdf", args.save_as_pdf)) if on
    ] or ["pdf"]
    config = load_config(args.config_yaml)
    plot_metrics_multi_model(
        args.task_type,
        args.fig_dir,
        args.task_dir,
        args.fig_name,
        config,
        MINIMUM_GROUP_SIZE,
        args.metrics_list,
        verbose_model=args.verbose_model,
        show_scatter=args.show_scatter,
        show_label_name=args.show_label_name,
        radar_cell_inches=args.radar_cell_inches,
        label_col=args.label_col,
        legend_col=args.legend_col,
        formats=formats,
    )


if __name__ == "__main__":
    main()
