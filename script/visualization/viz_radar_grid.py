"""
Plot per-model radar-chart grids across all three benchmark tasks.

One row per model, six polar subplots per row:

    Detection Recall | Detection Precision | Detection F1 | Angle MRE | Distance MRE | T/L MRE

i.e. the same six radars that viz_radar.py spreads over three multi-model
figures (3 detection metrics, the AD Angle/Distance groups, and TL), but drawn
as a single-model trace per cell, with that model's per-sample distribution
overlaid as violin + box plots on every spoke (viz_radar.py's --verbose_model
rendering). The models are split over two figures (_part1/_part2), each with a
label number-to-name mapping block for all four spoke sets below the grid.

Spokes, filters, metric inversion (MRE: outer ring = best), violins, and
per-model colours are all inherited from viz_radar.py, so these figures read as
one family with the existing per-task radars. Models are joined ACROSS tasks by
display name (the per-task result directory names differ), and each model's
colour is bound to its index in the FULL config order before any exclusion, so
colours still match the multi-model figures after MedVision is dropped.

Usage (see viz_radar_grid.sh for the driver):
    python viz_radar_grid.py \
        --config_detect <yaml> --task_dir_detect <dir> \
        --config_ad <yaml> --task_dir_ad <dir> \
        --config_tl <yaml> --task_dir_tl <dir> \
        --fig_dir <dir> --fig_name <name.pdf> \
        [--parsed_dirname <name>] [--cell_inches N] \
        [--exclude_display MedVision] \
        [--save_as_png] [--save_as_pdf]
"""

import os
from math import ceil, pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import viz_radar as vr
from medvision_bm.utils.configs import (
    C_ANATOMY_LABEL,
    C_TUMOR_LESION_LABEL,
    MINIMUM_GROUP_SIZE,
    SUMMARY_FILENAME_AD_METRICS,
    SUMMARY_FILENAME_DETECT_METRICS,
    SUMMARY_FILENAME_TL_METRICS,
)
from medvision_bm.utils.plot_utils import save_fig_capped

DETECT_METRICS = ["Recall", "Precision", "F1"]

# How many figures the model rows are split over.
N_PARTS = 2

# Cell / text geometry. The radar cells are small multiples (~3 in), so fonts are
# scaled well below viz_radar.py's 8-inch-cell sizes.
WSPACE = 0.30
HSPACE = 0.18
HEADER_FONTSIZE = 15
NAME_FONTSIZE = 13
RING_FONTSIZE = 6.5

# Label-mapping block below the grid: column count per spoke set, chosen so the
# tallest block stays around 9 lines.
MAP_COLS = {"Detection": 3, "Angle": 1, "Distance": 2, "T/L": 1}
MAP_FONTSIZE = 10
MAP_TITLE_IN = 0.35
MAP_LINE_IN = 0.21
MAP_COL_GAP_IN = 0.35
MAP_BLOCK_GAP_IN = 0.8


def load_task_data(config_yaml, task_dir, json_filename, metrics, task_type,
                   parsed_dirname, exclude_display):
    """Load one task's per-model metric DataFrames, keyed by DISPLAY name.

    Mirrors viz_radar.plot_metrics_multi_model's loading: same summary files,
    same Detection minimum-group-size filter, hard error on a missing summary.

    Returns (display_names_in_config_order, {display_name: df},
    {display_name: model_dir_name}).
    """
    config = vr.load_config(config_yaml)
    model_display_name = config.get("model_display_name", {})
    if not model_display_name:
        raise ValueError(f"No model_display_name entries in {config_yaml}")

    all_display = list(model_display_name.values())
    data, dirs = {}, {}
    for model_dir, display in model_display_name.items():
        if exclude_display and exclude_display in display:
            continue
        json_file = os.path.join(task_dir, model_dir, parsed_dirname, json_filename)
        if not os.path.exists(json_file):
            raise ValueError(f"Summary JSON not found: {json_file}")
        df = vr.load_model_metrics(json_file, metrics, task_type=task_type)
        if task_type == "Detection":
            df = df[df["num_samples"] >= MINIMUM_GROUP_SIZE]
            if len(df) == 0:
                raise ValueError(
                    f"{display}: no Detection labels with >= {MINIMUM_GROUP_SIZE} samples."
                )
        data[display] = df
        dirs[display] = model_dir
    return all_display, data, dirs


def common_labels(data):
    """Sorted intersection of Target labels across all models (as in viz_radar)."""
    label_sets = [set(df["Target"]) for df in data.values()]
    labels = sorted(set.intersection(*label_sets)) if label_sets else []
    if not labels:
        raise ValueError("No common labels across models.")
    return labels


def draw_model_radar(ax, values, should_invert, color, label_numbers, label_names,
                     show_rings, samples_by_label=None):
    """One single-model radar cell, styled after viz_radar.plot_radar_chart.

    samples_by_label: optional {label: per-sample values}; overlaid as violin +
    box plots per spoke via viz_radar.plot_violin_on_spoke, in the model's own
    colour (the single-model analogue of --verbose_model).
    """
    n = len(label_numbers)
    angles = [i / float(n) * 2 * pi for i in range(n)]
    closed_angles = angles + angles[:1]

    if should_invert:
        plot_values = [1 - min(max(v, 0), 1) for v in values]
    else:
        plot_values = list(values)
    closed = plot_values + plot_values[:1]

    ax.plot(closed_angles, closed, linestyle="-", linewidth=2.0, alpha=0.9, color=color)
    ax.fill(closed_angles, closed, color=color, alpha=0.15, linewidth=0)

    if samples_by_label:
        for spoke_idx, name in enumerate(label_names):
            samples = samples_by_label.get(name, [])
            if not samples:
                continue
            vr.plot_violin_on_spoke(
                ax,
                angle=angles[spoke_idx],
                values=samples,
                should_invert=should_invert,
                color=color,
                N=n,
            )

    ax.set_xticks(angles)
    spoke_fontsize = 7.0 if n <= 16 else 6.0
    ax.set_xticklabels([str(num) for num in label_numbers], fontsize=spoke_fontsize)
    for tick_label, name in zip(ax.get_xticklabels(), label_names):
        if vr.is_purple_label(name):
            tick_label.set_color(C_TUMOR_LESION_LABEL)
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color(C_ANATOMY_LABEL)

    y_limit_max = 1.1
    ax.set_ylim(-0.3, y_limit_max)
    y_ticks = np.arange(0, y_limit_max, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([])

    if show_rings:
        for y_val in y_ticks:
            if y_val == 0:
                if not should_invert:
                    continue
                label_text = "≥1"
            else:
                label_text = f"{(1 - y_val):.1f}" if should_invert else f"{y_val:.1f}"
            ax.text(
                0,
                y_val,
                label_text,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=RING_FONTSIZE,
                color="#333333",
                zorder=10,
            )
    ax.grid(True)


def draw_label_mapping_row(ax, blocks, fig):
    """Render the number-to-name mapping blocks side by side in one wide axes.

    blocks: list of (title, labels, n_cols, use_monospace, force_purple).
    Text is laid out in inches (converted through the axes' final position), so
    the block widths track the actual figure geometry.
    """
    ax.axis("off")
    fig_w, fig_h = fig.get_size_inches()
    bbox = ax.get_position()
    ax_w_in = bbox.width * fig_w
    ax_h_in = bbox.height * fig_h
    line_h = MAP_LINE_IN / ax_h_in
    title_h = MAP_TITLE_IN / ax_h_in

    x_in = 0.0
    for title, labels, n_cols, use_monospace, force_purple in blocks:
        entries = [(i + 1, name) for i, name in enumerate(labels)]
        rows = ceil(len(entries) / n_cols)
        columns = [entries[c * rows:(c + 1) * rows] for c in range(n_cols)]

        ax.text(
            x_in / ax_w_in, 1.0, title,
            transform=ax.transAxes, fontsize=MAP_FONTSIZE + 2, fontweight="bold",
            verticalalignment="top",
        )
        char_in = MAP_FONTSIZE * (0.60 if use_monospace else 0.52) / 72
        col_x_in = x_in
        for col in columns:
            texts = [f"{num}: {vr.abbreviate_label_name(name)}" for num, name in col]
            for i, ((num, name), text) in enumerate(zip(col, texts)):
                purple = force_purple or vr.is_purple_label(name)
                kwargs = dict(
                    transform=ax.transAxes,
                    fontsize=MAP_FONTSIZE,
                    verticalalignment="top",
                    color=C_TUMOR_LESION_LABEL if purple else C_ANATOMY_LABEL,
                    fontweight="bold" if purple else "normal",
                )
                if use_monospace:
                    kwargs["fontfamily"] = "monospace"
                ax.text(col_x_in / ax_w_in, 1.0 - title_h - i * line_h, text, **kwargs)
            col_x_in += max(len(t) for t in texts) * char_in + MAP_COL_GAP_IN
        x_in = col_x_in - MAP_COL_GAP_IN + MAP_BLOCK_GAP_IN


def load_model_samples(display, columns, parsed_dirs):
    """Per-sample values for one model, for every (task, metric) a column needs.

    Detection sweeps the sample JSONLs once per metric (mirroring viz_radar's
    per-metric loading); AD is swept once and serves both the Angle and the
    Distance column, so the sweep must cover the UNION of the labels of every
    column sharing a samples_key — not just the first column's subset.

    Returns {samples_key: {label: [values]}}.
    """
    key_labels = {}
    for _, _, _, labels, _, _, samples_key in columns:
        bucket = key_labels.setdefault(samples_key, [])
        bucket.extend(label for label in labels if label not in bucket)

    samples = {}
    for samples_key, labels in key_labels.items():
        task, metric = samples_key
        samples[samples_key] = vr.load_per_sample_values(
            parsed_dirs[task], metric, labels, task_type=task
        )
        n_loaded = sum(len(v) for v in samples[samples_key].values())
        print(f"[Info] {display} / {task} / {metric}: {n_loaded} samples")
    return samples


def plot_model_grid(args, formats):
    Path(args.fig_dir).mkdir(exist_ok=True, parents=True)
    parsed = args.parsed_dirname
    tl_json = SUMMARY_FILENAME_TL_METRICS.replace(
        ".json", f"_filtered{vr._parsed_dir_suffix(parsed)}.json"
    )

    detect_all, detect_data, detect_dirs = load_task_data(
        args.config_detect, args.task_dir_detect, SUMMARY_FILENAME_DETECT_METRICS,
        DETECT_METRICS, "Detection", parsed, args.exclude_display,
    )
    ad_all, ad_data, ad_dirs = load_task_data(
        args.config_ad, args.task_dir_ad, SUMMARY_FILENAME_AD_METRICS,
        ["avgMRE"], "AD", parsed, args.exclude_display,
    )
    tl_all, tl_data, tl_dirs = load_task_data(
        args.config_tl, args.task_dir_tl, tl_json,
        ["avgMRE"], "TL", parsed, args.exclude_display,
    )

    # Display name is the cross-task join key, so the three configs must list
    # the same models in the same order (they also define the palette index).
    if not (detect_all == ad_all == tl_all):
        raise ValueError(
            "The three configs must list identical display names in identical order; "
            f"got Detection={detect_all}, AD={ad_all}, TL={tl_all}"
        )

    # Colour bound by index in the FULL config order (excluded models included),
    # so remaining models keep the colour they have in the multi-model figures.
    palette = vr.model_palette(len(detect_all))
    color_by_display = dict(zip(detect_all, palette))
    models = [d for d in detect_all if d in detect_data]
    for display in models:
        if display not in ad_data or display not in tl_data:
            raise ValueError(f"{display} missing from AD or TL data.")

    det_labels = common_labels(detect_data)
    ad_labels = common_labels(ad_data)
    ad_groups = vr.split_ad_labels(ad_labels)
    tl_labels = common_labels(tl_data)

    # (header, task_data, metric, labels, invert, mapping_key, samples_key)
    # samples_key = (task_type, metric): AD's Angle and Distance columns share
    # one key, so the AD sample JSONLs are swept once per model, not twice.
    columns = [
        (f"Detection: {m} ↑", detect_data, m, det_labels, False, "Detection",
         ("Detection", m))
        for m in DETECT_METRICS
    ]
    columns += [
        (f"{name}: MRE ↓", ad_data, "avgMRE", labels, True, name, ("AD", "avgMRE"))
        for name, labels in ad_groups
    ]
    columns += [
        ("T/L: MRE ↓", tl_data, "avgMRE", tl_labels, True, "T/L", ("TL", "avgMRE"))
    ]
    n_cols = len(columns)

    map_blocks = []
    seen = set()
    for _, _, _, labels, _, key, _ in columns:
        if key in seen:
            continue
        seen.add(key)
        map_blocks.append(
            (key, labels, MAP_COLS.get(key, 1), key != "T/L", key == "T/L")
        )
    map_rows = max(ceil(len(b[1]) / b[2]) for b in map_blocks)
    map_in = MAP_TITLE_IN + map_rows * MAP_LINE_IN + 0.25

    # Split the model rows over N_PARTS figures, first parts largest.
    per_part = ceil(len(models) / N_PARTS)
    parts = [models[i:i + per_part] for i in range(0, len(models), per_part)]

    cell = args.cell_inches
    # Model names are rotated 90°, so the left margin only needs one line height.
    left_in = 0.75
    right_in, top_in, bottom_in = 0.3, 0.7, 0.3
    fig_w = left_in + right_in + cell * (n_cols + (n_cols - 1) * WSPACE)

    stem_base = os.path.splitext(os.path.join(args.fig_dir, args.fig_name))[0]
    for part_idx, part_models in enumerate(parts, start=1):
        n_rows = len(part_models)
        fig_h = top_in + bottom_in + cell * n_rows * (1 + HSPACE) + map_in

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            n_rows + 1, n_cols, figure=fig,
            height_ratios=[cell] * n_rows + [map_in],
        )
        fig.subplots_adjust(
            left=left_in / fig_w, right=1 - right_in / fig_w,
            top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
            wspace=WSPACE, hspace=HSPACE,
        )

        for r, display in enumerate(part_models):
            color = color_by_display[display]
            parsed_dirs = {
                "Detection": os.path.join(
                    args.task_dir_detect, detect_dirs[display], parsed),
                "AD": os.path.join(args.task_dir_ad, ad_dirs[display], parsed),
                "TL": os.path.join(args.task_dir_tl, tl_dirs[display], parsed),
            }
            model_samples = load_model_samples(display, columns, parsed_dirs)

            for c, (header, task_data, metric, labels, invert, _, samples_key) \
                    in enumerate(columns):
                ax = fig.add_subplot(gs[r, c], projection="polar")
                df = task_data[display].set_index("Target").reindex(labels)
                values = df[metric].tolist()
                numbers = list(range(1, len(labels) + 1))
                # Ring value labels only where the scale changes: the first column
                # of each scale type (col 0: higher-is-better, col 3: inverted MRE).
                show_rings = c in (0, len(DETECT_METRICS))
                draw_model_radar(
                    ax, values, invert, color, numbers, labels, show_rings,
                    samples_by_label=model_samples[samples_key],
                )
                if r == 0:
                    ax.set_title(
                        header, fontsize=HEADER_FONTSIZE, fontweight="bold", pad=16
                    )
                if c == 0:
                    ax.text(
                        -0.12, 0.5, display,
                        transform=ax.transAxes, rotation=90,
                        horizontalalignment="center", verticalalignment="center",
                        fontsize=NAME_FONTSIZE, fontweight="bold", color=color,
                    )

        mapping_ax = fig.add_subplot(gs[n_rows, :])
        draw_label_mapping_row(mapping_ax, map_blocks, fig)

        stem = f"{stem_base}_part{part_idx}{vr._parsed_dir_suffix(parsed)}"
        for fmt in formats:
            save_fig_capped(f"{stem}.{fmt}", bbox_inches="tight", transparent=True)
            print(f"Figure saved to: {stem}.{fmt}")
        plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Per-model grids of all six task radar charts with per-sample violin "
            "overlays (one model per row, models split over two figures)."
        )
    )
    parser.add_argument("--config_detect", type=str, required=True)
    parser.add_argument("--task_dir_detect", type=str, required=True)
    parser.add_argument("--config_ad", type=str, required=True)
    parser.add_argument("--task_dir_ad", type=str, required=True)
    parser.add_argument("--config_tl", type=str, required=True)
    parser.add_argument("--task_dir_tl", type=str, required=True)
    parser.add_argument("--fig_dir", type=str, required=True)
    parser.add_argument(
        "--fig_name", type=str, default="fig_radar_grid.pdf",
        help="Output figure basename; parts are suffixed _part1/_part2 "
             "(default: fig_radar_grid.pdf).",
    )
    parser.add_argument(
        "--parsed_dirname", type=str, default="parsed",
        help=(
            "Per-model subdirectory to read summaries/samples from. Non-default "
            "sources suffix the output figure names with __{parsed_dirname}. "
            "Default: parsed."
        ),
    )
    parser.add_argument(
        "--cell_inches", type=float, default=3.0,
        help="Width in inches for each radar cell (default: 3).",
    )
    parser.add_argument(
        "--exclude_display", type=str, default="MedVision",
        help=(
            "Drop models whose display name contains this substring "
            "(default: MedVision; pass '' to keep every model)."
        ),
    )
    parser.add_argument("--save_as_png", action="store_true", help="Save figures as PNG.")
    parser.add_argument("--save_as_pdf", action="store_true", help="Save figures as PDF.")
    args = parser.parse_args()

    formats = [
        f for f, on in (("png", args.save_as_png), ("pdf", args.save_as_pdf)) if on
    ] or ["pdf"]
    plot_model_grid(args, formats)


if __name__ == "__main__":
    main()
