import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from medvision_bm.utils.configs import (
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_MEAN_METRICS,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS,
)

SAMPLE_SIZE_THRESHOLD_LABEL = (
    100  # Minimum sample size for a label to be included in the plot
)
SAMPLE_SIZE_THRESHOLD_BOX = (
    30  # Minimum sample size for a box-image ratio group to be included in the plot
)


def plot_label_composition_and_metrics(
    all_data,
    model_name_display_map,
    figsize=(22, 26),
    min_sample_size=50,
    output_dir=None,
    output_filename=None,
    formats=("pdf",),
):
    """
    Plot the composition of sample size and metrics for each label across all models.

    Parameters:
    all_data: dict of DataFrames, key is model name, value is DataFrame
    model_name_display_map: dict mapping folder names to display names
    figsize: tuple, figure size (width, height)
    min_sample_size: minimum total sample size for a label to be included
    output_dir: directory to save the output plot
    """

    first_model = list(all_data.keys())[0]
    df = all_data[first_model]

    ordered_box_groups = [
        "<0.05",
        "0.05~0.10",
        "0.10~0.15",
        "0.15~0.20",
        "0.20~0.25",
        "0.25~0.30",
        "0.30~0.35",
        "0.35~0.40",
        "0.40~0.45",
        "0.45~0.50",
        "0.50~0.55",
        "0.55~0.60",
        "0.60~0.65",
        "0.65~0.70",
        "0.70~0.75",
        "0.75~0.80",
        "0.80~0.85",
        "0.85~0.90",
        ">=0.90",
    ]

    box_groups = [bg for bg in ordered_box_groups if bg in df["box_img_group"].unique()]

    pivot_df = df.pivot_table(
        index="label", columns="box_img_group", values="sample_size", fill_value=0
    )

    pivot_df["total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df[pivot_df["total"] >= min_sample_size]
    pivot_df = pivot_df.sort_values("total", ascending=False)
    label_order = pivot_df.index.tolist()
    pivot_df = pivot_df.drop("total", axis=1)

    bar_width = 1 * 1.5
    bar_positions = np.arange(0, len(pivot_df) * (bar_width * 1.5), bar_width * 1.5)
    bar_positions = bar_positions + bar_width / 2

    max_sample_size = pivot_df.sum(axis=1).max()

    metrics_to_plot = ["Recall", "Precision", "F1"]

    height_ratios = [1.2] * len(metrics_to_plot) + [0.7]
    fig, axes = plt.subplots(
        len(metrics_to_plot) + 1,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )

    colors = [
        plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)]
        for i in range(len(ordered_box_groups))
    ]

    markers = [
        "o",
        "s",
        "D",
        "p",
        "d",
        "^",
        "v",
        "<",
        ">",
        "X",
        "P",
        "H",
        "*",
        "h",
        "8",
        "1",
        "2",
        "3",
        "4",
        "x",
    ]

    for metric_idx, metric in enumerate(metrics_to_plot):
        ax = axes[metric_idx]

        section_width = bar_width * 1.5
        for i in range(0, len(bar_positions), 2):
            start_pos = bar_positions[i] - section_width / 2
            end_pos = start_pos + section_width
            ax.axvspan(start_pos, end_pos, facecolor="lightgrey", alpha=0.2, zorder=0)

        for model_idx, (model_name, model_data) in enumerate(all_data.items()):
            marker = markers[model_idx % len(markers)]

            model_pivot = model_data.pivot_table(
                index="label", columns="box_img_group", values=metric, fill_value=0
            )
            model_sample_pivot = model_data.pivot_table(
                index="label",
                columns="box_img_group",
                values="sample_size",
                fill_value=0,
            )

            model_pivot = model_pivot.reindex(label_order, fill_value=0)
            model_sample_pivot = model_sample_pivot.reindex(label_order, fill_value=0)

            for bg_idx, box_group in enumerate(box_groups):
                if box_group in model_pivot.columns:
                    values = model_pivot[box_group].values
                    sample_sizes = (
                        model_sample_pivot[box_group].values
                        if box_group in model_sample_pivot.columns
                        else np.zeros_like(values)
                    )
                    color = colors[ordered_box_groups.index(box_group)]

                    mask = (values > 0) & (sample_sizes >= SAMPLE_SIZE_THRESHOLD_BOX)
                    if np.any(mask):
                        face_color = (
                            color if marker in ["1", "2", "3", "4", "x"] else "none"
                        )
                        ax.scatter(
                            bar_positions[mask],
                            values[mask],
                            marker=marker,
                            facecolors=face_color,
                            edgecolors=color,
                            linewidths=2,
                            s=150,
                            alpha=0.9,
                            label=model_name if bg_idx == 0 else "",
                        )

        ax.set_xticks(bar_positions)
        ax.set_xticklabels([])
        ax.set_ylabel(metric, fontsize=20, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(bar_positions[0] - bar_width, bar_positions[-1] + bar_width)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="y", labelsize=14)

    ax_sample = axes[-1]

    section_width = bar_width * 1.5
    for i in range(0, len(bar_positions), 2):
        start_pos = bar_positions[i] - section_width / 2
        end_pos = start_pos + section_width
        ax_sample.axvspan(
            start_pos, end_pos, facecolor="lightgrey", alpha=0.2, zorder=0
        )

    left = np.zeros(len(pivot_df))
    for i, box_group in enumerate(box_groups):
        if box_group in pivot_df.columns:
            values = pivot_df[box_group].values
            ax_sample.bar(
                bar_positions,
                values,
                bottom=left,
                width=bar_width,
                label=box_group,
                color=colors[ordered_box_groups.index(box_group)],
                alpha=0.8,
            )
            left += values

    ax_sample.set_xticks(bar_positions)
    ax_sample.set_xticklabels(pivot_df.index, fontsize=14, rotation=90, ha="center")

    for tick in ax_sample.get_xticklabels():
        text = tick.get_text().lower()
        if any(
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
        ):
            tick.set_color("#770087")

    ax_sample.set_ylabel("Sample Size", fontsize=20, fontweight="bold")
    ax_sample.grid(axis="y", alpha=0.3)
    ax_sample.set_xlim(bar_positions[0] - bar_width, bar_positions[-1] + bar_width)
    ax_sample.set_ylim(0, max_sample_size * 1.1)
    ax_sample.tick_params(axis="y", labelsize=14)

    plt.tight_layout()

    box_legend = ax_sample.legend(
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        fontsize=16,
        ncol=2,
    )
    box_legend.set_title("Box-to-Image Ratio", prop={"weight": "bold", "size": 16})

    model_handles, model_labels = [], []
    handles, labels = axes[0].get_legend_handles_labels()
    for model_name in all_data.keys():
        for handle, label in zip(handles, labels):
            if label == model_name:
                model_handles.append(handle)
                model_labels.append(model_name_display_map.get(model_name, model_name))
                break

    model_legend = fig.legend(
        handles=model_handles,
        labels=model_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        fontsize=16,
        ncol=6,
    )
    model_legend.set_title("Models", prop={"weight": "bold", "size": 16})

    ax_sample.add_artist(box_legend)

    # Dynamically compute the required bottom margin so the legend does not
    # overlap the rotated x-tick labels on ax_sample.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_height_px = fig.get_figheight() * fig.dpi

    # Legend height in figure fraction (legend is anchored at figure y=0)
    legend_h_frac = model_legend.get_window_extent(renderer).height / fig_height_px

    # How far the rotated x-tick labels extend below ax_sample's bottom edge
    ax_bottom_frac = ax_sample.get_position().y0
    tick_ext_frac = 0.0
    for tick in ax_sample.get_xticklabels():
        tb = tick.get_window_extent(renderer)
        ext = ax_bottom_frac - tb.y0 / fig_height_px
        if ext > tick_ext_frac:
            tick_ext_frac = ext

    plt.subplots_adjust(bottom=legend_h_frac + tick_ext_frac + 0.01)

    if output_dir and output_filename:
        stem = os.path.splitext(os.path.join(output_dir, output_filename))[0]
        for fmt in formats:
            output_path = f"{stem}.{fmt}"
            plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
            print(f"Saved figure to {output_path}")

    plt.show()

    return fig, axes


def _parsed_dir_suffix(parsed_dirname):
    """Mirror of the summarizers' output naming: ``""`` for ``"parsed"``, else
    ``"__{parsed_dirname}"``. Applied to the output figure stem so figures from a
    non-default source (e.g. llm-parsed_gemma-4-31b) cannot overwrite published ones."""
    return "" if parsed_dirname == "parsed" else f"__{parsed_dirname}"


def main(
    in_dir,
    out_dir,
    model_name_display_map,
    folders,
    use_label_level=True,
    formats=("pdf",),
    parsed_dirname="parsed",
):
    if use_label_level:
        csv_filename = SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_MEAN_METRICS
        fig_filename = "fig_detection__metrics-boxSize__labelLevel"
    else:
        csv_filename = (
            SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS
        )
        fig_filename = "fig_detection__metrics-boxSize__anatomyLevel"
    fig_filename += _parsed_dir_suffix(parsed_dirname)

    if not folders:
        raise ValueError("No models listed under model_display_name in the config.")

    all_data = {}
    missing = []
    for folder in folders:
        csv_path = os.path.join(in_dir, folder, parsed_dirname, csv_filename)
        if os.path.exists(csv_path):
            all_data[folder] = pd.read_csv(csv_path)
            print(f"Loaded data for {folder}")
        else:
            missing.append(csv_path)

    # Fail loudly: with a source knob, a wrong parsed_dirname would otherwise skip
    # every model and render nothing, which reads as "no data" not "wrong source".
    if missing:
        raise FileNotFoundError(
            f"Missing per-label CSV for {len(missing)}/{len(folders)} configured "
            f"models under '{parsed_dirname}'. Generate them first with:\n"
            f"  python -m medvision_bm.benchmark.analyze_detection_task_boxsize "
            f"--task_dir {in_dir} --parsed_dirname {parsed_dirname}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    plot_label_composition_and_metrics(
        all_data,
        model_name_display_map,
        min_sample_size=SAMPLE_SIZE_THRESHOLD_LABEL,
        output_dir=out_dir,
        output_filename=fig_filename,
        formats=formats,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot metrics and sample size distribution per label and box size for multiple models."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (model_display_name mapping)",
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Directory containing model subfolders (each with a {parsed_dirname}/ subdirectory)",
    )
    parser.add_argument(
        "--parsed_dirname",
        type=str,
        default="parsed",
        help=(
            "Per-model subdirectory to read the per-label CSVs from, e.g. "
            "llm-parsed_gemma-4-31b. Non-default sources suffix the output figure "
            "name with __{parsed_dirname}. Default: parsed."
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save the output figure",
    )

    level_group = parser.add_mutually_exclusive_group()
    level_group.add_argument(
        "--label_level",
        action="store_true",
        help="Read fine-grained label CSV (default). Outputs fig_detection__metrics-boxSize__labelLevel.pdf",
    )
    level_group.add_argument(
        "--anatomy_level",
        action="store_true",
        help="Read anatomy-grouped label CSV. Outputs fig_detection__metrics-boxSize__anatomyLevel.pdf",
    )
    parser.add_argument(
        "--save_as_png", action="store_true", help="Save figures as PNG."
    )
    parser.add_argument(
        "--save_as_pdf", action="store_true", help="Save figures as PDF."
    )

    args = parser.parse_args()
    use_label_level = not args.anatomy_level  # default: label_level

    formats = [
        f for f, on in (("png", args.save_as_png), ("pdf", args.save_as_pdf)) if on
    ] or ["pdf"]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name_display_map = cfg["model_display_name"]
    folders = list(model_name_display_map.keys())

    main(
        args.in_dir,
        args.out_dir,
        model_name_display_map,
        folders,
        use_label_level=use_label_level,
        formats=formats,
        parsed_dirname=args.parsed_dirname,
    )
