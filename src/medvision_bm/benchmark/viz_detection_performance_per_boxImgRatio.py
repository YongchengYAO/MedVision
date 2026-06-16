import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from medvision_bm.utils.configs import (
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS,
)


def load_model_metrics(json_path):
    """Load and process metrics from a single model's JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    targets = []
    f1_values = []
    precision_values = []
    recall_values = []
    num_samples = []

    for target, metrics in data.items():
        if "F1" in metrics:
            targets.append(target)
            f1_values.append(metrics["F1"])
            precision_values.append(metrics["Precision"])
            recall_values.append(metrics["Recall"])
            num_samples.append(metrics["num_samples"])

    df = pd.DataFrame(
        {
            "Target": targets,
            "F1": f1_values,
            "Precision": precision_values,
            "Recall": recall_values,
            "num_samples": num_samples,
        }
    )

    return df.sort_values("Recall", ascending=False)


def extract_ratio_midpoint(range_str):
    mapping = {
        "Box/Image < 5%": 0.025,
        "5% <= Box/Image < 10%": 0.075,
        "10% <= Box/Image < 15%": 0.125,
        "15% <= Box/Image < 20%": 0.175,
        "20% <= Box/Image < 25%": 0.225,
        "25% <= Box/Image < 30%": 0.275,
        "30% <= Box/Image < 35%": 0.325,
        "35% <= Box/Image < 40%": 0.375,
        "40% <= Box/Image < 45%": 0.425,
        "45% <= Box/Image < 50%": 0.475,
        "50% <= Box/Image < 55%": 0.525,
        "55% <= Box/Image < 60%": 0.575,
        "60% <= Box/Image < 65%": 0.625,
        "65% <= Box/Image < 70%": 0.675,
        "70% <= Box/Image < 75%": 0.725,
        "75% <= Box/Image < 80%": 0.775,
        "80% <= Box/Image < 85%": 0.825,
        "85% <= Box/Image < 90%": 0.875,
        "Box/Image >= 90%": 0.95,
    }
    return mapping.get(range_str)


def plot_metrics_multi_model(in_dir, out_dir, model_name_display_map, folders):
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    model_data = {}
    for model_dir in folders:
        model_path = os.path.join(in_dir, model_dir)
        if os.path.isdir(model_path):
            json_file = os.path.join(
                model_path, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS
            )
            if os.path.exists(json_file):
                model_data[model_dir] = load_model_metrics(json_file)

    # Re-process to get ratio-keyed series per model
    processed_data = {}
    for model_name in model_data:
        display_name = model_name_display_map.get(model_name, model_name)
        json_file = os.path.join(
            in_dir,
            model_name,
            SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS,
        )
        with open(json_file, "r") as f:
            ratio_data = json.load(f)

        ratios, recalls, precisions, f1s = [], [], [], []
        for ratio_range, metrics in ratio_data.items():
            if "F1" in metrics:
                ratio = extract_ratio_midpoint(ratio_range)
                if ratio is None:
                    continue
                ratios.append(ratio)
                recalls.append(metrics["Recall"])
                precisions.append(metrics["Precision"])
                f1s.append(metrics["F1"])

        sorted_data = sorted(zip(ratios, recalls, precisions, f1s))
        processed_data[display_name] = {
            "ratios": [x[0] for x in sorted_data],
            "Recall": [x[1] for x in sorted_data],
            "Precision": [x[2] for x in sorted_data],
            "F1": [x[3] for x in sorted_data],
        }

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    metrics = ["Recall", "Precision", "F1"]

    base_colors = plt.cm.tab10.colors
    colors = []
    for i in range(len(folders)):
        if i < len(base_colors):
            colors.append(base_colors[i])
        else:
            base_color = base_colors[i % len(base_colors)]
            darker_color = tuple(c * 0.8 for c in base_color[:3]) + (
                base_color[3] if len(base_color) > 3 else ()
            )
            colors.append(darker_color)

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

    for i, metric in enumerate(metrics):
        ax = axs[i]
        for j, (model_name, data) in enumerate(processed_data.items()):
            if model_name == "Random":
                color = "black"
                marker = "*"
                markersize = 12
                line_style = "--"
            else:
                color = colors[j % len(colors)]
                marker = markers[j % len(markers)]
                markersize = 8
                line_style = "-"
            ax.plot(
                data["ratios"],
                data[metric],
                marker=marker,
                linestyle=line_style,
                label=model_name,
                color=color,
                linewidth=2,
                markersize=markersize,
            )
        ax.set_xlabel("Box-to-Image Ratio", fontsize=20)
        ax.set_ylabel(metric, fontsize=20, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 0.55, 0.05))
        ax.set_xticklabels([f"{x:.2f}" for x in np.arange(0, 0.55, 0.05)], fontsize=14)
        ax.tick_params(axis="y", labelsize=14)

    # Sample-size bar chart (fourth subplot)
    ax = axs[3]
    first_model = list(model_data.keys())[0]
    first_df = model_data[first_model].copy()
    first_df["Ratio"] = first_df["Target"].apply(extract_ratio_midpoint)
    first_df = first_df.sort_values("Ratio")

    boximg_ratio_map = {
        "Box/Image < 5%": "<0.05",
        "5% <= Box/Image < 10%": "0.05~0.10",
        "10% <= Box/Image < 15%": "0.10~0.15",
        "15% <= Box/Image < 20%": "0.15~0.20",
        "20% <= Box/Image < 25%": "0.20~0.25",
        "25% <= Box/Image < 30%": "0.25~0.30",
        "30% <= Box/Image < 35%": "0.30~0.35",
        "35% <= Box/Image < 40%": "0.35~0.40",
        "40% <= Box/Image < 45%": "0.40~0.45",
        "45% <= Box/Image < 50%": "0.45~0.50",
        "50% <= Box/Image < 55%": "0.50~0.55",
        "55% <= Box/Image < 60%": "0.55~0.60",
        "60% <= Box/Image < 65%": "0.60~0.65",
        "65% <= Box/Image < 70%": "0.65~0.70",
        "70% <= Box/Image < 75%": "0.70~0.75",
        "75% <= Box/Image < 80%": "0.75~0.80",
        "80% <= Box/Image < 85%": "0.80~0.85",
        "85% <= Box/Image < 90%": "0.85~0.90",
        "Box/Image >= 90%": ">=0.90",
    }

    y_pos = np.arange(len(first_df))
    bars = ax.barh(
        y_pos,
        first_df["num_samples"],
        color="#FEB05C",
        edgecolor="#F37600",
        linewidth=2,
    )
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=14)

    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    x_disp, _ = ax.transAxes.transform((0.2, 0))
    threshold = ax.transData.inverted().transform((x_disp, 0))[0]

    for bar, size in zip(bars, first_df["num_samples"]):
        width = bar.get_width()
        if width < threshold:
            text_x = width * 1.05
            ha = "left"
        else:
            text_x = width * 0.95
            ha = "right"
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(size):,}",
            va="center",
            ha=ha,
            fontsize=14,
            fontweight="bold",
        )

    formatted_labels = [boximg_ratio_map.get(t) for t in first_df["Target"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(formatted_labels, fontsize=14, rotation=45)
    ax.set_ylabel("Box-to-Image Ratio", fontsize=20)
    ax.set_xlabel("Sample Size", fontsize=20, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7, axis="x")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=min(5, len(processed_data)),
        fontsize=14,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    output_path = os.path.join(out_dir, "metrics_boxImgRatio-dotline.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot detection metrics vs box-to-image ratio for multiple models."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config-detect-boxImgRatio.yaml"),
        help="Path to the YAML config file (default: config-detect-boxImgRatio.yaml next to this script)",
    )
    parser.add_argument(
        "--in_dir",
        required=True,
        help="Directory containing model subfolders with metrics JSON files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save the output figure",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name_display_map = cfg["model_display_name"]
    folders = list(model_name_display_map.keys())

    plot_metrics_multi_model(args.in_dir, args.out_dir, model_name_display_map, folders)
