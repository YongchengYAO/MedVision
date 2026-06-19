"""
Assemble a cross-model comparison grid from pre-generated per-sample subfigures.

Reads a base directory whose immediate subfolders are model output directories
(one per model), each containing per-dataset subfolders of per-sample PNGs. Those
PNGs are produced by the per-task visualizers in this folder:
viz_tl_axes.py (TL), viz_ad_landmarks.py (AD), viz_detection_boxes.py (Detection).
A fixed random subset (seeded from configs.py SEED, overridable via --seed) is
sampled equally across the datasets common to all models, then laid out into a
single comparison figure.

Three layout modes:
    default          rows are grouped per model (--row_per_model rows each),
                     samples flow left-to-right then top-to-bottom.
    --dataset_as_col each column = one dataset; rows = samples within it.
                     Split into vertically stacked panels with
                     --dataset_as_col_num_panel to reduce width.
    --dataset_as_row each row = one dataset; columns = samples within it.
                     Split into side-by-side panels with --dataset_as_row_num_panel
                     to reduce height; wrap samples across multiple rows per dataset
                     with --dataset_as_row_num_row_per_ds.
When --dir_model selects a single model, the rotated model-name label column and
inter-model separators are omitted.

Usage:
    python viz_compile_grid.py \
        --dir_subfigures <base_dir_of_model_subfolders> \
        --limit_subfigures <N >= num_datasets> \
        --output <output.png> \
        [--row_per_model N] \
        [--dir_model <model_folder_name>] \
        [--seed N] \
        [--dataset_as_col [--dataset_as_col_num_panel N]] \
        [--dataset_as_row [--dataset_as_row_num_panel N] [--dataset_as_row_num_row_per_ds N]]
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from medvision_bm.utils.plot_utils import FIG_DPI, save_fig_capped, save_img_capped

from medvision_bm.utils.configs import SEED

MODEL_NAME_MAP = {
    "MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250": "fullRFT",
    "MedVision__fullSFT__Qwen2.5VL-7B__D110k-AD5k-TL5k__CoT__512x512__v2": "fullSFT",
}


def _select_samples(models, limit, rng):
    """Return ordered list of (dataset, filename) with equal samples per dataset.

    Each dataset receives ceil(limit / n_datasets) samples, so total may exceed
    limit by up to n_datasets-1 when limit is not divisible by n_datasets.
    Samples from the same dataset are consecutive in the returned list.
    """
    datasets_per_model = {
        m.name: sorted([d.name for d in m.iterdir() if d.is_dir()]) for m in models
    }
    common_datasets = sorted(
        set.intersection(*[set(v) for v in datasets_per_model.values()])
    )
    n_datasets = len(common_datasets)
    if limit < n_datasets:
        raise ValueError(
            f"--limit_subfigures ({limit}) is less than the number of datasets ({n_datasets}). "
            f"Datasets: {common_datasets}"
        )

    common_files = {}
    for dataset in common_datasets:
        files_sets = [set(f.name for f in (m / dataset).glob("*.png")) for m in models]
        common_files[dataset] = sorted(set.intersection(*files_sets))
        if not common_files[dataset]:
            raise ValueError(
                f"No common PNG files found across all models for dataset '{dataset}'"
            )

    per_dataset = math.ceil(limit / n_datasets)
    selected_by_dataset = {}
    for dataset in common_datasets:
        pool = list(common_files[dataset])
        rng.shuffle(pool)
        selected_by_dataset[dataset] = pool[:per_dataset]

    return [(ds, f) for ds in common_datasets for f in selected_by_dataset[ds]]


def _draw_model_labels_and_separators(fig, models, anchor_axes, show_model_label):
    """Draw rotated model name text and horizontal separator lines between model groups."""
    n_models = len(models)
    for model_idx, model in enumerate(models):
        bbox_first = anchor_axes[model_idx][0].get_position()
        bbox_last = anchor_axes[model_idx][-1].get_position()

        if show_model_label:
            model_name = MODEL_NAME_MAP.get(model.name, model.name)
            y_center = (bbox_first.y1 + bbox_last.y0) / 2
            x_label = (bbox_first.x0 + bbox_first.x1) / 2
            fig.text(
                x_label,
                y_center,
                model_name,
                rotation=90,
                va="center",
                ha="center",
                fontsize=11,
                fontweight="bold",
                transform=fig.transFigure,
            )

        if model_idx < n_models - 1:
            bbox_next = anchor_axes[model_idx + 1][0].get_position()
            y_sep = (bbox_last.y0 + bbox_next.y1) / 2
            fig.add_artist(
                Line2D(
                    [0.0, 1.0],
                    [y_sep, y_sep],
                    transform=fig.transFigure,
                    color="black",
                    linewidth=1.5,
                    clip_on=False,
                )
            )


def _compile_figure(models, samples, row_per_model, output, show_model_label=True):
    n_cols_img = math.ceil(len(samples) / row_per_model)
    n_models = len(models)
    n_total_rows = n_models * row_per_model

    cell_h = 3.0
    cell_w = 3.0
    label_w = 0.6

    col_offset = 1 if show_model_label else 0
    n_cols = n_cols_img + col_offset
    fig_w = (label_w if show_model_label else 0) + n_cols_img * cell_w
    fig_h = n_total_rows * cell_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios = ([label_w / cell_w] if show_model_label else []) + [1.0] * n_cols_img
    gs = GridSpec(
        n_total_rows,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.04,
        wspace=0.04,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )

    # anchor_axes[model_idx][row_in_model] = leftmost axis of that row (for label/separator positioning)
    anchor_axes = []
    for model_idx, model in enumerate(models):
        model_anchors = []
        for row_in_model in range(row_per_model):
            row_idx = model_idx * row_per_model + row_in_model

            if show_model_label:
                ax_lbl = fig.add_subplot(gs[row_idx, 0])
                ax_lbl.axis("off")
                model_anchors.append(ax_lbl)

            for col_idx in range(n_cols_img):
                sample_idx = row_in_model * n_cols_img + col_idx
                ax_img = fig.add_subplot(gs[row_idx, col_idx + col_offset])
                ax_img.axis("off")
                if not show_model_label and col_idx == 0:
                    model_anchors.append(ax_img)
                if sample_idx < len(samples):
                    dataset, filename = samples[sample_idx]
                    img = mpimg.imread(str(model / dataset / filename))
                    ax_img.imshow(img, aspect="auto")

        anchor_axes.append(model_anchors)

    fig.canvas.draw()
    _draw_model_labels_and_separators(fig, models, anchor_axes, show_model_label)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig_capped(str(out_path), fig=fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _make_dataset_as_col_panel(
    models, ds_group, samples_by_dataset, rows_per_model, panel_size, show_model_label
):
    """Render one dataset-as-col panel into a new Figure and return it (caller must close)."""
    n_models = len(models)
    n_total_rows = n_models * rows_per_model
    cell_h, cell_w, label_w, header_h = 3.0, 3.0, 0.6, 0.15
    col_offset = 1 if show_model_label else 0

    fig_w = (label_w if show_model_label else 0) + panel_size * cell_w
    fig_h = n_total_rows * cell_h + header_h * cell_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios = ([label_w / cell_w] if show_model_label else []) + [1.0] * panel_size
    gs = GridSpec(
        1 + n_total_rows,
        panel_size + col_offset,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[header_h] + [1.0] * n_total_rows,
        hspace=0.00,
        wspace=0.04,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )

    for ds_idx, dataset in enumerate(ds_group):
        ax_hdr = fig.add_subplot(gs[0, ds_idx + col_offset])
        ax_hdr.axis("off")
        ax_hdr.text(
            0.5,
            0.05,
            dataset,
            va="bottom",
            ha="center",
            fontsize=16,
            fontweight="bold",
            transform=ax_hdr.transAxes,
        )

    anchor_axes = []
    for model_idx, model in enumerate(models):
        model_anchors = []
        for row_in_model in range(rows_per_model):
            row_idx = 1 + model_idx * rows_per_model + row_in_model

            if show_model_label:
                ax_lbl = fig.add_subplot(gs[row_idx, 0])
                ax_lbl.axis("off")
                model_anchors.append(ax_lbl)

            for ds_idx, dataset in enumerate(ds_group):
                ax_img = fig.add_subplot(gs[row_idx, ds_idx + col_offset])
                ax_img.axis("off")
                if not show_model_label and ds_idx == 0:
                    model_anchors.append(ax_img)
                ds_files = samples_by_dataset[dataset]
                if row_in_model < len(ds_files):
                    img = mpimg.imread(str(model / dataset / ds_files[row_in_model]))
                    ax_img.imshow(img, aspect="auto")

        anchor_axes.append(model_anchors)

    fig.canvas.draw()
    _draw_model_labels_and_separators(fig, models, anchor_axes, show_model_label)
    return fig


def _compile_figure_dataset_as_col(
    models, samples, output, show_model_label=True, num_panel=1
):
    """Dataset-as-columns layout: each column = one dataset; rows = samples within that dataset.

    If num_panel > 1, datasets are split into vertically stacked panels to reduce figure width.
    Each panel is rendered as a separate Figure and stitched with PIL.
    """
    samples_by_dataset = {}
    dataset_order = []
    for ds, f in samples:
        if ds not in samples_by_dataset:
            samples_by_dataset[ds] = []
            dataset_order.append(ds)
        samples_by_dataset[ds].append(f)

    rows_per_model = max(len(v) for v in samples_by_dataset.values())
    panel_size = math.ceil(len(dataset_order) / num_panel)
    dataset_groups = [
        dataset_order[i * panel_size : (i + 1) * panel_size] for i in range(num_panel)
    ]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if num_panel == 1:
        fig = _make_dataset_as_col_panel(
            models,
            dataset_groups[0],
            samples_by_dataset,
            rows_per_model,
            panel_size,
            show_model_label,
        )
        save_fig_capped(str(out_path), fig=fig, bbox_inches="tight")
        plt.close(fig)
    else:
        import io

        from PIL import Image

        panel_images = []
        for ds_group in dataset_groups:
            fig = _make_dataset_as_col_panel(
                models,
                ds_group,
                samples_by_dataset,
                rows_per_model,
                panel_size,
                show_model_label,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=FIG_DPI)
            plt.close(fig)
            buf.seek(0)
            panel_images.append(Image.open(buf).copy())

        max_w = max(img.width for img in panel_images)
        total_h = sum(img.height for img in panel_images)
        result = Image.new("RGB", (max_w, total_h), "white")
        y = 0
        for img in panel_images:
            result.paste(img, (0, y))
            y += img.height
        save_img_capped(result, str(out_path))

    print(f"Saved: {out_path}")


def _make_dataset_as_row_panel(
    models,
    ds_group,
    samples_by_dataset,
    cols_per_dataset,
    panel_size,
    show_model_label,
    num_row_per_ds=1,
):
    """Render one dataset-as-row panel into a new Figure and return it (caller must close)."""
    n_models = len(models)
    cell_h, cell_w, label_w, ds_label_w = 3.0, 3.0, 0.6, 0.6
    n_label_cols = (1 if show_model_label else 0) + 1

    fig_w = (
        (label_w if show_model_label else 0) + ds_label_w + cols_per_dataset * cell_w
    )
    fig_h = n_models * panel_size * num_row_per_ds * cell_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    width_ratios = (
        ([label_w / cell_w] if show_model_label else [])
        + [ds_label_w / cell_w]
        + [1.0] * cols_per_dataset
    )
    total_rows = n_models * panel_size * num_row_per_ds
    gs = GridSpec(
        total_rows,
        n_label_cols + cols_per_dataset,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.04,
        wspace=0.04,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
    )

    ds_label_col_idx = 1 if show_model_label else 0
    img_col_offset = n_label_cols

    anchor_axes = []
    for model_idx, model in enumerate(models):
        model_anchors = []
        for ds_idx, dataset in enumerate(ds_group):
            base_row = model_idx * panel_size * num_row_per_ds + ds_idx * num_row_per_ds

            if show_model_label:
                ax_model_lbl = fig.add_subplot(
                    gs[base_row : base_row + num_row_per_ds, 0]
                )
                ax_model_lbl.axis("off")
                model_anchors.append(ax_model_lbl)

            ax_ds_lbl = fig.add_subplot(
                gs[base_row : base_row + num_row_per_ds, ds_label_col_idx]
            )
            ax_ds_lbl.axis("off")
            ax_ds_lbl.text(
                0.5,
                0.5,
                dataset,
                va="center",
                ha="center",
                fontsize=16,
                fontweight="bold",
                rotation=90,
                transform=ax_ds_lbl.transAxes,
            )
            if not show_model_label:
                model_anchors.append(ax_ds_lbl)

            ds_files = samples_by_dataset[dataset]
            for row_in_ds in range(num_row_per_ds):
                row_idx = base_row + row_in_ds
                for col_idx in range(cols_per_dataset):
                    sample_idx = row_in_ds * cols_per_dataset + col_idx
                    ax_img = fig.add_subplot(gs[row_idx, img_col_offset + col_idx])
                    ax_img.axis("off")
                    if sample_idx < len(ds_files):
                        img = mpimg.imread(str(model / dataset / ds_files[sample_idx]))
                        ax_img.imshow(img, aspect="auto")

        anchor_axes.append(model_anchors)

    fig.canvas.draw()
    _draw_model_labels_and_separators(fig, models, anchor_axes, show_model_label)
    return fig


def _compile_figure_dataset_as_row(
    models, samples, output, show_model_label=True, num_panel=1, num_row_per_ds=1
):
    """Dataset-as-rows layout: each row = one dataset; columns = samples within that dataset.

    If num_panel > 1, datasets are split into horizontally arranged side-by-side panels
    to reduce figure height. Each panel is rendered as a separate Figure and stitched with PIL.
    """
    samples_by_dataset = {}
    dataset_order = []
    for ds, f in samples:
        if ds not in samples_by_dataset:
            samples_by_dataset[ds] = []
            dataset_order.append(ds)
        samples_by_dataset[ds].append(f)

    cols_per_dataset = math.ceil(
        max(len(v) for v in samples_by_dataset.values()) / num_row_per_ds
    )
    panel_size = math.ceil(len(dataset_order) / num_panel)
    dataset_groups = [
        dataset_order[i * panel_size : (i + 1) * panel_size] for i in range(num_panel)
    ]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if num_panel == 1:
        fig = _make_dataset_as_row_panel(
            models,
            dataset_groups[0],
            samples_by_dataset,
            cols_per_dataset,
            panel_size,
            show_model_label,
            num_row_per_ds=num_row_per_ds,
        )
        save_fig_capped(str(out_path), fig=fig, bbox_inches="tight")
        plt.close(fig)
    else:
        import io

        from PIL import Image

        panel_images = []
        for ds_group in dataset_groups:
            fig = _make_dataset_as_row_panel(
                models,
                ds_group,
                samples_by_dataset,
                cols_per_dataset,
                panel_size,
                show_model_label,
                num_row_per_ds=num_row_per_ds,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=FIG_DPI)
            plt.close(fig)
            buf.seek(0)
            panel_images.append(Image.open(buf).copy())

        total_w = sum(img.width for img in panel_images)
        max_h = max(img.height for img in panel_images)
        result = Image.new("RGB", (total_w, max_h), "white")
        x = 0
        for img in panel_images:
            result.paste(img, (x, 0))
            x += img.width
        save_img_capped(result, str(out_path))

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile a cross-model comparison figure from pre-generated subfigures."
    )
    parser.add_argument(
        "--dir_subfigures",
        required=True,
        help="Base directory containing model subfolders (e.g. Figures/MedVision-TL-v2-CoT)",
    )
    parser.add_argument(
        "--limit_subfigures",
        type=int,
        required=True,
        help="Total number of samples to show per model (must be >= number of datasets)",
    )
    parser.add_argument(
        "--row_per_model",
        type=int,
        default=1,
        help="Number of rows per model group (default: 1)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG file path",
    )
    parser.add_argument(
        "--dir_model",
        default=None,
        help="If set, only plot rows for this model folder (path or folder name)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides the default from configs.py)",
    )
    parser.add_argument(
        "--dataset_as_col",
        action="store_true",
        help="If set, each column = one dataset; rows = samples within that dataset",
    )
    parser.add_argument(
        "--dataset_as_col_num_panel",
        type=int,
        default=1,
        help="Number of vertically stacked panels for --dataset_as_col (default: 1). "
        "When >1, datasets are split across panels to reduce figure width.",
    )
    parser.add_argument(
        "--dataset_as_row",
        action="store_true",
        help="If set, each row = one dataset; columns = samples within that dataset",
    )
    parser.add_argument(
        "--dataset_as_row_num_panel",
        type=int,
        default=1,
        help="Number of horizontally arranged panels for --dataset_as_row (default: 1). "
        "When >1, datasets are split across panels to reduce figure height.",
    )
    parser.add_argument(
        "--dataset_as_row_num_row_per_ds",
        type=int,
        default=1,
        help="Number of rows per dataset for --dataset_as_row (default: 1). "
        "When >1, samples within each dataset wrap across multiple rows.",
    )
    args = parser.parse_args()

    if args.dataset_as_col and args.dataset_as_row:
        raise ValueError(
            "--dataset_as_col and --dataset_as_row are mutually exclusive."
        )

    base_dir = Path(args.dir_subfigures)
    models = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not models:
        raise ValueError(f"No model subdirectories found in: {base_dir}")

    if args.dir_model is not None:
        target = Path(args.dir_model).name
        models = [m for m in models if m.name == target]
        if not models:
            raise ValueError(f"--dir_model '{args.dir_model}' not found in {base_dir}")

    show_model_label = args.dir_model is None
    if args.dataset_as_col or args.dataset_as_row:
        import warnings

        flag = "--dataset_as_col" if args.dataset_as_col else "--dataset_as_row"
        warnings.warn(
            f"--row_per_model is ignored when {flag} is set. "
            "Rows per model are determined automatically as ceil(limit_subfigures / num_datasets).",
            UserWarning,
            stacklevel=2,
        )

    seed = args.seed if args.seed is not None else SEED
    rng = random.Random(seed)
    samples = _select_samples(models, args.limit_subfigures, rng)

    if args.dataset_as_col:
        _compile_figure_dataset_as_col(
            models,
            samples,
            args.output,
            show_model_label=show_model_label,
            num_panel=args.dataset_as_col_num_panel,
        )
    elif args.dataset_as_row:
        _compile_figure_dataset_as_row(
            models,
            samples,
            args.output,
            show_model_label=show_model_label,
            num_panel=args.dataset_as_row_num_panel,
            num_row_per_ds=args.dataset_as_row_num_row_per_ds,
        )
    else:
        _compile_figure(
            models,
            samples,
            args.row_per_model,
            args.output,
            show_model_label=show_model_label,
        )


if __name__ == "__main__":
    main()
