"""
Assemble a cross-model comparison grid from pre-generated per-sample subfigures.

Reads a base directory whose immediate subfolders are model output directories
(one per model), each containing per-dataset subfolders of per-sample subfigures.
Those subfigures are produced by the per-task visualizers in this folder:
viz_tl_axes.py (TL), viz_ad_landmarks.py (AD), viz_detection_boxes.py (Detection).
A fixed random subset (seeded from configs.py SEED, overridable via --seed) is
sampled equally across the datasets common to all models, then laid out into a
single comparison figure.

Input/output formats (png, svg, pdf) are independent and controlled by
--input_format (single) and --output_format (one or more), both defaulting to pdf.
The grid is composited with PyMuPDF: vector inputs (pdf/svg) are placed as true
vector content (show_pdf_page), so vector -> vector output stays vector; png inputs
are embedded as raster, and png output rasterizes the final page. svg inputs are
converted to pdf via cairosvg before placement. Because vector pdf/svg output is not
pixel-capped, --pdf_image_dpi optionally downsamples the pdf's embedded raster images
via Ghostscript (overlays stay vector) to bound file size.

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
        --output <output_path> \
        [--input_format {png,svg,pdf}] \
        [--output_format {png,svg,pdf} [{png,svg,pdf} ...]] \
        [--pdf_image_dpi N] \
        [--row_per_model N] \
        [--dir_model <model_folder_name>] \
        [--dataset_order A,B,C] \
        [--seed N] \
        [--dataset_as_col [--dataset_as_col_num_panel N]] \
        [--dataset_as_row [--dataset_as_row_num_panel N] [--dataset_as_row_num_row_per_ds N]]
"""

import argparse
import math
import random
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from medvision_bm.utils.plot_utils import FIG_DPI, MAX_FIG_MP

from medvision_bm.utils.configs import SEED

PT_PER_IN = 72.0  # PDF points per inch
_LABEL_FONT = fitz.Font("hebo")  # Helvetica-Bold, matches the old fontweight="bold"

MODEL_NAME_MAP = {
    "MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250": "fullRFT",
    "MedVision__fullSFT__Qwen2.5VL-7B__D110k-AD5k-TL5k__CoT__512x512__v2": "fullSFT",
}


def _select_samples(models, limit, input_format, rng, dataset_order=None):
    """Return ordered list of (dataset, filename) with equal samples per dataset.

    Each dataset receives ceil(limit / n_datasets) samples, so total may exceed
    limit by up to n_datasets-1 when limit is not divisible by n_datasets.
    Samples from the same dataset are consecutive in the returned list.

    ``dataset_order`` optionally overrides the default alphabetical dataset order;
    listed datasets come first in the given order, any remaining ones follow
    alphabetically. ``None`` keeps the alphabetical order unchanged.
    """
    datasets_per_model = {
        m.name: sorted([d.name for d in m.iterdir() if d.is_dir()]) for m in models
    }
    common_datasets = sorted(
        set.intersection(*[set(v) for v in datasets_per_model.values()])
    )
    if dataset_order:
        first = [d for d in dataset_order if d in common_datasets]
        common_datasets = first + [d for d in common_datasets if d not in first]
    n_datasets = len(common_datasets)
    if limit < n_datasets:
        raise ValueError(
            f"--limit_subfigures ({limit}) is less than the number of datasets ({n_datasets}). "
            f"Datasets: {common_datasets}"
        )

    common_files = {}
    for dataset in common_datasets:
        files_sets = [
            set(f.name for f in (m / dataset).glob(f"*.{input_format}")) for m in models
        ]
        common_files[dataset] = sorted(set.intersection(*files_sets))
        if not common_files[dataset]:
            raise ValueError(
                f"No common '.{input_format}' files found across all models for dataset '{dataset}'"
            )

    per_dataset = math.ceil(limit / n_datasets)
    selected_by_dataset = {}
    for dataset in common_datasets:
        pool = list(common_files[dataset])
        rng.shuffle(pool)
        selected_by_dataset[dataset] = pool[:per_dataset]

    return [(ds, f) for ds in common_datasets for f in selected_by_dataset[ds]]


# ── PyMuPDF compositing primitives ───────────────────────────────────────────


def _rect_pts(bbox, w_pt, h_pt):
    """Convert a matplotlib figure-fraction bbox (y bottom-up) to a fitz.Rect in
    points on a page of size (w_pt, h_pt) (y top-down)."""
    x0, y0, x1, y1 = (
        (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if hasattr(bbox, "x0") else bbox
    )
    return fitz.Rect(x0 * w_pt, (1 - y1) * h_pt, x1 * w_pt, (1 - y0) * h_pt)


def _open_src_as_pdf(path, input_format):
    """Return a 1-page fitz Document for a vector subfigure (pdf directly, svg via cairosvg)."""
    if input_format == "pdf":
        return fitz.open(str(path))
    import cairosvg  # only needed for svg inputs

    return fitz.open("pdf", cairosvg.svg2pdf(url=str(path)))


def _place_subfig(page, rect, path, input_format, subdocs):
    """Place one subfigure into rect. Vector inputs are embedded as vector via
    show_pdf_page; png inputs are embedded as raster. keep_proportion=False
    stretches to fill the cell, matching the old imshow(aspect="auto")."""
    if input_format == "png":
        page.insert_image(rect, filename=str(path), keep_proportion=False)
    else:
        src = _open_src_as_pdf(path, input_format)
        subdocs.append(src)
        page.show_pdf_page(rect, src, 0, keep_proportion=False)


def _draw_text(page, rect, text, fontsize, rotate=0):
    """Draw bold text centered in rect, optionally rotated (90 = CCW, bottom-to-top)."""
    cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
    tl = _LABEL_FONT.text_length(text, fontsize=fontsize)
    tw = fitz.TextWriter(page.rect)
    tw.append((cx - tl / 2, cy + fontsize * 0.35), text, font=_LABEL_FONT, fontsize=fontsize)
    if rotate:
        tw.write_text(page, morph=(fitz.Point(cx, cy), fitz.Matrix(rotate)))
    else:
        tw.write_text(page)


def _model_label_elements(models, anchor_positions, show_model_label, fontsize=11):
    """Build rotated model-name text placements and inter-model separator y-fractions
    from per-model anchor bboxes (leftmost cell of each row). Returns
    (text_places, sep_fracs) where text_places = [(bbox, text, fontsize, rotate), ...]."""
    text_places, sep_fracs = [], []
    n_models = len(models)
    for i, model in enumerate(models):
        first = anchor_positions[i][0]
        last = anchor_positions[i][-1]
        if show_model_label:
            name = MODEL_NAME_MAP.get(model.name, model.name)
            bbox = (first.x0, last.y0, first.x1, first.y1)
            text_places.append((bbox, name, fontsize, 90))
        if i < n_models - 1:
            nxt = anchor_positions[i + 1][0]
            sep_fracs.append((last.y0 + nxt.y1) / 2)
    return text_places, sep_fracs


def _build_page(w_in, h_in, img_places, text_places, sep_fracs, input_format):
    """Render a single grid page into a new 1-page fitz Document.

    img_places  = [(bbox, src_path), ...]
    text_places = [(bbox, text, fontsize, rotate), ...]
    sep_fracs   = [y_fraction, ...]   (full-width horizontal separators)

    Returns (doc, subdocs) where subdocs are the per-cell source Documents that
    must stay open until the page is saved/rasterized.
    """
    w_pt, h_pt = w_in * PT_PER_IN, h_in * PT_PER_IN
    doc = fitz.open()
    page = doc.new_page(width=w_pt, height=h_pt)
    subdocs = []
    for bbox, path in img_places:
        _place_subfig(page, _rect_pts(bbox, w_pt, h_pt), path, input_format, subdocs)
    for bbox, text, fontsize, rotate in text_places:
        _draw_text(page, _rect_pts(bbox, w_pt, h_pt), text, fontsize, rotate)
    for y_frac in sep_fracs:
        y = (1 - y_frac) * h_pt
        page.draw_line(
            fitz.Point(0, y), fitz.Point(w_pt, y), color=(0, 0, 0), width=1.5
        )
    return doc, subdocs


def _compose_panels(panel_docs, horizontal):
    """Stitch multiple 1-page panel Documents into one page, preserving vectors.

    horizontal=True -> side by side (top-aligned); False -> stacked (left-aligned).
    """
    sizes = [(d[0].rect.width, d[0].rect.height) for d in panel_docs]
    if horizontal:
        w_pt, h_pt = sum(w for w, _ in sizes), max(h for _, h in sizes)
    else:
        w_pt, h_pt = max(w for w, _ in sizes), sum(h for _, h in sizes)

    out = fitz.open()
    page = out.new_page(width=w_pt, height=h_pt)
    x = y = 0
    for d, (w, h) in zip(panel_docs, sizes):
        page.show_pdf_page(fitz.Rect(x, y, x + w, y + h), d, 0)
        if horizontal:
            x += w
        else:
            y += h
    return out


def _downsample_pdf_images(pdf_path, dpi):
    """Downsample embedded raster images in pdf_path (in place) to `dpi` via Ghostscript,
    leaving vector paths and text untouched (only image XObjects are re-encoded). This
    bounds the file size of a vector-composited grid whose bulk is the per-cell raster
    medical-image, without rasterizing the overlays. Requires `gs` on PATH; Flate keeps
    the downsample lossless."""
    import shutil
    import subprocess

    gs = shutil.which("gs")
    if gs is None:
        raise RuntimeError(
            "--pdf_image_dpi requires Ghostscript ('gs') on PATH, but it was not found."
        )
    pdf_path = Path(pdf_path)
    tmp = pdf_path.with_name(pdf_path.stem + ".gs_tmp.pdf")
    cmd = [
        gs,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        # pdfwrite defaults to auto-rotating pages by text orientation; our rotated
        # model/dataset labels would otherwise flip the whole grid 90 degrees.
        "-dAutoRotatePages=/None",
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={dpi}",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dGrayImageResolution={dpi}",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/FlateEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/FlateEncode",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-sOutputFile={tmp}",
        str(pdf_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Ghostscript image downsampling failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    tmp.replace(pdf_path)


def _export(doc, out_path, output_formats, pdf_image_dpi=None):
    """Write the single-page doc to each requested format, swapping out_path's suffix.
    png rasterizes at up to FIG_DPI, clamped so the pixel count stays within MAX_FIG_MP.
    pdf_image_dpi (>0) downsamples the pdf's embedded rasters via Ghostscript (pdf only)."""
    page = doc[0]
    p = Path(out_path)
    for fmt in output_formats:
        target = p.with_suffix(f".{fmt}")
        if fmt == "pdf":
            doc.save(str(target))
            if pdf_image_dpi and pdf_image_dpi > 0:
                _downsample_pdf_images(target, pdf_image_dpi)
        elif fmt == "svg":
            with open(target, "w") as f:
                f.write(page.get_svg_image())
        elif fmt == "png":
            w_in = page.rect.width / PT_PER_IN
            h_in = page.rect.height / PT_PER_IN
            max_dpi = (MAX_FIG_MP * 1e6 / (w_in * h_in)) ** 0.5
            dpi = int(min(FIG_DPI, max_dpi))
            page.get_pixmap(dpi=dpi, alpha=True).save(str(target))
        else:
            raise ValueError(f"Unsupported output format: {fmt}")
        print(f"Saved: {target}")


def _close_all(*docs):
    for d in docs:
        try:
            d.close()
        except Exception:
            pass


def _group_by_dataset(samples):
    """Return (samples_by_dataset, dataset_order) preserving first-seen order."""
    samples_by_dataset, dataset_order = {}, []
    for ds, f in samples:
        if ds not in samples_by_dataset:
            samples_by_dataset[ds] = []
            dataset_order.append(ds)
        samples_by_dataset[ds].append(f)
    return samples_by_dataset, dataset_order


# ── Layout modes ─────────────────────────────────────────────────────────────


def _compile_figure(
    models, samples, row_per_model, output, output_formats, show_model_label=True,
    input_format="pdf", pdf_image_dpi=None, model_label_fontsize=11,
    hide_model_separator=False
):
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

    img_places = []
    # anchor_positions[model_idx][row_in_model] = leftmost cell bbox of that row
    anchor_positions = []
    for model_idx, model in enumerate(models):
        model_anchors = []
        for row_in_model in range(row_per_model):
            row_idx = model_idx * row_per_model + row_in_model

            if show_model_label:
                model_anchors.append(gs[row_idx, 0].get_position(fig))

            for col_idx in range(n_cols_img):
                sample_idx = row_in_model * n_cols_img + col_idx
                pos = gs[row_idx, col_idx + col_offset].get_position(fig)
                if not show_model_label and col_idx == 0:
                    model_anchors.append(pos)
                if sample_idx < len(samples):
                    dataset, filename = samples[sample_idx]
                    img_places.append((pos, model / dataset / filename))

        anchor_positions.append(model_anchors)

    texts, seps = _model_label_elements(
        models, anchor_positions, show_model_label, model_label_fontsize
    )
    if hide_model_separator:
        # The rotated block labels already mark where one model block ends and the next begins,
        # so the rules are redundant when the labels are large enough to read.
        seps = []

    doc, subdocs = _build_page(fig_w, fig_h, img_places, texts, seps, input_format)
    plt.close(fig)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _export(doc, out_path, output_formats, pdf_image_dpi)
    _close_all(doc, *subdocs)


def _make_dataset_as_col_panel(
    models, ds_group, samples_by_dataset, rows_per_model, panel_size, show_model_label, input_format
):
    """Render one dataset-as-col panel into a 1-page fitz Document. Returns (doc, subdocs)."""
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

    text_places = []
    for ds_idx, dataset in enumerate(ds_group):
        text_places.append((gs[0, ds_idx + col_offset].get_position(fig), dataset, 16, 0))

    img_places = []
    anchor_positions = []
    for model_idx, model in enumerate(models):
        model_anchors = []
        for row_in_model in range(rows_per_model):
            row_idx = 1 + model_idx * rows_per_model + row_in_model

            if show_model_label:
                model_anchors.append(gs[row_idx, 0].get_position(fig))

            for ds_idx, dataset in enumerate(ds_group):
                pos = gs[row_idx, ds_idx + col_offset].get_position(fig)
                if not show_model_label and ds_idx == 0:
                    model_anchors.append(pos)
                ds_files = samples_by_dataset[dataset]
                if row_in_model < len(ds_files):
                    img_places.append((pos, model / dataset / ds_files[row_in_model]))

        anchor_positions.append(model_anchors)

    labels, seps = _model_label_elements(models, anchor_positions, show_model_label)
    doc, subdocs = _build_page(
        fig_w, fig_h, img_places, text_places + labels, seps, input_format
    )
    plt.close(fig)
    return doc, subdocs


def _compile_figure_dataset_as_col(
    models, samples, output, output_formats, show_model_label=True, num_panel=1,
    input_format="pdf", pdf_image_dpi=None
):
    """Dataset-as-columns layout: each column = one dataset; rows = samples within that dataset.

    If num_panel > 1, datasets are split into vertically stacked panels to reduce figure width.
    Each panel is rendered as its own fitz page and stitched vector-preserving with show_pdf_page.
    """
    samples_by_dataset, dataset_order = _group_by_dataset(samples)

    rows_per_model = max(len(v) for v in samples_by_dataset.values())
    panel_size = math.ceil(len(dataset_order) / num_panel)
    dataset_groups = [
        dataset_order[i * panel_size : (i + 1) * panel_size] for i in range(num_panel)
    ]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_docs, all_subdocs = [], []
    for ds_group in dataset_groups:
        if not ds_group:
            continue
        d, sub = _make_dataset_as_col_panel(
            models,
            ds_group,
            samples_by_dataset,
            rows_per_model,
            panel_size,
            show_model_label,
            input_format,
        )
        panel_docs.append(d)
        all_subdocs += sub

    if len(panel_docs) == 1:
        final, composed = panel_docs[0], None
    else:
        final = _compose_panels(panel_docs, horizontal=False)
        composed = final

    _export(final, out_path, output_formats, pdf_image_dpi)
    _close_all(*panel_docs, *all_subdocs, *([composed] if composed else []))


def _make_dataset_as_row_panel(
    models,
    ds_group,
    samples_by_dataset,
    cols_per_dataset,
    panel_size,
    show_model_label,
    input_format,
    num_row_per_ds=1,
):
    """Render one dataset-as-row panel into a 1-page fitz Document. Returns (doc, subdocs)."""
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

    img_places = []
    text_places = []
    anchor_positions = []
    for model_idx, model in enumerate(models):
        model_anchors = []
        for ds_idx, dataset in enumerate(ds_group):
            base_row = model_idx * panel_size * num_row_per_ds + ds_idx * num_row_per_ds

            if show_model_label:
                model_anchors.append(
                    gs[base_row : base_row + num_row_per_ds, 0].get_position(fig)
                )

            ds_lbl_pos = gs[
                base_row : base_row + num_row_per_ds, ds_label_col_idx
            ].get_position(fig)
            text_places.append((ds_lbl_pos, dataset, 16, 90))
            if not show_model_label:
                model_anchors.append(ds_lbl_pos)

            ds_files = samples_by_dataset[dataset]
            for row_in_ds in range(num_row_per_ds):
                row_idx = base_row + row_in_ds
                for col_idx in range(cols_per_dataset):
                    sample_idx = row_in_ds * cols_per_dataset + col_idx
                    pos = gs[row_idx, img_col_offset + col_idx].get_position(fig)
                    if sample_idx < len(ds_files):
                        img_places.append(
                            (pos, model / dataset / ds_files[sample_idx])
                        )

        anchor_positions.append(model_anchors)

    labels, seps = _model_label_elements(models, anchor_positions, show_model_label)
    doc, subdocs = _build_page(
        fig_w, fig_h, img_places, text_places + labels, seps, input_format
    )
    plt.close(fig)
    return doc, subdocs


def _compile_figure_dataset_as_row(
    models,
    samples,
    output,
    output_formats,
    show_model_label=True,
    num_panel=1,
    num_row_per_ds=1,
    input_format="pdf",
    pdf_image_dpi=None,
):
    """Dataset-as-rows layout: each row = one dataset; columns = samples within that dataset.

    If num_panel > 1, datasets are split into horizontally arranged side-by-side panels
    to reduce figure height. Each panel is rendered as its own fitz page and stitched
    vector-preserving with show_pdf_page.
    """
    samples_by_dataset, dataset_order = _group_by_dataset(samples)

    cols_per_dataset = math.ceil(
        max(len(v) for v in samples_by_dataset.values()) / num_row_per_ds
    )
    panel_size = math.ceil(len(dataset_order) / num_panel)
    dataset_groups = [
        dataset_order[i * panel_size : (i + 1) * panel_size] for i in range(num_panel)
    ]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel_docs, all_subdocs = [], []
    for ds_group in dataset_groups:
        if not ds_group:
            continue
        d, sub = _make_dataset_as_row_panel(
            models,
            ds_group,
            samples_by_dataset,
            cols_per_dataset,
            panel_size,
            show_model_label,
            input_format,
            num_row_per_ds=num_row_per_ds,
        )
        panel_docs.append(d)
        all_subdocs += sub

    if len(panel_docs) == 1:
        final, composed = panel_docs[0], None
    else:
        final = _compose_panels(panel_docs, horizontal=True)
        composed = final

    _export(final, out_path, output_formats, pdf_image_dpi)
    _close_all(*panel_docs, *all_subdocs, *([composed] if composed else []))


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
        help="Output file path; its extension is replaced per --output_format",
    )
    parser.add_argument(
        "--input_format",
        choices=["png", "svg", "pdf"],
        default="pdf",
        help="Format of the input subfigures to read (default: pdf)",
    )
    parser.add_argument(
        "--output_format",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["pdf"],
        help="One or more output formats to write (default: pdf). "
        "pdf/svg keep vector inputs as vectors; png rasterizes the final page.",
    )
    parser.add_argument(
        "--pdf_image_dpi",
        type=int,
        default=None,
        help="For pdf output only: downsample the embedded raster images to this dpi via "
        "Ghostscript, keeping vector overlays sharp, to bound file size. Omit or <=0 to "
        "disable (full-resolution vector). Requires 'gs' on PATH. ~120-150 is a good "
        "balance; going much lower can be counterproductive.",
    )
    parser.add_argument(
        "--dir_model",
        default=None,
        help="If set, only plot rows for this model folder (path or folder name)",
    )
    parser.add_argument(
        "--dataset_order",
        default=None,
        help="Comma-separated dataset (subfigure folder) names fixing row/column order; "
        "listed ones come first in this order, the rest follow alphabetically. "
        "Omit to keep the default alphabetical order.",
    )
    parser.add_argument(
        "--model_label_fontsize",
        type=float,
        default=11,
        help="Point size of the rotated model row-block labels (default 11). Default layout "
        "mode only; --dataset_as_col/--dataset_as_row keep 11.",
    )
    parser.add_argument(
        "--hide_model_separator",
        action="store_true",
        help="Omit the horizontal rules drawn between model row-blocks. Default layout mode "
        "only; --dataset_as_col/--dataset_as_row keep theirs.",
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
    dataset_order = (
        [d.strip() for d in args.dataset_order.split(",") if d.strip()]
        if args.dataset_order
        else None
    )
    samples = _select_samples(
        models, args.limit_subfigures, args.input_format, rng, dataset_order
    )

    if args.pdf_image_dpi and args.pdf_image_dpi > 0 and "pdf" not in args.output_format:
        import warnings

        warnings.warn(
            "--pdf_image_dpi only affects pdf output, but 'pdf' is not in --output_format; "
            "it will have no effect.",
            UserWarning,
            stacklevel=2,
        )

    if args.dataset_as_col:
        _compile_figure_dataset_as_col(
            models,
            samples,
            args.output,
            args.output_format,
            show_model_label=show_model_label,
            num_panel=args.dataset_as_col_num_panel,
            input_format=args.input_format,
            pdf_image_dpi=args.pdf_image_dpi,
        )
    elif args.dataset_as_row:
        _compile_figure_dataset_as_row(
            models,
            samples,
            args.output,
            args.output_format,
            show_model_label=show_model_label,
            num_panel=args.dataset_as_row_num_panel,
            num_row_per_ds=args.dataset_as_row_num_row_per_ds,
            input_format=args.input_format,
            pdf_image_dpi=args.pdf_image_dpi,
        )
    else:
        _compile_figure(
            models,
            samples,
            args.row_per_model,
            args.output,
            args.output_format,
            show_model_label=show_model_label,
            input_format=args.input_format,
            pdf_image_dpi=args.pdf_image_dpi,
            model_label_fontsize=args.model_label_fontsize,
            hide_model_separator=args.hide_model_separator,
        )


if __name__ == "__main__":
    main()
