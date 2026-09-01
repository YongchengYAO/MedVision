"""Combine PNG/PDF figures into one grid composite, vector where possible.

One layout pass feeds two renderers, so the engines can never drift apart:

* ``render_vector`` places single-page PDFs onto one output page with a
  scale/offset transform -- text and line art stay **vector**, nothing is
  rasterized.
* ``render_raster`` draws the same boxes with matplotlib ``imshow``, pulling PDF
  panels through poppler on the way in.

Format rules
------------
A ``.pdf`` output requires **every** input to be a PDF: a PNG panel is already
pixels and cannot be turned back into vector art. A ``.png`` output accepts any
mix. So the output suffix alone picks the engine.

Layout
------
Panels fill an ``nrow`` x ``ncol`` grid in row-major order and are labelled
(a), (b), (c), ... unless ``show_labels`` is off.

* ``base_size`` is the full-scale panel width (``fit="width"``) or height
  (``fit="height"``, meant for single-row strips), in inches.
* ``subplot_scales`` multiplies each panel's size along that fitted axis, aspect
  preserved: 0.5 is half, 2.0 is double. Columns take the width of their widest
  panel and rows the height of their tallest, so a shrunken panel leaves
  whitespace rather than pulling its neighbours in.
* ``shifts_x``/``shifts_y`` nudge a panel within its slot as a fraction of the
  slot. Labels stay at the unshifted slot corner.

The background is transparent unless ``white_bg`` is set.

Run ``python figure_concat.py --help`` for the CLI.
"""

import argparse
import io
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from medvision_bm.utils.plot_utils import save_fig_capped

PT_PER_IN = 72.0
SUPPORTED = {".png", ".pdf"}


def wants_vector(paths, output):
    """Validate the format mix and report whether the job stays vector.

    Raises ValueError on an unsupported suffix, or on a PDF output fed by any
    PNG panel -- that conversion cannot preserve vector art.
    """
    suffixes = [Path(p).suffix.lower() for p in paths]
    out = Path(output).suffix.lower()

    for path, suffix in zip([*paths, output], [*suffixes, out]):
        if suffix not in SUPPORTED:
            raise ValueError(f"unsupported format {suffix!r}: {path}")

    if out == ".pdf":
        raster = [p for p, s in zip(paths, suffixes) if s != ".pdf"]
        if raster:
            raise ValueError(
                "a PDF output needs every input to be a PDF, but these are not: "
                + ", ".join(str(p) for p in raster)
                + "\nWrite a .png output instead to combine mixed formats."
            )
    return out == ".pdf"


def resolve_grid(n, nrow=None, ncol=None):
    """Fill in whichever of nrow/ncol was left out; default to a single column."""
    if nrow is None and ncol is None:
        nrow, ncol = n, 1
    elif ncol is None:
        ncol = math.ceil(n / nrow)
    elif nrow is None:
        nrow = math.ceil(n / ncol)
    if nrow * ncol < n:
        raise ValueError(f"grid {nrow}x{ncol} has no room for {n} panels")
    return nrow, ncol


def panel_sizes(paths):
    """Native (width, height) per panel, without rasterizing anything.

    Only the aspect ratio is used downstream, so mixing points (PDF) and pixels
    (PNG) across panels is fine.
    """
    sizes = []
    for path in paths:
        if Path(path).suffix.lower() == ".pdf":
            from pypdf import PdfReader

            # cropbox is what a viewer shows, and falls back to mediabox when unset.
            box = PdfReader(str(path)).pages[0].cropbox
            sizes.append((float(box.width), float(box.height)))
        else:
            from PIL import Image

            with Image.open(path) as im:
                sizes.append((float(im.width), float(im.height)))
    return sizes


def grid_layout(
    sizes,
    *,
    nrow,
    ncol,
    base_size,
    subplot_scales,
    shifts_x,
    shifts_y,
    fit="width",
    hspace=0.02,
    wspace=0.02,
    with_labels=True,
):
    """Lay panels out in a top-left-origin inch space.

    Returns ``(panels, slots, page_w, page_h)``, each box a ``(left, top, w, h)``
    tuple already normalized so the page starts at (0, 0).
    """
    drawn = []
    for (w, h), scale in zip(sizes, subplot_scales):
        if fit == "width":
            dw = base_size * scale
            dh = dw * h / w
        else:  # fit == "height"
            dh = base_size * scale
            dw = dh * w / h
        drawn.append((dw, dh))

    # A column is as wide as its widest panel; a row as tall as its tallest.
    col_w = [
        max((drawn[i][0] for i in range(len(sizes)) if i % ncol == c), default=0.0)
        for c in range(ncol)
    ]
    row_h = [
        max((drawn[i][1] for i in range(len(sizes)) if i // ncol == r), default=0.0)
        for r in range(nrow)
    ]
    gap_x = wspace * (sum(col_w) / len(col_w))
    gap_y = hspace * (sum(row_h) / len(row_h))

    panels, slots = [], []
    for i, ((dw, dh), shift_x, shift_y) in enumerate(zip(drawn, shifts_x, shifts_y)):
        r, c = divmod(i, ncol)
        sl = sum(col_w[:c]) + c * gap_x
        st = sum(row_h[:r]) + r * gap_y
        sw, sh = col_w[c], row_h[r]
        slots.append((sl, st, sw, sh))
        # The panel sits flush to the slot's leading edge along the fitted axis
        # (leftover whitespace goes right/down) and is centred on the other one.
        pad_x = 0.0 if fit == "width" else (sw - dw) / 2
        pad_y = (sh - dh) / 2 if fit == "width" else 0.0
        panels.append((sl + pad_x + shift_x * sw, st + pad_y + shift_y * sh, dw, dh))

    # Tight page: union of the drawn panels, widened to keep every label anchor in.
    # With labels off there is nothing to reserve, so the page hugs the panels.
    corners = [(l, t, l + w, t + h) for l, t, w, h in panels]
    if with_labels:
        corners += [
            (sl + 0.01 * sw, st + 0.01 * sh, sl + 0.01 * sw, st + 0.01 * sh)
            for sl, st, sw, sh in slots
        ]
    min_x = min(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    page_w = max(c[2] for c in corners) - min_x
    page_h = max(c[3] for c in corners) - min_y

    def rebase(boxes):
        return [(l - min_x, t - min_y, w, h) for l, t, w, h in boxes]

    return rebase(panels), rebase(slots), page_w, page_h


def read_panel(path, px_width):
    """Read a PNG or PDF panel as an image array.

    PDFs are rasterized with poppler's ``pdftocairo`` at exactly ``px_width``
    pixels across -- the width the panel occupies in the rendered composite --
    so the panel is neither upsampled (blurry) nor needlessly oversized.
    ``-transp`` keeps the transparent background the source figures are saved with.
    """
    if Path(path).suffix.lower() != ".pdf":
        return mpimg.imread(path)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftocairo", "-png", "-transp", "-singlefile",
             "-scale-to-x", str(px_width), "-scale-to-y", "-1",
             str(path), str(prefix)],
            check=True,
        )
        return mpimg.imread(prefix.with_suffix(".png"))


def _mpl_page(fig, transparent):
    """Render a matplotlib figure to a PDF page object of exactly its figsize.

    Goes through a buffer rather than a temp file so the page stays readable
    after this returns. No bbox_inches="tight" -- the page size must not shift.
    """
    from pypdf import PdfReader

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", transparent=transparent)
    plt.close(fig)
    buf.seek(0)
    return PdfReader(buf).pages[0]


def _label_anchors(slots, page_w, page_h):
    """Figure-fraction (x, y) for each label, 1% inside its slot's top-left."""
    return [
        ((sl + 0.01 * sw) / page_w, 1 - (st + 0.01 * sh) / page_h)
        for sl, st, sw, sh in slots
    ]


def render_vector(
    paths, panels, slots, page_w, page_h, output, labels, label_fontsize, white_bg
):
    """Compose single-page PDFs onto one page without rasterizing anything."""
    from pypdf import PdfReader, PdfWriter, Transformation

    writer = PdfWriter()
    out_page = writer.add_blank_page(
        width=page_w * PT_PER_IN, height=page_h * PT_PER_IN
    )

    if white_bg:
        # A blank PDF page paints nothing, so the white has to be merged in as
        # real content -- and first, so every panel lands on top of it.
        out_page.merge_page(_mpl_page(plt.figure(figsize=(page_w, page_h)), False))

    for path, (left, top, w, h) in zip(paths, panels):
        src = PdfReader(str(path)).pages[0]
        x0, y0 = float(src.cropbox.left), float(src.cropbox.bottom)
        factor = w * PT_PER_IN / float(src.cropbox.width)
        out_page.merge_transformed_page(
            src,
            Transformation()
            .translate(-x0, -y0)  # normalize a non-zero source origin
            .scale(factor)
            .translate(left * PT_PER_IN, (page_h - top - h) * PT_PER_IN),  # PDF y runs up
        )

    if labels:
        # A same-size overlay page keeps the labels as real text, not pixels.
        fig = plt.figure(figsize=(page_w, page_h))
        for (x, y), label in zip(_label_anchors(slots, page_w, page_h), labels):
            fig.text(x, y, label, fontsize=label_fontsize, fontweight="bold",
                     va="top", ha="left")
        out_page.merge_page(_mpl_page(fig, True))

    # merge_transformed_page leaves the combined content stream uncompressed and
    # strands each source page's original stream as an orphan; without this the
    # output balloons ~40x (47 MB for a 1.2 MB set of panels).
    out_page.compress_content_streams()
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)

    with open(output, "wb") as fh:
        writer.write(fh)


def render_raster(
    paths, panels, slots, page_w, page_h, output, labels, label_fontsize, dpi, white_bg
):
    """Draw the same boxes with matplotlib, rasterizing PDF panels on read."""
    fig = plt.figure(figsize=(page_w, page_h))

    for path, (left, top, w, h) in zip(paths, panels):
        im = read_panel(path, round(w * dpi))
        # Axes are positioned explicitly, so the layout matches render_vector exactly.
        ax = fig.add_axes([left / page_w, 1 - (top + h) / page_h, w / page_w, h / page_h])
        ax.imshow(im)
        ax.axis("off")

    if labels:
        for (x, y), label in zip(_label_anchors(slots, page_w, page_h), labels):
            fig.text(x, y, label, fontsize=label_fontsize, fontweight="bold",
                     va="top", ha="left")

    # No bbox_inches="tight" -- the page is already exactly the size we want.
    save_fig_capped(output, fig=fig, dpi=dpi, transparent=not white_bg)
    plt.close(fig)


def combine(
    paths,
    output,
    *,
    nrow=None,
    ncol=None,
    base_size=8.0,
    subplot_scales=None,
    shifts_x=None,
    shifts_y=None,
    fit="width",
    dpi=300,
    label_fontsize=18,
    white_bg=False,
    show_labels=True,
):
    """Combine PNG/PDF panels into one composite, vector where possible.

    Returns the engine used, ``"vector"`` or ``"raster"``.
    """
    paths = [Path(p) for p in paths]
    output = Path(output)
    n = len(paths)

    def fill(values, default):
        values = list(values) if values else [default] * n
        if len(values) != n:
            raise ValueError(f"expected {n} values to match the panels, got {len(values)}")
        return values

    vector = wants_vector(paths, output)
    nrow, ncol = resolve_grid(n, nrow, ncol)
    panels, slots, page_w, page_h = grid_layout(
        panel_sizes(paths),
        nrow=nrow,
        ncol=ncol,
        base_size=base_size,
        subplot_scales=fill(subplot_scales, 1.0),
        shifts_x=fill(shifts_x, 0.0),
        shifts_y=fill(shifts_y, 0.0),
        fit=fit,
        with_labels=show_labels,
    )
    labels = [f"({chr(ord('a') + i)})" for i in range(n)] if show_labels else None

    args = (paths, panels, slots, page_w, page_h, output, labels, label_fontsize)
    if vector:
        render_vector(*args, white_bg)
        return "vector"
    render_raster(*args, dpi, white_bg)
    return "raster"


EXAMPLES = """
examples:
  # 3x1 column
  figure_concat.py --images a.pdf b.pdf c.pdf --output out.pdf \\
      --base_size 8 --subplot_scales 1.0 0.75 0.95 --shifts_x 0 0.02 0

  # 1x3 row, third panel at 50% height
  figure_concat.py --images a.pdf b.pdf c.pdf --output out.pdf \\
      --nrow 1 --fit height --base_size 6 --subplot_scales 1 1 0.5

  # 2x2 grid, mixed inputs, raster output
  figure_concat.py --images a.pdf b.png c.pdf d.png --output out.png --ncol 2
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine PNG/PDF figures into a grid composite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="Panel paths in row-major order (PNG or PDF)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--nrow", type=int, default=None, help="Grid rows (default: one per panel)"
    )
    parser.add_argument(
        "--ncol", type=int, default=None, help="Grid columns (default: 1)"
    )
    parser.add_argument(
        "--fit",
        choices=("width", "height"),
        default="width",
        help="Whether base_size is each panel's width or its height",
    )
    parser.add_argument(
        "--base_size",
        type=float,
        default=8.0,
        help="Full-scale panel width (fit=width) or height (fit=height), in inches",
    )
    parser.add_argument(
        "--subplot_scales",
        type=float,
        nargs="+",
        default=None,
        help="Per-panel size multiplier along the fitted axis, aspect preserved: "
        "0.5 is half, 2.0 is double (default: all 1.0)",
    )
    parser.add_argument(
        "--shifts_x",
        type=float,
        nargs="+",
        default=None,
        help="Per-panel rightward shift as a fraction of the slot (default: all 0)",
    )
    parser.add_argument(
        "--shifts_y",
        type=float,
        nargs="+",
        default=None,
        help="Per-panel downward shift as a fraction of the slot (default: all 0)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Hide the (a)/(b)/(c) panel labels (shown by default)",
    )
    parser.add_argument(
        "--white_bg",
        action="store_true",
        help="Paint a white background (default: transparent)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output only")
    args = parser.parse_args()

    engine = combine(
        args.images,
        args.output,
        nrow=args.nrow,
        ncol=args.ncol,
        base_size=args.base_size,
        subplot_scales=args.subplot_scales,
        shifts_x=args.shifts_x,
        shifts_y=args.shifts_y,
        fit=args.fit,
        dpi=args.dpi,
        white_bg=args.white_bg,
        show_labels=not args.no_labels,
    )
    print(f"Saved ({engine}): {args.output}")
