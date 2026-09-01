#!/usr/bin/env bash
# Sample usages of figure_concat.py: combine PNG/PDF panels into one m x n composite.
# The output suffix picks the engine -- a .pdf output keeps every panel vector (so every
# input must be a PDF); a .png output rasterizes and accepts any mix of PNG/PDF.
#
# Every example runs on four generated demo panels with different aspect ratios, so the
# effect of each knob is visible in the outputs.
#
# Usage:
#   bash figure_concat.sh
#
# Optional:
#   FIG_DIR=<path>   Output root (default: <MEDVISION_DIR>/Figures); demo panels and
#                    composites are written to <FIG_DIR>/figure_concat_demo/
#
# Requires: pypdf (any PDF input/output), poppler's pdftocairo (PDF panels into a .png output).
#
# Knobs (python figure_concat.py --help):
#   --nrow N / --ncol N      grid shape, filled row-major; give one and the other is derived
#                            (default: a single column). Trailing slots may stay empty.
#   --fit width|height       which panel edge --base_size (inches) sets at scale 1.0;
#                            use height for one-row strips
#   --subplot_scales s ...   per-panel multiplier of the fitted edge, aspect preserved
#                            (0.5 = half). A column is as wide as its widest panel and a
#                            row as tall as its tallest, so a shrunken panel leaves
#                            whitespace instead of pulling its neighbours in.
#   --shifts_x dx ...        per-panel nudge inside its slot, as a fraction of the slot
#   --shifts_y dy ...        width / height; +x = right, +y = down, negatives allowed.
#                            Labels stay at the unshifted slot corner.
#   --no_labels              drop the (a) (b) (c) labels
#   --white_bg               opaque white instead of a transparent background
#   --dpi N                  .png output only (default: 300)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDVISION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

FIG_DIR="${FIG_DIR:-$MEDVISION_DIR/Figures}"
OUT_DIR="$FIG_DIR/figure_concat_demo"
mkdir -p "$OUT_DIR"

# ── Demo panels ─────────────────────────────────────────────────────────────
# A 4:3, B 1:1, C 3:4 (portrait), D 5:2 (wide); each saved as PDF and PNG.

python - "$OUT_DIR" <<'PY_EOF'
import sys
from pathlib import Path
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
for name, w, h, color in [
    ("A", 4, 3, "lightblue"),
    ("B", 3, 3, "navajowhite"),
    ("C", 3, 4, "lightgreen"),
    ("D", 5, 2, "lightpink"),
]:
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor(color)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, f"{name}\n{w}:{h}", ha="center", va="center",
            fontsize=36, transform=ax.transAxes)
    for ext in (".pdf", ".png"):
        fig.savefig(out / f"panel_{name}{ext}", dpi=150)
    plt.close(fig)
PY_EOF

A="$OUT_DIR/panel_A.pdf"
B="$OUT_DIR/panel_B.pdf"
C="$OUT_DIR/panel_C.pdf"
D="$OUT_DIR/panel_D.pdf"

# ── Grid shape ──────────────────────────────────────────────────────────────

# 3x1 column (the default when neither --nrow nor --ncol is given); every panel 4 in wide.
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" \
    --output "$OUT_DIR/grid_3x1_column.pdf" \
    --base_size 4

# 1x3 row: fit=height so every panel is 3 in tall and widths follow their aspect ratio.
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" \
    --output "$OUT_DIR/grid_1x3_row.pdf" \
    --nrow 1 \
    --fit height \
    --base_size 3

# 2x2 grid filled row-major: A B / C D.
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" "$D" \
    --output "$OUT_DIR/grid_2x2.pdf" \
    --ncol 2 \
    --base_size 3

# A grid need not be full: 3 panels with --nrow 2 derive ncol=2, and the last slot stays empty.
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" \
    --output "$OUT_DIR/grid_2x2_partial.pdf" \
    --nrow 2 \
    --base_size 3

# ── Scale subplots ──────────────────────────────────────────────────────────

# B at 60% and D at 80% width. Their column keeps the width of the full-size panel, so each
# shrunken panel sits flush left in its slot with whitespace to the right (and is centred
# vertically in its row, the non-fitted axis).
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" "$D" \
    --output "$OUT_DIR/scale_2x2.pdf" \
    --ncol 2 \
    --base_size 3 \
    --subplot_scales 1.0 0.6 1.0 0.8

# ── Shift subplots (x / y position) ─────────────────────────────────────────

# Shift x: shrink B to 60% and push it right by 20% of the slot width, which re-centres it
# under A and C. The (b) label does not move.
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" \
    --output "$OUT_DIR/shift_x_3x1.pdf" \
    --base_size 4 \
    --subplot_scales 1.0 0.6 1.0 \
    --shifts_x 0.0 0.2 0.0

# Shift y: no scaling needed -- drop B by 10% of the slot height in a one-row strip
# (a negative value would raise it instead).
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$B" "$C" \
    --output "$OUT_DIR/shift_y_1x3.pdf" \
    --nrow 1 \
    --fit height \
    --base_size 3 \
    --shifts_y 0.0 0.1 0.0

# ── Raster output ───────────────────────────────────────────────────────────

# A .png output accepts mixed inputs; PDF panels are rasterized (pdftocairo) at exactly the
# width they occupy. --dpi applies here only. Also: no labels, opaque white background.
python "$SCRIPT_DIR/figure_concat.py" \
    --images "$A" "$OUT_DIR/panel_B.png" "$C" "$OUT_DIR/panel_D.png" \
    --output "$OUT_DIR/mixed_2x2.png" \
    --ncol 2 \
    --base_size 3 \
    --dpi 150 \
    --no_labels \
    --white_bg
