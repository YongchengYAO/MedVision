"""
Visualize per-sample model responses for the Detection (bounding box) benchmark task.

Usage:
    python viz_detection_responses.py \
        --jsonl <path/to/parsed/samples.jsonl> \
        --output_dir <output/dir> \
        [--reshape_hw 512 512] \
        [--limit 10] \
        [--sample_ids 0 1 5]
"""

import argparse
import glob
import json
import os
import re

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.plot_utils import _get_appropriate_scale, save_fig_capped

# ── Color palette (light) ─────────────────────────────────────────────────────
C_FIG_BG = "#FFFFFF"
C_BOX_EDGE = "#4B5563"
C_SEP = "#D1D5DB"
C_HEADER = "#1D4ED8"
C_TEXT = "#111827"
C_THINK = "#9CA3AF"
C_REASON = "#0284C7"
C_STEP_ANS = "#16A34A"
C_ANS = "#EA580C"
C_TOOL = "#7C3AED"

# Prompt section highlights
C_IMG_PROMPT = "#D97706"
C_LABEL_NAME = "#059669"

# 4 unique coordinate colors — lower-left (x, y) and upper-right (x, y)
_COORD_COLORS = [
    "#DC2626",  # red-600    — lower-left x
    "#F97316",  # orange-500 — lower-left y
    "#06B6D4",  # cyan-500   — upper-right x
    "#0EA5E9",  # sky-500    — upper-right y
]

# Overlay bbox colors
C_GT_BOX = "#2ECC71"  # green  — ground truth
C_PRED_BOX = "#F37020"  # orange — model prediction

_TAG_COLORS = frozenset({C_THINK, C_REASON, C_STEP_ANS, C_ANS, C_TOOL})
C_TAG_GREY = "#6B7280"
_PROMPT_HEADERS = frozenset(
    {
        "Task:",
        "Additional information:",
        "Format requirement:",
        "Reasoning steps:",
    }
)

FIG_W = 30.24  # 25.2 × 1.2 — widened 20% (height auto-decreases as text reflows)
FONTSIZE_IN = 18.0
FONTSIZE_RS = 18.0
FONTSIZE_MT = 18.0
LH_MULT = 1.4
CHAR_RATIO = 0.601

# ── Webpage "Case Viewer" section-box palette (from medvision-vlm.github.io index.css) ──
C_SEG_FILL = "#ffffff"
C_SEG_EDGE = "#e6e9ef"
C_RAIL_TEAL = "#0E8C8B"  # Prompt & Metrics left rail + pill
C_RAIL_INDIGO = "#4f46e5"  # Response left rail + pill
C_FIGBOX_FILL = "#ffffff"
C_FIGBOX_EDGE = "#e6e9ef"

# ── Section-box / figure-box geometry (inches; FIG_W-relative). No outer card/stage. ──
OUTER_M = 0.22  # figure edge → content (small margin; outer card/stage removed)
COL_GAP_IN = 0.34  # left text column ↔ right figure box
LEFT_FRAC = 0.68  # text column : figure column (text widened)
SEG_GAP = 0.26  # gap between the 3 section boxes
SEG_PAD_X = 0.26  # seg border → text (both sides)
SEG_PAD_TOP = 0.24  # seg top border → pill top (margin above tag)
SEG_PAD_BOT = 0.26  # text bottom → seg bottom border (margin below text)
SEG_RADIUS = 0.14
RAIL_W = 0.06  # colored left rail thickness
PILL_H = 0.44  # tag height (enlarged)
PILL_PAD_X = 0.20  # text inset inside pill (enlarged)
PILL_GAP_BELOW = 0.20  # pill bottom → body text top (margin below tag)
PILL_FS = 18  # pill/tag font size (pt, sans bold; enlarged)
BODY_PAD = 0.10  # slack so last text line isn't flush to border
FIGBOX_PAD = 0.20  # figure box border → image region
FIGBOX_RADIUS = 0.14
TITLE_H = 0.42  # title band height above each figure
IMG_GAP = 0.24  # gap between the two stacked images
IMG_MAX_H = 9.5  # per-image height cap (raised so images fill the figure-box width)
GUT_L = 0.60  # left gutter (ylabel + yticks) for rotated bottom panel
GUT_B = 0.48  # bottom gutter (xlabel + xticks) for rotated bottom panel
# _draw_tokens wraps at max_x=0.985 of the axis while _count_wrapped_lines assumes the
# full width; shrink the estimate width to match so the box is tall enough (no clipping).
_WRAP_SAFETY = 0.95
SEG_EXTRA = SEG_PAD_TOP + PILL_H + PILL_GAP_BELOW + SEG_PAD_BOT  # pill+pad chrome per box

# Response tag patterns
_RESP_PATTERNS = [
    (re.compile(r"</?think>"), C_THINK),
    (re.compile(r"</?step-\d+-reasoning>"), C_REASON),
    (re.compile(r"</?step-\d+-answer>"), C_STEP_ANS),
    (re.compile(r"</?answer>"), C_ANS),
    (re.compile(r"</?tool_call>|</?tool_response>"), C_TOOL),
]

_STEP_TAG_RE = re.compile(r"<(/?)(step-(\d+)-(reasoning|answer))>")
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Detection task line: "Given ... return ... bounding box for the [label]."
_TASK_LINE_RE = re.compile(
    r"(Given the input medical image:\s*)(.*?)(,\s*return the coordinates.*?bounding box for the\s*)(.*?)(\..*)",
    re.DOTALL,
)


# ── Color extraction ───────────────────────────────────────────────────────────


def _extract_detection_color_maps(raw_text):
    """Extract 4 bbox coordinate strings from step-1-answer, map each to a color."""
    coord_map = {}
    m = re.search(r"<step-1-answer>(.*?)</step-1-answer>", raw_text, re.DOTALL)
    if m:
        for i, n in enumerate(_NUM_RE.findall(m.group(1))[:4]):
            coord_map.setdefault(n, _COORD_COLORS[i])
    # Fallback: use <answer> tag when step-1-answer is absent
    if not coord_map:
        m = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)
        if m:
            for i, n in enumerate(_NUM_RE.findall(m.group(1))[:4]):
                coord_map.setdefault(n, _COORD_COLORS[i])
    return coord_map


def _colorize_segment(text, in_coord_ctx, coord_map):
    out = []
    for piece in re.split(r"(\d+\.?\d*)", text):
        if not piece:
            continue
        if re.fullmatch(r"\d+\.?\d*", piece) and in_coord_ctx:
            c = coord_map.get(piece, C_TEXT)
            out.append((piece, c, False))
        else:
            out.append((piece, C_TEXT, False))
    return out


def _add_detection_number_colors(tokens, coord_map):
    """Color numbers in <step-1-answer> and <answer> contexts using coord_map."""
    result = []
    ctx_step, ctx_part, in_final_answer = None, None, False

    for seg, color, bold in tokens:
        if color in (C_REASON, C_STEP_ANS):
            m = _STEP_TAG_RE.match(seg)
            if m:
                if m.group(1) == "/":
                    ctx_part = None
                else:
                    ctx_step = int(m.group(3))
                    ctx_part = m.group(4)
        elif color == C_ANS:
            in_final_answer = seg == "<answer>"

        in_coord_ctx = (ctx_step == 1) or in_final_answer
        if color == C_TEXT and in_coord_ctx:
            result.extend(_colorize_segment(seg, True, coord_map))
        else:
            result.append((seg, color, bold))
    return result


# ── Tokenizers ─────────────────────────────────────────────────────────────────


def _tokenize_resp(text):
    tokens, pos, n = [], 0, len(text)
    while pos < n:
        best_m, best_c = None, C_TEXT
        for pat, color in _RESP_PATTERNS:
            m = pat.search(text, pos)
            if m and (best_m is None or m.start() < best_m.start()):
                best_m, best_c = m, color
        if best_m is None:
            tokens.append((text[pos:], C_TEXT, False))
            break
        if best_m.start() > pos:
            tokens.append((text[pos : best_m.start()], C_TEXT, False))
        tokens.append((best_m.group(), best_c, best_c in _TAG_COLORS))
        pos = best_m.end()
    return tokens


def _tokenize_number_line(line, number_color):
    out, pos = [], 0
    for m in re.finditer(r"\d+(?:\.\d+)?", line):
        if m.start() > pos:
            out.append((line[pos : m.start()], C_TEXT, False))
        out.append((m.group(), number_color, False))
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], C_TEXT, False))
    return out


def _tokenize_task_line(line):
    m = _TASK_LINE_RE.match(line)
    if m:
        return [
            (m.group(1), C_TEXT, False),
            (m.group(2), C_IMG_PROMPT, False),
            (m.group(3), C_TEXT, False),
            (m.group(4), C_LABEL_NAME, False),
            (m.group(5), C_TEXT, False),
        ]
    return [(line, C_TEXT, False)]


def _tokenize_input(text):
    tokens = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in _PROMPT_HEADERS:
            tokens.append((line, C_TEXT, True))
        elif "Given the input medical image:" in line:
            tokens.extend(_tokenize_task_line(line))
        else:
            tokens.append((line, C_TEXT, False))
        tokens.append(("\n", C_TEXT, False))
    if tokens and tokens[-1] == ("\n", C_TEXT, False):
        tokens.pop()
    return tokens


def _preprocess_response(s):
    for pat in [
        r"</?think>",
        r"</?step-\d+-reasoning>",
        r"</?step-\d+-answer>",
        r"</?answer>",
        r"</?tool_call>",
        r"</?tool_response>",
    ]:
        s = re.sub(r"[^\S\n]*(" + pat + r")[^\S\n]*", r"\n\1\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ── Layout helpers ─────────────────────────────────────────────────────────────


def _count_wrapped_lines(tokens, fontsize, ax_w_in):
    char_w_in = fontsize * CHAR_RATIO / 72
    max_chars = ax_w_in / char_w_in
    line_count, x = 1, 0.0
    for seg, *_ in tokens:
        parts = seg.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                line_count += 1
                x = 0.0
            for wi, word in enumerate(part.split(" ")):
                piece_len = float(len(word) if wi == 0 else len(word) + 1)
                if x > 0 and x + piece_len > max_chars:
                    line_count += 1
                    x = float(len(word))
                else:
                    x += piece_len
    return line_count


def _text_block_h_in(tokens, fontsize, ax_w_in):
    """Pure wrapped-text height in inches (no floor / no pad).

    The box chrome (pill + paddings, ``SEG_EXTRA``) supplies vertical padding
    separately, so — unlike the old ``_section_h_in`` — this must NOT add its own
    pad or 1.5in floor, else the section-box height double-counts that space.

    Estimates wrap at ``_WRAP_SAFETY`` × the axis width to match ``_draw_tokens``
    (which wraps at max_x=0.985), so the box is tall enough to show every line.
    """
    n = _count_wrapped_lines(tokens, fontsize, ax_w_in * _WRAP_SAFETY)
    return n * fontsize * LH_MULT / 72 + BODY_PAD


def _draw_tokens(ax, tokens, x0, y0, fontsize, ax_w_in, ax_h_in):
    char_w_ax = (fontsize * CHAR_RATIO / 72) / ax_w_in
    line_h_ax = (fontsize * LH_MULT / 72) / ax_h_in
    max_x = 0.985

    lines = [[]]
    for seg, color, bold in tokens:
        parts = seg.split("\n")
        lines[-1].append((parts[0], color, bold))
        for p in parts[1:]:
            lines.append([(p, color, bold)])

    y = y0
    for line_segs in lines:
        x = x0
        for seg, color, bold in line_segs:
            if not seg:
                continue
            display_color = C_TEXT if color in _TAG_COLORS else color
            fw = "bold" if (bold or color != C_TEXT) else "normal"
            words = seg.split(" ")
            for wi, word in enumerate(words):
                piece = word if wi == 0 else " " + word
                need_w = len(piece) * char_w_ax
                if x > x0 and x + need_w > max_x:
                    y -= line_h_ax
                    x = x0
                    piece = word
                    need_w = len(piece) * char_w_ax
                if piece:
                    ax.text(
                        x,
                        y,
                        piece,
                        color=display_color,
                        fontsize=fontsize,
                        fontweight=fw,
                        fontfamily="monospace",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        clip_on=True,
                    )
                    x += need_w
        y -= line_h_ax


# ── Webpage-card chrome helpers (shared verbatim by AD / TL / detection) ───────
# All chrome is drawn into ONE full-figure inch-axes (xlim=(0,FIG_W), ylim=(0,fig_h)),
# so 1 data-unit == 1 inch on BOTH axes and FancyBboxPatch corners stay circular
# regardless of the content-driven fig_h.


def _round_rect(
    ax, x, y, w, h, *, radius, facecolor, edgecolor="none", linewidth=1.0, zorder=1
):
    """Rounded rectangle in inch-axes data coords (circular corners)."""
    r = min(radius, w / 2, h / 2)
    patch = mpatches.FancyBboxPatch(
        (x + r, y + r),
        w - 2 * r,
        h - 2 * r,
        boxstyle=f"round,pad={r},rounding_size={r}",
        mutation_aspect=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        transform=ax.transData,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def _pill(ax, x, y, text, *, color, height=PILL_H, fs=PILL_FS, pad_x=PILL_PAD_X, zorder=6):
    """Fully-rounded pill with centered white bold uppercase text. Returns width (in)."""
    label = text.upper()
    char_w_in = fs * 0.60 / 72  # sans-bold avg advance ≈ 0.60·fontsize
    w = len(label) * char_w_in + 2 * pad_x
    _round_rect(ax, x, y, w, height, radius=height / 2, facecolor=color, zorder=zorder)
    ax.text(
        x + w / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=fs,
        fontweight="bold",
        fontfamily="sans-serif",
        zorder=zorder + 1,
    )
    return w


def _left_rail(ax, x, y, h, *, color, width=RAIL_W, zorder=5):
    """Thin rounded vertical capsule (section-box left rail)."""
    _round_rect(ax, x, y, width, h, radius=width / 2, facecolor=color, zorder=zorder)


def _place_image_box(fig, box_rect_frac, img_aspect):
    """Contain+center an image of physical aspect (h/w) inside a box.

    Returns an ``[l, b, w, h]`` fraction rect whose physical inch-aspect == img_aspect,
    so imshow(aspect=...) fills it exactly with no matplotlib re-fit margin.
    """
    fig_w, fig_h = fig.get_size_inches()
    l, b, w, h = box_rect_frac
    bw, bh = w * fig_w, h * fig_h
    box_aspect = bh / bw
    if img_aspect >= box_aspect:  # image taller → height-limited
        ih = bh
        iw = ih / img_aspect
    else:  # width-limited
        iw = bw
        ih = iw * img_aspect
    return [l + (bw - iw) / 2 / fig_w, b + (bh - ih) / 2 / fig_h, iw / fig_w, ih / fig_h]


def _rotated_overlay_chrome(ax, image_2d, psz, slice_dim):
    """Add L-shaped mm scale bar + orientation axis labels + ticks to a rotated
    overlay panel drawn via ``imshow(image_2d.T, origin='lower')``.

    Mirrors the webpage renderer (``plot_utils.plot_*_on_image``) but scales the
    28/20/16 pt fonts down by the panel's on-figure width so labels fit the reserved
    gutters. Returns the scaled legend fontsize. Assumes fixed limits already set.
    """
    H, W = image_2d.shape
    pos = ax.get_position()
    fig_w, _ = ax.figure.get_size_inches()
    ax_w_in = pos.width * fig_w
    s = min(0.6, max(0.35, ax_w_in / 10.0))
    lbl_fs, tick_fs, scale_fs, leg_fs = (
        round(28 * s),
        round(20 * s),
        round(20 * s),
        round(16 * s),
    )

    # L-shaped white scale bar at lower-left (origin='lower'); mirrors plot_utils.
    min_idx = int(np.argmin(image_2d.shape[:2]))
    scale_mm, n_min = _get_appropriate_scale(psz[min_idx], image_2d.shape[min_idx], 10)
    n_max = int(scale_mm / psz[1 - min_idx])
    sp0, sp1 = (n_min, n_max) if min_idx == 0 else (n_max, n_min)  # (dim0, dim1) px
    sx, sy = int(H * 0.05), int(W * 0.05)
    ax.plot([sx, sx + sp0], [sy, sy], "w-", lw=3)
    ax.plot([sx, sx], [sy, sy + sp1], "w-", lw=3)
    ax.text(
        sx + sp0 + H * 0.01, sy, f"{scale_mm} mm", color="white", ha="left", fontsize=scale_fs
    )

    xl, yl = {
        0: ("Anterior →", "Superior →"),
        1: ("Right →", "Superior →"),
        2: ("Right →", "Anterior →"),
    }[slice_dim]
    ax.set_xlabel(xl, fontsize=lbl_fs)
    ax.set_ylabel(yl, fontsize=lbl_fs)
    ax.tick_params(labelsize=tick_fs)
    return leg_fs


# ── Sample filtering (mirrors summarize_detection_task.py) ────────────────────


# ── Image / NIfTI loading ──────────────────────────────────────────────────────


def _load_nifti_2d(nii_path, slice_dim, slice_idx, new_shape_hw=None):
    img_nib = nib.load(nii_path)
    voxel_size = img_nib.header.get_zooms()
    image_3d = img_nib.get_fdata().astype("float32")
    if slice_dim == 0:
        image_2d = image_3d[slice_idx, :, :]
        psz = (float(voxel_size[1]), float(voxel_size[2]))
    elif slice_dim == 1:
        image_2d = image_3d[:, slice_idx, :]
        psz = (float(voxel_size[0]), float(voxel_size[2]))
    else:
        image_2d = image_3d[:, :, slice_idx]
        psz = (float(voxel_size[0]), float(voxel_size[1]))
    if new_shape_hw is not None:
        orig = image_2d.shape
        f = (new_shape_hw[0] / orig[0], new_shape_hw[1] / orig[1])
        image_2d = zoom(image_2d, f, order=0 if len(np.unique(image_2d)) <= 2 else 1)
        psz = (psz[0] / f[0], psz[1] / f[1])
    return psz, image_2d


def _load_image(doc, reshape_hw):
    _, image_2d = _load_nifti_2d(
        doc["image_file"],
        doc["slice_dim"],
        doc["slice_idx"],
        new_shape_hw=reshape_hw,
    )
    return normalize_img(doc, image_2d)


# ── Overlay (GT + predicted bounding boxes) ───────────────────────────────────


def _parse_bbox(coords_str):
    """Parse bbox coordinate string into [x_min, y_min, x_max, y_max] floats, or None."""
    nums = [s.replace(",", "") for s in _NUM_RE.findall(coords_str or "")]
    if len(nums) < 4:
        return None
    try:
        return [float(x) for x in nums[-4:]]
    except ValueError:
        return None


def _bbox_to_dim01(bbox_rel, H, W):
    """Convert normalized [x_min, y_min, x_max, y_max] (lower-left origin) → rotated plot
    space: dim0 = plot-x (horizontal), dim1 = plot-y (vertical)."""
    x_min, y_min, x_max, y_max = bbox_rel
    d0_min = H * (1 - y_max)
    d0_max = H * (1 - y_min)
    d1_min = x_min * W
    d1_max = x_max * W
    return d0_min, d1_min, d0_max, d1_max


def _draw_detection_overlay_on_ax(ax, doc, sample):
    """Draw the original-resolution image with GT (green dashed) and predicted (orange solid)
    bounding boxes, rotated 90° CCW to match the webpage renderer: imshow(image_2d.T,
    origin='lower', aspect=psz[1]/psz[0]); Rectangle anchored at (dim0_min, dim1_min)."""
    try:
        psz, image_2d = _load_nifti_2d(
            doc["image_file"], doc["slice_dim"], doc["slice_idx"]
        )
        image_2d = normalize_img(doc, image_2d)
    except Exception:
        ax.set_axis_off()
        return

    H, W = image_2d.shape
    ax.imshow(image_2d.T, cmap="gray", origin="lower", aspect=psz[1] / psz[0], zorder=-1)
    ax.set_xlim(-0.5, H - 0.5)
    ax.set_ylim(-0.5, W - 0.5)
    ax.set_autoscale_on(False)

    # GT bbox
    raw_target = sample.get("target", "[]")
    gt_bbox = json.loads(raw_target) if isinstance(raw_target, str) else raw_target
    if gt_bbox and len(gt_bbox) == 4:
        d0min, d1min, d0max, d1max = _bbox_to_dim01(gt_bbox, H, W)
        ax.add_patch(mpatches.Rectangle((d0min, d1min), d0max - d0min, d1max - d1min, linewidth=4, edgecolor=C_GT_BOX, facecolor="none", linestyle="--", zorder=2))

    # Predicted bbox
    pred_str = (sample.get("filtered_resps") or [""])[0]
    pred_bbox = _parse_bbox(pred_str)
    if pred_bbox:
        d0min, d1min, d0max, d1max = _bbox_to_dim01(pred_bbox, H, W)
        ax.add_patch(mpatches.Rectangle((d0min, d1min), d0max - d0min, d1max - d1min, linewidth=4, edgecolor=C_PRED_BOX, facecolor="none", linestyle="-", zorder=3))

    leg_fs = _rotated_overlay_chrome(ax, image_2d, psz, doc["slice_dim"])
    legend_handles = [
        mlines.Line2D([], [], color=C_GT_BOX, ls="--", linewidth=4, label="GT bbox"),
        mlines.Line2D([], [], color=C_PRED_BOX, ls="-", linewidth=4, label="Pred bbox"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=leg_fs, facecolor="white", edgecolor="#D1D5DB", framealpha=0.85)


# ── Sample data helpers ────────────────────────────────────────────────────────


def _build_metrics_tokens(sample):
    """Return token list for the metrics section."""
    raw_target = sample.get("target", "[]")
    gt_bbox = json.loads(raw_target) if isinstance(raw_target, str) else raw_target

    pred_str = (sample.get("filtered_resps") or [""])[0]
    pred_bbox = _parse_bbox(pred_str)

    iou = sample.get("avgIoU", {}).get("IoU")
    f1 = sample.get("F1", {}).get("F1")
    precision = sample.get("Precision", {}).get("Precision")
    recall = sample.get("Recall", {}).get("Recall")

    # Build coord color map for prediction values
    pred_color = {}
    if pred_bbox:
        for i, v in enumerate(pred_bbox):
            key = f"{v:.3f}"
            if key not in pred_color:
                pred_color[key] = _COORD_COLORS[i]

    def _tokenize_line(text, highlight=None):
        toks = []
        if highlight:
            pos = 0
            for m in re.finditer(r"\d+\.\d+", text):
                if m.start() > pos:
                    toks.append((text[pos : m.start()], C_TEXT, False))
                c = highlight.get(m.group(), C_TEXT)
                toks.append((m.group(), c, c != C_TEXT))
                pos = m.end()
            if pos < len(text):
                toks.append((text[pos:], C_TEXT, False))
        else:
            toks.append((text, C_TEXT, False))
        toks.append(("\n", C_TEXT, False))
        return toks

    fmt4 = lambda v: f"{v:.4f}" if v is not None else "N/A"

    if gt_bbox and len(gt_bbox) == 4:
        gt_str = (
            f"[{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]"
        )
    else:
        gt_str = "N/A"

    if pred_bbox:
        pred_str_fmt = f"[{pred_bbox[0]:.3f}, {pred_bbox[1]:.3f}, {pred_bbox[2]:.3f}, {pred_bbox[3]:.3f}]"
    else:
        pred_str_fmt = "N/A"

    tokens = []
    tokens += _tokenize_line(f"Ground Truth:  bbox = {gt_str}")
    tokens += _tokenize_line(
        f"Prediction:    bbox = {pred_str_fmt}", highlight=pred_color
    )
    tokens += _tokenize_line(
        f"Metrics:       Recall = {fmt4(recall)}    Precision = {fmt4(precision)}    F1 = {fmt4(f1)}    IoU = {fmt4(iou)}"
    )

    if tokens and tokens[-1] == ("\n", C_TEXT, False):
        tokens.pop()
    return tokens


# ── Filename helpers ───────────────────────────────────────────────────────────


def _get_label_str(doc):
    bp = doc.get("biometric_profile", {})
    if "metric_key" in bp:
        return bp["metric_key"]
    return str(doc.get("label", "unknown"))


def _make_filename(sample, dataset_name):
    doc = sample["doc"]
    doc_id = sample["doc_id"]
    task_id = int(doc.get("taskID", 0))
    slice_dim = doc.get("slice_dim", 0)
    slice_idx = doc.get("slice_idx", 0)
    label = re.sub(r"[^\w\-]", "_", _get_label_str(doc))
    return (
        f"{dataset_name}__Task{task_id:02d}"
        f"__doc{doc_id}__dim{slice_dim}__idx{slice_idx}__{label}.png"
    )


# ── Main per-sample plot ───────────────────────────────────────────────────────


def _plot_sample(sample, out_path, reshape_hw, formats, dpi=100):
    doc = sample["doc"]
    doc_id = sample["doc_id"]

    try:
        img = _load_image(doc, reshape_hw)
    except FileNotFoundError as e:
        print(f"  [Warning] Skipping doc_id={doc_id}: {e}")
        return None

    raw_resp = sample["resps"][0][0]
    input_text = sample.get("input", "")
    resp_text = _preprocess_response(raw_resp).replace("\n", " ")

    in_tokens = _tokenize_input(input_text)
    coord_map = _extract_detection_color_maps(raw_resp)
    rs_tokens = _add_detection_number_colors(_tokenize_resp(resp_text), coord_map)
    mt_tokens = _build_metrics_tokens(sample)

    # ── Layout: webpage "Case Viewer" section boxes (no outer card) ────────────
    toks = [in_tokens, rs_tokens, mt_tokens]
    fss = [FONTSIZE_IN, FONTSIZE_RS, FONTSIZE_MT]
    rail_colors = [C_RAIL_TEAL, C_RAIL_INDIGO, C_RAIL_TEAL]
    pill_texts = ["Prompt", "Response", "GT · Prediction · Metrics"]

    # Horizontal budget (inches) — text width drives wrapping, so widths come first.
    content_w = FIG_W - 2 * OUTER_M
    avail = content_w - COL_GAP_IN
    left_w = LEFT_FRAC * avail
    right_w = avail - left_w
    text_w = left_w - RAIL_W - 2 * SEG_PAD_X

    # Section-box heights = chrome (SEG_EXTRA) + pure wrapped-text height.
    t_h = [_text_block_h_in(toks[i], fss[i], text_w) for i in range(3)]
    seg_h = [SEG_EXTRA + t_h[i] for i in range(3)]
    left_stack_h = sum(seg_h) + 2 * SEG_GAP

    # Right-column image aspects: top = unrotated display; bottom = rotated overlay.
    H_img, W_img = img.shape
    disp_aspect = H_img / W_img
    _nii = nib.load(doc["image_file"])
    _shape = _nii.header.get_data_shape()
    _z = _nii.header.get_zooms()
    _sd = doc["slice_dim"]
    if _sd == 0:
        _H, _W, _px0, _px1 = _shape[1], _shape[2], _z[1], _z[2]
    elif _sd == 1:
        _H, _W, _px0, _px1 = _shape[0], _shape[2], _z[0], _z[2]
    else:
        _H, _W, _px0, _px1 = _shape[0], _shape[1], _z[0], _z[1]
    rot_aspect = (_W * _px1) / (_H * _px0)  # real-world mm-height / mm-width, rotated

    # Figure box hugs the two images (each with a title). The figure DYNAMICALLY
    # SHRINKS to fit within the text-column height, so it never forces extra height
    # (short responses no longer leave the figure taller than the text).
    fig_inner_w = right_w - 2 * FIGBOX_PAD
    top_nat = min(fig_inner_w * disp_aspect, IMG_MAX_H)
    bot_nat = min((fig_inner_w - GUT_L) * rot_aspect, IMG_MAX_H)
    fig_overhead = 2 * FIGBOX_PAD + 2 * TITLE_H + IMG_GAP + GUT_B  # non-image chrome
    content_h = max(left_stack_h, fig_overhead)  # text column drives overall height
    nat_img_h = top_nat + bot_nat
    scale = min(1.0, (content_h - fig_overhead) / nat_img_h) if nat_img_h > 0 else 1.0
    top_h = top_nat * scale
    bot_h = bot_nat * scale
    figbox_h = fig_overhead + top_h + bot_h
    fig_h = content_h + 2 * OUTER_M

    # ── Figure + full-figure inch-axes carrying the section / figure boxes ────
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=C_FIG_BG)
    fx = lambda v: v / FIG_W
    fy = lambda v: v / fig_h
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax_bg.set_xlim(0, FIG_W)
    ax_bg.set_ylim(0, fig_h)
    ax_bg.set_facecolor("none")
    ax_bg.axis("off")

    content_top = fig_h - OUTER_M
    content_bot = OUTER_M
    left_x0 = OUTER_M
    right_x0 = left_x0 + left_w + COL_GAP_IN

    # ── Left column: 3 section boxes (rail + pill + text), top-aligned ────────
    y = content_top
    for i in range(3):
        seg_top = y
        seg_bot = seg_top - seg_h[i]
        _round_rect(
            ax_bg, left_x0, seg_bot, left_w, seg_h[i], radius=SEG_RADIUS,
            facecolor=C_SEG_FILL, edgecolor=C_SEG_EDGE, linewidth=1.0, zorder=4,
        )
        _left_rail(
            ax_bg, left_x0, seg_bot + SEG_RADIUS, seg_h[i] - 2 * SEG_RADIUS,
            color=rail_colors[i], zorder=5,
        )
        text_x0 = left_x0 + RAIL_W + SEG_PAD_X
        _pill(
            ax_bg, text_x0, seg_top - SEG_PAD_TOP - PILL_H, pill_texts[i],
            color=rail_colors[i], zorder=6,
        )
        text_top = seg_top - SEG_PAD_TOP - PILL_H - PILL_GAP_BELOW
        text_bot = seg_bot + SEG_PAD_BOT
        ax_txt = fig.add_axes(
            [fx(text_x0), fy(text_bot), fx(text_w), fy(text_top - text_bot)]
        )
        ax_txt.set_facecolor("none")
        ax_txt.axis("off")
        _draw_tokens(ax_txt, toks[i], 0.002, 0.998, fss[i], text_w, text_top - text_bot)
        y = seg_bot - SEG_GAP

    # ── Right column: figure box (vertically centered) hugging the two images ──
    figbox_top = content_top - (content_h - figbox_h) / 2
    figbox_bot = figbox_top - figbox_h
    _round_rect(
        ax_bg, right_x0, figbox_bot, right_w, figbox_h, radius=FIGBOX_RADIUS,
        facecolor=C_FIGBOX_FILL, edgecolor=C_FIGBOX_EDGE, linewidth=1.0, zorder=4,
    )
    inner_x0 = right_x0 + FIGBOX_PAD
    inner_cx = right_x0 + right_w / 2
    inner_top = figbox_top - FIGBOX_PAD
    title_kw = dict(
        ha="center", va="center", fontsize=16, fontweight="bold", color=C_TEXT,
        fontfamily="sans-serif", zorder=6,
    )

    # Top: title + input image (unrotated).
    ax_bg.text(inner_cx, inner_top - TITLE_H / 2, "Input Image", **title_kw)
    top_img_top = inner_top - TITLE_H
    ax_img = fig.add_axes(
        _place_image_box(
            fig,
            [fx(inner_x0), fy(top_img_top - top_h), fx(fig_inner_w), fy(top_h)],
            disp_aspect,
        )
    )
    ax_img.imshow(img, cmap="gray", aspect="equal")
    ax_img.axis("off")

    # Bottom: title + rotated overlay (gutters reserved for labels/ticks).
    bunit_top = top_img_top - top_h - IMG_GAP
    ax_bg.text(
        inner_cx, bunit_top - TITLE_H / 2, "Image with GT and Prediction", **title_kw
    )
    bot_img_top = bunit_top - TITLE_H
    ax_ovl = fig.add_axes(
        _place_image_box(
            fig,
            [fx(inner_x0 + GUT_L), fy(bot_img_top - bot_h), fx(fig_inner_w - GUT_L), fy(bot_h)],
            rot_aspect,
        )
    )
    _draw_detection_overlay_on_ax(ax_ovl, doc, sample)

    stem = os.path.splitext(out_path)[0]
    for fmt in formats:
        save_fig_capped(f"{stem}.{fmt}", dpi=dpi, transparent=True)
    plt.close(fig)
    return f"{stem}.{formats[0]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Single JSONL file (mutually exclusive with --model_dir).",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Model results folder; loops all *.jsonl in {model_dir}/parsed/.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--reshape_hw", nargs=2, type=int, default=None, metavar=("H", "W")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit_per_jsonl",
        type=int,
        default=None,
        help="Max figures per JSONL file (--model_dir mode only).",
    )
    parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--save_as_png", action="store_true", help="Save figures as PNG."
    )
    parser.add_argument(
        "--save_as_pdf", action="store_true", help="Save figures as PDF."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure save DPI (clamped down to the 34MP arXiv cap). Default 100.",
    )
    args = parser.parse_args()

    if args.jsonl and args.model_dir:
        parser.error("--jsonl and --model_dir are mutually exclusive.")
    if not args.jsonl and not args.model_dir:
        parser.error("One of --jsonl or --model_dir is required.")
    if args.limit_per_jsonl is not None and not args.model_dir:
        parser.error("--limit_per_jsonl requires --model_dir.")

    reshape_hw = tuple(args.reshape_hw) if args.reshape_hw else None
    formats = [
        f for f, on in (("png", args.save_as_png), ("pdf", args.save_as_pdf)) if on
    ] or ["pdf"]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_dir:
        # ── Batch mode: loop all JSONL files in {model_dir}/parsed/ ──────────
        parsed_dir = os.path.join(args.model_dir, "parsed")
        if not os.path.isdir(parsed_dir):
            raise FileNotFoundError(f"Parsed directory not found: {parsed_dir}")
        all_jsonls = sorted(glob.glob(os.path.join(parsed_dir, "*.jsonl")))
        all_jsonls = [
            f
            for f in all_jsonls
            if "_proc_acc" not in os.path.basename(f)
            and "_eq_acc" not in os.path.basename(f)
        ]
        if not all_jsonls:
            raise FileNotFoundError(f"No JSONL files found in: {parsed_dir}")

        if args.limit_per_jsonl is not None:
            per_file_limit = args.limit_per_jsonl
        elif args.limit is not None:
            per_file_limit = max(1, args.limit // len(all_jsonls))
        else:
            per_file_limit = None

        model_name = os.path.basename(args.model_dir.rstrip("/"))

        for jsonl_path in all_jsonls:
            m = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
            dataset_name = m.group(1) if m else "unknown"

            per_file_out_dir = os.path.join(args.output_dir, model_name, dataset_name)
            os.makedirs(per_file_out_dir, exist_ok=True)

            with open(jsonl_path) as f:
                samples = [json.loads(l) for l in f if l.strip()]

            if args.sample_ids is not None:
                samples = [s for s in samples if s["doc_id"] in set(args.sample_ids)]
            if per_file_limit is not None:
                samples = samples[:per_file_limit]

            print(
                f"[{os.path.basename(jsonl_path)}] {len(samples)} sample(s) → {per_file_out_dir}"
            )
            for i, sample in enumerate(samples):
                doc_id = sample["doc_id"]
                fname = _make_filename(sample, dataset_name)
                out_path = os.path.join(per_file_out_dir, fname)
                out = _plot_sample(sample, out_path, reshape_hw, formats, dpi=args.dpi)
                if out:
                    print(f"  [{i+1}/{len(samples)}] doc_id={doc_id} → {out}")

    else:
        # ── Single-file mode (existing behavior) ──────────────────────────────
        with open(args.jsonl) as f:
            samples = [json.loads(l) for l in f if l.strip()]

        if args.sample_ids is not None:
            samples = [s for s in samples if s["doc_id"] in set(args.sample_ids)]
        if args.limit is not None:
            samples = samples[: args.limit]

        print(f"Processing {len(samples)} sample(s) → {args.output_dir}")
        for i, sample in enumerate(samples):
            doc_id = sample["doc_id"]
            out_path = os.path.join(args.output_dir, f"doc_{doc_id:04d}.png")
            out = _plot_sample(sample, out_path, reshape_hw, formats, dpi=args.dpi)
            if out:
                print(f"  [{i+1}/{len(samples)}] doc_id={doc_id} → {out}")


if __name__ == "__main__":
    main()
