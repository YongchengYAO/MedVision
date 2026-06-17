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

FIG_W = 25.2
FONTSIZE_IN = 18.0
FONTSIZE_RS = 18.0
FONTSIZE_MT = 18.0
LH_MULT = 1.4
CHAR_RATIO = 0.601

# Two-column layout (figure fractions)
LM = 0.025
_IX = LM + 0.008
COL_GAP = 0.020
COL2_W_FRAC = 0.2
COL2_X_FRAC = 1.0 - COL2_W_FRAC - LM - 0.005
COL1_W_FRAC = COL2_X_FRAC - _IX - COL_GAP

COL1_AX_W_IN = (COL1_W_FRAC - 0.014) * FIG_W
COL2_W_IN = COL2_W_FRAC * FIG_W

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


def _section_h_in(tokens, fontsize, ax_w_in, pad_in=0.40):
    n = _count_wrapped_lines(tokens, fontsize, ax_w_in)
    return max(1.5, n * fontsize * LH_MULT / 72 + pad_in)


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


# ── Sample filtering (mirrors summarize_detection_task.py) ────────────────────


def _build_removed_set(json_path):
    _dim_map = {"x": 0, "y": 1, "z": 2}
    with open(json_path) as f:
        entries = json.load(f)
    return frozenset(
        (
            e["image_file"],
            _dim_map[e["slice_dim"]],
            int(e["slice_idx"]),
            int(e["task_ID"]),
        )
        for e in entries
    )


def _relative_image_file(full_path, dataset_name):
    marker = f"/{dataset_name}/"
    idx = full_path.find(marker)
    return full_path[idx + len(marker) :] if idx >= 0 else os.path.basename(full_path)


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


def _bbox_to_image_coords(bbox_rel, H, W):
    """Convert normalized [x_min, y_min, x_max, y_max] (lower-left origin) → col/row for imshow(origin='upper')."""
    x_min, y_min, x_max, y_max = bbox_rel
    col_min = x_min * W
    col_max = x_max * W
    row_min = H * (1 - y_max)
    row_max = H * (1 - y_min)
    return col_min, row_min, col_max, row_max


def _draw_detection_overlay_on_ax(ax, doc, sample):
    """Draw original-resolution image with GT (green dashed) and predicted (orange solid) bounding boxes."""
    try:
        _, image_2d = _load_nifti_2d(
            doc["image_file"], doc["slice_dim"], doc["slice_idx"]
        )
        image_2d = normalize_img(doc, image_2d)
    except Exception:
        ax.axis("off")
        return

    H, W = image_2d.shape
    ax.imshow(image_2d, cmap="gray", origin="upper")

    # GT bbox
    raw_target = sample.get("target", "[]")
    gt_bbox = json.loads(raw_target) if isinstance(raw_target, str) else raw_target
    if gt_bbox and len(gt_bbox) == 4:
        c0, r0, c1, r1 = _bbox_to_image_coords(gt_bbox, H, W)
        ax.add_patch(
            mpatches.Rectangle(
                (c0, r0),
                c1 - c0,
                r1 - r0,
                linewidth=4,
                edgecolor=C_GT_BOX,
                facecolor="none",
                linestyle="--",
                zorder=2,
            )
        )

    # Predicted bbox
    pred_str = (sample.get("filtered_resps") or [""])[0]
    pred_bbox = _parse_bbox(pred_str)
    if pred_bbox:
        c0, r0, c1, r1 = _bbox_to_image_coords(pred_bbox, H, W)
        ax.add_patch(
            mpatches.Rectangle(
                (c0, r0),
                c1 - c0,
                r1 - r0,
                linewidth=4,
                edgecolor=C_PRED_BOX,
                facecolor="none",
                linestyle="-",
                zorder=3,
            )
        )

    legend_handles = [
        mlines.Line2D([], [], color=C_GT_BOX, ls="--", linewidth=4, label="GT bbox"),
        mlines.Line2D([], [], color=C_PRED_BOX, ls="-", linewidth=4, label="Pred bbox"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=14,
        facecolor="white",
        edgecolor="#D1D5DB",
        framealpha=0.85,
    )
    ax.axis("off")


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


def _plot_sample(sample, out_path, reshape_hw):
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
    in_tokens.append(("\n", C_TEXT, False))
    coord_map = _extract_detection_color_maps(raw_resp)
    rs_tokens = _add_detection_number_colors(_tokenize_resp(resp_text), coord_map)
    mt_tokens = _build_metrics_tokens(sample)

    # ── Layout constants (inches) ─────────────────────────────────────────────
    HDR_H = 0.35
    PAD = 0.20
    BOX_GAP = 0.40
    OUTER_TOP = 0.40
    OUTER_BOT = 0.20

    in_h = _section_h_in(in_tokens, FONTSIZE_IN, COL1_AX_W_IN, PAD)
    rs_h = _section_h_in(rs_tokens, FONTSIZE_RS, COL1_AX_W_IN, PAD)
    mt_h = _section_h_in(mt_tokens, FONTSIZE_MT, COL1_AX_W_IN, PAD)
    col1_content_h = (
        (HDR_H + in_h) + 2 * BOX_GAP + (HDR_H + rs_h) + BOX_GAP + (HDR_H + mt_h)
    )

    fig_h = col1_content_h + OUTER_TOP + OUTER_BOT

    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor=C_FIG_BG)

    tf = lambda v: v / fig_h

    # ── Column 1: section y-positions (bottom-up) ─────────────────────────────
    y = OUTER_BOT
    mt_bot = tf(y)
    mt_hf = tf(HDR_H + mt_h)
    y += HDR_H + mt_h + BOX_GAP
    rs_bot = tf(y)
    rs_hf = tf(HDR_H + rs_h)
    y += HDR_H + rs_h + 2 * BOX_GAP
    in_bot = tf(y)
    in_hf = tf(HDR_H + in_h)

    hkw = dict(
        transform=fig.transFigure,
        fontsize=22,
        fontweight="bold",
        color=C_TEXT,
        va="bottom",
        fontfamily="monospace",
        zorder=0,
    )
    top_pad = tf(0.02)
    fig.text(_IX + 0.006, in_bot + in_hf - tf(HDR_H * 0.3), "[ Prompt ]", **hkw)
    fig.text(_IX + 0.006, rs_bot + rs_hf - tf(HDR_H * 0.3), "[ Response ]", **hkw)
    fig.text(
        _IX + 0.006,
        mt_bot + mt_hf - tf(HDR_H * 0.3),
        "[ GT / Prediction / Metrics ]",
        **hkw,
    )

    COL1_AX_L = _IX + 0.004
    COL1_AX_FRAC = COL1_W_FRAC - 0.008

    def col1_axes(sec_bot, sec_hf):
        bot = sec_bot + top_pad
        h = max(0.01, sec_hf - tf(HDR_H) - top_pad)
        return fig.add_axes([COL1_AX_L, bot, COL1_AX_FRAC, h])

    ax_in = col1_axes(in_bot, in_hf)
    ax_in.set_facecolor("none")
    ax_in.axis("off")
    _draw_tokens(
        ax_in,
        in_tokens,
        0.002,
        0.998,
        FONTSIZE_IN,
        COL1_AX_FRAC * FIG_W,
        (in_hf - tf(HDR_H) - top_pad) * fig_h,
    )

    ax_rs = col1_axes(rs_bot, rs_hf)
    ax_rs.set_facecolor("none")
    ax_rs.axis("off")
    _draw_tokens(
        ax_rs,
        rs_tokens,
        0.002,
        0.998,
        FONTSIZE_RS,
        COL1_AX_FRAC * FIG_W,
        (rs_hf - tf(HDR_H) - top_pad) * fig_h,
    )

    ax_mt = col1_axes(mt_bot, mt_hf)
    ax_mt.set_facecolor("none")
    ax_mt.axis("off")
    _draw_tokens(
        ax_mt,
        mt_tokens,
        0.002,
        0.998,
        FONTSIZE_MT,
        COL1_AX_FRAC * FIG_W,
        (mt_hf - tf(HDR_H) - top_pad) * fig_h,
    )

    # Separator line at bottom of section 1 (Prompt)
    fig.add_artist(
        mlines.Line2D(
            [_IX, 1.0 - LM],
            [in_bot, in_bot],
            transform=fig.transFigure,
            color=C_SEP,
            linewidth=1.5,
            zorder=0,
        )
    )

    # Col 2: shared panel width for both images so they render at identical visible widths.
    # Pick a width such that BOTH (img aspect) and (overlay aspect) fit within 0.35*fig_h.
    disp_H, disp_W = img.shape
    disp_aspect = disp_H / disp_W
    orig_H, orig_W = doc["image_size_2d"]
    orig_aspect = orig_H / orig_W

    max_h_in = 0.35 * fig_h
    max_aspect = max(disp_aspect, orig_aspect)
    panel_w_in = min(COL2_W_IN, max_h_in / max_aspect)

    panel_w_frac = panel_w_in / FIG_W
    panel_x_frac = COL2_X_FRAC + (COL2_W_FRAC - panel_w_frac) / 2

    img_h_frac = panel_w_in * disp_aspect / fig_h
    ovl_h_frac = panel_w_in * orig_aspect / fig_h

    # Image 1: centered vertically in section 1 (Prompt)
    sec1_center = in_bot + in_hf / 2
    ax_img = fig.add_axes(
        [panel_x_frac, max(0.0, sec1_center - img_h_frac / 2), panel_w_frac, img_h_frac]
    )
    ax_img.imshow(img, cmap="gray", aspect="equal")
    ax_img.axis("off")

    # Image 2: centered vertically in section 2+3 (Response + Metrics)
    sec23_center = (mt_bot + rs_bot + rs_hf) / 2
    ax_ovl = fig.add_axes(
        [
            panel_x_frac,
            max(0.0, sec23_center - ovl_h_frac / 2),
            panel_w_frac,
            ovl_h_frac,
        ]
    )
    _draw_detection_overlay_on_ax(ax_ovl, doc, sample)

    plt.savefig(out_path, dpi=120, facecolor=C_FIG_BG)
    plt.close(fig)
    return out_path


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
        "--removed_samples_dir",
        default=None,
        help="Root dir with per-dataset removed_samples JSON files.",
    )
    parser.add_argument(
        "--removed_samples_filename",
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Filename of the removed-samples JSON within each dataset subdirectory.",
    )
    args = parser.parse_args()

    if args.jsonl and args.model_dir:
        parser.error("--jsonl and --model_dir are mutually exclusive.")
    if not args.jsonl and not args.model_dir:
        parser.error("One of --jsonl or --model_dir is required.")
    if args.limit_per_jsonl is not None and not args.model_dir:
        parser.error("--limit_per_jsonl requires --model_dir.")

    reshape_hw = tuple(args.reshape_hw) if args.reshape_hw else None
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
            if not f.endswith("_proc_acc.jsonl") and not f.endswith("_eq_acc.jsonl")
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

            if args.removed_samples_dir:
                json_path = os.path.join(
                    args.removed_samples_dir,
                    dataset_name,
                    args.removed_samples_filename,
                )
                if os.path.exists(json_path):
                    removed_set = _build_removed_set(json_path)
                    print(
                        f"[Info] Loaded removed-samples filter: {json_path} ({len(removed_set)} entries)"
                    )
                    before = len(samples)
                    samples = [
                        s
                        for s in samples
                        if (
                            _relative_image_file(
                                s["doc"].get("image_file", ""), dataset_name
                            ),
                            s["doc"].get("slice_dim"),
                            s["doc"].get("slice_idx"),
                            int(s["doc"].get("taskID", 0)),
                        )
                        not in removed_set
                    ]
                    print(
                        f"[Info] Filtered out {before - len(samples)} removed samples, {len(samples)} remaining"
                    )

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
                out = _plot_sample(sample, out_path, reshape_hw)
                if out:
                    print(f"  [{i+1}/{len(samples)}] doc_id={doc_id} → {out}")

    else:
        # ── Single-file mode (existing behavior) ──────────────────────────────
        with open(args.jsonl) as f:
            samples = [json.loads(l) for l in f if l.strip()]

        removed_set = None
        if args.removed_samples_dir:
            m = re.search(r"samples_([^_]+)_", os.path.basename(args.jsonl))
            dataset_name = m.group(1) if m else None
            if dataset_name:
                json_path = os.path.join(
                    args.removed_samples_dir,
                    dataset_name,
                    args.removed_samples_filename,
                )
                if os.path.exists(json_path):
                    removed_set = _build_removed_set(json_path)
                    print(
                        f"[Info] Loaded removed-samples filter: {json_path} ({len(removed_set)} entries)"
                    )
                else:
                    print(f"[Warning] Removed-samples file not found: {json_path}")
            else:
                print(
                    "[Warning] Could not extract dataset name from JSONL filename; filtering skipped"
                )

        if removed_set is not None:
            m = re.search(r"samples_([^_]+)_", os.path.basename(args.jsonl))
            dataset_name = m.group(1)
            before = len(samples)
            samples = [
                s
                for s in samples
                if (
                    _relative_image_file(s["doc"].get("image_file", ""), dataset_name),
                    s["doc"].get("slice_dim"),
                    s["doc"].get("slice_idx"),
                    int(s["doc"].get("taskID", 0)),
                )
                not in removed_set
            ]
            print(
                f"[Info] Filtered out {before - len(samples)} removed samples, {len(samples)} remaining"
            )

        if args.sample_ids is not None:
            samples = [s for s in samples if s["doc_id"] in set(args.sample_ids)]
        if args.limit is not None:
            samples = samples[: args.limit]

        print(f"Processing {len(samples)} sample(s) → {args.output_dir}")
        for i, sample in enumerate(samples):
            doc_id = sample["doc_id"]
            out_path = os.path.join(args.output_dir, f"doc_{doc_id:04d}.png")
            out = _plot_sample(sample, out_path, reshape_hw)
            if out:
                print(f"  [{i+1}/{len(samples)}] doc_id={doc_id} → {out}")


if __name__ == "__main__":
    main()
