"""
Visualize per-sample model responses for the AD (Angle/Distance) benchmark task.

Usage:
    python viz_ad_responses.py \
        --jsonl <path/to/parsed/samples.jsonl> \
        --output_dir <output/dir> \
        [--proc_acc_jsonl <path/to/proc_acc.jsonl>] \
        [--eq_acc_jsonl <path/to/eq_acc.jsonl>] \
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
C_LABEL_NAME = "#059669"  # green-600 — landmark / line description
C_IMG_SIZE = "#1D4ED8"
C_PIXEL_SIZE = "#6D28D9"

# 8 coordinate colors — step 1 uses [0:4], step 2 uses [4:8]
_COORD_COLORS = [
    "#DC2626",  # red-600
    "#F97316",  # orange-500
    "#EAB308",  # yellow-500
    "#84CC16",  # lime-400
    "#06B6D4",  # cyan-500
    "#0EA5E9",  # sky-500
    "#A21CAF",  # fuchsia-700
    "#4F46E5",  # indigo-600
]
C_MAJ_LEN = "#0F766E"  # teal-700 — measurement result

# GT overlay colors
C_GT_MAJOR = "#A21CAF"  # fuchsia-700 — GT landmark / line 1
C_GT_MINOR = "#4F46E5"  # indigo-600  — GT line 2 (angle only)

_TAG_COLORS = frozenset({C_THINK, C_REASON, C_STEP_ANS, C_ANS, C_TOOL})
_PROMPT_HEADERS = frozenset(
    {
        "Task:",
        "Additional information:",
        "Format requirement:",
        "Reasoning steps:",
    }
)

_DOT_COLORS = ["#4285F4", "#EA4335", "#FDB813", "#34A853"]

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
COL2_W_FRAC = 0.30
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
_NUM_RE = re.compile(r"\d+\.?\d*")
_TASK_LINE_RE = re.compile(
    r"(Given the input medical image:\s*)(.*?)(,\s*estimate.*)",
    re.DOTALL,
)
_ENTITY_SPLIT_RE = re.compile(r"(\((?:landmark|line)\s+\d+\))")


# ── Color extraction ───────────────────────────────────────────────────────────


def _extract_ad_color_maps(raw_text):
    coord_map, result_map = {}, {}
    pixel_strs, image_strs = set(), set()

    m = re.search(r"<step-1-answer>(.*?)</step-1-answer>", raw_text, re.DOTALL)
    if m:
        for i, n in enumerate(_NUM_RE.findall(m.group(1))[:4]):
            coord_map.setdefault(n, _COORD_COLORS[i])

    m = re.search(r"<step-2-answer>(.*?)</step-2-answer>", raw_text, re.DOTALL)
    if m:
        for i, n in enumerate(_NUM_RE.findall(m.group(1))[:4]):
            coord_map.setdefault(n, _COORD_COLORS[4 + i])

    m = re.search(r"<step-3-reasoning>(.*?)</step-3-reasoning>", raw_text, re.DOTALL)
    if m:
        txt = m.group(1)
        pm = re.search(r"\(pixel_width.*?\)\s*=\s*\(([^)]+)\)", txt) or re.search(
            r"\(pw,\s*ph\)\s*=\s*\(([^)]+)\)", txt
        )
        if pm:
            pixel_strs.update(_NUM_RE.findall(pm.group(1)))
        im = re.search(r"\(image_width.*?\)\s*=\s*\(([^)]+)\)", txt) or re.search(
            r"\(W,\s*H\)\s*=\s*\(([^)]+)\)", txt
        )
        if im:
            image_strs.update(_NUM_RE.findall(im.group(1)))

    m = re.search(r"<step-3-answer>(.*?)</step-3-answer>", raw_text, re.DOTALL)
    if m:
        nums = _NUM_RE.findall(m.group(1))
        if nums:
            result_map[nums[-1]] = C_MAJ_LEN

    # Tool-use fallback: result from <answer> tag when step-3-answer is absent
    if not result_map:
        m = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)
        if m:
            nums = re.findall(r"\d+\.\d+", m.group(1))
            if nums:
                result_map.setdefault(nums[0], C_MAJ_LEN)

    return coord_map, result_map, pixel_strs, image_strs


def _pick_num_color(
    n,
    ctx_step,
    ctx_part,
    in_final_answer,
    coord_map,
    result_map,
    pixel_strs,
    image_strs,
):
    if in_final_answer:
        return result_map.get(n, C_TEXT)
    if ctx_step == 3 and ctx_part == "answer":
        return result_map.get(n, C_TEXT)
    if ctx_step == 3:
        if n in pixel_strs:
            return C_PIXEL_SIZE
        if n in image_strs:
            return C_IMG_SIZE
        if n in coord_map:
            return coord_map[n]
        return result_map.get(n, C_TEXT)
    if ctx_step in (1, 2):
        return coord_map.get(n, C_TEXT)
    return C_TEXT


def _colorize_segment(
    text,
    ctx_step,
    ctx_part,
    in_final_answer,
    coord_map,
    result_map,
    pixel_strs,
    image_strs,
):
    out = []
    for piece in re.split(r"(\d+\.?\d*)", text):
        if not piece:
            continue
        if re.fullmatch(r"\d+\.?\d*", piece):
            c = _pick_num_color(
                piece,
                ctx_step,
                ctx_part,
                in_final_answer,
                coord_map,
                result_map,
                pixel_strs,
                image_strs,
            )
            out.append((piece, c, False))
        else:
            out.append((piece, C_TEXT, False))
    return out


def _add_ad_number_colors(tokens, coord_map, result_map, pixel_strs, image_strs):
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

        if color == C_TEXT and (ctx_step is not None or in_final_answer):
            result.extend(
                _colorize_segment(
                    seg,
                    ctx_step,
                    ctx_part,
                    in_final_answer,
                    coord_map,
                    result_map,
                    pixel_strs,
                    image_strs,
                )
            )
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


def _tokenize_estimate_clause(text):
    """Color the name/description that follows each (landmark N)/(line N) marker."""
    parts = _ENTITY_SPLIT_RE.split(text)
    tokens = []
    for i, part in enumerate(parts):
        if not part:
            continue
        if _ENTITY_SPLIT_RE.fullmatch(part):
            tokens.append((part, C_TEXT, False))  # (landmark N) / (line N) — plain
        elif i > 0 and _ENTITY_SPLIT_RE.fullmatch(parts[i - 1]):
            # Text immediately after a marker — color name/description, keep trailing punct
            # Non-greedy .*? + $ finds the LAST comma or period (trailing separator)
            m = re.match(r"(\s*)(.*?)([,\.]\s*)$", part, re.DOTALL)
            if m:
                if m.group(1):
                    tokens.append((m.group(1), C_TEXT, False))
                if m.group(2):
                    tokens.append((m.group(2), C_LABEL_NAME, False))
                if m.group(3):
                    tokens.append((m.group(3), C_TEXT, False))
            else:
                m2 = re.match(r"(\s*)(.*)", part, re.DOTALL)
                if m2.group(1):
                    tokens.append((m2.group(1), C_TEXT, False))
                if m2.group(2):
                    tokens.append((m2.group(2), C_LABEL_NAME, False))
        else:
            tokens.append((part, C_TEXT, False))
    return tokens


def _tokenize_task_line(line):
    m = _TASK_LINE_RE.match(line)
    if not m:
        return [(line, C_TEXT, False)]
    return [
        (m.group(1), C_TEXT, False),  # "Given the input medical image: "
        (m.group(2), C_IMG_PROMPT, False),  # modality name
        *_tokenize_estimate_clause(
            m.group(3)
        ),  # ", estimate ... (landmark/line N) <name> ..."
    ]


def _tokenize_input(text):
    tokens = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in _PROMPT_HEADERS:
            tokens.append((line, C_TEXT, True))
        elif re.match(r"The image size is", stripped):
            tokens.extend(_tokenize_number_line(line, C_IMG_SIZE))
        elif re.match(r"The pixel size", stripped):
            tokens.extend(_tokenize_number_line(line, C_PIXEL_SIZE))
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


# ── Sample filtering ───────────────────────────────────────────────────────────


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


# ── Overlay (GT + predicted landmarks / lines) ────────────────────────────────


def _draw_ad_overlay_on_ax(ax, doc, proc_acc):
    """
    Draw original-resolution image with GT landmarks/lines (dashed)
    and predicted landmarks/lines (solid) + dots.
    No resize: uses full NIfTI resolution.
    """
    try:
        _, image_2d = _load_nifti_2d(
            doc["image_file"], doc["slice_dim"], doc["slice_idx"]
        )
        image_2d = normalize_img(doc, image_2d)
    except Exception:
        ax.axis("off")
        return

    H, W = image_2d.shape

    def to_col_row(x_rel, y_rel):
        # model lower-left origin → imshow(origin='upper') col/row
        return x_rel * W, H * (1 - y_rel)

    ax.imshow(image_2d, cmap="gray", origin="upper", aspect="equal")

    legend_handles = []

    if proc_acc:
        metric_type = proc_acc.get("metric_type", "distance")

        if metric_type == "distance":
            gt1 = proc_acc.get("gt_landmark_1_wh")
            gt2 = proc_acc.get("gt_landmark_2_wh")
            if gt1 and gt2:
                c1, r1 = to_col_row(gt1[0], gt1[1])
                c2, r2 = to_col_row(gt2[0], gt2[1])
                ax.plot(
                    [c1, c2], [r1, r2], color=C_GT_MAJOR, ls="--", linewidth=4, zorder=2
                )
                ax.scatter(
                    [c1, c2],
                    [r1, r2],
                    color=[_DOT_COLORS[0], _DOT_COLORS[1]],
                    edgecolors="black",
                    s=80,
                    linewidth=1.5,
                    zorder=4,
                )

            pred1 = proc_acc.get("step1_pred")
            pred2 = proc_acc.get("step2_pred")
            if pred1 and len(pred1) == 2 and pred2 and len(pred2) == 2:
                pc1, pr1 = to_col_row(pred1[0], pred1[1])
                pc2, pr2 = to_col_row(pred2[0], pred2[1])
                ax.plot(
                    [pc1, pc2],
                    [pr1, pr2],
                    color="#F37020",
                    ls="-",
                    linewidth=3,
                    zorder=3,
                )
                ax.scatter(
                    [pc1, pc2],
                    [pr1, pr2],
                    color=[_DOT_COLORS[2], _DOT_COLORS[3]],
                    edgecolors="black",
                    s=60,
                    linewidth=1.5,
                    zorder=4,
                )

            legend_handles = [
                mlines.Line2D(
                    [], [], color=C_GT_MAJOR, ls="--", linewidth=3, label="GT landmarks"
                ),
                mlines.Line2D(
                    [], [], color="#F37020", ls="-", linewidth=3, label="Pred landmarks"
                ),
            ]

        else:  # angle
            for (k1, k2), color in [
                (("gt_line1_point1_wh", "gt_line1_point2_wh"), C_GT_MAJOR),
                (("gt_line2_point1_wh", "gt_line2_point2_wh"), C_GT_MINOR),
            ]:
                p1 = proc_acc.get(k1)
                p2 = proc_acc.get(k2)
                if p1 and p2:
                    c1, r1 = to_col_row(p1[0], p1[1])
                    c2, r2 = to_col_row(p2[0], p2[1])
                    ax.plot(
                        [c1, c2], [r1, r2], color=color, ls="--", linewidth=4, zorder=2
                    )

            all_pts = []
            for pred, color in [
                (proc_acc.get("step1_pred"), "#F37020"),
                (proc_acc.get("step2_pred"), "#FBBC05"),
            ]:
                if pred and len(pred) == 4:
                    c1, r1 = to_col_row(pred[0], pred[1])
                    c2, r2 = to_col_row(pred[2], pred[3])
                    ax.plot(
                        [c1, c2], [r1, r2], color=color, ls="-", linewidth=3, zorder=3
                    )
                    all_pts += [(c1, r1), (c2, r2)]
            for j, (cx, ry) in enumerate(all_pts[:4]):
                ax.scatter(
                    cx,
                    ry,
                    color=_DOT_COLORS[j],
                    edgecolors="black",
                    s=60,
                    linewidth=1.5,
                    zorder=4,
                )

            legend_handles = [
                mlines.Line2D(
                    [], [], color=C_GT_MAJOR, ls="--", linewidth=3, label="GT line 1"
                ),
                mlines.Line2D(
                    [], [], color=C_GT_MINOR, ls="--", linewidth=3, label="GT line 2"
                ),
                mlines.Line2D(
                    [], [], color="#F37020", ls="-", linewidth=3, label="Pred line 1"
                ),
                mlines.Line2D(
                    [], [], color="#FBBC05", ls="-", linewidth=3, label="Pred line 2"
                ),
            ]

    if legend_handles:
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


def _parse_prediction(filtered_resps):
    text = (filtered_resps[0] if filtered_resps else "") or ""
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _build_metrics_tokens(sample, proc_acc, eq_acc=None):
    """Return a token list for the metrics section with prediction value bold+colored."""
    target = float(sample["target"])
    pred = _parse_prediction(sample.get("filtered_resps", []))

    metric_type = (proc_acc or {}).get("metric_type") or sample["doc"][
        "biometric_profile"
    ]["metric_type"]
    unit = "mm" if metric_type == "distance" else "°"
    pred_str = f"{pred:.3f} {unit}" if pred is not None else "N/A"

    pred_color = {}
    if pred is not None:
        pred_color[f"{pred:.3f}"] = C_MAJ_LEN

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

    mre = sample.get("avgMRE", {}).get("MRE")
    mre_str = f"{mre:.4f}" if mre is not None else "N/A"

    tokens = []
    tokens += _tokenize_line(f"Ground Truth:  {target:.3f} {unit}")
    tokens += _tokenize_line(f"Prediction:    {pred_str}", highlight=pred_color)

    if proc_acc:
        f4 = lambda v: f"{v:.4f}" if v is not None else "N/A"
        lbl1 = "landmark 1" if metric_type == "distance" else "line 1 endpoints"
        lbl2 = "landmark 2" if metric_type == "distance" else "line 2 endpoints"
        step3_line = f"Measurement Error  ({metric_type}):   MRE = {f4(proc_acc.get('step3_MRE'))}"
        if metric_type == "distance":
            step3_line += f"    nMAE = {f4(proc_acc.get('step3_nMAE'))}"
        for line in [
            f"Localization Error ({lbl1}):  normazlied L2 distance = {f4(proc_acc.get('step1_normL2'))}",
            f"Localization Error ({lbl2}):  normalized L2 distance = {f4(proc_acc.get('step2_normL2'))}",
        ]:
            tokens += _tokenize_line(line)
        if eq_acc:
            ma = eq_acc.get("step3_model_answer")
            pe = eq_acc.get("step3_python_eval")
            ma_str = f4(ma)
            line = (
                f"Arithmetic Error   ({metric_type}):   "
                f"{f4(eq_acc.get('step3_equation_MRE'))} "
                f"(model: {ma_str} | python evaluator: {f4(pe)})"
            )
            tokens += _tokenize_line(
                line, highlight={ma_str: C_MAJ_LEN} if ma is not None else {}
            )
        tokens += _tokenize_line(step3_line)
    else:
        tokens += _tokenize_line("Process accuracy: N/A (no proc_acc file found)")

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


def _plot_sample(sample, proc_acc, out_path, reshape_hw, eq_acc=None):
    doc = sample["doc"]
    doc_id = sample["doc_id"]

    try:
        img = _load_image(doc, reshape_hw)
    except FileNotFoundError as e:
        print(f"  [Warning] Skipping doc_id={doc_id}: {e}")
        return None
    H_img, W_img = img.shape

    raw_resp = sample["resps"][0][0]
    input_text = sample.get("input", "")
    resp_text = _preprocess_response(raw_resp).replace("\n", " ")

    in_tokens = _tokenize_input(input_text)
    in_tokens.append(("\n", C_TEXT, False))
    coord_map, result_map, pixel_strs, image_strs = _extract_ad_color_maps(raw_resp)
    rs_tokens = _add_ad_number_colors(
        _tokenize_resp(resp_text), coord_map, result_map, pixel_strs, image_strs
    )
    mt_tokens = _build_metrics_tokens(sample, proc_acc, eq_acc)

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
    # Pick width such that BOTH (display aspect) and (overlay aspect) fit within 0.35*fig_h.
    disp_aspect = H_img / W_img
    _nii_shape = nib.load(doc["image_file"]).header.get_data_shape()
    _sd = doc["slice_dim"]
    if _sd == 0:
        _ovl_H, _ovl_W = _nii_shape[1], _nii_shape[2]
    elif _sd == 1:
        _ovl_H, _ovl_W = _nii_shape[0], _nii_shape[2]
    else:
        _ovl_H, _ovl_W = _nii_shape[0], _nii_shape[1]
    orig_aspect = _ovl_H / _ovl_W

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
    _draw_ad_overlay_on_ax(ax_ovl, doc, proc_acc)

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
    parser.add_argument("--proc_acc_jsonl", default=None)
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
        help="Root dir with per-dataset removed_samples JSON files (e.g. .../Data/Datasets).",
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

            proc_acc_by_id = {}
            proc_acc_path = jsonl_path.replace(".jsonl", "_proc_acc.jsonl")
            if os.path.exists(proc_acc_path):
                with open(proc_acc_path) as f:
                    for l in f:
                        if l.strip():
                            r = json.loads(l)
                            proc_acc_by_id[r["doc_id"]] = r

            eq_acc_by_id = {}
            eq_acc_path = jsonl_path.replace(".jsonl", "_eq_acc.jsonl")
            if os.path.exists(eq_acc_path):
                with open(eq_acc_path) as f:
                    for l in f:
                        if l.strip():
                            r = json.loads(l)
                            eq_acc_by_id[r["doc_id"]] = r

            print(
                f"[{os.path.basename(jsonl_path)}] {len(samples)} sample(s) → {per_file_out_dir}"
            )
            for i, sample in enumerate(samples):
                doc_id = sample["doc_id"]
                fname = _make_filename(sample, dataset_name)
                out_path = os.path.join(per_file_out_dir, fname)
                out = _plot_sample(
                    sample,
                    proc_acc_by_id.get(doc_id),
                    out_path,
                    reshape_hw,
                    eq_acc=eq_acc_by_id.get(doc_id),
                )
                if out:
                    print(f"  [{i+1}/{len(samples)}] doc_id={doc_id} → {out}")

    else:
        # ── Single-file mode (existing behavior) ──────────────────────────────
        proc_acc_path = args.proc_acc_jsonl
        if proc_acc_path is None:
            candidate = args.jsonl.replace(".jsonl", "_proc_acc.jsonl")
            if os.path.exists(candidate):
                proc_acc_path = candidate
                print(f"[Info] Auto-detected proc_acc: {candidate}")
            else:
                print("[Info] No proc_acc JSONL found; step metrics will show N/A")

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

        proc_acc_by_id = {}
        if proc_acc_path and os.path.exists(proc_acc_path):
            with open(proc_acc_path) as f:
                for l in f:
                    if l.strip():
                        r = json.loads(l)
                        proc_acc_by_id[r["doc_id"]] = r

        if args.sample_ids is not None:
            samples = [s for s in samples if s["doc_id"] in set(args.sample_ids)]
        if args.limit is not None:
            samples = samples[: args.limit]

        print(f"Processing {len(samples)} sample(s) → {args.output_dir}")
        for i, sample in enumerate(samples):
            doc_id = sample["doc_id"]
            out_path = os.path.join(args.output_dir, f"doc_{doc_id:04d}.png")
            out = _plot_sample(sample, proc_acc_by_id.get(doc_id), out_path, reshape_hw)
            if out:
                print(f"  [{i+1}/{len(samples)}] doc_id={doc_id} → {out}")


if __name__ == "__main__":
    main()
