"""
Used to generate samples for the case viewers in MedVision project page:
*********************************
https://medvision-vlm.github.io/
*********************************

Export MedVision benchmark results as case-study data for the project webpage's
interactive case viewer (medvision-vlm.github.io).

For each task (Detection, Tumor/Lesion size, Angle/Distance) it:
  1. Randomly selects samples per dataset (seeded; default 1234) and reports each
     sample's original metric (Detection: precision/recall/F1; TL & AD: MRE).
  2. Saves ONE overlay PNG per case into <page_dir>/figure/cases/<model>/:
       *_overlay.png — GT vs prediction overlay (pixel-size-scaled renderers in
                       medvision_bm.utils.plot_utils). If the model response can't
                       be parsed into coords/box, an IMAGE-ONLY figure is saved
                       instead and the case is flagged parseFailed.
  3. Builds the COMPLETE Prompt / Response / Metrics text as color-coded HTML panels,
     reusing the viz_*_responses.py tokenizers. On parse failure the Metrics panel
     shows a parsing-failure note instead of (invalid) numbers.
  4. Writes <page_dir>/static/js/cases.js  (window.MEDVISION_CASES, nested
     task -> model -> [cases]).

This MUST run in the MedVision conda env (needs nibabel + matplotlib + the
medvision_bm package). Figures cannot be produced without it.

Multiple models per task: pass "Name=dir" entries (usually via export_webpage_cases.sh).
Example:
    python export_webpage_cases.py \
        --tl_models  "MedVision-V0=/mnt/.../MedVision-TL-v2-CoT/MedVision__fullRFT__...s250" \
                     "Qwen2.5-VL-7B=/mnt/.../MedVision-TL-v2-CoT/Qwen2.5-VL-7B-Instruct" \
        --det_models "MedVision-V0=/mnt/.../MedVision-detect-v2/MedVision__fullRFT__...s250_CoT" \
        --ad_models  "MedVision-V0=/mnt/.../MedVision-AD-v2-CoT/MedVision__fullRFT__...s250" \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io \
        --per_dataset 3 --per_task_max 60
"""

import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
from matplotlib import pyplot as plt

# Make the sibling viz_* scripts importable as libraries (reuse their parsers).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import viz_ad_landmarks as AD  # noqa: E402
import viz_ad_responses as ADR  # noqa: E402
import viz_detection_boxes as DET  # noqa: E402
import viz_detection_responses as DETR  # noqa: E402
import viz_tl_axes as TL  # noqa: E402  (overlay renderers)
import viz_tl_responses as TLR  # noqa: E402  (full prompt/response/metrics tokenizers)

# Reuse the exact removed-sample logic the TL summarizer uses, so the case viewer
# excludes the same v1.0.0 -> v1.1.0 multi-cluster T/L samples that the benchmark drops.
from medvision_bm.benchmark.summarize_TL_task import (  # noqa: E402
    _build_removed_set,
    _relative_image_file,
)
from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (  # noqa: E402
    _load_nifti_2d,
)
from medvision_bm.utils.configs import label_map_rename  # noqa: E402
from medvision_bm.utils.plot_utils import (  # noqa: E402
    plot_ad_on_image,
    plot_detection_on_image,
    plot_tl_axes_on_image,
    save_fig_capped,
)

# ── Generic helpers ───────────────────────────────────────────────────────────


def _dataset_of(jsonl_path):
    m = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    return m.group(1) if m else "unknown"


def _main_jsonls(model_dir):
    """Return sample JSONLs under {model_dir}[/parsed], excluding companions."""
    parsed = os.path.join(model_dir, "parsed")
    base = parsed if os.path.isdir(parsed) else model_dir
    files = sorted(glob.glob(os.path.join(base, "*.jsonl")))
    return [
        f
        for f in files
        if not f.endswith("_proc_acc.jsonl")
        and not f.endswith("_eq_acc.jsonl")
        and not f.endswith("_filtered.jsonl")
    ]


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_companion(jsonl_path, suffix):
    """Load {jsonl}_<suffix>.jsonl into {doc_id: record}, or {} if absent."""
    p = jsonl_path.replace(".jsonl", f"_{suffix}.jsonl")
    out = {}
    if os.path.exists(p):
        for r in _load_jsonl(p):
            out[r["doc_id"]] = r
    return out


def _resp(sample):
    return TL._extract_resp_text(sample["resps"])


def _num(v, n=2):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "N/A"


# ── Title parsing (modality / target / landmark names) ────────────────────────


def _modality(inp):
    m = re.search(
        r"Given the input medical image:\s*(.+?)\s*,\s*(?:estimate|return|measure)",
        inp,
        re.DOTALL,
    )
    return m.group(1).strip() if m else "medical image"


def _short_modality(modality):
    """Compact label for case titles: prefer a parenthetical abbreviation (CT, MRI, …)."""
    abbr = re.findall(r"\(([A-Za-z0-9/ +-]{2,8})\)", modality)
    return abbr[-1] if abbr else re.sub(r"\s+scan$", "", modality)


# ── Full prompt/response/metrics → color-coded HTML (mirrors viz_*_responses.py) ─


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tokens_to_html(tokens, MOD):
    """Convert a viz_*_responses token list [(text, color, bold)] to styled HTML,
    replicating the figures' display rule: tag tokens render as base text but bold;
    colored tokens keep their hex and are bold. Newlines preserved (CSS pre-wrap)."""
    out = []
    for seg, color, bold in tokens:
        if not seg:
            continue
        disp = MOD.C_TEXT if color in MOD._TAG_COLORS else color
        weight = "bold" if (bold or color != MOD.C_TEXT) else "normal"
        txt = _esc(seg)
        if disp == MOD.C_TEXT and weight == "normal":
            out.append(txt)
        else:
            out.append(f'<span style="color:{disp};font-weight:{weight}">{txt}</span>')
    return "".join(out)


def _three_panels(prompt_html, resp_html, metrics_html):
    return [
        {"label": "Prompt", "html": prompt_html},
        {"label": "Response", "html": resp_html},
        {"label": "GT · Prediction · Metrics", "html": metrics_html},
    ]


def _title_tl(sample):
    inp = sample.get("input", "")
    tgt = re.search(
        r"enclosing the\s*(.+?)\s*,\s*in (?:mm|millimeters)", inp, re.DOTALL
    )
    target = tgt.group(1).strip() if tgt else "tumor/lesion"
    return f"{target.capitalize()} size — {_short_modality(_modality(inp))}"


def _title_det(sample):
    inp = sample.get("input", "")
    s = re.search(r"bounding box for the\s*(.+?)\s*[\.\n]", inp, re.DOTALL)
    structure = s.group(1).strip() if s else f"label {sample['doc'].get('label')}"
    return f"{structure.capitalize()} detection — {_short_modality(_modality(inp))}"


def _title_ad(sample):
    inp = sample.get("input", "")
    bp = sample["doc"].get("biometric_profile", {})
    mtype = bp.get("metric_type")
    if isinstance(mtype, list):
        mtype = mtype[0]
    if mtype == "distance":
        lm = re.findall(r"\(landmark \d+\)\s*([^,()\n.]+)", inp)
        names = [n.strip() for n in lm[:2]] or ["landmark 1", "landmark 2"]
        return f"{names[0]}–{names[1]} distance — {_short_modality(_modality(inp))}"
    mk = bp.get("metric_key", "angle")
    mk = mk[0] if isinstance(mk, list) else mk
    return f"{mk} angle — {_short_modality(_modality(inp))}"


def _final_answer(sample, n):
    """Recover the final n numeric answers from filtered_resps — same rule as the
    benchmark pipeline: last n numbers within <answer>…</answer> tag.
    Returns None when the benchmark marked this sample as a parse failure."""
    fr = sample.get("filtered_resps") or [""]
    text = (fr[0] if fr else "") or ""
    if not text:
        return None
    nums = [
        s.replace(",", "")
        for s in re.findall(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", text
        )
    ]
    if len(nums) < n:
        return None
    try:
        return [float(x) for x in nums[-n:]]
    except ValueError:
        return None


def _physical_diagonal(doc):
    """Physical diagonal (native unit, e.g. mm) of the 2D slice from the doc's own
    voxel_size + image_size_2d + slice_dim — mirrors _compute_physical_diagonal for the
    regular (non-scaledPS) tasks these case exports use. None if fields are missing."""
    vs, sd, hw = doc.get("voxel_size"), doc.get("slice_dim"), doc.get("image_size_2d")
    if not vs or sd is None or not hw:
        return None
    try:
        if sd == 0:
            px_h, px_w = float(vs[1]), float(vs[2])
        elif sd == 1:
            px_h, px_w = float(vs[0]), float(vs[2])
        elif sd == 2:
            px_h, px_w = float(vs[0]), float(vs[1])
        else:
            return None
        H, W = hw
        return ((H * px_h) ** 2 + (W * px_w) ** 2) ** 0.5
    except (TypeError, ValueError, IndexError):
        return None


def _f4(v):
    return f"{v:.4f}" if v is not None else "N/A"


def _span(color, text):
    """Inline-styled bold colored span — bypasses _tokens_to_html's _TAG_COLORS filter."""
    return f'<span style="color:{color};font-weight:bold">{text}</span>'


def _partial_profile_tl(sample, pred, gtn):
    """Partial profile HTML for TL: GT / Prediction (C_ANS) / Measurement Error rows.
    Returns HTML directly so the C_ANS prediction color is not filtered to C_TEXT by
    _tokens_to_html's _TAG_COLORS guard (C_ANS is a tag color used for the <answer> tag
    itself, which _tokens_to_html converts to plain bold — bypassing that is the point).
    """
    diag = _physical_diagonal(sample.get("doc", {}))
    mre = lambda p, g: abs(p - g) / abs(g) if g else None
    nmae = lambda p, g: abs(p - g) / diag if diag else None
    c = TLR.C_ANS
    p0 = _span(c, f"{pred[0]:.3f}")
    p1 = _span(c, f"{pred[1]:.3f}")
    lines = [
        f"Ground Truth:    major axis length = {gtn[0]:.3f} mm    minor axis length = {gtn[1]:.3f} mm",
        f"Prediction:      major axis length = {p0} mm    minor axis length = {p1} mm",
        f"Measurement Error  (major axis):   MRE  = {_f4(mre(pred[0], gtn[0]))}    nMAE = {_f4(nmae(pred[0], gtn[0]))}",
        f"Measurement Error  (minor axis):   MRE  = {_f4(mre(pred[1], gtn[1]))}    nMAE = {_f4(nmae(pred[1], gtn[1]))}",
    ]
    return "\n".join(lines)


def _partial_profile_ad(sample, pred, gt):
    """Partial profile HTML for AD: GT / Prediction (C_ANS) / Measurement Error row.
    Returns HTML directly for the same reason as _partial_profile_tl."""
    bp = sample["doc"].get("biometric_profile", {})
    mt = bp.get("metric_type")
    mt = mt[0] if isinstance(mt, list) else mt
    unit = "mm" if mt == "distance" else "°"
    mre = abs(pred - gt) / abs(gt) if gt else None
    line3 = f"Measurement Error  ({mt}):   MRE = {_f4(mre)}"
    if mt == "distance":
        diag = _physical_diagonal(sample.get("doc", {}))
        nmae = abs(pred - gt) / diag if diag else None
        line3 += f"    nMAE = {_f4(nmae)}"
    c = ADR.C_ANS
    lines = [
        f"Ground Truth:  {gt:.3f} {unit}",
        f"Prediction:    {_span(c, f'{pred:.3f}')} {unit}",
        line3,
    ]
    return "\n".join(lines)


def _partial_metrics(task, sample):
    """Partial profile (GT / Prediction / MRE) for non-MedVision TL/AD when the overlay
    WAS drawn (not failed). No warning note. C_ANS colors match the response highlight.
    """
    n = 2 if task == "TL" else 1
    gt = sample["target"]
    gt = json.loads(gt) if isinstance(gt, str) else gt
    if not isinstance(gt, list):
        gt = [gt]
    gtn = [float(g) for g in gt[:n]]
    pred = _final_answer(sample, n)
    if pred is None:
        return "Prediction could not be extracted; metrics unavailable."
    return (
        _partial_profile_tl(sample, pred, gtn)
        if task == "TL"
        else _partial_profile_ad(sample, pred[0], gtn[0])
    )


def _failure_metrics(task, sample):
    """Metrics-panel HTML when the rule-based parser can't extract intermediate
    coordinates. For TL/AD recovers the final <answer> and renders a partial profile
    (GT / Prediction / Measurement Error) — C_ANS colors match the response highlight.
    """
    warn = '<span style="color:#b91c1c;font-weight:700">⚠️Fail to parse reasoning text.</span> '
    if task == "Detection":
        note = warn + (
            "Bounding box coordinates could not be extracted, so the figure shows "
            "the input image with the GT bounding box but without the predicted bounding box."
        )
    else:
        note = warn + (
            "Landmark coordinates could not be extracted, so the figure shows "
            "the input image with the GT overlay but without the model prediction."
        )
    no_answer = (
        '<span style="color:#b91c1c;font-weight:700">⚠️Fail to extract the final answer.</span> '
        "Metrics are unavailable."
    )
    if task == "Detection":
        return note + "\n\n" + no_answer
    n = 2 if task == "TL" else 1
    gt = sample["target"]
    gt = json.loads(gt) if isinstance(gt, str) else gt
    if not isinstance(gt, list):
        gt = [gt]
    pred = _final_answer(sample, n)
    if pred is None:
        return note + "\n\n" + no_answer
    gtn = [float(g) for g in gt[:n]]
    prof = (
        _partial_profile_tl(sample, pred, gtn)
        if task == "TL"
        else _partial_profile_ad(sample, pred[0], gtn[0])
    )
    return note + "\n\n" + prof


# ── Off-the-shelf localization error under an assumed origin (top-left vs lower-left) ──
# The benchmark's localization error (_cal_norm_L2_dist in analyze_process_accuracy_*.py)
# compares the model's reported (x, y) — read in lower-left wh-space — against GT, normalized
# by sqrt(2). Off-the-shelf TL/AD models may instead use the standard top-left image origin;
# we recompute the error under each assumption so the viewer can show both. The final
# size/distance/angle answer is reflection-invariant, so ONLY this line differs by origin.


def _to_wh(pt_arr, H, W):
    """Array-space (dim0=row, dim1=col) → relative wh (x=col/W, y=1-row/H, lower-left)."""
    d0, d1 = pt_arr
    return (d1 / W, 1.0 - d0 / H)


def _nl2(a, b):
    """Normalized L2 in the unit wh-square (matches _cal_norm_L2_dist: /sqrt(2))."""
    return (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) / (2**0.5)


def _origin_wh(p_wh, origin):
    """p_wh = the model's reported (x, y) read as lower-left. Under 'topleft' the model meant
    y measured from the top, so its lower-left-equivalent y is (1 - y)."""
    return (p_wh[0], 1.0 - p_wh[1]) if origin == "topleft" else p_wh


def _localization_line(task, sample, origin):
    """One 'Localization error (normalized L2)' line for an off-the-shelf TL/AD prediction
    under the assumed origin, or None if predicted/GT coords are unavailable."""
    doc = sample["doc"]
    H, W = doc["image_size_2d"]
    f4 = lambda v: f"{v:.4f}" if v is not None else "N/A"
    if task == "TL":
        maj, minr = TL.parse_axis_coords(_resp(sample), (H, W))
        gt_maj, gt_min = TL._extract_gt_axis_pts(doc)
        if maj is None or gt_maj is None:
            return None

        def axis_err(pred_pts, gt_pts):
            p = [_origin_wh(_to_wh(pt, H, W), origin) for pt in pred_pts]
            g = [_to_wh(pt, H, W) for pt in gt_pts]
            return min(
                (_nl2(p[0], g[0]) + _nl2(p[1], g[1])) / 2,
                (_nl2(p[0], g[1]) + _nl2(p[1], g[0])) / 2,
            )  # endpoints unordered

        return (
            f"Localization error (normalized L2):   "
            f"major axis = {f4(axis_err(maj, gt_maj))}    minor axis = {f4(axis_err(minr, gt_min))}"
        )
    # AD
    mtype = doc["biometric_profile"]["metric_type"]
    mtype = mtype[0] if isinstance(mtype, list) else mtype
    pred = (
        AD._parse_dist_preds(_resp(sample), H, W)
        if mtype == "distance"
        else AD._parse_angle_preds(_resp(sample), H, W)
    )
    gt, _ = AD._get_gt_coords(doc)
    if pred is None or gt is None:
        return None
    gpts = (
        [gt["p1"], gt["p2"]]
        if gt["metric_type"] == "distance"
        else [gt["l1p1"], gt["l1p2"], gt["l2p1"], gt["l2p2"]]
    )
    n = min(len(pred), len(gpts))
    if n == 0:
        return None
    avg = (
        sum(
            _nl2(_origin_wh(_to_wh(pred[i], H, W), origin), _to_wh(gpts[i], H, W))
            for i in range(n)
        )
        / n
    )
    return f"Localization error (normalized L2):   {f4(avg)}"


_ORIGIN_LABEL = {"topleft": "top-left", "lowerleft": "lower-left"}


def _offshelf_metrics(task, sample, origin, failed):
    """Metrics HTML for an off-the-shelf TL/AD case under an assumed origin: an origin header
    + the per-origin localization line + the standard partial profile (flip-invariant).
    """
    hdr = (
        '<span style="color:#3b4cc8;font-weight:700">Assumed coordinate origin: '
        f"{_ORIGIN_LABEL.get(origin, origin)}.</span>"
    )
    base = _failure_metrics(task, sample) if failed else _partial_metrics(task, sample)
    loc = None if failed else _localization_line(task, sample, origin)
    return hdr + (("\n" + loc) if loc else "") + "\n\n" + base


# Response HTML for non-MedVision models: plain black EXCEPT numbers inside <answer>,
# which are bold + colored to match the Prediction line in the metrics section.
#
# answer_colors: None  → all <answer> numbers get MOD.C_ANS (TL/AD: uniform orange).
#               [c…]  → positional color per number (Detection: 4 bbox coord colors).
def _answer_colored_resp(raw, MOD, answer_colors=None):
    text = MOD._preprocess_response(raw).replace("\n", " ")
    m = re.search(r"(<answer>)(.*?)(</answer>)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return _esc(text)
    if answer_colors:
        nums = re.findall(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", m.group(2)
        )
        color_map = {
            n: answer_colors[i] for i, n in enumerate(nums[: len(answer_colors)])
        }

        def _color(mm):
            return _span(color_map.get(mm.group(), MOD.C_ANS), mm.group())

    else:

        def _color(mm):
            return _span(MOD.C_ANS, mm.group())

    inner = re.sub(r"-?\d+\.?\d*", _color, _esc(m.group(2)))
    return (
        _esc(text[: m.start()])
        + _esc(m.group(1))
        + inner
        + _esc(m.group(3))
        + _esc(text[m.end() :])
    )


# Prompt HTML: keep Task + Additional information; collapse the long
# "Format requirement:" + "Reasoning steps:" tail into a short omission note.
_OMIT_RE = re.compile(r"\n?[ \t]*Format requirement:", re.I)


def _prompt_html(inp, MOD):
    inp = inp or ""
    m = _OMIT_RE.search(inp)
    head = inp[: m.start()].rstrip() if m else inp
    html = _tokens_to_html(MOD._tokenize_input(head), MOD)
    if m:
        html += (
            '\n<span style="color:#6b7280">&lt;<b>Format Requirement</b> &amp; '
            "<b>Reasoning Instruction</b> omitted here&gt;</span>"
        )
    return html


_AD_DS_PREFIX = {"Ceph-Biometrics-400": "Ceph"}
_ANGLE_KEY_RE = re.compile(r"A-L_(\d+)_(\d+)-L_(\d+)_(\d+)")
_DIST_KEY_RE = re.compile(r"L-(\d+)-(\d+)")


def _format_ad_label(ds, metric_type, metric_key):
    """Format AD target: 'Ceph: a({P2,P5},{P2,P6})' / 'FeTA24: d(P9,P10)'."""
    prefix = _AD_DS_PREFIX.get(ds, ds)
    if metric_type == "angle":
        m = _ANGLE_KEY_RE.search(metric_key or "")
        if m:
            return (
                f"{prefix}: a("
                f"{{P{m.group(1)},P{m.group(2)}}},"
                f"{{P{m.group(3)},P{m.group(4)}}})"
            )
    else:
        m = _DIST_KEY_RE.search(metric_key or "")
        if m:
            return f"{prefix}: d(P{m.group(1)},P{m.group(2)})"
    return f"{prefix}: {metric_type}"


def _target_modality(task, sample, ds=None):
    """(target_label, short_modality) for the case-info line."""
    inp = sample.get("input", "")
    modality = _short_modality(_modality(inp))
    if task == "TL":
        m = re.search(
            r"enclosing the\s*(.+?)\s*,\s*in (?:mm|millimeters)", inp, re.DOTALL
        )
        target = m.group(1).strip() if m else "tumor / lesion"
    elif task == "Detection":
        m = re.search(r"bounding box for the\s*(.+?)\s*[\.\n]", inp, re.DOTALL)
        target = m.group(1).strip() if m else f"label {sample['doc'].get('label')}"
    else:  # AD — use structured label from dataset + biometric_profile
        bp = sample["doc"].get("biometric_profile", {})
        mt = bp.get("metric_type")
        mt = mt[0] if isinstance(mt, list) else mt
        mk = bp.get("metric_key")
        mk = mk[0] if isinstance(mk, list) else mk
        if ds and mk:
            target = _format_ad_label(ds, mt, mk)
        elif mt == "distance":
            lm = re.findall(r"\(landmark \d+\)\s*([^,()\n.]+)", inp)
            names = [n.strip() for n in lm[:2]]
            target = (
                f"distance({names[0]}, {names[1]})" if len(names) == 2 else "distance"
            )
        else:
            target = f"angle: {mk or 'angle'}"
    target = label_map_rename.get(target, target)
    return target, modality


def _blocks_tl(sample, proc, eq, failed=False, colored=True, origin=None):
    raw = _resp(sample)  # depth-robust: handles [[text]] and HuatuoGPT's [[[text]]]
    prompt = _prompt_html(sample.get("input", ""), TLR)
    if colored:
        toks = TLR._add_tl_number_colors(
            TLR._tokenize_resp(TLR._preprocess_response(raw).replace("\n", " ")),
            *TLR._extract_tl_color_maps(raw),
        )
        resp = _tokens_to_html(toks, TLR)
        metrics = (
            _failure_metrics("TL", sample)
            if failed
            else _tokens_to_html(TLR._build_metrics_tokens(sample, proc, eq), TLR)
        )
    else:
        resp = _answer_colored_resp(raw, TLR)
        # Non-MedVision: always use partial profile (C_ANS) so Prediction colors
        # match the <answer> highlight in the response. With origin set (dual-origin
        # off-the-shelf cases) prepend the origin header + per-origin localization line.
        if origin:
            metrics = _offshelf_metrics("TL", sample, origin, failed)
        else:
            metrics = (
                _failure_metrics("TL", sample)
                if failed
                else _partial_metrics("TL", sample)
            )
    return _title_tl(sample), _three_panels(prompt, resp, metrics)


def _blocks_det(sample, proc, eq, failed=False, colored=True):
    raw = _resp(sample)  # depth-robust: handles [[text]] and HuatuoGPT's [[[text]]]
    prompt = _prompt_html(sample.get("input", ""), DETR)
    if colored:
        toks = DETR._add_detection_number_colors(
            DETR._tokenize_resp(DETR._preprocess_response(raw).replace("\n", " ")),
            DETR._extract_detection_color_maps(raw),
        )
        resp = _tokens_to_html(toks, DETR)
    else:
        resp = _answer_colored_resp(raw, DETR, answer_colors=DETR._COORD_COLORS[:4])
    metrics = (
        _failure_metrics("Detection", sample)
        if failed
        else _tokens_to_html(DETR._build_metrics_tokens(sample), DETR)
    )
    return _title_det(sample), _three_panels(prompt, resp, metrics)


def _blocks_ad(sample, proc, eq, failed=False, colored=True, origin=None):
    raw = _resp(sample)  # depth-robust: handles [[text]] and HuatuoGPT's [[[text]]]
    prompt = _prompt_html(sample.get("input", ""), ADR)
    if colored:
        toks = ADR._add_ad_number_colors(
            ADR._tokenize_resp(ADR._preprocess_response(raw).replace("\n", " ")),
            *ADR._extract_ad_color_maps(raw),
        )
        resp = _tokens_to_html(toks, ADR)
        metrics = (
            _failure_metrics("AD", sample)
            if failed
            else _tokens_to_html(ADR._build_metrics_tokens(sample, proc, eq), ADR)
        )
    else:
        resp = _answer_colored_resp(raw, ADR)
        if origin:
            metrics = _offshelf_metrics("AD", sample, origin, failed)
        else:
            metrics = (
                _failure_metrics("AD", sample)
                if failed
                else _partial_metrics("AD", sample)
            )
    return _title_ad(sample), _three_panels(prompt, resp, metrics)


# ── Overlay (Image-2) renderers — reuse canonical pixel-size-scaled plotters ───


def _overlay_tl(sample, out_path, topleft=False):
    doc = sample["doc"]
    H, W = doc["image_size_2d"]
    maj, minr = TL.parse_axis_coords(_resp(sample), (H, W))
    # parse_axis_coords assumes the model used a lower-left origin (idx_dim0 = H*(1-y)).
    # Off-the-shelf models (e.g. Claude-Fable-5) use the standard top-left image origin
    # (y = row/H), so their predicted row is mirrored. Un-flip dim0 (predicted pts only;
    # GT is unaffected) to plot the landmarks where the model actually localized them.
    if topleft and maj is not None:
        maj = [(H - d0, d1) for (d0, d1) in maj]
        minr = [(H - d0, d1) for (d0, d1) in minr]
    psz, img = TL.load_nifti_slice(
        doc["image_file"], doc["slice_dim"], doc["slice_idx"], doc
    )
    mask = None
    mf = doc.get("mask_file")
    if mf and os.path.exists(mf):
        _, m = _load_nifti_2d(mf, doc["slice_dim"], doc["slice_idx"])
        mask = (m == doc["label"]).astype(np.float32)
    gt_maj, gt_min = TL._extract_gt_axis_pts(doc)
    # maj/minr may be None (parse failed) — plot_tl_axes_on_image guards None → GT-only
    plot_tl_axes_on_image(
        image_2d=img,
        pixel_sizes=psz,
        major_axis_pts=maj,
        minor_axis_pts=minr,
        slice_dim=doc["slice_dim"],
        slice_idx=doc["slice_idx"],
        fig_path=out_path,
        mask_2d=mask,
        gt_major_pts=gt_maj,
        gt_minor_pts=gt_min,
    )
    return maj is not None


def _overlay_det(sample, out_path):
    doc = sample["doc"]
    H, W = doc["image_size_2d"]
    pred_norm = DET.parse_box_coords(_resp(sample))
    gt = sample["target"]
    gt_norm = json.loads(gt) if isinstance(gt, str) else gt
    psz, img = TL.load_nifti_slice(
        doc["image_file"], doc["slice_dim"], doc["slice_idx"], doc
    )
    # pred_norm may be None (parse failed) — plot_detection_on_image guards None → GT-only
    plot_detection_on_image(
        image_2d=img,
        pixel_sizes=psz,
        gt_box=DET._box_to_array_space(gt_norm, H, W),
        pred_box=(
            DET._box_to_array_space(pred_norm, H, W) if pred_norm is not None else None
        ),
        slice_dim=doc["slice_dim"],
        slice_idx=doc["slice_idx"],
        fig_path=out_path,
    )
    return pred_norm is not None


def _overlay_ad(sample, out_path, topleft=False):
    doc = sample["doc"]
    H, W = doc["image_size_2d"]
    mtype = doc["biometric_profile"]["metric_type"]
    if isinstance(mtype, list):
        mtype = mtype[0]
    resp = _resp(sample)
    pred = (
        AD._parse_dist_preds(resp, H, W)
        if mtype == "distance"
        else AD._parse_angle_preds(resp, H, W)
    )
    # Same lower-left vs top-left origin mismatch as TL (AD prompts also omit the origin).
    # Un-flip dim0 of the predicted landmarks (pred only; GT untouched) for off-the-shelf
    # models that use the standard top-left origin. See _overlay_tl for the rationale.
    if topleft and pred is not None:
        pred = tuple((H - d0, d1) for (d0, d1) in pred)
    gt_pts, err = AD._get_gt_coords(doc)
    if gt_pts is None:
        # No GT data — fall back to image-only
        _image_only(sample, out_path)
        return False
    psz, img = TL.load_nifti_slice(
        doc["image_file"], doc["slice_dim"], doc["slice_idx"], doc
    )
    # pred may be None (parse failed) — plot_ad_on_image guards None → GT-only
    plot_ad_on_image(
        image_2d=img,
        pixel_sizes=psz,
        metric_type=mtype,
        gt_pts=gt_pts,
        pred_pts=pred,
        slice_dim=doc["slice_dim"],
        slice_idx=doc["slice_idx"],
        fig_path=out_path,
    )
    return pred is not None


# Parse-only success checks (no NIfTI load / no render) — mirror the parse half of
# the _overlay_* functions so --skip_existing can recover the parseFailed flag for a
# case whose overlay PNG is reused from disk.


def _parse_ok_tl(sample):
    H, W = sample["doc"]["image_size_2d"]
    maj, _ = TL.parse_axis_coords(_resp(sample), (H, W))
    return maj is not None


def _parse_ok_det(sample):
    return DET.parse_box_coords(_resp(sample)) is not None


def _parse_ok_ad(sample):
    doc = sample["doc"]
    gt_pts, _ = AD._get_gt_coords(doc)
    if gt_pts is None:
        return False
    H, W = doc["image_size_2d"]
    mtype = doc["biometric_profile"]["metric_type"]
    if isinstance(mtype, list):
        mtype = mtype[0]
    resp = _resp(sample)
    pred = (
        AD._parse_dist_preds(resp, H, W)
        if mtype == "distance"
        else AD._parse_angle_preds(resp, H, W)
    )
    return pred is not None


def _image_only(sample, out_path):
    """Fallback figure when prediction parsing fails: the input slice with the
    same 90° CCW rotation + pixel-size aspect as the overlays, but NO GT/pred."""
    doc = sample["doc"]
    psz, img = TL.load_nifti_slice(
        doc["image_file"], doc["slice_dim"], doc["slice_idx"], doc
    )
    h, w = img.shape
    img_aspect = h / w
    base = 10
    figsize = (base * img_aspect, base) if img_aspect > 1 else (base, base / img_aspect)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img.T, cmap="gray", origin="lower", aspect=psz[1] / psz[0])
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_fig_capped(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── Original per-task metrics (for the log) ────────────────────────────────────


def _pct(v):
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _metric_str(task, s):
    """Original metric for the selected sample: Detection P/R/F1; TL & AD MRE."""
    g = lambda d, k: (s.get(d) or {}).get(k)
    if task == "Detection":
        return (
            f"P={_num(g('Precision','Precision'),3)} "
            f"R={_num(g('Recall','Recall'),3)} "
            f"F1={_num(g('F1','F1'),3)}"
        )
    mre = s.get("avgMRE")
    mre = mre.get("MRE") if isinstance(mre, dict) else mre
    return f"MRE={_pct(mre)}"


# ── Per-task driver ────────────────────────────────────────────────────────────


def _slug(name):
    """Folder-safe slug for a model name."""
    return re.sub(r"[^\w.-]+", "-", name).strip("-") or "model"


def _parse_models(entries):
    """Parse "Name=dir" CLI entries into [(name, dir)]; bare dir -> basename name."""
    out = []
    for e in entries or []:
        if "=" in e:
            name, d = e.split("=", 1)
        else:
            name, d = os.path.basename(e.rstrip("/")), e
        out.append((name.strip(), d.strip()))
    return out


_BUILDERS = {
    "TL": (_blocks_tl, _overlay_tl, _parse_ok_tl),
    "Detection": (_blocks_det, _overlay_det, _parse_ok_det),
    "AD": (_blocks_ad, _overlay_ad, _parse_ok_ad),
}


def _collect_model(
    task, model_dir, removed_samples_dir=None, removed_samples_filename=None
):
    """Map every available sample of one (task, model) by its shared join key.

    Returns {(dataset, doc_id): (sample, proc, eq)}. doc_id is consistent across model
    folders (same dataset, same eval order), so this key lets every model be rendered on
    the SAME samples. No success filtering — a sample the model failed to parse is kept and
    later rendered image-only (parseFailed), so failures show up in the side-by-side too.

    For task == "TL", when removed_samples_dir is given, samples listed in each dataset's
    removed-samples JSON (v1.0.0 -> v1.1.0 multi-cluster exclusions, keyed exactly as in
    summarize_TL_task.py) are dropped from membership for all models."""
    if not model_dir or not os.path.isdir(model_dir):
        print(f"  [{task}] model_dir missing, skipping: {model_dir}")
        return {}

    use_companions = task in ("TL", "AD")
    collected = {}
    removed_cache = {}  # dataset -> frozenset | None
    n_removed = 0
    for jl in _main_jsonls(model_dir):
        ds = _dataset_of(jl)
        proc_map = _load_companion(jl, "proc_acc") if use_companions else {}
        eq_map = _load_companion(jl, "eq_acc") if use_companions else {}
        removed_set = None
        if task == "TL" and removed_samples_dir:
            if ds not in removed_cache:
                jp = os.path.join(removed_samples_dir, ds, removed_samples_filename)
                removed_cache[ds] = (
                    _build_removed_set(jp) if os.path.exists(jp) else None
                )
            removed_set = removed_cache[ds]
        for s in _load_jsonl(jl):
            doc = s.get("doc", {})
            if removed_set is not None:
                rkey = (
                    _relative_image_file(doc.get("image_file", ""), ds),
                    doc.get("slice_dim"),
                    doc.get("slice_idx"),
                    int(doc.get("taskID")),
                )
                if rkey in removed_set:
                    n_removed += 1
                    continue
            # Unique, model-stable join key. doc_id is unique only WITHIN a file; a dataset
            # has several task-config files that reuse doc_ids (e.g. BraTS24 Task05 & Task10
            # both have doc_id 199), so include taskID + slice_dim to disambiguate.
            key = (ds, str(doc.get("taskID")), doc.get("slice_dim"), s["doc_id"])
            collected[key] = (s, proc_map.get(s["doc_id"]), eq_map.get(s["doc_id"]))
    if n_removed:
        print(f"  [{task}] excluded {n_removed} removed-sample(s) from membership")
    return collected


def _select_shared_keys(collected, per_dataset, per_task_max, seed):
    """Pick a SHARED, ordered list of (dataset, doc_id) present in EVERY model, seeded.

    collected: {model_name: {(ds, doc_id): rec}}. Returns the keys common to all models,
    sampled up to per_dataset per dataset then capped to per_task_max. Computed ONCE and
    reused for every model, so each model's case list is built from identical samples in
    identical order (case N == same sample across models)."""
    if not collected:
        return []
    common = set.intersection(*[set(c.keys()) for c in collected.values()])
    if not common:
        return []
    rng = random.Random(seed)
    by_ds = {}
    for key in common:
        by_ds.setdefault(key[0], []).append(key)
    selected = []
    for ds in sorted(by_ds):
        keys = sorted(by_ds[ds])  # deterministic order before sampling
        selected.extend(rng.sample(keys, min(per_dataset, len(keys))))
    if len(selected) > per_task_max:
        selected = rng.sample(selected, per_task_max)
    return selected


def _build_case(
    task,
    ds,
    model_slug,
    model_name,
    sample,
    proc,
    eq,
    cases_dir,
    colored,
    skip_existing=False,
    cases_dirname="cases",
    dual_origin=False,
):
    """Render one model's overlay + Prompt/Response/Metrics panels for a single sample.

    Writes the PNG into cases_dir/<model_slug>/ and returns the case dict, or None if
    rendering raised. On unparseable coordinates the figure is image-only and the case is
    flagged parseFailed (metrics panel shows the parsing-failure note).

    skip_existing: reuse an overlay PNG already on disk instead of re-rendering it;
    the parseFailed flag is then recovered via the parse-only check.
    cases_dirname: subfolder under figure/ used in the manifest's image path.
    dual_origin: off-the-shelf TL/AD — render BOTH the top-left (default) and lower-left
    (alt) interpretations + per-origin localization metrics, and flag the case so the viewer
    shows an origin toggle. Off otherwise (MedVision uses lower-left; Detection states it).
    """
    block_fn, ovl_fn, parse_fn = _BUILDERS[task]
    doc = sample.get("doc", {})
    doc_id = sample["doc_id"]
    # Include taskID + slice_dim so PNGs don't collide across a dataset's task-config files.
    stem = f"{task.lower()}_{ds}_T{doc.get('taskID')}_S{doc.get('slice_dim')}_{doc_id}"
    png = lambda sfx="": os.path.join(cases_dir, model_slug, f"{stem}_overlay{sfx}.png")
    rel = lambda sfx="": f"figure/{cases_dirname}/{model_slug}/{stem}_overlay{sfx}.png"
    tgt, mod = _target_modality(task, sample, ds)
    base = {"target": tgt, "modality": mod, "holdMs": 4200}
    try:
        if dual_origin:
            # Default = top-left (likely model intent); alt = lower-left (as the benchmark scores).
            cached_d = skip_existing and os.path.exists(png(""))
            overlaid = (
                parse_fn(sample) if cached_d else ovl_fn(sample, png(""), topleft=True)
            )
            if overlaid:
                if not (skip_existing and os.path.exists(png("_lowerleft"))):
                    ovl_fn(sample, png("_lowerleft"), topleft=False)
                title, segs_d = block_fn(
                    sample, proc, eq, failed=False, colored=colored, origin="topleft"
                )
                _, segs_a = block_fn(
                    sample, proc, eq, failed=False, colored=colored, origin="lowerleft"
                )
                case = {
                    **base,
                    "title": title,
                    "image": rel(""),
                    "segments": segs_d,
                    "image_alt": rel("_lowerleft"),
                    "segments_alt": segs_a,
                    "originToggle": True,
                    "originLabel": "top-left",
                    "originLabelAlt": "lower-left",
                    "parseFailed": False,
                }
            else:
                # Parse failed → single image-only figure, no toggle.
                title, segs = block_fn(sample, proc, eq, failed=True, colored=colored)
                case = {
                    **base,
                    "title": title,
                    "image": rel(""),
                    "segments": segs,
                    "parseFailed": True,
                }
        else:
            cached = skip_existing and os.path.exists(png(""))
            overlaid = parse_fn(sample) if cached else ovl_fn(sample, png(""))
            title, segs = block_fn(
                sample, proc, eq, failed=not overlaid, colored=colored
            )
            case = {
                **base,
                "title": title,
                "image": rel(""),
                "segments": segs,
                "parseFailed": not overlaid,
            }
    except Exception as e:
        print(f"  [{task}/{model_slug}/{ds}/{doc_id}] render error: {e}")
        return None
    tag = "PARSE-FAIL (GT-only)" if case["parseFailed"] else _metric_str(task, sample)
    if dual_origin and not case["parseFailed"]:
        tag += ", dual-origin"
    print(f"  [{task}/{model_slug}/{ds}/{doc_id}] ok ({tag})")
    return {
        **case,
    }


# ── cases.js emitter ───────────────────────────────────────────────────────────


def _write_cases_js(path, by_task, task_key_suffix=""):
    header = (
        "// Auto-generated by script/visualization/export_webpage_cases.py\n"
        "// Schema: window.MEDVISION_CASES = { Detection:{}, TL:{}, AD:{} }\n"
        "//   each task maps model name -> [ case, ... ]\n"
        "//   per task, every model's list is the SAME samples in the SAME order, so case[i]\n"
        "//   is one underlying sample across models (switching models compares answers).\n"
        "//   case = { title, image:overlay_png, segments:[{label,html}], holdMs, parseFailed }\n"
        "//   segments = the full Prompt / Response / Metrics panels as inline-styled HTML\n"
        "//   (color-coded to match script/visualization/viz_*_responses.py figures).\n"
    )
    # task_key_suffix lets a variant export (e.g. the API pilot) emit distinct keys
    # ("TL-Pilot") into a separate manifest file. Object.assign merges into any
    # MEDVISION_CASES already defined, so the main and pilot manifests coexist
    # regardless of <script> load order.
    body = json.dumps(
        {
            f"Detection{task_key_suffix}": by_task.get("Detection", {}),
            f"TL{task_key_suffix}": by_task.get("TL", {}),
            f"AD{task_key_suffix}": by_task.get("AD", {}),
        },
        indent=2,
        ensure_ascii=False,
    )
    with open(path, "w") as f:
        f.write(
            header
            + "\nwindow.MEDVISION_CASES = Object.assign(window.MEDVISION_CASES || {}, "
            + body
            + ");\n"
        )
    print(f"[cases.js] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    # Multi-model: each entry is "ModelName=/path/to/model_dir" (repeatable).
    ap.add_argument("--det_models", nargs="*", default=[], metavar="NAME=DIR")
    ap.add_argument("--tl_models", nargs="*", default=[], metavar="NAME=DIR")
    ap.add_argument("--ad_models", nargs="*", default=[], metavar="NAME=DIR")
    # Back-compat single-model dirs (treated as model "MedVision-V0").
    ap.add_argument("--det_dir")
    ap.add_argument("--tl_dir")
    ap.add_argument("--ad_dir")
    ap.add_argument(
        "--page_dir", required=True, help="Project page repo (medvision-vlm.github.io)"
    )
    ap.add_argument(
        "--per_dataset",
        type=int,
        default=None,
        help="Samples per dataset (fallback when per-task arg not given).",
    )
    ap.add_argument(
        "--per_dataset_det",
        type=int,
        default=10,
        help="Samples per dataset for Detection (default 10).",
    )
    ap.add_argument(
        "--per_dataset_tl",
        type=int,
        default=20,
        help="Samples per dataset for TL (default 20).",
    )
    ap.add_argument(
        "--per_dataset_ad",
        type=int,
        default=20,
        help="Samples per dataset for AD (default 20).",
    )
    ap.add_argument("--per_task_max", type=int, default=50)
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sample selection (default 1234)",
    )
    # TL-only sample filtering, mirroring summarize_TL_task.py. When given, T/L
    # samples listed in <removed_samples_dir>/<dataset>/<removed_samples_filename>
    # are excluded before selection. Off by default (no dir -> no filtering).
    ap.add_argument(
        "--removed_samples_dir",
        default=None,
        help="Root dir with per-dataset removed-samples JSON (e.g. .../Data/Datasets). "
        "Enables TL removed-sample exclusion when set.",
    )
    ap.add_argument(
        "--removed_samples_filename",
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Removed-samples JSON filename within each dataset subdirectory.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Append mode: skip rendering overlay PNGs that already exist on disk "
        "(reuse the file); default overwrites every PNG. cases.js is always "
        "rebuilt in full. Existing PNGs are trusted — if model JSONLs were "
        "re-generated since the last export, run without this flag.",
    )
    # Variant export (e.g. the API pilot viewer): write PNGs + manifest to separate
    # locations and emit distinct task keys, so the main 13-model viewer's assets are
    # never touched. Defaults reproduce the original behavior exactly.
    ap.add_argument(
        "--cases_dirname",
        default="cases",
        help="Subfolder under <page_dir>/figure/ for overlay PNGs (default 'cases'). "
        "Also the orphan-cleanup scope and the manifest image-path prefix.",
    )
    ap.add_argument(
        "--cases_js",
        default="static/js/cases.js",
        help="Manifest output path relative to <page_dir> (default static/js/cases.js).",
    )
    ap.add_argument(
        "--task_key_suffix",
        default="",
        help="Suffix appended to manifest task keys, e.g. '-Pilot' -> 'TL-Pilot' "
        "(default '' = Detection/TL/AD).",
    )
    ap.add_argument(
        "--nonmedvision_topleft",
        action="store_true",
        help="Dual-origin mode for off-the-shelf (non-MedVision) TL & AD cases: render "
        "BOTH a top-left (default) and a lower-left (alt) overlay + per-origin "
        "localization metrics, and flag the case so the viewer shows an origin "
        "toggle. The TL/AD prompts omit the origin, so off-the-shelf models may use "
        "either; the benchmark/GT/MedVision-V0 use lower-left. Detection is excluded "
        "(its prompt states the origin). Final size/distance/angle answers are "
        "reflection-invariant, so only the localization line differs by origin.",
    )
    args = ap.parse_args()

    cases_dir = os.path.join(args.page_dir, "figure", args.cases_dirname)
    os.makedirs(cases_dir, exist_ok=True)
    # NOTE: stale overlays are NOT deleted up-front. Doing so would wipe the page's
    # figures before regeneration, so an interrupted/failed run would leave the old
    # cases.js pointing at deleted PNGs -> "figure pending". Instead we regenerate
    # first and remove only orphans at the end (see below).

    task_models = {
        "Detection": _parse_models(args.det_models)
        or ([("MedVision-V0", args.det_dir)] if args.det_dir else []),
        "TL": _parse_models(args.tl_models)
        or ([("MedVision-V0", args.tl_dir)] if args.tl_dir else []),
        "AD": _parse_models(args.ad_models)
        or ([("MedVision-V0", args.ad_dir)] if args.ad_dir else []),
    }

    # Per-task sample quota: explicit per-task arg used unless global --per_dataset is given.
    per_dataset_by_task = {
        "Detection": (
            args.per_dataset_det if args.per_dataset is None else args.per_dataset
        ),
        "TL": args.per_dataset_tl if args.per_dataset is None else args.per_dataset,
        "AD": args.per_dataset_ad if args.per_dataset is None else args.per_dataset,
    }

    by_task = {}
    for task in ("Detection", "TL", "AD"):
        print(f"=== {task} ===")
        # Collect every model's available samples keyed by (dataset, doc_id).
        collected = {}
        for name, d in task_models[task]:
            print(f"--- collect: {name} ({d}) ---")
            cm = _collect_model(
                task,
                d,
                removed_samples_dir=args.removed_samples_dir,
                removed_samples_filename=args.removed_samples_filename,
            )
            if cm:
                collected[name] = cm

        # One shared, seeded sample set, rendered by EVERY model in the same order.
        selected = _select_shared_keys(
            collected, per_dataset_by_task[task], args.per_task_max, args.seed
        )
        print(
            f"  [{task}] {len(selected)} shared case(s) across {len(collected)} model(s)"
        )

        # Render each model on the shared keys; then keep only keys that rendered for
        # ALL models so the per-model lists stay equal-length and index-aligned.
        rendered = {name: {} for name in collected}  # name -> {key: case}
        for name in collected:
            colored = (
                "medvision" in name.lower()
            )  # color response only for MedVision-V0
            # Off-the-shelf (non-MedVision) TL/AD: the prompt omits the origin, so render both
            # top-left and lower-left interpretations + a toggle (--nonmedvision_topleft).
            # MedVision (lower-left) and Detection (origin stated) keep a single version.
            dual_origin = (
                args.nonmedvision_topleft and not colored and task in ("TL", "AD")
            )
            for key in selected:
                ds = key[0]
                sample, proc, eq = collected[name][key]
                case = _build_case(
                    task,
                    ds,
                    _slug(name),
                    name,
                    sample,
                    proc,
                    eq,
                    cases_dir,
                    colored,
                    skip_existing=args.skip_existing,
                    cases_dirname=args.cases_dirname,
                    dual_origin=dual_origin,
                )
                if case:
                    rendered[name][key] = case
        good = [k for k in selected if all(k in rendered[name] for name in collected)]
        dropped = len(selected) - len(good)
        if dropped:
            print(
                f"  [{task}] dropped {dropped} case(s) that failed to render for some model"
            )

        models = {}
        for name in collected:  # preserve task-defined model order
            lst = [rendered[name][k] for k in good]
            if lst:
                models[name] = lst
        by_task[task] = models

    _write_cases_js(
        os.path.join(args.page_dir, args.cases_js),
        by_task,
        task_key_suffix=args.task_key_suffix,
    )

    # Deferred cleanup: now that the fresh cases.js is written, every referenced PNG
    # exists on disk; remove only overlays NOT referenced by it (orphans from prior
    # runs / dropped or unselected samples). This never leaves a dangling reference.
    referenced = {
        os.path.normpath(os.path.join(args.page_dir, c[k]))
        for m in by_task.values()
        for cases in m.values()
        for c in cases
        for k in ("image", "image_alt")
        if c.get(k)
    }
    existing = glob.glob(os.path.join(cases_dir, "*_overlay*.png")) + glob.glob(
        os.path.join(cases_dir, "*", "*_overlay*.png")
    )
    orphans = [p for p in existing if os.path.normpath(p) not in referenced]
    for p in orphans:
        os.remove(p)
    if orphans:
        print(f"[cleanup] removed {len(orphans)} orphan overlay PNG(s)")

    total = sum(len(cs) for m in by_task.values() for cs in m.values())
    summary = ", ".join(
        f"{t}={sum(len(cs) for cs in by_task[t].values())}c/{len(by_task[t])}m"
        for t in ("Detection", "TL", "AD")
    )
    print(f"\nDone: {total} cases  ({summary})")


if __name__ == "__main__":
    main()
