#!/usr/bin/env python3
"""Export the interactive box-ratio curve data for the project webpage.

Reads the same per-ratio summary JSONs that ``viz_detection_performance_per_boxImgRatio.py``
plots, and writes ``<page_dir>/static/js/boxratio-data.js`` defining the
``window.MEDVISION_BOXRATIO`` global that ``static/js/boxratio.js`` consumes. Mirrors
``export_radar_data.py`` (same ``--page_dir`` convention, minified emit, DO-NOT-EDIT header,
fail-loud contract).

This is the interactive twin of ``metrics_boxImgRatio-dotline.pdf``: detection quality as a
function of how much of the image the target occupies, one line per model, against the random-box
baseline. The figure's rules are ported rather than re-invented:

  * one point per ratio bin, placed at the bin MIDPOINT (``extract_ratio_midpoint``), lines sorted
    by that midpoint; the page draws x over 0-0.5, as the figure's ``set_xlim`` does;
  * short bin labels ("<0.05", "0.05~0.10", ...) from the figure's ``boximg_ratio_map`` -- the same
    vocabulary the box-size explorer uses, so the two widgets read as one story;
  * per-model colour AND marker by config position, from the figure's ``base_colors`` /
    ``darker_color`` block: tab10, hues past the first cycle darkened to 80%. That rule is
    byte-identical to ``viz_radar.model_palette``, which ``export_radar_data.py`` uses, so a model
    keeps ONE colour across this page, the radar and every PDF. It is reimplemented here from
    ``configs.radar_model_colors`` only because importing viz_radar would drag matplotlib into an
    otherwise stdlib exporter;
  * the ``random_detection`` baseline keeps the figure's treatment -- black, star marker, dashed --
    and is flagged so boxratio.js can draw it that way; it lives beside the model folders rather
    than under ``<model>/<parsed_dirname>/``, exactly as run_analysis.sh resolves it.

Sample sizes are a property of the evaluation set, not the model, so they are stored ONCE per bin
(what the figure's fourth subplot draws) -- asserted across every model, not assumed.

Example
-------
    PYTHONPATH=src python script/visualization/export_detection_performance_per_boxImgRatio_data.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io \
        --parsed_dirname llm-parsed_gemma-4-31b
"""
import argparse
import json
import math
import os
import sys

import yaml

from medvision_bm.utils.configs import (
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS,
    radar_model_colors,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_RESULTS_DIR = os.path.join(REPO, "Results")
DEFAULT_CONFIG = os.path.join(
    REPO, "script", "analyze", "detection--target-size", "config-detect-boxImgRatio.yaml"
)
TASK_DIR = "MedVision-detect-v2"

# The random-box baseline is produced by analyze_detection_task_boxsize_vs_random.py into its own
# top-level folder, so it has no per-model parsed/ subdirectory (see run_analysis.sh).
BASELINE_KEY = "random_detection"

# Mirrors of viz_detection_performance_per_boxImgRatio.py: bin -> (midpoint, short label). Kept as
# literals because that module imports matplotlib/pandas at module scope, which this stdlib-only
# exporter must not pull in. If the figure's binning changes, change it here too.
BINS = [
    ("Box/Image < 5%", 0.025, "<0.05"),
    ("5% <= Box/Image < 10%", 0.075, "0.05~0.10"),
    ("10% <= Box/Image < 15%", 0.125, "0.10~0.15"),
    ("15% <= Box/Image < 20%", 0.175, "0.15~0.20"),
    ("20% <= Box/Image < 25%", 0.225, "0.20~0.25"),
    ("25% <= Box/Image < 30%", 0.275, "0.25~0.30"),
    ("30% <= Box/Image < 35%", 0.325, "0.30~0.35"),
    ("35% <= Box/Image < 40%", 0.375, "0.35~0.40"),
    ("40% <= Box/Image < 45%", 0.425, "0.40~0.45"),
    ("45% <= Box/Image < 50%", 0.475, "0.45~0.50"),
    ("50% <= Box/Image < 55%", 0.525, "0.50~0.55"),
    ("55% <= Box/Image < 60%", 0.575, "0.55~0.60"),
    ("60% <= Box/Image < 65%", 0.625, "0.60~0.65"),
    ("65% <= Box/Image < 70%", 0.675, "0.65~0.70"),
    ("70% <= Box/Image < 75%", 0.725, "0.70~0.75"),
    ("75% <= Box/Image < 80%", 0.775, "0.75~0.80"),
    ("80% <= Box/Image < 85%", 0.825, "0.80~0.85"),
    ("85% <= Box/Image < 90%", 0.875, "0.85~0.90"),
    ("Box/Image >= 90%", 0.95, ">=0.90"),
]
MARKERS = [
    "o", "s", "D", "p", "d", "^", "v", "<", ">", "X",
    "P", "H", "*", "h", "8", "1", "2", "3", "4", "x",
]
WRAP_DARKEN = 0.8

# The figure panels Recall/Precision/F1; IoU rides along because the same summary carries it and it
# answers the obvious follow-up ("the box is found — how well is it placed?").
METRICS = [
    {"key": "F1", "label": "F1", "default": True},
    {"key": "Recall", "label": "Recall", "default": False},
    {"key": "Precision", "label": "Precision", "default": False},
    {"key": "IoU", "label": "IoU", "default": False},
]
METRIC_KEYS = [m["key"] for m in METRICS]


def model_palette(n):
    """``n`` model colours: tab10 in order, reused hues darkened to ``WRAP_DARKEN``.

    Byte-identical to ``viz_radar.model_palette`` (and to the figure's own base_colors/
    darker_color block): matplotlib's ``to_hex(to_rgb(c) * 0.8)`` rounds each channel the same way
    ``round(v * 0.8)`` on the 0-255 integer does. Note the darkening does NOT compound, so >= 2
    full wraps (21+ models) would repeat a colour; with 18 models every colour is distinct.
    """
    out = []
    for i in range(n):
        base = radar_model_colors[i % len(radar_model_colors)]
        if i < len(radar_model_colors):
            out.append(base)
        else:
            rgb = [int(base[k:k + 2], 16) for k in (1, 3, 5)]
            out.append("#" + "".join("%02x" % int(round(c * WRAP_DARKEN)) for c in rgb))
    return out


def _round(v):
    """Compact float for the JS blob; None/NaN/inf become null (never a bare NaN token)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 4)


def cell_blob(cell):
    """One bin's metrics for the JS blob; None where the summary has no scored entry."""
    if cell is None:
        return None
    out = {k: _round(cell.get(k)) for k in METRIC_KEYS}
    out["SR"] = _round(cell.get("SuccessRate"))
    return out


def summary_path(results_dir, folder, parsed_dirname):
    """Where run_analysis.sh resolves a model's per-ratio summary."""
    task_dir = os.path.join(results_dir, TASK_DIR)
    if folder == BASELINE_KEY:
        return os.path.join(task_dir, folder, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS)
    return os.path.join(
        task_dir, folder, parsed_dirname, SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_DETECT_METRICS
    )


def main():
    ap = argparse.ArgumentParser(description="Export interactive box-ratio data (boxratio-data.js).")
    ap.add_argument("--page_dir", required=True, help="Project page repo (medvision-vlm.github.io).")
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="MedVision Results/ directory.")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="YAML with the model_display_name map.")
    ap.add_argument("--out", default=None, help="Output JS path (default <page_dir>/static/js/boxratio-data.js).")
    ap.add_argument(
        "--parsed_dirname",
        default="parsed",
        help=(
            "Per-model subdirectory to read the summaries from, e.g. llm-parsed_gemma-4-31b. "
            "The random baseline always reads its own top-level folder. Default: parsed."
        ),
    )
    args = ap.parse_args()

    with open(args.config) as fh:
        display_map = (yaml.safe_load(fh) or {}).get("model_display_name") or {}
    if not display_map:
        sys.exit(f"[boxratio] {args.config}: no model_display_name map.")

    folders = list(display_map.keys())
    raw, missing = {}, []
    for folder in folders:
        path = summary_path(args.results_dir, folder, args.parsed_dirname)
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as fh:
            raw[folder] = json.load(fh)
    # Fail loudly: a wrong --parsed_dirname would otherwise ship a blob missing whole models,
    # which reads on the page as "this model has no curve" rather than "wrong source".
    if missing:
        sys.exit(
            f"[boxratio] missing {len(missing)}/{len(folders)} per-ratio summaries under "
            f"'{args.parsed_dirname}'. Generate them first with:\n"
            f"  bash script/analyze/detection--target-size/run_analysis.sh --task_dir "
            f"{os.path.join(args.results_dir, TASK_DIR)} --parsed_dirname {args.parsed_dirname}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    # Bins actually present, in the figure's order. A bin counts only where the summary carries
    # metrics for it, matching the figure's `if "F1" in metrics` guard.
    def scored(folder, name):
        cell = raw[folder].get(name)
        return cell if cell and "F1" in cell else None

    bins = [b for b in BINS if any(scored(f, b[0]) for f in folders)]
    if not bins:
        sys.exit("[boxratio] no scored ratio bins found.")
    unknown = set()
    for folder in folders:
        for name, cell in raw[folder].items():
            if "F1" in cell and name not in {b[0] for b in BINS}:
                unknown.add(name)
    if unknown:
        sys.exit(f"[boxratio] unknown ratio bin(s) {sorted(unknown)}; extend BINS.")

    # Sample size is a property of the evaluation set. Assert that before storing it once.
    reference = folders[0]
    sizes = {}
    for name, _mid, _label in bins:
        for folder in folders:
            cell = scored(folder, name)
            if not cell:
                continue
            n = int(cell.get("num_samples", 0))
            if name in sizes and sizes[name] != n:
                sys.exit(
                    f"[boxratio] num_samples disagrees at '{name}': {display_map[reference]} has "
                    f"{sizes[name]}, {display_map[folder]} has {n} — the summaries are not from "
                    "one Results/ state."
                )
            sizes[name] = n

    colors = model_palette(len(folders))
    models, values = [], {}
    for i, folder in enumerate(folders):
        display = display_map[folder]
        baseline = folder == BASELINE_KEY
        models.append({
            "name": display,
            # The figure gives the random baseline its own treatment; boxratio.js dashes it.
            "color": "#000000" if baseline else colors[i],
            "marker": "*" if baseline else MARKERS[i % len(MARKERS)],
            "baseline": baseline,
        })
        values[display] = [cell_blob(scored(folder, name)) for name, _mid, _label in bins]

    blob = {
        "metrics": METRICS,
        "bins": [
            {"name": name, "label": label, "mid": mid, "n": sizes[name]}
            for name, mid, label in bins
        ],
        "models": models,
        "values": values,
    }

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "boxratio-data.js")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_detection_performance_per_boxImgRatio_data.py - DO NOT EDIT.\n"
        "// Schema: window.MEDVISION_BOXRATIO = {\n"
        "//   metrics:[ {key,label,default} ],\n"
        "//   bins:[ {name,label,mid,n} ],          // mid = bin midpoint, the plotted x\n"
        "//   models:[ {name,color,marker,baseline} ],\n"
        "//   values:{ '<model name>': [ per-bin {F1,Recall,Precision,IoU,SR} | null ] } }\n"
        "// Sample size lives on the bin (shared by every model, asserted at export time).\n"
        "// baseline=true is the random-box control: black, star, dashed, as in the PDF.\n"
    )
    with open(out_path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_BOXRATIO = ")
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[boxratio] wrote {out_path} ({size_kb:.1f} KB) | {len(models)} series, {len(bins)} bins")
    print(f"[boxratio]   bins: {', '.join(b['label'] + '(n=' + str(b['n']) + ')' for b in blob['bins'])}")
    base = [m["name"] for m in models if m["baseline"]]
    print(f"[boxratio]   baseline: {base[0] if base else 'none'}")


if __name__ == "__main__":
    main()
