#!/usr/bin/env python3
"""Export the interactive box-size x label data blob for the project webpage.

Reads the same per-label CSVs that ``viz_detection_sampleSize_per_label_x_boxSize.py`` plots,
and writes ``<page_dir>/static/js/boxsize-data.js`` defining the ``window.MEDVISION_BOXSIZE``
global that ``static/js/boxsize.js`` consumes. Mirrors ``export_radar_data.py`` (same
``--page_dir`` convention, minified emit, DO-NOT-EDIT header, fail-loud contract).

This is the interactive twin of ``fig_detection__metrics-boxSize__*.pdf``: how detection quality
moves with the box-to-image area ratio, per clinical target. All the faithful-port rules come
from the figure module rather than being re-invented:

  * labels kept when their TOTAL sample size across box groups is >= ``MIN_LABEL_SAMPLES`` (100),
    ordered by that total, descending (ties broken by name so the blob is reproducible -- the
    figure's ``sort_values`` is an unstable quicksort);
  * a metric marker is drawn only where the cell has >= ``MIN_CELL_SAMPLES`` (30) samples; that
    threshold ships in the blob and is applied by boxsize.js, so the tooltip can still be honest
    about the n behind a suppressed point;
  * box groups ordered by ``ORDERED_BOX_GROUPS`` and coloured by their index into
    ``configs.radar_model_colors`` (byte-identical to ``plt.cm.tab10.colors``) -- the SAME rule
    and the SAME palette the PDF uses, so a ratio band keeps one colour on page and in print;
  * tumor/lesion labels flagged via the figure's purple-term list (rendered ``#770087``);
  * models carry a matplotlib MARKER name, not a colour: in this figure colour encodes the box
    ratio and shape encodes the model. boxsize.js draws the glyph in the legend chips.

Sample sizes do not depend on the model (every model is scored on the same cases), so they are
stored ONCE per label as a per-box-group vector -- what the figure's stacked bar panel draws.
That is asserted across all configured models rather than assumed.

Example
-------
    PYTHONPATH=src python script/visualization/export_detection_sampleSize_per_label_x_boxSize_data.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io \
        --parsed_dirname llm-parsed_gemma-4-31b
"""
import argparse
import csv
import json
import math
import os
import sys

import yaml

from medvision_bm.utils.configs import (
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_MEAN_METRICS,
    SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS,
    radar_model_colors,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_RESULTS_DIR = os.path.join(REPO, "Results")
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config-detect-sampleSize-per-label-boxSize.yaml")
TASK_DIR = "MedVision-detect-v2"

# Mirrors of viz_detection_sampleSize_per_label_x_boxSize.py. Kept as literals because that module
# imports matplotlib/pandas at module scope, which this stdlib-only exporter must not pull in.
# If the figure's thresholds change, change them here too.
MIN_LABEL_SAMPLES = 100  # SAMPLE_SIZE_THRESHOLD_LABEL
MIN_CELL_SAMPLES = 30  # SAMPLE_SIZE_THRESHOLD_BOX
ORDERED_BOX_GROUPS = [
    "<0.05", "0.05~0.10", "0.10~0.15", "0.15~0.20", "0.20~0.25", "0.25~0.30",
    "0.30~0.35", "0.35~0.40", "0.40~0.45", "0.45~0.50", "0.50~0.55", "0.55~0.60",
    "0.60~0.65", "0.65~0.70", "0.70~0.75", "0.75~0.80", "0.80~0.85", "0.85~0.90",
    ">=0.90",
]
PURPLE_TERMS = [
    "tumor", "cancer", "cyst", "stroke", "lesion", "resection cavity", "edema",
    "metastatic", "vestibular schwannoma",
]
MARKERS = [
    "o", "s", "D", "p", "d", "^", "v", "<", ">", "X",
    "P", "H", "*", "h", "8", "1", "2", "3", "4", "x",
]

# Selectable y-axis metrics. The figure stacks Recall/Precision/F1 as three panels; the page shows
# one at a time behind a segmented control, and adds IoU, which the CSV also carries.
METRICS = [
    {"key": "F1", "label": "F1", "default": True},
    {"key": "Recall", "label": "Recall", "default": False},
    {"key": "Precision", "label": "Precision", "default": False},
    {"key": "IoU", "label": "IoU", "default": False},
]
METRIC_KEYS = [m["key"] for m in METRICS]

# Both label granularities the figure supports via --anatomy_level. "anatomy" leads on the page:
# 26 coarse targets read at a glance, where the 51 fine labels need horizontal scrolling.
LEVELS = [
    {
        "key": "anatomy",
        "label": "Anatomy group",
        "csv": SUMMARY_FILENAME_PER_BOX_IMG_RATIO_GROUP_LABEL_DETECT_MEAN_METRICS,
        "default": True,
    },
    {
        "key": "fine",
        "label": "Fine label",
        "csv": SUMMARY_FILENAME_PER_BOX_IMG_RATIO_FINELABEL_DETECT_MEAN_METRICS,
        "default": False,
    },
]


def is_purple_label(name):
    """Tumor/lesion target? Mirrors the figure's x-tick recolouring block."""
    text = name.lower()
    return any(term in text for term in PURPLE_TERMS)


def _round(v):
    """Compact float for the JS blob; None/NaN/inf become null.

    A cell whose metric is undefined must not reach the browser as a bare ``NaN`` token: that is
    invalid JSON and would evaluate to a NaN coordinate. boxsize.js skips nulls.
    """
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 4)


def read_cells(path):
    """Return ``{(label, box_group): row}`` for one model's per-label CSV."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cells = {}
    for r in rows:
        key = (r["label"], r["box_img_group"])
        if key in cells:
            sys.exit(f"[boxsize] duplicate cell {key} in {path}")
        cells[key] = r
    return cells


def build_level(spec, results_dir, display_map, parsed_dirname):
    """Build one label-granularity level: labels + shared sample sizes + per-model metrics."""
    task_dir = os.path.join(results_dir, TASK_DIR)

    per_model, missing = {}, []
    for folder, display in display_map.items():
        path = os.path.join(task_dir, folder, parsed_dirname, spec["csv"])
        if not os.path.exists(path):
            missing.append(path)
            continue
        per_model[display] = read_cells(path)
    # Fail loudly: a wrong --parsed_dirname would otherwise silently ship a partial blob, which
    # reads on the page as "this model has no data" rather than "wrong source".
    if missing:
        sys.exit(
            f"[boxsize] missing {len(missing)}/{len(display_map)} per-label CSVs under "
            f"'{parsed_dirname}' for level '{spec['key']}'. Generate them first with:\n"
            f"  python -m medvision_bm.benchmark.analyze_detection_task_boxsize "
            f"--task_dir {task_dir} --parsed_dirname {parsed_dirname}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    order = list(display_map.values())
    reference = order[0]

    # Sample size is a property of the evaluation set, not of the model, so it is stored once.
    # Assert that before relying on it -- a mismatch would mean the CSVs came from different runs.
    for display in order[1:]:
        for key, row in per_model[display].items():
            ref_row = per_model[reference].get(key)
            if ref_row is None or int(ref_row["sample_size"]) != int(row["sample_size"]):
                sys.exit(
                    f"[boxsize] sample_size disagrees between '{reference}' and '{display}' at "
                    f"{key} ({spec['key']} level): the CSVs are not from one Results/ state."
                )

    # Labels: total across box groups >= MIN_LABEL_SAMPLES, ordered by that total (desc).
    totals = {}
    for (label, _bg), row in per_model[reference].items():
        totals[label] = totals.get(label, 0) + int(row["sample_size"])
    labels = sorted(
        [lb for lb, t in totals.items() if t >= MIN_LABEL_SAMPLES],
        key=lambda lb: (-totals[lb], lb),
    )
    if not labels:
        sys.exit(f"[boxsize] {spec['key']}: no label reaches {MIN_LABEL_SAMPLES} samples.")

    present = {bg for (_lb, bg) in per_model[reference] if _lb in labels}
    unknown = present - set(ORDERED_BOX_GROUPS)
    if unknown:
        sys.exit(f"[boxsize] unknown box_img_group(s) {sorted(unknown)}; extend ORDERED_BOX_GROUPS.")
    box_groups = [bg for bg in ORDERED_BOX_GROUPS if bg in present]

    label_blobs = [
        {
            "name": lb,
            "purple": bool(is_purple_label(lb)),
            "total": totals[lb],
            "sizes": [
                int(per_model[reference][(lb, bg)]["sample_size"])
                if (lb, bg) in per_model[reference] else 0
                for bg in box_groups
            ],
        }
        for lb in labels
    ]

    values = {}
    for display in order:
        cells = per_model[display]
        values[display] = [
            [
                {k: _round(cells[(lb, bg)].get(k)) for k in METRIC_KEYS}
                if (lb, bg) in cells else None
                for bg in box_groups
            ]
            for lb in labels
        ]

    return {
        "key": spec["key"],
        "label": spec["label"],
        "default": spec["default"],
        "boxGroups": [
            {"name": bg, "color": radar_model_colors[ORDERED_BOX_GROUPS.index(bg) % len(radar_model_colors)]}
            for bg in box_groups
        ],
        "labels": label_blobs,
        "values": values,
    }


def emit_js(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_detection_sampleSize_per_label_x_boxSize_data.py - DO NOT EDIT.\n"
        "// Schema: window.MEDVISION_BOXSIZE = {\n"
        "//   minCellSamples,               // hide a metric marker below this cell n (figure rule)\n"
        "//   minLabelSamples,              // label kept only at/above this total n\n"
        "//   metrics:[ {key,label,default} ],\n"
        "//   models:[ {name, marker} ],    // marker = matplotlib glyph name; colour encodes the\n"
        "//                                 // box ratio, NOT the model (as in the PDF figure)\n"
        "//   levels:[ { key,label,default,\n"
        "//              boxGroups:[ {name,color} ],\n"
        "//              labels:[ {name,purple,total,sizes:[n per boxGroup]} ],\n"
        "//              values:{ '<model name>': [ per-label [ per-boxGroup {F1,Recall,Precision,IoU}\n"
        "//                                                     | null ] ] } } ] }\n"
        "// Sample sizes live on the label (shared by every model, asserted at export time), so a\n"
        "// value cell carries metrics only. null = that model has no row for the cell.\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_BOXSIZE = ")
        # allow_nan=False: raise rather than write a bare NaN/Infinity token (invalid JSON).
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")


def main():
    ap = argparse.ArgumentParser(description="Export interactive box-size data (boxsize-data.js).")
    ap.add_argument("--page_dir", required=True, help="Project page repo (medvision-vlm.github.io).")
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="MedVision Results/ directory.")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="YAML with the model_display_name map.")
    ap.add_argument("--out", default=None, help="Output JS path (default <page_dir>/static/js/boxsize-data.js).")
    ap.add_argument(
        "--parsed_dirname",
        default="parsed",
        help=(
            "Per-model subdirectory to read the per-label CSVs from, e.g. "
            "llm-parsed_gemma-4-31b. Default: parsed."
        ),
    )
    args = ap.parse_args()

    with open(args.config) as fh:
        display_map = (yaml.safe_load(fh) or {}).get("model_display_name") or {}
    if not display_map:
        sys.exit(f"[boxsize] {args.config}: no model_display_name map.")

    models = [
        {"name": display, "marker": MARKERS[i % len(MARKERS)]}
        for i, display in enumerate(display_map.values())
    ]

    blob = {
        "minCellSamples": MIN_CELL_SAMPLES,
        "minLabelSamples": MIN_LABEL_SAMPLES,
        "metrics": METRICS,
        "models": models,
        "levels": [build_level(s, args.results_dir, display_map, args.parsed_dirname) for s in LEVELS],
    }

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "boxsize-data.js")
    emit_js(out_path, blob)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[boxsize] wrote {out_path} ({size_kb:.1f} KB) | {len(models)} models")
    for lv in blob["levels"]:
        drawn = sum(
            1
            for lb_i, lb in enumerate(lv["labels"])
            for bg_i in range(len(lv["boxGroups"]))
            if lb["sizes"][bg_i] >= MIN_CELL_SAMPLES
        )
        print(
            f"[boxsize]   {lv['key']:8s} labels={len(lv['labels'])} "
            f"boxGroups={len(lv['boxGroups'])} cells>={MIN_CELL_SAMPLES}n={drawn}"
        )


if __name__ == "__main__":
    main()
