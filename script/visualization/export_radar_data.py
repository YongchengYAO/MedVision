#!/usr/bin/env python3
"""Export the interactive radar-chart data blob for the project webpage.

Reads the same per-model summary JSONs that ``viz_radar.py`` /
``viz_radar_batch_leaderboard.sh`` plot, and writes ``<page_dir>/static/js/radar-data.js``
defining the ``window.MEDVISION_RADAR`` global that ``static/js/radar.js`` consumes.
Mirrors ``export_explorer_data.py`` (same ``--page_dir`` convention, minified emit,
DO-NOT-EDIT header, fail-loud contract).

The three task radars are the interactive twin of the static Figure 3: each spoke is a
clinical target, the radius is one metric, and every model is a separate line trace. All the
faithful-port rules are reused directly from ``viz_radar.py`` rather than re-implemented:

  * spokes = the sorted INTERSECTION of targets across all models
    (``EXCLUDED_KEYS`` dropped for Detection/TL; Detection also drops targets with
    ``num_samples < MINIMUM_GROUP_SIZE``) — ``load_model_metrics`` + set intersection;
  * tumor/lesion spokes flagged via ``is_purple_label`` (rendered ``#770087`` bold on the page);
  * A/D targets split into an Angle group and a Distance group via ``split_ad_labels``,
    names abbreviated via ``abbreviate_label_name`` (``..._distance_L-1-2`` -> ``Ceph: d(P1,P2)``);
  * per-model colours from ``viz_radar.model_palette(N)`` — the SAME palette the PDF radar and the
    angle box plot use (tab10, reused hues darkened to 80%), so a model has one colour across the
    web page and the paper figures. Assigned by the shared model order (identical across the three
    task configs), so a model keeps one colour everywhere it appears.

The RADIUS transform (``1 - clamp(v,0,1)`` for MRE/MAE, raw otherwise) is applied in the browser,
not here — this blob stores the ORIGINAL metric values so the hover tooltip can show them.

Example
-------
    PYTHONPATH=src python script/visualization/export_radar_data.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io
"""
import argparse
import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# viz_radar.py is a sibling script (not an installed module) — reuse its helpers directly.
sys.path.insert(0, SCRIPT_DIR)
import viz_radar  # noqa: E402  (load_config, load_model_metrics, is_purple_label, ...)

from medvision_bm.utils.configs import (  # noqa: E402
    MINIMUM_GROUP_SIZE,
    SUMMARY_FILENAME_AD_METRICS,
    SUMMARY_FILENAME_DETECT_METRICS,
    SUMMARY_FILENAME_TL_METRICS,
)

# ── static config ────────────────────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_DIR = os.path.join(REPO, "Results")

# One entry per task section on the project page. ``metrics`` are the selectable spoke metrics;
# ``higher_better=False`` means the browser inverts the radius (outer ring = best).
TASKS = {
    "Detection": {
        "task_dir": "MedVision-detect-v2",
        "config": "config-detect-CoT.yaml",
        "summary": SUMMARY_FILENAME_DETECT_METRICS,
        "min_samples": MINIMUM_GROUP_SIZE,
        "split_ad": False,
        "metrics": [
            {"key": "F1", "label": "F1", "higher_better": True, "default": True},
            {"key": "Recall", "label": "Recall", "higher_better": True, "default": False},
            {"key": "Precision", "label": "Precision", "higher_better": True, "default": False},
            {"key": "IoU", "label": "IoU", "higher_better": True, "default": False},
        ],
    },
    "TL": {
        "task_dir": "MedVision-TL-v2-CoT",
        "config": "config-TL-CoT.yaml",
        "summary": SUMMARY_FILENAME_TL_METRICS.replace(".json", "_filtered.json"),
        "min_samples": None,
        "split_ad": False,
        "metrics": [
            {"key": "avgMRE", "label": "MRE", "higher_better": False, "default": True},
        ],
    },
    "AD": {
        "task_dir": "MedVision-AD-v2-CoT",
        "config": "config-AD-CoT.yaml",
        "summary": SUMMARY_FILENAME_AD_METRICS,
        "min_samples": None,
        "split_ad": True,
        "metrics": [
            {"key": "avgMRE", "label": "MRE", "higher_better": False, "default": True},
        ],
    },
    # Pilot study: MedVision-V0 vs frontier API models (Claude-Fable-5, Gemini-3.1-Pro) on the
    # limit100 T/L subset (750 samples). Same shape as TL — clinical-target spokes, MRE — just a
    # 3-model set with its own summary filename. task_type "TL" so miscellaneous/others are excluded.
    "TL-Pilot": {
        "task_dir": "MedVision-TL-CoT-limit100",
        "config": "config-TL-pilot-CoT.yaml",
        "summary": "summary_metrics_TL_Task_filtered_limit100.json",
        "min_samples": None,
        "split_ad": False,
        "metrics": [
            {"key": "avgMRE", "label": "MRE", "higher_better": False, "default": True},
        ],
    },
}


def _round(v):
    """Compact float for the JS blob (None/NaN/inf pass through as null).

    A target the model failed on every sample (SuccessRate 0) has avgMRE/avgMAE = NaN
    (mean over zero parsed values). Emit null so the browser can skip it — leaving a gap
    like matplotlib — rather than a bare ``NaN`` token that would eval to a NaN coordinate.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 4)


def _load_task_frames(task_dir, config_path, summary, metric_keys, min_samples, task_type):
    """Return ``(model_display_order, {display_name: {target: row}})`` for one task.

    Reads every model listed in the config's ``model_display_name`` map, extracts the requested
    metrics (+ SuccessRate + num_samples) via ``viz_radar.load_model_metrics``, and applies the
    Detection minimum-sample filter. Fails loudly on any missing model directory / JSON so the
    shipped blob can never contain a partial task.
    """
    config = viz_radar.load_config(config_path)
    display_map = config.get("model_display_name", {})
    if not display_map:
        sys.exit(f"[radar] {config_path}: no model_display_name map.")

    extract = list(metric_keys) + ["SuccessRate"]
    order, frames = [], {}
    for model_key, display in display_map.items():
        parsed_dir = os.path.join(task_dir, model_key, "parsed")
        json_file = os.path.join(parsed_dir, summary)
        if not os.path.exists(json_file):
            sys.exit(f"[radar] missing summary for '{display}': {json_file}")
        df = viz_radar.load_model_metrics(json_file, extract, task_type=task_type)
        if min_samples is not None:
            df = df[df["num_samples"] >= min_samples]
        if len(df) == 0:
            sys.exit(f"[radar] '{display}' has no targets after filtering: {json_file}")
        order.append(display)
        frames[display] = df.set_index("Target").to_dict("index")
    return order, frames


def build_task(spec, results_dir, task_type):
    """Build one task's blob: metric list + spoke groups + per-model per-spoke values."""
    task_dir = os.path.join(results_dir, spec["task_dir"])
    config_path = os.path.join(SCRIPT_DIR, spec["config"])
    metric_keys = [m["key"] for m in spec["metrics"]]

    order, frames = _load_task_frames(
        task_dir, config_path, spec["summary"], metric_keys, spec["min_samples"], task_type
    )

    # Spokes = sorted intersection of targets across ALL models (faithful to viz_radar).
    common = set.intersection(*[set(rows.keys()) for rows in frames.values()])
    common = sorted(common)
    if not common:
        sys.exit(f"[radar] {task_type}: no common targets across models.")

    if spec["split_ad"]:
        label_groups = viz_radar.split_ad_labels(common)
        # Webpage display order: Distance before Angle (split_ad_labels yields Angle first). Any
        # residual "Other" group stays last. This drives the "Task" control's button order.
        _order = {"Distance": 0, "Angle": 1}
        label_groups = sorted(label_groups, key=lambda grp: _order.get(grp[0], 99))
    else:
        label_groups = [(None, common)]

    groups = []
    for group_name, labels in label_groups:
        spokes = [
            {
                "n": i + 1,
                "name": viz_radar.abbreviate_label_name(lbl),
                "full": lbl,
                "purple": bool(viz_radar.is_purple_label(lbl)),
            }
            for i, lbl in enumerate(labels)
        ]
        values = {}
        for display in order:
            rows = frames[display]
            per_spoke = []
            for lbl in labels:
                row = rows[lbl]
                cell = {k: _round(row.get(k)) for k in metric_keys}
                cell["SR"] = _round(row.get("SuccessRate"))
                cell["n"] = int(row.get("num_samples", 0))
                per_spoke.append(cell)
            values[display] = per_spoke
        groups.append({"name": group_name, "spokes": spokes, "values": values})

    return {"metrics": spec["metrics"], "groups": groups}, order


def emit_js(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_radar_data.py — DO NOT EDIT.\n"
        "// Schema: window.MEDVISION_RADAR = {\n"
        "//   models:[ {name, color} ],   // UNION of all tasks' models; a task may use a subset\n"
        "//   tasks:{ '<Detection|TL|AD|TL-Pilot>': {\n"
        "//     metrics:[ {key,label,higher_better,default} ],   // higher_better=false -> radius inverted\n"
        "//     groups:[ { name, spokes:[ {n,name,full,purple} ],\n"
        "//               values:{ '<model name>': [ per-spoke {<metric>..,SR,n} ] } } ] } } }\n"
        "// Radius transform (applied in radar.js, NOT here): r = higher_better ? clamp(v,0,1)\n"
        "//   : 1 - clamp(v,0,1). Stored values are ORIGINAL (avgMRE is a ratio, tooltip shows v*100%).\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_RADAR = ")
        # allow_nan=False: raise rather than write a bare NaN/Infinity token (invalid JSON, and it
        # would eval to a NaN coordinate in the browser). _round already nulls non-finite metrics.
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")


def main():
    ap = argparse.ArgumentParser(description="Export interactive radar data (radar-data.js).")
    ap.add_argument("--page_dir", required=True, help="Project page repo (medvision-vlm.github.io).")
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="MedVision Results/ directory.")
    ap.add_argument("--out", default=None, help="Output JS path (default <page_dir>/static/js/radar-data.js).")
    args = ap.parse_args()

    task_types = {"Detection": "Detection", "TL": "TL", "AD": "AD", "TL-Pilot": "TL"}
    tasks, orders = {}, {}
    for task, spec in TASKS.items():
        tasks[task], orders[task] = build_task(spec, args.results_dir, task_types[task])

    # Model identity is the display name. The three full-benchmark tasks must share the same 18-model
    # set (assert it); the pilot deliberately compares a different, smaller set. So build the UNION,
    # keeping the 18-model order first and appending any extras (e.g. the API pilot models). Colours
    # are assigned over the union, so a model keeps ONE colour everywhere it appears (radar.js shows
    # only the models present in each task).
    base = orders["Detection"]
    for task in ("TL", "AD"):
        if set(orders[task]) != set(base):
            sys.exit(
                f"[radar] model set mismatch: Detection {sorted(set(base))} vs "
                f"{task} {sorted(set(orders[task]))}"
            )
    union = list(base)
    for od in orders.values():
        for name in od:
            if name not in union:
                union.append(name)
    colors = viz_radar.model_palette(len(union))
    models = [{"name": name, "color": colors[i]} for i, name in enumerate(union)]

    blob = {"models": models, "tasks": tasks}

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "radar-data.js")
    emit_js(out_path, blob)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[radar] wrote {out_path} ({size_kb:.1f} KB) | {len(models)} models")
    for task, t in tasks.items():
        spoke_desc = ", ".join(
            f"{g['name'] or 'all'}={len(g['spokes'])}" for g in t["groups"]
        )
        print(f"[radar]   {task:9s} metrics={[m['key'] for m in t['metrics']]} spokes: {spoke_desc}")


if __name__ == "__main__":
    main()
