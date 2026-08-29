#!/usr/bin/env python3
"""Export the per-spoke distribution blob that overlays violins on the webpage radars.

Writes ``<page_dir>/static/js/violin-data.js`` defining ``window.MEDVISION_VIOLIN``, which
``static/js/radar.js`` reads to draw a violin + box plot along every radar spoke — the
interactive twin of ``viz_radar.py --verbose_model``, generalised from one named model to
every model in the task config.

Relationship to ``export_radar_data.py``
----------------------------------------
That script emits ONE number per (model, spoke, metric) — the summary value the radar line
passes through. This one emits the DISTRIBUTION behind that number: the per-sample values
are re-read from the ``*_samples_*.jsonl`` records, reduced to a KDE curve plus box-plot
quantiles, and stored per (model, spoke, metric).

Spoke identity is not re-derived here. ``export_radar_data.build_task`` is called directly
and its ``groups[i].spokes[j].full`` labels are reused verbatim, so violin spoke *j* is
always radar spoke *j* — the two blobs cannot drift out of alignment.

Plotted space
-------------
Values are stored ALREADY TRANSFORMED into the radar's plotted space, exactly as
``viz_radar.plot_violin_on_spoke`` does before calling ``gaussian_kde``::

    plotted = clamp(v, 0, 1)  if higher_better else  1 - clamp(v, 0, 1)

so ``radar.js`` maps a stored coordinate to pixels with its existing ``pixelR`` and needs no
metric-specific logic. Densities are sampled on a fixed grid of ``GRID`` points spanning
plotted ``[0, 1]`` and normalised so the peak is 100 — the renderer only ever needs the
SHAPE, since angular half-width is ``(density / max) * 0.3 * (2*pi/N)``.

Cost note
---------
Detection alone is ~85 MB of JSONL per model. The per-sample reader here extracts every
metric in ONE pass over each file; calling ``viz_radar.load_per_sample_values`` once per
metric would re-read the same ~1.5 GB four times.

Example
-------
    PYTHONPATH=src python script/visualization/export_violin_data.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Sibling scripts (not installed modules) — reuse their helpers rather than re-implementing.
sys.path.insert(0, SCRIPT_DIR)
import viz_radar  # noqa: E402
import export_radar_data as erd  # noqa: E402

# Number of density samples across plotted [0, 1]. viz_radar uses 200 for the PDF; the web
# violin spans ~205 px of radius, so 33 points (~6 px apart) is visually indistinguishable
# once the polygon is drawn, at a sixth of the blob size.
GRID = 33

# Every radar on the page, so the violin control appears on all of them rather than on three of
# four. TL-Pilot rides along cheaply: three models over the 750-sample subset. radar.js only
# offers the control for a task present in this blob, so dropping one here just disables it there.
TASKS = ["Detection", "TL", "AD", "TL-Pilot"]

# Detection JSONL stores IoU under "avgIoU" while summaries/--metrics_list use "IoU"
# (same aliasing viz_radar.load_per_sample_values applies).
_DETECT_KEY_ALIASES = {"IoU": "avgIoU"}


def _sample_label(doc, task_type):
    """Reconstruct a sample's target label, matching viz_radar.load_per_sample_values."""
    if task_type == "TL":
        return viz_radar._reconstruct_tl_label(doc)
    if task_type == "Detection":
        return viz_radar._reconstruct_detect_label(doc)
    bp = doc.get("biometric_profile")
    if bp is None or "metric_key" not in bp:
        return None
    return f"{doc['dataset_name']}_{bp['metric_type']}_{bp['metric_key']}"


def _scalar(raw):
    """Pull the metric scalar out of a JSONL field, or None if the sample did not succeed.

    Mirrors viz_radar.load_per_sample_values: AD/TL dicts carry an explicit ``success``
    flag, Detection dicts do not (failures are encoded as 0.0).
    """
    if isinstance(raw, dict):
        if "success" in raw and not raw["success"]:
            return None
        return next(
            (v for k, v in raw.items() if k != "success" and isinstance(v, (int, float))),
            None,
        )
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def load_per_sample_multi(parsed_dir, metric_names, keep_labels, task_type):
    """Return ``{label: {metric: [values]}}`` in a single pass over the model's JSONL files.

    Single-pass is the whole point: see the cost note in the module docstring.
    """
    keep = set(keep_labels)
    out = {lbl: {m: [] for m in metric_names} for lbl in keep}
    jsonl_key = {
        m: (_DETECT_KEY_ALIASES.get(m, m) if task_type == "Detection" else m)
        for m in metric_names
    }

    for path in glob.glob(os.path.join(parsed_dir, "*_samples_*.jsonl")):
        with open(path) as fh:
            for line in fh:
                sample = json.loads(line)
                label = _sample_label(sample.get("doc", {}), task_type)
                if label is None or label not in keep:
                    continue
                bucket = out[label]
                for m in metric_names:
                    value = _scalar(sample.get(jsonl_key[m]))
                    if value is not None:
                        bucket[m].append(float(value))
    return out


def summarize(values, higher_better):
    """Reduce raw per-sample values to the violin payload, or None if too few to fit a KDE.

    Returns ``{d: [int], q: [w_lo, q1, median, q3, w_hi], n: int}`` in plotted space.
    ``d`` is the KDE sampled on the GRID, peak-normalised to 100.
    """
    if len(values) < 3:
        return None

    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if not higher_better:
        v = 1.0 - v
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None

    grid = np.linspace(0.0, 1.0, GRID)
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(v)(grid)
    except Exception:
        # Degenerate input (every sample identical) makes the KDE covariance singular.
        # Fall back to a histogram so the spoke still shows where the mass sits.
        density, _ = np.histogram(v, bins=GRID, range=(0.0, 1.0), density=False)
        density = density.astype(float)

    peak = float(density.max()) if len(density) else 0.0
    if not math.isfinite(peak) or peak <= 0:
        return None
    d = [int(round(x)) for x in (density / peak) * 100.0]

    q1, median, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside_lo, inside_hi = v[v >= lo_fence], v[v <= hi_fence]
    w_lo = float(inside_lo.min()) if len(inside_lo) else float(v.min())
    w_hi = float(inside_hi.max()) if len(inside_hi) else float(v.max())
    # Quartiles are interpolated, so on a spiky distribution (e.g. 75% of samples exactly 0,
    # the rest far above the fence) q3 can fall on a value no sample takes, leaving every
    # in-fence sample BELOW it — a whisker pointing back inside the box. Clamp so the whiskers
    # always enclose the box; identical to the raw result whenever the fence behaves normally.
    w_lo, w_hi = min(w_lo, q1), max(w_hi, q3)

    return {
        "d": d,
        "q": [round(x, 4) for x in (w_lo, q1, median, q3, w_hi)],
        "n": int(len(v)),
    }


def build_task(task, results_dir, task_type, parsed_dirname):
    """Build one task's violin blob, reusing export_radar_data's spokes verbatim."""
    spec = erd.TASKS[task]
    radar_blob, order = erd.build_task(spec, results_dir, task_type, parsed_dirname)

    metrics = {m["key"]: bool(m["higher_better"]) for m in spec["metrics"]}
    metric_names = list(metrics)

    task_dir = os.path.join(results_dir, spec["task_dir"])
    dirname = "parsed" if spec.get("pin_parsed") else parsed_dirname
    display_map = viz_radar.load_config(os.path.join(SCRIPT_DIR, spec["config"]))["model_display_name"]
    dir_of = {display: key for key, display in display_map.items()}

    # Every label this task needs, across all groups — one read per model covers them all.
    all_labels = [s["full"] for grp in radar_blob["groups"] for s in grp["spokes"]]

    per_model = {}
    for display in order:
        parsed_dir = os.path.join(task_dir, dir_of[display], dirname)
        if not os.path.isdir(parsed_dir):
            sys.exit(f"[violin] missing parsed dir for '{display}': {parsed_dir}")
        per_model[display] = load_per_sample_multi(parsed_dir, metric_names, all_labels, task_type)
        print(f"[violin]   {task:9s} read {display}", flush=True)

    groups = []
    for grp in radar_blob["groups"]:
        labels = [s["full"] for s in grp["spokes"]]
        values = {}
        for display in order:
            by_label = per_model[display]
            values[display] = {
                m: [summarize(by_label[lbl][m], metrics[m]) for lbl in labels]
                for m in metric_names
            }
        groups.append({"name": grp["name"], "spokes": labels, "values": values})

    return {"groups": groups}


def emit_js(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_violin_data.py — DO NOT EDIT.\n"
        "// Per-spoke distributions behind the radar lines; drawn by static/js/radar.js as a\n"
        "// violin + box plot along each spoke (the web twin of viz_radar.py --verbose_model).\n"
        "// Schema: window.MEDVISION_VIOLIN = {\n"
        "//   grid: <int>,                         // density samples spanning plotted [0,1]\n"
        "//   tasks:{ '<Detection|TL|AD>': { groups:[ { name, spokes:[<full label>],\n"
        "//     values:{ '<model>': { '<metric>': [ per-spoke {d,q,n} | null ] } } } ] } } }\n"
        "// Spoke j here IS spoke j of the same group in radar-data.js (spokes are reused from\n"
        "// export_radar_data.build_task, never re-derived).\n"
        "// Coordinates are PLOTTED space — clamp(v,0,1), inverted for higher_better=false — so\n"
        "// radar.js maps them with its existing pixelR(). d[] is peak-normalised to 100 (shape\n"
        "// only); q = [whisker_lo, q1, median, q3, whisker_hi]; n = samples behind the spoke.\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_VIOLIN = ")
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")


def main():
    ap = argparse.ArgumentParser(description="Export radar violin-overlay data (violin-data.js).")
    ap.add_argument("--page_dir", required=True, help="Project page repo (medvision-vlm.github.io).")
    ap.add_argument("--results_dir", default=erd.DEFAULT_RESULTS_DIR, help="MedVision Results/ directory.")
    ap.add_argument("--out", default=None, help="Output JS path (default <page_dir>/static/js/violin-data.js).")
    ap.add_argument(
        "--parsed_dirname",
        default="parsed",
        help="Per-model subdirectory to read from, e.g. llm-parsed_gemma-4-31b. Default: parsed.",
    )
    args = ap.parse_args()

    task_types = {"Detection": "Detection", "TL": "TL", "AD": "AD", "TL-Pilot": "TL"}
    tasks = {}
    for task in TASKS:
        tasks[task] = build_task(task, args.results_dir, task_types[task], args.parsed_dirname)

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "violin-data.js")
    emit_js(out_path, {"grid": GRID, "tasks": tasks})

    size_kb = os.path.getsize(out_path) / 1024
    print(f"[violin] wrote {out_path} ({size_kb:.1f} KB)")
    for task, t in tasks.items():
        for grp in t["groups"]:
            filled = sum(
                1
                for mv in grp["values"].values()
                for arr in mv.values()
                for cell in arr
                if cell is not None
            )
            total = sum(len(arr) for mv in grp["values"].values() for arr in mv.values())
            print(
                f"[violin]   {task:9s} {str(grp['name'] or 'all'):12s} "
                f"spokes={len(grp['spokes']):3d}  distributions={filled}/{total}"
            )


if __name__ == "__main__":
    main()
