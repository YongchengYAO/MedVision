#!/usr/bin/env python3
"""Export the interactive performance-vs-release-date data blob for the project webpage.

Reads the same three per-task summary text files that ``viz_benchmark_leaderboard_timeline.py`` plots, and
writes ``<page_dir>/static/js/timeline-data.js`` defining the ``window.MEDVISION_TIMELINE`` global
that ``static/js/timeline.js`` consumes. Mirrors ``export_radar_data.py`` /
``export_detection_performance_per_boxImgRatio_data.py`` (same ``--page_dir`` convention, minified emit, DO-NOT-EDIT header,
fail-loud contract).

This is the interactive twin of ``leaderboard_timeline.pdf``: benchmark accuracy against the date
the weights first appeared, one point per (model, task). The figure's rules are REUSED rather than
re-implemented -- ``viz_benchmark_leaderboard_timeline`` is imported as a sibling module, so the release
dates, the model-series grouping, the summary parsers and the four task/metric definitions have
exactly one home. A number on the page and the same number in the paper cannot drift.

Two things the page does differently from the PDF, both because the widget shows ONE task at a
time while the figure stacks all four:

  * colour encodes the MODEL, not the task -- from the shared page palette
    (``configs.radar_model_colors``, position taken from the detection config's
    ``model_display_name`` order), so a model keeps one colour across this widget, the radar, the
    box-size explorers and every PDF. Marker still encodes the model SERIES, which is what makes
    the dotted same-family link readable;
  * model names come from the task configs, not from ``viz_benchmark_leaderboard_timeline.MODELS``, so the
    widget agrees with the leaderboard tables above it (e.g. "MiniMax-M3 (428B, int4)").

Both the plotted score and the quantity behind it are stored: Detection plots IoU directly, while
T/L, Distance and Angle plot 1/MRE, whose raw MRE the tooltip shows so a reader never has to invert
a number in their head.

Example
-------
    PYTHONPATH=src python script/visualization/export_benchmark_leaderboard_timeline_data.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io \
        --ad_summary Results/MedVision-AD-v2-CoT/summary_AD_task__llm-parsed_gemma-4-31b.txt \
        --tl_summary Results/MedVision-TL-v2-CoT/summary_TL_task_filtered__llm-parsed_gemma-4-31b.txt \
        --detect_summary Results/MedVision-detect-v2/summary_detection_task__llm-parsed_gemma-4-31b.txt
"""
import argparse
import json
import math
import os
import sys

import yaml

from medvision_bm.utils.configs import radar_model_colors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# viz_benchmark_leaderboard_timeline.py is a sibling script (not an installed module) -- reuse it directly so
# the release dates, series grouping, summary parsers and task list have a single home.
sys.path.insert(0, SCRIPT_DIR)
import viz_benchmark_leaderboard_timeline as TL  # noqa: E402

DEFAULT_RESULTS_DIR = os.path.join(REPO, "Results")
# Colour order comes from the DETECTION config, exactly as export_radar_data.py resolves it; the
# other two only contribute aliases, since each task file names the same model differently.
DEFAULT_COLOR_CONFIG = os.path.join(SCRIPT_DIR, "config-detect-CoT.yaml")
DEFAULT_ALIAS_CONFIGS = [
    os.path.join(SCRIPT_DIR, name)
    for name in ("config-detect-CoT.yaml", "config-TL-CoT.yaml", "config-AD-CoT.yaml")
]

WRAP_DARKEN = 0.8

# What each panel measures, in the figure's order. ``mre`` marks the three tasks whose plotted
# value is 1/MRE, so the page can label the axis and show the raw rate in the tooltip.
TASK_META = {
    "detection": {"label": "Detection", "metric": "IoU", "mre": False},
    "tl": {"label": "T/L size", "metric": "1 / MRE", "mre": True},
    "distance": {"label": "Distance", "metric": "1 / MRE", "mre": True},
    "angle": {"label": "Angle", "metric": "1 / MRE", "mre": True},
}


def model_palette(n):
    """``n`` model colours: tab10 in order, reused hues darkened to ``WRAP_DARKEN``.

    Byte-identical to ``viz_radar.model_palette`` and to ``export_detection_performance_per_boxImgRatio_data.model_palette``;
    duplicated here for the same reason the latter duplicates it -- so this exporter depends on
    ``configs`` rather than on the 57 KB radar module.
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


def _sig(v, digits=5):
    """Compact float for the JS blob, rounded to ``digits`` SIGNIFICANT figures.

    None/NaN/inf become null (never a bare NaN token). Significant figures, not decimal places:
    every value here lands on a LOG axis, where decimal rounding is not scale-invariant and can
    erase a point outright. MedGemma (4B) scores 1/MRE = 2.7e-05 on T/L, which ``round(v, 4)``
    turns into 0.0 -- a coordinate no log axis can place.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    if f == 0.0:
        return 0.0
    return round(f, -(int(math.floor(math.log10(abs(f)))) - (digits - 1)))


def load_display_map(paths):
    """Union of every config's ``model_display_name``: raw summary key -> page display name."""
    out = {}
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"[timeline] config not found: {path}")
        with open(path) as fh:
            mapping = (yaml.safe_load(fh) or {}).get("model_display_name") or {}
        if not mapping:
            sys.exit(f"[timeline] {path}: no model_display_name map.")
        for raw, display in mapping.items():
            if out.get(raw, display) != display:
                sys.exit(
                    f"[timeline] '{raw}' is displayed as both '{out[raw]}' and '{display}' across "
                    "the task configs; the widget cannot label it two ways."
                )
            out[raw] = display
    return out


def load_color_order(path):
    """Display names in the detection config's order -- the page's shared colour assignment."""
    with open(path) as fh:
        mapping = (yaml.safe_load(fh) or {}).get("model_display_name") or {}
    if not mapping:
        sys.exit(f"[timeline] {path}: no model_display_name map.")
    return list(mapping.values())


def resolve(display_map, color_order):
    """-> [{name, series, marker, color, release, ours}] in release order, and a name lookup.

    Each ``viz_benchmark_leaderboard_timeline.MODELS`` entry is matched to a page display name through its
    aliases. Anything unmatched is fatal: a silently dropped model reads on the page as "this model
    was never evaluated" rather than "the roster and the configs disagree".
    """
    colors = model_palette(len(color_order))
    color_of = {name: colors[i] for i, name in enumerate(color_order)}

    resolved, unmatched, uncoloured = {}, [], []
    for m in TL.MODELS:
        names = {display_map[a] for a in m["aliases"] if a in display_map}
        if not names:
            unmatched.append(m["name"])
            continue
        if len(names) > 1:
            sys.exit(
                f"[timeline] '{m['name']}' resolves to several display names {sorted(names)}; "
                "its aliases span two models."
            )
        display = names.pop()
        if display not in color_of:
            uncoloured.append(display)
        resolved[m["name"]] = display
    if unmatched:
        sys.exit(
            "[timeline] no config alias for: " + ", ".join(unmatched) + "\n"
            "  Add the model's summary key to script/visualization/config-*-CoT.yaml, or drop it "
            "from viz_benchmark_leaderboard_timeline.MODELS."
        )
    if uncoloured:
        sys.exit(
            "[timeline] not in the detection config's colour order: " + ", ".join(uncoloured) +
            " — the widget would give it a colour no other figure uses."
        )

    models = []
    for m in sorted(TL.MODELS, key=lambda m: (m["release"], m["name"])):
        display = resolved[m["name"]]
        models.append({
            "name": display,
            "series": m["series"],
            "marker": TL.SERIES_MARKERS[m["series"]],
            "color": color_of[display],
            "release": m["release"].isoformat(),
            "ours": m["series"] == "MedVision-V0",
        })
    return models, resolved


def main():
    ap = argparse.ArgumentParser(description="Export interactive timeline data (timeline-data.js).")
    ap.add_argument("--page_dir", required=True, help="Project page repo (medvision-vlm.github.io).")
    ap.add_argument("--ad_summary", required=True, help="summary_AD_task*.txt")
    ap.add_argument("--tl_summary", required=True, help="summary_TL_task*.txt")
    ap.add_argument("--detect_summary", required=True, help="summary_detection_task*.txt")
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "YAML with a model_display_name map; repeatable. Defaults to the three "
            "config-{detect,TL,AD}-CoT.yaml files, whose union covers every per-task summary key."
        ),
    )
    ap.add_argument(
        "--color_config",
        default=DEFAULT_COLOR_CONFIG,
        help="Config whose model order fixes the shared page palette (default: config-detect-CoT.yaml).",
    )
    ap.add_argument("--out", default=None, help="Output JS path (default <page_dir>/static/js/timeline-data.js).")
    args = ap.parse_args()

    for path in (args.ad_summary, args.tl_summary, args.detect_summary):
        if not os.path.exists(path):
            sys.exit(f"[timeline] summary not found: {path}")

    display_map = load_display_map(args.config or DEFAULT_ALIAS_CONFIGS)
    color_order = load_color_order(args.color_config)
    models, resolved = resolve(display_map, color_order)

    # The figure's own collection pass: same parsers, same 1/MRE inversion, same near-zero drops.
    records, unmapped = TL.collect(args)
    if unmapped:
        print("[timeline] not on the leaderboard roster (skipped, as in the PDF):")
        for raw in sorted(unmapped):
            print(f"[timeline]   - {raw}")

    # Raw MRE per (model, task), so the tooltip can show the rate the score inverts. Re-read from
    # the same parsers rather than inverting the plotted value, which would round-trip a rounding.
    tl_mre = TL.parse_tl(args.tl_summary)
    ad_mre = TL.parse_ad(args.ad_summary)
    raw_mre = {}
    for raw, mre in tl_mre.items():
        if raw in display_map:
            raw_mre.setdefault(display_map[raw], {})["tl"] = mre
    for raw, groups in ad_mre.items():
        if raw in display_map:
            for key in ("distance", "angle"):
                if key in groups:
                    raw_mre.setdefault(display_map[raw], {})[key] = groups[key]

    values = {}
    for r in records:
        display = resolved[r["name"]]
        cell = {"v": _sig(r["value"])}
        mre = raw_mre.get(display, {}).get(r["task"])
        if TASK_META[r["task"]]["mre"] and mre is not None:
            cell["mre"] = _sig(mre)
        values.setdefault(display, {})[r["task"]] = cell

    tasks = []
    for key, label, metric, color in TL.TASKS:
        meta = TASK_META[key]
        tasks.append({
            "key": key,
            "label": meta["label"],
            "metric": meta["metric"],
            # The PDF's per-panel accent. The widget colours points by model, but keeps this for
            # the task chip and the axis title, so the two artefacts still read as one family.
            "color": color,
            "mre": meta["mre"],
        })
        if label != meta["label"] or metric != meta["metric"].replace(" ", ""):
            # Non-fatal: the labels are cosmetic, but a silent divergence from the PDF is worth
            # seeing in the log.
            print(f"[timeline] note: '{key}' label/metric differs from the PDF ({label} / {metric}).")

    scored = sum(len(v) for v in values.values())
    blob = {"tasks": tasks, "models": models, "values": values}

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "timeline-data.js")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_benchmark_leaderboard_timeline_data.py - DO NOT EDIT.\n"
        "// Schema: window.MEDVISION_TIMELINE = {\n"
        "//   tasks:[ {key,label,metric,color,mre} ],   // mre=true => the plotted value is 1/MRE\n"
        "//   models:[ {name,series,marker,color,release,ours} ],   // release order, ISO date\n"
        "//   values:{ '<model name>': { '<task key>': {v, mre?} } } }\n"
        "// v is the plotted score (higher is better everywhere); mre is the raw rate it inverts.\n"
        "// Colour = model (shared page palette); marker = model series, as in leaderboard_timeline.pdf.\n"
    )
    with open(out_path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_TIMELINE = ")
        json.dump(blob, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(
        f"[timeline] wrote {out_path} ({size_kb:.1f} KB) | {len(models)} models, "
        f"{len(tasks)} tasks, {scored} scored points"
    )
    print(f"[timeline]   release span: {models[0]['release']} -> {models[-1]['release']}")
    thin = [m["name"] for m in models if len(values.get(m["name"], {})) < len(tasks)]
    if thin:
        print(f"[timeline]   incomplete rows ({len(thin)}): {', '.join(thin)}")


if __name__ == "__main__":
    main()
