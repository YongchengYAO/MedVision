"""Scatter model benchmark performance against model release date.

Reads the three per-task LLM-parsed summary text files and plots one point per
(model, task):

    Detection  -> IoU            (sample-weighted mean of the ANATOMY and T/L groups)
    T/L size   -> 1 / MRE        (weighted-average MRE over all T/L labels)
    Distance   -> 1 / MRE        ("Distance" cross-dataset group row)
    Angle      -> 1 / MRE        ("Angle" cross-dataset group row)

All four are "higher is better" accuracy scores. MRE in the summary files is a
fraction (0.26 == 26%), so 1/MRE needs no rescaling and lands on the same order
of magnitude as IoU.

Layout: one panel per task on a grid set by --layout (ROWSxCOLS, e.g. 4x1 for a
stacked column or 1x4 for a side-by-side row), all sharing a single release-date
axis that is labelled only on the bottom-row panels. Each panel keeps its own
log y range, since the four metrics span very different magnitudes.

Encoding: colour = task, marker = model series, dotted line = same series within
one task, ordered by release date.
"""

import argparse
import os
import re
from datetime import date

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from medvision_bm.utils.plot_utils import save_fig_capped

# ----------------------------------------------------------------------------
# Model metadata.
#
# release: earliest public trace of the weights = HuggingFace repo creation date
#   (`createdAt` from https://huggingface.co/api/models/<id>). Labs often create
#   the repo a few days before the public announcement, so this is a lower bound.
#   Two rows use the upstream model rather than the artifact we benchmark:
#     MiniMax-M3      -> MiniMaxAI/MiniMax-M3 (2026-06-02), not the INT4 requant
#     LLaVA-OneVision -> lmms-lab/...-72b-ov  (2024-08-06), not the llava-hf port
#   MedGemma (27B) is the multimodal checkpoint; the MedGemma family first
#   shipped 2025-05-19 as medgemma-27b-text-it.
#   MedVision-V0 is ours and unreleased, so it carries the paper date.
#
# series: models sharing a series share a marker and are joined by a dotted line.
# aliases: raw "Model:" keys, which differ per task summary file.
# ----------------------------------------------------------------------------
MODELS = [
    {
        "name": "Qwen2.5-VL (7B)",
        "release": date(2025, 1, 26),
        "series": "Qwen-VL",
        "aliases": ["Qwen2.5-VL-7B-Instruct"],
    },
    {
        "name": "Qwen2.5-VL (32B)",
        "release": date(2025, 3, 21),
        "series": "Qwen-VL",
        "aliases": ["Qwen2.5-VL-32B-Instruct"],
    },
    {
        "name": "Qwen3-VL-Thinking (32B)",
        "release": date(2025, 10, 19),
        "series": "Qwen-VL",
        "aliases": ["Qwen3-VL-32B-Thinking"],
    },
    {
        "name": "Gemma-3 (27B)",
        "release": date(2025, 3, 1),
        "series": "Gemma",
        "aliases": ["gemma-3-27b-it"],
    },
    {
        "name": "Gemma-4 (31B)",
        "release": date(2026, 3, 11),
        "series": "Gemma",
        "aliases": ["gemma-4-31B-it"],
    },
    {
        "name": "MedGemma (4B)",
        "release": date(2025, 5, 19),
        "series": "MedGemma",
        "aliases": ["medgemma-4b-it", "google__medgemma-4b-it"],
    },
    {
        "name": "MedGemma (27B)",
        "release": date(2025, 7, 9),
        "series": "MedGemma",
        "aliases": ["medgemma-27b-it", "MedGemma-27b-it-Budget8k"],
    },
    {
        "name": "GLM-4.6V (106B)",
        "release": date(2025, 12, 7),
        "series": "GLM-4.6V",
        "aliases": ["GLM-4.6V", "GLM-4.6V-Budget16k"],
    },
    {
        "name": "GLM-4.6V-Flash (9B)",
        "release": date(2025, 12, 7),
        "series": "GLM-4.6V",
        "aliases": ["GLM-4.6V-Flash", "GLM-4.6V-Flash-Budget16k"],
    },
    {
        "name": "InternVL3 (38B)",
        "release": date(2025, 4, 10),
        "series": "InternVL3",
        "aliases": ["InternVL3-38B", "InternVL3-38B_bugfix-2eb7706"],
    },
    {
        "name": "Llama-3.2-Vision (11B)",
        "release": date(2024, 9, 18),
        "series": "Llama-3.2-Vision",
        "aliases": [
            "Llama-3.2-11B-Vision-Instruct-Budget16k",
            "Llama-3.2-11B-Vision-Instruct_bugfix-2eb7706",
        ],
    },
    {
        "name": "LLaVA-OneVision (72B)",
        "release": date(2024, 8, 6),
        "series": "LLaVA-OneVision",
        "aliases": ["llava-onevision-qwen2-72b-ov-hf", "LLaVA-OneVision_bugfix-0a4c5e2"],
    },
    {
        "name": "Lingshu (32B)",
        "release": date(2025, 6, 5),
        "series": "Lingshu",
        "aliases": ["lingshu-32b", "lingshu-medical-mllm__Lingshu-32B"],
    },
    {
        "name": "MedDr (40B)",
        "release": date(2024, 4, 22),
        "series": "MedDr",
        "aliases": ["MedDr__BF16"],
    },
    {
        "name": "HuatuoGPT-Vision (34B)",
        "release": date(2024, 6, 26),
        "series": "HuatuoGPT-Vision",
        "aliases": [
            "FreedomIntelligence__HuatuoGPT-Vision-34B",
            "HuatuoGPT-Vision-34B_bugfix-09206a2",
            "HuatuoGPT-Vision-34B_bugfix-2eb7706-wStopStrings",
        ],
    },
    {
        "name": "HealthGPT (14B)",
        "release": date(2025, 2, 17),
        "series": "HealthGPT",
        "aliases": [
            "HealthGPT-L14",
            "HealthGPT-L14_bugfix-2eb7706",
            "HealthGPT-L14_bugfix-0a4c5e2",
        ],
    },
    {
        "name": "MiniMax-M3 (428B)",
        "release": date(2026, 6, 2),
        "series": "MiniMax-M3",
        "aliases": ["MiniMax-M3-INT4"],
    },
    {
        "name": "MedVision-V0 (7B)",
        "release": date(2026, 8, 1),
        "series": "MedVision-V0",
        "aliases": [
            "MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250",
            "MedVision__fullRFT__qwen25vl-7b-fullSFT__AD-TL-D__512x512__PRxAnswer_s250_CoT",
        ],
    },
]

# Marker per series. Series with more than one member are joined by a dotted line.
SERIES_MARKERS = {
    "Qwen-VL": "o",
    "Gemma": "s",
    "MedGemma": "D",
    "GLM-4.6V": "^",
    "InternVL3": "v",
    "Llama-3.2-Vision": "<",
    "LLaVA-OneVision": ">",
    "Lingshu": "P",
    "MedDr": "X",
    "HuatuoGPT-Vision": "h",
    "HealthGPT": "p",
    "MiniMax-M3": "d",
    "MedVision-V0": "*",
}

# One stacked panel per task: (key, panel name, metric, colour).
TASKS = [
    ("detection", "Detection", "IoU", "#1F77B4"),
    ("tl", "T/L size", "1/MRE", "#D62728"),
    ("distance", "Distance", "1/MRE", "#2CA02C"),
    ("angle", "Angle", "1/MRE", "#9467BD"),
]


def _alias_lookup():
    """raw summary key -> model metadata dict."""
    out = {}
    for m in MODELS:
        for a in m["aliases"]:
            out[a] = m
    return out


def parse_detection(path):
    """-> {raw_name: IoU}, sample-weighted across the ANATOMY and T/L groups."""
    group_re = re.compile(
        r"^\s*(ANATOMY|T/L)\s*\(\s*\d+\s*regions,\s*(\d+)\s*samples\):(.*)$"
    )
    iou_re = re.compile(r"(?:^|,)\s*IoU=([0-9.eE+-]+)")
    out, raw, num, den = {}, None, 0.0, 0.0
    with open(path) as f:
        for line in f:
            if line.startswith("Model: "):
                if raw is not None and den > 0:
                    out[raw] = num / den
                raw, num, den = line[len("Model: ") :].strip(), 0.0, 0.0
                continue
            m = group_re.match(line)
            if m and raw is not None:
                n = float(m.group(2))
                iou = iou_re.search(m.group(3))
                if iou:
                    num += n * float(iou.group(1))
                    den += n
    if raw is not None and den > 0:
        out[raw] = num / den
    return out


def parse_tl(path):
    """-> {raw_name: MRE} from each model's 'Weighted Average' line."""
    mre_re = re.compile(r"^Weighted Average .*?\bMRE:\s*([0-9.eE+-]+)")
    out, raw = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("Model: "):
                raw = line[len("Model: ") :].strip()
                continue
            m = mre_re.match(line)
            if m and raw is not None:
                out[raw] = float(m.group(1))
                raw = None
    return out


def parse_ad(path):
    """-> {raw_name: {"distance": MRE, "angle": MRE}} from the cross-dataset
    'Distance' and 'Angle' rows of each model's 'Group averages' table."""
    wanted = {"Distance": "distance", "Angle": "angle"}
    out, raw, in_groups = {}, None, False
    with open(path) as f:
        for line in f:
            if line.startswith("Model: "):
                raw = line[len("Model: ") :].strip()
                in_groups = False
                continue
            if line.startswith("Group averages:"):
                in_groups = True
                continue
            if line.startswith("Label-specific metrics:"):
                in_groups = False
                continue
            if not (in_groups and raw and "|" in line):
                continue
            cells = [c.strip() for c in line.split("|")]
            key = wanted.get(cells[0])
            if key and len(cells) > 2:
                try:
                    out.setdefault(raw, {})[key] = float(cells[2])
                except ValueError:
                    pass
    return out


def collect(args):
    """-> (records, unmapped) where records is a list of dicts, one per point."""
    lookup = _alias_lookup()
    detection = parse_detection(args.detect_summary)
    tl = parse_tl(args.tl_summary)
    ad = parse_ad(args.ad_summary)

    scores, unmapped = {}, set()

    def put(raw, task, value):
        meta = lookup.get(raw)
        if meta is None:
            unmapped.add(raw)
            return
        if value is None or value != value or value <= 0:  # NaN / non-positive
            return
        scores.setdefault(meta["name"], {})[task] = value

    for raw, iou in detection.items():
        put(raw, "detection", iou)
    for raw, mre in tl.items():
        put(raw, "tl", 1.0 / mre if mre else None)
    for raw, per_group in ad.items():
        for key in ("distance", "angle"):
            mre = per_group.get(key)
            put(raw, key, 1.0 / mre if mre else None)

    records = []
    for m in MODELS:
        for task, _, _, _ in TASKS:
            value = scores.get(m["name"], {}).get(task)
            if value is not None:
                records.append(
                    {
                        "name": m["name"],
                        "series": m["series"],
                        "release": m["release"],
                        "task": task,
                        "value": value,
                    }
                )
    return records, unmapped


def _text_bbox(ann, renderer):
    """Display bbox of an annotation's TEXT only.

    Annotation.get_window_extent unions the text with its leader line, which
    would make every box span back to its marker and defeat collision testing.
    Calling it first refreshes the cached layout; Text's own implementation then
    reports just the glyph box.
    """
    ann.get_window_extent(renderer=renderer)
    return Text.get_window_extent(ann)


def _declutter(fig, per_axis, step=10.0, max_steps=20, pad=1.10):
    """Nudge point labels vertically until they clear other labels and markers.

    matplotlib has no label repulsion and adjustText is not in the pinned env, so
    this does one greedy pass in display space: within a panel, labels are placed
    in x order and each takes the first free slot searched outwards from its
    natural position. A label may sit on its own marker but not on anyone else's.

    Obstacle radius is derived from the marker's own area (scatter ``s`` is in
    points squared), so it stays correct if the marker size changes. Requires a
    draw first, so the renderer can measure text.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    for ax, items in per_axis:
        ax_bb = ax.get_window_extent(renderer=renderer)
        markers = {}
        for ann, size in items:
            cx, cy = ax.transData.transform((mdates.date2num(ann.xy[0]), ann.xy[1]))
            r = (size**0.5) / 2.0 * px_per_pt
            markers[ann] = Bbox.from_extents(cx - r, cy - r, cx + r, cy + r)
        placed = []
        for ann, _ in sorted(items, key=lambda it: it[0].xy[0]):
            others = [bb for a, bb in markers.items() if a is not ann]
            dx = ann.xyann[0]
            chosen = None
            for k in range(max_steps):
                for dy in (0.0,) if k == 0 else (k * step, -k * step):
                    ann.xyann = (dx, dy)
                    bb = _text_bbox(ann, renderer).expanded(1.0, pad)
                    if bb.y0 < ax_bb.y0 or bb.y1 > ax_bb.y1:
                        continue  # would spill outside the panel
                    if not any(bb.overlaps(o) for o in placed) and not any(
                        bb.overlaps(o) for o in others
                    ):
                        chosen = bb
                        break
                if chosen is not None:
                    break
            if chosen is None:  # give up rather than fling the label off-panel
                ann.xyann = (dx, 0.0)
                chosen = _text_bbox(ann, renderer).expanded(1.0, pad)
            placed.append(chosen)


def plot(records, args):
    series_in_use = [s for s in SERIES_MARKERS if any(r["series"] == s for r in records)]

    nrows, ncols = args.layout_grid
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, figsize=(args.fig_width, args.fig_height)
    )
    axes = axes.ravel()

    per_axis_labels = []
    for ax, (task, name, metric, color) in zip(axes, TASKS):
        anns = []
        # Dotted line per series, in release order. Single-model series have
        # nothing to join, so they contribute markers only.
        linked = set()
        for series in series_in_use:
            pts = sorted(
                (r for r in records if r["series"] == series and r["task"] == task),
                key=lambda r: r["release"],
            )
            if len(pts) > 1:
                linked.add(series)
                ax.plot(
                    [p["release"] for p in pts],
                    [p["value"] for p in pts],
                    linestyle=":",
                    linewidth=2.5,
                    color=color,
                    alpha=1.0,
                    zorder=1,
                )

        for r in (r for r in records if r["task"] == task):
            is_ours = r["series"] == "MedVision-V0"
            size = 520 if is_ours else 170
            ax.scatter(
                r["release"],
                r["value"],
                marker=SERIES_MARKERS[r["series"]],
                s=size,
                facecolors="none",
                edgecolors=color,
                linewidths=2.5,
                alpha=0.95,
                zorder=3 if is_ours else 2,
            )
            # Label only points a dotted line actually joins, plus our own model,
            # which is always named. Everything else is read off the marker legend.
            if not (is_ours or r["series"] in linked):
                continue
            # Offset past the marker's own radius (scatter s is points squared).
            anns.append((ax.annotate(
                r["name"],
                xy=(r["release"], r["value"]),
                xytext=((size**0.5) / 2.0 + 3.0, 0),
                textcoords="offset points",
                fontsize=args.label_fontsize,
                color="#333333",
                va="center",
                ha="left",
                zorder=4,
                annotation_clip=False,
                # Leader line: the declutter pass can push a label well away from
                # its marker in a dense cluster, which would misattribute it.
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=0.6,
                    color="0.55",
                    shrinkA=(size**0.5) / 2.0,
                    shrinkB=1.0,
                ),
            ), size))

        per_axis_labels.append((ax, anns))
        ax.set_yscale("log")
        ax.set_ylabel(
            f"{name}\n{metric}", fontsize=24, fontweight="bold", color=color
        )
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.25)
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.3, alpha=0.15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Headroom for labels nudged off their marker by the declutter pass.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo / 1.9, hi * 1.9)

    # The x axis is shared; matplotlib hides the tick labels of panels above the
    # bottom row, so only the bottom-row panels need date formatting and a label.
    for bottom in axes[-ncols:]:
        bottom.set_xlabel("Model release date", fontsize=13)
        bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        bottom.tick_params(axis="x", labelsize=10)
        plt.setp(bottom.get_xticklabels(), rotation=45, ha="right")
    # Room on the right for the last model's label (shared axis: sets all panels).
    xlo, xhi = axes[-1].get_xlim()
    axes[-1].set_xlim(xlo - 20, xhi + 120)

    # Hollow markers: the legend keys the marker shape only, so filling it with a
    # colour would imply a task the entry does not belong to.
    series_handles = [
        mlines.Line2D(
            [],
            [],
            color="0.25",
            marker=SERIES_MARKERS[s],
            linestyle="",
            markerfacecolor="none",
            markeredgewidth=1.2,
            markersize=11 if s == "MedVision-V0" else 8,
            label=s,
        )
        for s in series_in_use
    ]
    if args.title:
        fig.suptitle(args.title, fontsize=14)
    # Reserve the strip the shared legend sits in; the panels tile the rest.
    fig.tight_layout(rect=[0, args.legend_frac, 1, 0.99 if args.title else 1])
    fig.legend(
        handles=series_handles,
        title="Model series",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=args.legend_col,
        fontsize=15,
        title_fontsize=16.5,
        frameon=False,
    )

    _declutter(fig, per_axis_labels)

    os.makedirs(args.fig_dir, exist_ok=True)
    stem = os.path.splitext(os.path.join(args.fig_dir, args.fig_name))[0]
    formats = ["png"] if args.save_as_png else ["pdf"]
    if args.save_as_png and args.save_as_pdf:
        formats = ["png", "pdf"]
    written = []
    for fmt in formats:
        out = f"{stem}.{fmt}"
        # No bbox_inches="tight": trimming would change the canvas aspect, and the
        # layout is already managed by tight_layout above, so the file keeps the
        # exact fig_width x fig_height ratio.
        save_fig_capped(out, fig=fig, transparent=True)
        written.append(out)
    plt.close(fig)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ad_summary", required=True, help="summary_AD_task*.txt")
    p.add_argument("--tl_summary", required=True, help="summary_TL_task*.txt")
    p.add_argument("--detect_summary", required=True, help="summary_detection_task*.txt")
    p.add_argument("--fig_dir", default="Figures")
    p.add_argument("--fig_name", default="leaderboard_timeline.pdf")
    # Canvas: width widened 20% from the original 9.0 (h:w was 2:1, now ~1.67:1).
    p.add_argument("--fig_width", type=float, default=10.8)
    p.add_argument("--fig_height", type=float, default=18.0)
    p.add_argument("--label_fontsize", type=float, default=12.0)
    p.add_argument("--legend_col", type=int, default=4)
    p.add_argument(
        "--layout",
        default="4x1",
        help="panel grid as ROWSxCOLS, e.g. 4x1 (stacked column) or 1x4 (one row)",
    )
    p.add_argument(
        "--legend_frac",
        type=float,
        default=0.10,
        help="fraction of figure height reserved for the bottom legend",
    )
    p.add_argument("--title", default="")
    p.add_argument("--save_as_png", action="store_true")
    p.add_argument("--save_as_pdf", action="store_true")
    args = p.parse_args()

    try:
        nrows, ncols = (int(v) for v in args.layout.lower().split("x"))
    except ValueError:
        p.error(f"--layout must be ROWSxCOLS, got {args.layout!r}")
    if nrows < 1 or ncols < 1 or nrows * ncols != len(TASKS):
        p.error(f"--layout {args.layout} must cover exactly {len(TASKS)} panels")
    args.layout_grid = (nrows, ncols)

    records, unmapped = collect(args)
    if unmapped:
        print("Skipped (no release date / not on the leaderboard roster):")
        for raw in sorted(unmapped):
            print(f"  - {raw}")

    heads = "".join(f"{name + ' (' + metric + ')':<20}" for _, name, metric, _ in TASKS)
    print(f"\n{'Model':<24}{'Release':<12}{heads}")
    by_model = {}
    for r in records:
        by_model.setdefault(r["name"], {})[r["task"]] = r["value"]
    for m in sorted(MODELS, key=lambda m: m["release"]):
        row = by_model.get(m["name"])
        if not row:
            continue
        cells = "".join(
            f"{row[k]:<20.4f}" if k in row else f"{'-':<20}" for k, _, _, _ in TASKS
        )
        print(f"{m['name']:<24}{m['release'].isoformat():<12}{cells}")

    for out in plot(records, args):
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
