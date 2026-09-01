"""
Plot label clouds of ``target @ modality`` for the Detection and T/L tasks,
contrasting each task's in-distribution roster against its target-OOD roster.

Four clouds are drawn in one 2x2 figure:

                    In-distribution                 Target-OOD
    Detection   tasks_MedVision-detect-CoT      OOD/...-detect-CoT-taskOOD
    T/L         tasks_MedVision-TL__train_SFT   OOD/...-TL-CoT-taskOOD

The in-distribution rosters are the ones MedVision-V0 was post-trained on; the
target-OOD rosters hold targets held out from that training.

Each roster JSON maps a task key (``<dataset>_<annotation>_Task<NN>_<plane>``,
optionally suffixed ``-CoT``) to a sample count. The key selects a task from
that dataset's ``benchmark_plan`` in medvision_ds, which supplies the labels,
the modality, and the acquisition description.

  * Detection - the task's ``labels_map`` contributes EVERY label in the task;
    one detection task covers up to 15 organs.
  * T/L - the task's ``target_label`` selects a single label.

Labels are reconciled through ``label_map_rename`` - NOT ``label_map_regroup``,
which viz_radar.py applies on the detection path - so the clouds show
fine-grained targets while still merging each upstream dataset's naming
variants ("gall bladder"/"gallbladder", the "(ivc)"/"(lag)"/"(rag)"
parentheticals). Modality is normalised to MR / CT / US / XR / PET and the
slice plane is dropped, so the detection OOD roster's Sagittal/Coronal/Axial
copies of one task collapse into a single entry.

Acquisition disambiguation
--------------------------
One ``target @ modality`` string can hide genuinely different acquisitions: the
ISLES24 stroke infarct is an ADC map in Task01 but a DWI scan in Task02, and
BraTS24 images the same paediatric tumour as contrast-enhanced T1 in Task10 and
non-contrast T1 in Task11. Where the tasks behind one label disagree on
acquisition, the label is SPLIT and each variant tagged - "stroke infarct @ MR
(ADC)" vs "stroke infarct @ MR (DWI)". Tags come from ACQUISITION_TAGS below,
matched against the plan's ``image_description``.

Labels that collide only because several datasets image the same target the
same way are NOT split: "spleen @ CT" is drawn once, not six times, because
AMOS22 / BCV15 / FLARE22 / MSD / AbdomenCT-1K / AbdomenAtlas all acquire it as
a plain abdominal CT. The distinction the split encodes is acquisition, not
provenance.

Layout
------
Word size carries NO meaning, and ONE size is used across all four panels, so a
panel's block size reflects how many targets its roster holds. Detection sample
counts are not available per label anyway (a task's single count covers all of
its labels), so encoding magnitude would be a fabrication on that path.

Words are drawn in alphabetical order, filling top-to-bottom down each column
before moving to the next, so a panel reads like an index. Each block hangs
from the top of its panel rather than floating in the middle. Tumour/lesion
targets are purple (``is_purple_label``), all others black, matching the radar
figures.

A label that still reads the same in a task's two panels after the acquisition
split is separated by provenance instead - "bladder @ CT (AMOS22)" in-distribution
against "bladder @ CT (BCV15 Cervix)" at target-OOD. Two panels showing an
identical word invite the reading that the rosters overlap; naming the dataset,
and its cohort where the plan records one, shows what actually differs. Anything
still identical afterwards is reported on stdout.

Usage (see viz_label_cloud.sh for the driver):
    python viz_label_cloud.py \
        --detect_indist <json> --detect_ood <json> \
        --tl_indist <json> --tl_ood <json> \
        --fig_dir <dir> --fig_name <name.pdf> \
        [--cell_width N] [--cell_height N] [--tl_cell_height N] \
        [--save_as_png] [--save_as_pdf]
"""

import argparse
import importlib
import json
import os
import re
from collections import defaultdict
from math import ceil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from viz_radar import is_purple_label
from medvision_bm.utils.configs import (
    C_ANATOMY_LABEL,
    C_TUMOR_LESION_LABEL,
    DATASETS_NAME2PACKAGE,
    label_map_rename,
)
from medvision_bm.utils.plot_utils import save_fig_capped

# Roster task keys: <dataset>_<annotation>_Task<NN>_<plane>, optionally "-CoT".
TASK_KEY_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<annotation>BoxCoordinate|TumorLesionSize)"
    r"_Task(?P<task_id>\d+)_(?P<plane>Axial|Coronal|Sagittal)(?:-CoT)?$"
)

# Same normalisation as viz_radar.py's label reconstruction.
MODALITY_MAP = {"MRI": "MR", "CT": "CT", "ultrasound": "US", "X-ray": "XR", "PET": "PET"}

# Acquisition tags, matched IN ORDER against a plan task's image_description;
# the first hit wins and later patterns are not tried. Order matters twice:
# "T2 Fluid Attenuated Inversion Recovery (FLAIR)" must read as FLAIR rather
# than T2w, and "non-contrast T1-weighted" must be caught before the
# contrast-enhanced pattern, whose "contrast" would otherwise match it.
# Gadolinium IS the contrast agent, so gadolinium-enhanced and contrast-enhanced
# T1 collapse to one tag. A description with no hit is generic for its modality
# (plain abdominal CT, knee MRI, echocardiography) and carries no tag.
ACQUISITION_TAGS = [
    (re.compile(r"apparent diffusion coefficient", re.I), "ADC"),
    (re.compile(r"diffusion-weighted", re.I), "DWI"),
    (re.compile(r"fluid[- ]attenuated inversion recovery|\bFLAIR\b", re.I), "FLAIR"),
    (re.compile(r"time of flight|TOF-MRA", re.I), "TOF-MRA"),
    (re.compile(r"non-contrast T1[- ]weighted", re.I), "T1w"),
    (re.compile(r"(?:gadolinium|contrast)[- ]enhanced T1[- ]weighted", re.I), "T1w-CE"),
    (re.compile(r"T2[- ]weighted", re.I), "T2w"),
    (re.compile(r"corticomedullary-phase", re.I), "corticomedullary phase"),
    (re.compile(r"kidney contrast", re.I), "contrast"),
]

# Cohort names that only restate the modality ("AMOS22-CT", "AMOS22-MRI") and so
# add nothing to a provenance tag.
MODALITY_LIKE = {name.upper() for name in MODALITY_MAP} | {
    code.upper() for code in MODALITY_MAP.values()
}

# Inter-word gap, in axes fractions. Split per axis because a panel is wider
# than it is tall, so equal fractions are not equal distances on the page.
PAD_X = 0.020
PAD_Y = 0.006

# Binary search bounds for the shared font size, in points.
FONT_LO = 1.0
FONT_HI = 90.0
FONT_TOL = 0.1

TITLE_FONTSIZE = 15


def benchmark_plan(dataset, task_kind):
    """The medvision_ds benchmark plan for one dataset's segmentation (Detection)
    or biometry (T/L) preprocessing module."""
    package = DATASETS_NAME2PACKAGE[dataset]
    module_name = (
        "preprocess_segmentation" if task_kind == "detect" else "preprocess_biometry"
    )
    module = importlib.import_module(f"medvision_ds.datasets.{package}.{module_name}")
    return module.benchmark_plan


def acquisition_tag(description):
    """Short acquisition tag for a plan task's image_description, or None when
    the description is generic for its modality."""
    for pattern, tag in ACQUISITION_TAGS:
        if pattern.search(description):
            return tag
    return None


def provenance_tag(dataset, image_folder):
    """Where a task's images come from: the dataset, plus its cohort when the
    plan's image_folder names one.

    Datasets that ship a single cohort just use an "Images" folder and get the
    bare dataset name. Datasets split into cohorts prefix the folder with their
    own name - "BCV15-Cervix/Images", "HNTSMRG24-preRT/Images" - and the part
    after that prefix is the cohort. A cohort that only restates the modality
    ("AMOS22-CT") is dropped, since the label already carries it.
    """
    cohort = image_folder.split("/")[0]
    prefix = f"{dataset}-"
    cohort = cohort[len(prefix):] if cohort.startswith(prefix) else ""
    if not cohort or cohort.upper() in MODALITY_LIKE:
        return dataset
    return f"{dataset} {cohort}"


def roster_entries(json_path, task_kind):
    """Resolve one roster JSON to its target entries.

    Returns (entries, n_tasks, n_samples) where entries is a list of
    (base_label, acq_tag, provenance) with base_label as
    ``"<target> @ <MODALITY>"`` and acq_tag possibly None.
    """
    with open(json_path) as handle:
        roster = json.load(handle)

    entries = []
    n_samples = 0
    for task_key, count in roster.items():
        match = TASK_KEY_RE.match(task_key)
        if match is None:
            raise ValueError(f"Unrecognised task key in {json_path}: {task_key!r}")
        dataset = match["dataset"]
        task = benchmark_plan(dataset, task_kind)["tasks"][int(match["task_id"]) - 1]
        n_samples += count

        labels_map = task["labels_map"]
        if task_kind == "detect":
            raw_names = list(labels_map.values())
        else:
            raw_names = [labels_map[str(task["target_label"])]]

        modality = MODALITY_MAP.get(task["image_modality"], task["image_modality"])
        tag = acquisition_tag(task["image_description"])
        provenance = provenance_tag(dataset, task.get("image_folder", ""))
        for raw_name in raw_names:
            renamed = label_map_rename.get(raw_name)
            if renamed is None:
                raise ValueError(
                    f"{raw_name!r} ({dataset} {match['task_id']}) is missing from "
                    "label_map_rename in configs.py"
                )
            entries.append((f"{renamed} @ {modality}", tag, provenance))

    return entries, len(roster), n_samples


def split_ambiguous(*entry_lists):
    """Decide which base labels need an acquisition tag.

    A base label is tagged only when the tasks behind it - across ALL the rosters
    passed in, so a task's two panels stay consistent - disagree on acquisition.
    Returns the set of base labels to tag.
    """
    tags_seen = defaultdict(set)
    for entries in entry_lists:
        for base, tag, _ in entries:
            tags_seen[base].add(tag)
    return {base for base, tags in tags_seen.items() if len(tags) > 1}


def acquisition_word(base, tag, ambiguous):
    """The label as it reads once acquisition splits are applied."""
    return f"{base} ({tag})" if base in ambiguous and tag else base


def display_word(base, tag, provenance, ambiguous, shared):
    """The label as drawn.

    Carries the acquisition tag where the acquisitions behind it disagree, and
    the provenance where the label would otherwise be drawn identically in a
    task's in-distribution AND target-OOD panels. Two panels showing the same
    word invite the reading that the roster overlaps; naming the cohort shows
    what actually differs.
    """
    parts = []
    if base in ambiguous and tag:
        parts.append(tag)
    if acquisition_word(base, tag, ambiguous) in shared:
        parts.append(provenance)
    return f"{base} ({', '.join(parts)})" if parts else base


def display_words(entries, ambiguous, shared):
    """Sorted, de-duplicated display strings for one panel."""
    return sorted(
        {display_word(base, tag, prov, ambiguous, shared) for base, tag, prov in entries}
    )


def _measure(ax, word, fontsize, renderer):
    """Width and height of ``word`` at ``fontsize``, in axes fractions."""
    artist = ax.text(0, 0, word, fontsize=fontsize, transform=ax.transAxes)
    bbox = artist.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(
        [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]]
    )
    artist.remove()
    return x1 - x0, y1 - y0


def column_layout(ax, words, fontsize, renderer):
    """Lay the words out as an alphabetical multi-column block, filled
    top-to-bottom down each column.

    Every column count that fits the panel is tried; the one whose block covers
    the most area wins, so the block fills the panel rather than stringing out
    into a single tall column. Returns (positions, line_height) with one
    (x_left, y_centre) per word in the input order, or None if no column count
    fits at this font size.
    """
    sizes = [_measure(ax, word, fontsize, renderer) for word in words]
    line_height = max(height for _, height in sizes) + PAD_Y

    best = None
    for n_cols in range(1, len(words) + 1):
        n_rows = ceil(len(words) / n_cols)
        # Skip counts that would leave a trailing column empty; the same shape is
        # reachable with fewer columns.
        if n_rows * (n_cols - 1) >= len(words):
            continue
        block_height = n_rows * line_height
        if block_height > 1.0:
            continue
        columns = [
            list(range(c * n_rows, min((c + 1) * n_rows, len(words))))
            for c in range(n_cols)
        ]
        widths = [max(sizes[i][0] for i in column) + PAD_X for column in columns]
        block_width = sum(widths)
        if block_width > 1.0:
            continue
        area = block_width * block_height
        if best is None or area > best[0]:
            best = (area, columns, widths, block_width, block_height, line_height)

    if best is None:
        return None
    _, columns, widths, block_width, block_height, line_height = best

    # Columns hang from the top of the panel, so every panel's first row starts
    # at the same height under its title however few entries it holds.
    positions = {}
    x_left = (1.0 - block_width) / 2
    y_top = 1.0
    for column_index, column in enumerate(columns):
        x = x_left + sum(widths[:column_index])
        for row_index, word_index in enumerate(column):
            positions[word_index] = (x, y_top - (row_index + 0.5) * line_height)
    return [positions[i] for i in range(len(words))], line_height


def fit_shared_fontsize(axes, word_lists, renderer):
    """Largest font size (to within FONT_TOL) at which EVERY panel lays out.

    One size for the whole figure, so block size reflects roster size rather
    than per-panel autofit. The binding panel is whichever holds the most words
    or the widest one.
    """
    lo, hi = FONT_LO, FONT_HI
    best = None
    while hi - lo > FONT_TOL:
        mid = (lo + hi) / 2
        layouts = [
            column_layout(ax, words, mid, renderer)
            for ax, words in zip(axes, word_lists)
        ]
        if any(layout is None for layout in layouts):
            hi = mid
        else:
            best = (mid, layouts)
            lo = mid
    if best is None:
        best = (
            FONT_LO,
            [
                column_layout(ax, words, FONT_LO, renderer)
                for ax, words in zip(axes, word_lists)
            ],
        )
    return best


def style_panel(ax, title, n_words):
    """Frame and title one panel. Called before tight_layout so the word fit is
    measured against the axes' final geometry."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        f"{title}  ({n_words} targets)",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=10,
    )


def place_words(ax, words, fontsize, layout):
    """Render one label cloud into an already-styled ``ax``."""
    positions, _ = layout
    for word, (x, y) in zip(words, positions):
        ax.text(
            x,
            y,
            word,
            fontsize=fontsize,
            color=C_TUMOR_LESION_LABEL if is_purple_label(word) else C_ANATOMY_LABEL,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect_indist", required=True)
    parser.add_argument("--detect_ood", required=True)
    parser.add_argument("--tl_indist", required=True)
    parser.add_argument("--tl_ood", required=True)
    parser.add_argument("--fig_dir", required=True)
    parser.add_argument("--fig_name", default="fig_OOD_label.pdf")
    parser.add_argument("--cell_width", type=float, default=8.0)
    parser.add_argument("--cell_height", type=float, default=8.0)
    parser.add_argument("--tl_cell_height", type=float, default=2.6)
    parser.add_argument("--save_as_png", action="store_true")
    parser.add_argument("--save_as_pdf", action="store_true")
    args = parser.parse_args()

    panels = [
        ("Detection", "In-distribution", args.detect_indist, "detect"),
        ("Detection", "Target-OOD", args.detect_ood, "detect"),
        ("T/L", "In-distribution", args.tl_indist, "tl"),
        ("T/L", "Target-OOD", args.tl_ood, "tl"),
    ]

    loaded = []
    for task, split, path, kind in panels:
        entries, n_tasks, n_samples = roster_entries(path, kind)
        loaded.append((task, split, entries, n_tasks, n_samples))

    # Acquisition splits are decided per task row, so a task's two panels agree.
    resolved = []
    for row in (0, 2):
        entries_indist, entries_ood = loaded[row][2], loaded[row + 1][2]
        ambiguous = split_ambiguous(entries_indist, entries_ood)
        # Whatever still reads the same in both panels after the acquisition
        # split is separated by provenance instead.
        acquisition_sets = [
            {acquisition_word(base, tag, ambiguous) for base, tag, _ in entries}
            for entries in (entries_indist, entries_ood)
        ]
        shared = acquisition_sets[0] & acquisition_sets[1]
        print(
            f"{loaded[row][0]}: {len(ambiguous)} label(s) split by acquisition: "
            f"{sorted(ambiguous)}"
        )
        print(
            f"{loaded[row][0]}: {len(shared)} label(s) in both panels, "
            f"tagged by provenance: {sorted(shared)}"
        )
        for offset in (0, 1):
            task, split, entries, n_tasks, n_samples = loaded[row + offset]
            words = display_words(entries, ambiguous, shared)
            resolved.append((task, split, words))
            print(
                f"{task:>9} {split:<16} {len(words):>3} targets "
                f"from {n_tasks:>2} tasks / {n_samples:,} samples"
            )
            for word in words:
                print(f"              {word}")

    for row in (0, 2):
        identical = sorted(set(resolved[row][2]) & set(resolved[row + 1][2]))
        print(
            f"\n{resolved[row][0]}: {len(identical)} target(s) still drawn "
            f"identically in both panels: {identical}"
        )

    # The rows get different heights. One shared font size means the Detection
    # row - 78 alphabetical entries against the T/L row's 9 - is what binds the
    # whole figure, and at equal row heights no column count fits it above
    # ~5 pt: two columns overflow the height, three overflow the width. Giving
    # the Detection row the bulk of the page lets it use two columns and lifts
    # the shared size for every panel, while the short T/L row is exactly as
    # tall as its handful of entries needs.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(2 * args.cell_width, args.cell_height + args.tl_cell_height),
        gridspec_kw={"height_ratios": [args.cell_height, args.tl_cell_height]},
    )

    flat_axes = list(axes.ravel())
    word_lists = [words for _, _, words in resolved]
    for ax, (task, split, words) in zip(flat_axes, resolved):
        style_panel(ax, f"{task} - {split}", len(words))
    fig.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fontsize, layouts = fit_shared_fontsize(flat_axes, word_lists, renderer)
    print(f"\nShared font size across all panels: {fontsize:.1f} pt")

    for ax, (task, split, words), layout in zip(flat_axes, resolved, layouts):
        place_words(ax, words, fontsize, layout)

    os.makedirs(args.fig_dir, exist_ok=True)
    stem = os.path.splitext(args.fig_name)[0]
    if args.save_as_pdf:
        path = os.path.join(args.fig_dir, f"{stem}.pdf")
        save_fig_capped(path, fig=fig, bbox_inches="tight", transparent=True)
        print(f"Wrote {path}")
    if args.save_as_png:
        path = os.path.join(args.fig_dir, f"{stem}.png")
        save_fig_capped(path, fig=fig, bbox_inches="tight", transparent=True)
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
