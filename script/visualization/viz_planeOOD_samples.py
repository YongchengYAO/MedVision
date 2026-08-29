"""
Render plane-OOD comparison subfigures: the same volume + target seen in the in-distribution
plane and in both out-of-distribution planes.

MedVision-V0 was SFT'd on **axial slices only**, so coronal and sagittal are the plane-OOD
evaluation planes (see script/ablation/OOD/eval__MedVision-V0-7B__*__planeOOD.sh). The rosters
that define the split live in tasks_list/:

    in-distribution   tasks_list/tasks_MedVision-{detect,TL}-CoT.json          (*_Axial-CoT)
    plane-OOD         tasks_list/OOD/tasks_MedVision-{detect,TL}-CoT-planeOOD.json
                                                                (*_Sagittal-CoT, *_Coronal-CoT)

Only Detection and Tumor/Lesion-Size have an OOD split, so A/D is not handled here.

Like viz_gt_annotations.py this reads nothing but the data folder -- no model output, no
HuggingFace build. Annotations come from the on-disk benchmark plans, and panels are drawn with
the same house helpers (normalize_img + plot_utils), so the output is stylistically identical to
the GT annotation figure.

    Detection  bboxes live in benchmark_plan_detection_*.json.gz, under
               <case>.slice_profiles_{x,y,z}[k].slice_profile[j].bboxes (array space).
    T/L        axis endpoints P1-P4 live in <landmark_folder>/<case>.json.gz; the plan's slice
               entry carries n_total_clusters and the per-cluster major/minor lengths in mm.

RAS+ storage fixes the plane <-> axis convention (plan_utils.AXIS_TO_PLANE):
    slice_dim 0 = x = Sagittal (OOD)
    slice_dim 1 = y = Coronal  (OOD)
    slice_dim 2 = z = Axial    (in-distribution)

── Output tree, and why it is shaped this way ────────────────────────────────
One tree per task, with the PLANE occupying the level viz_compile_grid.py calls "model":

    <fig_dir>/Detection/Axial (ID)/KiTS23/KiTS23__Task01__case_00123__kidney.pdf
    <fig_dir>/Detection/Coronal (OOD)/KiTS23/KiTS23__Task01__case_00123__kidney.pdf
    <fig_dir>/Detection/Sagittal (OOD)/KiTS23/KiTS23__Task01__case_00123__kidney.pdf
    <fig_dir>/Tumor-Lesion-Size/...

The filename is the PAIRING KEY -- dataset, task, case and target, with NO slice index -- and is
byte-identical across the three plane folders. That is what makes the compiled figure line up:

  * viz_compile_grid.py's default layout mode builds ONE ordered sample list and reuses it for
    every model block, varying only the folder prefix (model / dataset / filename). Identical
    filenames therefore put the same volume + target in the same column of every plane block,
    which is requirement "OOD planes of one volume share a column".
  * _select_samples INTERSECTS filenames across model folders, so a volume that failed to render
    in one plane drops out of all three rather than producing a misaligned column. This script
    already only emits complete triples, so the intersection is a belt-and-braces check.
  * models are taken in sorted() order and the folder name IS the rotated row label, so the names
    above sort in-distribution first (A < C < S) with no change to the compositor.

--num_col x --num_row_per_type triples are selected round-robin across datasets, so the widest
possible spread of datasets is shown and the rendered count matches the grid exactly. Groups are
kept only when the same (case, target) yields a usable slice in ALL THREE planes.

The displayed slice per plane is the one whose annotation is LARGEST in physical units (bbox
area in mm^2 for Detection, major-axis length in mm for T/L). Picking per plane rather than
reusing one 3D structure's centroid keeps every panel non-degenerate, which matters far more
here than in the GT figure: a column is only readable if all three panels show the target well.

Annotation version is the usual ceiling ("newest plan published at or before --version"), so at
--version 1.1.1 detection resolves back to v1.0.0 while T/L resolves to a real v1.1.1. T/L needs
1.1.1 -- that is the release which added sagittal and coronal T/L slices, without which the OOD
rows would be empty. plan_utils announces every fallback on stderr.

Output formats:
    - No flags -> ["pdf"] (default; viz_compile_grid.py reads pdf by default).
    - --save_as_png / --save_as_pdf -> one file per requested format.

See viz_planeOOD_samples.sh for the two-stage render + compile driver.
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    _load_nifti_2d,
)
from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.configs import SEED
from medvision_bm.utils.plan_utils import (
    load_benchmark_plan,
    resolve_plan_path,
    split_cases,
)
from medvision_bm.utils.plot_utils import (
    plot_detection_on_image,
    plot_tl_axes_on_image,
)

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(_REPO_DIR, "Data", "Datasets")
DEFAULT_TASKS_LIST_DIR = os.path.join(_REPO_DIR, "tasks_list")

# (axis letter in the plan's slice_profiles_* key, slice_dim int used everywhere else)
_AXES = (("x", 0), ("y", 1), ("z", 2))
_LM_KEY = {0: "slice_landmarks_x", 1: "slice_landmarks_y", 2: "slice_landmarks_z"}

# slice_dim -> folder name, which becomes the rotated row-block label in the compiled figure.
# Sorted order decides the row order, and "Axial (ID)" < "Coronal (OOD)" < "Sagittal (OOD)"
# puts the in-distribution block on top, as required.
PLANE_DIR = {2: "Axial (ID)", 1: "Coronal (OOD)", 0: "Sagittal (OOD)"}
ID_SLICE_DIM = 2
OOD_SLICE_DIMS = (1, 0)
ALL_SLICE_DIMS = (ID_SLICE_DIM,) + OOD_SLICE_DIMS

# Per task: output folder, roster filenames, the token used in roster task keys, and the plan
# family to read. Only these two tasks have an OOD split.
TASKS = {
    "Detection": {
        "dir": "Detection",
        "roster_id": "tasks_MedVision-detect-CoT.json",
        "roster_ood": os.path.join("OOD", "tasks_MedVision-detect-CoT-planeOOD.json"),
        "roster_token": "BoxCoordinate",
        "plan_type": "detection",
        "task_type": "Box-Size",
    },
    "Tumor-Lesion-Size": {
        "dir": "Tumor-Lesion-Size",
        "roster_id": "tasks_MedVision-TL-CoT.json",
        "roster_ood": os.path.join("OOD", "tasks_MedVision-TL-CoT-planeOOD.json"),
        "roster_token": "TumorLesionSize",
        "plan_type": "biometry",
        "task_type": "Tumor-Lesion-Size",
    },
}

_ROSTER_KEY_RE = re.compile(r"^(?P<dataset>.+?)_(?P<token>[A-Za-z]+)_Task(?P<task>\d+)_"
                            r"(?P<plane>Axial|Coronal|Sagittal)-CoT$")


def _load_json_gz(path):
    """Load a JSON or gzip-compressed JSON file."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _project_3d_to_2d(coord, slice_dim):
    """Drop the slice axis from a 3D voxel coord, giving [idx_dim0, idx_dim1] in array space."""
    if slice_dim == 0:
        return [coord[1], coord[2]]
    if slice_dim == 1:
        return [coord[0], coord[2]]
    return [coord[0], coord[1]]


def _display_aspect(case, slice_dim):
    """Physical width/height of the rendered panel, or None when geometry is missing.

    plot_utils draws imshow(img.T, aspect=pixel_sizes[1]/pixel_sizes[0]), so the panel's aspect
    is the PHYSICAL extent ratio, not the array-shape ratio. This guard bites much harder here
    than in the GT figure: the OOD planes of a thin-slab volume are exactly the degenerate ones.
    """
    info = case.get("image_file_info") or {}
    array_size, voxel_size = info.get("array_size"), info.get("voxel_size")
    if not array_size or not voxel_size:
        return None
    d0, d1 = [i for i in range(3) if i != slice_dim]
    denom = array_size[d1] * voxel_size[d1]
    if not denom:
        return None
    return (array_size[d0] * voxel_size[d0]) / denom


def _aspect_ok(case, slice_dim, max_aspect):
    """True when the panel is not so anisotropic that it renders as an uninformative stripe."""
    if not max_aspect:
        return True
    aspect = _display_aspect(case, slice_dim)
    if aspect is None:
        return True
    return (1.0 / max_aspect) <= aspect <= max_aspect


def _safe(text):
    """Filename-safe token. Dots are stripped because the plot helpers split the output path on
    os.path.splitext, which would otherwise truncate names like 'AbdomenAtlas1.0Mini'."""
    return str(text).replace(".", "-").replace("/", "-").replace(" ", "-")


def _slice_landmarks(dataset_dir, landmark_file, slice_dim, slice_idx):
    """Return the merged landmark mapping for one T/L slice, or None.

    T/L landmark files store a LIST of per-cluster dicts; only single-cluster slices are ever
    requested here, so element 0 is unambiguous. Matching entries are merged rather than the
    first one taken, mirroring viz_gt_annotations._slice_landmarks.
    """
    data = _load_json_gz(os.path.join(dataset_dir, landmark_file))
    merged = {}
    for entry in data.get(_LM_KEY[slice_dim], []) or []:
        if entry.get("slice_idx") != slice_idx:
            continue
        lms = entry.get("landmarks") or {}
        if isinstance(lms, list):
            if not lms:
                continue
            lms = lms[0]
        merged.update(lms)
    return merged or None


# ── Roster ───────────────────────────────────────────────────────────────────


def load_roster(tasks_list_dir, task_cfg):
    """(dataset, task_ID) pairs benchmarked in the in-distribution AND both OOD planes.

    Returns {dataset: {task_ID, ...}}. A pair is kept only when the axial task exists in the
    in-distribution roster and BOTH the sagittal and coronal tasks exist in the OOD roster, so
    the figure can never show a plane the OOD evaluation never ran.
    """
    def planes_of(path, token):
        found = defaultdict(set)
        for key in _load_json_gz(path):
            m = _ROSTER_KEY_RE.match(key)
            if not m or m.group("token") != token:
                continue
            found[(m.group("dataset"), m.group("task"))].add(m.group("plane"))
        return found

    token = task_cfg["roster_token"]
    id_planes = planes_of(os.path.join(tasks_list_dir, task_cfg["roster_id"]), token)
    ood_planes = planes_of(os.path.join(tasks_list_dir, task_cfg["roster_ood"]), token)

    roster = defaultdict(set)
    for pair, planes in id_planes.items():
        if "Axial" not in planes:
            continue
        if {"Sagittal", "Coronal"} <= ood_planes.get(pair, set()):
            roster[pair[0]].add(pair[1])
    return roster


# ── Candidate enumeration ────────────────────────────────────────────────────
#
# A "group" is one (dataset, task_ID, case_ID, target) -- i.e. one volume + one target, which is
# the unit that becomes a column. For each group we keep the best slice in each of the three
# planes, and drop the group unless all three are present.


def _best_detection_slices(case, max_aspect):
    """{label: {slice_dim: (slice_idx, score)}} keeping the largest single-cluster bbox per plane.

    Score is the bbox's PHYSICAL area in mm^2 (bbox["sizes"] is already in mm), so the choice is
    comparable across planes with different voxel spacing.
    """
    best = defaultdict(dict)
    for axis, slice_dim in _AXES:
        if not _aspect_ok(case, slice_dim, max_aspect):
            continue
        for entry in case.get(f"slice_profiles_{axis}") or []:
            slice_idx = entry["slice_idx"]
            for label_profile in entry.get("slice_profile") or []:
                boxes = label_profile.get("bboxes") or []
                if len(boxes) != 1:  # skip multi-cluster slices
                    continue
                box = boxes[0]
                sizes = box.get("sizes") or [0, 0]
                score = float(sizes[0]) * float(sizes[1])
                label = int(label_profile["label"])
                prev = best[label].get(slice_dim)
                if prev is None or score > prev[1]:
                    best[label][slice_dim] = (slice_idx, score, box)
    return best


def _best_tl_slices(case, max_aspect):
    """{slice_dim: (slice_idx, score)} keeping the longest-major-axis single-cluster slice.

    The biometry slice_profile is nested one level deeper than detection's: a list per cluster,
    each holding that cluster's metric dicts (major "L-1-2" and minor "L-3-4", both in mm). Only
    n_total_clusters == 1 slices are used, so cluster 0 is the whole story, and the major axis is
    simply the larger of the two lengths.
    """
    best = {}
    for axis, slice_dim in _AXES:
        if not _aspect_ok(case, slice_dim, max_aspect):
            continue
        for entry in case.get(f"slice_profiles_{axis}") or []:
            if entry.get("n_total_clusters") != 1:  # skip multi-cluster slices
                continue
            clusters = entry.get("slice_profile") or []
            if not clusters:
                continue
            metrics = clusters[0]
            if not isinstance(metrics, list):  # defensive: unexpected schema
                continue
            lengths = [
                float(m.get("metric_value", 0.0))
                for m in metrics
                if m.get("metric_type") == "distance"
            ]
            if not lengths:
                continue
            score = max(lengths)
            prev = best.get(slice_dim)
            if prev is None or score > prev[1]:
                best[slice_dim] = (entry["slice_idx"], score)
    return best


def _target_name(labels_map, label):
    """Human-readable target for the pairing key, falling back to 'labelN'."""
    if labels_map:
        name = labels_map.get(str(label)) or labels_map.get(label)
        if name:
            return _safe(name)
    return f"label{label}"


def _collect_detection(dataset, plan, task_ids, split, max_aspect, pool, rng):
    """Complete (case, label) groups for one dataset's detection plan."""
    groups = []
    for task in plan["tasks"]:
        task_id = task["task_ID"]
        if task_id not in task_ids:
            continue
        labels_map = task.get("labels_map") or {}
        cases = list(split_cases(task, split))
        # Shuffled so the early-exit below samples the dataset rather than its first few cases.
        rng.shuffle(cases)
        for case in cases:
            if len(groups) >= pool:
                break
            for label, per_dim in _best_detection_slices(case, max_aspect).items():
                if not all(d in per_dim for d in ALL_SLICE_DIMS):
                    continue
                box = {d: per_dim[d][2] for d in ALL_SLICE_DIMS}
                groups.append(
                    {
                        "dataset": dataset,
                        "task_id": task_id,
                        "task_type": "Box-Size",
                        "case_id": case["case_ID"],
                        "image_file": case["image_file"],
                        "label": label,
                        "target": _target_name(labels_map, label),
                        "slices": {d: per_dim[d][0] for d in ALL_SLICE_DIMS},
                        "gt_box": {
                            d: [
                                box[d]["min_coords"][0],
                                box[d]["min_coords"][1],
                                box[d]["max_coords"][0],
                                box[d]["max_coords"][1],
                            ]
                            for d in ALL_SLICE_DIMS
                        },
                    }
                )
    return groups


def _collect_tl(dataset, plan, task_ids, split, max_aspect, pool, rng):
    """Complete (case,) groups for one dataset's T/L tasks; the target is the task's label."""
    groups = []
    for task in plan["tasks"]:
        task_id = task["task_ID"]
        if task_id not in task_ids or task.get("target_label") is None:
            continue
        target_label = int(task["target_label"])
        labels_map = task.get("labels_map") or {}
        cases = list(split_cases(task, split))
        rng.shuffle(cases)
        for case in cases:
            if len(groups) >= pool:
                break
            if not case.get("landmark_file"):
                continue
            per_dim = _best_tl_slices(case, max_aspect)
            if not all(d in per_dim for d in ALL_SLICE_DIMS):
                continue
            groups.append(
                {
                    "dataset": dataset,
                    "task_id": task_id,
                    "task_type": "Tumor-Lesion-Size",
                    "case_id": case["case_ID"],
                    "image_file": case["image_file"],
                    "landmark_file": case["landmark_file"],
                    "mask_file": case.get("mask_file"),
                    "label": target_label,
                    "target": _target_name(labels_map, target_label),
                    "slices": {d: per_dim[d][0] for d in ALL_SLICE_DIMS},
                }
            )
    return groups


def collect_groups(
    data_dir, version, split, task_key, roster, datasets, pool_per_dataset, rng,
    max_plan_mb=0, max_aspect=0.0,
):
    """{dataset: [group, ...]} of volume+target groups usable in all three planes.

    One plan is resident at a time (cache cleared per dataset): detection plans store every bbox
    on every slice, and the roster includes AbdomenAtlas1.0Mini at ~600 MB gzipped. ``max_plan_mb``
    > 0 skips (and reports) plans above that size; 0 means no cap.
    """
    cfg = TASKS[task_key]
    plan_type = cfg["plan_type"]
    collect = _collect_detection if task_key == "Detection" else _collect_tl

    by_dataset, skipped = {}, []
    for dataset in datasets:
        task_ids = roster.get(dataset)
        if not task_ids:
            continue
        dataset_dir = os.path.join(data_dir, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"  {dataset:22s} MISSING under {data_dir}", file=sys.stderr)
            continue

        plan_path = resolve_plan_path(dataset_dir, plan_type, version)
        if not plan_path:
            print(f"  {dataset:22s} no {plan_type} plan at or before v{version}", file=sys.stderr)
            continue
        plan_mb = os.path.getsize(plan_path) / 1e6
        if max_plan_mb and plan_mb > max_plan_mb:
            skipped.append((dataset, plan_mb))
            continue

        plan = load_benchmark_plan(dataset_dir, plan_type, version)
        if plan:
            found = collect(dataset, plan, task_ids, split, max_aspect, pool_per_dataset, rng)
            if found:
                by_dataset[dataset] = found
            print(f"  {dataset:22s} {len(found)} complete triple(s)  [{plan_mb:.0f} MB plan]")
        # Drop the plan before the next dataset: two giant detection plans resident at once
        # would peak well above 100 GB.
        load_benchmark_plan.cache_clear()

    if skipped:
        print(
            f"\n[skipped] {len(skipped)} plan(s) above --max_plan_mb "
            f"(re-run with --max_plan_mb 0 to include them):",
            file=sys.stderr,
        )
        for dataset, mb in skipped:
            print(f"    {dataset} ({mb:.0f} MB gz)", file=sys.stderr)
    return by_dataset


def order_round_robin(by_dataset, rng):
    """All groups, ordered by cycling datasets so the columns span as many datasets as possible.

    Round-robin (rather than an even per-dataset quota) keeps the rendered count exactly equal to
    the grid size: viz_compile_grid.py refuses a --limit_subfigures below the number of dataset
    folders, and since at most n_needed datasets can contribute one group each, that can't happen.

    The FULL order is returned rather than the first n_needed, so the caller can walk past a group
    that fails to render and still fill the grid -- and the replacement is still drawn round-robin,
    so one bad volume does not collapse the dataset spread.

    The DATASET visiting order is shuffled (seeded), not alphabetical. With fewer columns than
    datasets -- the normal case for detection, which has 17 -- cycling sorted() would always take
    the alphabetically first N, i.e. AMOS22, AbdomenAtlas1.0Mini, AbdomenCT-1K, BCV15 ... which is
    three abdominal CT sets in a row. Pass --datasets to pick them by hand instead.
    """
    order = sorted(by_dataset)
    rng.shuffle(order)
    pools = {}
    for dataset in order:
        pool = list(by_dataset[dataset])
        rng.shuffle(pool)
        pools[dataset] = pool

    ordered = []
    while any(pools.values()):
        for dataset in order:
            if pools[dataset]:
                ordered.append(pools[dataset].pop())
    return ordered


# ── Rendering ────────────────────────────────────────────────────────────────


def pairing_key(group):
    """Filename stem shared by a group's three plane panels.

    Deliberately carries NO slice index: the three panels come from different slices, and it is
    the identical filename that makes viz_compile_grid.py place them in one column.
    """
    return "__".join(
        [
            _safe(group["dataset"]),
            f"Task{_safe(group['task_id'])}",
            _safe(group["case_id"]),
            group["target"],
        ]
    )


def render_panel(group, slice_dim, data_dir, task_root, show_mask, formats):
    """Draw one plane's panel for one group. Raises on failure; the caller counts and reports."""
    dataset_dir = os.path.join(data_dir, group["dataset"])
    slice_idx = group["slices"][slice_dim]

    pixel_sizes, img_2d = _load_nifti_2d(
        os.path.join(dataset_dir, group["image_file"]), slice_dim, slice_idx
    )
    doc = {
        "dataset_name": group["dataset"],
        "taskID": group["task_id"],
        "taskType": group["task_type"],
        "label": group["label"],
    }
    img_2d = normalize_img(doc, img_2d)

    fig_path = os.path.join(
        task_root, PLANE_DIR[slice_dim], group["dataset"], pairing_key(group) + ".png"
    )

    if group["task_type"] == "Box-Size":
        plot_detection_on_image(
            image_2d=img_2d,
            pixel_sizes=pixel_sizes,
            gt_box=group["gt_box"][slice_dim],
            pred_box=None,
            slice_dim=slice_dim,
            slice_idx=slice_idx,
            fig_path=fig_path,
            formats=formats,
        )
        return

    lms = _slice_landmarks(dataset_dir, group["landmark_file"], slice_dim, slice_idx)
    if lms is None:
        raise ValueError("landmarks not found for slice")

    mask_2d = None
    if show_mask and group.get("mask_file"):
        _, mask_raw = _load_nifti_2d(
            os.path.join(dataset_dir, group["mask_file"]), slice_dim, slice_idx
        )
        mask_2d = (mask_raw == group["label"]).astype(np.float32)

    plot_tl_axes_on_image(
        image_2d=img_2d,
        pixel_sizes=pixel_sizes,
        major_axis_pts=None,
        minor_axis_pts=None,
        slice_dim=slice_dim,
        slice_idx=slice_idx,
        fig_path=fig_path,
        mask_2d=mask_2d,
        gt_major_pts=[
            _project_3d_to_2d(lms["P1"], slice_dim),
            _project_3d_to_2d(lms["P2"], slice_dim),
        ],
        gt_minor_pts=[
            _project_3d_to_2d(lms["P3"], slice_dim),
            _project_3d_to_2d(lms["P4"], slice_dim),
        ],
        formats=formats,
    )


def render_group(group, data_dir, task_root, show_mask, formats):
    """Draw all three planes of one group, or none of them.

    Panels are written to a staging list and only committed once every plane has succeeded, so a
    half-rendered group can never leave a stray file that would break the column pairing. (The
    compositor's filename intersection would drop it anyway; failing atomically keeps the on-disk
    tree honest and the counts meaningful.)
    """
    written = []
    try:
        for slice_dim in ALL_SLICE_DIMS:
            render_panel(group, slice_dim, data_dir, task_root, show_mask, formats)
            stem = os.path.join(
                task_root, PLANE_DIR[slice_dim], group["dataset"], pairing_key(group)
            )
            written.extend(f"{stem}.{fmt}" for fmt in formats)
    except Exception:
        for path in written:
            if os.path.exists(path):
                os.remove(path)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Render plane-OOD (axial vs coronal/sagittal) MedVision subfigures."
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Root of Data/Datasets")
    parser.add_argument(
        "--tasks_list_dir",
        default=DEFAULT_TASKS_LIST_DIR,
        help="Root holding the in-distribution rosters and the OOD/ subfolder",
    )
    parser.add_argument(
        "--version",
        default="1.1.1",
        help="Annotation version pin, resolved by the ceiling rule. Must be >= 1.1.1 for T/L: "
        "that release added the sagittal and coronal T/L slices (default: 1.1.1)",
    )
    parser.add_argument(
        "--fig_dir",
        default=os.path.join(_REPO_DIR, "Figures", "planeOOD-samples"),
        help="Output base directory; panels land in <fig_dir>/<task>/<plane>/<dataset>/",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(TASKS),
        help=f"Comma-separated subset of {list(TASKS)} (OOD is defined for these two only)",
    )
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated dataset filter (default: whole roster)"
    )
    parser.add_argument(
        "--num_col", type=int, default=6, help="Columns in the compiled figure (default: 6)"
    )
    parser.add_argument(
        "--num_row_per_type",
        type=int,
        default=1,
        help="Rows per plane type (default: 1). The compiled figure has 3 x this many rows: the "
        "in-distribution axial block on top, then the two OOD plane blocks.",
    )
    parser.add_argument(
        "--pool_per_dataset",
        type=int,
        default=0,
        help="Groups enumerated per dataset before selection; 0 (default) uses "
        "num_col * num_row_per_type, which is enough for any round-robin draw.",
    )
    parser.add_argument(
        "--show_mask",
        action="store_true",
        help="For T/L, also draw the GT mask contour the ellipse was fitted to",
    )
    parser.add_argument(
        "--max_plan_mb",
        type=float,
        default=0,
        help="Skip plans larger than this gzipped size in MB, reporting which were dropped. "
        "Parsing a detection plan costs order 250x its gzipped size in RAM (measured: the 59 MB "
        "AbdomenCT-1K plan peaked at ~16 GB), so the roster's big three -- AbdomenAtlas1.0Mini "
        "(~600 MB), BraTS24 (~290 MB), MSD (~220 MB) -- cannot be parsed on a 32 GB box and get "
        "OOM-KILLED with no traceback, the log simply stopping mid-dataset. Check the CGROUP "
        "limit (/sys/fs/cgroup/memory.max), NOT `free`, which reports the host. 100 keeps the "
        "other 14 detection datasets and peaks near 16 GB; do not raise it much above that "
        "without checking your ceiling. 0 (default) means no cap.",
    )
    parser.add_argument(
        "--max_aspect",
        type=float,
        default=4.0,
        help="Skip slices whose rendered panel would be more anisotropic than this ratio. The "
        "compositor stretches every panel into a square cell, so a 25:1 panel becomes an "
        "uninformative stripe -- and the OOD planes of a thin-slab volume are exactly those. "
        "Judged on PHYSICAL extent. 0 disables the guard.",
    )
    parser.add_argument("--save_as_png", action="store_true", help="Save panels as PNG.")
    parser.add_argument("--save_as_pdf", action="store_true", help="Save panels as PDF.")
    parser.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    formats = [
        f for f, on in (("png", args.save_as_png), ("pdf", args.save_as_pdf)) if on
    ] or ["pdf"]

    task_keys = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in task_keys if t not in TASKS]
    if unknown:
        parser.error(f"unknown --tasks entries {unknown}; choose from {list(TASKS)}")

    n_needed = args.num_col * args.num_row_per_type
    pool = args.pool_per_dataset or n_needed
    dataset_filter = (
        {d.strip() for d in args.datasets.split(",") if d.strip()} if args.datasets else None
    )

    print(
        f"Grid: {args.num_col} col x {args.num_row_per_type} row per plane type "
        f"-> {n_needed} volume+target group(s), {3 * args.num_row_per_type} rows"
    )
    print("Row order (sorted folder names): " + " / ".join(PLANE_DIR[d] for d in ALL_SLICE_DIMS))

    total_ok, total_fail = 0, []
    for task_key in task_keys:
        cfg = TASKS[task_key]
        # A fresh RNG per task keeps each task's selection independent of the other's presence.
        rng = random.Random(args.seed if args.seed is not None else SEED)

        roster = load_roster(args.tasks_list_dir, cfg)
        datasets = sorted(roster)
        if dataset_filter is not None:
            datasets = [d for d in datasets if d in dataset_filter]
        print(
            f"\n[{task_key}] roster: {len(datasets)} dataset(s) benchmarked in all three planes "
            f"at version ceiling v{args.version} ({args.split} split)"
        )
        if not datasets:
            print(f"  no datasets left after filtering -- skipping {task_key}", file=sys.stderr)
            continue

        by_dataset = collect_groups(
            args.data_dir, args.version, args.split, task_key, roster, datasets,
            pool, rng, args.max_plan_mb, args.max_aspect,
        )
        ordered = order_round_robin(by_dataset, rng)
        if not ordered:
            print(f"  no complete plane triples found for {task_key}", file=sys.stderr)
            continue

        task_root = os.path.join(args.fig_dir, cfg["dir"])
        n_ok, failures = 0, []
        # Walk the round-robin order until the grid is full, so a group that fails to render is
        # replaced rather than leaving a hole the compile stage would refuse.
        progress = tqdm(total=n_needed, desc=f"  {task_key}", leave=False)
        for group in ordered:
            if n_ok >= n_needed:
                break
            try:
                render_group(group, args.data_dir, task_root, args.show_mask, formats)
                n_ok += 1
                progress.update(1)
            except Exception as e:  # noqa: BLE001 - report and continue
                failures.append((task_key, group["dataset"], group["case_id"], repr(e)))
        progress.close()

        if n_ok < n_needed:
            print(
                f"  only {n_ok}/{n_needed} groups rendered from a pool of {len(ordered)}; the "
                f"compiled grid would have empty cells. Lower --num_col/--num_row_per_type, "
                f"raise --pool_per_dataset or --max_aspect, or drop --max_plan_mb.",
                file=sys.stderr,
            )
        print(
            f"  rendered {n_ok}/{n_needed} group(s) x {len(ALL_SLICE_DIMS)} planes "
            f"into {task_root}"
        )
        for slice_dim in ALL_SLICE_DIMS:
            d = os.path.join(task_root, PLANE_DIR[slice_dim])
            n = sum(len(files) for _, _, files in os.walk(d)) if os.path.isdir(d) else 0
            print(f"    {PLANE_DIR[slice_dim]:18s} {n}")
        total_ok += n_ok
        total_fail.extend(failures)

    print(f"\nRendered {total_ok} complete plane triple(s) into {args.fig_dir}")
    if total_fail:
        print(f"\n{len(total_fail)} failures:", file=sys.stderr)
        for f in total_fail[:20]:
            print("  " + " | ".join(map(str, f)), file=sys.stderr)


if __name__ == "__main__":
    main()
