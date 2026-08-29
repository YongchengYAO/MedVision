"""
Render ground-truth-only MedVision annotation subfigures straight from the data folder.

Unlike the per-task response visualizers in this folder (viz_detection_boxes.py, viz_tl_axes.py,
viz_ad_landmarks.py), this script reads no model output at all: it walks
<data_dir>/<dataset>/benchmark_plan_*.json.gz plus the per-case landmark JSONs and draws the
normalized image with only the GT overlay, in the same house viz format.

Everything needed is already on disk, so no HuggingFace dataset build is involved:
    Detection  bboxes live in benchmark_plan_detection_*.json.gz, under
               <case>.slice_profiles_{x,y,z}[k].slice_profile[j].bboxes (array space).
    T/L        axis endpoints P1-P4 live in <landmark_folder>/<case>.json.gz, as a LIST of
               per-cluster dicts; the plan's slice entry carries n_total_clusters.
    A/D        landmarks live in Landmarks/<case>.json.gz as a DICT; which of them form the
               measured line(s) is resolved through the plan's lines_map / angles_map.
A biometry task is T/L when it has a "target_label" key, and A/D when it does not.

Annotation version is resolved with plan_utils.resolve_plan_path's ceiling rule -- "the newest
plan published at or before --version". This matters because a family may never have published
the pinned version: at --version 1.1.1 the detection plans resolve back to v1.0.0 (no detection
plan exists at 1.1.x) while T/L resolves to a real v1.1.1. Fallbacks are announced on stderr by
plan_utils itself, so the version actually used is never silent.

The lines_map / angles_map lookup is deliberately done against the ON-DISK task dict at the
resolved version, NOT via viz_ad_landmarks._get_benchmark_plan -- that helper imports
benchmark_plan from the installed medvision_ds package, which is version-blind and would mix
annotation versions into a run that is supposed to be pinned.

Output is one subfigure per sample, grouped into folders named after the task:

    <fig_dir>/GT/Detection/<stem>.pdf
    <fig_dir>/GT/Tumor-Lesion-Size/<stem>.pdf
    <fig_dir>/GT/Distance/<stem>.pdf
    <fig_dir>/GT/Angle/<stem>.pdf

viz_compile_grid.py takes its row label from that folder name, so feeding this tree to
    viz_compile_grid.py --dir_subfigures <fig_dir> --dir_model GT --dataset_as_row
produces a single figure with one labelled row per task. See viz_gt_annotations.sh.

Output formats:
    - No flags -> ["pdf"] (default; viz_compile_grid.py reads pdf by default).
    - --save_as_png / --save_as_pdf -> one file per requested format.
"""

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from medvision_bm.medvision_lmms_eval.lmms_eval.tasks.medvision.medvision_utils import (
    _load_nifti_2d,
)
from medvision_bm.sft.sft_utils import normalize_img
from medvision_bm.utils.configs import AD_NEAR_ZERO_GT_THRESHOLD, SEED
from medvision_bm.utils.plan_utils import (
    load_benchmark_plan,
    resolve_plan_path,
    split_cases,
)
from medvision_bm.utils.plot_utils import (
    plot_ad_on_image,
    plot_detection_on_image,
    plot_tl_axes_on_image,
)

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(_REPO_DIR, "Data", "Datasets")

# (axis letter in the plan's slice_profiles_* key, slice_dim int used everywhere else)
_AXES = (("x", 0), ("y", 1), ("z", 2))
_LM_KEY = {0: "slice_landmarks_x", 1: "slice_landmarks_y", 2: "slice_landmarks_z"}

# Folder names, which become the row labels in the compiled figure.
G_DET, G_TL, G_DIST, G_ANG = "Detection", "Tumor-Lesion-Size", "Distance", "Angle"
GROUPS = (G_DET, G_TL, G_DIST, G_ANG)


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
    is the PHYSICAL extent ratio (H*ph)/(W*pw), not the array-shape ratio. That distinction
    matters: CrossMoDA's 512x120 sagittal array looks alarming but its 120 slices are 1.5 mm
    each, giving aspect 1.17 -- a perfectly square-ish panel. The degenerate ones are volumes
    with very few slices (ACDC has 9, CAMUS 14), which reach 25:1 and render as bare stripes.
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
    """Return the merged landmark mapping for one slice, or None.

    A slice can appear in MORE THAN ONE entry: FeTA24 writes one entry per biometric
    measurement, each holding only that measurement's landmarks (e.g. {P1,P2} in one entry and
    {P3,P4} in the next, for the same slice), so matching entries must be merged rather than the
    first one taken. This mirrors viz_tl_axes._extract_gt_axis_pts.

    T/L landmark files store a LIST of per-cluster dicts; A/D files store a single dict. Callers
    only ever ask for single-cluster T/L slices, so taking element 0 of a list is unambiguous.
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


# ── Candidate enumeration ────────────────────────────────────────────────────


def _enum_detection(dataset, dataset_dir, plan, split, max_aspect):
    """One candidate per (case, slice, label) carrying exactly one bbox."""
    out = []
    for task in plan["tasks"]:
        task_id = task["task_ID"]
        for case in split_cases(task, split):
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
                        out.append(
                            {
                                "group": G_DET,
                                "task_type": "Box-Size",
                                "dataset": dataset,
                                "task_id": task_id,
                                "case_id": case["case_ID"],
                                "image_file": case["image_file"],
                                "slice_dim": slice_dim,
                                "slice_idx": slice_idx,
                                "label": int(label_profile["label"]),
                                "gt_box": [
                                    box["min_coords"][0],
                                    box["min_coords"][1],
                                    box["max_coords"][0],
                                    box["max_coords"][1],
                                ],
                            }
                        )
    return out


def _enum_tl(dataset, dataset_dir, task, split, max_aspect):
    """One candidate per single-cluster T/L slice."""
    out = []
    task_id = task["task_ID"]
    target_label = int(task["target_label"])
    for case in split_cases(task, split):
        landmark_file = case.get("landmark_file")
        if not landmark_file:
            continue
        for axis, slice_dim in _AXES:
            if not _aspect_ok(case, slice_dim, max_aspect):
                continue
            for entry in case.get(f"slice_profiles_{axis}") or []:
                if entry.get("n_total_clusters") != 1:  # skip multi-cluster slices
                    continue
                out.append(
                    {
                        "group": G_TL,
                        "task_type": "Tumor-Lesion-Size",
                        "dataset": dataset,
                        "task_id": task_id,
                        "case_id": case["case_ID"],
                        "image_file": case["image_file"],
                        "landmark_file": landmark_file,
                        "mask_file": case.get("mask_file"),
                        "slice_dim": slice_dim,
                        "slice_idx": entry["slice_idx"],
                        "label": target_label,
                    }
                )
    return out


def _resolve_ad_keys(task, metric):
    """Landmark keys forming a measurement, resolved against the on-disk task maps.

    distance -> [p1, p2];  angle -> [l1p1, l1p2, l2p1, l2p2].
    """
    metric_map = task[metric["metric_map_name"]]
    entry = metric_map[metric["metric_key"]]
    if metric["metric_type"] == "distance":
        return list(entry["element_keys"])
    lines_map = task[entry["element_map_name"]]
    line1, line2 = entry["element_keys"]
    return list(lines_map[line1]["element_keys"]) + list(lines_map[line2]["element_keys"])


def _enum_ad(dataset, dataset_dir, task, split, max_aspect):
    """One candidate per (case, slice, measurement) above the near-zero GT threshold."""
    out = []
    task_id = task["task_ID"]
    for case in split_cases(task, split):
        landmark_file = case.get("landmark_file")
        if not landmark_file:
            continue
        for axis, slice_dim in _AXES:
            if not _aspect_ok(case, slice_dim, max_aspect):
                continue
            for entry in case.get(f"slice_profiles_{axis}") or []:
                for metric in entry.get("slice_profile") or []:
                    metric_type = metric.get("metric_type")
                    if metric_type not in ("distance", "angle"):
                        continue
                    # Mirror the benchmark's near-zero GT exclusion (unbounded MRE there,
                    # degenerate overlay here).
                    if abs(metric.get("metric_value", 0.0)) < AD_NEAR_ZERO_GT_THRESHOLD:
                        continue
                    try:
                        lm_keys = _resolve_ad_keys(task, metric)
                    except (KeyError, ValueError):
                        continue
                    out.append(
                        {
                            "group": G_ANG if metric_type == "angle" else G_DIST,
                            "task_type": "Biometrics-From-Landmarks-"
                            + ("Angle" if metric_type == "angle" else "Distance"),
                            "metric_type": metric_type,
                            "metric_key": metric["metric_key"],
                            "dataset": dataset,
                            "task_id": task_id,
                            "case_id": case["case_ID"],
                            "image_file": case["image_file"],
                            "landmark_file": landmark_file,
                            "slice_dim": slice_dim,
                            "slice_idx": entry["slice_idx"],
                            "lm_keys": lm_keys,
                        }
                    )
    return out


def collect_candidates(
    data_dir, version, split, datasets, pool_per_dataset, rng, max_plan_mb=0,
    max_aspect=0.0,
):
    """Enumerate GT samples per dataset, keeping at most pool_per_dataset per (group, dataset).

    Each dataset's candidates are subsampled before moving on and the plan cache is cleared, so
    only one plan is resident at a time. That matters because detection plans store every bbox on
    every slice and get very large -- TotalSegmentator's is 676 MB gzipped and costs ~7 min and
    ~81 GB to parse. ``max_plan_mb`` > 0 skips (and reports) detection plans above that gzipped
    size; 0 means no cap, i.e. full coverage.
    """
    pools = defaultdict(list)
    skipped = []
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        per_group = defaultdict(list)

        det_path = resolve_plan_path(dataset_dir, "detection", version)
        det_mb = os.path.getsize(det_path) / 1e6 if det_path else 0
        if det_path and max_plan_mb and det_mb > max_plan_mb:
            skipped.append((dataset, det_mb))
            det_plan = None
        else:
            det_plan = load_benchmark_plan(dataset_dir, "detection", version)
        if det_plan:
            for cand in _enum_detection(dataset, dataset_dir, det_plan, split, max_aspect):
                per_group[cand["group"]].append(cand)

        bio_plan = load_benchmark_plan(dataset_dir, "biometry", version)
        if bio_plan:
            for task in bio_plan["tasks"]:
                if task.get("target_label") is not None:
                    found = _enum_tl(dataset, dataset_dir, task, split, max_aspect)
                else:
                    found = _enum_ad(dataset, dataset_dir, task, split, max_aspect)
                for cand in found:
                    per_group[cand["group"]].append(cand)

        for group, cands in per_group.items():
            rng.shuffle(cands)
            pools[group].extend(cands[:pool_per_dataset])
        print(
            f"  {dataset:22s} "
            + ", ".join(f"{g}={min(len(c), pool_per_dataset)}" for g, c in sorted(per_group.items()))
        )
        # Drop the plan before the next dataset: two giant detection plans resident at once
        # would peak well above 100 GB.
        load_benchmark_plan.cache_clear()

    if skipped:
        print(
            f"\n[skipped] {len(skipped)} detection plan(s) above --max_plan_mb "
            f"(re-run with --max_plan_mb 0 to include them):",
            file=sys.stderr,
        )
        for dataset, mb in skipped:
            print(f"    {dataset} ({mb:.0f} MB gz)", file=sys.stderr)
    return pools


# ── Rendering ────────────────────────────────────────────────────────────────


def _stem(cand):
    parts = [
        _safe(cand["dataset"]),
        f"Task{_safe(cand['task_id'])}",
        _safe(cand["case_id"]),
        f"dim{cand['slice_dim']}",
        f"idx{cand['slice_idx']}",
    ]
    if cand.get("metric_key"):
        parts.append(_safe(cand["metric_key"]))
    else:
        parts.append(f"label{cand.get('label')}")
    return "__".join(parts)


def render(cand, data_dir, out_root, show_mask, formats):
    """Draw one GT-only subfigure. Raises on failure; the caller counts and reports."""
    dataset_dir = os.path.join(data_dir, cand["dataset"])
    slice_dim, slice_idx = cand["slice_dim"], cand["slice_idx"]

    pixel_sizes, img_2d = _load_nifti_2d(
        os.path.join(dataset_dir, cand["image_file"]), slice_dim, slice_idx
    )
    # normalize_img picks the CT HU window from the label; A/D tasks have no label, and the
    # helper keys off the absence of "label" in the doc, so it must not be present at all.
    doc = {
        "dataset_name": cand["dataset"],
        "taskID": cand["task_id"],
        "taskType": cand["task_type"],
    }
    if cand.get("label") is not None:
        doc["label"] = cand["label"]
    img_2d = normalize_img(doc, img_2d)

    fig_path = os.path.join(out_root, cand["group"], _stem(cand) + ".png")

    if cand["group"] == G_DET:
        plot_detection_on_image(
            image_2d=img_2d,
            pixel_sizes=pixel_sizes,
            gt_box=cand["gt_box"],
            pred_box=None,
            slice_dim=slice_dim,
            slice_idx=slice_idx,
            fig_path=fig_path,
            formats=formats,
        )
        return

    lms = _slice_landmarks(dataset_dir, cand["landmark_file"], slice_dim, slice_idx)
    if lms is None:
        raise ValueError("landmarks not found for slice")

    if cand["group"] == G_TL:
        mask_2d = None
        if show_mask and cand.get("mask_file"):
            _, mask_raw = _load_nifti_2d(
                os.path.join(dataset_dir, cand["mask_file"]), slice_dim, slice_idx
            )
            mask_2d = (mask_raw == cand["label"]).astype(np.float32)
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
        return

    keys = cand["lm_keys"]
    pts = [_project_3d_to_2d(lms[k], slice_dim) for k in keys]
    if cand["metric_type"] == "distance":
        gt_pts = {"p1": pts[0], "p2": pts[1]}
    else:
        gt_pts = {"l1p1": pts[0], "l1p2": pts[1], "l2p1": pts[2], "l2p2": pts[3]}
    plot_ad_on_image(
        image_2d=img_2d,
        pixel_sizes=pixel_sizes,
        metric_type=cand["metric_type"],
        gt_pts=gt_pts,
        pred_pts=None,
        slice_dim=slice_dim,
        slice_idx=slice_idx,
        fig_path=fig_path,
        formats=formats,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render GT-only MedVision annotation subfigures from the data folder."
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Root of Data/Datasets")
    parser.add_argument(
        "--version",
        default="1.1.1",
        help="Annotation version pin, resolved by the ceiling rule (default: 1.1.1)",
    )
    parser.add_argument(
        "--fig_dir",
        default=os.path.join(_REPO_DIR, "Figures", "GT-annotations"),
        help="Output base directory; subfigures land in <fig_dir>/GT/<task>/",
    )
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated dataset filter (default: all)"
    )
    parser.add_argument(
        "--pool_per_dataset",
        type=int,
        default=8,
        help="Max subfigures rendered per (task, dataset), so no dataset floods a row",
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
        help="Skip detection plans larger than this gzipped size in MB, reporting which were "
        "dropped. Detection plans store every bbox on every slice, so the biggest "
        "(TotalSegmentator 676 MB, AbdomenAtlas1.0Mini 574 MB) cost minutes and tens of GB "
        "each to parse. 0 (default) means no cap.",
    )
    parser.add_argument(
        "--max_aspect",
        type=float,
        default=4.0,
        help="Skip slices whose rendered panel would be more anisotropic than this ratio. The "
        "compositor stretches every subfigure into a square cell, so a 25:1 panel (ACDC/CAMUS "
        "sagittal, only 9-14 slices) becomes an uninformative stripe. Judged on PHYSICAL extent, "
        "so thin-but-thick-sliced volumes like CrossMoDA are kept. 0 disables the guard.",
    )
    parser.add_argument("--save_as_png", action="store_true", help="Save subfigures as PNG.")
    parser.add_argument("--save_as_pdf", action="store_true", help="Save subfigures as PDF.")
    parser.add_argument("--seed", type=int, default=None, help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    formats = [
        f for f, on in (("png", args.save_as_png), ("pdf", args.save_as_pdf)) if on
    ] or ["pdf"]
    rng = random.Random(args.seed if args.seed is not None else SEED)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = sorted(
            d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))
        )

    out_root = os.path.join(args.fig_dir, "GT")
    print(f"Collecting GT candidates at version ceiling v{args.version} ({args.split} split)")
    pools = collect_candidates(
        args.data_dir,
        args.version,
        args.split,
        datasets,
        args.pool_per_dataset,
        rng,
        args.max_plan_mb,
        args.max_aspect,
    )

    if not pools:
        print("No candidates found -- nothing to render.", file=sys.stderr)
        return

    print("\nPool sizes: " + ", ".join(f"{g}={len(pools.get(g, []))}" for g in GROUPS))

    n_ok, failures = 0, []
    for group in GROUPS:
        cands = pools.get(group) or []
        for cand in tqdm(cands, desc=f"  {group}", leave=False):
            try:
                render(cand, args.data_dir, out_root, args.show_mask, formats)
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - report and continue
                failures.append((group, cand["dataset"], cand["case_id"], repr(e)))

    print(f"\nRendered {n_ok} subfigures into {out_root}")
    for group in GROUPS:
        d = os.path.join(out_root, group)
        n = len([f for f in os.listdir(d)]) if os.path.isdir(d) else 0
        print(f"  {group:20s} {n}")
    if failures:
        print(f"\n{len(failures)} failures:", file=sys.stderr)
        for f in failures[:20]:
            print("  " + " | ".join(map(str, f)), file=sys.stderr)


if __name__ == "__main__":
    main()
