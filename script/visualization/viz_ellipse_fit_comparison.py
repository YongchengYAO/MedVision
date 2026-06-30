"""Figure(s): ellipse fitting in IMAGE space vs REAL (physical) space.

Educational companion to ``doc/ellipse-fitting-image-vs-real-space.md``. Under
*anisotropic* pixel spacing, an ellipse fit to a lesion mask in pixel space is a
different ellipse than one fit in real-world space (anisotropic scaling is a
congruence ``S -> D^T S D``, not a similarity, so the principal axes rotate and
the major/minor lengths change). This script makes that concrete on REAL
benchmark volumes by overlaying both fits' axes on one slice, rendered two ways:
with and without the physical aspect-ratio correction in ``imshow``.

Why we re-slice: every shipped TL sample is sliced axially, where the in-plane
spacing is isotropic for all datasets, so the two fits coincide there. The
anisotropy lives in the through-plane axis. The Coronal/Sagittal TL-CoT configs
slice the SAME real volumes along slice_dim 1/0, pulling that large spacing into
the 2D plane.

Coverage: the candidate pool is the UNION of anisotropic (dataset, taskID) tasks
across the ds_v1.0.0 Coronal + Sagittal pixel-size summaries in
``tasks_list/pixel_sizes__ds_v1.0.0/`` (the authoritative anisotropy oracle).
Per-case test-split records (mask_file, label, slice_dim, slice_idx, pixel_size)
come from the HF MedVision per-task configs, loaded offline from local data.
This reaches all 14 anisotropic tasks — including subsets the axial test parquet
never indexed (BraTS24-MET, HNTSMRG24-preRT, MSD-Colon/Lung/Pancreas).

Examples:
    python script/visualization/viz_ellipse_fit_comparison.py --orientation sagittal --n 8
    python script/visualization/viz_ellipse_fit_comparison.py --orientation coronal --n 6 --seed 7
"""

import argparse
import gzip
import json
import os
import random
import re
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = "/mnt/vincent-pvc-rwm/Github/MedVision"
sys.path.insert(0, os.path.join(REPO, "src"))

# Local-only HF loader resolution: data already lives under Data/; setting these
# (if unset) makes the cached MedVision loader return local paths, no download.
os.environ.setdefault("MedVision_DATA_DIR", os.path.join(REPO, "Data"))
os.environ.setdefault("MedVision_PLANNER_VERSION", "1.0.0")  # matches ds_v1.0.0 summaries
os.environ.setdefault("MedVision_ACK_RELEASE", "1.1.1")  # ack older version vs latest release

from datasets import load_dataset  # noqa: E402

from medvision_bm.utils.configs import SEED  # noqa: E402
from medvision_bm.utils.plot_utils import _get_appropriate_scale, save_fig_capped  # noqa: E402

PIXSIZE_DIR = os.path.join(REPO, "tasks_list/pixel_sizes__ds_v1.0.0")
DEFAULT_OUT = os.path.join(REPO, "Figures/ellipse_fit")
HF_REPO = "YongchengYAO/MedVision"

# Orientation -> slice axis (RAS+ volumes; axis 2 carries the large spacing).
ORIENT_SLICE_DIM = {"sagittal": 0, "coronal": 1}
RECORD_COLS = ["image_file", "mask_file", "label", "slice_dim", "slice_idx",
               "pixel_size", "voxel_size", "dataset_name", "taskID"]

IMG_COLOR = "#EA4335"   # image-space fit (red)
REAL_COLOR = "#4285F4"  # real-space fit (blue)
MASK_COLOR = "#2ECC71"  # green


# --------------------------------------------------------------------------- #
# Ellipse fitting (mirrors medvision_ds __fit_ellipses, with correct A1 pairing)
# --------------------------------------------------------------------------- #
def _phys_len(a, b, ps):
    """Physical length (mm) of the segment a-b, points in array space (dim0, dim1)."""
    return float(np.hypot((a[0] - b[0]) * ps[0], (a[1] - b[1]) * ps[1]))


def fit_ellipse(contour_xy, pixel_sizes, space):
    """Fit an ellipse to a contour and return its major/minor axis endpoints.

    contour_xy : (N, 2) cv2 contour points, ordered (x=col=dim1, y=row=dim0).
    pixel_sizes: [ps_dim0, ps_dim1] mm.
    space      : "image" -> fit on raw pixel coords (wrong under anisotropy);
                 "real"  -> scale to mm with correct axis pairing, fit, then map
                            each endpoint back per-coordinate (correct).

    Returns dict with major/minor endpoint pairs in array space (dim0, dim1) and
    their physical lengths (mm). Major/minor are labelled by each method's NATIVE
    metric: image-space by pixel length (reproducing the A2 swap), real-space by
    physical length.
    """
    pts = contour_xy.astype(np.float32).copy()
    if space == "real":
        # cv2 x=dim1 -> * ps_dim1 ; y=dim0 -> * ps_dim0  (correct pairing)
        pts = pts * np.array([pixel_sizes[1], pixel_sizes[0]], np.float32)

    (cx, cy), (ax0, ax1), angle = cv2.fitEllipse(pts)
    ar = np.deg2rad(angle)
    a, b = ax0 / 2.0, ax1 / 2.0
    mvx, mvy = a * np.cos(ar), a * np.sin(ar)        # half-vector, axis-0
    nvx, nvy = -b * np.sin(ar), b * np.cos(ar)       # half-vector, axis-1 (perp)

    def back(px, py):
        """Fit-space (x, y) -> array space (dim0, dim1)."""
        if space == "real":
            px, py = px / pixel_sizes[1], py / pixel_sizes[0]  # per-coordinate D^-1
        return (py, px)

    P1, P2 = back(cx + mvx, cy + mvy), back(cx - mvx, cy - mvy)  # axis-0 ends
    P3, P4 = back(cx + nvx, cy + nvy), back(cx - nvx, cy - nvy)  # axis-1 ends

    axisA, axisB = (P1, P2), (P3, P4)
    if space == "image":
        lenA = np.hypot(P1[0] - P2[0], P1[1] - P2[1])  # pixel length
        lenB = np.hypot(P3[0] - P4[0], P3[1] - P4[1])
    else:
        lenA = _phys_len(P1, P2, pixel_sizes)          # physical length
        lenB = _phys_len(P3, P4, pixel_sizes)
    major, minor = (axisA, axisB) if lenA >= lenB else (axisB, axisA)

    return {
        "major": major,
        "minor": minor,
        "maj_phys": _phys_len(major[0], major[1], pixel_sizes),
        "min_phys": _phys_len(minor[0], minor[1], pixel_sizes),
    }


def _major_angle_deg(fit):
    """Orientation of the major axis in array (display) space, degrees."""
    (d0a, d1a), (d0b, d1b) = fit["major"]
    return float(np.degrees(np.arctan2(d1b - d1a, d0b - d0a)) % 180.0)


# --------------------------------------------------------------------------- #
# Candidate pool: union of anisotropic tasks (both summaries) -> HF test records
# --------------------------------------------------------------------------- #
def anisotropic_task_union():
    """Union of (dataset, taskID) with anisotropic samples across the ds_v1.0.0
    Coronal + Sagittal summaries — the "combine the two summaries" step."""
    tasks = set()
    for plane in ("Coronal", "Sagittal"):
        path = os.path.join(PIXSIZE_DIR, f"pixelsizes_MedVision-TL-CoT__{plane}__Test__summary.json")
        for name, counts in json.load(open(path)).items():
            if name == "__all_tasks__" or counts.get("anisotropic", 0) <= 0:
                continue
            dataset = name.split("_TumorLesionSize_")[0]
            task_id = name.split("_Task")[1].split("_")[0]
            tasks.add((dataset, task_id))
    return sorted(tasks)


def _is_aniso(ps, min_ratio):
    h, w = round(float(ps[0]), 3), round(float(ps[1]), 3)
    return max(h, w) / min(h, w) >= min_ratio


def task_records(dataset, task_id, orientation, min_aniso):
    """Anisotropic test-split records for one HF per-task config (or [] on failure)."""
    cfg = f"{dataset}_TumorLesionSize_Task{task_id}_{orientation.capitalize()}_Test"
    try:
        ds = load_dataset(HF_REPO, name=cfg, split="test", trust_remote_code=True, streaming=True)
        keep = [c for c in RECORD_COLS if c in ds.features]
        ds = ds.select_columns(keep)
        return [dict(r) for r in ds if _is_aniso(r["pixel_size"], min_aniso)]
    except Exception as e:  # config missing / schema cast / load error
        print(f"  WARNING: skip {cfg}: {type(e).__name__}: {str(e)[:120]}")
        return []


def build_records(orientation, min_aniso):
    """All anisotropic test records across the union tasks, tagged for sampling."""
    records = []
    for dataset, task_id in anisotropic_task_union():
        recs = task_records(dataset, task_id, orientation, min_aniso)
        records.extend(recs)
        if recs:
            print(f"  {dataset} Task{task_id}: {len(recs)} anisotropic test slices")
    return records


# benchmark-plan structure-name lookup (cached), for nicer figure titles
_PLAN_CACHE = {}


def structure_name(dataset, task_id):
    """Human-readable target structure for a task. Keyed by taskID (not label):
    the same label integer means different structures across a dataset's subsets
    (e.g. MSD label 2 = brain tumor / liver tumour / pancreas cancer)."""
    if dataset not in _PLAN_CACHE:
        p = os.path.join(REPO, f"Data/Datasets/{dataset}/benchmark_plan_biometry_v1.0.0.json.gz")
        lut = {}
        try:
            plan = json.load(gzip.open(p, "rt"))
            for i, t in enumerate(plan["tasks"], 1):  # Task01 = tasks[0]
                m = re.search(r"Label(\d+)", t.get("landmark_folder", ""))
                lut[f"{i:02d}"] = t.get("labels_map", {}).get(m.group(1), "") if m else ""
        except Exception:
            pass
        _PLAN_CACHE[dataset] = lut
    return _PLAN_CACHE[dataset].get(str(task_id).zfill(2), "")


def _inplane_ps(vox, slice_dim):
    return [vox[1], vox[2]] if slice_dim == 0 else [vox[0], vox[2]]


def prepare_record(rec, min_div=25.0, min_pts=120):
    """Load the record's QC slice, binarize by its label, fit both ellipses.

    Returns the render inputs (image, ROI, both fits, divergence) or None if the
    lesion is too small or the two majors are < ``min_div`` degrees apart.
    """
    slice_dim = int(rec["slice_dim"])
    slice_idx = int(rec["slice_idx"])
    mnib = nib.load(rec["mask_file"])
    vox = np.array(mnib.header.get_zooms(), float)
    ps = _inplane_ps(vox, slice_dim)

    sl = [slice(None)] * 3
    sl[slice_dim] = slice_idx
    mask2d = (mnib.get_fdata()[tuple(sl)] == int(rec["label"])).astype(np.uint8)
    if mask2d.sum() == 0:
        return None

    lbl, n = ndimage.label(mask2d)
    if n == 0:
        return None
    roi = (lbl == 1 + int(np.argmax(ndimage.sum(mask2d, lbl, range(1, n + 1))))).astype(np.uint8)
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).squeeze()
    if c.ndim != 2 or c.shape[0] < min_pts:
        return None

    img_fit = fit_ellipse(c, ps, "image")
    real_fit = fit_ellipse(c, ps, "real")
    d_ang = abs(_major_angle_deg(img_fit) - _major_angle_deg(real_fit))
    d_ang = min(d_ang, 180 - d_ang)
    if d_ang < min_div:
        return None

    img2d = nib.load(rec["image_file"]).get_fdata()[tuple(sl)].astype(np.float32)
    lo, hi = np.percentile(img2d, [1, 99])
    img2d = np.clip((img2d - lo) / (hi - lo + 1e-8), 0, 1)
    return {
        "image_2d": img2d, "roi": roi, "pixel_sizes": ps, "slice_idx": slice_idx,
        "img_fit": img_fit, "real_fit": real_fit, "d_ang": d_ang,
        "dataset": rec["dataset_name"], "task_id": rec["taskID"], "label": int(rec["label"]),
        "structure": structure_name(rec["dataset_name"], rec["taskID"]),
        "case": os.path.basename(rec["mask_file"]).split(".")[0],
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _roi_view(roi, pad_frac=0.35):
    """Plot-space view limits (d0_lo, d0_hi, d1_lo, d1_hi) around the ROI bbox."""
    ys, xs = np.where(roi > 0)  # ys = dim0, xs = dim1
    span0, span1 = ys.max() - ys.min(), xs.max() - xs.min()
    pad0, pad1 = max(span0 * pad_frac, 10), max(span1 * pad_frac, 10)
    return (ys.min() - pad0, ys.max() + pad0, xs.min() - pad1, xs.max() + pad1)


def _scale_bar(ax, view, pixel_sizes):
    """L-shaped white scale bar in the lower-left of the cropped view."""
    span = [view[1] - view[0], view[3] - view[2]]
    min_idx = int(np.argmin(span))
    scale_mm, n_min = _get_appropriate_scale(pixel_sizes[min_idx], span[min_idx], 10)
    n_max = int(scale_mm / pixel_sizes[1 - min_idx])
    s0, s1 = (n_min, n_max) if min_idx == 0 else (n_max, n_min)
    x0 = view[0] + 0.06 * span[0]
    y0 = view[2] + 0.06 * span[1]
    ax.plot([x0, x0 + s0], [y0, y0], "w-", lw=3)
    ax.plot([x0, x0], [y0, y0 + s1], "w-", lw=3)
    ax.text(x0 + s0 + 0.01 * span[0], y0, f"{scale_mm} mm", color="white", fontsize=12, va="bottom")


def _draw_fit(ax, fit, color):
    maj, mino = fit["major"], fit["minor"]
    ax.plot([maj[0][0], maj[1][0]], [maj[0][1], maj[1][1]], color=color, ls="-", lw=2.5, zorder=3)
    ax.plot([mino[0][0], mino[1][0]], [mino[0][1], mino[1][1]], color=color, ls="--", lw=2.0, zorder=3)
    for p in (maj[0], maj[1], mino[0], mino[1]):
        ax.scatter(p[0], p[1], color=color, edgecolors="black", s=35, lw=1.0, zorder=4)


def draw_panel(ax, image_2d, mask_2d, pixel_sizes, img_fit, real_fit, view, aspect_mode, title):
    aspect = (pixel_sizes[1] / pixel_sizes[0]) if aspect_mode == "physical" else 1.0
    ax.imshow(image_2d.T, cmap="gray", origin="lower", aspect=aspect, zorder=-1)
    ax.contour(mask_2d.T, levels=[0.5], colors=MASK_COLOR, linewidths=2, zorder=0)
    _draw_fit(ax, img_fit, IMG_COLOR)
    _draw_fit(ax, real_fit, REAL_COLOR)
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    _scale_bar(ax, view, pixel_sizes)
    ax.text(
        0.02, 0.98,
        f"image-fit major: {img_fit['maj_phys']:.1f} mm\n"
        f"real-fit  major: {real_fit['maj_phys']:.1f} mm",
        transform=ax.transAxes, va="top", ha="left", fontsize=11, color="white",
        bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"),
    )
    ax.set_xlabel("Anterior →", fontsize=14)
    ax.set_ylabel("Superior →", fontsize=14)
    ax.set_title(title, fontsize=15)


def render_case(prep, orientation, out_dir):
    """Write one 1x2 figure (no-scaling vs physical-aspect) for a prepared case."""
    image_2d, roi, ps = prep["image_2d"], prep["roi"], prep["pixel_sizes"]
    img_fit, real_fit = prep["img_fit"], prep["real_fit"]
    view = _roi_view(roi)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    draw_panel(axes[0], image_2d, roi, ps, img_fit, real_fit, view,
               "none", "Ellipse Fitting in Image Space")
    draw_panel(axes[1], image_2d, roi, ps, img_fit, real_fit, view,
               "physical", "Ellipse Fitting in Real Space")

    handles = [
        mlines.Line2D([], [], color=IMG_COLOR, ls="-", lw=2.5, label="Image-space fit — major"),
        mlines.Line2D([], [], color=IMG_COLOR, ls="--", lw=2.0, label="Image-space fit — minor"),
        mlines.Line2D([], [], color=REAL_COLOR, ls="-", lw=2.5, label="Real-space fit — major"),
        mlines.Line2D([], [], color=REAL_COLOR, ls="--", lw=2.0, label="Real-space fit — minor"),
        mlines.Line2D([], [], color=MASK_COLOR, lw=2, label="Lesion mask"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    struct = f" — {prep['structure']}" if prep["structure"] else ""
    fig.suptitle(
        f"Ellipse fitting: image space vs. real space — {prep['dataset']} Task{prep['task_id']}"
        f"{struct}\n{prep['case']} ({orientation}, pixel size {ps[0]:.2f}×{ps[1]:.2f} mm)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(
        out_dir,
        f"{prep['dataset']}_Task{prep['task_id']}_{prep['case']}_{orientation}_slice{prep['slice_idx']}.png",
    )
    save_fig_capped(out, fig=fig, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orientation", choices=list(ORIENT_SLICE_DIM), default="sagittal")
    ap.add_argument("--n", type=int, default=8, help="number of figures to generate")
    ap.add_argument("--seed", type=int, default=SEED, help=f"random seed (default {SEED} from configs)")
    ap.add_argument("--out-dir", default=None, help="output dir (default: .../Figures/ellipse_fit/<orientation>)")
    ap.add_argument("--min-divergence", type=float, default=25.0,
                    help="require image/real major axes ≥ this many degrees apart (deg)")
    ap.add_argument("--min-anisotropy", type=float, default=1.5,
                    help="require in-plane spacing ratio ≥ this (per-slice)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(DEFAULT_OUT, args.orientation)
    print(f"orientation : {args.orientation} (slice_dim={ORIENT_SLICE_DIM[args.orientation]})")
    print("building candidate pool from HF test configs (union of both summaries)...")
    records = build_records(args.orientation, args.min_anisotropy)
    print(f"total anisotropic test slices: {len(records)} "
          f"across {len({r['dataset_name'] for r in records})} datasets")
    if not records:
        print("No anisotropic records available for this plane.")
        return

    # Group by task and draw round-robin so one run spans many subsets (a few
    # large tasks like KiTS23 would otherwise dominate a flat random draw).
    rng = random.Random(args.seed)
    rng.shuffle(records)
    groups = {}
    for r in records:
        groups.setdefault((r["dataset_name"], r["taskID"]), []).append(r)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    ptr = {k: 0 for k in group_keys}

    made, seen, gi = 0, set(), 0
    while made < args.n and any(ptr[k] < len(groups[k]) for k in group_keys):
        k = group_keys[gi % len(group_keys)]
        gi += 1
        if ptr[k] >= len(groups[k]):
            continue
        rec = groups[k][ptr[k]]
        ptr[k] += 1
        if rec["mask_file"] in seen:  # one figure per volume -> diverse gallery
            continue
        prep = prepare_record(rec, min_div=args.min_divergence)
        if prep is None:
            continue
        seen.add(rec["mask_file"])
        out = render_case(prep, args.orientation, out_dir)
        made += 1
        print(f"  [{made}/{args.n}] {prep['dataset']} Task{prep['task_id']} {prep['case']} "
              f"slice{prep['slice_idx']}  "
              f"pixel_size=[{prep['pixel_sizes'][0]:.2f},{prep['pixel_sizes'][1]:.2f}]mm  "
              f"major_axis_divergence={prep['d_ang']:.0f}°  "
              f"image_fit_major={prep['img_fit']['maj_phys']:.1f}mm  "
              f"real_fit_major={prep['real_fit']['maj_phys']:.1f}mm")

    print(f"saved {made} figure(s) to {out_dir}")
    if made < args.n:
        print(f"WARNING: only {made}/{args.n} cases qualified (try lowering --min-divergence).")


if __name__ == "__main__":
    main()
