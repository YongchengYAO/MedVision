"""Summarize the local ``Data/Datasets`` collection from the benchmark plans.

Reads each dataset's ``benchmark_plan_*.json.gz`` (no HF) and produces:

* ``dataset_files.jsonl``   — one row per on-disk ``.nii.gz`` file (image/mask), with geometry,
  split, labels present, anatomy groups, and per-plane 2D-sample counts.
* ``dataset_summary_filtered.json`` — nested rollups (per dataset + a collection-level
  ``__all__``); the default filtered view (``dataset_summary_raw.json`` is the raw-count twin).
* ``dataset_summary.csv``   — WIDE, one comprehensive row per dataset (+ a TOTAL row).
* ``dataset_label_stats.csv`` — LONG companion, one row per (dataset, modality, category).

Canonical source of the case inventory is the **segmentation** plan (labels + ROI areas).
**Measurement targets** (T/L ellipse major/minor axes; A/D angle/distance) come from the
**biometry** plan. The large **detection** plans are read only for the BoxSize benchmark counts
(``count_benchmark_annotations``; skipped with ``--no_detection``). ``_regen`` folders are
skipped entirely.

Per-task ``labels_map`` is sourced **live from the ``medvision_ds`` preprocess code** (via
``get_labelsMap_imgModality_from_seg_benchmark_plan``) because the label names baked into the
``.json.gz`` are a stale snapshot; it falls back to the ``.json.gz`` map if ``medvision_ds`` is
unavailable. This can transitively import ``nibabel`` (used only when the live lookup succeeds).

All paths in the outputs are relative to the ``Datasets`` root.
"""

import argparse
import csv
import gzip
import json
import math
import os
import statistics
from collections import Counter, defaultdict

from medvision_bm.utils import configs
from medvision_bm.utils.parse_utils import get_labelsMap_imgModality_from_seg_benchmark_plan
from medvision_bm.utils.plan_utils import (
    AXIS_TO_PLANE,
    anatomy_group,
    dataset_exists_at,
    load_benchmark_plan,
    resolve_plan_path,
    slice_entries,
)

NII_EXT = ".nii.gz"
_SPLITS = ("train", "test")


def _mean_median(values):
    if not values:
        return None, None
    return round(statistics.fmean(values), 4), round(statistics.median(values), 4)


def _array_range(counts):
    """Compact per-axis 'minX-maxX x minY-maxY x minZ-maxZ' from an 'XxYxZ'->n counter dict."""
    if not counts:
        return ""
    dims = [[], [], []]
    for key in counts:
        for i, v in enumerate(key.split("x")[:3]):
            dims[i].append(int(v))
    return " x ".join(f"{min(d)}-{max(d)}" for d in dims if d)


def _voxel_range(counts):
    """Compact 'min-max' voxel spacing (mm) across all axes from a 'vXxvYxvZ'->n counter dict."""
    if not counts:
        return ""
    vals = [float(v) for key in counts for v in key.split("x")]
    return f"{round(min(vals), 3)}-{round(max(vals), 3)}"


def scan_ondisk(dataset_dir):
    """Count on-disk .nii.gz files by role: (n_image_nii, n_mask_nii, n_other_nii)."""
    n_img = n_mask = n_other = 0
    for dirpath, _, files in os.walk(dataset_dir):
        rel_segs = os.path.relpath(dirpath, dataset_dir).split(os.sep)
        for fn in files:
            if not fn.endswith(NII_EXT):
                continue
            if any(s.startswith("Images") for s in rel_segs):
                n_img += 1
            elif "Masks" in rel_segs:
                n_mask += 1
            else:
                n_other += 1
    return n_img, n_mask, n_other


def fine_labels_of_case(case, labels_map):
    """Union of fine label names present anywhere in the case (all axes)."""
    ints = set()
    for axis in ("x", "y", "z"):
        for entry in slice_entries(case, axis):
            for item in entry.get("slice_profile", []) or []:
                if isinstance(item, dict) and "label" in item:
                    ints.add(int(item["label"]))
    names = set()
    for i in ints:
        name = (labels_map or {}).get(str(i))
        if name:
            names.add(name)
    return names


def process_measurements(dataset_dir, plan_version):
    """From the biometry plan, collect measurement values by category.

    Returns a dict of value-lists: major_axis_mm, minor_axis_mm (T/L ellipse), angle_deg,
    distance_mm (A/D), and n_volumes per category. Empty dict if no biometry plan.
    """
    plan = load_benchmark_plan(dataset_dir, "biometry", plan_version)
    if plan is None:
        return {}
    vals = defaultdict(list)
    vols = defaultdict(set)
    landmark_files = set()
    for task in plan.get("tasks", []):
        lines = task.get("lines_map", {}) or {}
        for split in _SPLITS:
            for case in task.get(f"{split}_cases", []) or []:
                cid = case.get("case_ID")
                if case.get("landmark_file"):
                    landmark_files.add(case["landmark_file"])
                for axis in ("x", "y", "z"):
                    for entry in slice_entries(case, axis):
                        for item in entry.get("slice_profile", []) or []:
                            cluster = item if isinstance(item, list) else [item]
                            for m in cluster:
                                if not isinstance(m, dict) or "metric_value" not in m:
                                    continue
                                cat = _classify_metric(m, lines)
                                if cat:
                                    vals[cat].append(m["metric_value"])
                                    vols[cat].add(cid)
    return {
        "values": dict(vals),
        "n_volumes": {k: len(v) for k, v in vols.items()},
        "n_landmark_files": len(landmark_files),
    }


def _classify_metric(metric, lines_map):
    """Map a biometry metric dict to a category key, or None."""
    mtype, mkey = metric.get("metric_type"), metric.get("metric_key")
    if mtype == "angle":
        return "angle_deg"
    if mtype == "distance":
        name = (lines_map.get(mkey, {}) or {}).get("name", "").lower()
        if "ellipse" in name and ("major" in name or "marjor" in name):  # 'marjor' typo in data
            return "major_axis_mm"
        if "ellipse" in name and "minor" in name:
            return "minor_axis_mm"
        return "distance_mm"
    return None


def labels_map_from_medvision_ds(dataset, task_id, fallback_labels_map):
    """Current ``labels_map`` from the ``medvision_ds`` preprocess code, or a fallback.

    The ``.json.gz`` plan's ``labels_map`` is a stale snapshot whose names can lag corrections
    made in ``medvision_ds.datasets.*.preprocess_segmentation``; sourcing it live keeps anatomy
    grouping aligned with current terminology. ``task_id`` is stringly-typed in the plan, so it
    is cast to ``int`` (the medvision_ds lookup indexes ``tasks[task_id - 1]``). On any failure
    (e.g. ``medvision_ds``/``nibabel`` unavailable, unknown dataset) it returns
    ``fallback_labels_map`` with a warning, so one dataset can't abort the run. Biometry tasks
    (``labels_map is None``) are passed through untouched.
    """
    if not fallback_labels_map:
        return fallback_labels_map
    try:
        res = get_labelsMap_imgModality_from_seg_benchmark_plan(dataset, int(task_id))
        if res and res[0]:
            return res[0]
    except Exception as e:
        print(f"  [warn] {dataset} task {task_id}: medvision_ds labels_map unavailable "
              f"({type(e).__name__}: {e}); using .json.gz labels")
    return fallback_labels_map


def _load_plan_uncached(dataset_dir, family, version=None):
    """Same resolution as ``load_benchmark_plan`` (see ``resolve_plan_path``), but uncached.

    Bypasses that function's ``functools.lru_cache`` on purpose: the detection plan can be
    hundreds of MB (multi-GB decompressed for TotalSegmentator/AbdomenAtlas), so it must not be
    retained in the cache.
    """
    path = resolve_plan_path(dataset_dir, family, version)
    if path is None:
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def _count_boxsize(dataset_dir, dataset, version=None):
    """BoxSize sample counts + per-anatomy breakdowns from the detection plan.

    ``version`` caps which detection plan is read (newest at or before it); see
    ``resolve_plan_path``. Datasets first published after ``version`` yield zeros.

    Computes BOTH the filtered and the raw (unfiltered) count in one pass over train+test and
    all three planes; anatomy is mapped via the medvision_ds-sourced ``labels_map``:

    * filtered — per ``(case, slice, label)`` item, keep iff exactly one connected component
      (``len(bboxes) == 1``) with a box ``>= 10`` px in both dims (``MedVision.py`` 9192/9195-8).
    * raw — one per ``(case, slice, label)`` item that has any box, no cluster-count or size
      filter (a label split into N connected components still counts once, not N).

    Returns ``(n_filtered, filtered_by_anatomy, n_raw, raw_by_anatomy)`` (Counters).
    """
    plan = _load_plan_uncached(dataset_dir, "detection", version)
    if plan is None:
        return 0, Counter(), 0, Counter()
    n_f = n_r = 0
    anat_f, anat_r = Counter(), Counter()
    for task in plan.get("tasks", []):
        labels_map = labels_map_from_medvision_ds(
            dataset, task.get("task_ID"), task.get("labels_map")
        )
        for split in _SPLITS:
            for case in task.get(f"{split}_cases", []) or []:
                for axis in ("x", "y", "z"):
                    for entry in slice_entries(case, axis):
                        for item in entry.get("slice_profile", []) or []:
                            bboxes = item.get("bboxes") or []
                            if not bboxes:
                                continue
                            label = item.get("label")
                            name = ((labels_map or {}).get(str(int(label)))
                                    if label is not None else None)
                            grp = anatomy_group(name) if name else None
                            n_r += 1                                 # raw: one per (slice, label)
                            if grp:
                                anat_r[grp] += 1
                            dims = bboxes[0].get("dimensions") or [0, 0]
                            if len(bboxes) == 1 and dims[0] >= 10 and dims[1] >= 10:
                                n_f += 1                             # filtered: single >=10px box
                                if grp:
                                    anat_f[grp] += 1
    return n_f, anat_f, n_r, anat_r


def _count_biometry_samples(dataset_dir, version=None):
    """T/L + A/D sample counts (filtered and raw) from the biometry plan for ``version``.

    ``version`` selects the ``benchmark_plan_biometry_v<version>`` plan (``None`` = highest
    available), falling back to the highest available version when the requested one is absent
    (so A/D-only datasets, which ship v1.0.0 only, still resolve). Only T/L differs across
    versions — the newer planners extract different ellipse sets.

    * T/L filtered — a ``(case, slice)`` holding a single cluster: ``n_total_clusters == 1`` on the
      v1.1.0+ planner, falling back to ``len(slice_profile) == 1`` on v1.0.0 (``MedVision.py``
      9337-45). T/L raw — every ellipse (``sum(len(slice_profile))``), no single-cluster filter.
    * A/D — one sample per metric (``sum(len(slice_profile))``); no filter, so raw == filtered.

    Returns ``(n_tl_filtered, n_ad, n_tl_raw)``.
    """
    plan = load_benchmark_plan(dataset_dir, "biometry", version)
    if plan is None:
        return 0, 0, 0
    n_tl_f = n_ad = n_tl_r = 0
    for task in plan.get("tasks", []):
        is_tl = task.get("target_label") is not None
        for split in _SPLITS:
            for case in task.get(f"{split}_cases", []) or []:
                for axis in ("x", "y", "z"):
                    for entry in slice_entries(case, axis):
                        sp = entry.get("slice_profile", []) or []
                        if is_tl:
                            n_tl_r += len(sp)
                            # single-instance = one cluster on the slice: n_total_clusters (v1.1.0+)
                            # or, on the v1.0.0 fallback that lacks it, a single ellipse.
                            if entry.get("n_total_clusters", len(sp)) == 1:
                                n_tl_f += 1
                        else:
                            n_ad += len(sp)
    return n_tl_f, n_ad, n_tl_r


def count_benchmark_annotations(dataset_dir, dataset, skip_detection=False, version=None):
    """Benchmark VQA-sample counts for the 3 non-MaskSize tasks, filtered AND raw.

    Returns ``(filtered, raw)`` where each is ``(annotations_by_task, boxsize_by_anatomy)``:
    ``annotations_by_task`` = ``{"BoxSize": n, "TumorLesionSize": n, "BiometricsFromLandmarks":
    n}`` (nonzero only), ``boxsize_by_anatomy`` = ``{anatomy_group: n}``. *filtered* applies the
    v1.0.0 loader filters; *raw* counts one per (slice,label) box-item plus every ellipse,
    unfiltered. BoxSize comes from the
    large detection plan and is skipped (0) when ``skip_detection`` is set.
    """
    task_f, task_r = {}, {}
    anat_f, anat_r = Counter(), Counter()
    if not skip_detection:
        n_bf, anat_f, n_br, anat_r = _count_boxsize(dataset_dir, dataset, version)
        if n_bf:
            task_f["BoxSize"] = n_bf
        if n_br:
            task_r["BoxSize"] = n_br
    n_tl_f, n_ad, n_tl_r = _count_biometry_samples(dataset_dir, version)
    if n_tl_f:
        task_f["TumorLesionSize"] = n_tl_f
    if n_tl_r:
        task_r["TumorLesionSize"] = n_tl_r
    if n_ad:
        task_f["BiometricsFromLandmarks"] = n_ad
        task_r["BiometricsFromLandmarks"] = n_ad
    return (task_f, dict(anat_f)), (task_r, dict(anat_r))


def process_dataset(datasets_root, dataset, plan_version, skip_detection=False):
    """Return (file_rows, ds_summary, label_rows) for one dataset."""
    dataset_dir = os.path.join(datasets_root, dataset)
    n_img_disk, n_mask_disk, n_other_disk = scan_ondisk(dataset_dir)

    # Every family is resolved against plan_version, so a dataset never reports a task type (or an
    # inventory) from a plan published after the requested version.
    task_types = [
        pt for pt in ("segmentation", "detection", "biometry")
        if resolve_plan_path(dataset_dir, pt, plan_version)
    ]

    # Canonical inventory: segmentation (labels + ROI) if present, else biometry (AD-only sets).
    canonical = load_benchmark_plan(dataset_dir, "segmentation", plan_version)
    if canonical is None:
        canonical = load_benchmark_plan(dataset_dir, "biometry", plan_version)

    file_rows = []
    subjects = set()  # unique case_IDs (a subject may span train/test across modality tasks)
    images_by_split = Counter()  # per-split IMAGE-FILE counts (unambiguous for multimodal)
    image_files, mask_files = set(), set()
    modalities = set()
    labels_present, anatomy_present = set(), set()
    unmapped_labels = set()

    # rollups
    img_by_modality = Counter()
    vol_by_label, vol_by_anatomy = Counter(), Counter()
    samp_by_modality, samp_by_plane = Counter(), Counter()
    samp_by_label, samp_by_anatomy = Counter(), Counter()
    n_annotations = 0
    array_size_counter = Counter()  # "XxYxZ" -> count of image volumes
    voxel_counter = Counter()       # "vXxvYxvZ" (rounded 3dp) -> count of image volumes
    # per (modality, label) ROI stats for the long CSV
    label_roi = defaultdict(lambda: {"n_volumes": 0, "n_samples": 0, "roi": [], "px": [], "anatomy": ""})

    for task in (canonical or {}).get("tasks", []):
        modality = task.get("image_modality", "unknown")
        task_id = task.get("task_ID")
        task_type = task.get("task_type")
        labels_map = labels_map_from_medvision_ds(dataset, task_id, task.get("labels_map"))
        modalities.add(modality)
        for split in _SPLITS:
            for case in task.get(f"{split}_cases", []) or []:
                cid = case.get("case_ID")
                subjects.add(cid)
                info = case.get("image_file_info", {}) or {}
                array_size = info.get("array_size")
                voxel_size = info.get("voxel_size")
                orientation = info.get("orientation")

                # Dedup the slice/annotation counters below by image: a multi-task plan could
                # list the same image_file under several tasks, which would otherwise inflate
                # them (image_files is already deduped). first_seen is evaluated BEFORE the image
                # is registered further down. Scope: only these counters and the image row are
                # deduped by image; mask rows (mask_files) and the labels_present/anatomy sets are
                # keyed independently, so a shared image with a *different* mask is not specially
                # reconciled (no such case exists in the current plans).
                img_rel = f"{dataset}/{case['image_file']}" if case.get("image_file") else None
                first_seen = img_rel is None or img_rel not in image_files

                # per-plane 2D-slice counts for this case
                plane_counts = {
                    AXIS_TO_PLANE[a]: len(slice_entries(case, a)) for a in ("x", "y", "z")
                }
                if first_seen:
                    for plane, n in plane_counts.items():
                        samp_by_plane[plane] += n
                        samp_by_modality[modality] += n

                # labels present in this case (union)
                case_labels = fine_labels_of_case(case, labels_map) if labels_map else set()
                case_anatomy = set()
                for lab in case_labels:
                    grp = anatomy_group(lab)
                    case_anatomy.add(grp)
                    if grp == "UNMAPPED":
                        unmapped_labels.add(lab)
                labels_present |= case_labels
                anatomy_present |= case_anatomy

                # image file row (one per unique image; covers per-modality images)
                if img_rel and img_rel not in image_files:
                    image_files.add(img_rel)
                    images_by_split[split] += 1
                    img_by_modality[modality] += 1
                    if array_size:
                        array_size_counter["x".join(str(int(d)) for d in array_size)] += 1
                    if voxel_size:
                        voxel_counter["x".join(f"{float(v):.3f}" for v in voxel_size)] += 1
                    for lab in case_labels:
                        vol_by_label[lab] += 1
                    for grp in case_anatomy:
                        vol_by_anatomy[grp] += 1
                    file_rows.append({
                        "dataset": dataset, "task_id": task_id, "task_type": task_type,
                        "modality": modality, "role": "image", "path": img_rel,
                        "case_ID": cid, "split": split, "array_size": array_size,
                        "voxel_size": voxel_size, "orientation": orientation,
                        "n_2D-slices": plane_counts,
                    })
                # mask file row (labels/anatomy attached here)
                mask_rel = f"{dataset}/{case['mask_file']}" if case.get("mask_file") else None
                if mask_rel and mask_rel not in mask_files:
                    mask_files.add(mask_rel)
                    file_rows.append({
                        "dataset": dataset, "task_id": task_id, "task_type": task_type,
                        "modality": modality, "role": "mask", "path": mask_rel,
                        "case_ID": cid, "split": split, "array_size": array_size,
                        "voxel_size": voxel_size, "orientation": orientation,
                        "labels": sorted(case_labels), "anatomy_groups": sorted(case_anatomy),
                        "n_2D-slices": plane_counts,
                    })

                # per-slice label counts + ROI stats + annotation count (unique images only)
                if first_seen:
                    for axis in ("x", "y", "z"):
                        for entry in slice_entries(case, axis):
                            items = entry.get("slice_profile", []) or []
                            slice_labels = set()
                            for item in items:
                                if not isinstance(item, dict) or "label" not in item:
                                    continue
                                n_annotations += 1
                                name = (labels_map or {}).get(str(int(item["label"])))
                                if not name:
                                    continue
                                slice_labels.add(name)
                                key = (modality, name)
                                st = label_roi[key]
                                st["anatomy"] = anatomy_group(name)
                                if item.get("ROI_area") is not None:
                                    st["roi"].append(float(item["ROI_area"]))
                                if item.get("pixel_count") is not None:
                                    st["px"].append(float(item["pixel_count"]))
                                st["n_samples"] += 1
                            for name in slice_labels:
                                samp_by_label[name] += 1
                                samp_by_anatomy[anatomy_group(name)] += 1

    # attach per-label volume counts to label_roi
    for (modality, name), st in label_roi.items():
        st["n_volumes"] = vol_by_label.get(name, 0)

    # measurements from biometry
    meas = process_measurements(dataset_dir, plan_version)

    n_slices_total = sum(samp_by_plane.values())
    ds_summary = {
        "modalities": sorted(modalities),
        "task_types": task_types,
        "anatomy_groups": sorted(a for a in anatomy_present if a != "UNMAPPED"),
        "labels": sorted(labels_present),
        "n_labels": len(labels_present),
        "n_subjects": len(subjects),
        "n_images_train": images_by_split.get("train", 0),
        "n_images_test": images_by_split.get("test", 0),
        "n_image_files": len(image_files),
        "n_mask_files": len(mask_files),
        "n_landmark_files": meas.get("n_landmark_files", 0),
        "n_nii_ondisk": n_img_disk + n_mask_disk + n_other_disk,
        "n_image_nii_ondisk": n_img_disk,
        "n_mask_nii_ondisk": n_mask_disk,
        # plan_vs_disk_image_delta = n_image_files (plan) - n_image_nii_ondisk (disk):
        # 0 = agree, >0 = plan lists images missing on disk, <0 = disk has extra image files.
        "plan_vs_disk_image_delta": len(image_files) - n_img_disk,
        "n_2D-slices_total": n_slices_total,
        "n_2D-slices_by_plane": dict(samp_by_plane),
        "n_annotations": n_annotations,
        "array_size_counts": dict(sorted(array_size_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "voxel_mm_counts": dict(sorted(voxel_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "images_by_modality": dict(img_by_modality),
        "volumes_by_label": dict(vol_by_label),
        "volumes_by_anatomy": dict(vol_by_anatomy),
        "2D-slices_by_modality": dict(samp_by_modality),
        "2D-slices_by_label": dict(samp_by_label),
        "2D-slices_by_anatomy": dict(samp_by_anatomy),
        "measurements": {},
    }
    if unmapped_labels:
        ds_summary["unmapped_labels"] = sorted(unmapped_labels)

    # measurement mean/median into summary (Q2e: no n_volumes)
    for cat, vlist in meas.get("values", {}).items():
        mean, median = _mean_median(vlist)
        ds_summary["measurements"][cat] = {"mean": mean, "median": median, "n": len(vlist)}

    # benchmark VQA-sample counts (BoxSize + T/L + A/D) for the donut — filtered (v1.0.0
    # filters) and raw (unfiltered). BoxSize reads the large detection plan; --no_detection skips.
    (task_f, anat_f), (task_r, anat_r) = count_benchmark_annotations(
        dataset_dir, dataset, skip_detection, version=plan_version)
    ds_summary["annotations_by_task"] = task_f
    ds_summary["n_benchmark_annotations"] = sum(task_f.values())
    ds_summary["boxsize_by_anatomy"] = anat_f
    ds_summary["annotations_by_task_raw"] = task_r
    ds_summary["n_benchmark_annotations_raw"] = sum(task_r.values())
    ds_summary["boxsize_by_anatomy_raw"] = anat_r

    # long-format label rows (ROI stats + measurement rows). mean_ROI_pixel_count = mean ROI area
    # in pixels (raw counterpart of the mm2 `mean`); blank for measurement rows (N/A).
    label_rows = []
    for (modality, name), st in sorted(label_roi.items()):
        roi_mean, roi_med = _mean_median(st["roi"])
        px_mean, _ = _mean_median(st["px"])
        label_rows.append({
            "dataset": dataset, "modality": modality, "category": "label", "name": name,
            "anatomy_group": st["anatomy"], "n_volumes": st["n_volumes"],
            "n_2D-slices": st["n_samples"], "mean": roi_mean, "median": roi_med,
            "mean_ROI_pixel_count": px_mean, "unit": "mm2",
        })
    _unit = {"major_axis_mm": "mm", "minor_axis_mm": "mm", "angle_deg": "degree", "distance_mm": "mm"}
    for cat, vlist in meas.get("values", {}).items():
        mean, median = _mean_median(vlist)
        label_rows.append({
            "dataset": dataset, "modality": ";".join(sorted(modalities)),
            "category": "measurement", "name": cat, "anatomy_group": "",
            "n_volumes": meas.get("n_volumes", {}).get(cat, 0), "n_2D-slices": len(vlist),
            "mean": mean, "median": median, "mean_ROI_pixel_count": None, "unit": _unit.get(cat, ""),
        })

    return file_rows, ds_summary, label_rows


# ── output writers ────────────────────────────────────────────────────────────────────────

_WIDE_COLS = [
    "dataset", "modalities", "anatomy_groups", "labels", "n_labels", "task_types",
    "n_subjects", "n_images_train", "n_images_test", "n_image_files", "n_mask_files",
    "n_landmark_files", "n_nii_ondisk", "n_2D-slices_total", "n_2D-slices_axial",
    "n_2D-slices_coronal", "n_2D-slices_sagittal", "n_annotations", "array_size_range",
    "voxel_mm_range", "mean_major_axis_mm", "median_major_axis_mm", "mean_minor_axis_mm",
    "median_minor_axis_mm", "mean_angle_deg", "median_angle_deg", "mean_distance_mm",
    "median_distance_mm", "mean_ROI_area_mm2", "median_ROI_area_mm2",
]


def _wide_row(dataset, s, roi_mean, roi_med):
    m = s.get("measurements", {})

    def mm(cat, k):
        return m.get(cat, {}).get(k) if m.get(cat) else None

    return {
        "dataset": dataset,
        "modalities": ";".join(s["modalities"]),
        "anatomy_groups": ";".join(s["anatomy_groups"]),
        "labels": ";".join(s["labels"]),
        "n_labels": s["n_labels"],
        "task_types": ";".join(s["task_types"]),
        "n_subjects": s["n_subjects"],
        "n_images_train": s["n_images_train"],
        "n_images_test": s["n_images_test"],
        "n_image_files": s["n_image_files"],
        "n_mask_files": s["n_mask_files"],
        "n_landmark_files": s["n_landmark_files"],
        "n_nii_ondisk": s["n_nii_ondisk"],
        "n_2D-slices_total": s["n_2D-slices_total"],
        "n_2D-slices_axial": s["n_2D-slices_by_plane"].get("Axial", 0),
        "n_2D-slices_coronal": s["n_2D-slices_by_plane"].get("Coronal", 0),
        "n_2D-slices_sagittal": s["n_2D-slices_by_plane"].get("Sagittal", 0),
        "n_annotations": s["n_annotations"],
        "array_size_range": _array_range(s["array_size_counts"]),
        "voxel_mm_range": _voxel_range(s["voxel_mm_counts"]),
        "mean_major_axis_mm": mm("major_axis_mm", "mean"),
        "median_major_axis_mm": mm("major_axis_mm", "median"),
        "mean_minor_axis_mm": mm("minor_axis_mm", "mean"),
        "median_minor_axis_mm": mm("minor_axis_mm", "median"),
        "mean_angle_deg": mm("angle_deg", "mean"),
        "median_angle_deg": mm("angle_deg", "median"),
        "mean_distance_mm": mm("distance_mm", "mean"),
        "median_distance_mm": mm("distance_mm", "median"),
        "mean_ROI_area_mm2": roi_mean,
        "median_ROI_area_mm2": roi_med,
    }


# ── visualization ─────────────────────────────────────────────────────────────────────────

# Fixed, colorblind-safe (Okabe-Ito) hue per modality, kept consistent across the two
# modality panels so a reader can track a modality between the images/slices views.
_MODALITY_COLORS = {
    "CT": "#0072B2",
    "MRI": "#D55E00",
    "PET": "#009E73",
    "ultrasound": "#CC79A7",
    "X Ray": "#E69F00",
}
# Anatomy panels drop only the label-map gap (UNMAPPED), then shade each bar by a
# sequential ramp of its (log) sample size (darkest = largest).
_ANATOMY_EXCLUDE = {"UNMAPPED"}
_ANATOMY_CMAP = "Blues"

# The 3 benchmark tasks (``annotations_by_task`` keys) in fixed display order, shared by the
# two per-task panels so a row is the same task across the filtered/raw pair. Colours are a
# fixed one-hue teal ramp, darkest = largest task (echoing the anatomy panels'
# darkest-=-largest shading), reused from neither the modality palette nor the anatomy Blues
# ramp. A size-ordered ramp, not a categorical trio: it passes the ordinal checks (monotone
# OKLCH L 0.48->0.66, adjacent step dL >= 0.069, lightest step 2.9:1 on white) rather than the
# categorical CVD gate -- task identity is carried by the y labels, never by colour alone.
_TASK_LABELS = {
    "BoxSize": "Detection",
    "TumorLesionSize": "Tumor/Lesion (T/L)",
    "BiometricsFromLandmarks": "Angle/Distance (A/D)",
}
_TASK_COLORS = {
    "Detection": "#05668D",
    "Tumor/Lesion (T/L)": "#028090",
    "Angle/Distance (A/D)": "#00A896",
}


# Tokens dropped when splitting anatomy label names for the wordcloud (articles, sides,
# position words, ordinals) — mirrors the reference plot_labels_wordcloud.py stopword list.
_WORDCLOUD_STOPWORDS = {
    "a", "the", "of", "in", "at", "from", "part", "left", "right", "top", "bottom",
    "upper", "lower", "middle", "surrounding", "peripheral", "central", "inferior",
    "suprior", "anterior", "posterior", "lateral", "medial", "zone", "region", "segment",
    "area", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th",
    "10th", "11th", "12th", "13th", "14th",
}


def _abbrev(n):
    """Compact count label using M for millions and K for thousands: 24276501 -> '24.3M',
    6690918 -> '6.7M', 291550 -> '292K', 15613 -> '15.6K', 400 -> '400'. Values keep one decimal
    below 100 of their unit and go whole above it (e.g. 118000000 -> '118M', 291550 -> '292K')."""
    n = float(n)
    for div, suf in ((1e6, "M"), (1e3, "K")):
        if n >= div:
            v = n / div
            return f"{v:.0f}{suf}" if v >= 100 else f"{v:.1f}{suf}"
    return f"{int(n)}"


def _barh_panel(ax, data, title, *, order=None, exclude=None, color_of=None, shade=None):
    """One horizontal, log-x bar panel with bold labels and end-of-bar value labels.

    ``order`` fixes the category order (top-to-bottom); without it, bars are sorted
    descending by count. ``exclude`` drops categories. Bars are colored by
    ``color_of(cat)``, or — when ``shade`` names a colormap — by a sequential ramp of
    each bar's (log) sample size. The x-axis is log-scaled (counts span ~3 orders of
    magnitude) but unlabelled — exact sample sizes are annotated at each bar end instead.
    """
    exclude = exclude or set()
    if order is not None:
        items = [(k, data[k]) for k in order if k in data and k not in exclude]
    else:
        items = sorted(((k, v) for k, v in data.items() if k not in exclude),
                       key=lambda kv: kv[1], reverse=True)
    cats = [k for k, _ in items]
    vals = [v for _, v in items]
    y = list(range(len(cats)))
    if shade:
        from matplotlib import colormaps

        cmap = colormaps[shade]
        lv = [math.log10(max(v, 1)) for v in vals]
        lo = min(lv) if lv else 0.0
        span = (max(lv) - lo) if lv else 1.0
        # 0.35..0.95 keeps the lightest bar visible and the darkest short of pure ink
        bar_colors = [cmap(0.35 + 0.6 * (x - lo) / (span or 1.0)) for x in lv]
    else:
        bar_colors = [color_of(c) for c in cats]
    ax.barh(y, vals, color=bar_colors, height=0.74, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=8, fontweight="bold")
    ax.invert_yaxis()  # first category at top
    ax.set_xscale("log")
    vmax = max(vals, default=1) or 1  # `or 1`: an all-zero panel (e.g. --no_detection) still scales
    vmin = min((v for v in vals if v > 0), default=1)
    ax.set_xlim(10 ** math.floor(math.log10(vmin)), vmax * 4)  # headroom for value labels
    for yi, v in zip(y, vals):
        ax.text(v * 1.15, yi, _abbrev(v), va="center", ha="left",
                fontsize=7, fontweight="bold", color="#333333")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", which="major", color="#CCCCCC", lw=0.5, alpha=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def viz_summary(all_stats, out_path):
    """Render the 7 collection-level distributions as one compact figure.

    Row 1 (2 panels): images-by-modality and 2D-slices-by-modality.
    Row 2 (2 panels): benchmark sample size per task -- Detection (BoxSize), Tumor/Lesion
    biometry (TumorLesionSize), Angle/Distance biometry (BiometricsFromLandmarks) -- as
    single-instance (``annotations_by_task``, loader-filtered) and multi-instance
    (``annotations_by_task_raw``, unfiltered -- a SUPERSET of the left panel, never its
    complement; A/D has no filter, so its bars match). Fixed ``_TASK_LABELS`` order on both
    panels, so a row is the same task across the pair; a task absent from the summary (e.g.
    BoxSize under ``--no_detection``) still renders as a zero-length bar.
    Row 3 (3 narrower panels): volumes-by-anatomy, then the two annotation views of the same
    anatomy groups -- single-instance (``boxsize_by_anatomy``, the v1.0.0-filtered benchmark
    BoxSize count) and multi-instance (``2D-slices_by_anatomy``, the unfiltered per-(slice,label)
    count). Each anatomy panel sorts by its OWN values descending, so the row order differs
    between panels -- read each panel's own y labels rather than comparing across a row. Bars are
    log-scaled with exact value labels (counts span ~3 orders of magnitude). Saved as a
    transparent vector PDF via ``save_fig_capped`` (project figure convention), plus ``.svg``
    and ``_whitebg.svg`` twins for README / webpage embedding.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from medvision_bm.utils.plot_utils import save_fig_capped

    # The two annotation panels are nested BY DESIGN, not disjoint:
    #   single-instance = (slice, label) items with exactly ONE cluster >= 10 px in both dims
    #   multi-instance  = (slice, label) items with ONE OR MORE clusters, unfiltered (a superset)
    # Both come from the DETECTION plan so the pair stays in the same benchmark units -- the
    # seg-derived 2D-slices_by_anatomy is numerically identical today, but only coincidentally
    # (cf. the same guard in _bench_anatomy). Either may be empty under --no_detection.
    anat_single = all_stats.get("boxsize_by_anatomy") or {}
    anat_multi = all_stats.get("boxsize_by_anatomy_raw") or {}
    anat_panels = [
        ("# 3D Images by Anatomy", all_stats["volumes_by_anatomy"]),
        ("# Single-instance Annotations per Anatomy", anat_single),
        ("# Multi-instance Annotations per Anatomy", anat_multi),
    ]
    # Per-task pair: the same nested single/multi semantics as the anatomy pair, in benchmark
    # units throughout. Dicts may lack a task (or be absent on pre-benchmark summaries) -- the
    # panel builder below fills every _TASK_LABELS row, zero-length bar included.
    task_panels = [
        ("# Single-instance Annotations per Task", all_stats.get("annotations_by_task") or {}),
        ("# Multi-instance Annotations per Task", all_stats.get("annotations_by_task_raw") or {}),
    ]
    n_mod = max(len(all_stats["images_by_modality"]), len(all_stats["2D-slices_by_modality"]), 1)
    n_task = len(_TASK_LABELS)
    n_anat = max(len({k for _, d in anat_panels for k in d} - _ANATOMY_EXCLUDE), 1)
    fig_h = 1.2 + (n_mod + n_task) * 0.30 + n_anat * 0.22  # scale height to the taller rows
    fig = plt.figure(figsize=(15, fig_h))
    # nested grid: rows 1-2 keep 2 half-width panels each, row 3 holds 3 narrower ones (own
    # wspace, since each anatomy panel must still clear its y-tick label column)
    outer = fig.add_gridspec(
        3, 1, height_ratios=[n_mod, n_task, n_anat], hspace=0.30,
        left=0.13, right=0.98, top=0.92, bottom=0.05,
    )
    gs_mod = outer[0].subgridspec(1, 2, wspace=0.34)
    gs_task = outer[1].subgridspec(1, 2, wspace=0.34)
    gs_anat = outer[2].subgridspec(1, 3, wspace=0.52)

    def mod_color(c):
        return _MODALITY_COLORS.get(c, "#888888")

    mod_order = list(_MODALITY_COLORS)  # fixed order, consistent across the two modality panels

    _barh_panel(fig.add_subplot(gs_mod[0, 0]), all_stats["images_by_modality"],
                "# 3D Images by Modality", order=mod_order, color_of=mod_color)
    _barh_panel(fig.add_subplot(gs_mod[0, 1]), all_stats["2D-slices_by_modality"],
                "# 2D Slices by Modality", order=mod_order, color_of=mod_color)
    for i, (title, by_task) in enumerate(task_panels):
        # fixed _TASK_LABELS order (a row = the same task in both panels), missing tasks at 0
        data = {label: by_task.get(key, 0) for key, label in _TASK_LABELS.items()}
        _barh_panel(fig.add_subplot(gs_task[0, i]), data, title,
                    order=list(_TASK_LABELS.values()),
                    color_of=lambda c: _TASK_COLORS.get(c, "#888888"))
    for i, (title, data) in enumerate(anat_panels):
        # no `order=`: each anatomy panel sorts by its OWN values, descending
        _barh_panel(fig.add_subplot(gs_anat[0, i]), data, title,
                    exclude=_ANATOMY_EXCLUDE, shade=_ANATOMY_CMAP)

    save_fig_capped(out_path, fig=fig, bbox_inches="tight", transparent=True)
    # SVG twins for inline README + webpage embedding (browsers can't render PDF in <img>).
    # The canonical .svg stays transparent; only the _whitebg twin is opaque, for dark backdrops.
    svg = os.path.splitext(out_path)[0] + ".svg"
    save_fig_capped(svg, fig=fig, bbox_inches="tight", transparent=True)
    save_fig_capped(os.path.splitext(svg)[0] + "_whitebg.svg", fig=fig,
                    bbox_inches="tight", transparent=False, facecolor="white", edgecolor="white")
    plt.close(fig)


def _bench_count(s):
    """Dataset donut value: benchmark annotation count, falling back to the seg-based count."""
    return s.get("n_benchmark_annotations", s.get("n_annotations", 0))


def _bench_anatomy(s):
    """Dataset anatomy breakdown for the outer ring (benchmark BoxSize).

    Only summaries that PREDATE the benchmark counts (no ``n_benchmark_annotations`` key) fall
    back to the seg-based ``2D-slices_by_anatomy``. A modern summary with an empty
    ``boxsize_by_anatomy`` (e.g. ``--no_detection`` or a pure-A/D dataset) returns ``{}`` so the
    outer ring stays in benchmark units instead of silently mixing in seg-annotation counts.
    """
    if "n_benchmark_annotations" in s:
        return s.get("boxsize_by_anatomy") or {}
    return s.get("2D-slices_by_anatomy") or {}


def _short_ds(name):
    """Shorten the one over-long dataset name so ring labels stay compact."""
    return name.replace("Ceph-Biometrics-400", "Ceph-Bio-400")


def _break_anat(name):
    """Break an anatomy label onto a new line before "Tumor/Lesion" (e.g. "Brain Tumor/Lesion"
    -> "Brain\\nTumor/Lesion") so the long compound names stack instead of overflowing a wedge."""
    return name.replace(" Tumor/Lesion", "\nTumor/Lesion")


# Outer-ring tint gradient: a FIXED blend increment per sub-label, so every dataset shades along
# the same gradient -- a 2-label ring shows that gradient's first 2 steps instead of stretching the
# whole 0..0.85 range across just 2 wedges (which made small rings read as two unrelated colours).
# Step = full range / (largest sub-label count - 1), i.e. 24 labels (TotalSegmentator) span 0..0.85.
_SHADE_STEP = 0.85 / 23
_SHADE_MAX = 0.85


def _shade_palette(base, n):
    """``n`` tints of ``base`` (an RGB(A) tuple): index 0 is the pure base (darkest), each later
    index blends one fixed ``_SHADE_STEP`` further toward white (capped at ``_SHADE_MAX``). Used to
    shade one dataset's sub-label wedges by count rank so a single dataset reads as one hue with a
    light-to-dark gradient (darkest = most annotations)."""
    r, g, b = base[0], base[1], base[2]
    out = []
    for i in range(n):
        f = min(_SHADE_STEP * i, _SHADE_MAX)               # 0 (darkest) .. 0.85 (lightest tint)
        out.append((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f, 1.0))
    return out


# Dataset donut palette. Named palettes live in configs.py (nature_palette_1/2); pick the active one
# here. Colours cycle + lighten per wrap (see _WRAP_LIGHTEN) and are assigned in count-desc ring
# order, so the palette sweeps the donut in listed sequence (biggest wedge = first colour).
_DATASET_COLORS = configs.nature_palette_2   # swap to configs.nature_palette_1 for the other palette

# Each time the colour list wraps, the reused hue is blended this much further toward white, so a
# later round of datasets gets a LIGHTER tint of the same hue (wrap 0 = pure, wrap 1 = 0.33 toward
# white, wrap 2 = 0.55, ...) and no two datasets share a colour.
#
# Deliberately NOT configs.extend_palette: that helper alternates darker/lighter per wrap, which
# scores better on paper (min pairwise dE 10.7 vs 5.3 here) but renders the second round DARKER
# than the base wedges, reading as emphasis rather than as a repeat. A lighten-only variant of it
# reaches 10.6 by rotating hue instead, at the cost of more saturated wedges. Both were tried and
# rejected for this figure: the muted, uniformly-pale wrap below is the intended look. The tradeoff
# accepted here is that rounds 2 and 3 are close to each other (5.3, vs ~2.3 = just-noticeable);
# the wedge labels and the named legend carry identity for the small datasets.
_WRAP_LIGHTEN = 0.33


def _dataset_palette(keys):
    """One colour per dataset, assigned to ``keys`` IN THE ORDER GIVEN. The caller passes datasets in
    ring-draw order (annotation count, desc), so ``_DATASET_COLORS`` sweeps the ring in listed
    sequence (biggest wedge = first colour) and the legend matches. Colours cycle through the list;
    each time it wraps the reused hue is blended further toward white by ``_WRAP_LIGHTEN`` (later
    rounds lighter) so no two datasets share a colour."""
    from matplotlib.colors import to_rgba

    out = {}
    for i, k in enumerate(keys):
        r, g, b, a = to_rgba(_DATASET_COLORS[i % len(_DATASET_COLORS)])
        t = 1.0 - (1.0 - _WRAP_LIGHTEN) ** (i // len(_DATASET_COLORS))   # 0 (pure) .. ->1 (white)
        out[k] = (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t, a)
    return out


def _curved_text(ax, text, radius, center_deg, char_deg, fs, color, flip=False):
    """Draw ``text`` following a circular arc, centred on ``center_deg`` with ``char_deg`` of arc
    per character; each glyph is rotated tangent to the arc. ``flip`` (for bottom-half arcs)
    reverses the glyph order and rotation so the caption stays upright and reads left-to-right."""
    import math

    import matplotlib.patheffects as pe

    stroke = [pe.withStroke(linewidth=2.2, foreground="white")]
    chars = text[::-1] if flip else text
    mid = (len(chars) - 1) / 2.0
    for i, ch in enumerate(chars):
        th = center_deg + (mid - i) * char_deg                 # first glyph at the higher angle
        a = math.radians(th)
        rot = th + 90 if flip else th - 90
        ax.text(radius * math.cos(a), radius * math.sin(a), ch, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, rotation=rot,
                rotation_mode="anchor", path_effects=stroke, zorder=6)


def _radial_rot(theta):
    """Rotation (deg) aligning text along the radius at ``theta`` (so it points to the center),
    flipped on the left half to stay upright."""
    t = theta % 360
    if t > 270:
        return t - 360
    if t > 90:
        return t - 180
    return t


def _curved_inner_label(ax, name, count, mid_deg, r_in, w, fs, color, arc=0.027):
    """Curved inner-ring dataset label: name on the outer arc, count on the inner arc of the inner
    ring, following the sector (used for the two widest datasets). ``arc`` = data-space arc length
    per character, converted to a per-radius ``char_deg`` so letters stay tightly and evenly spaced
    (a fixed ``char_deg`` would gap the letters once the rings are enlarged)."""
    import math

    flip = 90 < (mid_deg % 360) < 270
    r_name, r_count = r_in - w * 0.30, r_in - w * 0.74
    _curved_text(ax, name, r_name, mid_deg, math.degrees(arc / r_name), fs, color, flip)
    _curved_text(ax, count, r_count, mid_deg, math.degrees(arc / r_count), fs, color, flip)


def _place_outside_labels(ax, pending, r_out, r_lab, gap_deg):
    """Render the "at least one anatomy label" leaders that don't fit inside their wedge, floating
    just outside the ring near their own sector.

    ``pending`` = ``[(mid_angle_deg, text, fontsize), ...]``. Each label sits at ``r_lab`` roughly at
    its wedge angle; when neighbours would collide they are nudged apart angularly (min ``gap_deg``,
    then re-centred on the group) so they stay near the target sector rather than in a side column.
    A short leader connects the wedge edge (true angle) to the floated label (adjusted angle)."""
    import math

    import matplotlib.patheffects as pe

    if not pending:
        return
    stroke = [pe.withStroke(linewidth=1.6, foreground="white")]
    order = sorted(range(len(pending)), key=lambda i: pending[i][0])
    orig = [pending[i][0] for i in order]
    adj = list(orig)
    for i in range(1, len(adj)):                               # spread apart, ascending
        if adj[i] < adj[i - 1] + gap_deg:
            adj[i] = adj[i - 1] + gap_deg
    shift = (sum(orig) - sum(adj)) / len(orig)                 # re-centre on the group
    adj = [a + shift for a in adj]
    for k, i in enumerate(order):
        ang0, txt, fs = pending[i]
        a0, aa = math.radians(ang0), math.radians(adj[k])
        x0, y0 = r_out * math.cos(a0), r_out * math.sin(a0)    # wedge outer edge (true angle)
        xt, yt = r_lab * math.cos(aa), r_lab * math.sin(aa)    # floated label (adjusted angle)
        ax.plot([x0, xt], [y0, yt], color="#888888", lw=0.5, zorder=1)
        c = math.cos(aa)
        ha = "center" if abs(c) < 0.35 else ("left" if c > 0 else "right")
        dx = 0.03 if ha == "left" else -0.03 if ha == "right" else 0.0
        ax.text(xt + dx, yt, txt, ha=ha, va="center", fontsize=fs, fontweight="bold",
                color="#1a1a1a", path_effects=stroke, linespacing=0.9, zorder=6)


# Ring label font sizes: donut wedge labels scaled 1.5x, outside leaders match the inside anatomy
# size (so the "at least one anatomy label" fallback reads identically whether it fits or not).
INNER_FS, OUTER_FS = 16.875, 14.625

# Data-space arc length per character for a curved inner label (matches _curved_inner_label's
# default arc), used to test whether a too-long dataset name still fits curved along its own arc.
_CURVED_ARC_PER_CHAR = 0.027

# Datasets pinned to a curved inner label regardless of the geometric radial/curved default (an
# explicit preference; TotalSegmentator's name sits right at the radial-fit boundary otherwise).
_ALWAYS_CURVED = {"TotalSegmentator"}


def _draw_donut(ax, entries, ds_color, *, w, r_out):
    """Draw one 2-ring dataset donut on ``ax``.

    ``entries`` = ``[(key, summary, t1, t2), ...]`` with angles (deg) already assigned. Inner ring:
    one wedge per dataset in ``ds_color[key]``. Outer ring: that dataset's sub-labels
    (``_bench_anatomy``) ordered by count, each wedge a tint of the dataset's own colour.

    Label presence is decided purely by geometry — no hardcoded per-dataset lists. Each candidate
    label is measured (via the renderer) and drawn only where the wedge has room: a radial label's
    stacked lines must span the wedge arc (tangential = ``r*Δθ``) and its longest line must fit the
    ring width (radial = ``w``). A dataset name too long to sit radially falls back to a curved label
    along the arc when the sector is wide enough; otherwise it is omitted. Because the enlarged donut
    zooms the small datasets up, more of their labels clear this test there than in the main donut."""
    import math

    import matplotlib.patheffects as pe
    from matplotlib.patches import Wedge

    stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
    r_in = r_out - w
    renderer = ax.figure.canvas.get_renderer()
    sx = abs(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])  # px / data unit

    def _room(txt, r, dtheta_deg, radial_room, fs):        # True iff a radial label fits the wedge
        t = ax.text(0, 0, txt, fontsize=fs, fontweight="bold", ha="center", va="center",
                    linespacing=0.9)
        bb = t.get_window_extent(renderer)
        t.remove()
        arc = r * math.radians(dtheta_deg)                 # tangential room at this radius
        return bb.height / sx <= arc and bb.width / sx <= radial_room

    def _lab(t1, t2, r, txt, fs):                          # radial label: points to the center
        mid = (t1 + t2) / 2.0
        a = math.radians(mid)
        ax.text(r * math.cos(a), r * math.sin(a), txt, ha="center", va="center",
                rotation=_radial_rot(mid), rotation_mode="anchor",
                fontsize=fs, fontweight="bold", color="#1a1a1a",
                path_effects=stroke, linespacing=0.9, zorder=6)

    for k, s, t1, t2 in entries:
        base = ds_color[k]
        ax.add_patch(Wedge((0, 0), r_in, t1, t2, width=w, facecolor=base,
                           edgecolor="white", linewidth=0.7))
        anat = {g: n for g, n in _bench_anatomy(s).items()
                if g not in _ANATOMY_EXCLUDE and n}
        if anat:
            ordered = sorted(anat.items(), key=lambda gn: gn[1], reverse=True)   # by count desc
            shades = _shade_palette(base, len(ordered))
            asum = sum(anat.values())
            a2 = t2
            for (g, n), col in zip(ordered, shades):
                a1 = a2 - (t2 - t1) * n / asum
                ax.add_patch(Wedge((0, 0), r_out, a1, a2, width=w, facecolor=col,
                                   edgecolor="white", linewidth=0.3))
                txt = f"{_break_anat(g)}\n{_abbrev(n)}"     # draw anatomy label only if it fits
                if _room(txt, (r_in + r_out) / 2.0, a2 - a1, w, OUTER_FS):
                    _lab(a1, a2, (r_in + r_out) / 2.0, txt, OUTER_FS)
                a2 = a1
        else:
            ax.add_patch(Wedge((0, 0), r_out, t1, t2, width=w, facecolor="#DDDDDD",
                               edgecolor="white", linewidth=0.3))
        name, cnt = _short_ds(k), _abbrev(_bench_count(s))  # inner dataset label: radial, else curved
        curved_ok = len(name) * _CURVED_ARC_PER_CHAR <= (r_in - 0.30 * w) * math.radians(t2 - t1)
        if k in _ALWAYS_CURVED and curved_ok:               # explicit curved override (e.g. TotalSeg)
            _curved_inner_label(ax, name, cnt, (t1 + t2) / 2.0, r_in, w, INNER_FS, "#1a1a1a")
        elif _room(f"{name}\n{cnt}", r_in - w / 2.0, t2 - t1, w, INNER_FS):
            _lab(t1, t2, r_in - w / 2.0, f"{name}\n{cnt}", INNER_FS)
        elif curved_ok:
            _curved_inner_label(ax, name, cnt, (t1 + t2) / 2.0, r_in, w, INNER_FS, "#1a1a1a")


def viz_rings(all_summary, out_path, magnify=True, layout="2x1", variant="filtered"):
    """Render the collection as a 2-ring donut of benchmark annotation counts.

    * inner — one wedge per dataset, angle proportional to ``n_benchmark_annotations``; colour per
      the reference palette (``_dataset_palette``).
    * outer — that dataset's BoxSize annotations split by sub-label (anatomy), each wedge a tint of
      the dataset's own hue, ordered by count (darkest = most). Anatomy labels sit inside the wedge;
      each labelled dataset is guaranteed at least one anatomy label, floated outside with a leader
      when nothing fits inside.

    When ``magnify`` is true the datasets too small to label in the main donut are re-drawn in a
    second, same-size, same-style donut using the SAME proportional sectors (zoomed to fill the
    circle); the small-dataset arc in the main donut is enclosed by a dotted outline captioned
    "small datasets". ``layout`` is ``"2x1"`` (stacked) or ``"1x2"`` (side by side). The main donut's
    centre reports the collection totals; a dataset colour legend (width <= one donut) runs along the
    bottom. Transparent vector PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Wedge

    from medvision_bm.utils.plot_utils import save_fig_capped

    ordered = [(k, v) for k, v in all_summary.items() if k != "__all__"]
    if not any("n_benchmark_annotations" in v for _, v in ordered):
        print("  [warn] no n_benchmark_annotations in summary; donut falls back to seg-based counts")

    datasets = [(k, v) for k, v in ordered if _bench_count(v) > 0]
    datasets.sort(key=lambda kv: _bench_count(kv[1]), reverse=True)
    total = sum(_bench_count(v) for _, v in datasets) or 1

    # colours follow the RING order (datasets sorted by count desc), so _DATASET_COLORS sweeps the
    # ring in listed sequence -- biggest wedge = first colour -- and the legend below matches.
    ds_color = _dataset_palette([k for k, _ in datasets])

    W, R_OUT = 0.32, 1.0                       # ring width; center hole = 1 - 2*W = 0.36
    START, P_NAME = 90.0, 0.03                  # start at 12 o'clock; min proportion for an inner label
    INNER_MIN = P_NAME * 360.0                  # deg: below this a dataset is "small" -> enlarged panel

    # unfiltering lifts the collection total (x1.87) far more than it lifts CAMUS (x1.41, 951K ->
    # 1.34M -- US structures rarely trip the multi-cluster/size filters), so CAMUS's share falls
    # 3.92% -> 2.96% and slips just under the 3% "small" cutoff, where it then dominates the zoom;
    # keep it out of the enlarged panel (and its dotted arc) for raw.
    enlarge_exclude = {"CAMUS"} if variant == "raw" else set()

    # proportional spans for the main donut (clockwise from 12 o'clock); tail = too small to label
    spans = [360.0 * _bench_count(v) / total for _, v in datasets]
    main_entries, cur, tail = [], START, []
    for (k, s), span in zip(datasets, spans):
        main_entries.append((k, s, cur - span, cur))
        if span < INNER_MIN and k not in enlarge_exclude:
            tail.append((k, s))
        cur -= span

    # tight limits (no more leaders -> all content is within the ring; top room for the caption)
    xlim, ylim = (-1.02, 1.02), (-1.02, 1.06)
    two = magnify and bool(tail)
    if two and layout == "1x2":                             # side by side + legend below (compact-sized)
        fig = plt.figure(figsize=(34, 20))                 # narrower: closes the inter-donut gap
        ax = fig.add_axes([0.005, 0.14, 0.49, 0.85])
        axm = fig.add_axes([0.505, 0.14, 0.49, 0.85])
        leg_rect, leg_ncol = [0.04, 0.0, 0.92, 0.12], 11
    elif two:                                               # stacked + legend below (compact-sized)
        fig = plt.figure(figsize=(22.5, 37))               # height tuned: keep donut size, tighten panel gap
        ax = fig.add_axes([0.16, 0.545, 0.68, 0.43])
        axm = fig.add_axes([0.16, 0.115, 0.68, 0.43])
        leg_rect, leg_ncol = [0.16, 0.0, 0.68, 0.10], 4
    else:                                                   # single donut: tight square, legend flush
        fig = plt.figure(figsize=(22.5, 20))           # donut x1.5 (fonts fixed -> larger rings)
        ax = fig.add_axes([0.145, 0.155, 0.711, 0.8])  # lifted a touch so the legend title clears the ring
        axm = None
        leg_rect, leg_ncol = [0.145, 0.0, 0.711, 0.125], 4
        xlim, ylim = (-1.02, 1.02), (-1.02, 1.02)          # tight to content (no leaders in the main donut)
    for a in (ax, axm):
        if a is None:
            continue
        a.set_aspect("equal")
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        a.axis("off")

    _draw_donut(ax, main_entries, ds_color, w=W, r_out=R_OUT)

    # center: title (doubled) over the 3 collection totals
    allk = all_summary.get("__all__", {})
    n_img = allk.get("n_image_files") or sum(v.get("n_image_files", 0) for _, v in ordered)
    n_sl = allk.get("n_2D-slices_total") or sum(v.get("n_2D-slices_total", 0) for _, v in ordered)
    # white halo so the center text stays legible on a transparent background (any page theme)
    ax.text(0, 0.11, "MedVision", ha="center", va="center", fontsize=56.4, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=5, foreground="white")])
    ax.text(0, -0.09, f"{_abbrev(n_img)} 3D images\n{_abbrev(n_sl)} 2D slices\n{_abbrev(total)} annotations",
            ha="center", va="center", fontsize=28.2, fontweight="bold", linespacing=1.5,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])

    # enclose the small-dataset arc with a dotted outline + curved "small datasets" caption
    if two:
        ta = [(t1, t2) for k, _, t1, t2 in main_entries
              if (t2 - t1) < INNER_MIN and k not in enlarge_exclude]
        lo, hi = min(t1 for t1, _ in ta), max(t2 for _, t2 in ta)
        ax.add_patch(Wedge((0, 0), R_OUT + 0.02, lo, hi, width=2 * W + 0.04, fill=False,
                           edgecolor="#d62728", linewidth=1.8, linestyle=(0, (2, 2)), zorder=5))
        _curved_text(ax, "small datasets", R_OUT + 0.045, (lo + hi) / 2.0,
                     char_deg=1.9, fs=16, color="#d62728")

    # enlarged donut: the small (tail) datasets, drawn as an INCOMPLETE ring (open at the top) to
    # signal it is a zoom-in of the small-dataset arc, not a whole collection.
    if two:
        FAN = 350.0                                        # arc covered; the 10deg opening = zoom cue
        tail_total = sum(_bench_count(s) for _, s in tail) or 1
        # +180 rotates the enlarged donut half a turn (opening then sits at the bottom)
        m_entries, cur = [], START - (360.0 - FAN) / 2.0 + 180.0
        for k, s in tail:
            span = FAN * _bench_count(s) / tail_total
            m_entries.append((k, s, cur - span, cur))
            cur -= span
        # same geometric labeller: the tail is zoomed up here, so more labels clear the fit test
        _draw_donut(axm, m_entries, ds_color, w=W, r_out=R_OUT)
        axm.text(0, 0, "small datasets", ha="center", va="center",
                 fontsize=28, fontweight="bold", color="#d62728", linespacing=1.4)

    # dataset color legend along the bottom, width <= one donut (anatomy legend removed)
    axl = fig.add_axes(leg_rect)
    axl.axis("off")
    handles = [Patch(facecolor=ds_color[k], label=f"{_short_ds(k)} ({_abbrev(_bench_count(s))})")
               for k, s in datasets]
    # NESTED, NOT DISJOINT. "single-instance" = the items the v1.0.0 filters keep (BoxSize: exactly
    # one cluster, >= 10 px in both dims; T/L: a single-cluster slice); "multi-instance" = the same
    # items UNFILTERED, i.e. ">= 1 instance", so it is a strict SUPERSET (24.3M contained in 45.3M),
    # never the complement -- do not add the two variants. Neither counter tallies components: N
    # clusters of one label on one slice count ONCE, not N; the variants differ only in which items
    # are admitted. SCOPE: these legend values are _bench_count, the 3-task total (24,279,534 /
    # 45,338,754), while the outer ring -- and the anatomy panels in dataset_summary.pdf -- are
    # BoxSize-only (24,236,327 / 45,274,250). A/D is unfiltered, so its 7,925 is equal on both sides.
    leg_title = "# Multi-instance Annotations" if variant == "raw" else "# Single-instance Annotations"
    leg = axl.legend(handles=handles, loc="center", ncol=leg_ncol, frameon=False, fontsize=17,
                     title=leg_title, title_fontsize=28.5,
                     handlelength=1.1, columnspacing=1.4, labelspacing=0.5, handletextpad=0.5)
    leg.get_title().set_fontweight("bold")

    save_fig_capped(out_path, fig=fig, bbox_inches="tight", transparent=True)
    if magnify and layout in ("2x1", "1x2"):   # SVG twin for inline README + webpage embedding (GitHub can't render PDF)
        svg = os.path.splitext(out_path)[0] + ".svg"
        save_fig_capped(svg, fig=fig, bbox_inches="tight", transparent=True)
        # White-background twin, served to GitHub dark mode via a <picture> element in the README
        # (README HTML strips inline CSS, so the backing must live in a file). The canonical .svg
        # above stays transparent; only this twin is opaque.
        save_fig_capped(os.path.splitext(svg)[0] + "_whitebg.svg", fig=fig,
                        bbox_inches="tight", transparent=False, facecolor="white", edgecolor="white")
    plt.close(fig)


def viz_wordcloud(all_summary, out_path):
    """Word cloud of anatomy label names, weighted by dataset test-set size.

    Follows the reference ``plot_labels_wordcloud.py``: each dataset's labels
    (``2D-slices_by_label`` keys) are tokenized (split on whitespace and '/', lowercased, a
    stopword list of articles/sides/ordinals dropped), and every label is weighted EQUALLY by its
    dataset's test-set size ``ceil(n_images_test / 10)`` — so a word's size reflects how many
    test cases across datasets use it, not its per-slice prevalence. Datasets with no
    segmentation labels (e.g. Ceph-Biometrics-400) contribute nothing. Rendered on a TRANSPARENT
    background (per the figure-generation convention) and saved as a dpi-capped PDF via
    ``save_fig_capped``. Requires the ``wordcloud`` package; if it is not installed the figure is
    skipped with a warning (so the rest of ``--viz`` still runs).
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  [warn] wordcloud package not installed; skipping wordcloud figure "
              "(pip install wordcloud)")
        return

    from medvision_bm.utils.plot_utils import save_fig_capped

    freq = Counter()
    for k, v in all_summary.items():
        if k == "__all__":
            continue
        labels = v.get("2D-slices_by_label") or {}
        weight = int(math.ceil((v.get("n_images_test") or 0) / 10.0))
        if not labels or weight <= 0:
            continue
        for name in labels:                        # each label once, weighted by dataset test size
            for token in name.split():
                for part in token.split("/"):
                    word = part.lower().strip().strip("(").strip(")")
                    if word and word not in _WORDCLOUD_STOPWORDS:
                        freq[word] += weight
    if not freq:
        print("  [warn] no label words for wordcloud; skipping")
        return

    # style pinned to the reference plot_labels_wordcloud.py (1200x800, steelblue contour), but on a
    # TRANSPARENT background (background_color=None + RGBA mode) instead of white
    wc = WordCloud(width=1200, height=800, background_color=None, mode="RGBA", max_words=80,
                   contour_width=3, contour_color="steelblue").generate_from_frequencies(freq)
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    save_fig_capped(out_path, fig=fig, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _raw_view(all_summary):
    """A deep copy of the summary with the RAW (unfiltered) benchmark counts swapped into the
    canonical fields, so ``viz_rings`` renders the raw variant from it."""
    import copy

    out = copy.deepcopy(all_summary)
    for v in out.values():
        if "n_benchmark_annotations_raw" in v:
            v["n_benchmark_annotations"] = v["n_benchmark_annotations_raw"]
            v["annotations_by_task"] = v.get("annotations_by_task_raw", {})
            v["boxsize_by_anatomy"] = v.get("boxsize_by_anatomy_raw", {})
    return out


def render_figures(all_summary, out_dir):
    """Render every figure and return the output paths: bar panels, word cloud, and the donut in
    ``filtered`` + ``raw`` variants (each as a magnified-inset and a compact version)."""
    outs = []
    p = os.path.join(out_dir, "dataset_summary.pdf")
    viz_summary(all_summary["__all__"], p)
    outs.append(p)
    p = os.path.join(out_dir, "dataset_summary_wordcloud.pdf")
    viz_wordcloud(all_summary, p)
    outs.append(p)
    for tag, summ in (("filtered", all_summary), ("raw", _raw_view(all_summary))):
        for lay in ("2x1", "1x2"):                 # the 2-panel magnify figure in both arrangements
            p = os.path.join(out_dir, f"dataset_summary_rings_{tag}_{lay}.pdf")
            viz_rings(summ, p, magnify=True, layout=lay, variant=tag)
            outs.append(p)
        p = os.path.join(out_dir, f"dataset_summary_rings_{tag}_compact.pdf")
        viz_rings(summ, p, magnify=False, variant=tag)
        outs.append(p)
    return outs


def main():
    parser = argparse.ArgumentParser(description="Summarize the local Data/Datasets collection.")
    parser.add_argument("--data_dir", required=True, help="MedVision Data directory (contains Datasets/).")
    parser.add_argument("--datasets", default=None, help="Comma-separated subset (default: all).")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: a 'dataset-info/datasets_summary' dir alongside "
                             "--data_dir, e.g. <repo>/dataset-info/datasets_summary).")
    parser.add_argument("--plan_version", default=None, help="Biometry plan version (default: highest).")
    parser.add_argument("--no_detection", action="store_true",
                        help="Skip the v1.0.0 detection-plan pass (BoxSize benchmark counts). Much "
                             "faster (~8.5 min) and low-memory, but n_benchmark_annotations then omits "
                             "BoxSize. Default: BoxSize is counted (loads large detection plans).")
    parser.add_argument("--viz", action="store_true",
                        help="Also render dataset_summary.pdf (bar panels, incl. sample size "
                             "per task: detection / T/L / A/D), "
                             "dataset_summary_wordcloud.pdf (anatomy word cloud; needs `pip install "
                             "wordcloud` -- NOT a declared dependency, and the figure is silently "
                             "skipped with a warning when it is missing), and the 2-ring donut in "
                             "filtered + raw benchmark counts, each as a magnified-inset "
                             "(dataset_summary_rings_{filtered,raw}_{2x1,1x2}.pdf) and a compact "
                             "(…_compact.pdf) variant.")
    parser.add_argument("--viz_only", action="store_true",
                        help="Figure-only: skip the scan, render all figures from the existing "
                             "dataset_summary_filtered.json in --out_dir, then exit.")
    parser.add_argument("--reuse_from", default=None,
                        help="Efficient version regen: reuse the version-invariant Box / segmentation "
                             "/ inventory fields from this existing summary dir and recompute only the "
                             "biometry (T/L sample counts + measurement stats) for --plan_version. "
                             "Writes the summary JSONs (and figures with --viz) to --out_dir without "
                             "re-reading the multi-GB detection plans.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.data_dir)), "dataset-info", "datasets_summary")

    if args.viz_only:
        summary_path = os.path.join(out_dir, "dataset_summary_filtered.json")
        if not os.path.exists(summary_path):
            parser.error(f"--viz_only: no existing summary at {summary_path}; "
                         "run without --viz_only first (optionally with --viz).")
        with open(summary_path) as f:
            all_summary = json.load(f)
        print("viz (from existing dataset_summary_filtered.json) ->")
        for p in render_figures(all_summary, out_dir):
            print(f"  {p}")
        return

    if args.reuse_from:
        base_json = os.path.join(args.reuse_from, "dataset_summary_filtered.json")
        if not os.path.exists(base_json):
            parser.error(f"--reuse_from: no base summary at {base_json}; run a full scan there first.")
        with open(base_json) as f:
            per_dataset = {k: v for k, v in json.load(f).items() if k != "__all__"}
        datasets_root = os.path.join(args.data_dir, "Datasets")
        os.makedirs(out_dir, exist_ok=True)
        # Box, segmentation and all inventory fields are version-invariant; only the biometry-derived
        # fields change with the plan version, so recompute just those and patch them into the base.
        for name, s in per_dataset.items():
            ddir = os.path.join(datasets_root, name)
            n_tl_f, _n_ad, n_tl_r = _count_biometry_samples(ddir, args.plan_version)
            for key, n in (("annotations_by_task", n_tl_f), ("annotations_by_task_raw", n_tl_r)):
                abt = dict(s.get(key) or {})
                if n:
                    abt["TumorLesionSize"] = n
                else:
                    abt.pop("TumorLesionSize", None)
                s[key] = abt
            s["n_benchmark_annotations"] = sum(s["annotations_by_task"].values())
            s["n_benchmark_annotations_raw"] = sum(s["annotations_by_task_raw"].values())
            meas = process_measurements(ddir, args.plan_version)
            s["measurements"] = {}
            for cat, vlist in meas.get("values", {}).items():
                mean, median = _mean_median(vlist)
                s["measurements"][cat] = {"mean": mean, "median": median, "n": len(vlist)}
        all_summary = {"__all__": _collection_rollup(per_dataset), **per_dataset}
        with open(os.path.join(out_dir, "dataset_summary_filtered.json"), "w") as f:
            f.write(json.dumps(all_summary, indent=2))
        with open(os.path.join(out_dir, "dataset_summary_raw.json"), "w") as f:
            json.dump(_raw_view(all_summary), f, indent=2)
        if args.viz:
            for p in render_figures(all_summary, out_dir):
                print(f"  viz -> {p}")
        tot = all_summary["__all__"]
        print(f"reuse (biometry v{args.plan_version}) from {args.reuse_from} -> {out_dir}\n"
              f"  benchmark annotations: {tot['n_benchmark_annotations']:,} single-instance / "
              f"{tot['n_benchmark_annotations_raw']:,} multi-instance")
        return

    datasets_root = os.path.join(args.data_dir, "Datasets")
    os.makedirs(out_dir, exist_ok=True)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = sorted(
            d for d in os.listdir(datasets_root)
            if os.path.isdir(os.path.join(datasets_root, d)) and not d.endswith("_regen")
        )

    # Drop datasets that had not been published at --plan_version: they carry no plan at or before
    # it, so they belong to a later release and must be absent from this summary entirely (not
    # listed with zeros). This is what lets one Datasets/ root holding every dataset still produce
    # a faithful summary for an older version.
    if args.plan_version:
        skipped = [d for d in datasets
                   if not dataset_exists_at(os.path.join(datasets_root, d), args.plan_version)]
        if skipped:
            datasets = [d for d in datasets if d not in set(skipped)]
            print(f"[scope] v{args.plan_version}: skipping {len(skipped)} dataset(s) first "
                  f"published later: {', '.join(skipped)}")

    files_path = os.path.join(out_dir, "dataset_files.jsonl")
    per_dataset, all_label_rows = {}, []
    all_roi = []  # for TOTAL row ROI mean

    with open(files_path, "w") as fjsonl:
        for i, dataset in enumerate(datasets, 1):
            file_rows, ds_summary, label_rows = process_dataset(
                datasets_root, dataset, args.plan_version, skip_detection=args.no_detection)
            for row in file_rows:
                fjsonl.write(json.dumps(row) + "\n")
            per_dataset[dataset] = ds_summary
            all_label_rows.extend(label_rows)
            # gather ROI values for the dataset wide row + grand total
            roi_vals = [v for r in label_rows if r["category"] == "label"
                        for v in ([r["mean"]] * r["n_2D-slices"] if r["mean"] is not None else [])]
            all_roi.extend(roi_vals)
            print(
                f"[{i}/{len(datasets)}] {dataset}: {ds_summary['n_image_files']} images, "
                f"{ds_summary['n_mask_files']} masks, {ds_summary['n_2D-slices_total']} 2D-slices, "
                f"{ds_summary['n_annotations']} seg-annotation, "
                f"{ds_summary['n_benchmark_annotations']} benchmark-annotation (filtered) "
                f"{ds_summary['annotations_by_task']}, "
                f"modalities={ds_summary['modalities']}"
                + (f"  [UNMAPPED: {ds_summary.get('unmapped_labels')}]" if ds_summary.get('unmapped_labels') else "")
            )

    # collection-level rollup + summary JSON (full, plus filtered/raw benchmark-count variants)
    all_summary = {"__all__": _collection_rollup(per_dataset), **per_dataset}
    # dataset_summary_filtered.json is the default filtered view; _raw.json is the raw-count view.
    with open(os.path.join(out_dir, "dataset_summary_filtered.json"), "w") as f:
        f.write(json.dumps(all_summary, indent=2))
    with open(os.path.join(out_dir, "dataset_summary_raw.json"), "w") as f:
        json.dump(_raw_view(all_summary), f, indent=2)

    if args.viz:
        for p in render_figures(all_summary, out_dir):
            print(f"  viz -> {p}")

    # wide CSV (per dataset + TOTAL)
    def roi_stats_for(dataset):
        rows = [r for r in all_label_rows if r["dataset"] == dataset and r["category"] == "label"]
        vals = [v for r in rows for v in ([r["mean"]] * r["n_2D-slices"] if r["mean"] is not None else [])]
        return _mean_median(vals)

    wide_path = os.path.join(out_dir, "dataset_summary.csv")
    with open(wide_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_WIDE_COLS)
        w.writeheader()
        for dataset in datasets:
            rm, rmed = roi_stats_for(dataset)
            w.writerow(_wide_row(dataset, per_dataset[dataset], rm, rmed))
        w.writerow(_total_row(per_dataset, _mean_median(all_roi)))

    # long CSV
    long_path = os.path.join(out_dir, "dataset_label_stats.csv")
    long_cols = ["dataset", "modality", "category", "name", "anatomy_group", "n_volumes",
                 "n_2D-slices", "mean", "median", "mean_ROI_pixel_count", "unit"]
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        for row in all_label_rows:
            w.writerow(row)

    tot = all_summary["__all__"]
    print(
        f"\nWrote {len(datasets)} datasets to {out_dir}\n"
        f"  TOTAL: {tot['n_datasets']} datasets, {tot['n_image_files']} images, "
        f"{tot['n_mask_files']} masks, {tot['n_nii_ondisk']} nii on disk, "
        f"{tot['n_2D-slices_total']} 2D-slices, {tot['n_annotations']} annotations\n"
        f"  files -> {files_path}\n"
        f"  summary JSON -> dataset_summary_filtered.json + dataset_summary_raw.json\n"
        f"  CSV -> dataset_summary.csv (wide) + dataset_label_stats.csv (long)\n"
        f"  all in {out_dir}\n"
    )


def _collection_rollup(per_dataset):
    agg = {
        "n_datasets": len(per_dataset),
        "n_image_files": 0, "n_mask_files": 0, "n_nii_ondisk": 0, "n_subjects": 0,
        "n_2D-slices_total": 0, "n_annotations": 0,
        "n_benchmark_annotations": 0, "n_benchmark_annotations_raw": 0,
        "images_by_modality": Counter(), "2D-slices_by_modality": Counter(),
        "volumes_by_anatomy": Counter(), "2D-slices_by_anatomy": Counter(),
        "annotations_by_task": Counter(), "boxsize_by_anatomy": Counter(),
        "annotations_by_task_raw": Counter(), "boxsize_by_anatomy_raw": Counter(),
    }
    for s in per_dataset.values():
        for k in ("n_image_files", "n_mask_files", "n_nii_ondisk", "n_subjects",
                  "n_2D-slices_total", "n_annotations",
                  "n_benchmark_annotations", "n_benchmark_annotations_raw"):
            agg[k] += s.get(k, 0)
        agg["images_by_modality"].update(s["images_by_modality"])
        agg["2D-slices_by_modality"].update(s["2D-slices_by_modality"])
        agg["volumes_by_anatomy"].update(s["volumes_by_anatomy"])
        agg["2D-slices_by_anatomy"].update(s["2D-slices_by_anatomy"])
        agg["annotations_by_task"].update(s.get("annotations_by_task", {}))
        agg["boxsize_by_anatomy"].update(s.get("boxsize_by_anatomy", {}))
        agg["annotations_by_task_raw"].update(s.get("annotations_by_task_raw", {}))
        agg["boxsize_by_anatomy_raw"].update(s.get("boxsize_by_anatomy_raw", {}))
    for k in ("images_by_modality", "2D-slices_by_modality", "volumes_by_anatomy",
              "2D-slices_by_anatomy", "annotations_by_task", "boxsize_by_anatomy",
              "annotations_by_task_raw", "boxsize_by_anatomy_raw"):
        agg[k] = dict(agg[k])
    return agg


def _total_row(per_dataset, roi_meanmed):
    tot = {c: "" for c in _WIDE_COLS}
    tot["dataset"] = "TOTAL"
    for c in ("n_subjects", "n_images_train", "n_images_test", "n_image_files",
              "n_mask_files", "n_landmark_files", "n_nii_ondisk", "n_2D-slices_total",
              "n_2D-slices_axial", "n_2D-slices_coronal", "n_2D-slices_sagittal", "n_annotations"):
        tot[c] = 0
    modalities, labels, anatomy = set(), set(), set()
    for s in per_dataset.values():
        modalities |= set(s["modalities"])
        labels |= set(s["labels"])
        anatomy |= set(s["anatomy_groups"])
        for c in ("n_subjects", "n_images_train", "n_images_test", "n_image_files",
                  "n_mask_files", "n_landmark_files", "n_nii_ondisk", "n_2D-slices_total",
                  "n_annotations"):
            tot[c] += s[c]
        tot["n_2D-slices_axial"] += s["n_2D-slices_by_plane"].get("Axial", 0)
        tot["n_2D-slices_coronal"] += s["n_2D-slices_by_plane"].get("Coronal", 0)
        tot["n_2D-slices_sagittal"] += s["n_2D-slices_by_plane"].get("Sagittal", 0)
    # union (distinct across datasets), consistent with modalities; n_labels counts the union,
    # not the sum of per-dataset counts (which double-counts labels shared across datasets).
    tot["modalities"] = ";".join(sorted(modalities))
    tot["labels"] = ";".join(sorted(labels))
    tot["anatomy_groups"] = ";".join(sorted(anatomy))
    tot["n_labels"] = len(labels)
    tot["mean_ROI_area_mm2"], tot["median_ROI_area_mm2"] = roi_meanmed
    return tot


if __name__ == "__main__":
    main()
