"""Shared helpers for reading MedVision ``benchmark_plan_*.json.gz`` files locally.

Each dataset folder under ``Data/Datasets/<DS>/`` ships gzip-compressed benchmark plans
(``benchmark_plan_{segmentation,detection,biometry}_v*.json.gz``) that enumerate every case
with geometry (``array_size``, ``voxel_size``, ``orientation``) and per-slice profiles. These
helpers centralize the plane<->axis convention and the (two) slice_profile schemas so
``summarize_datasets.py`` can read them without touching HF or nibabel.
"""

import functools
import glob
import gzip
import json
import os
import sys

# (dataset_dir, plan_type, version) tuples already warned about, so a version-fallback prints once.
_warned_versions = set()

# All MedVision volumes are stored RAS+, so ``array_size`` is [X(R/L), Y(A/P), Z(S/I)].
# Slicing along an axis drops that axis; the 2D (H, W) is the remaining two, in array order.
AXIS_TO_PLANE = {"x": "Sagittal", "y": "Coronal", "z": "Axial"}
PLANE_TO_AXIS = {v: k for k, v in AXIS_TO_PLANE.items()}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Benchmark family -> the plan whose ``slice_profiles`` supply that family's 2D image sizes.
# ``array_size`` is identical across plan types for a given case, so we deliberately route the
# box family through the (small) segmentation plan instead of the enormous detection plan
# (which stores every bbox on every slice and can be 100s of MB). T/L samples are the ellipse-fit
# slices, which live in the biometry plan (major/minor axis metrics), not the raw masks.
FAMILY_TO_PLAN_TYPE = {
    "boxsize": "segmentation",
    "masksize": "segmentation",
    "tumorlesionsize": "biometry",
    "biometricsfromlandmarks": "biometry",
}


def find_plan_files(dataset_dir, plan_type):
    """Return plan files of ``plan_type`` sorted ascending by version string."""
    pattern = os.path.join(dataset_dir, f"benchmark_plan_{plan_type}_v*.json.gz")
    return sorted(glob.glob(pattern))


@functools.lru_cache(maxsize=2)
def load_benchmark_plan(dataset_dir, plan_type, version=None):
    """Load one benchmark plan dict, or ``None`` if absent.

    ``version=None`` picks the highest available version; otherwise an exact match on the
    ``_v<version>.json.gz`` suffix. If the requested version is absent for this dataset, fall back
    to the highest available version (warning once) rather than returning ``None`` — this keeps a
    pinned ``--plan_version`` from silently emptying datasets that only ship an older version
    (e.g. Ceph/FeTA have biometry ``v1.0.0`` only). Returns ``None`` only when the plan type is
    entirely absent. Cached (maxsize=2) so a dataset's plan is not re-read once per task; callers
    must treat the returned dict as read-only.
    """
    files = find_plan_files(dataset_dir, plan_type)
    if not files:
        return None
    if version is None:
        path = files[-1]
    else:
        path = next(
            (f for f in files if os.path.basename(f).endswith(f"_v{version}.json.gz")), None
        )
        if path is None:
            path = files[-1]  # requested version missing -> highest available
            key = (dataset_dir, plan_type, version)
            if key not in _warned_versions:
                _warned_versions.add(key)
                print(
                    f"[plan_utils] {os.path.basename(dataset_dir)}: {plan_type} plan v{version} "
                    f"not found; using {os.path.basename(path)} instead",
                    file=sys.stderr,
                )
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def split_cases(task, split):
    """Return the case list for ``split`` ('train'/'test'/'all')."""
    if split == "train":
        return task.get("train_cases", []) or []
    if split == "test":
        return task.get("test_cases", []) or []
    return (task.get("train_cases", []) or []) + (task.get("test_cases", []) or [])


def slice_2d_size(array_size, axis):
    """2D (H, W) of a slice cut along ``axis`` ('x'/'y'/'z'), dropping that axis."""
    dims = [d for i, d in enumerate(array_size) if i != _AXIS_INDEX[axis]]
    return int(dims[0]), int(dims[1])


def slice_entries(case, axis):
    """The per-slice entry list for one axis (``[]`` if missing)."""
    return case.get(f"slice_profiles_{axis}", []) or []


def anatomy_group(fine_label):
    """Coarse anatomy group for a fine label via ``configs.label_map_regroup``.

    Returns ``"UNMAPPED"`` for labels not in the map (rather than raising), so a summary run
    never crashes on a newly added label — unmapped labels can be surfaced by the caller.
    """
    from medvision_bm.utils.configs import label_map_regroup

    return label_map_regroup.get(fine_label, "UNMAPPED")
