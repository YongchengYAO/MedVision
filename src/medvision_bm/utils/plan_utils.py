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


def plan_version_of(path):
    """Version tuple parsed from a ``benchmark_plan_<type>_v<X.Y.Z>.json.gz`` filename."""
    return tuple(int(p) for p in os.path.basename(path).rsplit("_v", 1)[1].split(".json")[0].split("."))


def resolve_plan_path(dataset_dir, plan_type, version=None):
    """Path of the newest ``plan_type`` plan published at or before ``version``, or ``None``.

    This is the loader's **ceiling** rule — *the newest annotation that existed at or before this
    point* — and it is the single resolution rule for every plan family, so a summary run cannot
    mix versions across families.

    ``version=None`` means the newest available. ``None`` is returned when the family is absent,
    **or when nothing was published at or before ``version``** — i.e. the dataset did not exist yet
    at that version, so it must contribute nothing to that version's summary.

    This deliberately replaced an "exact match, else highest available" rule. The old fallback
    existed so a pinned ``--plan_version`` could not silently empty a dataset shipping only an
    OLDER version (Ceph/FeTA have biometry ``v1.0.0`` only) — ceiling resolution preserves that
    exactly. What it fixes is the case the old rule never anticipated: a plan published *above* the
    pin. Since v1.2.0 that is real (8 datasets ship v1.2.0 plans only), and "highest available"
    leaked them into 1.0.0/1.1.0/1.1.1 summaries. Verified: across the 22 pre-v1.2.0 datasets x 4
    pins x 3 families the two rules agree on every single case, so no historical summary moves.
    """
    files = find_plan_files(dataset_dir, plan_type)
    if version is not None:
        cap = tuple(int(p) for p in version.split("."))
        files = [f for f in files if plan_version_of(f) <= cap]
    if not files:
        return None
    return max(files, key=plan_version_of)


def dataset_exists_at(dataset_dir, version=None):
    """True if the dataset published ANY plan at or before ``version``.

    A dataset added in a later release has no plan at or before an earlier pin, so it did not exist
    then and must be skipped entirely rather than reported with zeros.
    """
    return any(resolve_plan_path(dataset_dir, pt, version)
               for pt in ("segmentation", "detection", "biometry"))


@functools.lru_cache(maxsize=2)
def load_benchmark_plan(dataset_dir, plan_type, version=None):
    """Load one benchmark plan dict, or ``None``. Resolution: see ``resolve_plan_path``.

    Cached (maxsize=2) so a dataset's plan is not re-read once per task; callers must treat the
    returned dict as read-only. Warns once per (dataset, type, version) when the resolved plan is
    not an exact version match, so a fallback is never silent.
    """
    path = resolve_plan_path(dataset_dir, plan_type, version)
    if path is None:
        return None
    if version is not None and not os.path.basename(path).endswith(f"_v{version}.json.gz"):
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
