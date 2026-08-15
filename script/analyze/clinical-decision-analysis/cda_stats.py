"""Clinical Decision Agreement (CDA) — statistics and small I/O helpers.

Self-contained (numpy-only) implementations of the categorization + agreement
statistics used by the CDA analysis, plus a few filesystem/serialization
helpers inlined so the ``clinical-decision-analysis`` folder does not depend on
the ``medvision_bm`` package.

The kappa implementations are validated to match
``sklearn.metrics.cohen_kappa_score`` (plain and ``weights="quadratic"``) to
machine precision, but sklearn is intentionally NOT imported (it is not a
declared dependency of this repo).

Because sklearn may not be installed, the kappas are also checkable against
hand-computed references that need nothing but this module:

  - plain: M[gt, pred] = [[20, 5], [10, 15]] (n=50) -> po=0.70, pe=0.50, kappa=0.40
  - quadratic-weighted: M = [[10, 2, 0], [3, 10, 3], [0, 2, 10]] (n=40, k=3)
    with weights ((i-j)/(k-1))^2 -> kappa=0.80

Note that ``labels`` ORDER defines the ordinal distance for
:func:`weighted_kappa`: scrambling the 3-class order above to A, C, B changes
the result from 0.80 to 0.54. Always pass the order from the proxy's config.
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Filesystem / serialization helpers (inlined for self-containment)
# ---------------------------------------------------------------------------


def get_subfolders(task_dir):
    """Return the paths of all immediate subdirectories, sorted.

    Sorted for reproducibility: ``os.scandir`` yields raw directory order, which
    is not sorted and can change if the directory is rewritten, copied or moved
    to another filesystem. See :func:`sorted_glob` for why that would silently
    perturb the bootstrap.
    """
    return sorted(entry.path for entry in os.scandir(task_dir) if entry.is_dir())


def sorted_glob(pattern):
    """``glob.glob`` with a deterministic order.

    ``glob`` returns raw ``readdir`` order, not sorted. That order is stable
    while a directory is untouched, but changes when it is rewritten, copied or
    moved — and CDA's output depends on it in a way that is easy to miss:
    ``cda_uncertainty._cluster_index`` builds its cluster list in *first
    appearance* order, so a permuted file order permutes the clusters, and the
    seeded ``rng.integers(0, n_clusters, n_clusters)`` then draws a different set
    of volumes. Same seed, different confidence interval. Sorting makes the
    resampling reproducible across machines and after a filesystem move, not just
    within one untouched checkout.
    """
    import glob as _glob

    return sorted(_glob.glob(pattern))


# ---------------------------------------------------------------------------
# Removed-samples filtering (mirrors summarize_TL_task.py)
# ---------------------------------------------------------------------------
# The T/L benchmark excludes slices whose mask has more than one connected
# component -- which is why its canonical report is summary_TL_task_filtered.txt.
# Without the same exclusion CDA scores 1,064 renal samples where the benchmark
# it proxies scores 1,025, so the two are not like-for-like.

_REMOVED_DIM_MAP = {"x": 0, "y": 1, "z": 2}


def build_removed_set(json_path):
    """Load a removed-samples JSON and return a frozenset of exclusion keys.

    Each key is ``(relative_image_file, slice_dim_int, slice_idx, task_id)``.
    """
    import json  # local: keeps this module's top-level imports to os + numpy

    with open(json_path) as f:
        entries = json.load(f)
    return frozenset(
        (
            e["image_file"],
            _REMOVED_DIM_MAP[e["slice_dim"]],
            int(e["slice_idx"]),
            int(e["task_ID"]),
        )
        for e in entries
    )


def load_removed_set(removed_samples_dir, dataset_name, removed_samples_filename):
    """Return the exclusion set for one dataset, or None when it ships no file.

    A dataset with no removed-samples file is simply unfiltered -- the same
    fallback ``summarize_TL_task.py`` uses. That is what makes passing the flag
    to an A/D run harmless: those datasets ship no such file.
    """
    if not removed_samples_dir or not dataset_name:
        return None
    json_path = os.path.join(
        removed_samples_dir, dataset_name, removed_samples_filename
    )
    return build_removed_set(json_path) if os.path.exists(json_path) else None


def removed_key(doc, dataset_name):
    """Build the exclusion key for one parsed record, or None if it lacks fields.

    The removed-samples JSON keys ``image_file`` relative to the dataset folder,
    so the absolute path in the parsed doc is trimmed. Note the task id lives in
    ``doc["taskID"]`` -- NOT ``task_ID``, which is the spelling used inside the
    removed-samples JSON itself; mixing the two silently matches nothing.
    """
    img = doc.get("image_file") or ""
    task_id = doc.get("taskID")
    if task_id is None:
        return None
    marker = f"/{dataset_name}/"
    idx = img.find(marker)
    rel_img = img[idx + len(marker):] if idx >= 0 else os.path.basename(img)
    return (rel_img, doc.get("slice_dim"), doc.get("slice_idx"), int(task_id))


def filtered_suffix(removed_samples_dir):
    """Output-filename suffix marking a run as removed-samples-filtered.

    Its sibling ``source_suffix`` (which marks the parsed-source folder) lives in
    ``cda_config.py``, beside the source table it depends on.
    """
    return "_filtered" if removed_samples_dir else ""


def convert_numpy_to_python(obj):
    """Recursively convert NumPy values to native Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


def load_model_display_map(config_yaml):
    """Load an ordered ``{results_folder_name: "Display Name"}`` map from a config.

    The YAML has a single top-level key ``model_display_name`` (same schema as the
    visualization configs). Each CDA config is task-specific -- ``config-AD-CoT.yaml``
    names the folders under an A/D results directory, ``config-TL-CoT.yaml`` those
    under a T/L one -- because a model's canonical folder can differ between tasks.
    Pass the config matching the task directory being analyzed.

    Args:
        config_yaml: Path to the config, or None/empty to disable the restriction.

    Returns:
        dict | None: Ordered folder -> display name, or None when ``config_yaml``
        is falsy.

    Raises:
        ValueError: If the config does not define a non-empty ``model_display_name``.

    ``yaml`` is imported lazily so the scripts run without PyYAML unless a config
    is actually requested.
    """
    if not config_yaml:
        return None
    import yaml  # lazy import: only needed when --config_yaml is passed

    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    mapping = (cfg or {}).get("model_display_name")
    if not mapping:
        raise ValueError(
            f"{config_yaml}: config must define a non-empty 'model_display_name' map."
        )
    return dict(mapping)  # insertion order preserved (Py3.7+)


def resolve_model_dirs(task_dir, display_map):
    """Return an ordered list of ``(model_dir, display_name)`` for a task directory.

    With a display map the config's folders are used, in config order. Without one
    every subdirectory is used, with its basename as the display name.

    A config-listed folder that is missing from ``task_dir`` is a hard error: it
    means the config is stale, and skipping it would silently drop a model from
    the report. Drop ``--config_yaml`` to analyze whatever is on disk instead.

    Raises:
        FileNotFoundError: If any config-listed folder is absent from ``task_dir``.
    """
    if display_map is None:
        return [(md, os.path.basename(md)) for md in get_subfolders(task_dir)]

    missing = [f for f in display_map if not os.path.isdir(os.path.join(task_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} config-listed model folder(s) not found in {task_dir}:\n  "
            + "\n  ".join(missing)
            + "\nEither the config is stale, or it is the wrong one for this task "
            "directory: config-AD-CoT.yaml names A/D folders and config-TL-CoT.yaml "
            "names T/L folders, and a model's folder can differ between the two."
        )
    return [
        (os.path.join(task_dir, folder), display)
        for folder, display in display_map.items()
    ]


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


def categorize(value, cutoffs, labels, right_closed=True):
    """Map a continuous value to a discrete clinical category via ascending cutoffs.

    Args:
        value: Numeric value, or None. None / non-finite values return None.
        cutoffs: Ascending list of numeric boundaries.
        labels: Category names; must have ``len(cutoffs) + 1`` entries.
        right_closed: Boundary handling for a value exactly equal to a cutoff,
            either one bool for every cutoff or a per-cutoff list.
            ``True``  -> value falls in the LOWER category (advance only when
            ``value > cutoff``; matches AJCC "<= x cm" and ANB "<= 4 deg" rules).
            ``False`` -> value falls in the UPPER category (advance when
            ``value >= cutoff``; for ">= x is the higher category" rules).
            A per-cutoff list is needed for a two-sided band such as Steiner's
            "82 +/- 2": the normal band is the CLOSED interval [80, 84], so its
            lower edge must open upward (False) while its upper edge stays closed
            (True). A single bool cannot express that and puts a value of exactly
            80 in the retrusive category.

    Returns:
        str | None: The category label, or None when ``value`` is None / non-finite.

    Raises:
        ValueError: If a ``right_closed`` list does not have one entry per cutoff.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    if isinstance(right_closed, bool):
        closed = [right_closed] * len(cutoffs)
    else:
        closed = list(right_closed)
        if len(closed) != len(cutoffs):
            raise ValueError(
                f"right_closed has {len(closed)} entries for {len(cutoffs)} cutoffs; "
                "pass a single bool or one bool per cutoff."
            )
    idx = 0
    for c, rc in zip(cutoffs, closed):
        advance = (v > c) if rc else (v >= c)
        if advance:
            idx += 1
        else:
            break
    return labels[idx]


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def _confusion_matrix(gt_cats, pred_cats, labels):
    """Return a ``len(labels) x len(labels)`` count matrix ``M[gt, pred]``.

    Pairs whose GT or prediction label is not in ``labels`` are ignored.
    """
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    mat = np.zeros((n, n), dtype=np.int64)
    for g, p in zip(gt_cats, pred_cats):
        if g in index and p in index:
            mat[index[g], index[p]] += 1
    return mat


def cohen_kappa(gt_cats, pred_cats, labels):
    """Cohen's kappa for two aligned lists of categorical labels.

    Returns ``np.nan`` when there are no samples, and also when kappa is undefined
    because there is no class variance (expected agreement == 1, i.e. both raters
    put every sample in a single class) — matching ``sklearn``'s behaviour.
    """
    mat = _confusion_matrix(gt_cats, pred_cats, labels).astype(np.float64)
    n = mat.sum()
    if n == 0:
        return float("nan")
    po = np.trace(mat) / n
    row = mat.sum(axis=1) / n
    col = mat.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if pe >= 1.0:
        return float("nan")  # undefined: no class variance (cf. sklearn)
    return float((po - pe) / (1.0 - pe))


def weighted_kappa(gt_cats, pred_cats, labels):
    """Quadratic-weighted Cohen's kappa for ORDERED categorical labels.

    Penalizes disagreements by the squared ordinal distance between categories,
    so (e.g.) T1a-vs-T2b counts as a worse error than T1a-vs-T1b. ``labels`` must
    be given in ordinal order. Returns ``np.nan`` when there are no samples, or
    when kappa is undefined because there is no class variance (single occupied
    category) — matching ``sklearn``'s behaviour.
    """
    k = len(labels)
    mat = _confusion_matrix(gt_cats, pred_cats, labels).astype(np.float64)
    n = mat.sum()
    if n == 0 or k < 2:
        return float("nan")
    idx = np.arange(k)
    weights = ((idx[:, None] - idx[None, :]) / (k - 1)) ** 2
    row = mat.sum(axis=1)
    col = mat.sum(axis=0)
    expected = np.outer(row, col) / n
    denom = float(np.sum(weights * expected))
    if denom == 0:
        return float("nan")  # undefined: no class variance (cf. sklearn)
    return float(1.0 - np.sum(weights * mat) / denom)


def cal_clinical_agreement(gt_cats, pred_cats, labels, ordinal=False, n_total=None):
    """Compute Clinical Decision Agreement statistics between two categorizations.

    Compares categories derived from the ground-truth measurement against those
    derived from the model's predicted measurement.

    Args:
        gt_cats, pred_cats: Aligned lists of category labels for samples whose
            prediction was parseable (both entries non-None).
        labels: Ordered list of category names for the proxy.
        ordinal: If True, also compute quadratic-weighted kappa.
        n_total: Total sample count INCLUDING prediction-parse failures, used for
            coverage-adjusted agreement. Defaults to ``len(gt_cats)``.

    Returns:
        dict: ``accuracy``, ``cohen_kappa``, ``confusion`` (nested
        ``gt -> pred -> count``), ``flip_rate`` (= 1 - accuracy over parsed
        samples), ``n_parsed``, ``n_total``, and coverage-adjusted
        ``accuracy_coverage_adjusted`` / ``flip_rate_coverage_adjusted`` (parse
        failures counted as disagreement). Includes ``weighted_kappa`` when
        ``ordinal`` is True.
    """
    n_parsed = len(gt_cats)
    if n_total is None:
        n_total = n_parsed
    agree = sum(1 for g, p in zip(gt_cats, pred_cats) if g == p)
    accuracy = agree / n_parsed if n_parsed > 0 else float("nan")
    mat = _confusion_matrix(gt_cats, pred_cats, labels)
    confusion = {
        g: {p: int(mat[i, j]) for j, p in enumerate(labels)}
        for i, g in enumerate(labels)
    }
    out = {
        "accuracy": accuracy,
        "cohen_kappa": cohen_kappa(gt_cats, pred_cats, labels),
        "confusion": confusion,
        "flip_rate": (1.0 - accuracy) if n_parsed > 0 else float("nan"),
        "n_parsed": n_parsed,
        "n_total": n_total,
        "accuracy_coverage_adjusted": (
            agree / n_total if n_total > 0 else float("nan")
        ),
        "flip_rate_coverage_adjusted": (
            1.0 - agree / n_total if n_total > 0 else float("nan")
        ),
    }
    if ordinal:
        out["weighted_kappa"] = weighted_kappa(gt_cats, pred_cats, labels)
    return out
