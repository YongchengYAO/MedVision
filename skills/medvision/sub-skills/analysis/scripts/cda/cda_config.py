"""Clinical Decision Agreement (CDA) — clinical cutoff tables.

Self-contained configuration for the CDA analysis. Each proxy maps a geometric
measurement (an angle in degrees, or a tumor/lesion axis length in mm) to a
discrete clinical category using a standard, published cutoff table.

Each proxy spec is a dict with:
  - "name":        human-readable proxy name (for reports).
  - "cutoffs":     ascending numeric boundaries.
  - "labels":      category names; ``len(labels) == len(cutoffs) + 1``.
  - "ordinal":     True for ordered categories (e.g. T-stage) -> weighted kappa.
  - "right_closed": boundary handling used by ``cda_stats.categorize``, either one
        bool for every cutoff or a per-cutoff list.
        True  -> a value exactly on a cutoff falls in the *lower* category
                 (used for AJCC "<= x cm" style rules).
        False -> a value exactly on a cutoff falls in the *upper* category
                 (for ">= x is the higher category" style rules).
    The correct direction is a property of each published rule, not a global
    convention, so it is stated per proxy -- and, for a two-sided band, per
    cutoff. A one-sided ">= x" rule (AJCC) is uniformly lower-closed; a
    norm +/- k band (SNA, SNB) is the closed interval [norm-k, norm+k], whose
    lower edge must open upward while its upper edge stays closed.
    Exact-boundary hits are not hypothetical: 2 of the Ceph-Biometrics-400
    ground-truth angles sit exactly on a lower band edge (SNA 80.0, SNB 78.0),
    and models emit round numbers on the cutoffs too.
"""

# Output filenames (written into each model's parsed-source folder).
SUMMARY_FILENAME_CDA_METRICS = "summary_metrics_CDA_Task.json"
SUMMARY_FILENAME_CDA_VALUES = "summary_values_CDA_Task.json"

# --- Parsed-output sources ---------------------------------------------------
# A model folder can hold more than one set of parsed results. The regex parser
# writes "parsed/"; the LLM-judge re-parsing pass writes its own folder with the
# SAME filenames and row schema but the prediction under a DIFFERENT key.
#
# That key is why a source cannot be a plain path swap: reading an LLM-judge
# folder while still looking for "filtered_resps" finds the field absent on every
# row, scores every sample a parse failure, and produces a complete-looking
# report in which every model has n_parsed = 0. Selecting a source therefore
# selects its field.
#
# Matched by PREFIX, not by exact name. The judge pass writes one folder per
# judge model and per debug limit -- "llm-parsed_gemma-4-31b",
# "llm-parsed_gemma-4-31b-limit100", and one more for every future judge -- so
# the set of names grows without bound and cannot be enumerated here. What is
# stable is the prefix, because the prefix is what determines the row schema.
# Longest prefix wins; a name matching none is rejected.
#
# The keys below are PREFIXES, not folders to point --parsed_dirname at. The
# only LLM-judge folders that exist are gemma-4-31b's; pass the real folder name.
CDA_PARSED_SOURCE_PREFIXES = {
    "llm-parsed": "LLM_filtered_resps",
    "parsed": "filtered_resps",
}
CDA_DEFAULT_PARSED_DIRNAME = "parsed"

# The LLM-judge source currently on disk, for docs and CLI examples. Nothing
# resolves against it -- a new judge needs no change here, only a different
# --parsed_dirname (env CDA_PARSED_DIR), since the prefix does the matching.
CDA_LLM_PARSED_DIRNAME = "llm-parsed_gemma-4-31b"


def parsed_source_field(parsed_dirname):
    """Return the prediction field name for a parsed-source folder.

    Raises:
        ValueError: If the name matches no known prefix.
    """
    for prefix in sorted(CDA_PARSED_SOURCE_PREFIXES, key=len, reverse=True):
        if parsed_dirname == prefix or parsed_dirname.startswith(prefix):
            return CDA_PARSED_SOURCE_PREFIXES[prefix]
    raise ValueError(
        f"Unknown parsed source {parsed_dirname!r}: it starts with none of "
        f"{', '.join(sorted(CDA_PARSED_SOURCE_PREFIXES))}. Add its prefix to "
        "CDA_PARSED_SOURCE_PREFIXES with the field its rows carry the "
        "prediction in."
    )


def validate_parsed_dirname(value):
    """argparse ``type`` hook: pass a known source through, reject anything else.

    Prefix matching means the flag can no longer use argparse ``choices``, so the
    rejection has to happen here instead.
    """
    parsed_source_field(value)
    return value


def source_suffix(parsed_dirname):
    """Output-filename marker distinguishing one parsed source from another.

    The default source takes no marker, so filenames from a ``parsed`` run are
    unchanged. Any other source gets one -- ``llm-parsed_gemma-4-31b`` ->
    ``_llm-parsed-gemma-4-31b`` -- so two sources cannot overwrite each other's
    task-level reports. Per-model outputs need no marker: they are written inside
    the source folder itself.

    The marker keeps the folder name legible rather than squashing it to
    alphanumerics: with one folder per judge model, ``_llm-parsed-gemma-4-31b``
    and ``_llm-parsed-gemma-4-31b-limit100`` have to be told apart at a glance.
    ``_`` becomes ``-`` so the underscore keeps its role as the suffix separator.
    """
    if not parsed_dirname or parsed_dirname == CDA_DEFAULT_PARSED_DIRNAME:
        return ""
    cleaned = "".join(
        c if (c.isalnum() or c == "-") else "-" for c in parsed_dirname
    )
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return "_" + cleaned.strip("-")

# Resampling seed for the uncertainty analysis (clustered bootstrap CIs and
# the p-value inverted from them). Mirrors ``medvision_bm.utils.configs.SEED``; duplicated here so this
# folder stays self-contained.
CDA_SEED = 1024

# --- Proxy A: cephalometric angle -> maxillary/mandibular position (Ceph-Biometrics-400) ---
# Keyed by biometric_profile.metric_key. Norms/thresholds: Steiner CC.
# Am J Orthod 1953;39:729-755.
#
# IMPORTANT (folded-angle limitation): the benchmark defines its angle target as
# arccos(|A.B| / (||A|| ||B||)). The absolute value FOLDS every angle into
# [0, 90] -- the stored value is min(theta, 180 - theta), not |signed theta|.
# Verified: the max stored value over all 8 angle keys is exactly 90.0000, none
# above. Consequence: SNA/SNB above 90 deg reflect back below it (a true SNA of
# 94.4 stores as 85.6), which can move a subject across a band edge.
# See the doc's limitations section and DESIGN.md.
CDA_CEPH_ANGLE_PROXIES = {
    # SNA: maxillary A-P position, norm 82 deg +/- 2 -> normal = [80, 84].
    # Per-cutoff boundaries: the band's lower edge opens upward (80.0 is normal,
    # not retrusive), its upper edge stays closed (84.0 is normal, not protrusive).
    "A-L_1_2-L_2_5": {
        "name": "SNA maxillary position",
        "cutoffs": [80.0, 84.0],
        "labels": ["retrusive maxilla", "normal maxilla", "protrusive maxilla"],
        "ordinal": True,
        "right_closed": [False, True],
    },
    # SNB: mandibular A-P position, norm 80 deg +/- 2 -> normal = [78, 82].
    "A-L_1_2-L_2_6": {
        "name": "SNB mandibular position",
        "cutoffs": [78.0, 82.0],
        "labels": ["retrusive mandible", "normal mandible", "protrusive mandible"],
        "ordinal": True,
        "right_closed": [False, True],
    },
}

# --- Proxy B: renal tumor greatest dimension -> AJCC 8th-ed T category (KiTS23, KiPA22) ---
# T1a <= 4 cm; T1b > 4-7 cm; T2a > 7-10 cm; T2b > 10 cm (organ-confined).
# Ref: AJCC Cancer Staging Manual, 8th ed., 2017.
CDA_RENAL_TL_DATASETS = ("KiTS23", "KiPA22")
CDA_RENAL_TSTAGE = {
    "name": "AJCC renal T category (greatest dimension)",
    "cutoffs": [40.0, 70.0, 100.0],  # mm
    "labels": ["T1a", "T1b", "T2a", "T2b"],
    "ordinal": True,
    "right_closed": True,  # "<= x cm" -> boundary in lower category
}
