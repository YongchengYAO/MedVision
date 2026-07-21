#!/usr/bin/env bash
set -euo pipefail

# Summarize the local Data/Datasets collection from the benchmark plans (no HF; nibabel only if
# the live labels_map lookup via medvision_ds succeeds).
# Writes dataset_files.jsonl, dataset_summary_{filtered,raw}.json, dataset_summary.csv, and
# dataset_label_stats.csv.

# Resolve the repo root from this script's location (<repo>/script/misc/<this>.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Paths are overridable via the environment; defaults assume the standard layout.
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/Data}"
# Dataset (biometry) version: selects which T/L plan is counted. Detection/segmentation ship v1.0.0
# only, so only T/L differs across versions. Output dir is version-suffixed.
PLAN_VERSION="${PLAN_VERSION:-1.1.1}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dataset-info/datasets_summary_v${PLAN_VERSION}}"
# Optional: reuse version-invariant Box/segmentation/inventory from this existing summary dir and
# recompute only biometry — skips the multi-GB detection scan (fast version regen).
REUSE_FROM="${REUSE_FROM:-}"

ARGS=(--data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" --plan_version "${PLAN_VERSION}")
[ -n "${REUSE_FROM}" ] && ARGS+=(--reuse_from "${REUSE_FROM}")

python -m medvision_bm.utils.summarize_datasets "${ARGS[@]}" "$@"

# The scan always writes dataset_summary_{filtered,raw}.json (filtered = v1.0.0-filtered
# benchmark counts; raw = the same 3 tasks counted unfiltered).
#
# --- Variations (uncomment / adapt as needed) ---
# Subset:   --datasets KiTS23,Ceph-Biometrics-400
# Pin ds:   --plan_version 1.1.1   (biometry only; measurement sample set differs by version)
# Fast:     --no_detection  skip the large detection plans (drops BoxSize; ~8.5 min vs ~30 min)
# Figures:  --viz    also render dataset_summary.pdf (bar panels), dataset_summary_wordcloud.pdf
#                    (needs `pip install wordcloud` -- see the note on figure 2 below),
#                    and the donut in filtered+raw x {2x1, 1x2, compact}:
#                    dataset_summary_rings_{filtered,raw}_{2x1,1x2,compact}.pdf
#                    (2x1/1x2 = the magnified small-dataset panel stacked / side-by-side)
# Fig only: --viz_only  skip the scan; render all figures from the existing dataset_summary_filtered.json
#
#
# =============================== Output figures (--viz) ===============================
# Rendered by render_figures() in medvision_bm/utils/summarize_datasets.py. All are transparent
# vector PDFs written through save_fig_capped (project figure convention).
#
# --- 1. dataset_summary.pdf -- 5 horizontal bar panels, 2 + 3 (collection-level "__all__") ---
# Row 1 holds the 2 modality panels at half width; row 2 holds the 3 anatomy panels at one-third
# width. Log-scaled x-axis with exact value labels (counts span ~3 orders of magnitude).
# Subfigures, in reading order:
#   (1) row1-left   "# 3D Images by Modality"   <- images_by_modality
#                   One count per 3D volume. Fixed modality order/colour across (1) and (2).
#   (2) row1-right  "# 2D Slices by Modality"   <- 2D-slices_by_modality
#                   2D slices SUMMED over all three planes (x/y/z), so one volume is counted
#                   three times -- once per slicing direction.
#   (3) row2-left   "# 3D Images by Anatomy"    <- volumes_by_anatomy
#                   A volume counts once per anatomy group it contains (a whole-abdomen CT adds
#                   to Liver, Kidney, Spleen, ... simultaneously). Because a volume is counted in
#                   every group it touches, the bars SUM TO MORE than the unique-image count
#                   (99,008 vs 29,031 at ds v1.0.0 = 3.41 groups per volume on average); read the
#                   bars individually, never as a partition.
#   (4) row2-mid    "# Single-instance Annotations per Anatomy"  <- boxsize_by_anatomy
#                   The v1.0.0-FILTERED benchmark BoxSize count (24,236,327) -- i.e. the samples
#                   the loader actually emits. A (case, slice, label) item is kept only when it
#                   forms exactly ONE connected component whose box is >= 10 px in both
#                   dimensions, so multi-component and tiny structures are dropped outright.
#   (5) row2-right  "# Multi-instance Annotations per Anatomy"   <- boxsize_by_anatomy_raw
#                   The UNFILTERED count (45,274,250): every (case, slice, label) item carrying
#                   ONE OR MORE instances, with no cluster-count and no size filter.
#
#                   >>> "MULTI-INSTANCE" IS INCLUSIVE: it means ">= 1 instance", so panel (5) is
#                   >>> a strict SUPERSET of panel (4), NOT its complement. The panels are nested
#                   >>> by design -- (4) is 53.5% of (5) (24,236,327 of 45,274,250), contained
#                   >>> entirely within it, verified with 0 containment violations across all 36
#                   >>> anatomy groups and all 22 datasets. They do NOT partition the total, so
#                   >>> never add them or read them as disjoint classes.
#
#                   Two further properties of (5) worth knowing:
#                     * It is a per-(slice,label) count, not an instance count: a label split into
#                       N connected components on one slice contributes 1, not N. "Multi-instance"
#                       describes which slices are ADMITTED, not how components are tallied.
#                     * (5) minus (4) is not purely multi-component: the panel-(4) filter is a
#                       conjunction (one cluster AND >= 10 px), so a single-instance box that is
#                       merely too small also sits only in (5). The shipped JSONs do not record
#                       the two rejection causes separately, so that 21,037,923 delta cannot be
#                       attributed to multiplicity alone.
#                   Panel (5) reads boxsize_by_anatomy_raw (DETECTION plan), deliberately not the
#                   seg-derived 2D-slices_by_anatomy: the two are byte-identical on the current
#                   data but that is a coincidence, and _bench_anatomy() carries the same guard
#                   against silently mixing seg-annotation counts into benchmark units.
#   Row order in (3)/(4)/(5): each panel sorts by its OWN values, DESCENDING (largest at top), so
#   the anatomy order differs between the three panels -- read each panel's own y labels; a given
#   row index is NOT the same anatomy across panels.
#   Caveats that apply to (4) and (5) alike:
#     * MULTI-label per slice: liver + spleen + pancreas on one slice adds 3.
#     * A coarse group can gain >1 from a single slice when several fine labels map into it
#       (e.g. left kidney + right kidney -> Kidney; Rib aggregates 25 fine labels, Spleen 1) --
#       so bar height is NOT comparable across groups of differing label granularity.
#     * Summed over the 3 planes, like panel (2).
#   All three anatomy panels drop the "UNMAPPED" group (_ANATOMY_EXCLUDE; currently empty).
#   Under --no_detection both detection keys are empty, so panels (4) AND (5) render as all-zero
#   columns -- honest (no detection scan => no box annotations) rather than silently substituting
#   the segmentation-derived count.
#
# --- 2. dataset_summary_wordcloud.pdf -- single panel, no subfigures ---
# Anatomy label names tokenized (split on whitespace and '/', lowercased, articles/sides/ordinals
# dropped). Every label of a dataset is weighted EQUALLY by that dataset's test-set size,
# ceil(n_images_test / 10) -- so word size reflects how many test cases across datasets use the
# word, NOT its per-slice prevalence. Datasets without segmentation labels (e.g.
# Ceph-Biometrics-400) contribute nothing.
#
# REQUIRES `pip install wordcloud` -- it is NOT declared in pyproject.toml (unlike matplotlib /
# scipy / nibabel, which are), so `pip install .` does NOT pull it in. Without it viz_wordcloud()
# only prints "[warn] wordcloud package not installed; skipping" and returns. Note the run still
# LISTS dataset_summary_wordcloud.pdf among its outputs in that case, so an older copy of the file
# left in --out_dir will look freshly regenerated when it was not; check the mtime after a run.
#
# --- 3. dataset_summary_rings_{filtered,raw}_{2x1,1x2,compact}.pdf -- 2-ring donuts (6 files) ---
# variant: filtered = benchmark counts after the v1.0.0 loader filters, legend titled
#          "# Single-instance Annotations"; raw = the same tasks unfiltered, legend titled
#          "# Multi-instance Annotations". Same geometry, different counts.
# layout:  2x1 = two donuts stacked; 1x2 = side by side; compact = main donut only (no zoom panel).
#          The 2x1/1x2 files also emit .svg and _whitebg.svg twins (README / webpage embedding;
#          GitHub cannot render PDF, and the white-background twin serves GitHub dark mode).
# Subfigures:
#   (A) main donut -- inner ring: one wedge per dataset, angle proportional to that dataset's
#       benchmark annotation count, coloured so the palette sweeps the ring in count-desc order
#       (biggest wedge = first colour; the bottom legend follows the same order).
#       Outer ring: the SAME dataset's BoxSize annotations split by anatomy, each wedge a tint of
#       the dataset's own hue ordered by count (darkest = most). Centre reports the three
#       collection totals (3D images / 2D slices / annotations). Datasets too small to label are
#       enclosed by a dotted red arc captioned "small datasets".
#   (B) enlarged donut (2x1 and 1x2 only) -- those same small datasets re-drawn at their true
#       relative proportions, zoomed to fill the circle. Drawn as an INCOMPLETE ring (a ~10 degree
#       opening at the bottom) to signal it is a zoom of the dotted arc in (A), not a whole
#       collection. `compact` omits this panel entirely.
# Bottom legend (all layouts): dataset colour swatches with per-dataset counts; its title states
# the instance semantics of the variant.
# ======================================================================================
