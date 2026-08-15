"""Clinical Decision Agreement (CDA) — uncertainty analysis.

Regenerates the **bootstrap 95% confidence intervals** and **p-values** reported
alongside the CDA point estimates in ``docs/clinical-decision-agreement.md``.

This script re-reads the per-sample categorizations already written by the two
analysis scripts (it never re-runs inference and never re-derives categories):

  - Track 1 (self-consistent): ``parsed/summary_values_CDA_Task.json`` — one
    record per sample with ``gt_category`` / ``pred_category``.
  - Track 2 (renal true-label): ``parsed/summary_CDA_renal_truelabel.json`` —
    the ``cases`` list, with ``true_stage`` / ``pred_cat`` per case.

Both procedures operate on the same aligned (reference, prediction) label pairs
that produce the point estimate, so the CI and the p-value describe exactly the
number they annotate.

**The resampling unit is the imaging volume, not the record.** A T/L proxy
scores one record per annotated 2D slice, and a single tumor contributes many
slices (KiTS23+KiPA22: 1,064 records from 121 volumes, up to 64 from one), so
records are correlated. Resampling records i.i.d. would treat a tumor measured
on 64 slices as 64 independent facts and understate the interval about
five-fold -- enough to move the renal CI from excluding zero to including it.
Both procedures therefore work on whole volumes. For the cephalometric proxies,
which contribute one record per subject, this is identical to record-level
resampling.

**Bootstrap CI** — the sampling uncertainty of the estimate. Resample whole
volumes *with replacement*, keeping each record's reference and prediction
together so the pairing is preserved, recompute the statistic on each resample,
and take the 2.5th and 97.5th percentiles of the resulting distribution (the
percentile method). Resamples in which the statistic is undefined -- kappa has
no chance-corrected value when the reference AND the prediction both collapse
onto the SAME single category (expected agreement 1); a prediction-only collapse
still gives a finite kappa of 0 -- are discarded and the count of usable
resamples is reported.

**p-value** — a one-sided test of kappa > 0, obtained by inverting the same
clustered bootstrap distribution: p = (#{replicate <= 0} + 1) / (n_valid + 1).
The +1 in numerator and denominator is the standard finite-sampling correction,
so p is never 0. One resampling pass therefore yields both the interval and the
p-value, and the two cannot disagree about whether zero is plausible.

This replaced a label-permutation test, which **cannot be made valid here**.
Under clustering the exchangeable unit is the volume, but volumes have unequal
sizes, so permuted prediction blocks cannot be re-paired against the references
without splitting clusters across reference boundaries — which removes
between-cluster variance from the null and makes the test anti-conservative.
Measured on a clustered null with reference and prediction independent by
construction (60 volumes, ~9 records each), the block-permutation test rejected
at 0.100 against a nominal 0.05; the bootstrap inversion rejected at 0.050.
Note the nulls differ: a permutation test asks about the sharp null of no
association, this asks whether the parameter is 0.

Because both come from one distribution, the interval and the p-value are two
views of a single calculation: p < 0.05 exactly when the one-sided interval
clears 0. Read the interval for the effect size and the p-value only as a
convenience.

Usage:
    # Track 1 — all self-consistent proxies, the config's canonical models
    python cda_uncertainty.py --task_dir Results/MedVision-AD-v2-CoT \\
        --config_yaml config-AD-CoT.yaml
    python cda_uncertainty.py --task_dir Results/MedVision-TL-v2-CoT \\
        --config_yaml config-TL-CoT.yaml

    # Track 2 — renal T-stage vs pathologic stage
    python cda_uncertainty.py --task_dir Results/MedVision-TL-v2-CoT \\
        --config_yaml config-TL-CoT.yaml --truelabel
"""

import argparse
import json
import os

import numpy as np

from cda_config import (
    CDA_CEPH_ANGLE_PROXIES,
    CDA_DEFAULT_PARSED_DIRNAME,
    CDA_RENAL_TSTAGE,
    CDA_SEED,
    SUMMARY_FILENAME_CDA_VALUES,
    source_suffix,
    validate_parsed_dirname,
)
from cda_stats import (
    cohen_kappa,
    load_model_display_map,
    resolve_model_dirs,
    weighted_kappa,
)

TRUELABEL_FILENAME = "summary_CDA_renal_truelabel.json"

# Rows below this many scored records are flagged "low_n". They are still
# emitted -- kappa is well defined here and suppressing it inside the stats layer
# would impose an arbitrary policy on every caller -- but a reader must not treat
# a 1-sample kappa as an estimate.
MIN_INFORMATIVE_N = 10

# proxy display name -> spec, so a values record can be matched back to its
# label set (the label order matters for weighted kappa).
PROXY_SPECS = {spec["name"]: spec for spec in CDA_CEPH_ANGLE_PROXIES.values()}
PROXY_SPECS[CDA_RENAL_TSTAGE["name"]] = CDA_RENAL_TSTAGE


def _stat_fn(ordinal):
    """Return the agreement statistic used for a proxy (weighted kappa if ordinal)."""
    return weighted_kappa if ordinal else cohen_kappa


def _cluster_index(groups, n):
    """Return a list of index arrays, one per cluster, in first-appearance order.

    ``groups`` is a per-record cluster id (the imaging volume a record came from).
    Records with no id become singleton clusters, which reduces to i.i.d.
    resampling for that record.
    """
    if groups is None:
        return [np.array([i]) for i in range(n)]
    order, members = [], {}
    for i, g in enumerate(groups):
        key = g if g is not None else ("__singleton__", i)
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(i)
    return [np.asarray(members[k]) for k in order]


def bootstrap_ci(gt, pred, labels, ordinal, n_boot, seed, groups=None, alpha=0.05):
    """Percentile bootstrap CI **and** one-sided p-value for kappa over (gt, pred).

    Returns ``(lo, hi, p, observed, n_valid)``. Both the interval and the p-value
    come from a single cluster-resampled bootstrap distribution:

      - interval: the ``alpha/2`` and ``1 - alpha/2`` percentiles;
      - p-value:  ``(#{replicate <= 0} + 1) / (n_valid + 1)``, a one-sided test of
        kappa > 0 by inverting that distribution, with the standard
        finite-sampling correction so p is never 0.

    A label-permutation test was used here previously and was **anti-conservative
    under clustering**: with unequal cluster sizes there is no way to re-pair
    permuted prediction blocks against the references without splitting clusters
    across reference boundaries, which strips between-cluster variance out of the
    null. Measured on a clustered null (60 volumes, ~9 records each, reference
    and prediction independent by construction), that test rejected at 0.100
    against a nominal 0.05, while the inversion below rejected at 0.050.

    Resamples whole CLUSTERS with replacement, keeping each record's reference
    and prediction together. The cluster is the imaging volume: a T/L proxy has
    many 2D slices per tumor, and those slices are not independent observations,
    so resampling them i.i.d. would treat one tumor measured on 64 slices as 64
    independent facts and understate the interval several-fold. One record per
    volume (as for the cephalometric proxies) makes this identical to the plain
    i.i.d. bootstrap.

    Resamples where kappa is undefined (no class variance) are dropped;
    ``n_valid`` reports how many contributed.
    """
    stat = _stat_fn(ordinal)
    gt = np.asarray(gt, dtype=object)
    pred = np.asarray(pred, dtype=object)
    n = len(gt)
    nan = float("nan")
    if n == 0:
        return nan, nan, nan, nan, 0
    observed = stat(gt, pred, labels)
    clusters = _cluster_index(groups, n)
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([clusters[k] for k in picked])
        s = stat(gt[idx], pred[idx], labels)
        if np.isfinite(s):
            vals.append(s)
    if not vals:
        return nan, nan, nan, float(observed), 0
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = (sum(1 for v in vals if v <= 0.0) + 1) / (len(vals) + 1)
    return float(lo), float(hi), float(p), float(observed), len(vals)


def _pairs_selfconsistent(values_json):
    """Yield (proxy_name, spec, gt_labels, pred_labels, groups) from a Track-1 values file.

    ``groups`` is the imaging volume each record came from (``image_file``), the
    resampling cluster. A T/L proxy contributes many 2D slices per tumor, so its
    records are not independent; the cephalometric proxies contribute one record
    per subject, for which clustering is a no-op.
    """
    with open(values_json, "r") as f:
        records = json.load(f)
    by_proxy = {}
    for rec in records:
        g, p = rec.get("gt_category"), rec.get("pred_category")
        if g is None or p is None:
            continue  # unparsed prediction: excluded from kappa, as in the point estimate
        gt, pred, groups = by_proxy.setdefault(rec["proxy"], ([], [], []))
        gt.append(g)
        pred.append(p)
        groups.append(rec.get("image_file"))
    for name, (gt, pred, groups) in by_proxy.items():
        spec = PROXY_SPECS.get(name)
        if spec is None:
            continue
        yield name, spec, gt, pred, groups


def _pairs_truelabel(truelabel_json):
    """Yield (stratum_name, spec, true_stages, pred_cats, groups) from a Track-2 file.

    Track 2 is already aggregated to one record per case, so ``groups`` is the
    case id and every cluster is a singleton.
    """
    with open(truelabel_json, "r") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    organ_confined = {"T1a", "T1b", "T2a", "T2b"}
    strata = {
        "renal T-stage vs pathologic (full 6-class)": cases,
        "renal T-stage vs pathologic (organ-confined pT1-pT2)": [
            c for c in cases if c.get("true_stage") in organ_confined
        ],
    }
    for name, subset in strata.items():
        scored = [c for c in subset if c.get("pred_cat") is not None]
        gt = [c["true_stage"] for c in scored]
        pred = [c["pred_cat"] for c in scored]
        groups = [c.get("case_id") for c in scored]
        # full 6-class needs the pT3/pT4 labels appended to the ordinal scale
        labels = list(CDA_RENAL_TSTAGE["labels"])
        if "full 6-class" in name:
            labels = labels + ["T3", "T4"]
        spec = {"labels": labels, "ordinal": True}
        yield name, spec, gt, pred, groups


def main():
    parser = argparse.ArgumentParser(
        description="Clustered bootstrap CIs and p-values for CDA agreement statistics."
    )
    parser.add_argument("--task_dir", required=True, help="Results/<experiment> directory.")
    parser.add_argument(
        "--config_yaml",
        default=None,
        help="CDA config listing the models to analyze: config-AD-CoT.yaml for an "
        "A/D task dir, config-TL-CoT.yaml for a T/L one.",
    )
    parser.add_argument(
        "--truelabel",
        action="store_true",
        help="Analyze the renal true-label track instead of the self-consistent track.",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Read the '_filtered' inputs written by a --removed_samples_dir run of "
        "the two analysis scripts, and write a '_filtered' output. This script does "
        "no filtering itself; it only needs to know which files to read.",
    )
    parser.add_argument(
        "--parsed_dirname",
        default=CDA_DEFAULT_PARSED_DIRNAME,
        type=validate_parsed_dirname,
        help="Which parsed-results folder to read the per-sample categorisations "
        "from. Must match the --parsed_dirname the analysis scripts ran with; the "
        "output filename gains a matching source marker.",
    )
    parser.add_argument("--n_boot", type=int, default=4000, help="Bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=CDA_SEED, help="Resampling seed.")
    args = parser.parse_args()

    display_map = load_model_display_map(args.config_yaml)
    model_items = resolve_model_dirs(args.task_dir, display_map)

    fmark = "_filtered" if args.filtered else ""
    base = TRUELABEL_FILENAME if args.truelabel else SUMMARY_FILENAME_CDA_VALUES
    filename = f"{base.removesuffix('.json')}{fmark}.json"
    pair_fn = _pairs_truelabel if args.truelabel else _pairs_selfconsistent

    rows = []
    without_input = []
    for model_dir, model in model_items:
        path = os.path.join(model_dir, args.parsed_dirname, filename)
        if not os.path.isfile(path):
            without_input.append(model)
            continue
        for name, spec, gt, pred, groups in pair_fn(path):
            ordinal = spec.get("ordinal", False)
            labels = spec["labels"]
            lo, hi, p, observed, n_boot_ok = bootstrap_ci(
                gt, pred, labels, ordinal, args.n_boot, args.seed, groups=groups
            )
            # Count the clusters the resampler actually uses, so a record with no
            # image_file (a singleton cluster) is reflected here rather than
            # silently collapsing into a smaller reported count.
            n_clusters = len(_cluster_index(groups, len(gt)))
            rows.append({
                "model": model,
                "proxy": name,
                "statistic": "weighted_kappa" if ordinal else "cohen_kappa",
                "n": len(gt),
                "n_clusters": n_clusters,
                "point_estimate": observed,
                "ci_lower": lo,
                "ci_upper": hi,
                "p_kappa_gt_zero": p,
                # A row can be arithmetically valid and informationally empty:
                # with one disagreeing pair kappa is exactly 0, the CI has zero
                # width and p is 1. Flag it rather than suppress it, so the
                # number stays reproducible but is not read as an estimate.
                "low_n": len(gt) < MIN_INFORMATIVE_N,
                "n_boot_valid": n_boot_ok,
            })

    # This script only re-reads what the two analysis scripts persist; with no
    # input at all the output would be an empty-but-successful report, which is
    # indistinguishable from "the models genuinely have no eligible samples".
    if not rows:
        producer = (
            "analyze_CDA_renal_truelabel.py" if args.truelabel else "summarize_CDA_task.py"
        )
        raise FileNotFoundError(
            f"No '{filename}' found under any model's {args.parsed_dirname}/ in "
            f"{args.task_dir}. Run {producer} on this task directory first, with "
            f"the same --parsed_dirname."
        )
    if without_input:
        print(
            f"[warn] {len(without_input)} model(s) had no '{filename}' and were "
            f"excluded: {', '.join(without_input)}"
        )

    suffix = (
        ("_truelabel" if args.truelabel else "")
        + source_suffix(args.parsed_dirname)
        + fmark
    )
    out_json = os.path.join(args.task_dir, f"summary_CDA_uncertainty{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "n_boot": args.n_boot,
                "seed": args.seed,
                "ci_method": (
                    "percentile bootstrap; paired, resampling whole imaging volumes "
                    "(clusters) with replacement, not individual records"
                ),
                "p_method": (
                    "one-sided test of kappa > 0 by inverting the same clustered "
                    "bootstrap: (#{replicate <= 0} + 1) / (n_boot_valid + 1)"
                ),
                "cluster_note": (
                    "n counts scored records; n_clusters counts the independent "
                    "imaging volumes they came from. Where n_clusters < n the "
                    "records are correlated and n overstates the information."
                ),
                "rows": rows,
            },
            f,
            indent=2,
        )

    header = (
        f"{'Model':<26} {'Proxy':<52} {'n':>5} {'vols':>5} {'est':>8} "
        f"{'95% CI':>20} {'p(k>0)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
        print(
            f"{r['model']:<26} {r['proxy']:<52} {r['n']:>5} {r['n_clusters']:>5} "
            f"{r['point_estimate']:>+8.4f} {ci:>20} {r['p_kappa_gt_zero']:>9.4f}"
            f"{'  <- low n' if r['low_n'] else ''}"
        )
    print(
        "\nn = scored records; vols = independent imaging volumes they came from. "
        "CIs resample volumes,\nso where vols < n the interval reflects the smaller "
        "effective sample size."
    )
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
