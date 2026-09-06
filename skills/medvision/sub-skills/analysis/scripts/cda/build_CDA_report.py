#!/usr/bin/env python3
"""Render the final Clinical Decision Agreement (CDA) report as one Markdown file.

Bundled copy of the MedVision repository's CDA renderer (adapted for the
``medvision`` skill: the repository-root path used to shorten paths in the
provenance table is now the optional ``--repo_root`` argument, and the
"regenerate" footer names the bundled ``run_cda.sh`` runner).

Every other CDA output lands in the ``Results/`` tree, scattered across one
directory per model. This script collects the leaderboards into a single
Markdown file (``--out``).

It reads only the JSON artifacts the analysis persists -- never the fixed-width
``.txt`` reports, whose columns are truncated for display (model names to 26
chars, proxy labels to 56) and would round-trip lossily. Nothing is recomputed
here: this is a renderer, so a number in the report is traceable to exactly one
JSON field.

Deliberately emits no timestamp. An unchanged analysis then regenerates a
byte-identical file, so ``git diff`` shows real numeric movement rather than a
churning date line.

Run after summarize_CDA_task.py and cda_uncertainty.py -- the bundled
``run_cda.sh`` does exactly that.
"""

import argparse
import json
import os

from cda_config import (
    CDA_CEPH_ANGLE_PROXIES,
    CDA_DEFAULT_PARSED_DIRNAME,
    CDA_RENAL_TSTAGE,
    SUMMARY_FILENAME_CDA_METRICS,
    parsed_source_field,
    source_suffix,
    validate_parsed_dirname,
)
from cda_stats import load_model_display_map, resolve_model_dirs

# Proxy order follows cda_config's declaration order -- the two cephalometric
# proxies, then the renal proxy. A proxy not listed here falls back to
# first-appearance order.
PROXY_ORDER = [spec["name"] for spec in CDA_CEPH_ANGLE_PROXIES.values()] + [
    CDA_RENAL_TSTAGE["name"]
]

# Root against which paths in the provenance table are shortened. Defaults to
# the current working directory; override with --repo_root (e.g. the benchmark
# directory that holds Results/). Paths outside it are printed as given.
REPO_ROOT = os.getcwd()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _num(x, prec=3):
    """Format a float for a Markdown cell; None/NaN render as an em dash."""
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.{prec}f}"


def _pval(p):
    """Format a bootstrap p-value, floored at the resolution the resamples allow."""
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _ci(row):
    if row is None:
        return "—"
    return f"[{_num(row.get('ci_lower'))}, {_num(row.get('ci_upper'))}]"


def _table(header, rows):
    """Render a Markdown table. ``rows`` is a list of pre-formatted cell lists."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return lines


def _relpath(path):
    """Repo-relative path, so the report is not machine-specific."""
    try:
        rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    except ValueError:  # different drive (Windows)
        return path
    return path if rel.startswith("..") else rel


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def resolve_artifact_set(task_dir, want_filtered, src_sfx=""):
    """Decide which artifact set (filtered or not) to read for one task dir.

    Returns ``(suffix, note)``. Filtering is a T/L-only concept: the exclusion
    list marks slices whose mask has more than one connected component, and A/D
    measurements come from landmarks, not masks. So an A/D directory publishes no
    ``_filtered`` artifacts even on a filtered run, and falling back to the
    unfiltered set is correct rather than a silent downgrade -- but the fallback
    is recorded in the report header either way, so a reader never has to guess
    which sample set produced a number.

    ``src_sfx`` is the parsed-source marker, which precedes ``_filtered`` in the
    task-level filenames and so must be part of the probe.
    """
    if not want_filtered:
        return "", "unfiltered (whole parsed sample set)"
    probe = os.path.join(
        task_dir, f"summary_CDA_uncertainty{src_sfx}_filtered.json"
    )
    if os.path.isfile(probe):
        return "_filtered", "filtered (multi-cluster slices excluded)"
    return "", (
        "unfiltered (this task publishes no filtered artifacts: its datasets "
        "ship no exclusion list)"
    )


def load_metrics(task_dir, config_yaml, suffix, parsed_dirname):
    """Collect ``overall`` metrics per proxy, in config model order.

    Returns ``(by_proxy, absent, datasets_by_proxy, n_models)`` where ``by_proxy``
    maps a proxy name to an ordered list of ``(display_name, metrics_entry)``.
    """
    display_map = load_model_display_map(config_yaml)
    model_items = resolve_model_dirs(task_dir, display_map)
    metrics_name = (
        f"{SUMMARY_FILENAME_CDA_METRICS.removesuffix('.json')}{suffix}.json"
    )

    by_proxy, absent, datasets = {}, [], {}
    for model_dir, display in model_items:
        path = os.path.join(model_dir, parsed_dirname, metrics_name)
        if not os.path.isfile(path):
            absent.append(display)
            continue
        with open(path) as f:
            metrics = json.load(f)
        for proxy, entry in metrics.get("overall", {}).items():
            by_proxy.setdefault(proxy, []).append((display, entry))
        # by_dataset keys are "<proxy> @ <dataset>" -- the only place the report
        # can learn which datasets fed a proxy.
        for key in metrics.get("by_dataset", {}):
            proxy, _, dataset = key.partition(" @ ")
            if dataset:
                datasets.setdefault(proxy, set()).add(dataset)
    return by_proxy, absent, datasets, len(model_items)


def load_uncertainty(path):
    """Index uncertainty rows by ``(model, proxy)``; also return the run metadata."""
    if not os.path.isfile(path):
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    index = {(r["model"], r["proxy"]): r for r in data.get("rows", [])}
    meta = {k: v for k, v in data.items() if k != "rows"}
    return index, meta


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _ordered_proxies(by_proxy):
    ordered = [p for p in PROXY_ORDER if p in by_proxy]
    ordered += [p for p in by_proxy if p not in ordered]
    return ordered


def agreement_section(by_proxy, datasets, unc, absent, n_models):
    """Per-proxy leaderboards, one table each."""
    out = []
    for proxy in _ordered_proxies(by_proxy):
        entries = by_proxy[proxy]
        ds = ", ".join(sorted(datasets.get(proxy, ()))) or "—"
        out.append(f"\n### {proxy}")
        out.append(f"\n_Data: {ds}_\n")

        rows = []
        for display, e in entries:
            u = unc.get((display, proxy))
            flag = " ⚠" if u and u.get("low_n") else ""
            rows.append([
                display,
                _num(e.get("accuracy")),
                _num(e.get("accuracy_coverage_adjusted")),
                _num(e.get("cohen_kappa")),
                _num(e.get("weighted_kappa")),
                _ci(u),
                _pval(u.get("p_kappa_gt_zero")) if u else "—",
                f"{e.get('n_parsed', 0)}{flag}",
                str(e.get("n_total", 0)),
            ])
        out += _table(
            ["Model", "Acc", "AccCov", "κ", "wκ", "95% CI", "p", "Nparsed", "Ntotal"],
            rows,
        )

    if absent:
        out.append(
            f"\n**Not reported** ({len(absent)} of {n_models} configured models "
            "produced no CDA-eligible samples): "
            + ", ".join(absent)
        )
    return out


def build_report(args):
    src = source_suffix(args.parsed_dirname)
    ad_sfx, ad_note = resolve_artifact_set(args.ad_task_dir, args.filtered, src)
    tl_sfx, tl_note = resolve_artifact_set(args.tl_task_dir, args.filtered, src)

    ad_by_proxy, ad_absent, ad_ds, ad_n = load_metrics(
        args.ad_task_dir, args.ad_config_yaml, ad_sfx, args.parsed_dirname
    )
    tl_by_proxy, tl_absent, tl_ds, tl_n = load_metrics(
        args.tl_task_dir, args.tl_config_yaml, tl_sfx, args.parsed_dirname
    )
    # Refuse to render a report in which nothing was found. Every model absent
    # produces a complete-looking document whose every row says "not reported" --
    # indistinguishable at a glance from a real result, and the likeliest cause is
    # a parsed source that does not exist on disk (a mistyped --parsed_dirname, or
    # an analysis step that was never run for this source). Since sources are
    # matched by prefix rather than a fixed list, that name cannot be validated
    # up front, so it has to fail here instead.
    if not ad_by_proxy and not tl_by_proxy:
        raise FileNotFoundError(
            f"No CDA metrics found under '{args.parsed_dirname}/' in either "
            f"{args.ad_task_dir} or {args.tl_task_dir}. Check the source folder "
            "exists for these models, and that summarize_CDA_task.py ran with the "
            "same --parsed_dirname."
        )

    ad_unc, ad_meta = load_uncertainty(
        os.path.join(args.ad_task_dir, f"summary_CDA_uncertainty{src}{ad_sfx}.json")
    )
    tl_unc, tl_meta = load_uncertainty(
        os.path.join(args.tl_task_dir, f"summary_CDA_uncertainty{src}{tl_sfx}.json")
    )
    boot = tl_meta or ad_meta
    L = []
    L.append("# Clinical Decision Agreement — Results")
    L.append(
        "\nGenerated by `build_CDA_report.py`. Do not edit by hand: it is "
        "overwritten on every run of `run_cda.sh`."
    )
    L.append(
        "\nDoes the model's measurement error change the clinical decision? Each "
        "measurement is mapped through a published cutoff table into a category, "
        "and the model's category is compared with the reference category. See "
        "the `medvision` skill's `analysis` sub-skill (`references/cda.md`) for "
        "the method and `cda_config.py` for the cutoff tables."
    )

    L.append("\n## Provenance\n")
    L += _table(
        ["Input", "Value"],
        [
            [
                "Parsed source",
                f"`{args.parsed_dirname}/` — predictions read from "
                f"`{parsed_source_field(args.parsed_dirname)}`",
            ],
            ["A/D results", f"`{_relpath(args.ad_task_dir)}` — {ad_note}"],
            ["T/L results", f"`{_relpath(args.tl_task_dir)}` — {tl_note}"],
            ["A/D config", f"`{_relpath(args.ad_config_yaml)}` ({ad_n} models)"],
            ["T/L config", f"`{_relpath(args.tl_config_yaml)}` ({tl_n} models)"],
            [
                "Uncertainty",
                f"{boot.get('n_boot', '—')} bootstrap resamples, seed "
                f"{boot.get('seed', '—')}, resampling whole imaging volumes",
            ],
        ],
    )

    L.append("\n## Reading these tables\n")
    L.append(
        "- **Acc** — agreement on parsed samples. **AccCov** — the same with "
        "unparseable predictions counted as wrong; the gap between them is coverage.\n"
        "- **κ / wκ** — Cohen's kappa and quadratic-weighted kappa (ordinal proxies "
        "only). 0 means no better than chance at the same marginals.\n"
        "- **95% CI / p** — percentile bootstrap over whole imaging volumes, and a "
        "one-sided test of κ > 0 inverted from the same resamples.\n"
        "- **⚠** on `Nparsed` marks fewer than 10 scored records: the statistic is "
        "arithmetically valid but practically meaningless.\n"
        "- The decision-flip rate is omitted here because it is exactly 1 − Acc.\n"
        "- κ is **not comparable across proxies** — it depends on how close the "
        "cohort's values sit to a cutoff. Compare models within a table."
    )

    L.append("\n## Decision agreement\n")
    L.append(
        "Category of the prediction vs category of the ground-truth measurement. "
        "Both sides pass through the same cutoff table, so any disagreement is "
        "caused by measurement error alone."
    )
    L += agreement_section(ad_by_proxy, ad_ds, ad_unc, ad_absent, ad_n)
    L += agreement_section(tl_by_proxy, tl_ds, tl_unc, tl_absent, tl_n)

    L.append("\n## Regenerating this report\n")
    L.append("```bash")
    L.append("bash scripts/run_cda.sh \\")
    L.append(f"  --ad-task-dir {_relpath(args.ad_task_dir)} "
             f"--tl-task-dir {_relpath(args.tl_task_dir)} \\")
    L.append(f"  --ad-config {_relpath(args.ad_config_yaml)} "
             f"--tl-config {_relpath(args.tl_config_yaml)} \\")
    if src:
        L.append(f"  --parsed-dirname {args.parsed_dirname} \\")
    if args.filtered:
        L.append("  --removed-samples-dir <data_dir>/Datasets \\")
    L.append(f"  --out {_relpath(args.out)}")
    L.append("```")
    L.append(
        "\nIt needs the `Results/` tree the benchmark wrote (and, for a filtered "
        "T/L run, the dataset folders holding the removed-samples JSONs), so the "
        "analysis code alone cannot reproduce it."
    )
    return "\n".join(L) + "\n"


def parse_args():
    p = argparse.ArgumentParser(
        description="Render the final CDA leaderboard report as Markdown."
    )
    p.add_argument("--ad_task_dir", required=True, help="A/D results directory.")
    p.add_argument("--tl_task_dir", required=True, help="T/L results directory.")
    p.add_argument("--ad_config_yaml", required=True, help="Config for the A/D dir.")
    p.add_argument("--tl_config_yaml", required=True, help="Config for the T/L dir.")
    p.add_argument(
        "--parsed_dirname",
        default=CDA_DEFAULT_PARSED_DIRNAME,
        type=validate_parsed_dirname,
        help="Which parsed-results folder the analysis wrote into. Must match the "
        "--parsed_dirname the analysis scripts ran with. Recorded in the report's "
        "provenance table; pass a distinct --out per source.",
    )
    p.add_argument(
        "--filtered",
        action="store_true",
        help="Read the '_filtered' artifacts where a task publishes them. A task "
        "that publishes none (A/D) falls back to unfiltered, and the report header "
        "records which set each task contributed.",
    )
    p.add_argument("--out", required=True, help="Markdown file to write.")
    p.add_argument(
        "--repo_root",
        default=None,
        help="Directory against which paths in the provenance table are "
        "shortened (default: current working directory). Paths outside it are "
        "printed as given.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.repo_root:
        REPO_ROOT = os.path.abspath(args.repo_root)
    report = build_report(args)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"CDA report written to {args.out}")
