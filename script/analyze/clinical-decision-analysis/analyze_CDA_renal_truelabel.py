"""Renal T-stage true-label validation for the CDA task (self-contained).

The main CDA summary (``summarize_CDA_task.py``) is *self-consistent*: it compares
the clinical category derived from the model's predicted measurement against the
category derived from the ground-truth measurement. This script adds the
complementary *external* check for the renal T-stage proxy: it compares the
model's size-derived AJCC T category against the **pathologic T stage** recorded
in the KiTS23 clinical table — an independent, non-imaging reference.

Two caveats are handled explicitly and reported:

1. **2D-slice vs 3D.** MedVision T/L measurements are per-slice ellipse major
   axes. We aggregate them per case (max major axis over the case's annotated
   slices) as a proxy for the 3D greatest dimension that AJCC staging uses.
2. **Invasion-based stages.** Pathologic pT3 (perinephric / venous invasion) and
   pT4 (beyond Gerota's fascia) are NOT size-defined, so a size-only rule can
   never assign them. We therefore report both the full 6-class confusion (which
   exposes this ceiling) and an organ-confined pT1-pT2 stratum (where size is the
   staging axis and the comparison is fair).

To isolate "loss from invasion-based staging" from "loss from the model", the
report also includes the *intrinsic* ceiling: the pathologic-3D-size-derived
category vs the pathologic stage. Requires the KiTS23 clinical table
(``kits23.json``); it is fetched from the public KiTS23 repository and cached if
not supplied. Only KiTS23 ships pathologic stage (KiPA22 does not), so this
analysis is KiTS23-only.

The script is standalone (imports only ``cda_config`` / ``cda_stats`` from the
same folder, numpy, and PyYAML when ``--config_yaml`` is used). Run it directly:

    python analyze_CDA_renal_truelabel.py --task_dir <Results/...> \
        --config_yaml config-TL-CoT.yaml

Run as a CLI; see :func:`parse_args` for the accepted arguments.
"""

import argparse
import ast
import json
import os
import urllib.request

from cda_config import (
    CDA_DEFAULT_PARSED_DIRNAME,
    CDA_LLM_PARSED_DIRNAME,
    CDA_RENAL_TSTAGE,
    parsed_source_field,
    source_suffix,
    validate_parsed_dirname,
)
from cda_stats import (
    cal_clinical_agreement,
    categorize,
    convert_numpy_to_python,
    filtered_suffix,
    load_model_display_map,
    load_removed_set,
    removed_key,
    resolve_model_dirs,
    sorted_glob,
)

# Public KiTS23 clinical table (per-case pathology). Fetched + cached on demand.
KITS23_JSON_URL = "https://raw.githubusercontent.com/neheller/kits23/main/dataset/kits23.json"

# kits23.json pathology_t_stage encoding -> our T-category labels.
KITS23_TSTAGE_MAP = {
    "1a": "T1a",
    "1b": "T1b",
    "2a": "T2a",
    "2b": "T2b",
    "3": "T3",
    "4": "T4",
}
# Full ordinal ordering (includes invasion-based T3/T4 that a size rule cannot produce).
TRUE_TSTAGE_ORDER = ["T1a", "T1b", "T2a", "T2b", "T3", "T4"]
# Organ-confined stratum where tumor size is the staging axis (fair comparison).
ORGAN_CONFINED_ORDER = ["T1a", "T1b", "T2a", "T2b"]

# Default cache location for the KiTS23 clinical table (relative to CWD / repo root).
DEFAULT_KITS23_CACHE = os.path.join(
    "Data", "Datasets", "KiTS23", "kits23_clinical.json"
)


def resolve_kits23_table(kits23_json=None, download=True):
    """Return the KiTS23 clinical table as a list of per-case dicts.

    Resolution order: an explicit ``kits23_json`` path if it exists; otherwise
    download from :data:`KITS23_JSON_URL` (and cache to ``kits23_json`` when a
    path is given). Raises a clear error if the table cannot be obtained.
    """
    if kits23_json and os.path.exists(kits23_json):
        print(f"Loading KiTS23 clinical table from {kits23_json}")
        with open(kits23_json) as f:
            return json.load(f)

    if not download:
        raise FileNotFoundError(
            f"KiTS23 clinical table not found at {kits23_json!r}. Provide --kits23_json "
            f"or allow download from {KITS23_JSON_URL}."
        )

    print(f"Downloading KiTS23 clinical table from {KITS23_JSON_URL} ...")
    try:
        with urllib.request.urlopen(KITS23_JSON_URL, timeout=30) as r:
            data = r.read()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download KiTS23 clinical table ({e}). Manually fetch "
            f"{KITS23_JSON_URL} and pass it via --kits23_json."
        )
    cases = json.loads(data)
    if kits23_json:
        os.makedirs(os.path.dirname(os.path.abspath(kits23_json)), exist_ok=True)
        with open(kits23_json, "wb") as f:
            f.write(data)
        print(f"Cached KiTS23 clinical table to {kits23_json}")
    return cases


def build_stage_table(cases):
    """Map ``case_id`` -> clinical fields used for renal true-label validation.

    Returns a dict ``case_id -> {"true_stage", "pathologic_size_mm",
    "radiographic_size_mm", "malignant"}``. ``true_stage`` is None for cases
    without a size/AJCC-mappable pathologic T stage (e.g. "na").
    """
    table = {}
    for c in cases:
        cid = c.get("case_id")
        if not cid:
            continue
        stage_raw = str(c.get("pathology_t_stage"))
        true_stage = KITS23_TSTAGE_MAP.get(stage_raw)  # None for "na"/"None"/unknown
        path_size = c.get("pathologic_size")
        rad_size = c.get("radiographic_size")
        table[cid] = {
            "true_stage": true_stage,
            "pathologic_size_mm": (float(path_size) * 10.0 if path_size is not None else None),
            "radiographic_size_mm": (float(rad_size) * 10.0 if rad_size is not None else None),
            "malignant": c.get("malignant"),
        }
    return table


def _parse_major(target, filtered_resps):
    """Extract (gt_major_mm, pred_major_mm) from one KiTS23 T/L sample.

    ``pred_major_mm`` is None when the prediction did not parse to two values
    (matching the benchmark's success rule)."""
    try:
        gt = float(ast.literal_eval(str(target))[0])
    except Exception:
        gt = None
    pred = None
    if filtered_resps:
        try:
            parts = [p.strip() for p in str(filtered_resps[0]).strip().split(",")]
            vals = [float(p) for p in parts if p != ""]
            if len(vals) == 2:
                pred = vals[0]
        except Exception:
            pred = None
    return gt, pred


def aggregate_cases_from_model(
    model_dir,
    removed_samples_dir=None,
    removed_samples_filename=None,
    parsed_dirname=CDA_DEFAULT_PARSED_DIRNAME,
):
    """Aggregate per-case max major-axis (GT and predicted) from a model's KiTS23 T/L files.

    Returns dict ``case_id -> {"gt_max", "pred_max", "n_slices", "n_pred_fail"}``.
    ``pred_max`` is None when no slice of the case had a parseable prediction.

    With ``removed_samples_dir`` the multi-cluster slices the T/L benchmark
    excludes are skipped here too. That matters more than for Track 1: this
    aggregates by MAX over slices, so a single excluded slice can set a whole
    case's ``gt_max`` and shift its stage.

    ``parsed_dirname`` selects the source folder and, with it, the row field
    holding the prediction.
    """
    resps_field = parsed_source_field(parsed_dirname)
    parsed_dir = os.path.join(model_dir, parsed_dirname)
    # sorted_glob, not glob: see cda_stats.sorted_glob -- file order propagates
    # into the per-case aggregation order and the uncertainty bootstrap.
    files = [
        f
        for f in sorted_glob(os.path.join(parsed_dir, "*KiTS23*TumorLesionSize*.jsonl"))
        if not ("_proc_acc" in os.path.basename(f) or "_eq_acc" in os.path.basename(f))
    ]
    removed_set = load_removed_set(
        removed_samples_dir, "KiTS23", removed_samples_filename
    )

    cases = {}
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                if not data:
                    continue
                doc = data.get("doc", {})
                img = doc.get("image_file")
                if not img:
                    continue
                if removed_set is not None:
                    key = removed_key(doc, "KiTS23")
                    if key is not None and key in removed_set:
                        continue
                cid = os.path.basename(img).split(".")[0]  # case_00149.nii.gz -> case_00149
                gt, pred = _parse_major(data.get("target"), data.get(resps_field))
                if gt is None:
                    continue
                rec = cases.setdefault(
                    cid, {"gt_max": None, "pred_max": None, "n_slices": 0, "n_pred_fail": 0}
                )
                rec["n_slices"] += 1
                rec["gt_max"] = gt if rec["gt_max"] is None else max(rec["gt_max"], gt)
                if pred is None:
                    rec["n_pred_fail"] += 1
                else:
                    rec["pred_max"] = (
                        pred if rec["pred_max"] is None else max(rec["pred_max"], pred)
                    )
    return cases


def _cat(size_mm):
    """Size (mm) -> AJCC renal T category (T1a..T2b), or None."""
    return categorize(
        size_mm,
        CDA_RENAL_TSTAGE["cutoffs"],
        CDA_RENAL_TSTAGE["labels"],
        CDA_RENAL_TSTAGE["right_closed"],
    )


def _compare(reference_cats, tested_cats, labels, n_total):
    """Agreement of ``tested`` categories against ``reference`` categories."""
    return cal_clinical_agreement(
        reference_cats, tested_cats, labels, ordinal=True, n_total=n_total
    )


def analyze_model(
    model_dir,
    stage_table,
    removed_samples_dir=None,
    removed_samples_filename=None,
    parsed_dirname=CDA_DEFAULT_PARSED_DIRNAME,
):
    """Compute renal true-label agreement for one model directory.

    Returns a metrics dict, or None when the model has no KiTS23 cases that join
    to a pathologic stage.
    """
    cases = aggregate_cases_from_model(
        model_dir, removed_samples_dir, removed_samples_filename, parsed_dirname
    )
    if not cases:
        return None

    # Join to clinical stage; keep only cases with a size/AJCC-mappable pathologic stage.
    joined = []
    unmatched = 0
    for cid, rec in cases.items():
        clin = stage_table.get(cid)
        if clin is None:
            unmatched += 1
            continue
        if clin["true_stage"] is None:
            continue
        joined.append((cid, rec, clin))

    if not joined:
        return None

    # Per-case derived categories.
    rows = []
    for cid, rec, clin in joined:
        rows.append(
            {
                "case_id": cid,
                "true_stage": clin["true_stage"],
                "gt_cat": _cat(rec["gt_max"]),  # 2D-slice-max GT -> category
                "pred_cat": _cat(rec["pred_max"]),  # 2D-slice-max prediction -> category
                "path_size_cat": _cat(clin["pathologic_size_mm"]),  # true 3D size -> category
                "pred_max_mm": rec["pred_max"],
                "gt_max_mm": rec["gt_max"],
                "pathologic_size_mm": clin["pathologic_size_mm"],
            }
        )

    def build(order, restrict_true=None):
        """Assemble the comparison set over cases whose true stage is in ``order``
        (and, if ``restrict_true`` given, additionally in that set)."""
        sub = [r for r in rows if r["true_stage"] in order]
        if restrict_true is not None:
            sub = [r for r in sub if r["true_stage"] in restrict_true]

        def one(tested_key):
            ref = [r["true_stage"] for r in sub if r[tested_key] is not None]
            tst = [r[tested_key] for r in sub if r[tested_key] is not None]
            return _compare(ref, tst, order, n_total=len(sub))

        return {
            "n_cases": len(sub),
            "pred_vs_pathologic": one("pred_cat"),
            "gt2Dslice_vs_pathologic": one("gt_cat"),
            "pathologic3Dsize_vs_pathologic": one("path_size_cat"),
        }

    # Self-consistent per-case (both size-derived -> 4-class); over organ-confined-capable cases.
    sc_ref = [r["gt_cat"] for r in rows if r["pred_cat"] is not None and r["gt_cat"] is not None]
    sc_tst = [r["pred_cat"] for r in rows if r["pred_cat"] is not None and r["gt_cat"] is not None]
    n_sc_total = sum(1 for r in rows if r["gt_cat"] is not None)
    self_consistent = cal_clinical_agreement(
        sc_ref, sc_tst, ORGAN_CONFINED_ORDER, ordinal=True, n_total=n_sc_total
    )

    return {
        "n_cases_joined": len(joined),
        "n_cases_unmatched_to_clinical_table": unmatched,
        "full_6class": build(TRUE_TSTAGE_ORDER),
        "organ_confined_pT1_pT2": build(TRUE_TSTAGE_ORDER, restrict_true=ORGAN_CONFINED_ORDER),
        "self_consistent_pred_vs_gt2Dslice": self_consistent,
        "cases": rows,
    }


def _fmt(x, width=8, prec=4):
    if x is None:
        x = float("nan")
    return f"{x:<{width}.{prec}f}"


def _print_block(emit, title, block):
    emit(f"\n  [{title}]  (n_cases={block['n_cases']})")
    emit(
        f"    {'Comparison':<34} | {'Acc':<8} | {'wKappa':<8} | {'Kappa':<8} | "
        f"{'Nparsed':<8} | {'Ntotal':<8}"
    )
    emit("    " + "-" * 88)
    for key in (
        "pred_vs_pathologic",
        "gt2Dslice_vs_pathologic",
        "pathologic3Dsize_vs_pathologic",
    ):
        m = block[key]
        emit(
            f"    {key:<34} | {_fmt(m.get('accuracy'))} | "
            f"{_fmt(m.get('weighted_kappa'))} | {_fmt(m.get('cohen_kappa'))} | "
            f"{m.get('n_parsed', 0):<8} | {m.get('n_total', 0):<8}"
        )


def _model_items(task_dir, model_dir, display_map):
    """Return ``(ordered [(model_dir, display_name)], out_dir)``.

    Task-directory mode delegates to the shared resolver, so this script and
    ``summarize_CDA_task.py`` always agree on which models a config selects.
    """
    if task_dir is not None:
        return resolve_model_dirs(task_dir, display_map), task_dir
    if model_dir is not None:
        return [(model_dir, os.path.basename(os.path.normpath(model_dir)))], model_dir
    raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def main(**kwargs):
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    kits23_json = kwargs.get("kits23_json") or DEFAULT_KITS23_CACHE
    no_download = kwargs.get("no_download", False)
    config_yaml = kwargs.get("config_yaml")
    removed_samples_dir = kwargs.get("removed_samples_dir")
    removed_samples_filename = kwargs.get("removed_samples_filename")
    parsed_dirname = kwargs.get("parsed_dirname") or CDA_DEFAULT_PARSED_DIRNAME
    _fsuffix = filtered_suffix(removed_samples_dir)
    # Per-model JSONs land inside the source folder and need no source marker;
    # the task-level report sits above it and does.
    _ssuffix = source_suffix(parsed_dirname)
    print(f"Parsed source: {parsed_dirname}/ "
          f"(prediction field: {parsed_source_field(parsed_dirname)})")

    stage_table = build_stage_table(
        resolve_kits23_table(kits23_json, download=not no_download)
    )

    display_map = load_model_display_map(config_yaml)
    model_items, out_dir = _model_items(task_dir, model_dir, display_map)

    output_lines = []
    leaderboard = []  # (display, wkappa, acc, acc_cov, n_parsed, n_total)

    def emit(text):
        print(text)
        output_lines.append(text)

    emit("\n===== RENAL T-STAGE TRUE-LABEL VALIDATION (KiTS23) =====")
    emit(
        "pred_vs_pathologic = model size->stage vs pathologic stage (primary true-label test)."
    )
    emit(
        "gt2Dslice_vs_pathologic = 2D-slice GT size->stage vs pathologic. A REFERENCE, "
        "not a ceiling: max-over-slices under-estimates the 3D greatest dimension, so a "
        "model that over-measures can beat it."
    )
    emit(
        "pathologic3Dsize_vs_pathologic = true 3D size->stage vs pathologic (intrinsic size ceiling; T3/T4 are invasion-based)."
    )

    absent = []  # models with no joinable KiTS23 cases — reported, not just skipped
    for md, display in model_items:
        res = analyze_model(
            md,
            stage_table,
            removed_samples_dir,
            removed_samples_filename,
            parsed_dirname,
        )
        if res is None:
            # No KiTS23 T/L files, or no case id joined to a pathologic stage.
            # Silently skipping would make a partial leaderboard look complete.
            absent.append(display)
            continue

        # Write per-model JSON into parsed/ if it exists, else the model dir.
        parsed_dir = os.path.join(md, parsed_dirname)
        dst_dir = parsed_dir if os.path.exists(parsed_dir) else md
        json_path = os.path.join(
            dst_dir, f"summary_CDA_renal_truelabel{_fsuffix}.json"
        )
        with open(json_path, "w") as f:
            json.dump(convert_numpy_to_python(res), f, indent=2)

        emit(f"\nModel: {display}  (joined {res['n_cases_joined']} cases, "
             f"{res['n_cases_unmatched_to_clinical_table']} unmatched)")
        _print_block(emit, "Full 6-class (incl. invasion-based pT3/pT4)", res["full_6class"])
        _print_block(emit, "Organ-confined pT1-pT2 (size is the staging axis)", res["organ_confined_pT1_pT2"])
        sc = res["self_consistent_pred_vs_gt2Dslice"]
        emit(
            f"\n  [Self-consistent per-case pred vs GT-2D-slice]  acc={_fmt(sc.get('accuracy'))} "
            f"wKappa={_fmt(sc.get('weighted_kappa'))} kappa={_fmt(sc.get('cohen_kappa'))} "
            f"n={sc.get('n_parsed')}/{sc.get('n_total')}"
        )
        emit("\n" + "=" * 100)
        oc = res["organ_confined_pT1_pT2"]["pred_vs_pathologic"]
        leaderboard.append((
            display,
            oc.get("weighted_kappa"),
            oc.get("accuracy"),
            oc.get("accuracy_coverage_adjusted"),
            oc.get("n_parsed"),
            oc.get("n_total"),
        ))

    if absent:
        emit(f"\nNOT REPORTED ({len(absent)} of {len(model_items)} models): "
             "no KiTS23 T/L files, or no case joined to a pathologic stage")
        for display in absent:
            emit(f"  {display}")

    # Leaderboard (config order) of the primary true-label metric.
    # AccCov (parse failures counted as disagreement) sits beside Acc because the
    # denominators here run from a handful of cases to 70: ranking on parsed-only
    # accuracy alone rewards a model that measured very few cases.
    if display_map is not None and leaderboard:
        emit("\n\n### Leaderboard (config order): organ-confined pred -> stage vs pathologic")
        emit(
            f"{'Model':<26} | {'wKappa':<8} | {'Acc':<8} | {'AccCov':<8} | "
            f"{'Nparsed':<8} | {'Ntotal':<8}"
        )
        emit("-" * 77)
        for display, wk, acc, acccov, npar, ntot in leaderboard:
            emit(
                f"{display[:26]:<26} | {_fmt(wk)} | {_fmt(acc)} | {_fmt(acccov)} | "
                f"{npar:<8} | {ntot:<8}"
            )

    suffix = _ssuffix + _fsuffix + ("_canonical" if display_map is not None else "")
    out_path = os.path.join(out_dir, f"summary_CDA_renal_truelabel{suffix}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nRenal true-label summary saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Renal T-stage true-label validation (KiTS23) for the CDA task."
    )
    parser.add_argument("--task_dir", type=str, help="Task directory containing model folders.")
    parser.add_argument("--model_dir", type=str, help="A single model directory.")
    parser.add_argument(
        "--parsed_dirname",
        default=CDA_DEFAULT_PARSED_DIRNAME,
        type=validate_parsed_dirname,
        help="Which parsed-results folder inside each model directory to read: "
        "'parsed' (regex parser), or any 'llm-parsed*' folder written by an "
        f"LLM-judge re-parse (e.g. '{CDA_LLM_PARSED_DIRNAME}'). Matched by prefix. "
        "The prefix also selects the row field holding the prediction. The "
        "per-model JSON is written back into that folder; the task-level report "
        "gains a source marker (e.g. '_llm-parsed-gemma-4-31b').",
    )
    parser.add_argument(
        "--kits23_json",
        type=str,
        default=None,
        help="Path to kits23.json clinical table. If absent, it is downloaded and "
        f"cached to {DEFAULT_KITS23_CACHE}.",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Do not download the KiTS23 clinical table; require --kits23_json to exist.",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="CDA config listing the models to analyze. This analysis reads "
        "tumor/lesion-size results, so it takes config-TL-CoT.yaml. When given, "
        "only those models are analyzed, in config order, with display names and "
        "a leaderboard.",
    )
    parser.add_argument(
        "--removed_samples_dir",
        type=str,
        default=None,
        help="Root directory of per-dataset removed-samples JSONs (e.g. "
        "Data/Datasets). Excludes the multi-cluster slices the benchmark's own "
        "summarize_TL_task.py excludes. Output filenames gain a '_filtered' marker.",
    )
    parser.add_argument(
        "--removed_samples_filename",
        type=str,
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Filename of the removed-samples JSON within each dataset "
        "subdirectory. Matches summarize_TL_task.py's default.",
    )
    args = parser.parse_args()
    if args.task_dir is None and args.model_dir is None:
        parser.error("Either --task_dir or --model_dir must be provided.")
    if args.config_yaml is not None and args.task_dir is None:
        parser.error("--config_yaml can only be used with --task_dir")
    return args


if __name__ == "__main__":
    main(**vars(parse_args()))
