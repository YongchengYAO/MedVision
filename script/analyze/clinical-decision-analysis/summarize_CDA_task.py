"""Summarize Clinical Decision Agreement (CDA) across models (self-contained).

CDA is a *downstream proxy* for the MedVision benchmark. Instead of reporting
raw measurement error (MAE / MRE), it maps each geometric measurement to a
discrete clinical category using a standard, published cutoff table, and reports
whether the model's *predicted* measurement yields the *same* clinical category
as the *ground-truth* measurement. It answers: "does the model's residual
measurement error flip the downstream clinical decision?"

This script reuses the already-parsed benchmark outputs (``parsed/*.jsonl``) — no
re-inference is performed. It supports two proxies (see ``cda_config.py`` for
the cutoff tables and citations):

- **ANB / SNA / SNB angle -> orthodontic skeletal class** (Ceph-Biometrics-400).
  ANB is primary; SNA/SNB are secondary companions. (nominal)
- **Renal tumor greatest dimension -> AJCC 8th-ed T category** (KiTS23, KiPA22).
  (ordinal -> also reports quadratic-weighted kappa)

The script is standalone: it imports only ``cda_config`` and ``cda_stats`` from
the same folder, plus numpy (and PyYAML only when ``--config_yaml`` is used). Run
it directly:

    python summarize_CDA_task.py --task_dir <Results/...> \
        --config_yaml config-AD-CoT.yaml --skip_model_wo_parsed_files

With ``--config_yaml`` the cross-model report is restricted to the config's
models, labelled with their display names, ordered as in the config, and
prefixed with a per-proxy leaderboard. Pass the config matching the task
directory -- ``config-AD-CoT.yaml`` for an A/D directory, ``config-TL-CoT.yaml``
for a T/L one -- because a model's results folder can differ between the two.

Run as a CLI; see :func:`parse_args` for the accepted arguments.
"""

import argparse
import ast
import json
import os
import re

from cda_config import (
    CDA_CEPH_ANGLE_PROXIES,
    CDA_DEFAULT_PARSED_DIRNAME,
    CDA_LLM_PARSED_DIRNAME,
    CDA_RENAL_TL_DATASETS,
    CDA_RENAL_TSTAGE,
    SUMMARY_FILENAME_CDA_METRICS,
    SUMMARY_FILENAME_CDA_VALUES,
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

# File-kind markers in the parsed JSONL filenames.
_ANGLE_FILE_MARKER = "BiometricsFromLandmarks_Angle"
_TL_FILE_MARKER = "TumorLesionSize"


def _file_kind(filename):
    """Return ``"angle"`` for a cephalometric-angle file, ``"tl"`` for a
    tumor/lesion-size file, or ``None`` for files with no CDA proxy (e.g.
    distance files)."""
    if _TL_FILE_MARKER in filename:
        return "tl"
    if _ANGLE_FILE_MARKER in filename:
        return "angle"
    return None


def _proxies_for_sample(file_kind, dataset_name, metric_key):
    """Return the list of ``(proxy_key, spec)`` a sample contributes to.

    - Angle samples map to a Ceph angle proxy only when their ``metric_key`` is a
      recognized cephalometric angle (ANB/SNA/SNB).
    - T/L samples map to the renal T-stage proxy for renal-tumor datasets, and to
      no proxy otherwise.
    """
    hits = []
    if file_kind == "angle":
        spec = CDA_CEPH_ANGLE_PROXIES.get(metric_key)
        if spec is not None:
            hits.append((spec["name"], spec))
    elif file_kind == "tl":
        if dataset_name in CDA_RENAL_TL_DATASETS:
            hits.append((CDA_RENAL_TSTAGE["name"], CDA_RENAL_TSTAGE))
    return hits


def _gt_pred_values(file_kind, target, filtered_resps):
    """Extract the (ground-truth, predicted) numeric measurement used for a proxy.

    For angle samples this is the single scalar; for T/L samples it is the major
    (longest) axis length in mm. Returns ``(gt, pred)`` where ``gt`` is a float
    (or None if unparseable) and ``pred`` is a float or None when the prediction
    could not be parsed into the expected number of values.
    """
    # Ground truth (should always parse; guarded defensively).
    try:
        parsed_gt = ast.literal_eval(str(target))
        if file_kind == "tl":
            gt = float(parsed_gt[0])  # major axis
        else:
            gt = float(parsed_gt[0] if isinstance(parsed_gt, (list, tuple)) else parsed_gt)
    except Exception:
        gt = None

    # Prediction: enforce the same value-count rule the MAE/MRE parsers use.
    pred = None
    if filtered_resps:
        try:
            parts = [p.strip() for p in str(filtered_resps[0]).strip().split(",")]
            vals = [float(p) for p in parts if p != ""]
            if file_kind == "tl":
                if len(vals) == 2:
                    pred = vals[0]  # major axis
            else:
                if len(vals) == 1:
                    pred = vals[0]
        except Exception:
            pred = None

    return gt, pred


def _metrics_filename(base, limit, removed_samples_dir):
    """Suffix an output filename with ``_filtered`` and/or ``_limit<N>``.

    Suffix order matches summarize_TL_task.py. Both markers are load-bearing: a
    filtered run and a full run must not overwrite each other's numbers, since
    they are computed over different sample sets.
    """
    stem = base.removesuffix(".json")
    stem += filtered_suffix(removed_samples_dir)
    if limit is not None:
        stem += f"_limit{limit}"
    return f"{stem}.json"


def _new_group():
    return {"spec": None, "gt": [], "pred": [], "n_fail": 0}


def process_parsed_file_in_model_folder(
    model_dir,
    limit=None,
    removed_samples_dir=None,
    removed_samples_filename=None,
    parsed_dirname=CDA_DEFAULT_PARSED_DIRNAME,
):
    """Accumulate CDA proxy categories over a model's parsed files and write summaries.

    Reads ``<model_dir>/<parsed_dirname>/*.jsonl`` (excluding analysis outputs),
    routes each sample to its applicable proxies, categorizes the GT and predicted
    measurements, and aggregates per proxy (overall) and per proxy-x-dataset.
    Writes ``summary_metrics_CDA_Task.json`` and ``summary_values_CDA_Task.json``
    back into that same source directory.

    ``parsed_dirname`` also selects which row field holds the prediction --
    ``llm-parsed*`` rows carry it under ``LLM_filtered_resps``.
    """
    resps_field = parsed_source_field(parsed_dirname)
    parsed_files_dir = os.path.join(model_dir, parsed_dirname)
    if not os.path.exists(parsed_files_dir):
        print(f"Parsed files directory does not exist: {parsed_files_dir}, skipping...")
        return

    # sorted_glob, not glob: file order determines the order records land in the
    # values JSON, which determines cluster order in the uncertainty bootstrap.
    jsonl_files = [
        f
        for f in sorted_glob(os.path.join(parsed_files_dir, "*.jsonl"))
        if not ("_proc_acc" in os.path.basename(f) or "_eq_acc" in os.path.basename(f))
    ]

    overall = {}  # proxy_key -> group
    by_dataset = {}  # "proxy_key @ dataset" -> group
    sample_values = []  # per-sample records for the values JSON

    for jsonl_path in jsonl_files:
        filename = os.path.basename(jsonl_path)
        file_kind = _file_kind(filename)
        if file_kind is None:
            continue

        match = re.search(r"samples_([^_]+)_", filename)
        if not match:
            continue
        dataset_name = match.group(1)
        removed_set = load_removed_set(
            removed_samples_dir, dataset_name, removed_samples_filename
        )

        count = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                if not data:
                    continue

                doc = data.get("doc", {})
                if removed_set is not None:
                    key = removed_key(doc, dataset_name)
                    if key is not None and key in removed_set:
                        continue
                biometric_profile = doc.get("biometric_profile", {}) or {}
                metric_key = biometric_profile.get("metric_key")
                target = data.get("target")
                filtered_resps = data.get(resps_field)

                hits = _proxies_for_sample(file_kind, dataset_name, metric_key)
                if not hits:
                    continue

                gt_val, pred_val = _gt_pred_values(file_kind, target, filtered_resps)
                if gt_val is None:
                    continue  # cannot categorize GT -> skip sample

                count += 1

                for proxy_key, spec in hits:
                    gt_cat = categorize(
                        gt_val, spec["cutoffs"], spec["labels"], spec["right_closed"]
                    )
                    pred_cat = categorize(
                        pred_val, spec["cutoffs"], spec["labels"], spec["right_closed"]
                    )
                    if gt_cat is None:
                        continue

                    for key in (proxy_key, f"{proxy_key} @ {dataset_name}"):
                        store = overall if key == proxy_key else by_dataset
                        grp = store.setdefault(key, _new_group())
                        grp["spec"] = spec
                        if pred_cat is None:
                            grp["n_fail"] += 1
                        else:
                            grp["gt"].append(gt_cat)
                            grp["pred"].append(pred_cat)

                    sample_values.append(
                        {
                            "proxy": proxy_key,
                            "dataset": dataset_name,
                            "metric_key": metric_key,
                            "gt_value": gt_val,
                            "pred_value": pred_val,
                            "gt_category": gt_cat,
                            "pred_category": pred_cat,
                            "image_file": (
                                os.path.basename(doc.get("image_file"))
                                if doc.get("image_file")
                                else None
                            ),
                            "slice_dim": doc.get("slice_dim"),
                            "slice_idx": doc.get("slice_idx"),
                        }
                    )

                if limit is not None and count >= limit:
                    break

    if not overall:
        # Remove any metrics from an earlier run. Returning without writing would
        # leave a stale file behind, and the report loads whatever it finds — so a
        # model that no longer yields eligible samples would keep reporting its
        # previous numbers as if they had just been recomputed.
        print(f"No CDA-eligible samples found in {parsed_files_dir}, skipping...")
        for stale_base in (SUMMARY_FILENAME_CDA_METRICS, SUMMARY_FILENAME_CDA_VALUES):
            stale = _metrics_filename(stale_base, limit, removed_samples_dir)
            stale_path = os.path.join(parsed_files_dir, stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
                print(f"  removed stale {stale}")
        return

    def _finalize(store):
        out = {}
        for key, grp in store.items():
            spec = grp["spec"]
            n_total = len(grp["gt"]) + grp["n_fail"]
            agreement = cal_clinical_agreement(
                grp["gt"],
                grp["pred"],
                spec["labels"],
                ordinal=spec["ordinal"],
                n_total=n_total,
            )
            out[key] = {"name": spec["name"], "ordinal": spec["ordinal"], **agreement}
        return out

    summary_metrics = {
        "overall": _finalize(overall),
        "by_dataset": _finalize(by_dataset),
    }

    values_filename = _metrics_filename(
        SUMMARY_FILENAME_CDA_VALUES, limit, removed_samples_dir
    )
    values_path = os.path.join(parsed_files_dir, values_filename)
    with open(values_path, "w") as f:
        json.dump(convert_numpy_to_python(sample_values), f, indent=2)
    print(f"Saved per-sample CDA values to {values_path}")

    metrics_filename = _metrics_filename(
        SUMMARY_FILENAME_CDA_METRICS, limit, removed_samples_dir
    )
    metrics_path = os.path.join(parsed_files_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(convert_numpy_to_python(summary_metrics), f, indent=2)
    print(f"Saved CDA summary metrics to {metrics_path}")


def _fmt(x, width=8, prec=4):
    """Format a possibly-None/NaN float for the fixed-width report."""
    if x is None:
        x = float("nan")
    return f"{x:<{width}.{prec}f}"


def _load_overall(metrics_file):
    with open(metrics_file, "r") as f:
        return json.load(f)


def _emit_leaderboards(rows, emit):
    """Emit a per-proxy leaderboard (one table per proxy) over ``rows``.

    ``rows`` is a list of ``(display_name, metrics_dict)``. Proxy order follows
    first appearance across models; model order follows ``rows`` order.
    """
    proxies = []
    for _, m in rows:
        for k in m.get("overall", {}):
            if k not in proxies:
                proxies.append(k)

    header = (
        f"{'Model':<26} | {'Acc':<8} | {'Kappa':<8} | {'wKappa':<8} | "
        f"{'Flip':<8} | {'AccCov':<8} | {'Nparsed':<8} | {'Ntotal':<8}"
    )
    sep = "-" * 102
    for proxy in proxies:
        emit(f"\n### Leaderboard (config order): {proxy}")
        emit(header)
        emit(sep)
        for display, m in rows:
            e = m.get("overall", {}).get(proxy)
            if e is None:
                continue
            emit(
                f"{display[:26]:<26} | "
                f"{_fmt(e.get('accuracy'))} | "
                f"{_fmt(e.get('cohen_kappa'))} | "
                f"{_fmt(e.get('weighted_kappa'))} | "
                f"{_fmt(e.get('flip_rate'))} | "
                f"{_fmt(e.get('accuracy_coverage_adjusted'))} | "
                f"{e.get('n_parsed', 0):<8} | "
                f"{e.get('n_total', 0):<8}"
            )


def print_model_summaries(
    task_dir,
    model_items,
    canonical,
    limit=None,
    skip_model_wo_parsed_files=False,
    removed_samples_dir=None,
    parsed_dirname=CDA_DEFAULT_PARSED_DIRNAME,
):
    """Print and save a cross-model CDA report over ``model_items``.

    ``model_items`` is the ordered ``(model_dir, display_name)`` list to report,
    and ``canonical`` says whether it came from a config. A canonical report is
    prefixed with per-proxy leaderboards and written to
    ``<task_dir>/summary_CDA_task_canonical.txt``; otherwise it covers whatever
    was on disk and is written to ``<task_dir>/summary_CDA_task.txt``. A
    removed-samples-filtered run adds a ``_filtered`` marker.
    """
    # Suffix order is "<_source><_filtered><_canonical><_limitN>", matching
    # analyze_CDA_renal_truelabel.py and the paths run_CDA_analysis.sh echoes.
    # The source marker leads because it names where the numbers came from.
    limit_sfx = "_limit" + str(limit) if limit is not None else ""
    canon_sfx = "_canonical" if canonical else ""
    output_filename = (
        f"summary_CDA_task{source_suffix(parsed_dirname)}"
        f"{filtered_suffix(removed_samples_dir)}{canon_sfx}{limit_sfx}.txt"
    )
    output_file_path = os.path.join(task_dir, output_filename)

    output_lines = []

    def emit(text):
        print(text)
        output_lines.append(text)

    emit("\n\n========== CLINICAL DECISION AGREEMENT (CDA) — MODEL SUMMARIES ==========\n")
    emit(
        "Acc = agreement on parsed samples; Kappa = Cohen's kappa; wKappa = "
        "quadratic-weighted kappa (ordinal proxies only);"
    )
    emit(
        "Flip = decision-flip rate (parsed); AccCov = coverage-adjusted accuracy "
        "(parse failures = disagreement)."
    )

    metrics_filename = _metrics_filename(
        SUMMARY_FILENAME_CDA_METRICS, limit, removed_samples_dir
    )

    # Load each model's metrics (in the chosen order).
    rows = []  # (display_name, metrics_dict)
    absent = []  # (display_name, reason) — reported, not just printed
    for model_dir, display in model_items:
        parsed_dir = os.path.join(model_dir, parsed_dirname)
        if skip_model_wo_parsed_files and not os.path.exists(parsed_dir):
            print(f"\nSkipping model directory (no {parsed_dirname} folder): {model_dir}")
            absent.append((display, f"no {parsed_dirname}/ folder"))
            continue
        metrics_file = os.path.join(parsed_dir, metrics_filename)
        if not os.path.exists(metrics_file):
            print(f"\nSkipping model (no CDA metrics file): {model_dir}")
            absent.append((display, "no CDA-eligible samples"))
            continue
        rows.append((display, _load_overall(metrics_file)))

    # A config-listed model that yields no metrics must be visible in the report.
    # Dropping it silently would make a partial evaluation set look complete.
    if absent:
        emit(f"\nNOT REPORTED ({len(absent)} of {len(model_items)} configured models):")
        for display, reason in absent:
            emit(f"  {display:<26} — {reason}")

    # Per-proxy leaderboards (only meaningful with an explicit config order).
    if canonical and rows:
        _emit_leaderboards(rows, emit)

    # Per-model detailed blocks. The label column must fit
    # "<proxy name> @ <dataset>": the renal proxy name alone is 42 chars, so a
    # narrower column truncated its KiTS23 and KiPA22 rows to the same string.
    label_w = 56
    header = (
        f"{'Proxy':<{label_w}} | {'Acc':<8} | {'Kappa':<8} | {'wKappa':<8} | "
        f"{'Flip':<8} | {'AccCov':<8} | {'Nparsed':<8} | {'Ntotal':<8}"
    )
    separator = "-" * len(header)
    for display, metrics in rows:
        emit(f"\nModel: {display}")
        for section in ("overall", "by_dataset"):
            section_metrics = metrics.get(section, {})
            if not section_metrics:
                continue
            emit(
                "\nOverall (per proxy):"
                if section == "overall"
                else "\nBy dataset (per proxy x dataset):"
            )
            emit(header)
            emit(separator)
            for key in sorted(section_metrics.keys()):
                m = section_metrics[key]
                emit(
                    f"{key[:label_w]:<{label_w}} | "
                    f"{_fmt(m.get('accuracy'))} | "
                    f"{_fmt(m.get('cohen_kappa'))} | "
                    f"{_fmt(m.get('weighted_kappa'))} | "
                    f"{_fmt(m.get('flip_rate'))} | "
                    f"{_fmt(m.get('accuracy_coverage_adjusted'))} | "
                    f"{m.get('n_parsed', 0):<8} | "
                    f"{m.get('n_total', 0):<8}"
                )
        emit("\n" + "=" * 100 + "\n")

    with open(output_file_path, "w") as output_file:
        output_file.write("\n".join(output_lines))
    print(f"\nSummary saved to {output_file_path}")


def _process_task_directory(
    task_dir,
    limit,
    skip_model_wo_parsed_files=False,
    config_yaml=None,
    removed_samples_dir=None,
    removed_samples_filename=None,
    parsed_dirname=CDA_DEFAULT_PARSED_DIRNAME,
):
    """Compute and report CDA over a task directory.

    The same resolved model list drives both steps, so the models whose metrics
    are (re)computed are exactly the models that appear in the report.
    """
    display_map = load_model_display_map(config_yaml)
    model_items = resolve_model_dirs(task_dir, display_map)
    for model_dir, _ in model_items:
        parsed_files_dir = os.path.join(model_dir, parsed_dirname)
        if skip_model_wo_parsed_files and not os.path.exists(parsed_files_dir):
            print(f"\nSkipping model directory (no {parsed_dirname} folder): {model_dir}")
            continue
        print(f"\nProcessing model directory: {model_dir}")
        process_parsed_file_in_model_folder(
            model_dir,
            limit,
            removed_samples_dir,
            removed_samples_filename,
            parsed_dirname,
        )
    print_model_summaries(
        task_dir,
        model_items,
        display_map is not None,
        limit,
        skip_model_wo_parsed_files,
        removed_samples_dir,
        parsed_dirname,
    )


def _process_single_model_directory(
    model_dir,
    limit,
    removed_samples_dir=None,
    removed_samples_filename=None,
    parsed_dirname=CDA_DEFAULT_PARSED_DIRNAME,
):
    print(f"\nProcessing model directory: {model_dir}")
    process_parsed_file_in_model_folder(
        model_dir, limit, removed_samples_dir, removed_samples_filename, parsed_dirname
    )


def main(**kwargs):
    """Process CDA for a task directory (all models + cross-model report) or a
    single model directory.

    Args:
        task_dir (str, optional): Path to task directory (mutually exclusive with
            model_dir).
        model_dir (str, optional): Path to a single model directory.
        limit (int, optional): Maximum number of samples to process per file.
        skip_model_wo_parsed_files (bool): Skip models without the selected
            parsed-source folder (task_dir mode only).
        parsed_dirname (str, optional): Parsed-results folder to read inside each
            model directory -- ``parsed`` or an ``llm-parsed*`` folder such as
            ``llm-parsed_gemma-4-31b``; also selects the prediction field.
            Defaults to ``parsed``.
        config_yaml (str, optional): CDA config listing the models to report; when
            given, the cross-model report is restricted/renamed/ordered accordingly.
            Must match the task: config-AD-CoT.yaml or config-TL-CoT.yaml.
        removed_samples_dir (str, optional): Root of per-dataset removed-samples
            JSONs. When given, excluded samples are skipped and every output
            filename gains a ``_filtered`` marker.
        removed_samples_filename (str, optional): Filename within each dataset
            subdirectory.

    Raises:
        ValueError: If neither task_dir nor model_dir is provided.
    """
    task_dir = kwargs.get("task_dir")
    model_dir = kwargs.get("model_dir")
    limit = kwargs.get("limit")
    skip_model_wo_parsed_files = kwargs.get("skip_model_wo_parsed_files", False)
    config_yaml = kwargs.get("config_yaml")
    removed_samples_dir = kwargs.get("removed_samples_dir")
    removed_samples_filename = kwargs.get("removed_samples_filename")
    parsed_dirname = kwargs.get("parsed_dirname") or CDA_DEFAULT_PARSED_DIRNAME

    if task_dir is not None:
        print(
            f"Using task_dir: {task_dir}\nModel directories within this folder will be looped over."
        )
        print(f"Parsed source: {parsed_dirname}/ "
              f"(prediction field: {parsed_source_field(parsed_dirname)})")
        _process_task_directory(
            task_dir,
            limit,
            skip_model_wo_parsed_files=skip_model_wo_parsed_files,
            config_yaml=config_yaml,
            removed_samples_dir=removed_samples_dir,
            removed_samples_filename=removed_samples_filename,
            parsed_dirname=parsed_dirname,
        )
    elif model_dir is not None:
        print(
            f"Using model_dir: {model_dir}\nProcessing all JSONL files within this directory."
        )
        print(f"Parsed source: {parsed_dirname}/ "
              f"(prediction field: {parsed_source_field(parsed_dirname)})")
        _process_single_model_directory(
            model_dir,
            limit,
            removed_samples_dir,
            removed_samples_filename,
            parsed_dirname,
        )
    else:
        raise ValueError("Either 'task_dir' or 'model_dir' must be provided.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Clinical Decision Agreement (CDA) proxy metrics."
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        help="Path to the task directory containing model result folders.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to a specific model directory containing a parsed-source folder.",
    )
    parser.add_argument(
        "--parsed_dirname",
        default=CDA_DEFAULT_PARSED_DIRNAME,
        type=validate_parsed_dirname,
        help="Which parsed-results folder inside each model directory to read: "
        "'parsed' (regex parser), or any 'llm-parsed*' folder written by an "
        f"LLM-judge re-parse (e.g. '{CDA_LLM_PARSED_DIRNAME}'). Matched by prefix, "
        "since the judge writes one folder per judge model. The prefix also "
        "selects the row field holding the prediction, so a source and its field "
        "cannot be mixed up. Per-model outputs are written back into the folder "
        "read; task-level reports gain a source marker "
        "(e.g. '_llm-parsed-gemma-4-31b').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples processed per JSONL file (default: all).",
    )
    parser.add_argument(
        "--skip_model_wo_parsed_files",
        action="store_true",
        help="Skip model directories without the selected --parsed_dirname folder. "
        "Only valid with --task_dir.",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="CDA config listing the models to report: config-AD-CoT.yaml for an "
        "A/D task dir, config-TL-CoT.yaml for a T/L one. When given, the "
        "cross-model report is restricted to those models, labelled, and ordered "
        "as in the config, with per-proxy leaderboards.",
    )
    parser.add_argument(
        "--removed_samples_dir",
        type=str,
        default=None,
        help="Root directory of per-dataset removed-samples JSONs (e.g. "
        "Data/Datasets). Excludes the multi-cluster T/L slices the benchmark's "
        "own summarize_TL_task.py excludes, so CDA scores the same sample set. "
        "Output filenames gain a '_filtered' marker.",
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
    if args.skip_model_wo_parsed_files and args.task_dir is None:
        parser.error("--skip_model_wo_parsed_files can only be used with --task_dir")
    if args.config_yaml is not None and args.task_dir is None:
        parser.error("--config_yaml can only be used with --task_dir")
    return args


if __name__ == "__main__":
    args_dict = vars(parse_args())
    main(**args_dict)
