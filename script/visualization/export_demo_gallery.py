"""Seed the MedVision-V0 demo gallery (one-time; run in the MedVision conda env).

For each sampled MedVision-V0-7B benchmark record this writes:
  <out_dir>/<case_id>/input.png  — the slice resized to 512x512 (the model feed size)
and appends to <out_dir>/examples.json a record with the VERBATIM prompt (`prompt`),
GT overlay coords (`gt`), the GT answer (`gt_answer`), and the info-row fields
`target` (structure / biometric name) + `modality` (imaging modality) parsed from the
prompt — mirroring export_webpage_cases._target_modality so the demo's case info matches
the project-page case viewer. `label` keeps the raw class id for provenance.

The demo app consumes only these assets — it never imports medvision_bm.

Usage:
  python script/visualization/export_demo_gallery.py \
    --det_dir Results/MedVision-detect-v2/MedVision-V0-7B \
    --tl_dir  Results/MedVision-TL-v2-CoT/MedVision-V0-7B \
    --ad_dir  Results/MedVision-AD-v2-CoT/MedVision-V0-7B \
    --out_dir /mnt/vincent-pvc-rwm/Github/medvision-v0-demo/examples \
    --per_subtask 2

Selection picks `--per_subtask` cases at random (seeded) from EACH subtask — one main results
JSONL per subtask (dataset + task-config + plane) — so every subtask appears in the gallery.
"""

import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import viz_ad_landmarks as AD  # noqa: E402
import viz_tl_axes as TL  # noqa: E402

# Reuse the TL summarizer's removed-sample logic so the demo gallery excludes the SAME
# v1.0.0 -> v1.1.0 multi-cluster T/L samples the benchmark drops (matches export_webpage_cases).
from medvision_bm.benchmark.summarize_TL_task import (  # noqa: E402
    _build_removed_set,
    _relative_image_file,
)
from medvision_bm.utils.configs import label_map_rename  # noqa: E402

FEED = 512

# Sample-selection seed. Matches export_webpage_cases.sh / export_webpage_cases.py (default
# 1234) so the demo gallery is drawn with the SAME seed as the project-page case viewer,
# rather than the benchmark-wide configs.SEED. Override via --seed. NOTE: identical seed does
# NOT yield identical cases — the case viewer samples per-dataset while this samples per-task,
# so the RNG call sequence (and selection) differs.
WEBPAGE_CASE_SEED = 1234


# ── Target / modality for the info-row (mirrors export_webpage_cases._target_modality) ──
# Both are parsed from the VERBATIM prompt — the structure name and imaging modality the
# model is actually told — NOT from the raw class id or the dataset name. label_map_rename
# normalizes target variants exactly as the project-page case viewer does.


def _modality(inp):
    m = re.search(
        r"Given the input medical image:\s*(.+?)\s*,\s*(?:estimate|return|measure)",
        inp,
        re.DOTALL,
    )
    return m.group(1).strip() if m else "medical image"


def _short_modality(modality):
    abbr = re.findall(r"\(([A-Za-z0-9/ +-]{2,8})\)", modality)
    return abbr[-1] if abbr else re.sub(r"\s+scan$", "", modality)


def _target_modality(task, ad_kind, prompt, label_id):
    mod = _short_modality(_modality(prompt))
    if task == "detection":
        m = re.search(r"bounding box for the\s*(.+?)\s*[\.\n]", prompt, re.DOTALL)
        tgt = m.group(1).strip() if m else f"label {label_id}"
    elif task == "tl":
        m = re.search(
            r"enclosing the\s*(.+?)\s*,\s*in (?:mm|millimeters)", prompt, re.DOTALL
        )
        tgt = m.group(1).strip() if m else "tumor/lesion"
    elif ad_kind == "distance":
        nm = re.search(
            r"estimate the distance of\s*(.+?)\s*in (?:mm|millimeters)",
            prompt,
            re.DOTALL,
        )
        if nm:
            tgt = nm.group(1).strip()
        else:
            lm = re.findall(r"\(landmark \d+\)\s*([^,()\n.]+)", prompt)
            names = [n.strip() for n in lm[:2]]
            tgt = (
                f"distance: {names[0]} → {names[1]}" if len(names) == 2 else "distance"
            )
    else:  # angle
        ln = re.findall(
            r"\(line \d+\)\s*the line connecting\s*(.+?)\s*and\s*(.+?)\s*[,.\n]",
            prompt,
            re.DOTALL,
        )
        tgt = (
            f"angle: {ln[0][0]}–{ln[0][1]} / {ln[1][0]}–{ln[1][1]}"
            if len(ln) == 2
            else "angle"
        )
    return label_map_rename.get(tgt, tgt), mod


def _rel_from_array(pt, H, W):
    d0, d1 = pt
    return [d1 / W, 1.0 - d0 / H]


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _main_jsonls(model_dir):
    return [
        f
        for f in sorted(glob.glob(os.path.join(model_dir, "*.jsonl")))
        if not f.endswith(("_proc_acc.jsonl", "_eq_acc.jsonl", "_filtered.jsonl"))
    ]


def _dataset_of(jsonl_path):
    """Dataset name from a sample JSONL filename (samples_<dataset>_...), as the TL summarizer
    keys its removed-sample files. 'unknown' if the pattern doesn't match."""
    m = re.search(r"samples_([^_]+)_", os.path.basename(jsonl_path))
    return m.group(1) if m else "unknown"


def _save_input_png(doc, case_dir):
    psz, img = TL.load_nifti_slice(
        doc["image_file"], doc["slice_dim"], doc["slice_idx"], doc
    )
    H, W = img.shape
    pil = Image.fromarray(img.astype(np.uint8), mode="L").convert("RGB")
    pil = pil.resize((FEED, FEED), Image.BILINEAR)
    os.makedirs(case_dir, exist_ok=True)
    pil.save(os.path.join(case_dir, "input.png"))
    # psz = [px_dim0, px_dim1] = [height-axis, width-axis] mm (array space).
    return H, W, psz


def _save_mask_png(doc, case_dir):
    """Save the GT segmentation mask for a TL case as a binary (H, W) PNG (white = label),
    the SAME mask the project-page TL figure contours (export_webpage_cases._overlay_tl /
    viz_tl_responses._draw_tl_overlay_on_ax). The demo renders it as the green GT contour;
    without it the TL overlay has no GT mask outline. Returns "mask.png" or None (no mask).

    Saved at the slice's native (H, W); the demo's figure renderer nearest-resizes it to the
    512x512 input on load, so it aligns with the resized input.png and the relative GT axes."""
    mf = doc.get("mask_file")
    if not mf or not os.path.exists(mf):
        return None
    label = doc.get("label", 1)
    _, mask_2d = TL._load_nifti_2d(mf, doc["slice_dim"], doc["slice_idx"])
    mask_bin = mask_2d == label
    if not mask_bin.any():
        return None
    Image.fromarray((mask_bin.astype(np.uint8) * 255), mode="L").save(
        os.path.join(case_dir, "mask.png")
    )
    return "mask.png"


def _select(
    model_dir,
    per_subtask,
    seed,
    task=None,
    removed_samples_dir=None,
    removed_samples_filename=None,
):
    """Randomly pick `per_subtask` samples from EACH subtask (seeded) and concatenate, so every
    subtask is represented in the gallery — rather than pooling the whole task and sampling a
    flat total, which lets large subtasks crowd out small ones (some never appear).

    A *subtask* is one main results JSONL: a specific dataset + task-config + plane, e.g.
    `samples_BraTS24_..._Task06_Axial` or `samples_Ceph-Biometrics-400_..._Angle`. This mirrors
    the per-group selection in export_webpage_cases.py (which samples within each dataset group).

    Seeded + deterministic: subtasks are visited in sorted filename order (via _main_jsonls) and
    each subtask's samples are sorted by (taskID, doc_id) before sampling, so the same seed always
    yields the same gallery. TL only: when removed_samples_dir is given, the v1.0.0 -> v1.1.0
    multi-cluster T/L samples the benchmark drops (same exclusion as summarize_TL_task.py /
    export_webpage_cases.py) are removed BEFORE sampling. Off when removed_samples_dir is None."""
    rng = random.Random(seed)
    filter_tl = task == "tl" and bool(removed_samples_dir)
    removed_cache = {}  # dataset -> frozenset | None (None = no removed-samples file)
    selected, n_removed = [], 0
    for jl in _main_jsonls(model_dir):  # one JSONL = one subtask
        ds = _dataset_of(jl)
        removed_set = None
        if filter_tl:
            if ds not in removed_cache:
                jp = os.path.join(removed_samples_dir, ds, removed_samples_filename)
                removed_cache[ds] = (
                    _build_removed_set(jp) if os.path.exists(jp) else None
                )
            removed_set = removed_cache[ds]
        pool = []
        for s in _load_jsonl(jl):
            if removed_set is not None:
                doc = s.get("doc", {})
                rkey = (
                    _relative_image_file(doc.get("image_file", ""), ds),
                    doc.get("slice_dim"),
                    doc.get("slice_idx"),
                    int(doc.get("taskID")),
                )
                if rkey in removed_set:
                    n_removed += 1
                    continue
            pool.append(s)
        # deterministic order within the subtask before sampling
        pool.sort(key=lambda s: (int(s["doc"].get("taskID", 0)), s["doc_id"]))
        selected.extend(rng.sample(pool, min(per_subtask, len(pool))))
    if n_removed:
        print(f"  [tl] excluded {n_removed} removed-sample(s) before selection")
    return selected


def _gt_detection(sample):
    gt = sample["target"]
    box = json.loads(gt) if isinstance(gt, str) else gt
    box = [float(v) for v in box]
    return {"box": box}, box


def _gt_tl(sample, doc, H, W):
    maj, minr = TL._extract_gt_axis_pts(doc)
    gt = {
        "major": [_rel_from_array(maj[0], H, W), _rel_from_array(maj[1], H, W)],
        "minor": [_rel_from_array(minr[0], H, W), _rel_from_array(minr[1], H, W)],
    }
    ans = sample["target"]
    ans = json.loads(ans) if isinstance(ans, str) else ans
    return gt, [float(ans[0]), float(ans[1])]


def _gt_ad(sample, doc, H, W):
    # AD._get_gt_coords returns (result_dict, err); the dict keys the array-space
    # [row, col] landmarks by name (distance: p1/p2; angle: l1p1/l1p2/l2p1/l2p2).
    result, err = AD._get_gt_coords(doc)
    if result is None:
        raise RuntimeError(f"GT coords unavailable: {err}")
    mtype = result["metric_type"]
    if mtype == "distance":
        pts = [result["p1"], result["p2"]]
    else:  # angle
        pts = [result["l1p1"], result["l1p2"], result["l2p1"], result["l2p2"]]
    gt = {"kind": mtype, "pts": [_rel_from_array(p, H, W) for p in pts]}
    ans = sample["target"]
    ans = json.loads(ans) if isinstance(ans, str) else ans
    scalar = float(ans[0] if isinstance(ans, (list, tuple)) else ans)
    return gt, scalar, mtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_dir")
    ap.add_argument("--tl_dir")
    ap.add_argument("--ad_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--per_subtask",
        type=int,
        default=2,
        help="Random samples drawn from EACH subtask (one main results JSONL = one "
        "dataset/task-config/plane). Total per task = per_subtask x (number of subtasks).",
    )
    ap.add_argument("--seed", type=int, default=WEBPAGE_CASE_SEED)
    # TL-only sample filtering, mirroring summarize_TL_task.py / export_webpage_cases.py. When
    # given, T/L samples listed in <removed_samples_dir>/<dataset>/<removed_samples_filename>
    # are excluded before selection. Off by default (no dir -> no filtering).
    ap.add_argument(
        "--removed_samples_dir",
        default=None,
        help="Root dir with per-dataset removed-samples JSON (e.g. .../Data/Datasets). "
        "Enables TL removed-sample exclusion when set.",
    )
    ap.add_argument(
        "--removed_samples_filename",
        default="multi_cluster_samples_v1.0.0_to_v1.1.0.json",
        help="Removed-samples JSON filename within each dataset subdirectory.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    jobs = [("detection", args.det_dir), ("tl", args.tl_dir), ("ad", args.ad_dir)]
    for task, model_dir in jobs:
        if not model_dir:
            continue
        for i, sample in enumerate(
            _select(
                model_dir,
                args.per_subtask,
                args.seed,
                task=task,
                removed_samples_dir=args.removed_samples_dir,
                removed_samples_filename=args.removed_samples_filename,
            )
        ):
            doc = sample["doc"]
            case_id = f"{task}_{i:02d}"
            case_dir = os.path.join(args.out_dir, case_id)
            H, W, psz = _save_input_png(doc, case_dir)
            # psz = [px_dim0, px_dim1] = [height, width]; the demo figure renderer
            # consumes spacing_w (width-axis) and spacing_h (height-axis).
            spacing_h, spacing_w = float(psz[0]), float(psz[1])
            slice_dim = int(doc["slice_dim"])
            ad_kind = ""
            mask_rel = None
            if task == "detection":
                gt, gt_answer = _gt_detection(sample)
                label = str(doc.get("label"))
            elif task == "tl":
                gt, gt_answer = _gt_tl(sample, doc, H, W)
                label = str(doc.get("label"))
                mask_rel = _save_mask_png(doc, case_dir)  # GT mask -> green contour
            else:
                gt, gt_answer, ad_kind = _gt_ad(sample, doc, H, W)
                label = "biometric"
            target, modality = _target_modality(task, ad_kind, sample["input"], label)
            records.append(
                {
                    "case_id": case_id,
                    "task": task,
                    "ad_kind": ad_kind,
                    "label": label,
                    "target": target,
                    "title": f"{task} example {i}",
                    "modality": modality,
                    "input": f"{case_id}/input.png",
                    "mask": f"{case_id}/{mask_rel}" if mask_rel else None,
                    "prompt": sample["input"],
                    "gt": gt,
                    "gt_answer": gt_answer,
                    "spacing_w": spacing_w,
                    "spacing_h": spacing_h,
                    "slice_dim": slice_dim,
                }
            )
            print(f"  wrote {case_id}")

    with open(os.path.join(args.out_dir, "examples.json"), "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Done: {len(records)} cases -> {args.out_dir}/examples.json")


if __name__ == "__main__":
    main()
