"""Seed the MedVision-V0 demo gallery (one-time; run in the MedVision conda env).

For each sampled MedVision-V0-7B benchmark record this writes:
  <out_dir>/<case_id>/input.png  — the slice resized to 512x512 (the model feed size)
and appends to <out_dir>/examples.json a record with the VERBATIM prompt (`input`),
GT overlay coords (relative [0,1]), and the GT answer (`target`).

The demo app consumes only these assets — it never imports medvision_bm.

Usage:
  python script/visualization/export_demo_gallery.py \
    --det_dir Results/MedVision-detect-v2/MedVision-V0-7B \
    --tl_dir  Results/MedVision-TL-v2-CoT/MedVision-V0-7B \
    --ad_dir  Results/MedVision-AD-v2-CoT/MedVision-V0-7B \
    --out_dir /mnt/vincent-pvc-rwm/Github/medvision-v0-demo/examples \
    --per_task 3
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import viz_tl_axes as TL            # noqa: E402
import viz_ad_landmarks as AD       # noqa: E402

from medvision_bm.utils.configs import SEED  # noqa: E402

FEED = 512


def _rel_from_array(pt, H, W):
    d0, d1 = pt
    return [d1 / W, 1.0 - d0 / H]


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _main_jsonls(model_dir):
    return [f for f in sorted(glob.glob(os.path.join(model_dir, "*.jsonl")))
            if not f.endswith(("_proc_acc.jsonl", "_eq_acc.jsonl", "_filtered.jsonl"))]


def _save_input_png(doc, case_dir):
    psz, img = TL.load_nifti_slice(doc["image_file"], doc["slice_dim"], doc["slice_idx"], doc)
    H, W = img.shape
    pil = Image.fromarray(img.astype(np.uint8), mode="L").convert("RGB")
    pil = pil.resize((FEED, FEED), Image.BILINEAR)
    os.makedirs(case_dir, exist_ok=True)
    pil.save(os.path.join(case_dir, "input.png"))
    return H, W


def _select(model_dir, per_task, seed):
    rng = random.Random(seed)
    samples = []
    for jl in _main_jsonls(model_dir):
        samples.extend(_load_jsonl(jl))
    return rng.sample(samples, min(per_task, len(samples)))


def _gt_detection(sample):
    gt = sample["target"]
    box = json.loads(gt) if isinstance(gt, str) else gt
    box = [float(v) for v in box]
    return {"box": box}, box


def _gt_tl(sample, doc, H, W):
    maj, minr = TL._extract_gt_axis_pts(doc)
    gt = {"major": [_rel_from_array(maj[0], H, W), _rel_from_array(maj[1], H, W)],
          "minor": [_rel_from_array(minr[0], H, W), _rel_from_array(minr[1], H, W)]}
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
    ap.add_argument("--per_task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    jobs = [("detection", args.det_dir), ("tl", args.tl_dir), ("ad", args.ad_dir)]
    for task, model_dir in jobs:
        if not model_dir:
            continue
        for i, sample in enumerate(_select(model_dir, args.per_task, args.seed)):
            doc = sample["doc"]
            case_id = f"{task}_{i:02d}"
            case_dir = os.path.join(args.out_dir, case_id)
            H, W = _save_input_png(doc, case_dir)
            ad_kind = ""
            if task == "detection":
                gt, gt_answer = _gt_detection(sample)
                label = str(doc.get("label"))
            elif task == "tl":
                gt, gt_answer = _gt_tl(sample, doc, H, W)
                label = str(doc.get("label"))
            else:
                gt, gt_answer, ad_kind = _gt_ad(sample, doc, H, W)
                label = "biometric"
            records.append({
                "case_id": case_id, "task": task, "ad_kind": ad_kind,
                "label": label, "title": f"{task} example {i}",
                "modality": doc.get("dataset_name", ""),
                "input": f"{case_id}/input.png",
                "prompt": sample["input"],
                "gt": gt, "gt_answer": gt_answer,
            })
            print(f"  wrote {case_id}")

    with open(os.path.join(args.out_dir, "examples.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"Done: {len(records)} cases -> {args.out_dir}/examples.json")


if __name__ == "__main__":
    main()
