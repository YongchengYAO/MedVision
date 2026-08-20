#!/usr/bin/env python3
"""Sample landmark/biometry figure PNGs for the project webpage's annotation-preview widget.
*********************************
https://medvision-vlm.github.io/
*********************************

Scans <data_dir>/<dataset>/ for the two pre-rendered landmark-figure folder shapes documented
in the medvision_ds project memory, classifies each dataset into a group by folder shape alone
(no hardcoded dataset list, so a newly copied dataset is picked up automatically), takes a
seeded random sample from EACH figure folder, copies the sampled PNGs into
<page_dir>/figure/annot-preview/<group>/<dataset>/, and writes
<page_dir>/static/js/annot-preview-data.js defining the window.MEDVISION_ANNOT_PREVIEW global
the widget reads.

The sample limit (--n_per_folder) is PER FIGURE FOLDER, not per dataset: a dataset with several
folders gets the limit applied to EACH of them independently, then all of them are concatenated
into that dataset's manifest entry. Pooling them first (sample N from the union) would let
whichever folder happens to have the most files crowd out the rest -- e.g. MSWAL's Label7 folder
(3,318 PNGs) would dominate a pooled sample over its Label5 folder (964 PNGs) purely by file
count, not by anything meaningful. Per-folder sampling guarantees every folder is represented.

Two groups, told apart structurally:
  AD (anatomical landmark)     — "Landmarks-fig/" holding either flat <case>_<plane>_<slice>.png
                                  (PDDCA/VerSe/AFIDs), or -- predating that convention -- one
                                  subdirectory per biometric measurement, each holding its own
                                  flat PNGs (FeTA24: "Landmarks-fig/<Measurement-Name>/<case>_
                                  ..._sliceN.png"). Either way this is the exact-slice figure set,
                                  NOT "Landmarks-fig-w-projection/": every point drawn is on the
                                  slice it's drawn over, which projection is not (see project
                                  memory "Landmark figures: exact-slice vs projected") --
                                  projection is for review, not a preview sample.
  TL (mask-derived tumor/lesion) — one or more "Landmarks-<tag>-fig-v<version>/<case>/<plane>_
                                  <slice>.png" folders (nested per case): multiple mask labels
                                  (MSWAL: 5), or modality-scoped variants (DEEP-PSMA: FDG/PSMA).
                                  Some datasets (BraTS24, MSD, HNTSMRG24) hold several SUBSETS
                                  under the dataset root (e.g. BraTS24/BraTS24-GLI,
                                  BraTS24/BraTS24-MEN-RT, ...), each with its own such folders one
                                  level deeper -- every subset is scanned and its samples folded
                                  into that dataset's single manifest entry, tagged with the
                                  subset's suffix (the dataset name stripped off the front) so
                                  cases can't collide across subsets or labels. When the SAME
                                  (subset, label) has more than one version folder on disk (e.g.
                                  KiPA22's Label4 at both v1.1.0 and v1.1.1, or BraTS24-GLI's
                                  Label1 at v1.1.0/v1.1.1 -- a correction leaves the old version's
                                  folder in place, never rewriting it), only the newest version is
                                  sampled.

Detection figures are out of scope entirely -- this script never looks at a dataset's
Detection-related output, only its two landmark-figure shapes.

Example
-------
    python script/visualization/export_annotation_preview.py \
        --page_dir /mnt/vincent-pvc-rwm/Github/medvision-vlm.github.io
"""
import argparse
import hashlib
import glob
import json
import os
import random
import re
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
DEFAULT_DATA_DIR = os.path.join(_REPO_DIR, "Data", "Datasets")

_TL_DIR_RE = re.compile(r"^Landmarks-(.+)-fig-(v[\d.]+)$")


def _vtuple(version):
    return tuple(int(p) for p in version.lstrip("v").split("."))


def _ad_pairs(ad_fig):
    """[(src, dest_name), ...] for an AD dataset's Landmarks-fig/ -- flat (PDDCA/VerSe/AFIDs)
    or one subdirectory per biometric measurement (FeTA24, predates the flat convention)."""
    flat = sorted(glob.glob(os.path.join(ad_fig, "*.png")))
    if flat:
        return [(f, os.path.basename(f)) for f in flat]
    pairs = []
    for measure_dir in sorted(glob.glob(os.path.join(ad_fig, "*"))):
        if not os.path.isdir(measure_dir):
            continue
        measure = os.path.basename(measure_dir)
        for f in sorted(glob.glob(os.path.join(measure_dir, "*.png"))):
            pairs.append((f, f"{measure}_{os.path.basename(f)}"))
    return pairs


def _tl_dir_candidates(ds_dir, dataset):
    """[(subset_suffix_or_None, folder_name, folder_path), ...] -- every Landmarks-*-fig-v*
    folder directly under ds_dir, PLUS one level deeper under any subset folder (BraTS24/
    BraTS24-GLI, MSD/MSD-BrainTumour, HNTSMRG24/HNTSMRG24-midRT, ...). subset_suffix is the
    subset dirname with the dataset's own name-prefix stripped (e.g. "GLI", "midRT"), or None
    for folders directly under the dataset root."""
    out = []
    for sub in sorted(os.listdir(ds_dir)):
        if _TL_DIR_RE.match(sub):
            out.append((None, sub, os.path.join(ds_dir, sub)))

    for entry in sorted(os.listdir(ds_dir)):
        entry_path = os.path.join(ds_dir, entry)
        if not os.path.isdir(entry_path) or _TL_DIR_RE.match(entry):
            continue
        suffix = entry[len(dataset) + 1:] if entry.startswith(dataset + "-") else entry
        for sub in sorted(os.listdir(entry_path)):
            if _TL_DIR_RE.match(sub):
                out.append((suffix, sub, os.path.join(entry_path, sub)))
    return out


def _discover(data_dir):
    """{"AD": {dataset: {folder: [(src_path, dest_name), ...]}}, "TL": {...}}, sorted, unsampled.

    Folder-scoped (not flattened) so sampling can apply --n_per_folder to each figure
    folder independently. AD datasets always resolve to a single synthetic folder key
    ("Landmarks-fig"); TL datasets may have several (one per _TL_DIR_RE match, at the
    dataset root or one level deeper under a subset folder)."""
    ad, tl = {}, {}
    for ds_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(ds_dir):
            continue
        dataset = os.path.basename(ds_dir)

        ad_fig = os.path.join(ds_dir, "Landmarks-fig")
        if os.path.isdir(ad_fig):
            pairs = _ad_pairs(ad_fig)
            if pairs:
                ad.setdefault(dataset, {})["Landmarks-fig"] = pairs

        # Group candidate TL folders by (subset, label) identity (everything but the trailing
        # version) and keep only the newest version of each. A correction never rewrites a
        # published annotation file in place (see project CLAUDE.md's annotation-identity rule)
        # -- it adds a new version ALONGSIDE the old one, so e.g. KiPA22 has both
        # Landmarks-Label4-fig-v1.1.0 (superseded) and -v1.1.1 (current) on disk. Sampling both
        # would silently mix a stale annotation version into what's supposed to be a preview of
        # the current release. Subset scoping matters here too: BraTS24-GLI's "Label1" and
        # BraTS24-PED's "Label1" are unrelated labels that happen to share a name.
        newest = {}  # (subset, label_tag) -> (version_tuple, folder_name, folder_path, version)
        for subset, sub, path in _tl_dir_candidates(ds_dir, dataset):
            m = _TL_DIR_RE.match(sub)
            label_tag, version = m.group(1), m.group(2)
            vt = _vtuple(version)
            key = (subset, label_tag)
            if key not in newest or vt > newest[key][0]:
                newest[key] = (vt, sub, path, version)

        for subset, label_tag in sorted(newest, key=lambda k: (k[0] or "", k[1])):
            _, sub, path, version = newest[(subset, label_tag)]
            # e.g. "Label1-v1.2.0", "FDG-Label1-v1.2.0", "GLI-Label1-v1.1.1", "midRT-Label1-v1.1.1"
            tag = f"{subset}-{label_tag}-{version}" if subset else f"{label_tag}-{version}"
            pairs = []
            for case_dir in sorted(glob.glob(os.path.join(path, "*"))):
                if not os.path.isdir(case_dir):
                    continue
                case = os.path.basename(case_dir)
                for f in sorted(glob.glob(os.path.join(case_dir, "*.png"))):
                    dest_name = f"{tag}_{case}_{os.path.basename(f)}"
                    pairs.append((f, dest_name))
            if pairs:
                folder_key = f"{subset}/{sub}" if subset else sub
                tl.setdefault(dataset, {})[folder_key] = pairs
    return ad, tl


def _sample_and_copy(groups, page_dir, group_key, n_per_folder, seed):
    out = {}
    for dataset in sorted(groups):
        folders = groups[dataset]
        # Keep each dataset independent: adding or removing another dataset must not
        # advance its random stream and reshuffle this preview.
        dataset_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{group_key}:{dataset}".encode()).digest()[:8], "big"
        )
        dataset_rng = random.Random(dataset_seed)
        chosen_all = []
        folder_report = []
        for folder in sorted(folders):
            pairs = folders[folder]
            k = min(n_per_folder, len(pairs))
            chosen_all.extend(sorted(dataset_rng.sample(pairs, k), key=lambda p: p[1]))
            folder_report.append(f"{folder}={k}/{len(pairs)}")

        dest_dir = os.path.join(page_dir, "figure", "annot-preview", group_key, dataset)
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        rel_paths = []
        for src, dest_name in chosen_all:
            shutil.copy2(src, os.path.join(dest_dir, dest_name))
            rel_paths.append(f"figure/annot-preview/{group_key}/{dataset}/{dest_name}")
        out[dataset] = rel_paths
        print(
            f"[annot-preview] {group_key}/{dataset}: " + ", ".join(folder_report)
            + f" -> {len(rel_paths)} total"
        )
    return out


def _emit_js(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "// Auto-generated by script/visualization/export_annotation_preview.py — DO NOT EDIT.\n"
        "// Schema: window.MEDVISION_ANNOT_PREVIEW = {\n"
        "//   seed, n_per_folder,\n"
        "//   AD:{ <dataset>: [relative png path, ...] },  -- anatomical-landmark datasets\n"
        "//   TL:{ <dataset>: [relative png path, ...] }   -- mask-derived tumor/lesion datasets\n"
        "// }\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        fh.write("window.MEDVISION_ANNOT_PREVIEW = ")
        json.dump(manifest, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")


def main():
    ap = argparse.ArgumentParser(
        description="Sample landmark/biometry figure PNGs (A/D + T/L) for the webpage's "
        "annotation-preview widget."
    )
    ap.add_argument(
        "--data_dir", default=DEFAULT_DATA_DIR,
        help="Root holding <dataset>/Landmarks-fig or Landmarks-*-fig-v* folders "
        f"(default {DEFAULT_DATA_DIR}).",
    )
    ap.add_argument(
        "--page_dir", required=True, help="Project page repo (medvision-vlm.github.io)."
    )
    ap.add_argument(
        "--n_per_folder", type=int, default=20,
        help="Figures sampled from EACH figure folder (default 20). A dataset with several "
        "folders (multiple mask labels, or modality-scoped variants) gets this limit applied "
        "to every one of them independently, not pooled across the dataset.",
    )
    ap.add_argument(
        "--seed", type=int, default=1234, help="Random seed for sample selection (default 1234)."
    )
    ap.add_argument(
        "--out", default=None,
        help="Manifest JS path (default <page_dir>/static/js/annot-preview-data.js).",
    )
    args = ap.parse_args()

    ad_groups, tl_groups = _discover(args.data_dir)
    if not ad_groups and not tl_groups:
        sys.exit(f"[annot-preview] no landmark figure folders found under {args.data_dir}")
    print(
        f"[annot-preview] discovered AD datasets: {sorted(ad_groups)} | "
        f"TL datasets: {sorted(tl_groups)}"
    )

    manifest = {
        "seed": args.seed,
        "n_per_folder": args.n_per_folder,
        "AD": _sample_and_copy(ad_groups, args.page_dir, "AD", args.n_per_folder, args.seed),
        "TL": _sample_and_copy(tl_groups, args.page_dir, "TL", args.n_per_folder, args.seed),
    }

    out_path = args.out or os.path.join(args.page_dir, "static", "js", "annot-preview-data.js")
    _emit_js(out_path, manifest)
    total = sum(len(v) for v in manifest["AD"].values()) + sum(
        len(v) for v in manifest["TL"].values()
    )
    print(f"[annot-preview] wrote {out_path} ({total} figures total)")

    # A dataset sampled by a PRIOR run but absent from THIS run's discovery (removed from
    # data_dir, or no longer has a landmark-fig folder) never gets re-visited by
    # _sample_and_copy above, so its old figure dir would otherwise linger un-referenced.
    for group_key, sampled in (("AD", manifest["AD"]), ("TL", manifest["TL"])):
        group_dir = os.path.join(args.page_dir, "figure", "annot-preview", group_key)
        if not os.path.isdir(group_dir):
            continue
        for existing in sorted(os.listdir(group_dir)):
            if existing not in sampled:
                shutil.rmtree(os.path.join(group_dir, existing))
                print(f"[annot-preview] removed orphaned {group_key}/{existing} (no longer discovered)")


if __name__ == "__main__":
    main()
