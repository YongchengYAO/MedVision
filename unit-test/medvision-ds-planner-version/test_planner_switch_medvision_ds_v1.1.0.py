"""
Demonstration: MedVision_PLANNER_VERSION fixes
==============================================

Two recent fixes in MedVision.py (YongchengYAO/MedVision Huggingface dataset repo)
changed how the loader handles ``MedVision_PLANNER_VERSION``:

  1. ``[src] fix: include planner_version in builder cache fingerprint``
     - Before: switching ``MedVision_PLANNER_VERSION`` between 1.0.0 and
       1.1.0 inside the same Python process collided in the same Hugging
       Face builder cache folder, raising::

           datasets.exceptions.NonMatchingSplitsSizesError:
             [{'expected': SplitInfo(...num_examples=303...),
               'recorded': SplitInfo(...num_examples=391...)}]

     - After: each planner version resolves to its own cache folder (the
       config name is suffixed with ``-planner_version=<v>``), so the
       two never share metadata.

  2. ``[src] feat: skip raw redownload when cached version >= requested planner``
     - Before: switching from 1.1.0 to 1.0.0 triggered a full re-download
       of the raw NIfTI images/masks plus a per-file reorientation pass,
       because the trigger condition was ``cached != requested``.
     - After: re-download only fires on a true upgrade
       (``cached < requested``). Downgrades reuse the on-disk data —
       annotation zips are backward compatible, so a v1.1.0 zip can
       satisfy v1.0.0 planner requests.

How the script works
--------------------
Two consecutive ``load_dataset()`` calls are made:

  Step 1. ``MedVision_PLANNER_VERSION=1.1.0`` (latest annotations)
  Step 2. ``MedVision_PLANNER_VERSION=1.0.0`` (original annotations)

After step 1, we snapshot the mtimes of the raw NIfTI image files in
``MedVision_DATA_DIR/Datasets/<dataset>/Images/``. We run step 2 and
then verify:

  * Both calls succeeded (otherwise commit 1 is missing and step 2
    would have raised ``NonMatchingSplitsSizesError``).
  * The two calls produced the expected, different sample counts
    (303 for 1.1.0, 391 for 1.0.0).
  * The two calls landed in distinct HF builder cache folders.
  * NIfTI mtimes did NOT change between step 1 and step 2 — proving
    step 2 did not trigger raw redownload + reorientation (commit 2).

Works for both new and existing users
-------------------------------------
* **New user** (no MedVision data on disk yet): step 1 performs the
  initial download, creating NIfTI files. Step 2 must not modify them.
* **Existing user** (raw data already present, the local download
  tracker may be at any prior planner version including legacy
  ``True``): step 1 may either reuse or upgrade; step 2 must not
  modify the NIfTIs after step 1's snapshot is taken.

Requirements
------------
* The latest ``MedVision.py`` on the Hub must include both fixes.
  ``download_mode="force_redownload"`` below pulls the latest script
  from the Hub on each run.
* ``--data_dir`` (CLI arg) or ``MedVision_DATA_DIR`` (env var) must point
  to your local data directory. Defaults to ``./Data`` relative to this
  script if neither is set.

Run with::

    python test_planner_switch_medvision_ds_v1.1.0.py
    python test_planner_switch_medvision_ds_v1.1.0.py --data_dir /path/to/your/Data
"""

import argparse
import os
import sys
import time
import glob
import json
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Verify MedVision_PLANNER_VERSION fixes")
parser.add_argument(
    "--data_dir",
    default=None,
    help="Path to the MedVision data directory (sets MedVision_DATA_DIR). "
         "Defaults to MedVision_DATA_DIR env var, then ./Data next to this script.",
)
args = parser.parse_args()

# Resolve data directory: CLI arg > env var > default ./Data
if args.data_dir is not None:
    DATA_DIR = os.path.abspath(args.data_dir)
    os.environ["MedVision_DATA_DIR"] = DATA_DIR
else:
    DATA_DIR = os.environ.setdefault("MedVision_DATA_DIR", os.path.join(SCRIPT_DIR, "Data"))

# Pin the HF datasets cache to a scratch location so this test does not
# interfere with your normal builder cache. MUST be set BEFORE importing
# `datasets`, since the env var is read at import time. We also wipe the
# scratch cache at the start of every run so the result is deterministic
# regardless of leftovers from prior runs.
CACHE_ROOT = os.path.join(DATA_DIR, ".cache", "huggingface_test_planner")
if os.path.exists(CACHE_ROOT):
    shutil.rmtree(CACHE_ROOT)
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = CACHE_ROOT

from datasets import load_dataset

HUB_REPO = "YongchengYAO/MedVision"
CONFIG = "KiPA22_TumorLesionSize_Task01_Axial_Test"
SPLIT_NAME = "test"
DATASET_NAME = "KiPA22"  # raw-file folder under <DATA_DIR>/Datasets/
EXPECTED_111 = 303
EXPECTED_100 = 391

# After commit 1, each planner_version produces a cache folder named
#     <config>-planner_version=<v>/<dataset_version>/<hash>/dataset_info.json
CACHE_GLOB = os.path.join(CACHE_ROOT, "**", CONFIG + "*", "*", "*", "dataset_info.json")

# Raw NIfTI files reoriented in place if commit 2's gating condition allows
# a download. We compare their mtimes around step 2 to detect re-download.
RAW_IMAGES_GLOB = os.path.join(DATA_DIR, "Datasets", DATASET_NAME, "Images", "*.nii.gz")


def banner(text):
    line = "=" * 70
    print(f"\n{line}\n{text}\n{line}")


def load_with_planner(version):
    os.environ["MedVision_PLANNER_VERSION"] = version
    ds = load_dataset(
        HUB_REPO,
        name=CONFIG,
        trust_remote_code=True,
        split=SPLIT_NAME,
        download_mode="force_redownload",
    )
    print(f"  -> planner_version={version}: len(ds) = {len(ds)}")
    return len(ds)


def snapshot_mtimes():
    return {p: os.stat(p).st_mtime for p in sorted(glob.glob(RAW_IMAGES_GLOB))}


def show_cache_layout(expected=2, max_wait_s=5.0):
    # HF datasets may finalize the dataset_info.json write briefly after
    # load_dataset returns (especially on the first cold-cache run when
    # the loader script is freshly downloaded). Retry a few times before
    # declaring the cache layout final.
    deadline = time.monotonic() + max_wait_s
    while True:
        paths = sorted(glob.glob(CACHE_GLOB, recursive=True))
        if len(paths) >= expected or time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    print(f"  cache folders matching the config: {len(paths)}")
    for p in paths:
        d = json.load(open(p))
        splits = d.get("splits") or {}
        test_n = splits.get("test", {}).get("num_examples")
        rel = os.path.relpath(p, CACHE_ROOT)
        print(f"    - {rel}: test num_examples={test_n}")
    return paths


banner("Step 1: load with planner_version=1.1.0 (latest annotations)")
n_111 = load_with_planner("1.1.0")

# Snapshot raw NIfTI mtimes AFTER step 1. Whatever step 1 did
# (initial download, upgrade from older planner, or no-op) is fine —
# we only care that step 2 below does not modify these files further.
mtimes_after_step1 = snapshot_mtimes()
print(f"  raw NIfTI files on disk after step 1: {len(mtimes_after_step1)}")

banner("Step 2: load with planner_version=1.0.0 (original annotations)")
# Without commit 1, this raises NonMatchingSplitsSizesError because step
# 1's cached dataset_info.json (num_examples=303) is compared against the
# freshly generated v1.0.0 split (num_examples=391) in the same cache
# folder.
n_100 = load_with_planner("1.0.0")

# Snapshot mtimes again. With commit 2 in effect, mtimes are unchanged.
# Without commit 2, _split_generators would have re-fetched the
# annotations zip and re-oriented every NIfTI in place, bumping mtimes.
mtimes_after_step2 = snapshot_mtimes()

banner("Step 3: inspect cache layout")
folders = show_cache_layout()

banner("Step 4: check NIfTI mtimes between steps")
unchanged = mtimes_after_step1 == mtimes_after_step2
print(f"  raw NIfTI mtimes unchanged across step 2: {unchanged}")
if not unchanged:
    diffs = [
        p for p in mtimes_after_step1
        if mtimes_after_step1.get(p) != mtimes_after_step2.get(p)
    ]
    print(
        f"  files with changed mtimes: {len(diffs)}/{len(mtimes_after_step1)} "
        f"(sample: {diffs[:3]})"
    )

banner("Result")
checks = [
    (n_111 == EXPECTED_111, f"step 1 sample count == {EXPECTED_111} (got {n_111})"),
    (n_100 == EXPECTED_100, f"step 2 sample count == {EXPECTED_100} (got {n_100})"),
    (len(folders) >= 2, f"two distinct cache folders (got {len(folders)})"),
    (unchanged, "raw NIfTI mtimes unchanged across step 2"),
]
for ok, label in checks:
    marker = "[PASS]" if ok else "[FAIL]"
    print(f"  {marker} {label}")
print()
if all(ok for ok, _ in checks):
    print("All PASS")
    sys.exit(0)
else:
    print("FAIL: at least one check failed — see above.")
    sys.exit(1)
