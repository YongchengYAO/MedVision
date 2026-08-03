# Release v1.2.1

**v1.2.1 corrects the MAMA-MIA and PI-CAI annotations, which were recorded in the source orientation instead of RAS+.** Their v1.2.0 annotations are **withdrawn** — removed from the hub, so `1.2.1` is now the earliest version either dataset offers. **No other dataset is affected** — the other 28 resolve to exactly the same annotation files they did at v1.2.0, byte for byte.

```bash
export MedVision_PLANNER_VERSION=latest   # resolves to 1.2.1
```

If you have ever loaded a MAMA-MIA or PI-CAI config, [clear that cache once](#do-i-need-to-do-anything).

## Summary

| | Change | In one line | Action |
| --- | --- | --- | --- |
| **Major** | | | |
| 1 | [Annotations in the wrong orientation](#annotations-in-the-wrong-orientation) | MAMA-MIA and PI-CAI recorded coordinates in the source frame while the loader reoriented the images to RAS+ underneath them | **clear the cache** if you used either |
| 2 | [Withdrawal of v1.2.0](#withdrawal-of-v120) | removed from the hub; a `1.2.0` pin errors, naming 1.2.1 | **move your pin** if you pinned 1.2.0 |
| 3 | [Reproducible data preparation](#reproducible-data-preparation) | `scripts/gen-annotations/` rebuilds any dataset from source; the annotation version is now an explicit input | none |
| **Minor** | | | |
| 4 | [Consistent values across numpy versions](#consistent-values-across-numpy-versions) | the same code on numpy 2.x used to record `16.5` where numpy 1.26 recorded `16.49999976158142` | none |
| 5 | [Download paths](#download-paths) | both corrected mirrors republished and repinned; new fast paths for AMOS22, HNTSMRG24, MSD; AbdomenAtlas1.0Mini's removed for licence reasons | none |

Item 1 is a correctness fix and is the only one that can require anything of you.

## Do I need to do anything?

| Your situation | What to do |
| --- | --- |
| You have never loaded MAMA-MIA or PI-CAI | Nothing |
| You use `latest` and have loaded either | Load once more with the refresh below; `latest` now resolves to 1.2.1 |
| You pin `1.2.0` and load either | Loading now raises an error naming 1.2.1. Move your pin to `1.2.1` or `latest` |
| You pin `1.1.1` or older | Nothing. You could not load these two datasets at that pin anyway |

```python
import os
from datasets import load_dataset

os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "True"      # refresh the annotation file
ds = load_dataset(
    "YongchengYAO/MedVision",
    name="...",                                            # your MAMA-MIA / PI-CAI config
    trust_remote_code=True,
    split="test",
    download_mode="force_redownload",                      # rebuild the Arrow cache
)
```

Both layers must be refreshed: the annotation file on disk *and* the Arrow cache. From v1.2.0 the cache key is the annotation version that actually resolves, so the move from 1.2.0 to 1.2.1 changes the key on its own — but the images themselves were also rewritten, which is why the download flag is needed too.

---

## Annotations in the wrong orientation

**In plain English.** MedVision measures anatomy in voxels, then converts to millimetres, so every recorded coordinate is an index into an image array. That only works if the annotation and the image agree on which way is up. For MAMA-MIA and PI-CAI they did not. The annotations were computed on the volumes exactly as they arrived from the source, but the loader rotates every image into a standard anatomical orientation (RAS+) before handing it to you — and rotating an image does not renumber a coordinate that was written down beforehand. So the voxels moved and the annotations did not follow.

**Technically.** Neither dataset's `download_raw.py` reoriented to RAS+, and no preprocessing run passed `--reorient2RAS`, so the benchmark planner recorded bounding boxes and landmark indices in the source frame:

| Dataset | Cohort | Source orientation | Relation to RAS+ |
| --- | --- | --- | --- |
| MAMA-MIA | DUKE, ISPY2 | `('L','A','I')` | two axis flips |
| MAMA-MIA | ISPY1, NACT | `('P','S','L')` | axis permutation — the array shape itself changes |
| PI-CAI | all | `('L','P','S')` | two axis flips |

`MedVision.py` step 3.3 then applies `nibabel.as_closest_canonical` to the images at load time, and touches only the `*.nii.gz` — never the `*.json.gz` that holds the annotations.

> **Why does the loader reorient, when reorientation is already a preprocessing step?**
>
> It is a **guard**, and on data that already complies it does nothing: `_reorient_niigz_RASplus` returns early on an already-RAS file and, in place, writes no bytes.
>
> It is needed because not every dataset can be redistributed. 9 of the 30 have no redistributable copy, so the loader fetches them from the original source in whatever orientation the provider ships and pairs them with annotations from this repo — the annotations are generated from RAS-aligned images. The other 21 arrive already preprocessed and RAS+, where it is a no-op.

The fix is to reorient in the downloader, before any annotation is computed, which is what the other six datasets added in v1.2.0 already did. Both `download_raw.py` now call `reorient_niigz_RASplus_batch_inplace()` after the image header is copied onto the mask, and the images and annotations were regenerated together as v1.2.1.


### Effect on the data

Regeneration was verified against the pre-correction data on every axis that should not have moved.

| | MAMA-MIA | PI-CAI |
| --- | --- | --- |
| Cases | 1506 | 425 |
| Train / test split | 1054 / 452 — **identical**, membership *and* order | 297 / 128 — **identical** |
| Segmentation + detection entries | 214,408 — **identical** | 42,857 — **identical** |
| Per-label voxel totals | **identical** | **identical** |
| Biometry entries / QC figures | 5,066 / 5,066 | 409 / 409 |

Bounding boxes were also recomputed directly from the reoriented masks on disk: **65,195 of 65,195 reproduce exactly.** 

**Biometry measurements do change**, on roughly 27% of PI-CAI cases and 36% of MAMA-MIA cases. This is not a second bug. The tumour/lesion acceptance filter requires the fitted ellipse's four axis endpoints to fall inside an enlarged bounding box and outside a shrunk one, and it computes both box edges with `int()` truncation, which is not mirror-symmetric: mirroring an image can move a box edge by one voxel, and a measurement sitting exactly on the boundary can fall on the other side of the test. Holding the data fixed and changing only the rounding rule collapses the difference to zero on all 425 PI-CAI cases and on a 335-case MAMA-MIA sample that includes every one of the 235 axis-permutation cases — so the change is fully accounted for by borderline acceptance, and no measurement value itself is orientation-dependent.

**Two PI-CAI cases** shipped an expert mask whose grid did not match its T2W image:

| Case | T2W image grid |
| --- | --- |
| `10408_1000415` | `640 x 640 x 19` at `0.3 x 0.3 x 3.6` mm |
| `10459_1000467` | `640 x 640 x 23` at `0.3 x 0.3 x 3.6` mm |

Their masks are resampled onto the image grid with nearest-neighbour interpolation rather than inheriting a mismatched header. The check compares size, spacing, origin and direction; all 425 pairs now agree on all four.

## Withdrawal of v1.2.0

**In plain English.** The old MAMA-MIA and PI-CAI annotations were wrong, so they have been removed. They are no longer downloadable, and `1.2.1` is now the earliest version either dataset offers.

**Technically.** The `1.2.0` entries are gone from `_ANNOTATION_INDEX`, which states what the hub holds. A pin of `1.2.0` therefore finds nothing for these two and raises the standard "not published at the selected version" error naming `1.2.1` — the same path any dataset takes when pinned below its first release. Every other dataset resolves at `1.2.0` exactly as before; only these 36 configs are unavailable at that pin.

This is a **withdrawal**, which is a different state from a **pause**:

| | Pause | Withdrawal |
| --- | --- | --- |
| The file on the hub | still there | deleted |
| `_ANNOTATION_INDEX` | still lists the version | version removed |
| `_PAUSED_ANNOTATIONS` | names the version | not listed — nothing left to withhold |
| Pinning that version | resolves, then is refused | resolves to an older version, or errors if none |
| Used when | a defect is suspected and an investigation is open | the replacement has shipped and the bad file is gone |

MAMA-MIA and PI-CAI were paused while v1.2.1 was prepared, then withdrawn once it was published. The pause mechanism remains in the loader, unused, for the next incident.

## Reproducible data preparation

`scripts/gen-annotations/` rebuilds the preprocessed images/masks and the benchmark plans for any dataset in the catalogue, from the original public sources. It replaces a set of ad-hoc scripts that were never version-controlled.

The annotation version is now an explicit input rather than a side effect of which `medvision_ds` happens to be installed: every `preprocess_*.py` accepts `--annotation_version`, and the planner falls back to the installed version only when nothing is passed. The driver has exactly two modes — reproduce the newest published annotation of each *(dataset, task)*, or, for maintainers, mint a new version after bumping `__version__`, which is refused unless the new version is strictly above every existing one.

See [`scripts/gen-annotations/README.md`](../scripts/gen-annotations/README.md), which also records what is *not* reproducible this way: the tumour/lesion biometry at v1.1.0 and v1.1.1 came from `scripts/regenerate_tl_annotation_v1.1.*.py`, not from `preprocess_biometry.py`.

## Other changes

### Consistent values across numpy versions

**In plain English.** The same code, run on two different numpy versions, wrote slightly different numbers into the detection plans. A regeneration therefore could not be checked against what was published.

**Technically.** Recorded sizes were computed as `voxel_count * spacing`, where the spacing comes from the NIfTI header as `float32`. Before numpy 2.0, multiplying an `int` by a `float32` widened the result to `float64`; NEP 50 changed that to stay in `float32`. So `5 * 3.3` recorded `16.49999976158142` on numpy 1.26 and `16.5` on numpy 2.x. The spacing is now cast with `float()` before the multiplication, which pins the result to the pre-NEP-50 value — the one every published annotation already contains.

Two `scipy`-compatibility casts landed alongside it (`bool` → `int32` before `find_objects`, which `scipy >= 1.15` rejects). Those are numerically inert.

### Download paths

**The MAMA-MIA and PI-CAI mirrors were republished.** Their `-Lite` repositories now hold the RAS+ corrected volumes, and each dataset's `download_fast.py` is repinned to the new revision. The pin is an exact commit SHA, so until it moves a republished mirror reaches nobody — the two go together.

| Mirror | Revision | Contents |
| --- | --- | --- |
| `YongchengYAO/PI-CAI-Lite` | `5df7713c…` | 425 images + 425 masks, 1 shard |
| `YongchengYAO/MAMA-MIA-Lite` | `bbbfb30d…` | 1506 images + 1506 masks, 2 shards |

Both were verified by downloading the published revision back and re-deriving the orientation of every volume from its NIfTI affine: **850/850 and 3012/3012 RAS+**.

**Eleven other mirrors were repinned to their current revision**, none of which changes what a user receives — every bump was made only after confirming the zip inventories are byte-identical at the old and new revision.

All eleven had drifted only through card edits — README-only commits for ACDC, BCV15, CAMUS, CrossMoDA, FLARE22, KiTS23, TotalSegmentator and autoPET-III, and for DEEP-PSMA, LIDC-IDRI and Ceph-Biometrics-400 the config lists this release completed.

Ceph-Biometrics-400 is the exception worth naming: its landmark files genuinely differ between the two revisions. Every landmark moves by exactly `(0, 0, +1)` — the off-by-one mirror fix — uniformly across all 7,600 points in all 400 cases.

**A uniform 1-voxel shift changes neither distances nor angles, so no error is introduced.** Both are translation-invariant: a distance depends on the difference between two points, an angle on the directions of two lines, and shifting every point by the same vector leaves both untouched. Verified against the released v1.0.0 plan rather than argued — all **4,400 distance** and **3,200 angle** measurements reproduce from either revision with zero mismatches, and the angles are bit-identical between the two landmark sets. The plan is untouched, and users now receive the corrected coordinates.

**New fast download paths** for AMOS22, HNTSMRG24 and MSD, whose former `download.py` became `download_raw.py`.

**AbdomenAtlas1.0Mini's fast path was removed.** Its licence does not permit redistributing a preprocessed copy, so no mirror exists and the loader now builds it from source via `download_raw.py`. That moves the catalogue split to **9 datasets prepared locally, 21 from a preprocessed copy**.

## What did not change

**No dataset other than MAMA-MIA and PI-CAI was regenerated.** All 28 others resolve to exactly the same annotation version they did at v1.2.0, and yield byte-identical rows.

- `MedVision_PLANNER_VERSION=1.2.0` still resolves for every other dataset; only MAMA-MIA and PI-CAI refuse.
- No environment variable, config name, feature schema or split name changed.

Verified with `scripts/test_annotation_resolution.py` (950 configs × every pin) and `scripts/test_tl_ack_gate.py`, both in the dataset repo.

## For maintainers

**Version bump.** `src/medvision_ds/__version__.py` → `1.2.1`; `MedVision.py` config `version="1.2.1"`, which is what `latest` resolves to and what enters the builder-cache fingerprint.

**Regenerating.** `python scripts/gen-annotations/build_dataset.py --data_dir <dir> --dataset MAMA-MIA`. The plan files already exist, so an intentional rebuild needs `--force`.

**A published annotation file is never rewritten in place.** Corrections always get a new version number — the version string is the only identity the data has, and the code that produced it is part of that identity. This release is that policy applied: v1.2.0 was withheld while the fix was prepared, reissued as v1.2.1, and then withdrawn — never corrected under its own name.

## See also

- `doc/release-v1.2.0.md` — the 8 new datasets, annotation version resolution, the cache-key fix
- `doc/release-v1.2.0-datasets.md` — the 8 new datasets in detail
- `doc/release-v1.1.1.md` — TL ellipse-fit bugfix
- `doc/design-annotation-version-resolution.md` — design notes for the version control mechanism
- `scripts/gen-annotations/README.md` — rebuilding any dataset from source
