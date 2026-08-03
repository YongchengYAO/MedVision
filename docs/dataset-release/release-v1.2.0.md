# Release v1.2.0

> ⚠️ **Partly superseded — the MAMA-MIA and PI-CAI annotations from this release were withdrawn in v1.2.1.**
>
> Two of the eight datasets added in this release shipped annotations recorded in the **source image orientation instead of RAS+**, so the recorded coordinates did not describe the images the loader serves. Those files have been **deleted from the hub** and reissued as **v1.2.1**.
>
> - Pinning `1.2.0` now raises a *withdrawn* error for these two. `latest` resolves to `1.2.1` and loads the corrected annotations.
> - **If you loaded MAMA-MIA or PI-CAI at v1.2.0, discard those results and clear that cache** — the rows came from the defective annotation.
> - **The other 28 datasets are unaffected** and still resolve at `1.2.0` exactly as described below.
>
> See [`release-v1.2.1.md`](release-v1.2.1.md) for what did and did not change, and for the difference between a *withdrawn* and a *paused* version.
>
> **The rest of this note stands as written**, with two corrections it could not anticipate: `latest` now resolves to `1.2.1`, not `1.2.0`; and MAMA-MIA and PI-CAI are available only at `1.2.1`.

**v1.2.0 adds 8 datasets (130 configs) and changes no existing annotation.** Every dataset released before this version loads exactly the same annotation files it did at v1.1.1 — byte for byte.

`MedVision_PLANNER_VERSION` sets the newest annotations you are willing to load. To get v1.2.0:

```bash
export MedVision_PLANNER_VERSION=latest   # resolves to 1.2.0
```

Throughout this note, a **pin** means `MedVision_PLANNER_VERSION` set to a specific version rather than `latest`.

## Summary

Seven changes, most important first. Each links to its own section below.

| | Change | In one line | Action |
| --- | --- | --- | --- |
| 1 | [New datasets](#new-datasets) | 8 datasets, 130 configs; the catalogue grows from 820 to 950 | none |
| 2 | [Fixed: cached data could be stale](#fixed-cached-data-could-be-stale) | `load_dataset` could hand back old rows after the annotations behind them changed — and this happened to a real, shipped change | **check 4 conditions** |
| 3 | [How annotation versions are resolved](#how-annotation-versions-are-resolved) | each dataset now loads its own newest annotation at or below the version you ask for, instead of one rule for the whole catalogue | none |
| 4 | [Acknowledgement is now per dataset](#changed-acknowledgement-is-now-per-dataset) | `MedVision_ACK_RELEASE` is demanded only when *the dataset you are loading* has moved past your pin | none — fewer prompts |
| 5 | [Fixed: two data roots could share one cache](#fixed-two-data-roots-could-share-one-cache) | pointing at a second `MedVision_DATA_DIR` could return rows whose file paths point into the first | **clear once**, if it applies |
| 6 | [Fixed: download reliability](#fixed-download-reliability) | four defects: a failed download recorded as finished, a crash when one dataset was prepared twice at once, a broken relative data root, and ~27 GiB of needless re-downloading | none |
| 7 | [Stricter `MedVision_PLANNER_VERSION`](#changed-stricter-medvision_planner_version-values) | values nobody published, like `1.1.5` or `v1.1.1`, are refused instead of silently resolving to something older | none unless you set one |

Items 2 and 5 are correctness fixes and are the only ones that can require anything of you. Everything else applies automatically.

## Do I need to do anything?

| Your situation | What to do |
| --- | --- |
| You use `latest` | Nothing |
| You pin `1.1.1` or older | Nothing breaks. You just cannot load the 8 new datasets — [details](#pinning-a-version-older-than-v120) |
| You built a `Tumor-Lesion-Size` cache before v1.1.1 shipped | Check four conditions, then clear that cache once — [details](#fixed-cached-data-could-be-stale) |
| You have used two or more `MedVision_DATA_DIR` values without a separate `HF_DATASETS_CACHE` for each | Clear those caches once — [details](#fixed-two-data-roots-could-share-one-cache) |

Landed here from a version error? Start with [How annotation versions are resolved](#how-annotation-versions-are-resolved).

---

## New datasets

| Dataset | Tasks |
| --- | --- |
| AFIDs | Biometrics-From-Landmarks |
| DEEP-PSMA | Mask-Size, Box-Size, Tumor-Lesion-Size |
| LIDC-IDRI | Mask-Size, Box-Size, Tumor-Lesion-Size |
| LNQ2023 | Mask-Size, Box-Size, Tumor-Lesion-Size |
| MAMA-MIA | Mask-Size, Box-Size, Tumor-Lesion-Size |
| PDDCA | Mask-Size, Box-Size, Biometrics-From-Landmarks |
| PI-CAI | Mask-Size, Box-Size, Tumor-Lesion-Size |
| VerSe | Mask-Size, Box-Size, Biometrics-From-Landmarks |

All 8 publish annotation version `1.2.0`. Config lists live in the dataset repo under `info/`: `info/v1.2.0/` (950 configs) and `info/v1.0.0-v1.1.1/` (820 configs).

For what these datasets actually contain — anatomy, modality, case counts, licences, the preprocessing decisions behind each one, and the Hugging Face mirrors they download from — see [`doc/release-v1.2.0-datasets.md`](release-v1.2.0-datasets.md).

## Fixed: cached data could be stale

**This affects all versions before v1.2.0 and is worth two minutes of your time.**

**In plain English.** `load_dataset` keeps a local copy of the rows it built last time, so that loading the same thing again is fast. To decide whether that copy is still good, it compared the version you *asked for* (i.e., the version from `MedVision_PLANNER_VERSION`). But asking for a version does not pin down the data. The same request can point at different annotation files at different times, and several different requests can point at one file. When the data behind your request changed, the request did not — so you got the old copy back, and the new annotations never reached you. The annotation file sitting on your disk was not refreshed either, because that check compared the same two version strings.

**Technically.** The HuggingFace builder-cache fingerprint was derived from the value of `MedVision_PLANNER_VERSION`, not from the `benchmark_plan_{kind}_v{X}.json.gz` that value resolved to. The requested version is not an identity for the data; the resolved filename is. The download predicate had the same flaw, comparing the requested version against the version recorded in `.downloaded_datasets.json`, so neither the Arrow layer nor the on-disk layer noticed the change.

**This is not hypothetical.** The v1.1.1 release changed the already-published v1.1.0 annotations in place, with no version bump. It re-aligned the train/test split, relabelling about 41% of cases in six datasets: BraTS24, HNTSMRG24, KiPA22, KiTS23, MSD and autoPET-III. (See "Split alignment to v1.0.0" in `doc/release-v1.1.1.md`.)

The measurement values were byte-identical, so a stale cache looks completely normal. Only the train/test partition differs.

**You are affected only if all four are true:**

1. you loaded a `Tumor-Lesion-Size` config, and
2. the dataset was BraTS24, HNTSMRG24, KiPA22, KiTS23, MSD or autoPET-III, and
3. you built the cache *before* the v1.1.1 release, at `MedVision_PLANNER_VERSION=1.1.0` or at `latest` (which meant 1.1.0 then), and
4. you have reused that cache since.

If any one of them is false, you have nothing to do here.

**To clear it,** refresh both caches once: the annotation file on disk *and* the Arrow cache. Clearing the Arrow cache alone is not enough, because the annotation file is stale too.

```python
import os
from datasets import load_dataset

config = "..."        # one of your affected Tumor-Lesion-Size configs
split_name = "test"   # repeat for each split you cached

os.environ["MedVision_FORCE_DOWNLOAD_DATA"] = "True"      # refresh the annotation file
ds = load_dataset(
    "YongchengYAO/MedVision",
    name=config,
    trust_remote_code=True,
    split=split_name,
    download_mode="force_redownload",                     # rebuild the Arrow cache
)
```

From v1.2.0 the cache key is the annotation version that **actually loads**. If the annotations behind your pin ever change, the key changes with them, and the stale cache is correctly missed.

**A published annotation file is never rewritten in place. Corrections always get a new version number** — the version string is the only identity the data has.

Two consequences of the new key are worth knowing:

- **Two pins that resolve to the same annotation file now share one cache** instead of building two. Pinning `1.1.1` and pinning `1.0.0` both load ACDC's only annotation, `1.0.0`, so they land in the same place.
- **One-time rebuild, for everyone.** Because the key changed, existing Arrow caches are orphaned and rebuild on next use. The rebuild reads the annotation plan file and re-emits the rows; it does not re-transfer any images or annotations, so it costs seconds per config.

Old cache directories are not deleted automatically. `<hf_cache>` below is your `HF_DATASETS_CACHE` (default `~/.cache/huggingface`); `datasets` snake-cases the builder name, hence `med_vision`. List before deleting:

```bash
ls -d <hf_cache>/datasets/*med_vision*     # check first
rm -rf <hf_cache>/datasets/*med_vision*
```

## How annotation versions are resolved

📚 [Annotation Version Control](https://medvision-vlm.github.io/explorer.html)

**In plain English.** Two different things were both called "the version", and they moved at different speeds. One is the version of the *release* — it goes up every time anything ships. The other is the version of one dataset's *annotations for one task* — it goes up only when those particular annotations are regenerated, which is rare. Treating them as the same number meant that publishing a new release implied every dataset had new annotations, which was never true.

So `MedVision_PLANNER_VERSION` is now read as a **ceiling** rather than an exact match: *"give me the newest annotations that existed at or before this point"*. Each dataset answers that question for itself. `latest` means "the annotations as they stand now"; `1.1.1` means "the annotations as they stood at v1.1.1" — dataset by dataset.

**Technically.**

- The **release version** (`1.2.0`) is a property of the published `MedVision.py` and advances every release. It is deliberately hardcoded in the loader, so it reflects the remote release rather than whichever `medvision_ds` happens to be installed locally.
- The **annotation version** — the `_v{X}` in `benchmark_plan_{kind}_v{X}.json.gz`, where *kind* is `segmentation`, `detection` or `biometry` — is a property of a *(dataset, task)* pair.

`MedVision_PLANNER_VERSION` accepts **either kind of version** — a published annotation version such as `1.1.1`, or the `medvision_ds` release version — and resolves per (dataset, task) to the newest published annotation at or below it.

The accepted set is derived from the annotation index, so it is exactly:

| Value | |
| --- | --- |
| `latest` | resolves to the current release, `1.2.0` |
| `1.2.0` | adds 8 datasets; existing annotations unchanged — also the current release |
| `1.1.1` | fixes transposed in-plane voxel spacing in the TL ellipse fit |
| `1.1.0` | corrected TL filtering, cluster threshold 20px |
| `1.0.0` | original TL filtering, cluster threshold 200px |

Anything else is refused — see [Stricter `MedVision_PLANNER_VERSION` values](#changed-stricter-medvision_planner_version-values).

The release version stays acceptable even if a future release publishes no annotations of its own — otherwise `latest`, which resolves to it, would stop working.

Worked example, all at `MedVision_PLANNER_VERSION=latest`:

| Dataset / task | Published versions | Loads |
| --- | --- | --- |
| ACDC / Mask-Size | `1.0.0` | `1.0.0` |
| KiTS23 / Tumor-Lesion-Size | `1.0.0`, `1.1.0`, `1.1.1` | `1.1.1` |
| PDDCA / Mask-Size | `1.2.0` | `1.2.0` |

### Why the old rule had to go

For every task except Tumor-Lesion-Size (TL) the old loader fell back to `1.0.0`; for TL it did not fall back at all, and required an annotation file stamped with the exact release version.

Once the release became 1.2.0, `latest` would have gone looking for a `1.2.0` TL annotation for every dataset. The five new TL datasets publish one. **The 162 pre-existing TL configs do not, and would all have failed.**

The new rule also matches what v1.1.1 already documented: *"if a `1.1.x` plan is absent the loader transparently falls back"*. The behaviour for every combination that worked before is unchanged.

**A missing annotation is now caught up front.** The per-(dataset, task) rule replaces the hardcoded `1.0.0` fallback and its TL exclusion, so a missing file raises a named error at resolution time — before anything downloads — instead of crashing later, mid-way through generating rows.

The two kinds of version coincide today — every annotation version published so far is also a release version — so the distinction has not yet had to matter. It will the first time a correction ships as, say, `1.2.1` without a release of its own: that value becomes accepted as an annotation version, and reading the setting as a ceiling is what makes it behave sensibly.

## Changed: acknowledgement is now per dataset

**In plain English.** `MedVision_ACK_RELEASE` is the "yes, I know I am asking for something older than what exists" switch. Before, *older* was judged against the catalogue as a whole: publishing anything new made every pinned user set the switch, even for datasets the release never touched. Now it is judged against the dataset in front of you. Since v1.2.0 changed no existing annotation, nobody pinned at `1.1.1` is prompted at all.

**Technically.** The gate compares your pin against the newest annotation version published for *this* (dataset, task) pair, not against the release version.

```bash
export MedVision_PLANNER_VERSION=1.1.0    # older than KiTS23's newest TL annotation

# pick ONE of these two:
export MedVision_ACK_RELEASE=1.1.1        # KiTS23's newest TL annotation, or ...
# export MedVision_ACK_RELEASE=1.2.0      # ... the whole release
```

Two values are accepted, because they acknowledge different things:

- **The dataset's newest annotation** (`1.1.1` above) — "I know *this dataset* has moved past my pin." Use it when loading one dataset; it is the number the error message shows you. It stops working the next time that dataset is regenerated.
- **The release** (`1.2.0` above) — "I have read release 1.2.0." Use it for a catalogue sweep. It stops working at the next release.

A sweep cannot use the per-dataset value. Different datasets sit at different newest versions, so a sweep would need several values at once — and `MedVision_ACK_RELEASE` holds only one.

What changed is *when you are blocked*. Both columns assume the release is 1.2.0:

| Your pin | Config you load | Before | Now |
| --- | --- | --- | --- |
| `1.1.1` | ACDC / Mask-Size | blocked | loads — v1.2.0 did not touch ACDC |
| `1.1.1` | KiTS23 / Tumor-Lesion-Size | blocked | loads — `1.1.1` *is* its newest |
| `1.1.0` | KiTS23 / Tumor-Lesion-Size | blocked | still blocked — a newer TL annotation exists |

## Fixed: two data roots could share one cache

**In plain English.** Every row MedVision hands you contains absolute file paths, and those paths are built from `MedVision_DATA_DIR`. The data root was therefore baked into the rows — but it was not part of the name the cache was filed under, so a cache built for one root looked like a match for any other. Point at a second root and the first root's cache answered: nothing downloaded into the new location, and the paths you got back still led into the old one.

**Technically.** The canonicalised data root is now part of the builder-cache fingerprint, so each root keeps its own cache and two roots can be used side by side.

Whether this could ever have affected you depends on where your Arrow cache lives. For most users, it could not:

- **Never affected: [`medvision_bm`](https://github.com/YongchengYAO/MedVision) users** (the MedVision benchmark and finetuning codebase). `setup_env_hf_medvision_ds(data_dir)` sets `MedVision_DATA_DIR`, `HF_HOME` and `HF_DATASETS_CACHE` from the same `data_dir`, so the Arrow cache already moved with the data root. Every eval and training script goes through that call.
- **Never affected: setting `HF_HOME` / `HF_DATASETS_CACHE` yourself per data root, or passing `cache_dir=`** — for the same reason.
- **Exposed: calling `load_dataset` directly with only `MedVision_DATA_DIR` set** — the minimal usage shown on the dataset card. `HF_DATASETS_CACHE` then defaults to `~/.cache/huggingface/datasets`, which does not move with the data root, so both roots shared one cache.

If you are in that last group and you have used more than one data root, clear the affected caches once with `download_mode="force_redownload"`. Unlike the stale-annotation case above, the annotation files themselves were never wrong here, so rebuilding the Arrow layer is enough; the rebuild runs the loader, which downloads into the new root.

Setting `HF_DATASETS_CACHE` alongside each data root, as `medvision_bm` does, avoids the problem entirely and is worth doing regardless.

## Fixed: download reliability

Four defects in the download path. All are fixed automatically; none need action.

### A failed image download is no longer recorded as a finished install

**In plain English.** The loader writes a note saying "this dataset is fully installed", and every later load trusts that note and skips downloading. The note used to be written even when the image download had failed. So a dataset whose images never arrived was marked complete forever, and later loads produced rows pointing at files that do not exist.

**Technically.** The image step caught its own failure — a bare `except:` that blindly re-ran the entire multi-gigabyte transfer, wrapped by an outer `except subprocess.CalledProcessError` — and fell through to write the completion entry in `.downloaded_datasets.json` regardless of outcome. The failure now propagates and no entry is written, so the next load retries. The same bare handler also caught Ctrl-C, restarting a long download instead of stopping it.

### Two configs of one dataset can be prepared at the same time

**In plain English.** A dataset's annotations arrive as one zip file, which the loader downloads, unpacks, and deletes. But one dataset covers many configs — train and test of a single task are two of them, and BraTS24 has 228 in total — and HuggingFace prepares each config as an independent job. Two jobs for the same dataset therefore both downloaded that one zip, both unpacked it on top of each other, and whichever finished second crashed trying to delete a file the first had already removed. Loading a dataset's train and test splits in parallel was enough to trigger it.

**Technically.** HuggingFace's builder lock is per config, and the loader's own existing lock guards only `.downloaded_datasets.json` — which is not written until after the images finish, leaving the whole install unprotected. The download → unpack → delete sequence now runs under a per-dataset lock, with a re-check inside it so the waiting job skips the work instead of repeating it. The previous symptom was a bare `FileNotFoundError` on `Datasets/<name>.zip`.

### `MedVision_DATA_DIR` is resolved to an absolute path once

**In plain English.** A relative data root such as `./data` broke every download, because two different steps each changed the working directory and the second one then resolved against the first. Different spellings of one location also counted as different locations.

**Technically.** The root is canonicalised once, up front, with `expanduser` → `abspath` → `normpath`, so `~/data`, `/home/me/data` and `/home/me/data/` are one root rather than three — which also matters now that the root is part of the cache key. An empty value is rejected rather than silently promoting the current working directory to the data root.

### The download check now asks two separate questions

**In plain English.** "Which annotations do I have?" and "did the last install finish?" used to be answered by the same recorded number, which was the version you *requested* — a number the dataset might never have published. The first question is now answered by looking at the dataset directory; the second by whether the record exists at all.

**Technically.**

| Question | Now answered by | Previously answered by |
| --- | --- | --- |
| Which annotation version is on disk? | The dataset directory itself, compared against what is published for that dataset | The value recorded in `.downloaded_datasets.json`, compared against the version you requested |
| Did a previous install finish? | Whether a `.downloaded_datasets.json` entry exists | The same entry — but its *value* was also read as a version |

The entry is written only after the images download and the RAS+ reorientation completes, so a missing entry reliably marks a run that died partway. Under the old rule, a pin naming a version the dataset never published forced a full re-download — about 27 GiB of annotation archives across the 22 pre-existing datasets, plus a re-run of image download and reorientation over the whole corpus.

### `.downloaded_datasets.json` now records what is on disk

Before v1.2.0 this entry recorded the version you *requested*, not the version on disk. ACDC's only annotation is `1.0.0`, but loading it at `latest` under the v1.1.1 loader wrote `dataset_ACDC: "1.1.1"`, and pinning `1.2.0` wrote `"1.2.0"`. The entry now records the highest version actually present on disk.

| Dataset / plan kind | Versions on disk | Pin | Old entry | New entry |
| --- | --- | --- | --- | --- |
| KiTS23 / biometry | `1.0.0, 1.1.0, 1.1.1` | `1.2.0` | `1.2.0` ✗ | `1.1.1` |
| KiTS23 / biometry | `1.0.0, 1.1.0, 1.1.1` | `1.1.0` | `1.1.0` | `1.1.1` |
| ACDC / segmentation | `1.0.0` | `1.2.0` | `1.2.0` ✗ | `1.0.0` |

✗ marks an entry naming a version the directory does not contain.

The `1.1.0` pin row is deliberate. The entry describes what the directory *can serve*, not what this particular load asked for, so an older `MedVision.py` comparing `cached < requested` still reaches the right answer. (The entry is rewritten only when a download actually runs, so the "new entry" is what would be written on that dataset's next download.)

**You do not need to clean up old entries.** A wrong value is now inert — the version decision reads the dataset directory, not this field — and it is overwritten the next time that dataset downloads. The field's *presence* is what still matters: it is the "install finished" marker.

## Changed: stricter `MedVision_PLANNER_VERSION` values

**In plain English.** A version string nobody ever published is far more likely to be a typo than an intention, so it is now refused with the accepted values listed. Previously such a value was accepted and quietly resolved to something older — or left every config unloadable with no explanation.

**Technically.** Values are validated against the published annotation versions plus the current release ([the accepted set](#how-annotation-versions-are-resolved)). Two categories were previously mishandled:

*Malformed*, like `v1.1.1` or `1.2`. These matched no published annotation filename and fell through to the `1.0.0` fallback, which printed a banner naming the requested and loaded files — but only once per task type, so in a catalogue sweep later datasets degraded quietly. For Tumor-Lesion-Size, where there was no fallback at all, the same pin produced a missing path and a crash.

*Well-formed but never published*, like `1.1.5` or `0.0.0`. `1.1.5` silently resolved down to whatever each dataset published below it, and `0.0.0` left all 950 configs unloadable with no hint why.

Both are now refused at parse time. Surrounding whitespace is still stripped and accepted.

---

## Pinning a version older than v1.2.0

A dataset introduced in v1.2.0 cannot be loaded at an earlier pin — its annotations did not exist yet. Asking for one raises a clear error naming the dataset and the versions that do exist, instead of failing deep inside the loader.

Setting `MedVision_ACK_RELEASE` does not help here, and the error says so. To sweep the whole catalogue at a pinned version, iterate that release's config list instead. The path below is relative to a checkout of the dataset repo:

```python
configs = open("info/v1.0.0-v1.1.1/ConfigurationsList_All.csv").read().split()
```

## What did not change

**No pre-existing dataset was regenerated in this release.** Every (pin, dataset, task) combination that worked before yields byte-identical rows:

- Where your pin matched an annotation file exactly, it still resolves to that file.
- Where your pin fell back to `1.0.0` (every non-TL task), it still resolves to `1.0.0` — every non-TL dataset publishes exactly one annotation version.
- Legacy boolean `true` entries in `.downloaded_datasets.json` remain valid: their presence is read as "an install completed", which is what the download check now uses, so nothing re-downloads en masse. Their value is no longer interpreted as a version. See `doc/release-v1.1.0.md` for the original contract.
- No environment variable, config name, feature schema or split name changed.

Two behaviours changed deliberately, both from "silently wrong" to "explicitly refused": a malformed version string, and a dataset requested at a version predating its existence. Neither had a working counterpart before.

Verified with `scripts/test_annotation_resolution.py` (950 configs × every pin) and `scripts/test_tl_ack_gate.py`, both in the dataset repo.

## See also

- `doc/release-v1.2.1.md` — **corrects MAMA-MIA and PI-CAI, whose v1.2.0 annotations are withheld.** Read this if you use either dataset; everything else in this note still stands
- `doc/release-v1.2.0-datasets.md` — the 8 new datasets in detail
- `doc/release-v1.1.1.md` — TL ellipse-fit bugfix
- `doc/release-v1.1.0.md` — TL sample filtering; the legacy-boolean tracker contract
- `doc/design-annotation-version-resolution.md` — design notes for the resolution mechanism
