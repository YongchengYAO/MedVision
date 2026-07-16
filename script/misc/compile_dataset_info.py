"""Compile ``dataset_info`` for every MedVision dataset and audit its links.

Source of truth is the live ``medvision_ds`` preprocess modules — the same dicts the
benchmark planner stamps into ``benchmark_plan_*.json.gz`` — so the output can never
drift from the shipped annotations. Every dataset's ``dataset_info`` is asserted to be
identical across its preprocess_{segmentation,detection,biometry} modules.

Outputs (under --out_dir, default dataset-info/):
  datasets_info.json             the compiled data (feeds the webpage Dataset Explorer)
  datasets_info_link_audit.json  per-URL HTTP status (only with --audit_links)

Usage:
  PYTHONPATH=src python script/misc/compile_dataset_info.py [--audit_links]
"""

import argparse
import concurrent.futures
import json
import os
import importlib
import re
import sys

from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE

# medvision_ds is pip-installed as a NON-editable copy in site-packages, which silently shadows the
# working tree — edits to the preprocess files would be invisible here. Prepend the source checkout
# so this always compiles from the sources under version control, not a stale installed copy.
DEFAULT_DS_SRC = "/mnt/vincent-pvc-rwm/MedVision/src"

KINDS = ("segmentation", "detection", "biometry")
INFO_KEYS = ("dataset", "dataset_website", "dataset_data", "license", "paper")

# Licence strings are hand-written per dataset and spelled inconsistently. Normalise the
# SEPARATORS only ("CC-BY-NC" / "CC BY NC ND" -> "CC BY-NC" / "CC BY-NC-ND"). Deliberately
# never append a version: the source saying "CC-BY-NC" does not tell us it is 4.0, and
# asserting a version on a licence would be inventing a legal fact.
LICENSE_SPELLING = {
    "CC-BY-NC": "CC BY-NC",
    "CC-BY-NC 4.0": "CC BY-NC 4.0",
    "CC BY NC ND": "CC BY-NC-ND",
}


def normalize_license(value):
    if not isinstance(value, str):
        return value
    v = " ".join(value.split())
    return LICENSE_SPELLING.get(v, v)


# Redistributed datasets pull a HuggingFace copy on the loader path — from download_fast.py where one
# exists, else download.py (download_raw.py is excluded on purpose: AbdomenAtlas's raw script points at
# the upstream AbdomenAtlas/* org, which is not ours). Every repo is fetched with repo_type="dataset",
# so the public URL is huggingface.co/datasets/<id>. Two spellings occur:
#   ISLES24/TopCoW24/*_fast:  repo_id="YongchengYAO/X"
#   KiPA22/ToothFairy2/SKM:   repo_id=<var>, where <var> = os.environ.get("...", "YongchengYAO/X")
_LOADER_SCRIPTS = ("download.py", "download_fast.py")
_REPO_ID_RE = re.compile(r'repo_id\s*=\s*["\']([^"\']+)["\']')
_ENV_DEFAULT_RE = re.compile(r'os\.environ\.get\(\s*["\'][^"\']+["\']\s*,\s*["\']([^"\']+/[^"\']+)["\']')

# Datasets we redistribute on HF that have NO download_fast.py: the loader fetches the raw data from
# the original host, and the HF repo carries only our own preprocessed release. Nothing in the
# download scripts references these, so they must be listed by hand.
HF_EXTRA = {
    "FeTA24": ["https://huggingface.co/datasets/YongchengYAO/FeTA24-Biometrics"],
}

# NEVER publish these: private mirrors (401 to anonymous users). ToothFairy2/download.py and
# SKM_TEA/download.py default to these ids — by design, since users are meant to point the env var at
# their OWN private repo — so advertising them would send readers to a 401 instead of the source.
HF_PRIVATE = {
    "https://huggingface.co/datasets/YongchengYAO/ToothFairy2",
    "https://huggingface.co/datasets/YongchengYAO/SKM-TEA-nii",
}

# On HF, but NOT a redistribution: YongchengYAO/OAIZIB-CM is that dataset's ORIGINAL home (it is our
# own dataset, from CartiMorph), which is why README's table marks it "HF" and not "HF*". The link is
# already shown under Source. Keep this in step with that column.
HF_NOT_REDISTRIBUTED = {"OAIZIB-CM"}

# Per-dataset caveats the compiled fields cannot express. Each note is {text, url?, url_label?}.
# Wording follows the dataset card's own "Datasets" section so the two never disagree.
RAW_ACCESS_URL = "https://huggingface.co/datasets/YongchengYAO/MedVision#datasets"
_NO_REDIST = ("⚠️ This dataset does not allow redistribution, so you need to apply for access from "
              "the data owners")

DATASET_NOTES = {
    "FeTA24": [
        {"text": _NO_REDIST + ". MedVision downloads the raw data automatically from Synapse, so "
                 "once you have access, set the SYNAPSE_TOKEN environment variable before using "
                 "MedVision.",
         "url": RAW_ACCESS_URL, "url_label": "Access requirements →"},
        {"text": "📝 In this HF dataset (FeTA24-Biometrics) we only released the preprocessed data "
                 "and our landmarks."},
    ],
    "SKM-TEA": [
        {"text": _NO_REDIST + ", process the raw data, upload the preprocessed data to your own "
                 "private HF dataset repo, and set the MedVision_SKMTEA_HF_ID environment variable.",
         "url": RAW_ACCESS_URL, "url_label": "Access requirements →"},
    ],
    "ToothFairy2": [
        {"text": _NO_REDIST + ", process the raw data, upload the preprocessed data to your own "
                 "private HF dataset repo, and set the MedVision_ToothFairy2_HF_ID environment "
                 "variable.",
         "url": RAW_ACCESS_URL, "url_label": "Access requirements →"},
    ],
}


def hf_redistribution(dataset, package_dir):
    """Public HF dataset repos this project redistributes (README's "HF*" column).

    Parsed from the loader-path download scripts (a dataset may ship several, e.g. TotalSegmentator
    has CT-Lite + MR-Lite), plus any HF_EXTRA entry. Private repos and non-redistributions are
    dropped: sending a reader to a 401, or badging our own dataset as a redistribution, are both
    worse than showing nothing — the Source link still gets them there.
    """
    if dataset in HF_NOT_REDISTRIBUTED:
        return []
    repo_ids = []
    for script in _LOADER_SCRIPTS:
        path = os.path.join(package_dir, script)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            src = fh.read()
        repo_ids += _REPO_ID_RE.findall(src) + _ENV_DEFAULT_RE.findall(src)
    # dict.fromkeys dedupes (TotalSegmentator names each repo twice) while keeping source order
    urls = [f"https://huggingface.co/datasets/{r}" for r in dict.fromkeys(repo_ids)]
    urls += HF_EXTRA.get(dataset, [])
    return [u for u in dict.fromkeys(urls) if u not in HF_PRIVATE]

# Some hosts reject unknown agents with 403; identify as a normal browser.
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def collect_dataset_info():
    """Return {dataset: dataset_info}, failing loud on cross-module disagreement."""
    out, problems = {}, []
    for ds, pkg in sorted(DATASETS_NAME2PACKAGE.items()):
        seen, package_dir = {}, None
        for kind in KINDS:
            try:
                module = importlib.import_module(f"medvision_ds.datasets.{pkg}.preprocess_{kind}")
            except ImportError:
                continue
            package_dir = os.path.dirname(module.__file__)
            info = getattr(module, "dataset_info", None)
            if info is None:
                problems.append(f"{ds}: preprocess_{kind}.py has no dataset_info")
                continue
            seen[kind] = info
        if not seen:
            problems.append(f"{ds}: no preprocess module exposed dataset_info")
            continue
        uniq = {json.dumps(v, sort_keys=True) for v in seen.values()}
        if len(uniq) != 1:
            problems.append(f"{ds}: dataset_info differs across {list(seen)} — cannot pick one")
            continue
        info = next(iter(seen.values()))
        missing = [k for k in INFO_KEYS if k not in info]
        if missing:
            problems.append(f"{ds}: dataset_info missing key(s) {missing}")
        clean = {k: info.get(k) for k in INFO_KEYS}
        # ACDC's dataset_website is " https://..." with a leading space; strip defensively
        # so a stray character upstream can never ship a broken href.
        clean["dataset_website"] = (clean.get("dataset_website") or "").strip()
        for k in ("dataset_data", "paper"):
            clean[k] = [u.strip() for u in (clean.get(k) or []) if u and u.strip()]
        clean["license"] = [normalize_license(l) for l in (clean.get("license") or []) if l and l.strip()]
        clean["hf_data"] = hf_redistribution(ds, package_dir) if package_dir else []
        # Source must mean "where the data originally comes from". Ceph-Biometrics-400 and ISLES24
        # list their YongchengYAO repo in dataset_data as well as being redistributed by us, which
        # renders the identical URL under both Source and HF data. Reclassify rather than duplicate:
        # anything hf_data already claims is ours moves out of Source. No URL is lost — the two
        # fields together still hold every link — and datasets whose HF repo IS the origin
        # (OAIZIB-CM, our own dataset) have empty hf_data, so nothing is taken from them.
        clean["dataset_data"] = [u for u in clean["dataset_data"] if u not in set(clean["hf_data"])]
        if not clean["dataset_data"]:
            problems.append(f"{ds}: no upstream source link left in dataset_data — the only entry was "
                            f"our own HF redistribution, so the reader has nowhere to go for the origin")
        clean["notes"] = DATASET_NOTES.get(ds, [])
        # Ships expert segmentation masks. A preprocess_segmentation.py IS that fact — the module
        # declares the mask_folder the planner reads (verified: all of them do), and its absence is
        # what makes Ceph-Biometrics-400 the one landmark-only dataset. Derived, not hand-listed, so
        # a future mask-less dataset needs no edit here.
        clean["has_segmentation"] = "segmentation" in seen
        clean["_source_modules"] = sorted(seen)
        out[ds] = clean
    return out, problems


def readme_hf_star(readme_path):
    """Datasets the README's Dataset table marks as redistributed by us ("HF*" in Data Source).

    That table is the authority for what we redistribute; this reads it so the compiled hf_data can
    be cross-checked against it instead of trusting a proxy signal like "has a download_fast.py".
    Returns None if the table cannot be found, so a moved README degrades to a warning, not a crash.
    """
    # The table uses display names; map the ones that differ from the dataset keys.
    alias = {"Ceph-Bio-400": "Ceph-Biometrics-400", "AbdomenAtlas": "AbdomenAtlas1.0Mini"}
    if not os.path.exists(readme_path):
        return None
    star = set()
    with open(readme_path) as fh:
        for line in fh:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6 or not cells[0] or cells[0].startswith(("-", "**", "Dataset")):
                continue
            if "HF*" in cells[5]:
                star.add(alias.get(cells[0], cells[0]))
    return star or None


def check_url(url):
    """HEAD (then GET) a URL, following redirects. Never raises."""
    import requests

    for method in ("head", "get"):
        try:
            r = requests.request(
                method, url, allow_redirects=True, timeout=25,
                headers={"User-Agent": UA}, stream=(method == "get"),
            )
            if method == "head" and r.status_code >= 400:
                continue  # some hosts refuse HEAD; retry with GET before judging
            return {"url": url, "status": r.status_code, "final_url": r.url,
                    "redirected": r.url.rstrip("/") != url.rstrip("/")}
        except Exception as e:
            if method == "get":
                return {"url": url, "status": None, "error": f"{type(e).__name__}: {e}"[:160]}
    return {"url": url, "status": None, "error": "unreachable"}


def audit_links(info):
    """HTTP-check every unique URL across all datasets, in parallel."""
    urls = []
    for ds, di in info.items():
        if di.get("dataset_website"):
            urls.append(di["dataset_website"])
        urls += [u for u in (di.get("dataset_data") or []) if u]
        urls += [u for u in (di.get("hf_data") or []) if u]
        urls += [u for u in (di.get("paper") or []) if u]
    urls = sorted(set(urls))
    print(f"auditing {len(urls)} unique links ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check_url, urls))
    return {r["url"]: r for r in results}


def classify(r):
    """Verdict for a checked URL.

    Publishers answer bots with codes that are not failures: IEEE/figshare return 202
    (challenge page) and RSNA/grand-challenge return 403. Those links are fine in a
    browser, so they must not be reported as broken — only 4xx-not-403 and 5xx are.
    """
    if r.get("status") is None:
        return "UNREACHABLE"
    if r["status"] == 200:
        return "REDIRECT" if r.get("redirected") else "OK"
    if 200 < r["status"] < 300:
        return "BOT-CHALLENGE"  # reachable; server fobs off non-browser clients
    if 300 <= r["status"] < 400:
        return "REDIRECT"
    if r["status"] in (401, 403):
        return "BLOCKED"  # reachable, but refuses an automated client
    if 400 <= r["status"] < 500:
        return "BROKEN"
    return "SERVER-ERROR"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="dataset-info", help="output directory")
    ap.add_argument("--audit_links", action="store_true", help="HTTP-check every link")
    ap.add_argument("--medvision_ds_src", default=DEFAULT_DS_SRC,
                    help="medvision_ds source checkout to compile from (shadows the installed copy).")
    args = ap.parse_args()

    if args.medvision_ds_src:
        sys.path.insert(0, args.medvision_ds_src)

    import medvision_ds
    print(f"reading medvision_ds v{medvision_ds.__version__} from {os.path.dirname(medvision_ds.__file__)}")
    if args.medvision_ds_src and not medvision_ds.__file__.startswith(args.medvision_ds_src):
        raise SystemExit(
            f"medvision_ds resolved to {medvision_ds.__file__}, not the requested source "
            f"{args.medvision_ds_src} — refusing to compile from a possibly stale installed copy."
        )

    info, problems = collect_dataset_info()
    print(f"compiled dataset_info for {len(info)} datasets")

    # Cross-check hf_data against the README's "HF*" column — the authority for what we redistribute.
    star = readme_hf_star(os.path.join(args.medvision_ds_src or "", "..", "README.md"))
    if star is None:
        print("WARNING: could not read the README Dataset table — hf_data left unchecked")
    else:
        have = {k for k, v in info.items() if v.get("hf_data")}
        missing = sorted(star - have)
        extra = sorted(have - star - set(HF_EXTRA))
        print(f"hf_data: {len(have)} datasets | README marks {len(star)} as HF* | "
              f"extra by design: {sorted(HF_EXTRA)}")
        if missing:
            problems.append(f"README marks {missing} as HF* but no hf_data link was resolved")
        if extra:
            problems.append(f"hf_data links for {extra}, which the README does not mark HF*")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    from medvision_ds import __version__ as ds_version
    out_path = os.path.join(args.out_dir, "datasets_info.json")
    with open(out_path, "w") as f:
        json.dump({"medvision_ds_version": ds_version, "datasets": info}, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote {out_path}")

    if not args.audit_links:
        return

    audit = audit_links(info)
    buckets = {}
    for url, r in audit.items():
        buckets.setdefault(classify(r), []).append(r)

    print("\n" + "=" * 78)
    print("LINK AUDIT")
    print("=" * 78)
    for k in ("OK", "REDIRECT", "BOT-CHALLENGE", "BLOCKED", "BROKEN", "SERVER-ERROR", "UNREACHABLE"):
        if k in buckets:
            print(f"  {k:<13} {len(buckets[k])}")
    for k in ("BROKEN", "SERVER-ERROR", "UNREACHABLE", "BLOCKED", "BOT-CHALLENGE", "REDIRECT"):
        for r in sorted(buckets.get(k, []), key=lambda x: x["url"]):
            owners = [ds for ds, di in info.items()
                      if r["url"] == di.get("dataset_website")
                      or r["url"] in (di.get("dataset_data") or [])
                      or r["url"] in (di.get("hf_data") or [])
                      or r["url"] in (di.get("paper") or [])]
            print(f"\n  [{k}] {r['url']}")
            print(f"        used by : {', '.join(owners)}")
            if r.get("status") is not None:
                print(f"        status  : {r['status']}")
            if r.get("final_url") and r.get("redirected"):
                print(f"        final   : {r['final_url']}")
            if r.get("error"):
                print(f"        error   : {r['error']}")

    audit_path = os.path.join(args.out_dir, "datasets_info_link_audit.json")
    with open(audit_path, "w") as f:
        json.dump({url: {**r, "verdict": classify(r)} for url, r in sorted(audit.items())}, f, indent=2)
        f.write("\n")
    print(f"\nwrote {audit_path}")


if __name__ == "__main__":
    main()
