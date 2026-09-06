#!/usr/bin/env python
"""check_biomedparse_env.py - pre-flight check for the MedVision BiomedParse ablation.

Reports, without running any model:
  * installed versions of the packages the ablation needs vs the pins of the ablation's
    `requirements.txt` (embedded below), including the torch CUDA build and
    `torch.cuda.is_available()`;
  * whether the upstream checkout `third_party/BiomedParse` exists and is at the pinned
    commit (needs `--ablation-dir`; uses `git rev-parse HEAD`);
  * presence and size of `models/biomedparse_v2.ckpt` and `models/finetuned-detect/last.ckpt`;
  * the MedVision data directory markers (`Datasets/`, `src/`, `.downloaded_datasets.json`);
  * which launcher env knobs are set (TASK, GPU, CUDA_VISIBLE_DEVICES, CHECKPOINT, ENV_NAME,
    BIOMEDPARSE_DIR, MedVision_DATA_DIR, MedVision_PLANNER_VERSION, MedVision_ACK_RELEASE, HF_TOKEN).

Prerequisites: none beyond Python 3.8+. Safe on CPU-only hosts; never downloads or imports the
upstream model code.

Examples
    python check_biomedparse_env.py --help
    python check_biomedparse_env.py                                  # check the current interpreter
    python check_biomedparse_env.py --python /path/to/envs/biomedparse/bin/python \
        --ablation-dir /path/to/MedVision/script/ablation/biomedparse
    python check_biomedparse_env.py --ablation-dir ... --data-dir /data/MedVision --json

Exit codes: 0 = all required imports present (version mismatches are warnings);
            1 = at least one required import missing;
            2 = bad arguments (interpreter not executable, ablation dir missing).
"""

import argparse
import importlib
import json
import os
import subprocess
import sys

# ----------------------------------------------------------------------------- pins
# From the ablation's requirements.txt (extra index: https://download.pytorch.org/whl/cu124)
PINS = {
    "numpy": "1.26.4",
    "packaging": "23.0",
    "setuptools": "65.6.3",
    "ninja": "1.11.1.1",
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "torchaudio": "2.6.0+cu124",
    "pandas": "2.2.2",
    "scikit-learn": "1.4.2",
    "hydra-core": "1.3.2",
    "lightning": "2.3.0",
    "marshmallow": "3.23.2",
    "timm": "0.9.16",
    "deepspeed": "0.14.2",
    "transformers": "4.40.0",
    "open-clip-torch": "2.26.1",
    "sentencepiece": "0.2.0",
    "kornia": "0.7.3",
    "python-dotenv": "1.0.1",
    "huggingface-hub": "0.36.0",
    "datasets": "3.6.0",
}
UPSTREAM_COMMIT = "e02096c03af0d79c6994ffc2d60a49eeb0361e1f"  # microsoft/BiomedParse, v2 branch
PRETRAINED_CKPT_MIN_GB = 4.0  # biomedparse_v2.ckpt is ~4.2 GB

# (import name, distribution name) - required by the launchers' code path
REQUIRED = [
    ("torch", "torch"),
    ("detectron2", "detectron2 (built from source by setup.sh)"),
    ("lightning", "lightning"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface-hub"),
    ("hydra", "hydra-core"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
    ("nibabel", "nibabel"),
    ("scipy", "scipy"),
    ("medvision_ds", "medvision_ds (install_medvision_ds --data_dir <data_dir>)"),
    ("medvision_bm", "medvision_bm (repository source <repo>/src)"),
]
OPTIONAL = [
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("datasets", "datasets"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("matplotlib", "matplotlib"),
    ("timm", "timm"),
    ("deepspeed", "deepspeed"),
    ("kornia", "kornia"),
    ("open_clip", "open-clip-torch"),
    ("sentencepiece", "sentencepiece"),
    ("dotenv", "python-dotenv"),
    ("marshmallow", "marshmallow"),
    ("safetensors", "safetensors"),
    ("accelerate", "accelerate"),
    ("psutil", "psutil"),
    ("SimpleITK", "SimpleITK"),
    ("pydicom", "pydicom"),
    ("nrrd", "pynrrd"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("packaging", "packaging"),
    ("setuptools", "setuptools"),
    ("ninja", "ninja"),
]
ENV_VARS = [
    "TASK", "GPU", "CUDA_VISIBLE_DEVICES", "CHECKPOINT", "ENV_NAME", "BIOMEDPARSE_DIR",
    "MedVision_DATA_DIR", "MedVision_PLANNER_VERSION", "MedVision_ACK_RELEASE", "HF_TOKEN",
]
UPSTREAM_FILES = ["inference.py", "utils.py", "configs/model/biomedparse.yaml", "configs/model/biomedparse_3D.yaml"]


# ----------------------------------------------------------------------------- probe
def _version_of(mod, dist):
    v = getattr(mod, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version as _dist_version
            v = _dist_version(dist.split(" ")[0])
        except Exception:
            v = None
    return v


def probe(extra_paths):
    """Import every module in THIS interpreter and return a JSON-serialisable report."""
    for p in extra_paths:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    out = {"python": sys.executable, "python_version": sys.version.split()[0], "modules": {}, "torch": {}}
    for name, dist in REQUIRED + OPTIONAL:
        try:
            mod = importlib.import_module(name)
            out["modules"][name] = {"version": _version_of(mod, dist), "file": getattr(mod, "__file__", None), "error": None}
        except BaseException as e:  # noqa: BLE001 - some packages raise SystemExit/ImportError subclasses
            out["modules"][name] = {"version": None, "file": None, "error": f"{type(e).__name__}: {e}"}
    if out["modules"].get("torch", {}).get("error") is None:
        try:
            import torch  # noqa: WPS433
            out["torch"] = {
                "version": torch.__version__,
                "cuda_build": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        except Exception as e:  # noqa: BLE001
            out["torch"] = {"error": f"{type(e).__name__}: {e}"}
    return out


def run_probe(python, extra_paths):
    if python is None:
        return probe(extra_paths)
    cmd = [python, os.path.abspath(__file__), "--probe"] + [f"--extra-path={p}" for p in extra_paths]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return {"error": f"interpreter not found: {python}"}
    except subprocess.TimeoutExpired:
        return {"error": "probe timed out after 600 s"}
    if res.returncode != 0 or not res.stdout.strip():
        return {"error": f"probe failed (exit {res.returncode}): {res.stderr.strip()[-2000:]}"}
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"error": f"probe returned non-JSON output: {res.stdout[-2000:]}"}


# ----------------------------------------------------------------------------- filesystem checks
def _git_head(repo_dir):
    try:
        res = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return res.stdout.strip() if res.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_ablation_dir(ablation_dir, data_dir):
    rep = {"ablation_dir": ablation_dir, "exists": os.path.isdir(ablation_dir)}
    if not rep["exists"]:
        return rep
    for rel in ["setup.sh", "requirements.txt", "scripts/_env.sh", "src/run_inference.py", "src/eval_detect.py", "src/eval_tl.py"]:
        rep[rel] = os.path.isfile(os.path.join(ablation_dir, rel))
    rep["_env.local.sh"] = os.path.isfile(os.path.join(ablation_dir, "scripts", "_env.local.sh"))

    up = os.environ.get("BIOMEDPARSE_DIR") or os.path.join(ablation_dir, "third_party", "BiomedParse")
    upstream = {"dir": up, "exists": os.path.isdir(up), "head": None, "at_pinned_commit": None, "files": {}}
    if upstream["exists"]:
        upstream["head"] = _git_head(up)
        upstream["at_pinned_commit"] = upstream["head"] == UPSTREAM_COMMIT if upstream["head"] else None
        upstream["files"] = {f: os.path.isfile(os.path.join(up, f)) for f in UPSTREAM_FILES}
    rep["upstream"] = upstream

    ckpts = {}
    for rel in ["models/biomedparse_v2.ckpt", "models/finetuned-detect/last.ckpt"]:
        p = os.path.join(ablation_dir, rel)
        if os.path.isfile(p):
            size_gb = os.path.getsize(p) / 1e9
            ckpts[rel] = {"present": True, "size_gb": round(size_gb, 2)}
            if rel.endswith("biomedparse_v2.ckpt") and size_gb < PRETRAINED_CKPT_MIN_GB:
                ckpts[rel]["warning"] = f"smaller than {PRETRAINED_CKPT_MIN_GB} GB - truncated download?"
        else:
            ckpts[rel] = {"present": False}
    rep["checkpoints"] = ckpts

    repo_root = os.path.abspath(os.path.join(ablation_dir, "..", "..", ".."))
    rep["repo_root"] = repo_root
    rep["repo_src_medvision_bm"] = os.path.isdir(os.path.join(repo_root, "src", "medvision_bm"))
    for tj in ["tasks_MedVision-detect__train_SFT.json", "tasks_MedVision-TL__train_SFT.json"]:
        rep[f"tasks_list/{tj}"] = os.path.isfile(os.path.join(repo_root, "tasks_list", tj))

    dd = data_dir or os.environ.get("MedVision_DATA_DIR") or os.path.join(repo_root, "Data")
    rep["data_dir"] = {
        "dir": dd,
        "exists": os.path.isdir(dd),
        "Datasets/": os.path.isdir(os.path.join(dd, "Datasets")),
        "src/": os.path.isdir(os.path.join(dd, "src")),
        ".downloaded_datasets.json": os.path.isfile(os.path.join(dd, ".downloaded_datasets.json")),
        "note": "src/_paths.py imports medvision_ds from <repo>/Data/src regardless of MedVision_DATA_DIR",
    }
    return rep


def check_env_vars():
    out = {}
    for k in ENV_VARS:
        v = os.environ.get(k)
        if v is None:
            out[k] = None
        elif k == "HF_TOKEN":
            out[k] = "set (hidden)" + (" - trailing newline!" if v.endswith("\n") else "")
        else:
            out[k] = v
    return out


# ----------------------------------------------------------------------------- reporting
def compare_versions(modules):
    rows = []
    dist_by_import = {imp: dist.split(" ")[0] for imp, dist in REQUIRED + OPTIONAL}
    for imp, info in modules.items():
        dist = dist_by_import.get(imp, imp)
        pin = PINS.get(dist)
        inst = info.get("version")
        if info.get("error"):
            status = "MISSING"
        elif pin is None:
            status = "ok (unpinned)"
        elif inst == pin:
            status = "ok"
        elif inst and pin.split("+")[0] == str(inst).split("+")[0]:
            status = f"MISMATCH build (pin {pin})"
        else:
            status = f"MISMATCH (pin {pin})"
        rows.append((imp, dist, inst, pin, status, info.get("error")))
    return rows


def print_report(report):
    probe_rep = report["probe"]
    if "error" in probe_rep:
        print(f"[probe] ERROR: {probe_rep['error']}")
    else:
        print(f"[python] {probe_rep['python']} ({probe_rep['python_version']})")
        t = probe_rep.get("torch") or {}
        if t:
            print(f"[torch] version={t.get('version')} cuda_build={t.get('cuda_build')} "
                  f"cuda_available={t.get('cuda_available')} devices={t.get('device_count')}"
                  + (f" error={t.get('error')}" if t.get("error") else ""))
        print("\n[packages]  import  (distribution)  installed  ->  status")
        for imp, dist, inst, pin, status, err in report["versions"]:
            req = "REQUIRED" if imp in {r[0] for r in REQUIRED} else "optional"
            line = f"  {imp:<16} ({dist:<20}) {str(inst):<14} {status}   [{req}]"
            if err and status == "MISSING":
                line += f"  <- {err.splitlines()[0][:100]}"
            print(line)
    if report.get("ablation"):
        a = report["ablation"]
        print(f"\n[ablation dir] {a['ablation_dir']}  exists={a['exists']}")
        if a["exists"]:
            for k in ["setup.sh", "requirements.txt", "scripts/_env.sh", "src/run_inference.py", "src/eval_detect.py", "src/eval_tl.py", "_env.local.sh"]:
                print(f"  {k:<28} {a[k]}")
            up = a["upstream"]
            print(f"  upstream {up['dir']}\n    exists={up['exists']} head={up['head']} at_pinned_commit={up['at_pinned_commit']} (pin {UPSTREAM_COMMIT[:7]})")
            for f, ok in up["files"].items():
                print(f"    {f:<40} {ok}")
            for rel, info in a["checkpoints"].items():
                print(f"  {rel:<40} present={info['present']}" + (f" size={info['size_gb']} GB" if info.get("present") else "")
                      + (f"  WARNING: {info['warning']}" if info.get("warning") else ""))
            print(f"  repo_root={a['repo_root']} src/medvision_bm={a['repo_src_medvision_bm']}")
            for tj in ["tasks_list/tasks_MedVision-detect__train_SFT.json", "tasks_list/tasks_MedVision-TL__train_SFT.json"]:
                print(f"  {tj:<56} {a[tj]}")
            d = a["data_dir"]
            print(f"  data_dir={d['dir']} exists={d['exists']} Datasets/={d['Datasets/']} src/={d['src/']} .downloaded_datasets.json={d['.downloaded_datasets.json']}")
    print("\n[env vars]")
    for k, v in report["env"].items():
        print(f"  {k:<26} {'<unset>' if v is None else v}")
    print("\n[summary]")
    for m in report["missing_required"]:
        print(f"  MISSING required import: {m}")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
    print(f"  result: {'FAIL' if report['missing_required'] else 'OK'} "
          f"({len(report['missing_required'])} required import(s) missing, {len(report['warnings'])} warning(s))")


def build_parser():
    p = argparse.ArgumentParser(
        description="Check an interpreter and an ablation folder against the BiomedParse-ablation pins "
                    "(torch/CUDA, detectron2, lightning, transformers, huggingface-hub, medvision_ds, upstream commit, "
                    "checkpoint, env knobs). CPU-safe; exit 1 when a required import is missing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--python", default=None, help="interpreter to probe (e.g. the conda env's python); default: this one")
    p.add_argument("--ablation-dir", default=None, help="ablation folder (contains setup.sh, src/, third_party/, models/)")
    p.add_argument("--data-dir", default=None, help="MedVision data directory (default: $MedVision_DATA_DIR or <repo>/Data)")
    p.add_argument("--json", action="store_true", help="print the full report as JSON instead of text")
    # internal
    p.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--extra-path", action="append", default=[], help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.probe:
        print(json.dumps(probe(args.extra_path)))
        return 0

    ablation_dir = os.path.abspath(args.ablation_dir) if args.ablation_dir else None
    if ablation_dir and not os.path.isdir(ablation_dir):
        print(f"error: --ablation-dir does not exist: {ablation_dir}", file=sys.stderr)
        return 2
    if args.python and not (os.path.isfile(args.python) and os.access(args.python, os.X_OK)):
        print(f"error: --python is not an executable file: {args.python}", file=sys.stderr)
        return 2

    extra_paths = []
    if ablation_dir:  # mirror src/_paths.add_medvision_to_path(): <repo>/Data/src then <repo>/src
        repo_root = os.path.abspath(os.path.join(ablation_dir, "..", "..", ".."))
        extra_paths = [os.path.join(repo_root, "Data", "src"), os.path.join(repo_root, "src")]
        if args.data_dir:
            extra_paths.insert(0, os.path.join(os.path.abspath(args.data_dir), "src"))

    report = {
        "pins": PINS,
        "upstream_commit_pin": UPSTREAM_COMMIT,
        "probe": run_probe(args.python, extra_paths),
        "ablation": check_ablation_dir(ablation_dir, args.data_dir) if ablation_dir else None,
        "env": check_env_vars(),
        "missing_required": [],
        "warnings": [],
        "versions": [],
    }
    if "error" in report["probe"]:
        report["missing_required"].append(f"probe failed: {report['probe']['error']}")
    else:
        report["versions"] = compare_versions(report["probe"]["modules"])
        req_names = {r[0] for r in REQUIRED}
        for imp, dist, inst, pin, status, err in report["versions"]:
            if status == "MISSING" and imp in req_names:
                report["missing_required"].append(f"{imp} ({dist})")
            elif status.startswith("MISMATCH"):
                report["warnings"].append(f"{dist} installed {inst}, pinned {pin}")
        t = report["probe"].get("torch") or {}
        if t and not t.get("cuda_available", False):
            report["warnings"].append("torch.cuda.is_available() is False - inference/fine-tuning need a CUDA GPU")
    a = report["ablation"]
    if a and a["exists"]:
        up = a["upstream"]
        if not up["exists"]:
            report["warnings"].append("upstream BiomedParse checkout missing (run setup.sh or set BIOMEDPARSE_DIR)")
        elif up["at_pinned_commit"] is False:
            report["warnings"].append(f"upstream at {up['head']}, pinned {UPSTREAM_COMMIT}")
        if not a["checkpoints"]["models/biomedparse_v2.ckpt"]["present"]:
            report["warnings"].append("models/biomedparse_v2.ckpt missing (ensure_pretrained_ckpt downloads it, ~4.2 GB)")
        elif a["checkpoints"]["models/biomedparse_v2.ckpt"].get("warning"):
            report["warnings"].append("models/biomedparse_v2.ckpt " + a["checkpoints"]["models/biomedparse_v2.ckpt"]["warning"])
        if not a["repo_src_medvision_bm"]:
            report["warnings"].append("<repo>/src/medvision_bm not found - the ablation folder must live inside a MedVision checkout")
        if not a["data_dir"]["exists"]:
            report["warnings"].append(f"data directory missing: {a['data_dir']['dir']}")
    if report["env"].get("TASK") == "tl" and report["env"].get("MedVision_ACK_RELEASE") is None:
        report["warnings"].append("TASK=tl but MedVision_ACK_RELEASE is unset (the launchers export 1.4.0)")
    if isinstance(report["env"].get("HF_TOKEN"), str) and "trailing newline" in report["env"]["HF_TOKEN"]:
        report["warnings"].append("HF_TOKEN ends with a newline - strip it (tr -d '\\n')")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
    return 1 if report["missing_required"] else 0


if __name__ == "__main__":
    sys.exit(main())
