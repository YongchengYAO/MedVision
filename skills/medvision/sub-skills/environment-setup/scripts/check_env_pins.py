#!/usr/bin/env python3
"""check_env_pins.py -- compare installed package versions against a MedVision
requirements pin set. Read-only: it never installs, upgrades or removes anything.

Purpose
    The MedVision eval/SFT stacks are pinned per model (requirements_eval_<key>.txt /
    requirements_sft_<key>.txt). Several steps silently drift an environment away
    from those pins (a wheelhouse `pip install --force-reinstall` without --no-deps,
    an `install_medvision_ds` run that lifts huggingface_hub, a vLLM install that
    pulls a different torch). This script prints installed vs pinned versions for
    the packages that matter and exits 1 on any mismatch, so it can gate a launcher.

Packages tracked
    torch, torchvision, vllm, transformers, accelerate, huggingface_hub, datasets,
    medvision_bm, medvision_ds, flash-attn (plus numpy/protobuf/xformers when pinned).

Prerequisites
    Python >= 3.8 with only the standard library. `packaging` is used when present
    for exact PEP 440 specifier matching; otherwise a base-version string compare
    is used (a pin `torch==2.6.0` still accepts an installed `2.6.0+cu124`).

Usage
    check_env_pins.py --requirements <path/to/requirements_eval_qwen25vl.txt>
    check_env_pins.py --model qwen25vl --repo-root <repo>      # <repo>/requirements/requirements_eval_qwen25vl.txt
    check_env_pins.py --model sft_qwen25vl --repo-root <repo>  # <repo>/requirements/requirements_sft_qwen25vl.txt
    check_env_pins.py --model qwen25vl                         # falls back to the embedded pin snapshot
    check_env_pins.py --requirements req.txt --python /path/to/other/env/bin/python
    check_env_pins.py --list-models

Exit codes
    0 = every pinned tracked package matches; 1 = at least one mismatch or a pinned
    package is missing; 2 = usage / file error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

TRACKED = [
    "torch",
    "torchvision",
    "vllm",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "datasets",
    "medvision_bm",
    "medvision_ds",
    "flash-attn",
]
# Also reported when the requirements file pins them.
EXTRA_IF_PINNED = ["numpy", "protobuf", "xformers", "trl", "peft", "bitsandbytes", "deepspeed"]

# Model key -> requirements file name (repository `requirements/` directory).
MODEL_TO_REQUIREMENTS = {
    "claude": "requirements_eval_claude.txt",
    "gemini": "requirements_eval_gemini.txt",
    "gemma3": "requirements_eval_gemma3.txt",
    "gemma4": "requirements_eval_gemma4.txt",
    "glm4v": "requirements_eval_glm4v.txt",
    "gpt": "requirements_eval_gpt.txt",
    "healthgpt": "requirements_eval_healthgpt.txt",
    "huatuogpt_vision": "requirements_eval_huatuogpt_vision.txt",
    "internvl3": "requirements_eval_internvl3.txt",
    "kimi": "requirements_eval_kimi.txt",
    "lingshu": "requirements_eval_lingshu.txt",
    "llama3_vision": "requirements_eval_llama3_vision.txt",
    "llava_onevision": "requirements_eval_llava_onevision.txt",
    "llavamed": "requirements_eval_llavamed.txt",
    "meddr": "requirements_eval_meddr.txt",
    "medgemma": "requirements_eval_medgemma.txt",
    "medvision-v0": "requirements_eval_medvision-v0.txt",
    "minimax-m3-int4": "requirements_eval_minimax-m3-int4.txt",
    "qwen25vl": "requirements_eval_qwen25vl.txt",
    "qwen25vl_update1": "requirements_eval_qwen25vl_update1.txt",
    "qwen3vl": "requirements_eval_qwen3vl.txt",
    "sft_gemma4": "requirements_sft_gemma4.txt",
    "sft_medgemma": "requirements_sft_medgemma.txt",
    "sft_qwen25vl": "requirements_sft_qwen25vl.txt",
    "sft_qwen3.6vl": "requirements_sft_qwen3.6vl.txt",
}

# Snapshot of the tracked pins per requirements file (repository state at the time this
# skill was generated). Used only when --model is given and the file cannot be found.
EMBEDDED_PINS = {
    "requirements_eval_claude.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "0.36.0", "numpy": "2.4.6", "protobuf": "6.33.6", "transformers": "4.57.1"},
    "requirements_eval_gemini.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "0.36.0", "numpy": "2.4.6", "protobuf": "6.33.6", "transformers": "4.57.1"},
    "requirements_eval_gemma3.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.1", "vllm": "0.10.2", "xformers": "0.0.32.post1"},
    "requirements_eval_gemma4.txt": {"accelerate": "1.13.0", "datasets": "3.6.0", "huggingface_hub": "1.18.0", "numpy": "2.2.6", "protobuf": "6.33.6", "torch": "2.10.0", "torchvision": "0.25.0", "transformers": "5.10.2", "vllm": "0.19.0"},
    "requirements_eval_glm4v.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "1.20.1", "numpy": "2.2.6", "protobuf": "6.33.6", "torch": "2.10.0", "torchvision": "0.25.0", "transformers": "5.12.1", "vllm": "0.19.1"},
    "requirements_eval_gpt.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "0.36.0", "numpy": "2.4.6", "protobuf": "6.33.6", "transformers": "4.57.1"},
    "requirements_eval_healthgpt.txt": {"accelerate": "0.27.0", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "1.26.4", "protobuf": "3.20.0", "torch": "2.6.0", "torchvision": "0.21.0"},
    "requirements_eval_huatuogpt_vision.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "1.26.4", "protobuf": "3.20.0", "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.40.0"},
    "requirements_eval_internvl3.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.57.1", "vllm": "0.10.0", "xformers": "0.0.31"},
    "requirements_eval_kimi.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "0.36.0", "numpy": "2.4.6", "protobuf": "6.33.6", "transformers": "4.57.1"},
    "requirements_eval_lingshu.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "1.26.4", "protobuf": "3.20.0", "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.52.1"},
    "requirements_eval_llama3_vision.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.1", "vllm": "0.10.2", "xformers": "0.0.32.post1"},
    "requirements_eval_llava_onevision.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.57.1", "vllm": "0.10.0", "xformers": "0.0.31"},
    "requirements_eval_llavamed.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "1.26.4", "protobuf": "3.20.0", "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.37.2"},
    "requirements_eval_meddr.txt": {"accelerate": "0.34.2", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "1.26.4", "protobuf": "3.20.0", "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.37.2"},
    "requirements_eval_medgemma.txt": {"accelerate": "1.10.1", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.9.0", "torchvision": "0.24.0", "transformers": "4.57.1"},
    "requirements_eval_medvision-v0.txt": {"accelerate": "1.9.0", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.54.1", "vllm": "0.10.0", "xformers": "0.0.31"},
    "requirements_eval_minimax-m3-int4.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "1.20.1", "numpy": "2.3.5", "protobuf": "6.33.6", "torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.12.1"},
    "requirements_eval_qwen25vl.txt": {"accelerate": "1.9.0", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.54.1", "vllm": "0.10.0", "xformers": "0.0.31"},
    "requirements_eval_qwen25vl_update1.txt": {"accelerate": "1.9.0", "datasets": "3.6.0", "huggingface_hub": "1.3.3", "numpy": "2.2.6", "protobuf": "6.33.4", "torch": "2.9.1", "torchvision": "0.24.1", "transformers": "5.0.0rc2", "vllm": "0.14.0"},
    "requirements_eval_qwen3vl.txt": {"accelerate": "1.13.0", "datasets": "3.6.0", "huggingface_hub": "0.36.0", "numpy": "2.2.6", "protobuf": "6.33.6", "torch": "2.8.0", "torchvision": "0.23.0", "transformers": "4.57.0", "vllm": "0.11.0", "xformers": "0.0.32.post1"},
    "requirements_sft_gemma4.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "1.22.0", "numpy": "2.4.6", "protobuf": "6.33.0", "torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "transformers": "5.5.0", "trl": "0.19.1"},
    "requirements_sft_medgemma.txt": {"accelerate": "1.9.0", "datasets": "3.6.0", "huggingface_hub": "0.36.0", "numpy": "2.4.6", "protobuf": "6.33.0", "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.54.0", "trl": "0.19.1"},
    "requirements_sft_qwen25vl.txt": {"accelerate": "1.11.0", "datasets": "3.6.0", "huggingface_hub": "0.35.3", "numpy": "2.2.6", "protobuf": "6.33.0", "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.54.0", "trl": "0.19.1"},
    "requirements_sft_qwen3.6vl.txt": {"accelerate": "1.14.0", "datasets": "3.6.0", "huggingface_hub": "1.22.0", "numpy": "2.4.6", "protobuf": "6.33.0", "torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "transformers": "5.5.0", "trl": "0.19.1"},
}

# Probe executed either in-process or in the interpreter given by --python.
PROBE = r"""
import importlib, importlib.metadata as md, json, sys
def norm(n): return n.lower().replace("-", "_").replace(".", "_")
def version_of(name):
    for cand in {name, name.replace("-", "_"), name.replace("_", "-")}:
        try:
            return md.version(cand)
        except md.PackageNotFoundError:
            pass
    return None
def file_of(mod):
    try:
        return importlib.import_module(mod).__file__
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return f"<import failed: {type(exc).__name__}: {exc}>"
names = json.loads(sys.argv[1])
out = {"python": sys.version.split()[0], "executable": sys.executable,
       "versions": {n: version_of(n) for n in names},
       "files": {m: file_of(m) for m in ("medvision_bm", "medvision_ds")}}
print(json.dumps(out))
"""


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name.strip().lower())


def parse_requirements(path: str) -> dict[str, str]:
    """Return {normalized_name: specifier} for every requirement line with a version spec."""
    pins: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "git+", "http")):
                continue
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$", line)
            if not m:
                continue
            name, spec = m.group(1), m.group(3).strip()
            if spec:
                pins[_norm(name)] = spec
    return pins


def spec_matches(spec: str, installed: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore
        from packaging.version import Version  # type: ignore

        return SpecifierSet(spec).contains(Version(installed), prereleases=True)
    except Exception:  # packaging missing or unparsable -> lenient fallback
        m = re.match(r"^==\s*([^,\s]+)$", spec)
        if not m:
            return True  # cannot evaluate ranges without packaging; do not fail
        pinned = m.group(1)
        if "+" in pinned:
            return pinned == installed
        return installed.split("+", 1)[0] == pinned


def run_probe(python: str | None, names: list[str]) -> dict:
    if python is None:
        proc = subprocess.run([sys.executable, "-c", PROBE, json.dumps(names)], capture_output=True, text=True)
    else:
        proc = subprocess.run([python, "-c", PROBE, json.dumps(names)], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"error: version probe failed in interpreter {python or sys.executable}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def hub_transformers_hint(versions: dict[str, str | None]) -> str | None:
    tf, hub = versions.get("transformers"), versions.get("huggingface_hub")
    if not tf or not hub:
        return None
    try:
        tf_major = int(tf.split(".")[0])
        hub_parts = [int(p) for p in re.findall(r"\d+", hub)[:2]]
    except ValueError:
        return None
    if tf_major == 4 and hub_parts[0] >= 1:
        return (f"transformers {tf} requires huggingface_hub<1.0 but {hub} is installed "
                "-> expect `ImportError: cannot import name 'is_offline_mode'` / `huggingface-hub<1.0 is required`; "
                "re-pin: pip install \"huggingface_hub==0.36.0\" (check the model's requirements file first)")
    if tf_major >= 5 and (hub_parts[0] < 1 or (hub_parts[0] == 1 and hub_parts[1] < 5)):
        return (f"transformers {tf} requires huggingface_hub>=1.5 but {hub} is installed "
                "-> lift huggingface_hub to the version pinned in the model's requirements file")
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Read-only: nothing is installed. Exit 1 on any mismatch.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--requirements", metavar="FILE", help="requirements file to compare against")
    src.add_argument("--model", metavar="KEY", help="model key mapped to requirements_eval_<KEY>.txt (or sft_<KEY>); see --list-models")
    ap.add_argument("--repo-root", metavar="DIR", default=None,
                    help="MedVision checkout used to locate requirements/<file> for --model (default: current directory)")
    ap.add_argument("--python", metavar="INTERP", default=None, help="probe this interpreter instead of the current one")
    ap.add_argument("--list-models", action="store_true", help="print the model-key table and exit")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    args = ap.parse_args(argv)

    if args.list_models:
        for k, v in MODEL_TO_REQUIREMENTS.items():
            print(f"{k:20s} {v}")
        return 0

    pins: dict[str, str] = {}
    source = "(no requirements given: reporting installed versions only)"
    if args.requirements:
        if not os.path.isfile(args.requirements):
            print(f"error: requirements file not found: {args.requirements}", file=sys.stderr)
            return 2
        pins = parse_requirements(args.requirements)
        source = args.requirements
    elif args.model:
        fname = MODEL_TO_REQUIREMENTS.get(args.model)
        if fname is None:
            print(f"error: unknown model key '{args.model}'; use --list-models", file=sys.stderr)
            return 2
        root = args.repo_root or os.getcwd()
        candidate = os.path.join(root, "requirements", fname)
        if os.path.isfile(candidate):
            pins = parse_requirements(candidate)
            source = candidate
        else:
            pins = {_norm(k): f"=={v}" for k, v in EMBEDDED_PINS[fname].items()}
            source = f"embedded snapshot of {fname} (file not found under {root}/requirements)"

    names = list(TRACKED) + [n for n in EXTRA_IF_PINNED if _norm(n) in pins]
    try:
        probe = run_probe(args.python, names)
    except FileNotFoundError:
        print(f"error: interpreter not found: {args.python}", file=sys.stderr)
        return 2

    rows = []
    mismatches = 0
    for name in names:
        installed = probe["versions"].get(name)
        spec = pins.get(_norm(name))
        if spec is None:
            status = "installed" if installed else "absent"
        elif installed is None:
            status, mismatches = "MISSING (pinned)", mismatches + 1
        elif spec_matches(spec, installed):
            status = "ok"
        else:
            status, mismatches = "MISMATCH", mismatches + 1
        rows.append((name, spec or "-", installed or "-", status))

    hint = hub_transformers_hint(probe["versions"])
    files = probe["files"]
    editable_notes = []
    for mod, path in files.items():
        if isinstance(path, str) and "site-packages" in path:
            editable_notes.append(f"{mod} imports from site-packages ({path}); edits to a source checkout will NOT be picked up")
        elif isinstance(path, str) and path.startswith("<import failed"):
            editable_notes.append(f"{mod}: {path}")

    if args.json:
        print(json.dumps({"source": source, "python": probe["python"], "executable": probe["executable"],
                          "rows": [dict(zip(("package", "pinned", "installed", "status"), r)) for r in rows],
                          "module_files": files, "hint": hint, "mismatches": mismatches}, indent=2))
    else:
        print(f"Pins source : {source}")
        print(f"Interpreter : {probe['executable']} (python {probe['python']})")
        w = max(len(r[0]) for r in rows)
        print(f"{'package':{w}s}  {'pinned':22s}  {'installed':22s}  status")
        for r in rows:
            print(f"{r[0]:{w}s}  {r[1]:22s}  {r[2]:22s}  {r[3]}")
        for mod, path in files.items():
            print(f"{mod}.__file__ = {path}")
        for note in editable_notes:
            print(f"note: {note}")
        if hint:
            print(f"hint: {hint}")
        print(f"Result: {mismatches} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
