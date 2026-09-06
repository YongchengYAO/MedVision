#!/usr/bin/env python3
"""Collect safe, read-only facts about a MedVision environment.

Purpose
    Answer "is this Python ready for MedVision, and for which workflows?"
    without installing, downloading, or touching GPUs. It reports the
    `medvision_bm` / `medvision_ds` / vendored `lmms_eval` installs, the
    versions of the pinned foundation packages, GPU visibility, the `mvbm`
    console script, the MedVision environment variables, and the layout of
    the data directory when `MedVision_DATA_DIR` is set.

Prerequisites
    Run it with the Python interpreter you intend to use for MedVision
    (a plain `python` from an activated environment, or an absolute path).
    Nothing beyond the standard library is required for the report itself;
    every package probe is optional and reported as missing when absent.

Examples
    python check_medvision_env.py
    python check_medvision_env.py --json
    python check_medvision_env.py --repo-root /path/to/MedVision   # use src/ of a checkout
    python check_medvision_env.py --data-dir /path/to/Data

Exit status
    0 when `medvision_bm` imports, 1 when it does not (or when
    --require-gpu is given and no CUDA device is visible).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

CORE_PACKAGES = [
    # (import name, distribution name shown in reports)
    ("medvision_bm", "medvision_bm"),
    ("medvision_ds", "medvision_ds"),
    ("lmms_eval", "lmms_eval (vendored fork)"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("nibabel", "nibabel"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
]

# Optional stacks, grouped by the workflow that needs them. Absence is not an error.
OPTIONAL_PACKAGES = {
    "evaluation (vLLM wrappers)": ["vllm", "qwen_vl_utils", "decord", "bitsandbytes"],
    "evaluation (API wrappers)": ["anthropic", "openai", "google.genai"],
    "sft": ["trl", "peft", "deepspeed", "wandb", "flash_attn"],
    "rft parquet / verl": ["pyarrow", "verl"],
    "biomedparse ablation": ["detectron2", "lightning", "hydra"],
    "analysis": ["sklearn", "pandas", "yaml"],
}

ENV_VARS = [
    "MedVision_DATA_DIR",
    "MedVision_PLANNER_VERSION",
    "MedVision_ACK_RELEASE",
    "MedVision_FORCE_INSTALL_CODE",
    "MedVision_FORCE_DOWNLOAD_DATA",
    "MedVision_DISABLE_SAMPLE_FILTERING",
    "MedVision_DOWNLOAD_QC_FIGURES",
    "MEDVISION_RESP_CACHE",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "CUDA_VISIBLE_DEVICES",
]

SECRET_VARS = ["HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MOONSHOT_API_KEY", "SYNAPSE_TOKEN"]


def _version_of(module, dist_name: str) -> str | None:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as md_version

        return md_version(dist_name.split(" ")[0])
    except Exception:  # noqa: BLE001 - best effort
        return None


def probe_module(import_name: str, dist_name: str) -> dict:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - any import failure is a finding
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    info = {"present": True, "version": _version_of(module, dist_name)}
    location = getattr(module, "__file__", None)
    # Only the MedVision-owned packages are commonly installed editable; for them the
    # distinction matters (a site-packages copy silently shadows checkout edits).
    if location and import_name in {"medvision_bm", "medvision_ds", "lmms_eval"}:
        info["editable_install"] = "site-packages" not in location and "dist-packages" not in location
    return info


def probe_torch() -> dict:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    info = {
        "present": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        try:
            info["device_0"] = torch.cuda.get_device_name(0)
            info["compute_capability_0"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        except Exception as exc:  # noqa: BLE001
            info["device_error"] = str(exc)[:120]
    return info


def probe_nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"present": False}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        return {"present": True, "gpus": gpus, "returncode": out.returncode}
    except Exception as exc:  # noqa: BLE001
        return {"present": True, "error": str(exc)[:120]}


def probe_console_script(name: str) -> dict:
    exe = shutil.which(name)
    if not exe:
        return {"on_path": False}
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60, check=False)
        return {"on_path": True, "help_ok": out.returncode == 0}
    except Exception as exc:  # noqa: BLE001
        return {"on_path": True, "help_ok": False, "error": str(exc)[:120]}


def probe_data_dir(data_dir: str | None) -> dict:
    if not data_dir:
        return {"configured": False}
    root = Path(data_dir)
    info = {"configured": True, "path_exists": root.is_dir()}
    if not root.is_dir():
        return info
    datasets_dir = root / "Datasets"
    info["datasets_dir_exists"] = datasets_dir.is_dir()
    if datasets_dir.is_dir():
        names = sorted(p.name for p in datasets_dir.iterdir() if p.is_dir())
        info["dataset_count"] = len(names)
        info["datasets"] = names[:40]
        plans = sorted(datasets_dir.glob("*/benchmark_plan_*.json.gz"))
        info["benchmark_plan_files"] = len(plans)
    info["medvision_ds_source_present"] = (root / "src" / "medvision_ds").is_dir()
    tracker = root / ".downloaded_datasets.json"
    info["download_tracker_present"] = tracker.is_file()
    if tracker.is_file():
        try:
            data = json.loads(tracker.read_text())
            info["download_tracker_entries"] = len(data) if isinstance(data, dict) else None
        except Exception as exc:  # noqa: BLE001
            info["download_tracker_error"] = str(exc)[:120]
    info["hf_cache_present"] = (root / ".cache" / "huggingface").is_dir()
    return info


def pin_warnings(core: dict) -> list[str]:
    warnings: list[str] = []

    def ver(name: str) -> str | None:
        entry = core.get(name, {})
        return entry.get("version") if entry.get("present") else None

    def major(v: str | None) -> int | None:
        try:
            return int(str(v).split(".")[0])
        except Exception:  # noqa: BLE001
            return None

    hub, tf, ds = ver("huggingface_hub"), ver("transformers"), ver("datasets")
    if hub and tf:
        if major(tf) is not None and major(tf) < 5 and major(hub) is not None and major(hub) >= 1:
            warnings.append(
                f"transformers {tf} needs huggingface_hub<1.0 but {hub} is installed "
                "(symptom: ImportError: cannot import name 'is_offline_mode'); re-pin huggingface_hub==0.36.0."
            )
        if major(tf) is not None and major(tf) >= 5 and major(hub) is not None and major(hub) < 1:
            warnings.append(f"transformers {tf} needs huggingface_hub>=1.5 but {hub} is installed.")
    if ds and ds != "3.6.0":
        warnings.append(
            f"datasets {ds} installed; medvision_bm pins datasets==3.6.0 because trust_remote_code was removed in datasets>=4."
        )
    bm = core.get("medvision_bm", {})
    if bm.get("present") and bm.get("editable_install") is False:
        warnings.append(
            "medvision_bm is a site-packages copy (non-editable): edits under a checkout's src/ will NOT take effect "
            "until you reinstall (pip install -e <repo> --no-deps) or run with PYTHONPATH=<repo>/src."
        )
    if not core.get("medvision_ds", {}).get("present"):
        warnings.append(
            "medvision_ds missing: dataset loading, parse_outputs and the summarizers need it; install with "
            "`mvbm install mvds -d <data_dir>`."
        )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", help="Path to a MedVision checkout; its src/ is prepended to sys.path before probing.")
    parser.add_argument("--data-dir", help="Data directory to inspect (default: $MedVision_DATA_DIR).")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of text.")
    parser.add_argument("--skip-optional", action="store_true", help="Do not probe optional workflow packages (faster).")
    parser.add_argument("--require-gpu", action="store_true", help="Exit 1 when torch reports no CUDA device.")
    args = parser.parse_args()

    if args.repo_root:
        src = Path(args.repo_root).expanduser().resolve() / "src"
        sys.path.insert(0, str(src))

    report: dict = {
        "python": {"executable": sys.executable, "version": platform.python_version(), "platform": platform.platform()},
        "core": {},
        "torch": probe_torch(),
        "nvidia_smi": probe_nvidia_smi(),
        "console_scripts": {"mvbm": probe_console_script("mvbm")},
        "env": {name: os.environ.get(name) for name in ENV_VARS},
        "secrets_set": {name: bool(os.environ.get(name)) for name in SECRET_VARS},
        "secrets_with_trailing_whitespace": [
            name for name in SECRET_VARS if os.environ.get(name) and os.environ[name] != os.environ[name].strip()
        ],
    }
    for import_name, dist_name in CORE_PACKAGES:
        report["core"][import_name] = probe_module(import_name, dist_name) if import_name != "torch" else report["torch"]
    if not args.skip_optional:
        report["optional"] = {
            group: {name: probe_module(name, name).get("present", False) for name in names}
            for group, names in OPTIONAL_PACKAGES.items()
        }
    report["data_dir"] = probe_data_dir(args.data_dir or os.environ.get("MedVision_DATA_DIR"))
    report["warnings"] = pin_warnings(report["core"])
    if report["secrets_with_trailing_whitespace"]:
        report["warnings"].append(
            "These secrets carry leading/trailing whitespace (pod-injected newline): "
            + ", ".join(report["secrets_with_trailing_whitespace"])
            + ". Sanitize with: export VAR=\"$(printf '%s' \"$VAR\" | tr -d '[:space:]')\"."
        )
    if not report["torch"].get("cuda_available"):
        report["warnings"].append(
            "No CUDA device visible: local VLM evaluation, SFT/RFT training and the LLM judge cannot run here; "
            "parsing, summarizing, task-list work and analyses still can."
        )

    ok = bool(report["core"]["medvision_bm"].get("present"))
    if args.require_gpu and not report["torch"].get("cuda_available"):
        ok = False
    report["status"] = "ok" if ok else "not-ready"

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if ok else 1

    print(f"MedVision environment check  ->  {report['status'].upper()}")
    print(f"python  : {report['python']['version']}  ({report['python']['executable']})")
    for name, entry in report["core"].items():
        if entry.get("present"):
            extra = ""
            if name == "torch":
                extra = f"  cuda_build={entry.get('cuda_build')} cuda_available={entry.get('cuda_available')} devices={entry.get('device_count')}"
            elif "editable_install" in entry:
                extra = "  (editable)" if entry["editable_install"] else "  (site-packages copy)"
            print(f"  [ok]      {name:<17} {entry.get('version') or '?'}{extra}")
        else:
            print(f"  [missing] {name:<17} {entry.get('error', '')}")
    smi = report["nvidia_smi"]
    print(f"nvidia-smi: {'present, ' + str(len(smi.get('gpus', []))) + ' GPU(s)' if smi.get('present') else 'not found'}")
    mv = report["console_scripts"]["mvbm"]
    print(f"mvbm      : {'on PATH, --help ok' if mv.get('help_ok') else ('on PATH but --help failed' if mv.get('on_path') else 'not on PATH')}")
    if "optional" in report:
        print("optional packages by workflow:")
        for group, names in report["optional"].items():
            present = [n for n, p in names.items() if p]
            missing = [n for n, p in names.items() if not p]
            print(f"  {group:<30} present={present or '-'} missing={missing or '-'}")
    print("environment variables:")
    for name, value in report["env"].items():
        print(f"  {name:<36} {value if value is not None else '(unset)'}")
    print("secrets set (values never printed): " + ", ".join(k for k, v in report["secrets_set"].items() if v) or "none")
    dd = report["data_dir"]
    if dd.get("configured"):
        print(
            f"data dir  : exists={dd.get('path_exists')} datasets={dd.get('dataset_count', 0)} plans={dd.get('benchmark_plan_files', 0)} "
            f"medvision_ds_src={dd.get('medvision_ds_source_present')} tracker={dd.get('download_tracker_present')}"
        )
    else:
        print("data dir  : MedVision_DATA_DIR unset (pass --data-dir to inspect one)")
    if report["warnings"]:
        print("warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
