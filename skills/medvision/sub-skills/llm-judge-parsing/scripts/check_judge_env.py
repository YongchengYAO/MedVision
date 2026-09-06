#!/usr/bin/env python3
"""Check an interpreter against what the MedVision LLM-judge pipeline needs.

Purpose
-------
The LLM-as-judge second-pass parser (benchmark step 4) runs EVERY stage under one
interpreter, chosen by the ``PYTHON`` environment variable. That interpreter must
hold (a) a vLLM matching the judge registry's pin and a CUDA-capable torch for the
GPU stage, and (b) the CPU-stage imports (``medvision_bm`` with ``datasets`` and
``nibabel``, plus ``medvision_ds``) for the report stages. The single most common
failure is an unset ``PYTHON`` -- the first ``python3`` on PATH wins -- which shows
up as "No module named 'vllm'" or "no CUDA device" on a box with four GPUs.

This script probes a TARGET interpreter in a subprocess and reports:

* Python version
* vllm version vs the registry pin (FAIL when missing, MISMATCH when different)
* transformers version vs the reader's expected major line
* torch version, CUDA build, device count/names, and whether a CUDA allocation
  succeeds (the "driver too old" case fails here, not at device_count)
* whether the CPU-stage imports resolve (``medvision_bm.utils.parse_utils.cal_metrics``,
  ``datasets``, ``nibabel``, ``yaml``, ``medvision_ds``)
* whether ``judge_config`` imports from ``--llm-parsing-dir`` and what it registers
* the recommended ``export PYTHON=...`` line

Prerequisites
-------------
Standard library only. ``--help`` and the whole report work without vllm or torch
in the interpreter running THIS script; the target is probed in a subprocess.

Where the pins come from
------------------------
The judge registry (``JUDGE_MODELS``) lives in the repository checkout at
``<repo>/script/llm-parsing/judge_config.py``. It is NOT part of the installed
``medvision_bm`` package, so it cannot be located through the package. Pass
``--llm-parsing-dir <repo>/script/llm-parsing`` or ``--repo-root <repo>`` to read
the live registry and its requirements files. Without either, a built-in snapshot
of the registry (recorded 2026-09-04: key ``gemma-4-31b``, ``vllm==0.19.0``,
``transformers==5.10.2``, ``torch==2.10.0``, ``tensor_parallel=2``,
``out_suffix=_gemma-4-31b``) is used and labelled as a snapshot.

Examples
--------
  python check_judge_env.py --python <repo>/.cache/judge-env_gemma-4-31b/bin/python \\
      --repo-root <repo>
  python check_judge_env.py --python /path/to/judge-env/bin/python --json
  python check_judge_env.py                       # probe the current interpreter
  python check_judge_env.py --skip-cuda-alloc     # enumerate GPUs, do not allocate

Exit codes
----------
0  vllm present and equal to the pin (CUDA absence is a WARNING: CPU stages work)
1  vllm missing, or the target interpreter could not be probed
2  vllm present but not equal to the registry pin
3  bad arguments (interpreter not found, unreadable llm-parsing dir, unknown --judge)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

# Snapshot of JUDGE_MODELS as read from the repository on 2026-09-04. Used only
# when no live judge_config.py is reachable; the report says so when it applies.
_REGISTRY_SNAPSHOT: Dict[str, Dict[str, Any]] = {
    "gemma-4-31b": {
        "hf_id": "google/gemma-4-31B-it",
        "out_suffix": "_gemma-4-31b",
        "tensor_parallel": 2,
        "requirements": "requirements-gemma-4-31b.txt",
        "post_requirements": "requirements-gemma-4-31b-post.txt",
        "torch_pin": "torch==2.10.0",
        "transformers_major": 5,
        "vllm_pin": "0.19.0",
        "transformers_pin": "5.10.2",
        "env": ("vllm==0.19.0 (requirements-gemma-4-31b.txt) then transformers==5.10.2 "
                "(requirements-gemma-4-31b-post.txt, a SECOND pip pass)"),
    }
}
_SNAPSHOT_DEFAULT_KEY = "gemma-4-31b"


class BadArgs(Exception):
    """A caller mistake (unknown judge key); reported as exit code 3."""

# Executed under the TARGET interpreter. Prints one JSON object on stdout.
_PROBE = r"""
import importlib, json, sys
out = {"python": sys.version.split()[0], "executable": sys.executable, "modules": {}}
for mod in ("vllm", "transformers", "torch", "datasets", "nibabel", "yaml",
            "medvision_bm", "medvision_ds"):
    try:
        m = importlib.import_module(mod)
        out["modules"][mod] = {"version": getattr(m, "__version__", "present"), "error": None}
    except BaseException as e:  # a broken CUDA build can raise SystemExit-like errors
        out["modules"][mod] = {"version": None, "error": f"{type(e).__name__}: {e}"[:300]}
try:
    from medvision_bm.utils.parse_utils import cal_metrics  # noqa: F401
    out["cal_metrics"] = {"ok": True, "error": None}
except BaseException as e:
    out["cal_metrics"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
cuda = {"torch_cuda": None, "count": 0, "names": [], "alloc_ok": None, "error": None}
if out["modules"]["torch"]["version"]:
    try:
        import torch
        cuda["torch_cuda"] = torch.version.cuda
        cuda["count"] = torch.cuda.device_count()
        if cuda["count"]:
            cuda["names"] = [torch.cuda.get_device_name(i) for i in range(cuda["count"])]
            if __SKIP_ALLOC__:
                cuda["alloc_ok"] = None
            else:
                try:
                    torch.zeros(1, device="cuda")
                    cuda["alloc_ok"] = True
                except BaseException as e:
                    cuda["alloc_ok"] = False
                    msg = str(e).strip().splitlines()
                    cuda["error"] = (msg[-1] if msg else type(e).__name__)[:300]
    except BaseException as e:
        cuda["error"] = f"{type(e).__name__}: {e}"[:300]
out["cuda"] = cuda
print(json.dumps(out))
"""


def _parse_pin(req_path: str, package: str) -> Optional[str]:
    """Return the ``==`` pin for ``package`` in a requirements file, or None."""
    if not os.path.isfile(req_path):
        return None
    pat = re.compile(r"^\s*" + re.escape(package) + r"\s*==\s*([^\s#;]+)", re.IGNORECASE)
    with open(req_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                return m.group(1)
    return None


def load_registry(llm_parsing_dir: Optional[str], judge: Optional[str]) -> Dict[str, Any]:
    """Load the judge registry entry, live from judge_config.py when possible.

    Returns a dict with keys: source, key, entry (dict), vllm_pin, transformers_pin,
    registered (list of keys), import_error (str or None).
    """
    result: Dict[str, Any] = {"source": "snapshot", "import_error": None}
    if llm_parsing_dir:
        cfg_path = os.path.join(llm_parsing_dir, "judge_config.py")
        if not os.path.isfile(cfg_path):
            result["import_error"] = f"{cfg_path} not found"
        else:
            try:
                spec = importlib.util.spec_from_file_location("medvision_judge_config_probe", cfg_path)
                mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                assert spec and spec.loader
                spec.loader.exec_module(mod)  # judge_config imports only `os`
                models = dict(getattr(mod, "JUDGE_MODELS"))
                default_key = getattr(mod, "JUDGE_DEFAULT_KEY")
                key = judge or default_key
                if key not in models:
                    # The live registry WAS readable, so falling back to the
                    # built-in snapshot would answer a question nobody asked.
                    raise BadArgs(f"judge {key!r} is not registered in {cfg_path}; "
                                  f"registered: {', '.join(sorted(models))}")
                else:
                    entry = dict(models[key])
                    entry["vllm_pin"] = _parse_pin(
                        os.path.join(llm_parsing_dir, entry.get("requirements", "")), "vllm")
                    entry["transformers_pin"] = _parse_pin(
                        os.path.join(llm_parsing_dir, entry.get("post_requirements") or ""),
                        "transformers") or _parse_pin(
                        os.path.join(llm_parsing_dir, entry.get("requirements", "")), "transformers")
                    result.update({
                        "source": f"live ({cfg_path})",
                        "key": key,
                        "entry": entry,
                        "registered": sorted(models),
                        "default_key": default_key,
                    })
                    return result
            except BadArgs:
                raise
            except Exception as exc:  # pragma: no cover - diagnostic path
                result["import_error"] = f"{type(exc).__name__}: {exc}"
    key = judge or _SNAPSHOT_DEFAULT_KEY
    entry = _REGISTRY_SNAPSHOT.get(key)
    if entry is None:
        raise BadArgs(f"judge {key!r} is not in the built-in snapshot "
                      f"({', '.join(sorted(_REGISTRY_SNAPSHOT))}); pass --repo-root or "
                      f"--llm-parsing-dir to read a live registry.")
    result.update({
        "key": key,
        "entry": dict(entry),
        "registered": sorted(_REGISTRY_SNAPSHOT),
        "default_key": _SNAPSHOT_DEFAULT_KEY,
    })
    return result


def probe(python: str, timeout: int, skip_alloc: bool) -> Dict[str, Any]:
    """Run the probe under ``python`` and return its JSON report."""
    code = _PROBE.replace("__SKIP_ALLOC__", "True" if skip_alloc else "False")
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"probe of {python} timed out after {timeout}s (importing vllm can be "
                         f"slow; raise --timeout)")
    stdout = proc.stdout.strip().splitlines()
    for line in reversed(stdout):
        if line.startswith("{"):
            try:
                data = json.loads(line)
                data["_stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
                return data
            except json.JSONDecodeError:
                continue
    raise SystemExit(f"probe of {python} produced no report (exit {proc.returncode}).\n"
                     f"stderr tail:\n" + "\n".join(proc.stderr.strip().splitlines()[-10:]))


def _norm_version(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return str(v).split("+", 1)[0]


def build_report(python: str, reg: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    entry = reg["entry"]
    mods = data["modules"]
    vllm_v = mods["vllm"]["version"]
    vllm_pin = entry.get("vllm_pin")
    tf_v = mods["transformers"]["version"]
    tf_major_want = entry.get("transformers_major")
    tf_major_got = int(str(tf_v).split(".")[0]) if tf_v and str(tf_v)[0].isdigit() else None

    if vllm_v is None:
        vllm_status = "FAIL (vllm not importable)"
        exit_code = 1
    elif vllm_pin and _norm_version(vllm_v) != _norm_version(vllm_pin):
        vllm_status = f"MISMATCH (installed {vllm_v}, registry pins {vllm_pin})"
        exit_code = 2
    else:
        vllm_status = "OK" if vllm_pin else "OK (no pin found in requirements; version unverified)"
        exit_code = 0

    if tf_v is None:
        tf_status = "FAIL (transformers not importable)"
    elif tf_major_want is not None and tf_major_got != tf_major_want:
        tf_status = f"MISMATCH (installed {tf_v}, reader needs the {tf_major_want}.x line)"
    else:
        tf_status = "OK"

    cuda = data["cuda"]
    if mods["torch"]["version"] is None:
        cuda_status = "FAIL (torch not importable)"
    elif cuda["count"] == 0:
        cuda_status = "WARNING: no CUDA device -- prep/stage0/analyze work, smoke/pilot/full need a GPU"
    elif cuda["alloc_ok"] is False:
        cuda_status = f"FAIL: CUDA allocation failed ({cuda['error']}); torch build vs driver mismatch?"
    else:
        cuda_status = f"OK ({cuda['count']} device(s))"

    cpu_ok = data["cal_metrics"]["ok"] and mods["datasets"]["version"] and mods["nibabel"]["version"]
    cpu_status = "OK" if cpu_ok else "WARNING: Stages 2-4 (apply/summarize) will fail under this interpreter"
    ds_status = "OK" if mods["medvision_ds"]["version"] else (
        "WARNING: medvision_ds not importable (Stage 3 needs it; set MEDVISION_DS_SRC or install it)")

    return {
        "python": python,
        "python_version": data["python"],
        "registry_source": reg["source"],
        "registry_import_error": reg.get("import_error"),
        "judge_key": reg["key"],
        "registered_judges": reg.get("registered"),
        "hf_id": entry.get("hf_id"),
        "out_suffix": entry.get("out_suffix"),
        "tensor_parallel": entry.get("tensor_parallel"),
        "vllm": {"installed": vllm_v, "pin": vllm_pin, "status": vllm_status},
        "transformers": {"installed": tf_v, "pin": entry.get("transformers_pin"),
                         "expected_major": tf_major_want, "status": tf_status},
        "torch": {"installed": mods["torch"]["version"], "pin": entry.get("torch_pin"),
                  "cuda_build": cuda["torch_cuda"]},
        "cuda": {"count": cuda["count"], "names": cuda["names"], "alloc_ok": cuda["alloc_ok"],
                 "status": cuda_status},
        "cpu_stages": {"cal_metrics": data["cal_metrics"], "datasets": mods["datasets"]["version"],
                       "nibabel": mods["nibabel"]["version"], "yaml": mods["yaml"]["version"],
                       "status": cpu_status},
        "medvision_ds": {"installed": mods["medvision_ds"]["version"], "status": ds_status},
        "recommended_export": f"export PYTHON={python}",
        "exit_code": exit_code,
    }


def print_human(rep: Dict[str, Any]) -> None:
    print("=== MedVision LLM-judge environment check ===")
    print(f"interpreter   : {rep['python']} (Python {rep['python_version']})")
    print(f"registry      : {rep['registry_source']}")
    if rep["registry_import_error"]:
        print(f"                (live registry unavailable: {rep['registry_import_error']})")
    print(f"judge key     : {rep['judge_key']}  ->  {rep['hf_id']}  (out_suffix {rep['out_suffix']}, "
          f"registry tensor_parallel {rep['tensor_parallel']})")
    if rep["registered_judges"]:
        print(f"registered    : {', '.join(rep['registered_judges'])}")
    print(f"vllm          : {rep['vllm']['installed']}  pin {rep['vllm']['pin']}  -> {rep['vllm']['status']}")
    print(f"transformers  : {rep['transformers']['installed']}  pin {rep['transformers']['pin']}  "
          f"-> {rep['transformers']['status']}")
    print(f"torch         : {rep['torch']['installed']} (CUDA build {rep['torch']['cuda_build']})  "
          f"pin {rep['torch']['pin']}")
    names = ", ".join(rep["cuda"]["names"]) if rep["cuda"]["names"] else "-"
    print(f"GPUs          : {rep['cuda']['count']} [{names}]  -> {rep['cuda']['status']}")
    print(f"CPU stages    : cal_metrics={'ok' if rep['cpu_stages']['cal_metrics']['ok'] else 'FAIL'} "
          f"datasets={rep['cpu_stages']['datasets']} nibabel={rep['cpu_stages']['nibabel']} "
          f"yaml={rep['cpu_stages']['yaml']}  -> {rep['cpu_stages']['status']}")
    if not rep["cpu_stages"]["cal_metrics"]["ok"]:
        print(f"                {rep['cpu_stages']['cal_metrics']['error']}")
    print(f"medvision_ds  : {rep['medvision_ds']['installed']}  -> {rep['medvision_ds']['status']}")
    print()
    print("Use this interpreter for every driver step:")
    print(f"    {rep['recommended_export']}")
    if rep["exit_code"] == 1:
        print("\nRESULT: FAIL -- Stage 1 cannot run here. Build the judge environment (see "
              "references/judge-environment.md) and point PYTHON at it.")
    elif rep["exit_code"] == 2:
        print("\nRESULT: MISMATCH -- this vllm is not the registry pin. Judge output is not "
              "comparable across engines; rebuild the env from the pinned requirements.")
    else:
        print("\nRESULT: OK for Stage 1 imports" + ("" if rep["cuda"]["count"] else
                                                     " (GPU still required for smoke/pilot/full)"))


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check an interpreter against the MedVision LLM-judge pipeline's needs "
                    "(vllm pin, torch/CUDA, CPU-stage imports, judge registry).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n", 1)[-1])
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to probe (default: the current one). Point it at the "
                         "judge env, e.g. <repo>/.cache/judge-env_gemma-4-31b/bin/python")
    ap.add_argument("--judge", default=None, help="registry key (default: the registry's default)")
    ap.add_argument("--llm-parsing-dir", default=None,
                    help="directory holding judge_config.py and requirements-*.txt "
                         "(<repo>/script/llm-parsing). Overrides --repo-root.")
    ap.add_argument("--repo-root", default=None,
                    help="repository checkout; implies --llm-parsing-dir <repo>/script/llm-parsing")
    ap.add_argument("--timeout", type=int, default=300, help="probe timeout in seconds (default 300)")
    ap.add_argument("--skip-cuda-alloc", action="store_true",
                    help="enumerate GPUs but do not allocate a tensor on them")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    python = shutil.which(args.python) or args.python
    if not (os.path.isfile(python) and os.access(python, os.X_OK)):
        print(f"error: interpreter not found or not executable: {args.python}", file=sys.stderr)
        return 3

    llm_dir = args.llm_parsing_dir
    if llm_dir is None and args.repo_root:
        llm_dir = os.path.join(args.repo_root, "script", "llm-parsing")
    if llm_dir is not None and not os.path.isdir(llm_dir):
        print(f"error: --llm-parsing-dir {llm_dir} is not a directory", file=sys.stderr)
        return 3

    try:
        reg = load_registry(llm_dir, args.judge)
    except BadArgs as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    try:
        data = probe(python, args.timeout, args.skip_cuda_alloc)
    except SystemExit as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rep = build_report(python, reg, data)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print_human(rep)
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
