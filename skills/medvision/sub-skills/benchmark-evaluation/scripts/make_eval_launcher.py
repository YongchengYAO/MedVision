#!/usr/bin/env python3
"""Generate a MedVision benchmark launcher (bash) for one model x one task.

Purpose
    Reproduce the skeleton of the repository's ``script/benchmark-{detect,TL,AD}/eval__*.sh``
    launchers from a machine-readable catalog (``model_catalog.json`` next to this file):
    optional conda-env creation, the medvision_bm install block (wheel build on local disk,
    editable install, or skip), ``MedVision_PLANNER_VERSION`` / ``MedVision_ACK_RELEASE``
    exports, the load-bearing "Method 1" install trio (install_medvision_ds ->
    install_vendored_lmms_eval [--lmms_eval_opt_deps] -> pip install -r requirements --no-deps),
    the ``python -m medvision_bm.benchmark.eval__<model>`` command with every flag the
    repository launcher passes, and API-key sanitising for API models.

Prerequisites
    Python >= 3.8, standard library only. The generated script needs bash, a MedVision
    checkout (``benchmark_dir``) with ``tasks_list/`` and ``requirements/``, and -- for
    open-weight models -- CUDA GPUs. Nothing is executed by this generator.

Examples
    python make_eval_launcher.py --list-models
    python make_eval_launcher.py --model qwen25vl --task detect
    python make_eval_launcher.py --model claude --task AD --api-provider openrouter --out run_claude_AD.sh
    python make_eval_launcher.py --model medgemma-27b --task TL --cuda-visible-devices 0,1 --dry-run
    python make_eval_launcher.py --model qwen25vl --task TL --model-hf-id /path/to/checkpoint \
        --model-name my-sft-run --install-mode skip --no-conda-env --method 1

Exit codes
    0 success; 1 catalog/IO error; 2 invalid arguments (unknown model, bad values).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "model_catalog.json"
TASK_CHOICES = ("detect", "TL", "AD")
CONDA_PYTHON = {  # interpreters the repository launchers actually create; all others 3.11
    "meddr": "3.9",
    "llava-med": "3.10",
    "minimax-m3": "3.12",
    "minimax-m3-int4": "3.12",
}

COMMON_PATH_FLAGS = [
    "--results_dir $result_dir",
    "--data_dir $data_dir",
    "--tasks_list_json_path $tasks_list_json_path",
    "--task_status_json_path $task_status_json_path",
]


# --------------------------------------------------------------------------- helpers
def die(msg: str, code: int = 2) -> None:
    print(f"[make_eval_launcher] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_catalog(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cat = json.load(f)
    except FileNotFoundError:
        die(f"catalog not found: {path}", 1)
    except json.JSONDecodeError as e:
        die(f"catalog is not valid JSON ({path}): {e}", 1)
    for key in ("tasks", "models"):
        if key not in cat:
            die(f"catalog missing top-level key '{key}': {path}", 1)
    return cat


def list_models(cat: dict) -> None:
    rows = []
    for key, m in cat["models"].items():
        rows.append(
            (
                key,
                m.get("backend", "?"),
                str(m.get("launcher_method", "?")),
                m.get("display_name", ""),
                m["eval_module"].split(".")[-1],
                m.get("lmms_model_key") or "-",
                "yes" if m.get("launchers") else "no",
            )
        )
    widths = [max(len(r[i]) for r in rows + [("key", "backend", "method", "display name", "entry point", "lmms key", "repo launcher")]) for i in range(7)]
    hdr = ("key", "backend", "method", "display name", "entry point", "lmms key", "repo launcher")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * widths[i] for i in range(7)))
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    print(f"\n{len(rows)} catalog entries; tasks: {', '.join(cat['tasks'].keys())}")


def sh_comment(text: str, width: int = 96) -> str:
    """Wrap text as a bash comment block."""
    import textwrap

    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("#")
            continue
        for line in textwrap.wrap(para, width=width - 2):
            out.append(f"# {line}")
    return "\n".join(out)


def resolve_provider(model: dict, requested: str | None) -> str:
    api = model.get("api") or {}
    providers = api.get("providers", {})
    if not providers:
        return ""
    if requested is None:
        return api.get("launcher_default_provider") or api.get("direct_provider")
    if requested == "direct":
        return api["direct_provider"]
    if requested in providers:
        return requested
    die(f"--api-provider '{requested}' not valid for this model; choose from: direct, {', '.join(providers)}")
    return ""  # unreachable


# --------------------------------------------------------------------------- blocks
def block_header(key: str, m: dict, task: str, tinfo: dict, method: str, install_mode: str) -> str:
    gpu_note = (
        "Requires API credentials only (no local GPU)."
        if m["backend"] == "api"
        else "Requires CUDA GPU(s); expose exactly the GPUs to use via CUDA_VISIBLE_DEVICES "
        "(vLLM models: tensor_parallel_size = number of visible GPUs; HF models: one full replica per GPU)."
    )
    lines = [
        "#!/usr/bin/env bash",
        f"# MedVision benchmark launcher -- generated by make_eval_launcher.py",
        f"# model: {key} ({m.get('display_name', '')})   task: {task} ({tinfo['task_tag']})",
        f"# entry point: python -m {m['eval_module']}   lmms_eval key: {m.get('lmms_model_key') or 'n/a'}",
        f"# method: {method}   install mode: {install_mode}",
        f"# parallelism: {m.get('parallelism', 'n/a')}",
        f"# {gpu_note}",
        "# Edit the 'Set paths and configs' block, then run:  bash <this file>  (from any directory).",
    ]
    if m.get("notes"):
        lines.append(sh_comment("NOTE: " + m["notes"]))
    return "\n".join(lines)


def block_conda(env_name: str, py: str) -> str:
    return f'''ENV_NAME="{env_name}"

# Only create the env if it doesn't already exist
source activate base
eval "$(conda shell.bash hook)"
if [ -d "$(conda info --base)/envs/${{ENV_NAME}}" ]; then
    echo "Conda env '${{ENV_NAME}}' already exists. Skipping creation."
else
    conda create -n "${{ENV_NAME}}" python=={py} -y
fi
conda activate "${{ENV_NAME}}"'''


def block_config(args, key: str, m: dict, task: str, tinfo: dict, provider: str) -> str:
    profile = m["flag_profile"]
    lines = ["# Set paths and configs"]
    bd = args.benchmark_dir or '${MEDVISION_BENCHMARK_DIR:-$PWD}'
    lines.append(f'benchmark_dir="{bd}"  # MedVision checkout (contains tasks_list/, requirements/, Data/, Results/)')
    dd = args.data_dir or '${benchmark_dir}/Data'
    lines.append(f'data_dir="{dd}"')
    if profile in ("hf_third_party", "healthgpt"):
        tp = args.third_party_dir or '${benchmark_dir}/third_party'
        lines.append(f'dir_third_party="{tp}"')

    model_name = args.model_name or m.get("default_model_name")
    if profile == "healthgpt":
        lines.append(f'model_name="{model_name}"  # run label: output folder under Results/<task_tag>/ and task-status key')
        lines.append(f'model_choice="{m["model_choice"]}"  # architecture: HealthGPT-L14 or HealthGPT-XL32')
    elif profile == "api":
        lines.append(f'model_name="{model_name}"  # output folder under Results/<task_tag>/ and task-status key')
        lines.append(f'batch_size={args.batch_size_per_gpu or m.get("batch_size", 1)}')
    else:
        hf = args.model_hf_id or m.get("default_model_hf_id")
        if not hf:
            die(f"model '{key}' has no default HF id; pass --model-hf-id <hf-id-or-local-path>")
        if not model_name:
            die(f"model '{key}' has no default model name; pass --model-name <label>")
        lines.append(f'model_hf_id="{hf}"  # Hugging Face id or local checkpoint directory')
        lines.append(f'model_name="{model_name}"  # output folder under Results/<task_tag>/ and task-status key')
        if args.lora_path:
            lines.append(f'lora_path="{args.lora_path}"')
        if profile == "tooluse":
            lines.append(f'batch_size={args.batch_size_per_gpu or m.get("batch_size", 20)}')
            lines.append(f'gpu_memory_utilization={args.gpu_memory_utilization or m.get("gpu_memory_utilization", 0.95)}')
            lines.append("max_tokens_phase1=512  # <think> + <tool_call> budget (deliberate per-phase cap)")
            lines.append("max_tokens_phase2=64   # final <answer> budget")
        else:
            lines.append(f'batch_size_per_gpu={args.batch_size_per_gpu or m.get("batch_size_per_gpu")}')
            if m.get("gpu_memory_utilization") is not None or args.gpu_memory_utilization:
                lines.append(
                    f'gpu_memory_utilization={args.gpu_memory_utilization or m.get("gpu_memory_utilization")}'
                    "  # fraction of each GPU vLLM may claim; lower it on OOM at engine start"
                )

    # model-specific extra variables (sampling, stop strings, context caps ...)
    for name, spec in (m.get("extra_vars") or {}).items():
        if name == "reshape_image_hw" and args.reshape_image_hw:
            continue
        if name == "stop_string" and args.stop_strings:
            continue
        cmt = f"  # {spec['comment']}" if spec.get("comment") else ""
        lines.append(f"{name}={spec['value']}{cmt}")
    if args.reshape_image_hw and profile != "api":
        lines.append(f'reshape_image_hw="{args.reshape_image_hw}"  # resize slices before the model sees them (H x W)')
    if args.stop_strings:
        lines.append(f"stop_string='{args.stop_strings}'  # generation halts at this string")

    # API provider + key sanitising
    if profile == "api":
        api = m["api"]
        code_var = api["model_code_flag"].lstrip("-")
        lines.append("")
        lines.append("# API provider and model code")
        for pname, pinfo in api["providers"].items():
            lines.append(f"# - {pname}: model code \"{pinfo['model_code']}\", key {' or '.join(pinfo['env_vars'])}")
        code = args.model_code or api["providers"][provider]["model_code"]
        lines.append(f'api_provider="{provider}"')
        lines.append(f'{code_var}="{code}"')
        lines.append("")
        lines.append("# API key check + sanitization (injected secrets can carry a trailing newline,")
        lines.append("# which breaks HTTP auth headers)")
        direct = api["direct_provider"]
        denv = api["providers"][direct]["env_vars"]
        others = [p for p in api["providers"] if p != direct]
        lines.append(f'if [ "${{api_provider}}" = "{direct}" ]; then')
        if len(denv) == 1:
            lines.append(f'    api_key_var="{denv[0]}"')
        else:
            lines.append(f'    if [ -n "${{{denv[0]}:-}}" ]; then')
            lines.append(f'        api_key_var="{denv[0]}"')
            lines.append("    else")
            lines.append(f'        api_key_var="{denv[1]}"')
            lines.append("    fi")
        for o in others:
            lines.append("else")
            lines.append(f'    api_key_var="{api["providers"][o]["env_vars"][0]}"')
        lines.append("fi")
        lines.append('if [ -z "${!api_key_var:-}" ]; then')
        lines.append('    echo "[Error] ${api_key_var} is not set." >&2')
        lines.append("    exit 1")
        lines.append("fi")
        lines.append('''export "${api_key_var}"="$(printf '%s' "${!api_key_var}" | tr -d '\\n')"''')

    # task bookkeeping
    task_tag = args.task_tag or tinfo["task_tag"]
    tasks_json = args.tasks_json or ('${benchmark_dir}/' + tinfo["tasks_json"])
    if profile == "api":
        limit = args.sample_limit or m.get("sample_limit", 100)
    elif profile == "tooluse":
        limit = args.sample_limit or m.get("sample_limit", 100)
    else:
        limit = args.sample_limit or 1000
    lines.append("")
    lines.append("# Other configs (safe to leave as is)")
    lines.append(f'task_tag="{task_tag}"  # Results/<task_tag>/<model_name>/ ; use a distinct tag (e.g. -limit100) for pilot runs')
    lines.append('result_dir="${benchmark_dir}/Results/${task_tag}"')
    lines.append(f'tasks_list_json_path="{tasks_json}"')
    lines.append('task_status_json_path="${benchmark_dir}/completed_tasks/completed_tasks_${task_tag}.json"')
    lines.append(f"sample_limit={limit}  # benchmark convention: 1000 per subtask (open-weight), 100 per subtask (API pilot)")
    if profile == "api":
        rs = args.reshape_image_hw or m.get("reshape_image_hw", "512x512")
        lines.append(f'reshape_image_hw="{rs}"')
    return "\n".join(lines)


def block_install(mode: str) -> str:
    if mode == "skip":
        return "set -euo pipefail\n# medvision_bm install skipped (--install-mode skip): the active environment must already provide it."
    if mode == "editable":
        return '''set -euo pipefail
# Install medvision_bm in editable mode from the checkout (fine on local disks; on shared/CephFS
# volumes prefer the wheel build below, see --install-mode wheel).
python -m pip install -e "${benchmark_dir}"'''
    return '''# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
# tree). setuptools build_py caches created dirs in a process-global memo, and on
# CephFS a build subdir can transiently vanish (async delete/recreate lag or an
# unguarded concurrent writer), after which the cache refuses to recreate it and a
# later file copy dies with: could not create '...': No such file or directory.
# A private local build dir is immune; only the shared-env install needs the lock.
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \\
    pyproject.toml MANIFEST.in LICENSE src \\
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"'''


def block_exports(args, m: dict, task: str, tinfo: dict, planner_version: str) -> str:
    lines = [f"# Pin the MedVision dataset annotation version (the loader hard-fails without it)",
             f"export MedVision_PLANNER_VERSION='{planner_version}'"]
    if tinfo.get("ack_release"):
        lines.append(f"export MedVision_ACK_RELEASE='{tinfo['ack_release']}'  # acknowledge the newer T/L annotation release while pinning an older planner")
    if args.cuda_visible_devices:
        lines.append("")
        lines.append(f'export CUDA_VISIBLE_DEVICES="{args.cuda_visible_devices}"  # vLLM: tensor_parallel_size = number of ids listed; HF: one process per id')
    profile = m["flag_profile"]
    if profile == "api":
        lines.append("")
        lines.append("# Set output token limit (thinking + output share this budget)")
        lines.append(f"max_tokens={args.max_new_tokens or m.get('default_max_tokens', 16000)}")
    elif profile == "tooluse":
        pass
    else:
        budget = args.max_new_tokens or (m.get("per_task_overrides", {}).get(task, {}) or {}).get("max_new_tokens") or m.get("default_max_tokens", 4096)
        lines.append("")
        lines.append("# Set output token limit (repository default 4096; CoT on verbose/reasoning models may need 16000+)")
        lines.append(f"max_new_tokens={budget}")
    if m.get("third_party_subdir"):
        lines.append("")
        lines.append("# Important: fix module import failure in the distributed (accelerate) subprocesses")
        lines.append(f'export PYTHONPATH="${{dir_third_party}}/{m["third_party_subdir"]}:${{PYTHONPATH:-}}"')
    return "\n".join(lines)


def eval_flags(args, m: dict) -> list[str]:
    profile = m["flag_profile"]
    extra = list(m.get("extra_args") or [])
    if profile == "api":
        api = m["api"]
        code_var = api["model_code_flag"].lstrip("-")
        flags = ["--api_provider $api_provider", f"{api['model_code_flag']} ${code_var}", "--model_name $model_name",
                 "--max_tokens $max_tokens"] + COMMON_PATH_FLAGS + ["--batch_size $batch_size", "--sample_limit $sample_limit",
                                                                     "--reshape_image_hw $reshape_image_hw"] + extra
    elif profile == "healthgpt":
        flags = ["--model_name $model_name", "--model_choice $model_choice", "--results_dir $result_dir",
                 "--dir_third_party $dir_third_party", "--data_dir $data_dir", "--tasks_list_json_path $tasks_list_json_path",
                 "--task_status_json_path $task_status_json_path", "--batch_size_per_gpu $batch_size_per_gpu",
                 "--max_new_tokens $max_new_tokens", "--sample_limit $sample_limit"] + extra
    elif profile == "tooluse":
        flags = ["--model_hf_id $model_hf_id", "--model_name $model_name", "--tasks_list_json_path $tasks_list_json_path",
                 "--results_dir $result_dir", "--task_status_json_path $task_status_json_path", "--data_dir $data_dir",
                 "--sample_limit $sample_limit", "--batch_size $batch_size", "--gpu_memory_utilization $gpu_memory_utilization",
                 "--max_tokens_phase1 $max_tokens_phase1", "--max_tokens_phase2 $max_tokens_phase2"]
        if args.lora_path:
            flags.append("--lora_path $lora_path")
        return flags
    else:  # vllm / hf / hf_third_party / minimax
        flags = ["--model_hf_id $model_hf_id", "--model_name $model_name"] + COMMON_PATH_FLAGS
        if profile == "hf_third_party":
            flags.insert(3, "--dir_third_party $dir_third_party")
        flags.append("--batch_size_per_gpu $batch_size_per_gpu")
        if m.get("gpu_memory_utilization") is not None or args.gpu_memory_utilization:
            flags.append("--gpu_memory_utilization $gpu_memory_utilization")
        flags += ["--max_new_tokens $max_new_tokens", "--sample_limit $sample_limit"] + extra
        if args.lora_path:
            flags.append("--lora_path $lora_path")
    if args.reshape_image_hw and profile != "api" and not any(f.startswith("--reshape_image_hw") for f in flags):
        flags.append("--reshape_image_hw $reshape_image_hw")
    if args.stop_strings and not any(f.startswith("--stop_strings") for f in flags):
        flags.append('--stop_strings "$stop_string"')
    return flags


def fmt_cmd(module: str, flags: list[str], first: str | None = None, indent: str = "    ", comment: bool = False) -> str:
    parts = [f"python -m {module} \\"]
    if first:
        parts.append(f"{indent}{first} \\")
    for i, fl in enumerate(flags):
        tail = " \\" if i < len(flags) - 1 else ""
        parts.append(f"{indent}{fl}{tail}")
    if comment:
        return "\n".join("# " + p for p in parts)
    return "\n".join(parts)


def block_method1_installs(m: dict) -> str:
    lines = ['python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir}"']
    od = m.get("opt_deps")
    lines.append("python -m medvision_bm.benchmark.install_vendored_lmms_eval" + (f" --lmms_eval_opt_deps {od}" if od else ""))
    req = m.get("requirements_file")
    if req:
        lines.append(f'pip install -r "${{benchmark_dir}}/{req}" --no-deps')
    else:
        lines.append("# (no pinned requirements file exists for this model in the repository; pin via `pip freeze` after a successful Method 2 run)")
    return "\n".join(lines)


DEBUG_COMMENT = """# Add these arguments for debugging:
# --env_setup_only       # install everything, then exit (no inference)
# --skip_env_setup       # do not touch the environment (Method 1 relies on this)
# --skip_update_status   # do not mark the task done in completed_tasks/*.json"""


def block_run(args, m: dict, method: str) -> str:
    module = m["eval_module"]
    flags = eval_flags(args, m)
    if method == "two-pass":
        arr = "\n".join(f"    {f}" for f in flags)
        return f'''# Eval args shared by the setup-only and run passes (defined once to avoid drift).
common_args=(
{arr}
)

# --- Step 1: built-in env setup (torch, vendored lmms_eval, medvision_ds, transformers, vLLM). Exits after setup.
python -m {module} "${{common_args[@]}}" --env_setup_only

# --- Step 2: repair / replace the environment here if the model needs a vLLM build that pip cannot
#             provide (e.g. the AWQ-INT4 MiniMax-M3 checkpoint needs a patched vLLM fork). Keep this
#             step empty when the pip vLLM already registers the architecture.

# --- Step 3: run the eval against the prepared environment
python -m {module} "${{common_args[@]}}" --skip_env_setup'''
    m1 = f'''# (Method 1) Manually install requirements before running the eval script (more robust)
# ---
{block_method1_installs(m)}

{fmt_cmd(module, flags, first="--skip_env_setup")}
# ---'''
    m2 = f'''# (Method 2) Let the eval script install its own dependencies (simpler; may pick up incompatible
# newer package versions -- see the pins in the model catalog if it breaks)
{DEBUG_COMMENT}
{fmt_cmd(module, flags)}'''
    if method == "1":
        alt = "\n".join("# " + l if l and not l.startswith("#") else l for l in m2.splitlines())
        return f"{m1}\n\n{alt}"
    alt = "\n".join("# " + l if l and not l.startswith("#") else l for l in m1.splitlines())
    return f"{m2}\n\n{alt}"


def build_launcher(args, cat: dict) -> str:
    key = args.model
    m = cat["models"][key]
    tinfo = cat["tasks"][args.task]
    provider = resolve_provider(m, args.api_provider) if m["flag_profile"] == "api" else ""
    method = args.method or str(m.get("launcher_method", 1))
    if method == "two-pass" and m["flag_profile"] != "minimax":
        die("--method two-pass is only meaningful for the MiniMax-M3 entry point")
    planner = args.planner_version or cat.get("planner_version_default", "1.0.0")
    parts = [block_header(key, m, args.task, tinfo, method, args.install_mode)]
    if not args.no_conda_env:
        parts.append(block_conda(args.conda_env or m.get("conda_env", f"eval-{key}"), CONDA_PYTHON.get(key, "3.11")))
    parts.append(block_config(args, key, m, args.task, tinfo, provider))
    parts.append(block_install(args.install_mode))
    parts.append(block_exports(args, m, args.task, tinfo, planner))
    parts.append(block_run(args, m, method))
    if not args.no_conda_env:
        parts.append("conda deactivate\n# conda remove -n $ENV_NAME --all -y")
    return "\n\n".join(parts) + "\n"


def dry_run_summary(args, cat: dict) -> dict:
    m = cat["models"][args.model]
    tinfo = cat["tasks"][args.task]
    provider = resolve_provider(m, args.api_provider) if m["flag_profile"] == "api" else None
    method = args.method or str(m.get("launcher_method", 1))
    profile = m["flag_profile"]
    if profile == "api":
        limit = args.sample_limit or m.get("sample_limit", 100)
        budget = args.max_new_tokens or m.get("default_max_tokens")
    else:
        limit = args.sample_limit or m.get("sample_limit", 1000)
        budget = args.max_new_tokens or (m.get("per_task_overrides", {}).get(args.task, {}) or {}).get("max_new_tokens") or m.get("default_max_tokens")
    out = {
        "model_key": args.model,
        "display_name": m.get("display_name"),
        "eval_module": m["eval_module"],
        "lmms_model_key": m.get("lmms_model_key"),
        "backend": m["backend"],
        "flag_profile": profile,
        "task": args.task,
        "task_tag": args.task_tag or tinfo["task_tag"],
        "tasks_json": args.tasks_json or tinfo["tasks_json"],
        "planner_version": args.planner_version or cat.get("planner_version_default"),
        "ack_release": tinfo.get("ack_release"),
        "method": method,
        "install_mode": args.install_mode,
        "conda_env": None if args.no_conda_env else (args.conda_env or m.get("conda_env")),
        "model_hf_id": None if profile in ("api", "healthgpt") else (args.model_hf_id or m.get("default_model_hf_id")),
        "model_choice": m.get("model_choice"),
        "model_name": args.model_name or m.get("default_model_name"),
        "sample_limit": limit,
        "output_token_budget": budget,
        "batch_size_per_gpu": args.batch_size_per_gpu or m.get("batch_size_per_gpu", m.get("batch_size")),
        "gpu_memory_utilization": args.gpu_memory_utilization or m.get("gpu_memory_utilization"),
        "opt_deps": m.get("opt_deps"),
        "requirements_file": m.get("requirements_file"),
        "pins": m.get("pins"),
        "api_provider": provider,
        "api_model_code": (args.model_code or m["api"]["providers"][provider]["model_code"]) if provider else None,
        "api_env_vars": m["api"]["providers"][provider]["env_vars"] if provider else [],
        "extra_args": m.get("extra_args"),
        "repository_launchers": m.get("launchers"),
        "out": args.out,
    }
    return out


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a MedVision benchmark launcher script for one model and one task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="model_catalog.json to read (default: next to this script)")
    p.add_argument("--list-models", action="store_true", help="print the catalog keys and exit")
    p.add_argument("--model", help="catalog key (see --list-models), e.g. qwen25vl, medgemma-27b, claude")
    p.add_argument("--task", choices=TASK_CHOICES, help="benchmark task: detect | TL | AD")
    p.add_argument("--dry-run", action="store_true", help="print the resolved configuration as JSON instead of the launcher")
    p.add_argument("--out", help="write the launcher to this file (default: stdout)")
    # paths
    p.add_argument("--benchmark-dir", help="MedVision checkout path (default: ${MEDVISION_BENCHMARK_DIR:-$PWD} at run time)")
    p.add_argument("--data-dir", help="dataset directory (default: ${benchmark_dir}/Data)")
    p.add_argument("--third-party-dir", help="third-party code directory for MedDr/LLaVA-Med/HuatuoGPT/HealthGPT (default: ${benchmark_dir}/third_party)")
    p.add_argument("--task-tag", help="override the Results/<task_tag> and completed_tasks tag (e.g. MedVision-TL-CoT-limit100 for a pilot)")
    p.add_argument("--tasks-json", help="override the task-list JSON path")
    # model
    p.add_argument("--model-name", help="run label (Results/<task_tag>/<model_name>, completed_tasks key)")
    p.add_argument("--model-hf-id", help="Hugging Face id or local checkpoint directory (open-weight models)")
    p.add_argument("--lora-path", help="LoRA adapter path/id for entry points that accept --lora_path")
    p.add_argument("--model-code", help="API model code override (e.g. anthropic/claude-fable-5)")
    p.add_argument("--api-provider", help="'direct' (vendor API) or 'openrouter', or an explicit provider name from the catalog")
    # resources / budgets
    p.add_argument("--sample-limit", type=int, help="samples per subtask (repository: 1000 open-weight, 100 API pilot)")
    p.add_argument("--max-new-tokens", type=int, help="output token budget (--max_new_tokens, or --max_tokens for API models)")
    p.add_argument("--batch-size-per-gpu", type=int, help="batch size per GPU (--batch_size for API/tool-use entry points)")
    p.add_argument("--gpu-memory-utilization", type=float, help="vLLM gpu_memory_utilization fraction")
    p.add_argument("--reshape-image-hw", help="resize slices before the model sees them, e.g. 512x512")
    p.add_argument("--stop-strings", help="stop string passed as --stop_strings, e.g. '</answer>'")
    p.add_argument("--cuda-visible-devices", help="emit export CUDA_VISIBLE_DEVICES=... (e.g. 0,1,2,3)")
    p.add_argument("--planner-version", help="MedVision_PLANNER_VERSION to export (default 1.0.0)")
    # skeleton options
    p.add_argument("--method", choices=["1", "2", "two-pass"], help="1 = install trio + --skip_env_setup (default for most models); 2 = built-in env setup; two-pass = MiniMax pattern")
    p.add_argument("--install-mode", choices=["wheel", "editable", "skip"], default="wheel", help="how the launcher installs medvision_bm (default: wheel build on local disk)")
    p.add_argument("--conda-env", help="conda env name to create/activate (default: catalog value)")
    p.add_argument("--no-conda-env", action="store_true", help="omit the conda create/activate block (use the already active environment)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cat = load_catalog(args.catalog)
    if args.list_models:
        list_models(cat)
        return 0
    if not args.model or not args.task:
        die("--model and --task are required (or use --list-models)")
    if args.model not in cat["models"]:
        close = [k for k in cat["models"] if args.model.lower().replace("_", "-") in k or k in args.model.lower()]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        die(f"unknown model key '{args.model}'.{hint} Use --list-models.")
    if args.sample_limit is not None and args.sample_limit <= 0:
        die("--sample-limit must be a positive integer")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        die("--max-new-tokens must be a positive integer")
    if args.gpu_memory_utilization is not None and not (0.0 < args.gpu_memory_utilization <= 1.0):
        die("--gpu-memory-utilization must be in (0, 1]")
    m = cat["models"][args.model]
    if m["flag_profile"] == "api" and (args.sample_limit or 0) > 100:
        print("[make_eval_launcher] note: API models were benchmarked as a 100-sample pilot; "
              "larger limits cost real money and OpenRouter reserves the full max_tokens per request.", file=sys.stderr)
    if m["flag_profile"] == "tooluse" and not args.model_hf_id:
        die("the tool-use entry point requires --model-hf-id (no default checkpoint)")
    if args.dry_run:
        print(json.dumps(dry_run_summary(args, cat), indent=2))
        return 0
    text = build_launcher(args, cat)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        try:
            os.chmod(out, 0o755)
        except OSError:
            pass
        print(f"[make_eval_launcher] wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
