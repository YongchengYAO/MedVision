#!/usr/bin/env bash
#
# Create the environment the judge runs in, and prove it works.
#
# The pipeline needs an interpreter with vLLM and a torch whose CUDA build the
# pod's driver actually supports. That is a separate environment from the one used
# for evaluation, which pins a vLLM too old for the judge.
#
# It is also a separate environment PER READER, which is not a convenience:
# readers can impose contradictory pins. Gemma-4's config declares transformers
# 5.5.0.dev0, which no 4.x release can load, while a vLLM release declaring
# transformers<5 cannot host it -- there is no version satisfying both. Each
# reader therefore gets its own venv, its own requirements file and its own torch
# pin, all read from the registry in judge_config.py.
#
# Usage:
#   bash script/llm-parsing/setup_judge_env.sh                    # <repo>/.cache/judge-env_gemma-4-31b
#   bash script/llm-parsing/setup_judge_env.sh --judge gemma-4-31b
#   bash script/llm-parsing/setup_judge_env.sh /srv/envs/judge    # explicit target
#
# Then, as it will print:
#   export PYTHON=<target>/bin/python
#
# Environment:
#   JUDGE            reader key, same as --judge. `judge_config.py --list` shows them.
#   TORCH_INDEX_URL  wheel index for a torch build matching an OLDER driver, e.g.
#                    https://download.pytorch.org/whl/cu126 for a CUDA 12.6
#                    driver. Installed BEFORE vllm, which then accepts it because
#                    any CUDA variant of the pinned torch satisfies its own pin.
#                    Leave unset for the default (CUDA 12.8) build.
#   PYTHON_BIN       base interpreter used to create the venv (default: python3).
#                    Must be 3.10-3.12.

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE=script/llm-parsing

say()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

JUDGE="${JUDGE:-}"
TARGET=""
# The positional argument is a DIRECTORY THIS SCRIPT CREATES, so the parser has to
# be strict about what reaches it. Two guards, both from observed damage:
#
#   -*  A mistyped flag would otherwise fall through to the catch-all and become a
#       venv path. `--jugde gemma-4-31b` built an env at ./gemma-4-31b -- for the
#       DEFAULT reader, because the typo never set JUDGE -- then installed,
#       verified and reported success. Nothing about that is recoverable by
#       reading the output.
#
#   2nd A second positional used to overwrite the first silently, which is the
#       same failure with a different shape.
#
# A bare word that is not a path is still accepted, because "$(pwd)/pilot" and
# "pilot" are both legitimate targets. That is how a stray 21 MB `pilot/` venv
# appeared in the repo root on 2026-08-15: `setup_judge_env.sh pilot` was meant to
# be `run_llm_parsing.sh pilot` (pilot is a STEP of the driver, not a reader).
# Naming the sibling script in the error is the part that catches it.
while [ $# -gt 0 ]; do
  case "$1" in
    --judge) [ $# -ge 2 ] || die "--judge needs a value"; JUDGE="$2"; shift 2 ;;
    --list) "${PYTHON_BIN:-python3}" "${HERE}/judge_config.py" --list; exit 0 ;;
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
    -*) die "unknown option '$1'.
       Known: --judge KEY | --list | --help
       A bare word is taken as the venv DIRECTORY to create, so an unrecognised
       flag is refused rather than turned into one." ;;
    *)
      # Driver-step check FIRST: it produces the actionable message, and it must
      # win even when a target was already given (`setup_judge_env.sh /srv/env pilot`).
      case "$1" in
        prep|stage0|smoke|pilot|full|analyze)
          die "'$1' is a step of run_llm_parsing.sh, not a target for this script.
       This script only BUILDS the environment; it would create a venv at ./$1.
       You probably want:  bash ${HERE}/run_llm_parsing.sh $1" ;;
      esac
      [ -z "${TARGET}" ] || die "two target directories given ('${TARGET}' and '$1').
       This script takes at most one: the venv path to create."
      TARGET="$1"; shift ;;
  esac
done

# The reader's requirements file, torch pin, expected transformers major and venv
# name all come from judge_config.JUDGE_MODELS rather than being repeated here.
# A copy of those pins in this script is a copy that goes stale silently: it would
# still build an environment, just not the one the reader needs.
if ! _reg="$("${PYTHON_BIN:-python3}" "${HERE}/judge_config.py" --shell ${JUDGE:+--judge "${JUDGE}"} 2>&1)"; then
  printf '%s\n' "${_reg}" >&2; exit 1
fi
eval "${_reg}"; unset _reg

# Default target carries the reader suffix, so building the second reader's
# environment cannot overwrite the first reader's working one.
TARGET="${TARGET:-$(pwd)/.cache/${JUDGE_ENV_BASENAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

say "judge environment for ${JUDGE_KEY} -> ${TARGET}"
note "model        : ${JUDGE_MODEL_HF}"
note "requirements : ${HERE}/${JUDGE_REQUIREMENTS}"
if [ -n "${JUDGE_POST_REQUIREMENTS}" ]; then
  note "overrides    : ${HERE}/${JUDGE_POST_REQUIREMENTS} (separate pip pass)"
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || die "PYTHON_BIN=${PYTHON_BIN} not found. Set it to a python 3.10-3.12 binary."

# vLLM publishes wheels for 3.10-3.12. On 3.13 pip resolves nothing and reports
# it as "no matching distribution for vllm", which reads like a network problem.
"${PYTHON_BIN}" - <<'PY' || die "need python 3.10-3.12 (set PYTHON_BIN)"
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)
PY
note "base interpreter: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

if [ -x "${TARGET}/bin/python" ]; then
  note "reusing the existing venv"
else
  "${PYTHON_BIN}" -m venv "${TARGET}" || die "could not create a venv at ${TARGET}"
  note "created"
fi
VPY="${TARGET}/bin/python"

say "installing (a few GB; vLLM pulls torch and the CUDA runtime)"
"${VPY}" -m pip install --upgrade pip >/dev/null

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  note "torch first, from ${TORCH_INDEX_URL} (${JUDGE_TORCH_PIN})"
  "${VPY}" -m pip install "${JUDGE_TORCH_PIN}" --index-url "${TORCH_INDEX_URL}" \
    || die "could not install ${JUDGE_TORCH_PIN} from ${TORCH_INDEX_URL}"
fi

[ -f "${HERE}/${JUDGE_REQUIREMENTS}" ] \
  || die "${HERE}/${JUDGE_REQUIREMENTS} is missing (named by JUDGE_MODELS['${JUDGE_KEY}'])"
"${VPY}" -m pip install -r "${HERE}/${JUDGE_REQUIREMENTS}" || die "pip install failed"

# The CPU stages run under this same interpreter, and they import medvision_bm.
# Nothing in Stage 1 needs these, so an environment without them builds cleanly,
# verifies, sweeps for thirteen GPU-hours and only then dies on the first line of
# Stage 2. Reader-independent, hence one shared file rather than a copy per reader.
"${VPY}" -m pip install -r "${HERE}/requirements-cpu-stages.txt" \
  || die "could not install the CPU-stage dependencies (requirements-cpu-stages.txt)"

# Phase 2: pins that CONTRADICT what phase 1 installed, and therefore cannot be
# expressed in the same requirements file. pip resolves one file as a single
# constraint set, so `vllm==0.19.0` beside `transformers==5.10.2` is not untidy but
# unsatisfiable -- ResolutionImpossible, nothing installed. A separate invocation
# resolves against the INSTALLED set instead: pip prints a red "dependency resolver
# does not currently take into account all the packages that are installed" notice,
# exits 0, and the override sticks. eval__gemma4.py does the same thing for the
# same model, which is where this shape comes from.
if [ -n "${JUDGE_POST_REQUIREMENTS}" ]; then
  [ -f "${HERE}/${JUDGE_POST_REQUIREMENTS}" ] \
    || die "${HERE}/${JUDGE_POST_REQUIREMENTS} is missing (named by JUDGE_MODELS['${JUDGE_KEY}'])"
  say "applying overrides (${JUDGE_POST_REQUIREMENTS})"
  note "pip will print a block of red 'incompatible' lines here. EXPECTED."
  note "Overriding transformers makes EVERY installed package that declares"
  note "transformers<5 complain, not only the one being overridden -- on this"
  note "stack that is vllm, xgrammar and compressed-tensors. Those are DECLARED"
  note "bounds, not observed failures: the validated eval stack ships the same"
  note "combination and runs. The verification below imports each complainer and"
  note "is the real gate."
  "${VPY}" -m pip install -r "${HERE}/${JUDGE_POST_REQUIREMENTS}" \
    || die "override install failed (${JUDGE_POST_REQUIREMENTS})"
fi

say "verifying"
"${VPY}" - "${JUDGE_KEY}" "${JUDGE_TRANSFORMERS_MAJOR}" \
    "${JUDGE_POST_REQUIREMENTS:-${JUDGE_REQUIREMENTS}}" \
    <<'PY' || die "the environment imports but cannot use the GPU -- see above"
import sys
import torch, transformers, vllm

judge, want_major, reqs = sys.argv[1], int(sys.argv[2]), sys.argv[3]

print(f"  judge        : {judge}")
print(f"  python       : {sys.version.split()[0]}")
print(f"  vllm         : {vllm.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  torch        : {torch.__version__} (built for CUDA {torch.version.cuda})")

# The transformers major is checked against what THIS reader needs, in both
# directions. Neither bound is advisory: Gemma-4's config declares
# transformers_version 5.5.0.dev0, which no 4.x release can read, while a vLLM
# that declares transformers<5 will happily resolve the 4.x line and only fail
# later. Checking here turns that into a setup error instead of an unhelpful load
# error long after a 62 GB download.
got_major = int(transformers.__version__.split(".")[0])
if got_major != want_major:
    raise SystemExit(
        f"\n  ABORT transformers {transformers.__version__} is installed, but"
        f" {judge} needs the\n  {want_major}.x line. Reinstall with the pinned"
        f" {reqs}.")

# Override collateral. Forcing transformers past an upper bound makes EVERY
# installed package carrying that bound report a conflict, not just the one the
# override targeted -- here vllm, xgrammar and compressed-tensors. pip prints those
# in red and exits 0, which leaves a human to decide whether four scary lines are
# fine. They usually are: a declared bound is a claim by a package author, not an
# observation, and the validated eval stack
# (requirements/requirements_eval_gemma4.txt) ships this exact combination. What
# distinguishes noise from breakage is whether the complainers still IMPORT, so
# check that here instead of asking anyone to eyeball it. A version bound that is
# merely conservative passes; one that reflects a real API break fails, now,
# rather than at the first chat() call after a 62 GB download.
import importlib
import importlib.metadata as md

suspects = []
for dist in md.distributions():
    dname = (dist.metadata or {}).get("Name")
    if not dname:
        continue
    for req in (dist.requires or []):
        head = req.split(";")[0].strip()
        if head.split("[")[0].split()[0].split("<")[0].split(">")[0].split("=")[0].lower() != "transformers":
            continue
        if "<5" in head:
            suspects.append(dname)
            break

if suspects:
    # dist name != module name (compressed-tensors -> compressed_tensors), so ask
    # the metadata rather than guessing, and fall back to the usual normalisation.
    top = {}
    try:
        for mod, dists in md.packages_distributions().items():
            for d in dists:
                top.setdefault(d, mod)
    except Exception:
        pass
    print(f"\n  packages declaring transformers<5 (expected to complain): "
          f"{', '.join(sorted(set(suspects)))}")
    broken = []
    for d in sorted(set(suspects)):
        mod = top.get(d, d.replace("-", "_"))
        try:
            importlib.import_module(mod)
        except Exception as e:
            broken.append(f"{d} (import {mod}): {type(e).__name__}: {e}")
    if broken:
        raise SystemExit(
            "\n  ABORT a package that declares transformers<5 no longer imports"
            " under\n  the installed transformers, so the bound is real rather than"
            " conservative:\n    "
            + "\n    ".join(broken)
            + "\n\n  This is NOT the ordinary pip conflict message -- that one is"
              " harmless.\n  Pin the offender in the override file, or pin"
              f" transformers lower there:\n      {reqs}\n")
    print("  ...and all of them import cleanly, so those bounds are conservative.")

n = torch.cuda.device_count()
if not n:
    print("\n  NOTE no CUDA device visible. Fine on a CPU box -- the CPU steps"
          " (model, prep,\n       stage0, analyze) will work. Stage 1 needs a GPU.")
    raise SystemExit(0)

# The allocation, not the count, is the test: device_count() only enumerates the
# driver. A torch built for a newer CUDA than the driver provides passes the count
# and fails here -- which is the "driver is too old" error, reported against the
# interpreter that causes it rather than 13 hours into a sweep.
try:
    torch.zeros(1, device="cuda")
except RuntimeError as e:
    raise SystemExit(
        f"\n  ABORT CUDA will not initialize.\n    {e}\n\n"
        f"  'driver is too old' means this torch build wants a newer CUDA than the"
        f" driver\n  provides. Rebuild the env against the driver's version, e.g."
        f" for CUDA 12.6:\n"
        f"      TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \\\n"
        f"        bash script/llm-parsing/setup_judge_env.sh <target>\n")

print(f"  CUDA         : OK, {n} device(s), {torch.cuda.get_device_name(0)}")
PY

say "done"
cat <<EOF

  Use it:

      export PYTHON=${TARGET}/bin/python
      bash ${HERE}/run_llm_parsing.sh --judge ${JUDGE_KEY}

EOF
