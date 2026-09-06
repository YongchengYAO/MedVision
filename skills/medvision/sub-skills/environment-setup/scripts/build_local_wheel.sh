#!/usr/bin/env bash
# build_local_wheel.sh -- build the medvision_bm wheel on node-local disk, then
# (optionally) install it into the current Python environment under a file lock.
#
# Purpose
#   Adapted from the wheel-build block that every MedVision benchmark/SFT/RFT
#   launcher runs before `python -m medvision_bm...`. setuptools' build_py caches
#   the directories it created in a process-global memo; on shared network file
#   systems (CephFS and similar) a build subdirectory can transiently vanish, after
#   which the cache refuses to recreate it and the copy step dies with
#     could not create '<build>/lib/medvision_bm/...': No such file or directory
#   Building from a private copy of the source tree in a mktemp directory avoids
#   that race entirely; only the shared-environment `pip install` needs the lock.
#
# Prerequisites
#   - bash, tar, flock (util-linux), mktemp
#   - a Python with pip (default: `python` on PATH; override with --python)
#   - network access OR an already-installed setuptools>=61 + wheel when you pass
#     --no-build-isolation (pip otherwise downloads the build backend)
#
# Usage
#   build_local_wheel.sh --repo <path-to-MedVision-checkout> [--wheelhouse <dir>]
#                        [--python <interpreter>] [--lock <file>]
#                        [--no-install | --with-deps] [--no-build-isolation]
#
#   --repo         checkout that contains pyproject.toml, MANIFEST.in, LICENSE, src/  (required)
#   --wheelhouse   where the built wheel is copied           (default: <repo>/.wheelhouse)
#   --python       interpreter whose pip builds/installs     (default: python)
#   --lock         flock file guarding the install step      (default: <repo>/.medvision_build.lock)
#   --no-install   build + copy the wheel only; print its path and exit 0
#   --with-deps    install WITHOUT --no-deps (exactly what the repository launchers do).
#                  WARNING: this re-resolves medvision_bm's own pins (torch==2.6.0,
#                  torchvision==0.21.0, huggingface_hub==0.36.0, ...) and can downgrade a
#                  working vLLM/torch stack. The default install uses --no-deps.
#   --no-build-isolation   pass through to `pip wheel` (offline hosts)
#
# Examples
#   # refresh medvision_bm code in the active env without touching torch/hub pins
#   bash build_local_wheel.sh --repo "$HOME/MedVision"
#   # only build, keep the wheel in a scratch wheelhouse
#   bash build_local_wheel.sh --repo "$HOME/MedVision" --wheelhouse /tmp/wh --no-install
#
# Output: the last line on stdout is the absolute path of the wheel inside the wheelhouse.
set -euo pipefail

# Print the header comment block (everything after the shebang up to the first non-comment line).
usage() { awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

repo=""
wheelhouse=""
python_bin="python"
lockfile=""
do_install=1
no_deps="--no-deps"
build_isolation=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)        repo="${2:?--repo needs a value}"; shift 2 ;;
        --wheelhouse)  wheelhouse="${2:?--wheelhouse needs a value}"; shift 2 ;;
        --python)      python_bin="${2:?--python needs a value}"; shift 2 ;;
        --lock)        lockfile="${2:?--lock needs a value}"; shift 2 ;;
        --no-install)  do_install=0; shift ;;
        --with-deps)   no_deps=""; shift ;;
        --no-build-isolation) build_isolation="--no-build-isolation"; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "error: unknown argument '$1' (see --help)" >&2; exit 2 ;;
    esac
done

if [[ -z "${repo}" ]]; then
    echo "error: --repo <path> is required" >&2; exit 2
fi
repo="$(cd "${repo}" 2>/dev/null && pwd)" || { echo "error: --repo '${repo}' is not a directory" >&2; exit 2; }
for f in pyproject.toml MANIFEST.in LICENSE src; do
    if [[ ! -e "${repo}/${f}" ]]; then
        echo "error: '${repo}/${f}' not found -- --repo must point at a MedVision checkout" >&2; exit 2
    fi
done
if ! grep -q 'name = "medvision_bm"' "${repo}/pyproject.toml"; then
    echo "error: ${repo}/pyproject.toml does not declare project name medvision_bm" >&2; exit 2
fi
command -v "${python_bin}" >/dev/null 2>&1 || { echo "error: python interpreter '${python_bin}' not found" >&2; exit 2; }
if (( do_install )); then
    command -v flock >/dev/null 2>&1 || { echo "error: 'flock' (util-linux) is required for the install step; use --no-install to skip it" >&2; exit 2; }
fi

wheelhouse="${wheelhouse:-${repo}/.wheelhouse}"
lockfile="${lockfile:-${repo}/.medvision_build.lock}"
mkdir -p "${wheelhouse}"
wheelhouse="$(cd "${wheelhouse}" && pwd)"

# Private build copy on local disk: only the files setuptools needs, no stale egg-info/pycache.
build_tmp="$(mktemp -d "${TMPDIR:-/tmp}/medvision_build.XXXXXX")"
trap 'rm -rf "${build_tmp}"' EXIT
tar -cf - -C "${repo}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"

echo "[build_local_wheel] building medvision_bm wheel from ${repo} in ${build_tmp}" >&2
# shellcheck disable=SC2086
"${python_bin}" -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps ${build_isolation} >&2
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
final_wheel="${wheelhouse}/$(basename "${built_wheel}")"

if (( do_install )); then
    echo "[build_local_wheel] installing ${final_wheel} (flock ${lockfile}) ${no_deps}" >&2
    # shellcheck disable=SC2086
    flock "${lockfile}" "${python_bin}" -m pip install --force-reinstall ${no_deps} "${final_wheel}" >&2
    "${python_bin}" - <<'PY' >&2
import medvision_bm
print(f"[build_local_wheel] medvision_bm {medvision_bm.__version__} now imports from {medvision_bm.__file__}")
PY
else
    echo "[build_local_wheel] --no-install: wheel built and copied only" >&2
fi
echo "${final_wheel}"
