# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir_root="${benchmark_dir}/Data"
data_dir="${benchmark_dir}/Data/Datasets"
wd="${benchmark_dir}/script/analyze/process-accuracy"
results_dir="${benchmark_dir}/Results"

# Install medvision_bm: build the wheel on node-local disk (NOT the shared CephFS
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
tar -cf - -C "${benchmark_dir}" --exclude='*.egg-info' --exclude=__pycache__ \
    pyproject.toml MANIFEST.in LICENSE src \
  | tar -xf - -C "${build_tmp}"
python -m pip wheel "${build_tmp}" -w "${build_tmp}/wh" --no-deps
built_wheel="$(ls -t "${build_tmp}/wh"/medvision_bm-*.whl | head -n1)"
cp -f "${built_wheel}" "${wheelhouse}/"
flock "${lockfile}" python -m pip install --force-reinstall "${built_wheel}"

# Install medvision_ds
python -m medvision_bm.benchmark.install_medvision_ds --data_dir "${data_dir_root}"

# Install the vendored lmms_eval + its heavy deps (torch/transformers/nibabel) so
# the scaledPS nMAE diagonal helper (_compute_physical_diagonal) is importable;
# without it that import fails and every scaledPS nMAE comes out NaN.
python -m medvision_bm.benchmark.install_vendored_lmms_eval --lmms_eval_opt_deps medvision_v0

# AD, scaledPS in [0.5,3]
python $wd/analyze_process_accuracy_AD.py --model_dir ${results_dir}/MedVision-AD-CoT-scaledPS/MedVision-V0-7B

# TL, scaledPS in [0.5,3]
python $wd/analyze_process_accuracy_TL.py --model_dir ${results_dir}/MedVision-TL-CoT-scaledPS/MedVision-V0-7B

