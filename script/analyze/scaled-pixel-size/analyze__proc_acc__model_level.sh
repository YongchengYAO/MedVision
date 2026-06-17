# Set paths and configs
benchmark_dir="/root/Documents/MedVision"
data_dir_root="${benchmark_dir}/Data"
data_dir="${benchmark_dir}/Data/Datasets"
wd="${benchmark_dir}/script/analyze/process-accuracy"
results_dir="${benchmark_dir}/Results"

# Install medvision_bm (locked shared build)
set -euo pipefail
lockfile="${benchmark_dir}/.medvision_build.lock"
wheelhouse="${benchmark_dir}/.wheelhouse"
mkdir -p "${wheelhouse}"
flock "${lockfile}" bash -c '
    set -euo pipefail
    benchmark_dir="'"${benchmark_dir}"'"
    wheelhouse="'"${wheelhouse}"'"
    rm -rf "${benchmark_dir}/build" "${benchmark_dir}/src/medvision_bm.egg-info"
    python -m pip wheel "${benchmark_dir}" -w "${wheelhouse}" --no-deps
    latest_wheel="$(ls -t "${wheelhouse}"/medvision_bm-*.whl | head -n1)"
    python -m pip install --force-reinstall "${latest_wheel}"
'

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

