Each JSON file maps a subtask name to its sample size (`{"task name": sample_size, ...}`), covering every subtask in the MedVision benchmark for this dataset version.

These `all_tasks__ds_v1.0.0/`, `all_tasks__ds_v1.1.0/`, and `all_tasks__ds_v1.1.1/` folders share the same A/D and Detection subtasks; they differ **only in the T/L (tumor/lesion size) sample sizes**, which change with each dataset release:

- **v1.0.0 → v1.1.0**: new T/L sample filtering removes ambiguous cases and adds more single-small-target samples. See the [v1.1.0 release note](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/release-v1.1.0.md).
- **v1.1.0 → v1.1.1**: corrected T/L ellipse fit (fixes a transposed in-plane voxel-spacing bug); ~22% fewer T/L samples on anisotropic slices (e.g. sagittal/coronal), while isotropic slices (e.g. axial) are essentially unchanged. See the [v1.1.1 release note](https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/doc/release-v1.1.1.md).
