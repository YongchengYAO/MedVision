# Tasks List

This folder defines which MedVision tasks are used for benchmarking and SFT training. Each JSON file maps task names to per-task sample counts. The counts are informational only — the pipeline reads just the task names (the top-level keys; see `load_tasks` in `src/medvision_bm/utils/utils.py`).

| File | Purpose |
|------|---------|
| `tasks_MedVision-{AD,TL,detect}-CoT.json` | Evaluation tasks for the Angle/Distance, Tumor/Lesion size, and Detection benchmarks |
| `tasks_MedVision-{AD,TL,detect}__train_SFT.json` | SFT training tasks |
| `OOD/` | Plane-OOD and target-OOD evaluation task lists |
| `experimental/` | Experimental and legacy task lists (not part of the main pipeline) |

## FAQ

### Q1: What are "tasks" and MedVision dataset configs?

- A **task** is a unit of model evaluation (or training data preparation). Each task name maps to a task YAML in `src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/`, which defines prompts, ground-truth formatting, and metrics.
- A **dataset config** locates the corresponding subset of the MedVision dataset on Hugging Face; each task loads its data through one config.
- Configs for all released dataset versions are listed in `dataset-info/dataset-configs/<version>/`.

### Q2: Why do evaluation task names end in "-CoT" while training task names do not?

- Both point to the same dataset configs — the `-CoT` suffix does not select different data.
- CoT (chain-of-thought) content is constructed after data loading, for both training and evaluation.
- MedVision-V0 ([our model](https://huggingface.co/YongchengYAO/MedVision-V0-7B)) also uses CoT during training. The missing `-CoT` suffix in the training task names is a legacy naming inconsistency, not a functional difference.
