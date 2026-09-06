# Repository Provenance

## Purpose

Read this before deciding whether this skill is current for a checkout of the
MedVision repository. If the current repo commit, dirty state, package version,
or major evidence paths differ from this snapshot, run `refresh-repo-skill`.

## Snapshot

```json
{
  "schema": "disco.repo-provenance.v1",
  "generated_at_utc": "2026-09-05T14:15:00Z",
  "repository": {
    "name": "MedVision",
    "remote_url": "https://github.com/YongchengYAO/MedVision",
    "vcs": "git",
    "branch": "master",
    "tag": null,
    "nearest_tag": "v1.2.0 (11 commits ahead)",
    "commit": "780e247d393bafba6ac71bb9707b0e0e4506d2e7",
    "working_tree": "dirty",
    "dirty_paths": [
      ".readthedocs.yaml",
      "README.md",
      "docs/model-release/MedVision-V0-7B_ModelCard.md",
      "docsite/source/benchmarking/clinical-decision-agreement.md",
      "docsite/source/benchmarking/llm-judge-parsing.md",
      "docsite/source/benchmarking/overview.md",
      "docsite/source/benchmarking/parsing-and-summarizing.md",
      "docsite/source/benchmarking/running-evaluations.md",
      "docsite/source/conf.py",
      "docsite/source/dataset/concepts.md",
      "docsite/source/dataset/loading.md",
      "docsite/source/dataset/statistics.md",
      "docsite/source/extending/add-a-model.md",
      "docsite/source/extending/add-a-task.md",
      "docsite/source/fine-tuning/rft.md",
      "docsite/source/fine-tuning/sft.md",
      "docsite/source/getting-started/installation.md",
      "docsite/source/getting-started/quickstart.md",
      "docsite/source/index.md",
      "docsite/source/reference/api/index.md",
      "docsite/source/reference/api/verl_utils.md",
      "docsite/source/reference/cli.md",
      "docsite/source/releases/index.md",
      "docsite/source/releases/v1.1.1.md",
      "docsite/source/releases/v1.2.0.md",
      "docsite/source/resources.md",
      "requirements/requirements_eval_llavamed.txt",
      "requirements/requirements_sft_gemma4.txt",
      "requirements/requirements_sft_medgemma.txt",
      "script/analyze/clinical-decision-analysis/README.md",
      "parse_AD-jobs.sh",
      "parse_TL-jobs.sh",
      "parse_all.sh",
      "parse_detection-jobs.sh",
      "script/rrg/",
      "setup-env-test.sh",
      "skills/"
    ]
  },
  "packages": [
    {
      "name": "medvision_bm",
      "pypi_name": "medvision-bm",
      "version": "1.2.0",
      "import_names": ["medvision_bm"],
      "console_scripts": ["mvbm"],
      "requires_python": ">=3.9",
      "pinned_dependencies": ["datasets==3.6.0", "huggingface_hub==0.36.0", "torch==2.6.0", "torchvision==0.21.0", "accelerate==1.9.0", "psutil==7.2.2", "scipy", "nibabel", "matplotlib"]
    },
    {
      "name": "medvision_ds",
      "source": "src/ subdirectory of the Hugging Face dataset repo YongchengYAO/MedVision",
      "version": "1.4.0",
      "import_names": ["medvision_ds"],
      "declared_dependencies": ["huggingface_hub>=0.35.3,<2.0", "datasets==3.6.0", "opencv-python", "nibabel", "numpy", "SimpleITK", "scipy", "pandas", "scikit-image", "pynrrd", "matplotlib", "tqdm", "requests", "synapseclient", "gdown", "gdrive", "rarfile", "py7zr"],
      "note": "installed by `mvbm install mvds -d <data_dir>`; not part of this GitHub repository. The hub floor is deliberate: the loader reinstalls this package inside a live process, so an exact pin would mutate the caller's environment mid-run."
    },
    {
      "name": "lmms_eval",
      "version": "0.3.0",
      "import_names": ["lmms_eval"],
      "note": "vendored fork shipped inside medvision_bm (src/medvision_bm/medvision_lmms_eval), installed by `python -m medvision_bm.benchmark.install_vendored_lmms_eval`"
    }
  ],
  "evidence": {
    "source_roots": ["src/medvision_bm", "src/medvision_bm/medvision_lmms_eval/lmms_eval"],
    "docs": ["README.md", "docs", "tasks_list/README.md", "script/llm-parsing/README.md", "script/llm-parsing/DESIGN.md", "script/analyze/clinical-decision-analysis/README.md", "script/ablation/biomedparse/README.md"],
    "examples": ["script/benchmark-detect", "script/benchmark-TL", "script/benchmark-AD", "script/sft", "script/rft", "script/llm-parsing", "script/analyze", "script/visualization", "script/dataset", "script/misc", "script/ablation/biomedparse"],
    "tests": ["unit-test"],
    "configs": ["tasks_list", "dataset-info", "requirements", "dockerfile", "src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks"],
    "existing_agent_skills": [".claude/skills/medvision-paper", ".claude/skills/medvision-pipeline"]
  },
  "counts_at_snapshot": {
    "eval_entry_points": 21,
    "launchers_per_task_family": 24,
    "registered_model_keys": 20,
    "task_yaml_datasets": 22,
    "base_task_yamls": 199,
    "task_yamls": 1253
  },
  "excluded_from_extraction": ["third_party", "local", "Data", "Results", "Results-bak", "SFT", "Figures", "fig", "Paper-results-backup", "build", "docsite", "dev-test", "script/dev_plot", "script/rrg", "script/ablation/biomedparse/third_party", "script/ablation/biomedparse/{data,models,results,figures}", "*/dev"]
}
```

`dirty_paths` holds exactly what `git status --porcelain` prints, so it can be
compared literally. `README.md` now carries a table of contents, a new "🤖 Agent Skills" section, the install-example swap to `mvbm install mvds -d Data`, an ablation-bullet reword, a download-section heading rename and a rewritten QC-figure bullet — no longer a one-line
change swapping the dataset-install example to `mvbm install mvds -d Data`. Both
that form and
`python -m medvision_bm.benchmark.install_medvision_ds --data_dir <data_dir>`
exist and are documented here as aliases. Everything else in the list is
untracked — private parse launchers (`parse_*.sh`, `setup-env-test.sh`), an
unrelated report-generation directory (`script/rrg/`) and this skill's own tree
(`skills/`); none was used as evidence.

`src/medvision_bm/sft/config` does not exist at this commit and is not package
data. Training configuration is passed as CLI flags and accelerate/FSDP config
files; see the `sft` sub-skill.

The fine-tuning surface (`src/medvision_bm/sft/` and `script/sft/`) is unchanged
since the previous snapshot: it carries the two-stage dataset preparation,
true-size prepared-dataset naming, the stage-1 gate on `--prepared_ds_dir` +
`--skip_process_dataset`, and the launcher hand-off of the reported directory to
the training launch.

## Refresh Check

- If `git rev-parse HEAD` differs from `repository.commit`, treat the skill as
  potentially stale and run `refresh-repo-skill`.
- If the current working tree is dirty and the dirty paths differ from the list
  above, run `refresh-repo-skill`.
- If `medvision_bm.__version__` is no longer `1.2.0`, or `pyproject.toml`
  dependencies / console scripts / package data changed, run
  `refresh-repo-skill`.
- If the dataset package `medvision_ds` moved past `1.4.0` (new annotation
  release), or its declared `huggingface_hub` / `datasets` ranges changed,
  re-check `../sub-skills/dataset-and-tasks/` for planner-version and
  `MedVision_ACK_RELEASE` guidance and the install-order guidance in
  `../sub-skills/environment-setup/` and `../sub-skills/biomedparse-ablation/`.
- If files were added under `src/medvision_bm/benchmark/eval__*.py`,
  `script/benchmark-*/`, `requirements/`, or `dockerfile/`, re-check
  `model-roster.md` and `../sub-skills/benchmark-evaluation/`.
- If `src/medvision_bm/utils/install_utils.py` changes how `install_medvision_ds`
  installs the wheel, re-check the two-step install description in
  `../sub-skills/environment-setup/references/installation.md` and the pin-lift rows
  in `../sub-skills/environment-setup/references/troubleshooting.md` and
  `../sub-skills/biomedparse-ablation/`.
- If `src/medvision_bm/sft/sft_utils.py` or the `train__*.py` entry points
  change how the prepared dataset is named or handed to training, or the
  `script/sft/` launchers change their phase A / phase B shape, re-check
  `../sub-skills/sft/` (`references/data-preparation.md`, `workflows.md`,
  `launcher-catalog.md`, `troubleshooting.md`, `scripts/sft_launcher_template.sh`).

Two bundled scripts re-verify most of `counts_at_snapshot` cheaply and exit
non-zero on drift: `../sub-skills/extending-models-and-tasks/scripts/list_task_yamls.py`
(datasets, base/task YAML counts, duplicate task names, broken includes) and
`../sub-skills/extending-models-and-tasks/scripts/list_registered_models.py`
(registered keys against modules, decorators and dispatch branches).
