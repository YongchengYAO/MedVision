# Source script import map — `medvision`

Every repository-maintained script that a future agent might need, and what the generated skill
did with it. "Source repo artifact" names a path in the repository; "Bundled skill helper" names
a file inside the runtime skill. The two are deliberately distinguished so no agent tries to run
a repository path that the skill does not ship.

| Source repo artifact | Decision | Bundled skill helper | Owner | Reason | Check performed |
| --- | --- | --- | --- | --- | --- |
| `script/benchmark-{detect,TL,AD}/eval__*.sh` (72 launchers) | adapt | `benchmark-evaluation/scripts/make_eval_launcher.py` + `model_catalog.json` | benchmark-evaluation | One skeleton repeated per model and task; a generator driven by a 26-model catalogue preserves the load-bearing install order and the wheel-build block without shipping 72 near-identical files | `--list-models`; generated a detect and an A/D launcher and compared against the originals |
| the node-local wheel-build block inside every launcher | adapt | `environment-setup/scripts/build_local_wheel.sh` | environment-setup | Small, safe, and the fix for a real shared-filesystem race; worth having as a standalone helper | `--help`, `bash -n`, `--no-install` build |
| per-model requirements pins | adapt | `environment-setup/scripts/check_env_pins.py` | environment-setup | Diagnosing pin drift is the most common setup failure; the repository has no such checker | `--help`; run against a real requirements file, exits 1 on mismatch |
| `script/sft/train__*.sh` (21 launchers) | adapt | `sft/scripts/sft_launcher_template.sh` | sft | One parameterised two-phase template covers the family; variants are catalogued in a reference | `bash -n`, dry-run |
| `script/rft/build_parquet_ds__verl__*.sh` (5) | adapt | `rft/scripts/build_parquet_ds.sh` | rft | Small and safe once the conda and wheel preamble is replaced by explicit arguments | `bash -n`, `--help`, dry-run |
| `script/dataset/build_visualize_parquet_ds.sh` | adapt | `dataset-and-tasks/scripts/build_parquet_ds.sh` | dataset-and-tasks | Same reason; small default limits added | `--help`, dry-run |
| `python -m medvision_bm.benchmark.download_datasets` | wrap | `dataset-and-tasks/scripts/download_datasets.sh` | dataset-and-tasks | Adds the planner-version guard, token whitespace trimming, and the CoT-suffix strip that the raw CLI lacks | `--help`, dry-run, error paths |
| `medvision_bm.utils.plan_utils` | wrap | `dataset-and-tasks/scripts/inspect_benchmark_plan.py` | dataset-and-tasks | Makes the version-ceiling behaviour observable offline | run against two real dataset directories, including a ceiling fallback |
| `tasks_list/*.json` + `utils/data_utils.tasks_to_configs` | wrap | `dataset-and-tasks/scripts/list_tasks.py` | dataset-and-tasks | Name-to-config derivation is error-prone by hand; adds plane rewriting and CoT handling | run on five real task lists plus failure paths |
| `parse_outputs` + the three summarizers | wrap | `results-parsing-and-metrics/scripts/parse_and_summarize.sh` | results-parsing-and-metrics | Keeps the two steps' flags consistent and guards T/L-only options | `--help`, three dry-runs, three error paths |
| metric functions in `utils/parse_utils.py` | adapt | `results-parsing-and-metrics/scripts/metrics_demo.py` | results-parsing-and-metrics | Executable proof of the NaN-vs-0 and denominator semantics against the installed package | run: all checks pass, exits 1 if semantics drift |
| summary JSON layouts | adapt | `results-parsing-and-metrics/scripts/inspect_summary.py` | results-parsing-and-metrics | Read-only viewer for every summary type | run against nine real files and directories |
| `script/analyze/clinical-decision-analysis/*.py` | copy | `analysis/scripts/cda/` (5 modules + 2 configs) | analysis | Self-contained apart from public package imports; copying preserves the exact method | `--help` from the bundled location |
| `script/analyze/{process,equation}-accuracy/*.py` | copy | `analysis/scripts/analyze_{process,equation}_accuracy_{TL,AD}.py` | analysis | Same reason; repository-relative paths replaced by an optional argument | `--help` from the bundled location |
| `script/analyze/detection--target-size/run_analysis.sh` + YAML | adapt | `analysis/scripts/detection_target_size.sh` + `config-detect-boxImgRatio.yaml` | analysis | Explicit paths and a dry-run replace repository-rooted defaults | `--help`, dry-run |
| `script/ablation/biomedparse/scripts/_env.sh` | adapt | `biomedparse-ablation/scripts/env_template.sh` | biomedparse-ablation | The knob surface is the useful part; activation side effects made conditional | `bash -n` |
| BiomedParse pinned dependency set | adapt | `biomedparse-ablation/scripts/check_biomedparse_env.py` | biomedparse-ablation | The setup has many pins and a source build; a checker turns a long failure into a report | run: reports missing imports and pin drift, exits 1 |
| `script/llm-parsing/judge_config.py` registry | adapt | `llm-judge-parsing/scripts/check_judge_env.py` | llm-judge-parsing | The most common judge failure is an unset interpreter variable; works without the GPU stack | `--help`, real run, four error paths |
| roster YAML authoring | adapt | `llm-judge-parsing/scripts/make_roster_yaml.py` | llm-judge-parsing | Roster keys must be directory names holding parsed records; no repository tool exists | run on a synthetic tree with nine decoy entries; output round-tripped through the pipeline's own loader |
| model registry vs image-size dispatch | adapt | `extending-models-and-tasks/scripts/list_registered_models.py` | extending-models-and-tasks | Catches the "model registered but no resize branch" class of bug statically | run against the installed package |
| model file / eval entry / launcher / unit-test templates | adapt | `extending-models-and-tasks/scripts/scaffold_new_model.py` | extending-models-and-tasks | Turns a nine-site checklist into generated skeletons with TODO markers; never writes into a checkout by default | dry-run, real run into a temp directory, generated files byte-compiled |
| task YAML layout | adapt | `extending-models-and-tasks/scripts/list_task_yamls.py` | extending-models-and-tasks | Shows which YAMLs a task list actually uses | run against the installed package |
| environment facts across all workflows | adapt | root `scripts/check_medvision_env.py` | root | One safe entry point that answers "can this machine do X" | run on two interpreters; exits 1 when the package is absent |
| `script/llm-parsing/run_llm_parsing.sh` and stage modules | reference-only | `llm-judge-parsing/references/{pipeline,recipes,cli-reference}.md` | llm-judge-parsing | 46 KB driver that re-roots to the checkout, contains a destructive step, and orchestrates a 13-hour GPU sweep; its help and step table are generated from one source, so a wrapper would drift | every documented flag verified via `--help` on CPU |
| `script/llm-parsing/setup_judge_env.sh` | reference-only | `llm-judge-parsing/references/judge-environment.md` | llm-judge-parsing | Mutates an environment and downloads gigabytes | steps documented, not executed |
| `script/ablation/biomedparse/{setup.sh, scripts/**, src/*.py}` | reference-only | `biomedparse-ablation/references/{setup,tracks,cli-reference}.md` | biomedparse-ablation | Resolve paths relative to the ablation directory and require the upstream checkout plus a compiled dependency | flags transcribed from argparse source |
| `script/visualization/**` (28 shell + 24 Python) | reference-only | root `references/visualization-catalog.md` | root | Paper- and webpage-specific, with roster YAMLs and output trees that only exist in a checkout | catalogued by entry point with inputs |
| `script/misc/{regen_all_tasks,compile_dataset_info,summarize_datasets,...}` | reference-only | `dataset-and-tasks/references/maintainer-workflows.md` | dataset-and-tasks | Need the dataset source tree, network, and hours; one has a hard source-tree guard | CLI documented; guard explained |
| `src/medvision_bm/utils/push_hf_model.py` | reference-only | `sft/references/workflows.md` | sft | Publishes to a model hub with credentials | documented, not bundled |
| `dockerfile/build_and_push.sh` | exclude | – | – | Pushes images with registry credentials | – |
| `script/llm-parsing/run_llm_parsing_{ood,fullsft}.sh` | exclude | – | – | Retired wrappers; the environment-override pattern replaced them, and re-introducing one would contradict the current design | – |
| `unit-test/**` | exclude from runtime | – | – | Verification evidence, not runtime skill content | run as native ground truth instead |

## Rule check

No runtime instruction in the skill tells an agent to execute a repository path. Where a
repository script is named, it is named as provenance and the runnable replacement is a bundled
helper or a public package command. The two reference-only pipelines that genuinely live only in
a checkout (the judge driver and the BiomedParse launchers) state that prerequisite explicitly
and use a `<repo>` placeholder rather than any concrete path.
