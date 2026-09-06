# Coverage / depth matrix — `medvision`

One row per user-facing repository capability. "Kind" is primary workflow, support workflow,
maintainer workflow, or minor detail. "Native validation" uses the statuses from
`../verification/native-verification-report.md`.

| Capability | Kind | Evidence source | Skill location | Depth check | Synthetic validation | Native validation |
| --- | --- | --- | --- | --- | --- | --- |
| Install `medvision_bm` (PyPI / checkout / editable / nightly) | support | `pyproject.toml`, README Quick Start | `environment-setup` + `references/installation.md` | three install paths, verification command, editable-vs-copy trap | 2 cases | help-only PASS (`mvbm --help`) |
| Install `medvision_ds` dataset package | support | `utils/install_utils.py`, `cli.py` | `environment-setup` | what the installer does, env vars it sets, two-step reinstall rationale | covered | help-only PASS |
| Install vendored `lmms_eval` + per-model extras | support | `install_utils.py`, vendored `pyproject.toml` | `environment-setup` + `requirements-catalog.md` | every extra enumerated with its pins | covered | help-only PASS |
| Environment setup entry points and load-bearing order | support | `benchmark/env_setup.py`, `sft/env_setup.py`, `docs/debug_env_setup.md` | `environment-setup` | full flag lists, Method 1 vs Method 2, ordering rationale | covered | help-only PASS |
| Requirements catalogue (21 eval + 4 SFT) and Docker images | support | `requirements/`, `dockerfile/` | `environment-setup` + root `model-roster.md` | per-file pins table; docker tags and run recipe | covered | not runnable here |
| Version-pin traps (hub/transformers/datasets/torch, setuptools, protobuf, libGL) | support | source comments, `docs/debug_env_setup.md` | `environment-setup/references/troubleshooting.md` + root `troubleshooting.md` | symptom → cause → fix per trap | 1 case | `check_env_pins.py` exits 1 on mismatch |
| Dataset concepts, config naming, returned fields | primary | README Data sections, loader source | `dataset-and-tasks/references/concepts.md` | naming grammar, per-family field lists verified against the loader | covered | n/a |
| Task lists and name↔config derivation | primary | `tasks_list/`, `utils/data_utils.py` | `dataset-and-tasks` + `scripts/list_tasks.py` | CoT suffix rule, BoxCoordinate↔BoxSize bridge, SFT namespace, OOD lists | 1 case | script exercised on 5 real lists |
| Downloading datasets, download modes, tokens, QC figures | primary | README, `download_datasets.py`, loader | `dataset-and-tasks/references/downloading.md` | CLI flags, mode table, every env var, gated-source tokens | covered | help-only PASS; downloads skipped |
| Annotation versions, planner pin, acknowledgement, ceiling | primary | loader source, `utils/plan_utils.py` | `dataset-and-tasks` + `scripts/inspect_benchmark_plan.py` | exact banners, ceiling semantics, leaderboard version | 1 case | ceiling fallback demonstrated on real plans |
| `Data/` layout and benchmark plans | support | `docs/file-structure.md`, `plan_utils.py` | `dataset-and-tasks/references/data-layout.md` | tree, plan schemas, full plan_utils API | covered | inspector run on real datasets |
| Parquet snapshot + sample visualization | support | `dataset/*.py`, `script/dataset/` | `dataset-and-tasks/references/parquet-and-visualization.md` | 23 flags, 4-level limit hierarchy | covered | help-only PASS |
| Dataset-info regeneration (maintainer) | maintainer | `script/misc/`, `utils/configs_to_tasks.py` | `dataset-and-tasks/references/maintainer-workflows.md` | documented reference-only with the source-tree guard | – | help-only PASS |
| Run an evaluation (24 entry points) | primary | `benchmark/eval__*.py`, 72 launchers | `benchmark-evaluation` + `scripts/make_eval_launcher.py` | per-model wiring, full launcher anatomy, generator reproduces the skeleton | 2 cases (planned) | BLOCKED_REQUIRED_BACKEND (cuda); parsers PASS |
| Resume / response cache / completed-tasks tracker | support | `eval_utils.py`, README | `benchmark-evaluation/references/workflows.md` + `scripts/check_results_tree.py` | hash-keyed invalidation, disable switch, tracker semantics | covered | script `--help` PASS |
| Perceived image size and pixel-size invariant | primary | `medvision_utils.get_resized_img_shape`, `docs/Model-Image-Processing.md` | `benchmark-evaluation` + `extending-models-and-tasks/references/image-size-dispatch.md` | every dispatch branch, strategies A/B/C, API rules | 1 case | static registry check PASS |
| Token budgets | support | `docs/model-token-budget.md` | `benchmark-evaluation/references/image-processing-and-token-budgets.md` | three resolution channels, defaults, historical bug | covered | n/a |
| Hardware and parallelism | support | two `docs/Model-*.md` | `benchmark-evaluation` + root `model-roster.md` | per-model footprints, TP vs DP, MoE caveat | covered | n/a |
| Parse outputs (step 2) | primary | `benchmark/parse_outputs.py`, `utils/parse_utils.py` | `results-parsing-and-metrics` + `scripts/parse_and_summarize.sh` | flags, answer-scope rule, per-sample scoring, outputs | covered | help-only PASS |
| Summarize (step 3), three task families | primary | three summarizers, `utils/configs.py` | `results-parsing-and-metrics` | flags, exact output filenames, grouping, filters | 1 case | help-only PASS |
| Metric definitions and failure handling | primary | `parse_utils.py`, `medvision_utils.py` | `results-parsing-and-metrics/references/metrics.md` + `scripts/metrics_demo.py` | every metric, denominator, NaN-vs-0, units | 1 case | 4 native metric tests PASS |
| Deduplicate result records | support | `remove_duplicate_samples.py` | `results-parsing-and-metrics` | documented with the safe out-dir pattern | covered | help-only PASS |
| LLM-judge second parse (step 4) | primary | `script/llm-parsing/**`, `docs/LLM-Judge-Reproducibility.md` | `llm-judge-parsing` (6 references) | 6 stages, env knobs, artifacts, schemas, 11 recipes | 2 cases | 9 native tests PASS, 1 repo-data FAIL; inference BLOCKED |
| Judge environment and registry | support | `setup_judge_env.sh`, `judge_config.py` | `llm-judge-parsing/references/judge-environment.md` + `scripts/check_judge_env.py` | pins, registry, the PYTHON requirement | covered | checker exits 1 correctly |
| SFT dataset construction and sample limits | primary | `sft/sft_utils.py` | `sft/references/data-preparation.md` + `scripts/check_sample_limits.py` | limit hierarchy, bootstrap, validation carve-out, caches | 1 case (planned) | parser PASS |
| SFT training (LoRA and full-parameter, 9 entry points) | primary | `sft/train__*.py`, `script/sft/*.sh` | `sft` + `scripts/sft_launcher_template.sh` | two-phase pattern, every knob, FSDP configs, merging | 1 case (planned) | BLOCKED_REQUIRED_BACKEND; parsers PASS |
| Tool-use SFT variant and the execution sandbox | support | `train__qwen25vl_AD_TL_tooluse.py`, `utils/tool_execution.py` | `sft` | public CLI only; sandbox semantics | covered | 4 sandbox tests PASS |
| RFT parquet builders | primary | `rft/verl/*.py`, `script/rft/*.sh` | `rft` + `scripts/build_parquet_ds.sh` | flags, sharding, schema, family coupling | 2 cases | help-only PASS; verl prompt test PASS |
| GRPO recipe (external verl fork) | primary | README RFT section, fork docs | `rft/references/rft-recipes.md` | rewards, mixing, curriculum, recipe variables | covered | external, not run |
| Clinical Decision Agreement | primary | `script/analyze/clinical-decision-analysis/**` | `analysis/references/cda.md` + bundled `scripts/cda/` | method, config schema, kappa, volume bootstrap | 1 case (planned) | CPU-runnable; help PASS |
| Process and equation accuracy | primary | `script/analyze/{process,equation}-accuracy/` | `analysis/references/process-and-equation-accuracy.md` | step definitions, inputs, near-zero filter | covered | 5 equation tests PASS |
| Detection by target size | support | `script/analyze/detection--target-size/`, box-size analyzers | `analysis/references/detection-target-size.md` | ratio grouping, minimum group size, random baseline | covered | help-only PASS |
| Add a model (local and API) | maintainer | `docs/New-Models-Guide.md`, model templates | `extending-models-and-tasks` + `scripts/scaffold_new_model.py` | site-by-site checklist, API cap rules, scaffolder | 1 case (planned) | registry consistency check PASS |
| Add a task / dataset YAML pair | maintainer | `docs/New-Tasks-Guide.md`, task YAMLs | `extending-models-and-tasks/references/add-a-task.md` + `scripts/list_task_yamls.py` | YAML anatomy, factories, tags, registration | covered | lister run on the installed package |
| BiomedParse ablation (both tracks) | primary | `script/ablation/biomedparse/**` | `biomedparse-ablation` (6 references) | setup, both tracks, knobs, smoke tests, fairness rules | 2 cases | BLOCKED_REQUIRED_BACKEND; env checker PASS |
| Figures and webpage exports | minor | `script/visualization/**` | root `references/visualization-catalog.md` | catalogued by entry point with inputs; reference-only | – | not run |
| Paper facts and terminology | support | README, model card, paper skill | root `references/concepts-and-glossary.md` | vocabulary, units, recipe summary | 2 root cases | n/a |

## Breadth check

No public capability in the include scope is unmapped. Every primary workflow has a focused
sub-skill; every non-trivial support workflow is reachable from an owning sub-skill or a root
reference; both maintainer workflows have dedicated guidance.

## Depth check

Every sub-skill pairs a router `SKILL.md` (68-155 lines) with references carrying the flag
tables, schemas, and failure modes, plus bundled scripts. Depth was checked by asking whether a
future agent could complete the workflow without reopening the repository: the two areas where
that is only partially true are recorded in the long-tail gap register.
