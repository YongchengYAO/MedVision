# MedVision — Repository Evidence Map (create-repo-skill stage 2)

Generated 2026-09-04 from checkout `a2c6482e0dbeea7f5cd5a8eddac7c7581f30608c` (branch `master`,
`git describe` = `v1.2.0-8-ga2c6482`, working tree DIRTY: 9 modified + 6 untracked paths).
Decision policy: `extractionScope: ask`, `importAfterVerification: ask` (user gave no delegation).

Canonical skill id: `medvision` (no collision in the managed library; repo id `YongchengYAO/MedVision`).
Active skills root: `<repo>/skills/` (did not exist) -> generated skill at `skills/medvision/`.
Artifact root: `skills/tests/medvision/` (`test-cases/`, `reports/`).
Private inspection prefix (deferred until scope confirmed):
`/mnt/vincent-pvc-rwm/.disco/agent/envs/medvision-inspection` (DISCO_CODING_AGENT_DIR is on the PVC; `$HOME` is ephemeral).

## Package metadata

| Field | Value |
| --- | --- |
| Distribution / import | `medvision_bm` / `medvision_bm` (src layout, `package-dir = {"" = "src"}`) |
| Version | dynamic (`medvision_bm.__version__`) |
| Python | `>=3.9` |
| Pinned deps | datasets==3.6.0, huggingface_hub==0.36.0, torch==2.6.0, torchvision==0.21.0, accelerate==1.9.0, psutil==7.2.2, scipy, nibabel, matplotlib |
| Console script | `mvbm = medvision_bm.cli:main` (`mvbm install mvds -d <data_dir>`) |
| Package data | vendored `medvision_lmms_eval/**`, `sft/config/*.yaml` |
| Sibling package | `medvision_ds` (dataset codebase, installed by `install_medvision_ds` into `<data_dir>/src`; local source tree v1.4.0 at `/mnt/vincent-pvc-rwm/MedVision/src`) |
| License file | CC-BY 4.0 (GitHub API: `NOASSERTION`; per-commit license endpoint 404 -> resolver value `NO_LICENSE`) |

## Include (extraction evidence)

| Evidence source | Why it matters | Planned use |
| --- | --- | --- |
| `src/medvision_bm/{cli.py,utils/}` | CLI entry, task JSON loader, parse/metric utils, configs (SEED, thresholds), plan utils, install utils, tool execution | verify APIs/signatures; cross-cutting references and scripts |
| `src/medvision_bm/benchmark/` | eval__<model>.py (24), env_setup, install_medvision_ds, install_vendored_lmms_eval, download_datasets, parse_outputs, summarize_{AD,TL,detection}_task, eval_utils, remove_duplicate_samples, analyze/viz helpers | benchmark-evaluation + results sub-skills, CLI reference |
| `src/medvision_bm/medvision_lmms_eval/lmms_eval/{models,tasks,evaluator.py}` | vendored eval engine: per-VLM wrappers (vLLM + API), per-dataset task YAMLs + utils, `medvision/medvision_utils.py` shared prompt/metric code | benchmark internals, new-model/new-task maintainer guidance, metric definitions |
| `src/medvision_bm/dataset/` | build_parquet_ds, ds_utils, visualize_samples | dataset access / visualization support |
| `src/medvision_bm/sft/` | train__{SFT,fullFT}-CoT__{qwen2_5_vl,gemma4,medgemma,qwen3vl}.py, sft_utils (dataset construction), per-model utils, prompts, env_setup, config yamls | sft sub-skill |
| `src/medvision_bm/rft/verl/` | build_parquet_ds(+checkpointed), reward code mirrored from verl fork | rft sub-skill |
| `README.md` | Quick Start, Docker, Data, Benchmark steps 1-4, SFT, RFT, dataset concepts, download modes, analysis/viz catalogue | workflow references everywhere |
| `docs/` (selected) | New-Tasks-Guide, New-Models-Guide, Model-Image-Processing, Model-Hardware-Requirements, Model-Parallelism-Summary, model-token-budget, debug_env_setup, file-structure, SFT_model_checkpoints, LLM-Judge-Reproducibility, blog/{Dataset-Tutorial,SFT_Tutorial}, codebase/dataset/model release notes | references + troubleshooting |
| `script/benchmark-{detect,TL,AD}/` (72 launchers) | canonical per-model eval recipes (env setup order, flags, token budgets) | distil into command builder + launcher template |
| `script/sft/` | LoRA-CoT and full-FT launchers | sft workflows reference |
| `script/rft/` | parquet builder launchers | rft workflows reference |
| `script/llm-parsing/` | LLM-judge re-parsing pipeline (README, DESIGN, run_llm_parsing.sh, judge_*.py, configs, setup_judge_env.sh) | llm-judge sub-skill |
| `script/analyze/{clinical-decision-analysis,process-accuracy,equation-accuracy,detection--target-size}` | documented analyses (README lists them) | analysis sub-skill (pending user choice) |
| `script/visualization/*.sh` (28) + `*.py` (24) | documented figure entry points | reference-only catalogue (pending user choice) |
| `script/misc/` | compile_dataset_info, regen_all_tasks, summarize_datasets, convert_configs_to_tasks | maintainer/dataset-info support |
| `script/dataset/build_visualize_parquet_ds.sh` | dataset parquet + visualization | dataset sub-skill |
| `tasks_list/` (+README, OOD/, experimental/) | task-name namespace, SFT vs eval naming | dataset/tasks reference |
| `dataset-info/` | per-version dataset configs, datasets_info.json | dataset reference (metadata only, no 676MB JSONs) |
| `requirements/`, `dockerfile/` | per-model install variants, docker tags | environment sub-skill |
| `unit-test/` | CPU-runnable metric/parsing tests; model-download tests | native candidates |
| `.claude/skills/{medvision-paper,medvision-pipeline}/SKILL.md` | existing repo-local agent guidance (paper facts, 3-step pipeline) | reuse validated guidance, keep terminology |
| `completed_tasks/` | runtime task-status tracker (format only) | results reference |
| `pyproject.toml`, `MANIFEST.in`, `.github/workflows/publish-pypi.yml`, `LICENSE` | packaging/release facts | provenance, install guidance |

## Exclude (kept out of extraction context)

| Path | Reason |
| --- | --- |
| `script/ablation/` (392k files) | BiomedParse clone + outputs; separate ablation with its own README; at most a one-line pointer |
| `third_party/` | vendored model repos (InternVL, LLaVA-Med, HealthGPT, ...) |
| `local/` | private experiment launchers / pilot runs (never to be launched) |
| `Data/`, `Results/`, `Results-bak/`, `SFT/`, `Paper-results-backup/`, `Figures/`, `fig/` | data, results, checkpoints, figures |
| `build/`, `src/build/`, `src/medvision_bm.egg-info`, `.cache/`, `.wheelhouse/`, `.cuda-link`, `.mypy_cache`, `.pytest_cache`, `graphify-out/`, `.remember/` | build/cache/generated |
| `docsite/` | Sphinx site source/build (public docs are on readthedocs; README already carries the content) |
| `dev-test/`, `script/dev_plot/`, `script/analyze/dev-scalefactor/`, `*/dev/` subdirs, `docs/dev*`, `docs/literature-review__*`, `docs/superpowers/` | development scratch |
| `script/rrg/`, untracked `parse_*.sh`, `setup-env-test.sh` | untracked WIP |
| `medvision_lmms_eval/{miscs,tools,docs,wheels,*.egg-info}` | upstream lmms-eval leftovers |
| `.claude/` (other than skills as evidence), `CLAUDE.md` | agent config |

## Native test/example candidate map (initial; not run in this phase)

| Native artifact | Workflow | Safety | Backend | Criticality | CPU sub | Dependency variant | Hardware | Verification expectation | Candidate command | Expected signal | Planned owner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mvbm --help`, `mvbm install mvds --help` | CLI | help-only | any | alternative | full | base | none | native help | `mvbm --help` | exit 0, subcommands listed | root |
| `python -m medvision_bm.benchmark.{parse_outputs,summarize_AD_task,summarize_TL_task,summarize_detection_task,download_datasets,env_setup,install_medvision_ds,install_vendored_lmms_eval} --help` | pipeline CLIs | help-only | any | alternative | full | base | none | native help | as listed | exit 0, flags match reference | results / environment / dataset |
| `unit-test/nMAE/test-{1..5}.py` | metric definitions | safe-runnable | cpu | alternative | full | base | none | native test | `python unit-test/nMAE/test-1.py` | exit 0 | results |
| `unit-test/scaledPS/test-{1..5}.py` | scaled pixel-size pipeline | safe-runnable | cpu | alternative | full | base | none | native test | as listed | exit 0 | results |
| `unit-test/detection-metric-failure/test-{1..3}.py` | detection metric failure handling | safe-runnable | cpu | alternative | full | base | none | native test | as listed | exit 0 | results |
| `unit-test/equation-accuracy/test-{1..5}.py` | equation accuracy analysis | safe-runnable | cpu | alternative | full | base | none | native test | as listed | exit 0 | analysis |
| `unit-test/llm-parsing/test-{1..6,11}.py` | judge queue/parse invariants | safe-runnable (check per file) | cpu | alternative | full | base + script/llm-parsing on path | none | native test | as listed | exit 0 | llm-judge |
| `unit-test/tool-use/test-{1..4}.py` | tool execution sandbox | safe-runnable | cpu | alternative | full | base | none | native test | as listed | exit 0 | sft (tool-use) |
| `unit-test/detection-verl-nocot/test-1.py` | verl prompt building | check imports | cpu | optional | full | verl fork | none | maybe | tbd | tbd | rft |
| `unit-test/medvision-ds-planner-version/*` | planner version switch | skip-network | cpu | optional | full | medvision_ds + HF download | none | doc only | none | n/a | dataset |
| `unit-test/sft-loss-masking/*` | Gemma/Qwen collate masking | skip-network (processor download) | cpu/cuda | optional | partial | sft requirements | none/GPU | doc only | none | n/a | sft |
| `unit-test/perceived-size-resize/*` | perceived-size probes | skip-network | cpu | optional | partial | transformers processors | none | doc only | none | n/a | benchmark-evaluation |
| `unit-test/{claude,gemini,kimi,openai}-image-resize/*` | API image caps | skip-credentials | any | optional | n/a | API SDKs | keys | doc only | none | n/a | benchmark-evaluation |
| `script/llm-parsing/run_llm_parsing.sh --list/--help` | judge driver | help-only | any | alternative | full | base | none | native help | `bash script/llm-parsing/run_llm_parsing.sh --list` | steps printed | llm-judge |
| `script/llm-parsing/test-sweep.sh`, `smoke`/`pilot`/`full` steps | judge inference | skip-gpu-or-hardware | cuda | REQUIRED | none | vllm 0.11 judge env | >=1 H100-class GPU | BLOCKED while no GPU | none | n/a | llm-judge |
| `script/benchmark-*/eval__<open-weight>__*.sh` | local VLM evaluation | skip-gpu-or-hardware + skip-network | cuda | REQUIRED | none | per-model requirements_eval_*.txt (vLLM) | 1-4x 80GB GPUs | BLOCKED while no GPU | none | n/a | benchmark-evaluation |
| `script/benchmark-*/eval__{Claude,GPT,Gemini,Kimi,GLM}*.sh` | API evaluation | skip-credentials | any | optional | full (given keys) | API SDK requirements | keys | doc only | none | n/a | benchmark-evaluation |
| `script/sft/train__*.sh`, `src/medvision_bm/sft/train__*.py` | SFT | skip-gpu-or-hardware + skip-expensive | cuda | REQUIRED | none | requirements_sft_*.txt | 4x 80-140GB | BLOCKED while no GPU; `--help` parser check only | `python -m medvision_bm.sft.train__SFT-CoT__qwen2_5_vl --help` | exit 0 | sft |
| `script/rft/build_parquet_ds__*.sh` | verl parquet build | skip-network + skip-expensive (dataset download) | cpu | alternative | full | base | none | help-only; tiny-limit run only if data present | `python -m medvision_bm.rft.verl.build_parquet_ds --help` | exit 0 | rft |
| `script/analyze/clinical-decision-analysis/run_CDA_analysis.sh` | CDA | tiny-fixture-runnable (reads parsed/ only) | cpu | alternative | full | base | none | native or synthetic fixture | tbd | report written | analysis |
| `script/misc/regen_all_tasks.py`, `compile_dataset_info.py` | maintainer regen | help-only (needs medvision_ds src) | cpu | optional | full | medvision_ds | none | help-only | `--help` | exit 0 | dataset/maintainer |

## Backend verification plan (draft; depends on scope confirmation)

- REQUIRED `cuda` (CPU substitute: none): local open-weight VLM evaluation (vLLM/HF), SFT/full-FT training, LLM-judge inference. Host verdict TODAY: no GPU (`nvidia-smi` absent; cgroup RAM 32 GiB). Consequence: these native cases are `BLOCKED_REQUIRED_BACKEND` unless a GPU appears before final verification. Guidance for them is drafted from source + launchers and checked with `--help`/parser/import checks.
- ALTERNATIVE `cpu` (full substitute): parse_outputs, summarize_*, metric utils, task JSON loading, dataset loading (needs HF network for real data), parquet building (needs data), CDA/analysis, judge stage0/analyze CPU stages.
- OPTIONAL: API-model evaluation (credentials), image-size probes (model downloads).
- Minimum environment set: ONE private conda prefix (python 3.11) with `pip install -e <repo>` (torch 2.6.0 default wheel = CUDA-capable build, so a later GPU needs no reinstall) + `medvision_ds` from the local v1.4.0 source tree. Skip: 21 `requirements_eval_*.txt` vLLM stacks, `requirements_sft_*.txt`, judge env (vllm 0.11), verl.
- Preparation smoke: `import medvision_bm, medvision_bm.utils.parse_utils`, `mvbm --help`, pipeline `--help`s, `torch.cuda.is_available()` recorded as False.

## Source script inventory (initial decisions)

| Source script | Workflow | Decision | Bundled target | Rationale |
| --- | --- | --- | --- | --- |
| `script/benchmark-*/eval__*.sh` (72) | eval launchers | adapt | `sub-skills/benchmark-evaluation/scripts/make_eval_launcher.py` + `references/launcher-catalog.md` | same skeleton x model; generate launcher from model key/task; load-bearing install order preserved |
| `script/sft/train__*.sh` | SFT launchers | adapt | `sub-skills/sft/scripts/make_sft_launcher.py` or template | parameterised skeleton |
| `script/rft/build_parquet_ds__*.sh` | parquet builders | adapt | `sub-skills/rft/scripts/build_parquet_ds.sh` template | small, safe with explicit paths |
| `script/llm-parsing/run_llm_parsing.sh` + judge_*.py | judge driver | reference-only + wrap `--list/--help` | `sub-skills/llm-judge/references/pipeline.md`, `scripts/check_judge_env.py` | large, env-specific, destructive `prep` step |
| `script/llm-parsing/setup_judge_env.sh` | judge env | reference-only | troubleshooting | mutates envs, big downloads |
| `script/analyze/clinical-decision-analysis/*` | CDA | wrap | `sub-skills/analysis/scripts/run_cda.py` | CPU, reads parsed/ |
| `script/analyze/{process,equation}-accuracy/*.py` | analyses | wrap/adapt | `sub-skills/analysis/scripts/` | CPU |
| `script/visualization/*.sh` | figures | reference-only | `references/visualization-catalog.md` | paper-specific, many env-specific paths |
| `script/misc/{regen_all_tasks.py,compile_dataset_info.py,summarize_datasets.sh}` | maintainer | wrap/reference | dataset sub-skill scripts | need medvision_ds source guard |
| `script/dataset/build_visualize_parquet_ds.sh` | dataset viz | adapt | dataset sub-skill | small |
| `dockerfile/build_and_push.sh` | release | exclude | none | pushes images (credentials) |
| `src/medvision_bm/utils/push_hf_model.py` | release | reference-only | sft references | Hub push (credentials) |
| `unit-test/*` | tests | reference (native candidates) | none (verification only) | not runtime content |

## Scope decision (user-confirmed 2026-09-04 12:2x UTC via AskUserQuestion)

- Extraction scope: FULL pipeline. Sub-skills: environment-setup, dataset-and-tasks, benchmark-evaluation,
  results-parsing-and-metrics, llm-judge-parsing, sft, rft, extending-models-and-tasks, analysis,
  biomedparse-ablation (user added: from `script/ablation/biomedparse`, excluding upstream clone + outputs).
- Visualization: reference-only catalogue (no sub-skill).
- Required-backend limitation ACCEPTED by user: no GPU on this pod; CUDA-required native cases
  (open-weight eval, SFT/full-FT, LLM-judge inference, BiomedParse eval/finetune) will be recorded as
  `BLOCKED_REQUIRED_BACKEND`; env handoff = `partial`; auto-import disabled (import policy is `ask` anyway).
- Environment: NEW private conda prefix `/mnt/vincent-pvc-rwm/.disco/agent/envs/medvision-inspection`
  (python 3.11), `pip install -e <repo>` with default torch==2.6.0 wheel, plus `medvision_ds` from the local
  v1.4.0 source tree. No mutation of conda base or user envs.
- Exclusions confirmed as listed above except `script/ablation/biomedparse` (now INCLUDED; its upstream
  BiomedParse clone and result/output directories stay excluded).
