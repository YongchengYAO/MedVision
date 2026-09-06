# Sub-skill drafting handoffs (main-agent review log)

Review rubric applied to each: identity (dir = frontmatter name), files on disk, scope, depth, script bundling,
troubleshooting present, evidence-backed, self-contained (no out-of-tree links), executable scripts, frontmatter contract,
no private paths. Static checks run by the main agent: `grep` for out-of-tree Markdown links, `/mnt/`, `/home/`,
`/root/Documents`, `medvision-inspection`, `conda activate`; independent re-run of one bundled script.

## results-parsing-and-metrics — ACCEPTED (first pass)
- Files: SKILL.md (103 lines) + 6 references + 3 scripts (1,564 lines total). Frontmatter OK. No out-of-tree links, no leaks.
- Main-agent re-run: `scripts/metrics_demo.py` → ALL CHECKS PASSED (exit 0); `parse_and_summarize.sh --help` OK.
- Evidence: parse_outputs, 3 summarizers, parse_utils, configs constants, medvision_utils process/aggregate fns, README §2-4,
  llm-parsing README outputs, real Results/ records read-only; 36 signatures via inspect.signature.
- Source scripts: parse/summarize/remove_duplicate_samples WRAPPED; metrics_demo + inspect_summary NEW (adapted); box-size
  analyzers/viz + scaledPS launchers REFERENCE-ONLY (owned by analysis / benchmark-evaluation).
- Troubleshooting: 20 rows (missing parsed, resps_key, NaN nMAE, --limit, missing medvision_ds/torch/transformers, plan lookup,
  removed-samples root, IoU>0.5 vs mean, pixel-unit boxes, label maps, missing _results.json, duplicated doc_id, -p/pickling, …).
- Native: unit-test/detection-metric-failure/test-{1,2,3} PASS; unit-test/nMAE/test-2 PASS; nMAE/test-1 and scaledPS/test-1
  FAIL as STALE TESTS (API drift: `_compute_physical_diagonal` keyword-only `explicit_scale`; regex-exec captures decorator) —
  record as stale, not as skill gaps; nMAE/test-{3,4,5}, scaledPS/test-{2..5} skip-network.
- Difficult cases proposed: (1) mean IoU 0.62 vs IoU>0.5 0.41 explanation; (2) re-summarize T/L from llm-parsed dir for two models
  with removed-samples filter.
- Gaps: `viz_detection_performance_per_boxImgRatio` default YAML not shipped in the package; `remove_duplicate_samples` exit 0 on
  bad --dir; parse_outputs imports torch/transformers/medvision_ds at module level (echo in environment-setup + root troubleshooting).

## dataset-and-tasks — ACCEPTED (first pass)
- Files: SKILL.md (81 lines) + 7 references + 4 scripts (1,246 lines). Frontmatter OK. No out-of-tree links, no leaks.
- Evidence: README data sections, all tasks_list JSONs, utils/{data_utils,utils,plan_utils,install_utils,configs_to_tasks,
  size_dist_utils,summarize_datasets,configs_to_*_sizes}.py, download_datasets.py, dataset/*.py, sft_utils loader, script/dataset,
  script/misc, dataset-info listings, docs/file-structure, dataset tutorial, release notes, planner-version unit test, dataset
  source `MedVision.py` (banners, version tables, tracker, QC figures), local plans KiPA22/CrossMoDA read-only.
- Inspection: 10 `--help`s PASS; `tasks_to_configs` behaviour; feature dicts README==loader (4/4); all 4 scripts exercised incl.
  error paths and fallbacks; ceiling fallback demonstrated on KiPA22 (pin 1.2.0 → 1.1.1 with `[plan_utils]` warning).
- Source scripts: build_visualize_parquet_ds.sh ADAPTED; download CLI WRAPPED; list_tasks/inspect_benchmark_plan NEW;
  misc regen/compile/convert/size-probe scripts REFERENCE-ONLY (need dataset source checkout / hours / hard-coded paths).
- Troubleshooting: ~40 rows incl. all 10 loader banners, BuilderConfig-not-found for `-CoT`/`BoxCoordinate`, gated tokens,
  stale Arrow, NonMatchingSplitsSizesError, parquet asserts, plan OOM, compile_dataset_info guard.
- Native: help-only ×5 PASS; planner-version unit test = skip-network (documented).
- Difficult cases: (1) pinned 1.2.0 T/L study (ceiling + ACK + leaderboard 391 vs 303); (2) SFT list → coronal plane-OOD eval
  configs offline via `list_tasks.py --plane Coronal --cot add`.
- VERIFIED FINDING to propagate: `tasks_to_configs` does not strip `-CoT`, so `download_datasets --tasks_json <eval -CoT list>`
  yields `BuilderConfig '..._Axial-CoT_Test' not found`; use SFT-style lists, CSVs, or the bundled wrapper (strips suffix).
- Integration decisions: keep the single `list_tasks.py` in this sub-skill (root SKILL.md links to it; no root copy);
  strip harness-created `.claude/.cc-writes` and `__pycache__` before packaging.

## environment-setup — ACCEPTED (completed during the rate-limit outage; reviewed by main agent)
- Files: SKILL.md (68 lines) + 5 references (cli-reference, environment-variables, installation, requirements-catalog,
  troubleshooting; 651 lines total) + 2 scripts. Frontmatter contract OK.
- Static checks: no private-path or inspection-env leaks; no Markdown links leaving the skill tree; every relative link target
  exists.
- Main-agent script re-run: `build_local_wheel.sh --help` OK and `bash -n` clean; `check_env_pins.py --help` OK;
  `check_env_pins.py --requirements requirements/requirements_eval_qwen25vl.txt` reported 11 mismatches and exited 1 as
  documented, and correctly printed the resolved `medvision_bm.__file__` / `medvision_ds.__file__` so the editable-vs-copy
  trap is visible at runtime.

## biomedparse-ablation — ACCEPTED (completed during the outage; reviewed by main agent)
- Files: SKILL.md (108 lines) + 6 references (overview-and-fairness, setup, tracks, cli-reference, visualization-convention,
  troubleshooting; 958 lines) + 2 scripts. Frontmatter OK.
- Static checks: clean (no leaks, no out-of-tree links, no broken targets). The agent respected the instruction never to
  enumerate `third_party/`, `data/`, `models/`, `results/`, `figures/`.
- Main-agent script re-run: `env_template.sh` passes `bash -n`; `check_biomedparse_env.py --help` OK; a real run reported the
  three missing required imports (detectron2/lightning/upstream) plus 7 pin warnings and the absent GPU as findings rather
  than a traceback, exiting 1 as designed.

## rft — ACCEPTED (completed during the outage; reviewed by main agent)
- Files: SKILL.md (110 lines) + 5 references (workflows, cli-reference, parquet-schema, rft-recipes, troubleshooting;
  686 lines) + 2 scripts. Frontmatter OK.
- Static checks: clean. The verl fork is described as an external dependency with its recipes documented, not bundled.
- Main-agent script re-run: `build_parquet_ds.sh` passes `bash -n` and `--help`; `inspect_parquet_ds.py --help` OK.

## Relaunch note (rate limit)
The first drafting round hit an account session limit. `results-parsing-and-metrics` and `dataset-and-tasks` finished before it;
`environment-setup`, `biomedparse-ablation` and `rft` finished on a later retry. The remaining five
(`llm-judge-parsing`, `analysis`, `benchmark-evaluation`, `extending-models-and-tasks`, `sft`) left partial output on disk and
were relaunched as fresh agents on a different model with an explicit "what exists / what is missing" delta, instructed to
read and keep the partial files.

## llm-judge-parsing — ACCEPTED (relaunched agent; reviewed by main agent)
- Files: SKILL.md (155 lines) + 6 references (pipeline, recipes, design-notes, judge-environment, cli-reference,
  troubleshooting; 1,975 lines total) + 2 scripts. Frontmatter contract OK.
- Static checks: no private-path leaks, no out-of-tree links, all relative links resolve.
- Main-agent script re-run: `check_judge_env.py --help` OK; a real run on this GPU-less host exits 1 with a clear
  "Stage 1 cannot run here" verdict instead of a traceback, and prints the caller's own interpreter path (runtime-derived,
  not baked into the file — verified by grep). `make_roster_yaml.py --help` OK.
- The agent fixed two defects it found in its own earlier partial output: an unknown judge key falling back silently to a
  snapshot registry, and roster entries printed as "skip" while actually being kept.
- Native: 9 of 10 judge unit tests PASS. `unit-test/llm-parsing/test-8.py` fails, and the agent traced it to a stray
  non-roster `llm-parsed_gemma-4-31b/` directory under a `_CoT`-suffixed duplicate model folder in
  `Results/MedVision-detect-v2` that holds strict parsed records. All 19 roster models are clean. This is repository data
  hygiene, not a skill defect; it is documented in the sub-skill's troubleshooting with the fix. Independently reproduced
  by the main agent.
- Difficult cases proposed: (1) extend a finished 19-model campaign by one model without re-judging;
  (2) diagnose differing judge-invalid rates across machines. Both written as usability cases.
- Noted tension: the judge pipeline ships only in the repository checkout, not in the installed package, so runnable
  driver commands necessarily reference a checkout path. The agent used a `<repo>` placeholder and made the prerequisite
  explicit. ACCEPTED as-is: bundling a thin wrapper would drift from a driver whose help and step table are generated
  from a single source.

## benchmark-evaluation — ACCEPTED (relaunched agent; reviewed by main agent)
- Files: SKILL.md (148 lines) + 6 references (workflows, launcher-anatomy, cli-reference, model-catalog,
  image-processing-and-token-budgets, hardware-and-parallelism; 1,073 lines) + 3 scripts. Frontmatter OK.
- Static checks: clean.
- Main-agent script re-run: `make_eval_launcher.py --list-models` prints a 26-model table with backend, method, entry
  point, lmms_eval key and whether a repository launcher exists; generating a detect launcher for a vLLM model produces a
  faithful copy of the repository skeleton including the environment name, the wheel-build block and the parallelism
  note. `model_catalog.json` is valid JSON with provenance, task, flag-profile and 26 model entries.
  `check_results_tree.py --help` OK.
- Correctly defers hardware and pin tables to the root `references/model-roster.md` instead of duplicating them.

## analysis — ACCEPTED (relaunched agent; reviewed by main agent)
- Files: SKILL.md (137 lines) + 6 references + 14 script files including a copied clinical-analysis
  package (21 files total). Frontmatter OK; static checks clean.
- Main-agent script re-run: `--help` OK for all four bundled analyzers and both shell wrappers; both
  wrappers pass `bash -n`.
- The agent ran the full clinical-analysis pipeline end to end on a synthetic fixture inside a temp
  directory (two tasks, two models, three proxies) and produced all documented outputs, then removed
  the caches it created. It also added actionable import guards to the four copied analyzers, which
  previously produced bare tracebacks.
- Native: the five equation-accuracy tests pass.
- Correction it forced: the brief claimed a minimum-group-size constant is applied by the box-size
  analyzers. Verified false; that constant is used only by the two summarizers. The sub-skill
  documents the accurate behaviour and cross-links.
- Difficult cases proposed and written up: clinical analysis on a task with no cutoff table, and an
  equation-accuracy number that looks best because of low coverage.

## extending-models-and-tasks — ACCEPTED (relaunched agent; reviewed by main agent)
- Files: SKILL.md (148 lines) + 5 references + 3 scripts. Frontmatter OK; static checks clean.
- Main-agent script re-run: all three scripts `--help` OK; the registry checker reports 20 registered
  keys against 19 dispatch branches with no mismatches, listing every image-processing function.
- The agent verified its scaffolder by generating all three model kinds into a temp directory and
  byte-compiling every generated Python file; it refuses to write into a checkout by default.
- Two repository findings: the new-models guide embeds a stale registry snippet, and the task-YAML
  inventory found eight defects in unused variant YAMLs. Both recorded in the integration notes.

## sft — ACCEPTED (relaunched agent; reviewed by main agent)
- Files: SKILL.md (136 lines) + 6 references + 3 scripts. Frontmatter OK; static checks clean.
- Main-agent script re-run: both Python helpers `--help` OK; the launcher template passes `bash -n`
  and prints usage with `-h`.
- The agent verified that all ten training entry points expose an identical 55-flag surface by
  diffing their help output pairwise, and fixed a crash in the pre-existing limit checker.
- Correction it forced: the structure plan and my brief referenced sharded-training YAML files under
  the SFT package. Verified that directory does not exist at this commit even though the package
  metadata still declares it; all such settings are command-line flags. The sub-skill documents the
  flags and invents nothing.
- Difficult cases proposed and written up: full-parameter training out of memory at the first
  optimizer step, and a sample limit larger than the available pool.
