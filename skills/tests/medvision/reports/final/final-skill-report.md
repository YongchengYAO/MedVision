# Final Skill Report — `medvision`

## Source Snapshot

- Repository: `YongchengYAO/MedVision`, commit `a2c6482e0dbeea7f5cd5a8eddac7c7581f30608c`, branch `master`
- Working tree state: **dirty** (9 modified tracked files, 6 untracked paths; listed in the skill's provenance reference)
- Packages: `medvision_bm` 1.2.0 (this repository), `medvision_ds` 1.4.0 (ships with the Hugging Face dataset), vendored `lmms_eval` 0.3.0
- Evidence consulted: source roots, README, 15 documents under `docs/`, 200+ launcher and pipeline scripts, task lists, dataset-info catalogues, requirements and Dockerfiles, the unit-test suite, two pre-existing repository agent skills, and live inspection of the installed packages

## Generated Skill Summary

- Runtime skill: `skills/medvision/` — 115 files, about 11,900 lines of Markdown
- Root: a 139-line router, five references (glossary, model roster, troubleshooting, visualization catalogue, provenance), one environment-checking script, and the routing metadata
- Ten sub-skills, each a router plus references plus bundled scripts:

| Sub-skill | Purpose |
| --- | --- |
| `environment-setup` | install and repair the stack; requirements catalogue; pin traps |
| `dataset-and-tasks` | dataset concepts, task lists, downloads, annotation versions, data layout |
| `benchmark-evaluation` | run step 1 across 21 entry points and 24 launcher stems per task family |
| `results-parsing-and-metrics` | steps 2 and 3, and the exact meaning of every metric |
| `llm-judge-parsing` | step 4, the format-robust second parse |
| `sft` | supervised fine-tuning, both parameter-efficient and full-parameter |
| `rft` | verl parquet building and the GRPO recipe |
| `analysis` | clinical decision agreement, process and equation accuracy, target-size stratification |
| `extending-models-and-tasks` | add a model or a task |
| `biomedparse-ablation` | the segmentation-specialist comparison |

- 39 bundled script files, every one exercised at least with `--help`

## Coverage Matrix

Full table in `../integration/coverage-depth-matrix.md`. Summary: every public capability in the
confirmed scope maps to a sub-skill or a root reference. No unmapped primary workflow.

| Repo capability group | Required backend | Skill location | Coverage | Synthetic validation | Native validation |
| --- | --- | --- | --- | --- | --- |
| Installation, pins, Docker | any | `environment-setup` | covered | 2 cases pass | help-only pass; pin checker exits 1 correctly |
| Dataset selection, download, versions | any | `dataset-and-tasks` | covered | 2 cases pass | help-only pass; plan ceiling reproduced |
| Evaluation of local models | **cuda** | `benchmark-evaluation` | covered | 2 cases | **blocked**; 21 parsers pass |
| Evaluation of API models | any + credentials | `benchmark-evaluation` | covered | 1 case | skipped, no credentials |
| Parsing, summarizing, metrics | cpu | `results-parsing-and-metrics` | covered | 2 cases pass | 4 native metric tests pass |
| LLM-judge pipeline | **cuda** | `llm-judge-parsing` | covered | 2 cases | 9 native tests pass; inference **blocked** |
| Supervised fine-tuning | **cuda** | `sft` | covered | 2 cases | **blocked**; 10 parsers pass |
| RFT data preparation | cpu | `rft` | covered | 2 cases | help-only pass; prompt test passes |
| GRPO training | external | `rft` (documented) | partial by design | – | external framework |
| Post-hoc analyses | cpu | `analysis` | covered | 2 cases | 5 native tests pass; full pipeline run on a fixture |
| Adding models and tasks | cpu | `extending-models-and-tasks` | covered | 2 cases | registry consistency check passes |
| BiomedParse ablation | **cuda + nvcc** | `biomedparse-ablation` | covered | 2 cases | **blocked**; env checker passes |
| Figures and webpage exports | any | root catalogue | reference-only by design | – | not run |

## Long-Tail Gaps

Full register in `../integration/long-tail-gap-register.md`. The material ones:

| Gap | Risk | Note |
| --- | --- | --- |
| No GPU runtime evidence for four workflows | **high** | the central limitation of this build |
| GRPO training lives in an external fork | medium | recipe documented, training not owned |
| Dataset-package internals not covered | medium | it ships with the dataset, not this repository |
| Per-dataset annotation semantics not enumerated | medium | generic rules covered; a generated catalogue would close it |
| Figure scripts catalogued, not bundled | low | they depend on checkout-local roster files |
| API runtime behaviour not exercised | medium | cap tables documented; no live call made |

## Usability Validation

- 26 case directories: at least two per sub-skill (three for `sft` and `biomedparse-ablation`), two for the root, two integrated
- Every case has a copyable prompt, a reviewer README and gradeable assertions
- Six cases are deliberately refusal cases, where the correct answer declines or qualifies a
  plausible request
- Both integrated cases are adapted from repository-native end-to-end material, not synthesized
- Index and coverage notes: `../../test-cases/index.md`

## Self-Refine

Three fresh agents, each restricted to reading only the runtime skill, answered one real case prompt
each and were graded against the case assertions. **22 of 22 assertions passed**, across three evals, plus a fourth qualitative eval on a scenario that appears nowhere in the skill, which the agent answered by composing facts from four sub-skills. None needed the
repository. Details and caveats in `../self-refine/iteration-1.md`; the notable caveat is that the
metric-troubleshooting eval reuses an example that the skill itself documents, so it tests retrieval
more than generalisation.

Defects found and fixed before this pass: 55 broken cross-links (a systematic depth error in
reference files), an entry-point count error in the root router, an overstated token-budget rule, and
a stray build log inside the tree.

## Native Ground-Truth Verification

Full report: `../verification/native-verification-report.md`.

| Status | Count |
| --- | --- |
| PASS (native tests) | 27 |
| PASS (help/parser checks) | 22 |
| NATIVE_FAIL | 6 |
| SKILL_GAP | **0** |
| BLOCKED_REQUIRED_BACKEND | 5 |
| SKIP_UNSAFE | 4 |

All six native failures are pre-existing repository conditions, not skill defects: two stale test
suites that call a changed API and mis-extract source, one documented data condition where images are
whose predictions are not wrapped in `<answer>` tags (the earlier "images not at their recorded paths" reading is retracted in `reports/verification/native-verification-report.md`), and one stray results directory that breaks the judge invariants gate.
Each is reported to the repository owner in `human-review.md`.

## Import Readiness

- **Ready with an accepted backend limitation.**
- Environment handoff: **partial**. The authoring host has no CUDA device. The user accepted this
  limitation when confirming the extraction scope, choosing full-pipeline coverage with GPU-native
  cases recorded as blocked.
- Backend gate eligible for automatic import: **no**. Four workflows carry
  `BLOCKED_REQUIRED_BACKEND` and must not be presented as backend-verified.
- Blocking issues: none for correctness; all static, licence, link, privacy and structure gates pass.
- Import policy for this run was `ask`. **The user was asked and declined**, choosing to leave the
  skill in the repository for review rather than install it into the managed library. Nothing outside
  this repository was modified.
- To import later: recompute the digest, then run the importer. Both commands are in the handoff.
- Recommended follow-up: re-run the four blocked native cases on a GPU host, then update this report
  and the routing handoff digest before importing.
