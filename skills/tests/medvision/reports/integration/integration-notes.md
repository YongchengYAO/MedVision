# Whole-skill integration notes — `medvision`

Written by the main agent after all ten sub-skills were accepted, before verification.

## Accepted sub-skills and what they own

| Sub-skill | Owns | Files |
| --- | --- | --- |
| `environment-setup` | installing and repairing the stack, requirements catalogue, Docker, install order, pin traps | 8 |
| `dataset-and-tasks` | dataset concepts, task lists, downloading, annotation versions, data layout, parquet snapshots | 12 |
| `benchmark-evaluation` | running step 1: entry points, launchers, budgets, parallelism, resume, perceived image size | 10 |
| `results-parsing-and-metrics` | steps 2 and 3: parsing, summarizing, every metric definition | 10 |
| `llm-judge-parsing` | step 4: the judge pipeline, its environment, rosters, artifacts, reproducibility | 9 |
| `sft` | supervised fine-tuning: data construction, both training paths, merging, resuming | 9 |
| `rft` | verl parquet building and the GRPO recipe run in the external fork | 8 |
| `analysis` | clinical decision agreement, process and equation accuracy, detection by target size | 21 |
| `extending-models-and-tasks` | maintainer work: adding a model or a task | 9 |
| `biomedparse-ablation` | the segmentation-specialist comparison, both tracks | 9 |

Root: `SKILL.md`, five references, one script. Total runtime tree: 113 files, about 11,300 lines of
Markdown.

## Cross-references added or corrected during integration

- **Systematic link-depth defect.** 55 relative links were broken because several agents wrote
  sibling and root links from inside `references/` at the depth that would be correct from
  `SKILL.md`. Repaired all of them by resolving each intended target against the real tree and
  recomputing the relative path. The tree now has 202 relative links, 0 broken, 0 escaping the skill
  directory.
- One prose passage in the ablation sub-skill described how a repository launcher sources its own
  shared environment file. It read like a broken link, so it was rephrased; the behaviour statement
  is unchanged and remains correctly marked reference-only.
- The root roster is the single model table. `benchmark-evaluation` links to it for hardware and
  pins instead of duplicating them, and keeps only evaluation wiring locally.
- The task-list helper lives once, in `dataset-and-tasks`. The root and `benchmark-evaluation` link
  to it rather than shipping a second copy.

## Corrections the sub-skill agents forced on my root files

1. **Entry-point count.** The root router said 24 evaluation entry points. Verified: there are 21
   `eval__*.py` modules and 24 launcher stems per task family, because four modules each serve two
   checkpoints. Corrected.
2. **Token budgets.** The roster stated a flat 4096 local / 16000 API rule. Verified against all 72
   launchers: 46 use 4096, nine use 16000, six use 16384 (the MiniMax pair on all three tasks), five
   local launchers use 16000, and the GPT launchers use 4096, below the API default. Corrected to
   describe the rule plus its exceptions.
3. **A claim I rejected.** One agent reported that the `lingshu`, `meddr` and `huatuogpt_vision`
   optional-dependency extras do not exist. Verified directly in the vendored package metadata: all
   three are defined. The root roster was already correct and was left unchanged.

## Repository findings surfaced during integration

These concern the repository, not the generated skill, and are reported to the owner rather than
worked around:

1. Two stale test suites fail against current source: three files in `unit-test/nMAE/` call
   `_compute_physical_diagonal()` without its now-required keyword-only argument, and
   `unit-test/scaledPS/test-1.py` extracts source with a regex that now captures a trailing
   decorator.
2. A stray non-roster directory under `Results/MedVision-detect-v2` holds strict parsed records
   inside an `llm-parsed_<judge>/` folder, which makes the judge invariants test fail. All 19 roster
   models are clean.
3. `pyproject.toml` declares `sft/config/*.yaml` as package data, but that directory does not exist
   at this commit. All sharded-training settings are passed as command-line flags. The skill
   documents the flags and does not invent the files.
4. The new-models guide embeds a stale registry snippet, showing keys that have since been renamed
   or collapsed and omitting several current ones. The skill documents the live registry, verified
   by script.
5. The task-YAML inventory tool found eight defects in unused variant YAMLs: four duplicate task
   names caused by a trailing space plus a wrong include, and three files setting the dataset path
   key instead of the dataset name key. None affect published results.

## Duplicate or conflicting content removed

- Hardware tables existed in both the root roster and a sub-skill draft; the sub-skill now links.
- Metric definitions appeared briefly in two places; they live only in the metrics reference, with
  the glossary carrying one-line summaries.
- Two sub-skills described the visualization convention; it is stated once, in the ablation
  sub-skill, and linked from the root catalogue.

## Gate results

- Every primary workflow has a focused sub-skill; every non-trivial support workflow is reachable
  from an owning sub-skill or a root reference; both maintainer workflows have dedicated guidance.
- Every sub-skill has a `references/troubleshooting.md`.
- Every reference and script is linked from its nearest `SKILL.md`.
- No runtime instruction requires executing a repository path. Two pipelines that genuinely ship
  only in a checkout state that prerequisite explicitly and use a placeholder rather than a concrete
  path.
- Frontmatter contract passes on all 11 `SKILL.md` files; the license gate reports one consistent
  value across the tree; no caches, logs, symlinks or private paths remain.
