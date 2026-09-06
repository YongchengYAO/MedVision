# `skills/` — the MedVision repo skill

This directory holds a **repo-specific Agent Skill** for MedVision: a routed set of
Markdown instruction files plus runnable helper scripts that teach an AI coding agent
(or a new human contributor) how to operate this repository — environment, datasets,
evaluation, parsing, metrics, the LLM judge, SFT, RFT, analyses, and the BiomedParse
ablation.

It is documentation for *operating the repo*, not part of the `medvision_bm` package.
Nothing here is imported by the package, and nothing here runs automatically.

```
skills/
├── medvision/                 # the runtime skill — 115 files, ~11.9k lines of Markdown, 39 scripts
│   ├── SKILL.md               #   root router: read this first
│   ├── references/            #   cross-cutting: glossary, model roster, troubleshooting,
│   │                          #   visualization catalogue, provenance, routing metadata
│   ├── scripts/               #   check_medvision_env.py
│   └── sub-skills/<name>/     #   10 sub-skills, each: SKILL.md + references/ + scripts/
├── medvision-paper/           # companion skill: authoritative facts from the paper
├── medvision-pipeline/        # companion skill: the benchmark pipeline, download, env, SFT launchers
├── disco/routing_decision/    # where this skill sits in the DisCo skill-library taxonomy
└── tests/medvision/           # review artifacts: 26 usability cases + verification reports
```

Paths in this document are written relative to the repository root, and every shell snippet
resolves that root itself with `REPO_ROOT="$(git rev-parse --show-toplevel)"`, so the
commands are safe to paste from any working directory and contain no machine-specific path.


---

## 1. Use it as documentation (no install)

The skill is plain Markdown with a strict routing discipline, so the cheapest way to use
it is to follow that discipline by hand or point an agent at it:

1. Read `medvision/SKILL.md` — it states the three task families, the minimal install, and
   a one-line description of each sub-skill.
2. Jump to the **one** sub-skill that matches the task, e.g.
   `medvision/sub-skills/sft/SKILL.md`.
3. Open only the `references/*.md` the sub-skill names. Each sub-skill splits into
   `workflows.md` (what to run, in order), `cli-reference.md` (every flag and default) and
   `troubleshooting.md` (symptom → cause → fix).

Do not `cat` the tree: it is ~11.9k lines and is written to be read by routing, not in bulk.

With an agent, a prompt like this is enough:

```
Read skills/medvision/SKILL.md, route to the right sub-skill, and answer:
<your question>. Read only the reference files that sub-skill names.
```

---

## 2. Use it as a Claude Code skill

Claude Code discovers skills **one level deep** under a skills root — `<root>/<name>/SKILL.md`.
So you register `medvision` as one skill; its ten sub-skills stay ordinary files that the
root router opens by relative path. Do not flatten them into the skills root.

Resolve the two roots first, so the commands below work from any directory and on any
checkout. Claude Code's configuration directory is `$CLAUDE_CONFIG_DIR` when set and
`$HOME/.claude` otherwise — do not assume the two are the same:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CLAUDE_SKILLS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills"
```

Project scope (this repo only):

```bash
mkdir -p "$REPO_ROOT/.claude/skills"
ln -s "$REPO_ROOT/skills/medvision" "$REPO_ROOT/.claude/skills/medvision"
# or, to pin a copy that will not follow later edits:
# cp -r "$REPO_ROOT/skills/medvision" "$REPO_ROOT/.claude/skills/medvision"
```

User scope (every project):

```bash
mkdir -p "$CLAUDE_SKILLS"
ln -s "$REPO_ROOT/skills/medvision" "$CLAUDE_SKILLS/medvision"
```

Both use an absolute link target on purpose: a relative one breaks as soon as the link or
the checkout is moved. Re-point the link with `ln -sfn` if you relocate the repository.

Then restart the session and invoke it explicitly:

```
/medvision   how do I resume an interrupted multi-GPU evaluation?
```

**Explicit invocation is required.** Every `SKILL.md` here sets
`disable-model-invocation: true`, so the model will never load the skill on its own — a
deliberate choice from the skill-library convention (bulk-imported repo skills would
otherwise crowd the model-visible skill list). If you want it to trigger automatically for
MedVision work, delete that line from `medvision/SKILL.md`; leave it in the sub-skills,
which are not registered as skills anyway. Note that a modified runtime tree no longer
matches its recorded digest, which matters only if you later import it (§3).

Relationship to the two companion skills that sit beside it in `skills/`:

| Skill | Scope | Auto-invoked |
| --- | --- | --- |
| `medvision-paper` | paper facts: scale numbers, annotation rules, metric definitions, V0 recipe | yes |
| `medvision-pipeline` | short how-to: the 4-step pipeline, download, env, where SFT launchers live | yes |
| `skills/medvision` (this) | the full operating surface: every flag, launcher, pin trap and failure mode | no, `/medvision` |

They overlap by design: the two small ones are quick context, this one is the deep
reference. Keep all three, or retire `medvision-pipeline` if you register this one and are
happy to invoke it by hand.

---

## 3. Import it into the DisCo skill library

This skill was generated by DisCo's `create-repo-skill` and refreshed by
`refresh-repo-skill`; its intended long-term home is the managed collection under
`<disco-agent-root>/skills/repositories/repo-skills/` — where `<disco-agent-root>` is
`$DISCO_CODING_AGENT_DIR` when set and `$HOME/.disco/agent` otherwise — reachable through
`repo-skills-router`.
**It was deliberately not imported** — it lives here for review instead.

If you decide to import it:

1. Recompute the content digest — `disco/routing_decision/classification.json` carries a
   recorded `skill_content_sha256` that is **stale** relative to the current tree, so it must
   be regenerated before import:

   ```bash
   REPO_ROOT="$(git rev-parse --show-toplevel)"
   python "$REPO_ROOT/skills/tests/medvision/reports/routing/compute_skill_digest.py" \
       "$REPO_ROOT/skills/medvision" \
       --write "$REPO_ROOT/skills/disco/routing_decision/classification.json"
   ```

2. Import with the routing handoff (or ask Claude to run the
   `import-repo-skills-to-agent` skill, which wraps this with validation, router merging
   and rollback):

   ```bash
   CLAUDE_SKILLS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills"
   node "$CLAUDE_SKILLS/verify-repo-skill/scripts/import_repo_skill.mjs" \
       --routing-entry "$REPO_ROOT/skills/disco/routing_decision/classification.json" \
       "$REPO_ROOT/skills/medvision"
   ```

The importer refuses symbolic links anywhere inside the runtime tree, so import from the
real directory, never from a symlinked copy. Delete any `__pycache__` directories under
`skills/medvision/` first — `compute_skill_digest.py` hashes every file it finds, so stray `.pyc`
artifacts change the digest (the tree is exactly 115 source files when clean).

---

## 4. Routing map

| Sub-skill | Use it when |
| --- | --- |
| `environment-setup` | installing or repairing the stack; the 25 frozen requirements files, Docker images, load-bearing install order, version-pin traps |
| `dataset-and-tasks` | choosing and naming data: dataset configs, task-list JSONs, downloads, annotation versions and the planner pin, `Data/` layout, parquet snapshots |
| `benchmark-evaluation` | running step 1: the 21 `eval__<model>` entry points, launcher anatomy, sample limits and token budgets, parallelism, the resume cache, perceived image size |
| `results-parsing-and-metrics` | steps 2–3: `parse_outputs`, the three summarizers, and the exact denominator and failure handling of every metric |
| `llm-judge-parsing` | step 4: the format-robust second parse — stages, judge environment, roster YAMLs, reproducibility caveats |
| `sft` | supervised fine-tuning: dataset construction, sample-limit semantics, LoRA and full-parameter entry points, the two-phase launcher pattern, merging and resuming |
| `rft` | verl-ready parquet building and the GRPO recipe behind MedVision-V0 |
| `analysis` | post-hoc studies: clinical decision agreement, process and equation accuracy, detection by target size |
| `extending-models-and-tasks` | maintainer work: add a model across all required sites, or add a task/dataset YAML pair |
| `biomedparse-ablation` | the segmentation-specialist comparison: evaluating and fine-tuning BiomedParse v2 with MedVision's metrics |

Cross-cutting, at the root: `references/concepts-and-glossary.md` (vocabulary — task names
vs dataset configs, planes, splits, annotation versions, metric names),
`references/model-roster.md` (entry point, `lmms_eval` key, dependency stack, parallelism,
hardware and perceived image size per model), `references/troubleshooting.md`,
`references/visualization-catalog.md`, `references/repo-provenance.md`.
`references/repo-routing-metadata.json` is library infrastructure — there is nothing in it
for a task.

---

## 5. Bundled scripts

All 39 files are read-only or scaffolding: 35 executables (26 Python, 9 shell), every one exercised at
least with `--help`, plus 4 config/data files (`model_catalog.json`, three `*.yaml`). Start
with the environment check:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
python "$REPO_ROOT/skills/medvision/scripts/check_medvision_env.py" --data-dir "$REPO_ROOT/Data"
# --json | --skip-optional | --require-gpu | --repo-root <checkout>
```

It reports both packages, the pinned foundation versions, GPU visibility, the `mvbm` CLI
and the seven `MedVision_*` dataset/loader variables plus `MEDVISION_RESP_CACHE` (not the
`MEDVISION_SFT_*` / `MEDVISION_SCALED_PS_*` / `MEDVISION_DS_SRC` family), names the pin traps it finds, installs nothing, prints no
secret values, and exits non-zero when `medvision_bm` is missing.

The rest live under `medvision/sub-skills/<name>/scripts/`:

| Sub-skill | Scripts |
| --- | --- |
| `environment-setup` | `check_env_pins.py`, `build_local_wheel.sh` |
| `dataset-and-tasks` | `list_tasks.py`, `inspect_benchmark_plan.py`, `download_datasets.sh`, `build_parquet_ds.sh` |
| `benchmark-evaluation` | `check_results_tree.py`, `make_eval_launcher.py`, `model_catalog.json` |
| `results-parsing-and-metrics` | `parse_and_summarize.sh`, `inspect_summary.py`, `metrics_demo.py` |
| `llm-judge-parsing` | `check_judge_env.py`, `make_roster_yaml.py` |
| `sft` | `check_sample_limits.py`, `inspect_prepared_dataset.py`, `sft_launcher_template.sh` |
| `rft` | `build_parquet_ds.sh`, `inspect_parquet_ds.py` |
| `analysis` | `run_cda.sh`, `detection_target_size.sh`, `analyze_{process,equation}_accuracy_{TL,AD}.py`, `cda/` |
| `extending-models-and-tasks` | `list_registered_models.py`, `list_task_yamls.py`, `scaffold_new_model.py` |
| `biomedparse-ablation` | `check_biomedparse_env.py`, `env_template.sh` |

---

## 6. Rules the skill enforces

Worth knowing before you delegate anything to it, because it will push back:

- It will not start an evaluation, fine-tuning run, dataset download or judge sweep unless
  asked; it states the GPU-hour, credit or disk cost first. One dataset config downloads
  the whole source dataset.
- It treats `Data/`, `Results/`, `SFT/` and `completed_tasks/` as read-only data and
  deduplicates into a new directory rather than editing result JSONLs.
- It keeps `MedVision_PLANNER_VERSION` fixed for the life of a study and records it with
  the results — annotation versions change the T/L sample set.
- It never mixes model dependency stacks in one environment.
- It reports a metric only with its denominator convention.
- It states that MedVision models are research artifacts, not clinical tools.

---

## 7. Keeping it current

`medvision/references/repo-provenance.md` is the staleness baseline: it pins the source
commit, the working-tree state, `medvision_bm` 1.2.0 / `medvision_ds` 1.4.0 / vendored
`lmms_eval` 0.3.0, and the evidence paths. Re-read its *Refresh Check* section, which lists
the exact conditions that should trigger a refresh (HEAD moved, package version bumped, new
`eval__*` entry point, new annotation release, SFT prepared-dataset hand-off changed).

Current state: the snapshot was re-stamped to commit `780e247` on 2026-09-05.
`check_repo_provenance.py` now reports `stale` against this checkout — the commit still matches, but
`dirty_paths` does not, because the docsite/skills/README fixes are uncommitted. Re-stamp it (or commit
those changes) before importing. Re-run the refresh when one of
the triggers in that file fires:

```
Run the refresh-repo-skill skill on skills/medvision against this checkout.
```

---

## 8. Review artifacts and known limits

`tests/medvision/` is review material, not runtime content — do not import or register it.

- `test-cases/index.md` — 26 usability cases (at least two per sub-skill, two for the root, two
  integrated; six are deliberate refusal cases). Each directory has `user_request.txt` (a
  copyable prompt), `README.md` (persona and expected behaviour) and `assertions.json`
  (gradeable PASS/FAIL). To run one: give a fresh agent the skill and only the
  `user_request.txt` text, then grade its answer against `assertions.json`.
- `reports/final/human-review.md` — what to review first, the judgement calls made, and six
  repository defects the build surfaced. Five are fixed as of `780e247` (unit-test call
  sites, the `scaledPS` extraction regex, the stale registry snippet in
  `docs/New-Models-Guide.md`, eight unused task-YAML variants, the `sft/config/*.yaml`
  package-data entry). One is still open: `Results/MedVision-detect-v2` holds two `_CoT`-suffixed
  directories that sit alongside their non-suffixed siblings. They are not duplicates — they carry
  the `-CoT` prompt-variant runs (their JSONLs are named `*_BoxCoordinate_Task01_<Plane>-CoT.jsonl`). One of
  them **is** the roster's MedVision-V0 entry (`config-detect-CoT.yaml:4`); only `…__v2_CoT` is off-roster.
  The judge-invariants test walks every `<model>/llm-parsed_<judge>/` directory rather than the roster, so
  the off-roster one makes it fail on Detection. All 19 roster models are clean.
- `reports/final/final-skill-report.md` — coverage matrix, long-tail gap register, and the
  self-refine result (22/22 assertions passed across three agents reading only the skill).
- `reports/verification/` — native verification runs plus the 2026-09-04 and 2026-09-05
  staleness audits, refresh-verification notes and `verification-report.json`.

Two limits to carry with you:

1. **Four GPU workflows have no runtime evidence** — local model evaluation, fine-tuning,
   judge inference, and the ablation tracks. Their guidance comes from source, launchers and
   docs, checked with parser and import tests on a CPU-only host. Treat as unverified until
   the blocked cases are run on a GPU.
2. **Licence** — all eleven `SKILL.md` files now declare `license: CC-BY-4.0`, matching `LICENSE`
   and `pyproject.toml:11`. (GitHub's SPDX auto-detection still returns NOASSERTION for the
   repository, which is where the original placeholder came from.)
