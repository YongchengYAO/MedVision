# Human review notes — `medvision`

## What to look at first

1. `skills/medvision/SKILL.md` — does the routing match how you actually think about the project?
2. `skills/medvision/references/model-roster.md` — the single model table. It was corrected twice
   during integration and is the most likely place for drift as models are added.
3. `skills/medvision/sub-skills/*/references/troubleshooting.md` — these encode operational
   knowledge that is not written down anywhere in the repository. Worth a skim for anything wrong.

## Judgement calls you may want to revisit

- **Scope.** The BiomedParse ablation was included at your request. Visualization was catalogued
  rather than turned into a sub-skill, because those scripts depend on roster files and output trees
  that only exist inside a checkout.
- **The reinforcement stage.** Training runs in an external fork, so the `rft` sub-skill documents
  the recipe and owns only dataset preparation. It does not claim to run GRPO.
- **Two checkout-only pipelines.** The judge driver and the ablation launchers ship only in the
  repository, not in the installed package, so their runnable instructions necessarily name a
  checkout path. They are marked reference-only and use a placeholder. The alternative, bundling
  thin wrappers, was rejected because they would drift from drivers whose help text is generated
  from a single source.
- **Licence.** Now `NOASSERTION`: the 2026-09-04 refresh re-queried GitHub, which reports that it did
  not auto-detect a licence (the creation-time query had returned nothing usable, hence `NO_LICENSE`).
  Your repository ships a Creative Commons Attribution 4.0 file. Replace the value if you want the
  skill to state that.

## Refresh of 2026-09-04 (commit `980e9df`)

What changed in the repository: prepared SFT datasets are named from their true train sizes after a
load+split stage, every entry point skips that stage when given `--prepared_ds_dir` together with
`--skip_process_dataset`, and the 21 `script/sft` launchers now tee phase A to
`prepare_dataset.log`, read the reported directory back, and pass it to phase B.

Public files changed (all under `sub-skills/sft/`): `SKILL.md` (description, invariants 2 and 7),
`references/data-preparation.md`, `workflows.md`, `troubleshooting.md` (two new entries),
`cli-reference.md`, `launcher-catalog.md`, `scripts/check_sample_limits.py` (wording),
`scripts/sft_launcher_template.sh` (now mirrors the launchers' hand-off). Plus
`references/repo-provenance.md` (root) and the licence line in all eleven `SKILL.md` files.

Worth a look:

1. `sub-skills/sft/scripts/sft_launcher_template.sh` — the new capture block after phase A. It was
   exercised with a stub entry point (success and missing-report paths), not with a real preparation
   run.
2. `sub-skills/sft/references/troubleshooting.md` §2 — the two new entries describe the symptom a user
   sees when phase B is launched without `--prepared_ds_dir`, and the launchers' deliberate abort.
3. `skills/tests/medvision/test-cases/sub-skills/sft/phase-b-stalls-in-dataset-loading/` — the new
   usability case for the refreshed behaviour.

Reports: `reports/verification/staleness-audit-2026-09-04.md`,
`reports/verification/refresh-verification-2026-09-04.md`, `reports/routing/refresh-2026-09-04.md`,
`reports/license-resolution.json`. The skill remains **not imported**, as you decided at creation.

## Refresh of 2026-09-05 (commit `780e247`)

The repository moved by exactly one commit, and that commit is the tracked half of the dirty tree the
previous snapshot described — the maintainer fixes made while the skill was being reviewed. So most
of this refresh is a re-stamp. Three claims were genuinely stale, all about what the dataset-package
install does to your pins:

1. **`biomedparse-ablation`** said the installer "force-reinstalls its dependencies and lifts
   `huggingface-hub`/`packaging`". It no longer does. `install_medvision_ds` is a plain wheel install
   followed by `--force-reinstall --no-deps`, and `medvision_ds` declares
   `huggingface_hub>=0.35.3,<2.0` — the ablation's pinned 0.36.0 sits inside that and survives. The
   row now names what *can* still move: `pip install --upgrade build` (which can lift `packaging`
   past 23.0) and any dep that is missing or out of range (`opencv-python` versus the env's
   `opencv-python-headless`, the exact `datasets==3.6.0`).
2. **`environment-setup`** blamed the same mechanism in miniature: "`install_medvision_ds` … 
   re-resolved hub (0.36.0 → 1.x)". Rewritten around the trigger that is real — a hub *missing or
   below* the 0.35.3 floor gets resolved up to the newest in-range release, which is 1.x.
3. **`environment-setup/references/installation.md`** claimed the package ships `sft/config/*.yaml`
   as package data. `780e247` dropped that entry and the directory does not exist.

Everything else was re-verified and retained, including the two bundled checkers, which now both pass
clean against the fixed tree (`list_task_yamls.py` exit 0 — the eight YAML defects from the last
review are gone; `list_registered_models.py` exit 0 across 20 registered keys).

Public files changed: `references/repo-provenance.md` (new snapshot, plus a `counts_at_snapshot`
block and two new refresh triggers), `sub-skills/environment-setup/references/installation.md`,
`sub-skills/environment-setup/references/troubleshooting.md` (two rows),
`sub-skills/biomedparse-ablation/references/troubleshooting.md`,
`sub-skills/biomedparse-ablation/references/setup.md`. No file was added or removed, no route moved,
and the licence re-query at this commit returned `NOASSERTION` again, so no frontmatter changed.

Worth a look:

1. `sub-skills/biomedparse-ablation/references/troubleshooting.md` — the rewritten row is longer than
   its neighbours because it now discriminates rather than blanket-blames. Trim it if you prefer the
   table terse.
2. `references/repo-provenance.md` — two deliberate additions. The `counts_at_snapshot` block gives a
   future refresh six numbers that two bundled scripts can re-check in seconds. And `dirty_paths` is
   now the literal `git status --porcelain` list: the old annotated form
   (`"skills/ (untracked; …)"`) could never compare equal, so the freshness checker reported `stale`
   even against the checkout the snapshot was written from. It now reports `current`.
3. Usability cases: `sub-skills/environment-setup/hub-lifted-after-dataset-install` was rewritten so
   its scenario is a below-floor hub (0.34.4), and a new case
   `sub-skills/biomedparse-ablation/which-pins-the-dataset-install-can-still-move` fails if answered
   from the pre-fix behaviour.

Reports: `reports/verification/staleness-audit-2026-09-05.md`,
`reports/verification/refresh-verification-2026-09-05.md`,
`reports/verification/verification-report.json`, `reports/routing/refresh-2026-09-05.md`,
`reports/license-resolution.json`. The skill remains **not imported**.

One tooling note: the bundled licence resolver could not run here (this host's only `node` is
v6.13.1, which cannot parse ESM). The identical `gh api` query it wraps was run directly and its
result recorded in the resolver's own report shape.

## Findings about the repository itself

**Status as of `780e247`: items 1, 2, 4, 5 and 6 are fixed; only item 3 remains open.**

These came out of running your own tests as ground truth. None is caused by the skill.

1. **[FIXED]** Three files in `unit-test/nMAE/` call `_compute_physical_diagonal()` without its now-required
   keyword-only `explicit_scale` argument and fail immediately.
2. **[FIXED]** `unit-test/scaledPS/test-1.py` extracts source with a regex that now also captures a trailing
   decorator, producing a syntax error. Tests 2 to 5 of that suite pass.
3. **[OPEN]** Two stray directories under `Results/MedVision-detect-v2` — `_CoT`-suffixed duplicates of
   `MedVision__fullRFT__…PRxAnswer_s250` and `MedVision__fullSFT__…__v2` — hold strict parsed records
   inside an `llm-parsed_<judge>/` folder, which makes the judge invariants test fail on Detection.
   All 19 roster models are clean.
4. **[FIXED in `780e247`]** `pyproject.toml` declared `sft/config/*.yaml` as package data for a directory
   that does not exist.
5. **[FIXED]** `docs/New-Models-Guide.md` embedded a stale registry snippet: it still shows split HealthGPT keys
   and omits several currently registered models.
6. **[FIXED]** The task-YAML inventory found eight defects in unused variant files: four duplicate task names
   caused by a trailing space combined with a wrong include, and three files that set the dataset
   path key instead of the dataset name key. None affects published results.

## Residual risk

The four GPU workflows — local model evaluation, fine-tuning, judge inference, and the ablation
tracks — have no runtime evidence because the authoring host has no GPU. Their guidance comes from
source, launchers and documentation, and was checked with parser and import tests. Treat them as
unverified until someone runs the blocked cases on a GPU host.
