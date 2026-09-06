# Common brief for every MedVision sub-skill drafting subagent

You are drafting ONE sub-skill of the generated `medvision` repo skill (DisCo / AREX-Skill "Agent Skills" format).
Read this file fully, then your sub-skill-specific brief. Write files DIRECTLY to disk; return only a review handoff.

## Where things are (private research context — NEVER cite these paths inside the generated skill files)
- Source repository checkout (evidence): `/mnt/vincent-pvc-rwm/Github/MedVision` (commit a2c6482, branch master, dirty).
- Dataset package source (evidence): `/mnt/vincent-pvc-rwm/MedVision/src/medvision_ds` (v1.4.0).
- Inspection Python (use it to VERIFY every API/CLI claim; run `--help`, `inspect.signature`, imports):
  `/mnt/vincent-pvc-rwm/.disco/agent/envs/medvision-inspection/bin/python`
  (medvision_bm installed editable; medvision_ds installed; torch 2.6.0 CPU-only host — NO GPU; no vllm/verl/transformers-heavy stacks).
  Run it directly by absolute path; do NOT `conda activate`; do NOT pip install anything; never run GPU jobs, downloads, training, or anything that writes into the repo checkout, `Data/`, `Results/`, `SFT/`.
- Generated skill root: `/mnt/vincent-pvc-rwm/Github/MedVision/skills/medvision/` — you own ONLY `sub-skills/<your-id>/` beneath it.
- Review/test artifact root (do NOT write there; the main agent owns it): `/mnt/vincent-pvc-rwm/Github/MedVision/skills/tests/medvision/`.
- Structure plan + evidence map (read for boundaries): `skills/tests/medvision/reports/integration/sub-skill-structure-plan.md`, `repository-evidence-map.md`.
- Existing library example to imitate for shape/tone: `/mnt/vincent-pvc-rwm/.disco/agent/skills/repositories/repo-skills/monai/` (root + sub-skills).

## Output contract for `sub-skills/<id>/`
```
sub-skills/<id>/
  SKILL.md                 # 80-200 lines, router-like, valid YAML frontmatter (below)
  references/*.md          # workflows.md / cli-reference.md / api-reference.md / configuration.md / data-formats.md /
                           # troubleshooting.md (REQUIRED) ... distilled, task-oriented, verified facts
  scripts/*                # bundled helpers (copy/adapt/wrap repo scripts); each has a docstring/header with purpose,
                           # prerequisites, example invocation; argparse with --help; safe defaults; runnable from any cwd;
                           # accept --repo-root / explicit paths instead of assuming the checkout; catch ImportError and
                           # report the missing package/extra clearly; exit non-zero on genuine validation failure
```
Frontmatter (exact keys; description double-quoted, third person, trigger-rich):
```yaml
---
name: <id>
description: "..."
disable-model-invocation: true
license: NO_LICENSE
metadata:
  disco-role: operating
---
```

## Hard rules (verification will reject violations)
1. SELF-CONTAINED: a future agent must use the sub-skill after the checkout is gone. No Markdown links or "run/see `script/...`", `docs/...`, `unit-test/...`, absolute checkout paths, or `../..` links that leave `skills/medvision/`. Source repo artifacts may be NAMED as provenance ("the repository's `script/benchmark-*/` launchers") but every runnable instruction must point at a bundled `scripts/` file or a public command (`python -m medvision_bm.…`, `mvbm …`, `pip install …`).
2. PRIVACY: never mention the inspection env path, conda prefixes, `/mnt/...`, `/root/Documents/MedVision`, `/home/vincent`, pod names, tokens. Use placeholders like `<repo>`, `<data_dir>`, `${benchmark_dir}`.
3. EVIDENCE: every flag, default, function name, file name, env var, output path must be verified against source or the inspection env (`python -m <module> --help`). Do not invent flags. If something cannot be verified on CPU (vllm-only imports), say "verified from source" and cite the module name.
4. Repo terminology: Detection / Tumor-Lesion size (T/L) / Angle-Distance (A/D); task names vs dataset configs; `MedVision-V0`; `medvision_bm` / `medvision_ds`; `lmms_eval` (vendored fork); `verl` fork branch `medvision-rl`.
5. Links inside `SKILL.md` must resolve to files you actually create. Every reference/script is linked from your `SKILL.md` with one sentence saying when to read/run it. Cross-links to siblings use relative paths like `../results-parsing-and-metrics/SKILL.md` and to root files `../../references/<file>.md` (root files that WILL exist: `references/troubleshooting.md`, `references/concepts-and-glossary.md`, `references/model-roster.md`, `references/visualization-catalog.md`, `references/repo-provenance.md`, `scripts/check_medvision_env.py`, `scripts/list_tasks.py`).
6. `references/troubleshooting.md` is REQUIRED: symptom / error fragment → likely cause → concrete fix / validation → when to stop (needs GPU, credentials, network, large data).
7. Safe operating guidance: GPU workflows are documented with exact commands but flagged "requires GPU"; never instruct to run private launchers; never suggest `pip install` that mutates a user env without a warning about pins.
8. No `__pycache__`, logs, caches, test cases, review notes inside the skill tree.
9. Run at least a `--help`/parser or tiny-fixture check for every script you bundle, using the inspection Python; fix until it passes.
10. Keep `SKILL.md` router-like: workflows/depth/tables/long examples go into `references/`.

## Return (handoff ONLY — no file bodies)
- Files created (paths relative to `sub-skills/<id>/`) with one-line purpose each.
- Evidence consulted (repo-relative paths) and inspection commands run (with pass/fail).
- Source scripts: copied / adapted / wrapped / reference-only / excluded, with reasons.
- Troubleshooting failure modes covered.
- Native test/example candidates relevant to this sub-skill (repo-relative, with safety class).
- 1-2 proposed difficult synthetic usability cases (user request + expected assertions) and why repo tests are insufficient.
- Known gaps, uncertainties, questions for the main agent.
