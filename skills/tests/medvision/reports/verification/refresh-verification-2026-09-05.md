# Refresh verification — `medvision`, 2026-09-05

Machine-readable results: `verification-report.json` (10 PASS, 1 accepted WARN, 0 FAIL).

## Commands run against this checkout

| Command | Result |
| --- | --- |
| `check_repo_provenance.py --skill-dir skills/medvision --repo-path .` | `stale` before the refresh (commit, dirty paths, missing `src/medvision_bm/sft/config`) |
| `list_task_yamls.py --repo-root .` | exit 0 — 22 datasets, 199 base YAMLs, 1253 task YAMLs, no duplicate task names, every `include:` resolves |
| `list_registered_models.py --repo-root .` | exit 0 — 20 registered keys, each with module, decorator and dispatch branch |
| `check_medvision_env.py --data-dir <data_dir> --skip-optional` | runs, reports NOT-READY on this CPU host (packages absent from the base interpreter) — expected |
| `gh api repos/YongchengYAO/MedVision/license?ref=780e247…` | `NOASSERTION` (unchanged) |
| `git diff --stat 980e9df..HEAD -- src/medvision_bm/sft script/sft` | empty — SFT surface untouched |
| `check_repo_provenance.py` (after the refresh) | `current`, exit 0 — commit, working tree and dirty paths all match |

## Baseline that can actually pass

The previous snapshot annotated its dirty paths (`"skills/ (untracked; …)"`). The checker compares
them literally against `git status --porcelain` output, so an annotated list can never match and the
snapshot reported `stale` even on the checkout it was written from. `dirty_paths` is now the literal
path list the spec asks for, with the explanation moved to prose, and the checker returns `current`.

## Static checks

Frontmatter valid across all 11 `SKILL.md` files (lowercase ids, quoted descriptions,
`disable-model-invocation: true`, `license: NOASSERTION`, `metadata.disco-role: operating`).
Provenance present, schema-valid, commit equals HEAD, linked from the root router. No host paths,
conda prefixes or interpreter paths in the runtime tree. No `__pycache__`/`*.pyc`. 26 usability
cases, each with a copyable prompt, README and parseable assertions; `index.md` count matches.

## Accepted warning

Ten paths under `sub-skills/biomedparse-ablation/` name launchers in the repository checkout
(`${ABLATION_DIR}/scripts/{eval,finetune}/…`) rather than bundled skill scripts. All ten were
verified present under `script/ablation/biomedparse/scripts/`. This is the reference-only design
recorded at creation: those drivers generate their own help text, so bundling thin wrappers would
drift. The references use the `${ABLATION_DIR}` placeholder, not a host path.

## Note on tooling

`resolve_repo_license.mjs` could not execute here — the only `node` on this host is v6.13.1, which
cannot parse ESM. The identical `gh api` query the resolver wraps was run directly and its result
recorded in `../license-resolution.json` in the resolver's own report shape. The value is unchanged,
so no frontmatter edit was required.
