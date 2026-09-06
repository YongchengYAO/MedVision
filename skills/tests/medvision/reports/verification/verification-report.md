# Static verification report — `medvision`

Runtime skill: `skills/medvision`. All checks run by the main agent after whole-skill integration.

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Directory layout matches the intended structure | PASS | root `SKILL.md`, `references/`, `scripts/`, `sub-skills/<id>/{SKILL.md,references,scripts}`; 113 files |
| 2 | Every `SKILL.md` has valid YAML frontmatter | PASS | 11/11 parsed |
| 3 | Frontmatter `name` equals the directory basename | PASS | 11/11 |
| 4 | Every id is canonical lowercase-hyphen, ≤64 chars | PASS | 11/11 |
| 5 | Sub-skill ids do not repeat the root skill id | PASS | none contains "medvision" |
| 6 | Every `description` is double-quoted | PASS | 11/11 |
| 7 | Every file declares `disable-model-invocation: true` | PASS | 11/11 |
| 8 | Every file declares `metadata.disco-role: operating` | PASS | 11/11 |
| 9 | Exactly one single-line top-level `license` per file, consistent | PASS | bundled license validator: `valid: true`, 11 files, one value |
| 10 | License value recorded with resolver status | PASS | `NO_LICENSE`, status `unavailable`; see `../license-resolution.json` |
| 11 | Root `SKILL.md` routes rather than duplicating detail | PASS | 136 lines, routes to 10 sub-skills and 5 references |
| 12 | Root `SKILL.md` gives install and a verification command | PASS | install block plus import check plus the bundled environment checker |
| 13 | Root `SKILL.md` links repo provenance | PASS | cross-cutting references section |
| 14 | `references/repo-provenance.md` exists with the required schema and fields | PASS | schema `disco.repo-provenance.v1`; commit, branch, dirty paths, package versions, evidence paths |
| 15 | Provenance contains no local paths, prefixes or credentials | PASS | grep clean |
| 16 | `references/repo-routing-metadata.json` present and minimal-v2 | PASS | only the 6 allowed keys; assignments carry only area and family |
| 17 | Routing metadata matches the external handoff exactly | PASS | 3 assignments, identical sets; identity and taxonomy hash equal |
| 18 | Every sub-skill `SKILL.md` routes to nearby references and scripts | PASS | each links every file it owns |
| 19 | Every sub-skill has `references/troubleshooting.md` | PASS | 10/10 |
| 20 | All relative Markdown links resolve inside the skill tree | PASS | 202 links checked, 0 broken (55 repaired during integration) |
| 21 | No link escapes the skill directory | PASS | 0 escaping |
| 22 | No runtime instruction requires executing a repository path | PASS | reviewed; two checkout-only pipelines are marked reference-only with a placeholder path |
| 23 | Every named source artifact has a bundled replacement or a recorded reason | PASS | see `../integration/source-script-import-map.md` |
| 24 | Documented script options match the scripts' real parsers | PASS | every bundled script exercised with `--help`; option names taken from that output |
| 25 | Bundled scripts run from an arbitrary working directory | PASS | verified for the root checker and the sub-skill checkers |
| 26 | Diagnostics report missing dependencies instead of tracebacks | PASS | the judge and ablation checkers both degrade cleanly on this host |
| 27 | No local environment details in any runtime file | PASS | grep for machine paths, prefixes and the inspection environment: clean |
| 28 | No caches, build output, logs or symlinks in the tree | PASS | one stray build log and two harness directories were removed during integration |
| 29 | No check-only artifacts inside the runtime skill | PASS | all reports and cases live under the artifact root |
| 30 | Artifact root is organised into `test-cases/` and `reports/` | PASS | 24 cases; reports split into integration, verification, self-refine, final |
| 31 | Usability cases have the required three files each | PASS | 24/24 |
| 32 | Each sub-skill has one or two difficult synthetic cases | PASS | two each, plus two for the root |
| 33 | One or two integrated difficult cases exist | PASS | two, both adapted from repository-native end-to-end material |
| 34 | Native repository tests were run after integration | PASS | 27 passed, 6 failed, all six classified as pre-existing repository conditions |
| 35 | Required-backend work is blocked, not silently skipped | PASS | 5 `BLOCKED_REQUIRED_BACKEND` items recorded and carried into the final report |

## Warnings

- `NO_LICENSE` is a resolver fallback, not a legal conclusion. The repository ships a
  Creative Commons Attribution 4.0 licence file and the hosting platform reports a non-standard
  licence identifier for the repository, but the per-commit licence endpoint returned nothing usable,
  so the tooling recorded the fallback. A human may replace it after review.
- The skill was generated from a dirty checkout. The provenance file lists the modified paths so a
  later refresh can tell what moved.
