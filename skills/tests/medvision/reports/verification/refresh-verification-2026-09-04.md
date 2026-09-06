# Refresh verification report — `medvision`, 2026-09-04

Scope: the `refresh-repo-skill` pass triggered by repository commit `980e9df`. Runtime skill:
`skills/medvision` (external working copy inside the repository; not the managed DisCo copy). Claim
audit: `staleness-audit-2026-09-04.md` in this directory.

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Provenance drift detected before editing | PASS | `check_repo_provenance.py`: `status: stale` (commit `a2c6482` -> `980e9df`; dirty-path set changed) |
| 2 | `references/repo-provenance.md` rewritten for the current checkout | PASS | schema `disco.repo-provenance.v1`; commit `980e9dfb…`, branch `master`, `v1.2.0-10-g980e9df`, 23 dirty paths (repo-relative), packages and evidence paths retained; new sft-specific refresh trigger |
| 3 | Root `SKILL.md` still links the provenance file | PASS | cross-cutting references section, unchanged |
| 4 | Every `SKILL.md` frontmatter parses; `disable-model-invocation: true`; `metadata.disco-role: operating` | PASS | 11/11 (YAML parse) |
| 5 | One licence value across the tree, re-queried for the refresh commit | PASS | `NOASSERTION` in 11/11 files; `../license-resolution.json` (GitHub CLI, status `resolved`; the bundled `.mjs` resolver cannot run on this host's Node 6) |
| 6 | Stale claims replaced (phase-B naming requirement, "record that path", stage-1 always runs) | PASS | `sft/SKILL.md` invariants 2 and 7; `data-preparation.md` §1, §7; `workflows.md` §2, §4, §6; `cli-reference.md` §2; `launcher-catalog.md` shared block + anatomy 14-19; `troubleshooting.md` §2 (+2 entries) |
| 7 | Refreshed claims match current source | PASS | gate `if prepared_ds_dir is None or not kwargs.get("skip_process_dataset")` in 10/10 entry points; `[Info] Using user-specified prepared dataset directory` in 10/10; `[Info] Starting dataset preparation from` at `sft_utils.py:2128`; `prepare_dataset.log` tee + `sed` capture + `--prepared_ds_dir` in 21/21 `script/sft/train__*.sh` |
| 8 | Bundled template mirrors the launchers' hand-off | PASS | `bash -n` clean; `--help` exits 0; `DRY_RUN=1` phase AB prints the tee note and phase B carries `--prepared_ds_dir "<dir reported by phase A>"`; `DRY_RUN=1 --phase B` without a dir prints the load+split warning |
| 9 | Template capture path exercised end to end | PASS | stub entry-point package on `PYTHONPATH`: reporting stub -> directory captured, `prepare_dataset.log` written, exit 0; non-reporting stub -> `[error] could not read the prepared dataset directory …`, exit 1 |
| 10 | Edited bundled script still runs against the real parser | PASS | `check_sample_limits.py` executed with `medvision_bm` importable; new wording printed; limits resolved through `parse_sample_limits` |
| 11 | Documented script options match the scripts | PASS | `--prepared-ds-dir` help text updated; no option renamed |
| 12 | All relative links in the `sft` sub-skill resolve | PASS | 0 broken |
| 13 | No local checkout paths, environment names, or credentials in runtime files | PASS | grep for the host's mount, home, environment and job identifiers: 0 hits (the only `miniconda` hit is the Docker image's `/opt/miniconda` in `environment-setup/references/installation.md`, a container path from the Dockerfiles, pre-existing) |
| 14 | No caches, backups or scratch files in the runtime tree | PASS | 0 `__pycache__` / `*.pyc` / `*.bak` |
| 15 | Usability cases: one refreshed-behaviour case, one retained regression case, `index.md` updated | PASS | new `sub-skills/sft/phase-b-stalls-in-dataset-loading` (3 files); `sub-skills/sft/sample-limit-exceeds-pool` retained as regression guard; index now lists 25 cases |
| 16 | Routing handoff refreshed without reclassification | PASS | `skills/disco/routing_decision/classification.json`: `source_commit` -> `980e9dfb…`; README evidence ranges shifted +2 for a changelog line; registry range tightened to 19-65; three assignments retained (capability scope unchanged) |
| 17 | Runtime routing metadata unchanged and consistent with the handoff | PASS | `references/repo-routing-metadata.json` untouched; same three area-family pairs |
| 18 | Review artifacts kept outside the runtime tree | PASS | this report, the audit, the licence report and the routing note live under `skills/tests/medvision/reports/` |

## Warnings and accepted risks

- `NOASSERTION` is GitHub's statement that it did not auto-detect the licence; the repository ships a
  Creative Commons Attribution 4.0 `LICENSE` file. The value is preserved per policy and is not a legal
  conclusion. Replace it manually if the skill should state CC-BY-4.0.
- `skill_content_sha256` in the routing handoff is still the zero placeholder from creation. It must be
  recomputed immediately before any import (`reports/routing/compute_skill_digest.py`).
- The nine other sub-skills were not re-audited claim by claim; the commits since the baseline touch only
  the SFT surface, visualization/doc files and one README changelog line.
- GPU-native verification of the SFT workflow remains `BLOCKED_REQUIRED_BACKEND` on this host, as at
  creation.
