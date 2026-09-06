# Staleness audit — `medvision` refresh, 2026-09-04

Trigger: repository commit `980e9df` ("name prepared SFT datasets by their true train sizes and hand the path
from the prep run to training"). Provenance baseline: `a2c6482` (dirty; the `sft` sub-skill had been written
against a newer working tree that already carried the two-stage preparation and true-size naming, so most of
its dataset-construction claims were already current). `scripts/check_repo_provenance.py` reported
`status: stale` (commit drift plus a changed dirty-path set).

Repository evidence inspected: `src/medvision_bm/sft/sft_utils.py`, all ten `src/medvision_bm/sft/train__*.py`
entry points, the 21 `script/sft/train__*.sh` launchers, `README.md`, `git log a2c6482..HEAD`,
`git diff --stat a2c6482..HEAD -- src/medvision_bm/sft script/sft` (32 files, +1859/-1146).

## Claims

| Existing claim (file) | Current evidence | Decision | Action |
| --- | --- | --- | --- |
| Two-stage preparation: load+split per task, name the dir from true sizes, then format and save (`data-preparation.md` §1, §7; `workflows.md` §2) | entry points: `task_specs` loop → `_load_split_dataset_task` → name → `_format_dataset_task` → `save_to_disk` | retain | none |
| `prepare_dataset` = `load_split_limit_dataset` + `format_clean_dataset` (`data-preparation.md` §1, §4) | `sft_utils.py` | retain | none |
| `get_cgroup_memory_percent`, `save_to_disk` num_proc clamp (`data-preparation.md` §4; `troubleshooting.md` §1) | `sft_utils.py`; every entry point | retain | none |
| Non-main ranks receive the resolved dir by broadcast (`data-preparation.md` intro; `workflows.md` §2) | `broadcast_object_from_main` after `barrier()` in every entry point | retain | none |
| "`--skip_process_dataset true` still performs the load+split stage so the default name resolves" (`workflows.md` §2 note; `troubleshooting.md` §2) | true only when no `--prepared_ds_dir`; with an explicit dir the stage-1 loop is gated off (`if prepared_ds_dir is None or not skip_process_dataset`) | refresh | reworded in `workflows.md`, `troubleshooting.md`, `cli-reference.md`, `data-preparation.md` |
| "Both phases must resolve the same name, so phase A and phase B need identical limits and resize flags" (`SKILL.md` invariant 2; `data-preparation.md` §7; `troubleshooting.md` §2) | still true for a phase B without `--prepared_ds_dir`; the launchers now pass the phase-A path, which removes the requirement | refresh | invariant 2 rewritten; §7 rewritten; troubleshooting fix updated |
| "Phase B rebuilds nothing: needs `--skip_process_dataset true` and a prepared dataset on disk" (`SKILL.md` invariant 7) | plus: with `--prepared_ds_dir` it skips stage 1 entirely | refresh | invariant 7 extended |
| "It prints `Data processing completed...` — record that path" (`workflows.md` §2) | launchers tee phase A to `${lora_checkpoint_dir}/prepare_dataset.log`, `sed` the line, `exit 1` if missing, pass `--prepared_ds_dir` to phase B | add-replacement | `workflows.md` §2 rewritten; §4 command gains `--prepared_ds_dir`; §6 resume note |
| Launcher anatomy blocks 14-18 (`launcher-catalog.md`) | new `prep_log` definition, tee, capture block, `${prepared_ds_dir:+--prepared_ds_dir ...}` on phase A, `--prepared_ds_dir ${prepared_ds_dir}` on phase B, in all 21 launchers | refresh | blocks renumbered 14-19; shared-behaviour paragraph extended |
| `--skip_process_dataset` / `--prepared_ds_dir` rows (`cli-reference.md` §2) | gate semantics above | refresh | rows extended |
| Bundled template emits phase B without the phase-A path (`scripts/sft_launcher_template.sh`) | repository launchers now hand the path over | add-replacement | template tees phase A, captures the directory, appends `--prepared_ds_dir` to phase B, aborts on a missing report, warns on a phase-B-only run without it |
| `check_sample_limits.py` wording "replaces a 'full' token ... once the data is loaded" | correct in substance (true sizes for unset limits) | refresh (precision) | print text reworded |
| No troubleshooting entry for a slow phase B or for the launcher's abort | new failure modes introduced by the commit | add-support-workflow | two entries added to `troubleshooting.md` §2 |
| Launcher catalogue table rows (families, knobs, profiles) | launchers' variable blocks unchanged by the commit | retain | none |
| Trainer configuration, loss masking, resume, merge (`training-configuration.md`) | `prepare_trainer*`, collates, `train_resume_from_checkpoint`, `merge_models` unchanged since baseline | retain | none |
| RFT sub-skill: `--prepared_ds_dir` parsed but unused by the parquet builders | RFT builders untouched by the commit | retain | none |
| Routing: three area-family assignments | capability scope unchanged (an SFT-internal improvement) | retain | handoff `source_commit` refreshed; three evidence line ranges re-verified and adjusted for a 2-line README changelog insertion |
| Licence `NO_LICENSE` (resolver 404 at baseline) | GitHub licence endpoint now returns `NOASSERTION` for the repository | refresh | applied to root and all 10 sub-skills; report updated |

## Unknowns / accepted

- The bundled resolver (`resolve_repo_license.mjs`) cannot run on this host (Node 6, no ES-module support);
  the same GitHub endpoint was queried directly with the GitHub CLI. `NOASSERTION` is preserved per policy
  and is not a legal conclusion; the repository ships a CC-BY-4.0 `LICENSE` file.
- The other nine sub-skills were not re-audited claim by claim: `git log a2c6482..HEAD` touches only
  `src/medvision_bm/sft`, `script/sft`, visualization/doc files and a README changelog line.
