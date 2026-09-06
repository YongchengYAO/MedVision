# Staleness audit — `medvision`, 2026-09-05

Baseline: provenance snapshot at `980e9df` (dirty tree).
Current: `780e247d393bafba6ac71bb9707b0e0e4506d2e7`, branch `master`, 11 commits past `v1.2.0`.
First signal: `check_repo_provenance.py` → **stale** (commit drift, dirty-path drift, one evidence
path gone).

## Change surface

Exactly one commit separates baseline from current, and its 16 files are precisely the tracked files
the old snapshot listed as *dirty*. In other words the skill was authored against a working tree
whose content is now committed. That makes most of the drift a stamp problem rather than a content
problem — but not all of it.

`git diff --stat 980e9df..HEAD -- src/medvision_bm/sft script/sft` is empty, so the fine-tuning
surface refreshed on 2026-09-04 is untouched and its guidance carries forward unchanged.

## Claim-by-claim

| Existing claim | Current evidence | Decision | Action | Verification |
| --- | --- | --- | --- | --- |
| Provenance snapshot `980e9df`, dirty tree with 14 tracked files | HEAD is `780e247`; tracked tree clean apart from a one-line README edit | repo-drift | rewrote `references/repo-provenance.md` | commit compared to `git rev-parse HEAD` |
| `medvision_bm` ships `sft/config/*.yaml` as package data (`environment-setup/references/installation.md`) | `780e247` removed that entry; `src/medvision_bm/sft/config` does not exist | stale | removed the claim, said where training config actually lives | `git diff` of `pyproject.toml`; directory absent |
| Provenance evidence path `src/medvision_bm/sft/config` | same | stale | dropped from the evidence list | provenance checker no longer reports a missing path |
| "the dataset-package installer force-reinstalls its dependencies and lifts `huggingface-hub`/`packaging`" (`biomedparse-ablation/references/troubleshooting.md`) | `install_medvision_ds` is a plain install followed by `--force-reinstall --no-deps`; `medvision_ds` declares `huggingface_hub>=0.35.3,<2.0`, which the ablation's pinned 0.36.0 satisfies | stale | replaced with the two mechanisms that remain: `pip install --upgrade build` can lift `packaging`; the plain install fills deps that are missing or out of range (`opencv-python`, exact `datasets==3.6.0`) | `install_utils.py` L280-315; `Data/src/pyproject.toml`; `script/ablation/biomedparse/requirements.txt` |
| Same claim, restated as setup step 7 (`biomedparse-ablation/references/setup.md`) | same | stale | rewritten; re-running the requirements file is still the repair | as above |
| "`install_medvision_ds` or the loader's in-tree `pip install .` re-resolved hub (0.36.0 → 1.x)" (`environment-setup/references/troubleshooting.md`) | neither path re-resolves a satisfied dependency: the wheel install is plain + `--no-deps`, and `pip_install_medvision_ds()` is a bare `pip install git+…` | stale | cause rewritten around the real trigger — a hub **missing or below** the 0.35.3 floor is resolved up to the newest in-range release | `install_utils.py` L323-336; declared range in `Data/src/pyproject.toml` |
| "plain install step only fills *missing* deps" (same file, second row) | pip also replaces deps that are installed but outside the declared range | refresh | reworded to "missing or outside its declared ranges" | pip resolution semantics + declared ranges |
| Two-step install description (`installation.md`) | matches current code exactly | retain (sharpened) | added the below-floor case, which is the trap that still bites | `install_utils.py` comment and command string |
| Eight defective task YAMLs (duplicate names, `dataset_path` overrides) | fixed by `780e247` | retain | no runtime claim asserted the defects; bundled checker re-run | `list_task_yamls.py` exits 0: 22 datasets, 199 base, 1253 task YAMLs, no duplicates, every include resolves |
| Model roster and `AVAILABLE_MODELS` coverage | live registry has 20 active keys (5 commented out); roster names `kimi`, `vllm_minimax_m3`, `vllm_glm4v`, collapsed `healthgpt`, `vllm_qwen3vl` | retain | none | `list_registered_models.py` exits 0, "no mismatches" |
| "21 `eval__*` entry points, 24 launcher stems per family" | 21 and 24/24/24 | retain | none | directory counts |
| Judge tests run as `unit-test/llm-parsing/test-{1..7,9,11}.py` | files present (test-10 absent, as documented) | retain | none | directory listing |
| `mvbm install mvds -d <data_dir>` == `python -m medvision_bm.benchmark.install_medvision_ds --data_dir …` | both entry points exist; the uncommitted README edit prefers the `mvbm` form the skill already leads with | retain | recorded the README edit as a dirty path | `cli.py` L33; module file present |
| SFT sub-skill (prepared-dataset hand-off, two-phase launchers) | source untouched since baseline | retain | none | empty diff over `src/medvision_bm/sft` and `script/sft` |

## Unknowns and accepted risk

- The dataset codebase's own reinstall under `MedVision_FORCE_INSTALL_CODE=true` lives in the Hugging
  Face dataset repository, not in this checkout, so its exact pip invocation was not read. The
  refreshed text describes it by name and does not attribute force-reinstall behaviour to it.
- The four GPU workflows (local evaluation, fine-tuning, judge inference, ablation tracks) still have
  no runtime evidence on this host. Unchanged from the previous build.
