# Which pins the dataset install can still move

## User Persona
A user running the segmentation-specialist ablation who has just installed the dataset codebase into
the pinned ablation environment and wants to know what actually changed before hand-pinning.

## Scenario Coverage
- Skill area: `biomedparse-ablation`
- Capability: dependency-pin reasoning after `install_medvision_ds`
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/troubleshooting.md`, `references/setup.md`,
  `scripts/check_biomedparse_env.py`
- Trigger expectation: names the ablation setup plus a dependency symptom.

## Expected Successful Behavior
Discriminates rather than blanket-blames. The dataset install performs a plain wheel install (fills
only what is missing or outside the declared ranges) followed by a `--force-reinstall --no-deps`
refresh, so an in-range `huggingface-hub==0.36.0` survives. What can still move is `packaging`, via
the `pip install --upgrade build` that precedes the wheel build. `datasets==3.6.0` matches the
declared exact pin; the `opencv-python` versus `opencv-python-headless` split explains the expected
`pip check` noise. Repair is re-running the requirements file, which `setup.sh` already does twice,
verified with the bundled checker.

## Failure Signals
Claiming the installer force-reinstalls its dependencies or lifts the hub; recommending an unpinned
upgrade; telling the user to re-pin everything by hand without saying which pin actually moved; not
naming the bundled environment checker.

## Why This Case Exists
Added in the 2026-09-05 refresh. It requires the corrected installer semantics: an answer written
against the pre-fix behavior fails assertions 1 and 3.
