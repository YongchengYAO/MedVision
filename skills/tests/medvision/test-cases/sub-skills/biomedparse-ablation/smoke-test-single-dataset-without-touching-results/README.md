# Scoped smoke test that leaves shipped artifacts alone

## User Persona
A cautious reproducer validating the ablation pipeline before a long run.

## Scenario Coverage
- Skill area: `biomedparse-ablation`
- Capability: the smoke-test path and output isolation
- Difficulty: intermediate
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/tracks.md`, `references/setup.md`,
  `scripts/check_biomedparse_env.py`
- Trigger expectation: names the segmentation-specialist comparison and a MedVision dataset.

## Expected Successful Behavior
Uses the evaluation-track smoke test scoped to the single dataset, states that its outputs are
written under a separate smoke-test directory rather than the tracked result and figure trees, and
lists the three stages it exercises. Checks the environment first because the pipeline needs the
pinned upstream checkout, a compiled detection dependency and a CUDA GPU, and warns that smoke-test
metrics are not meaningful because the pool is tiny.

## Failure Signals
Running the full track; claiming results are overwritten or being vague about output location;
skipping the environment prerequisites; presenting smoke-test numbers as comparable.
