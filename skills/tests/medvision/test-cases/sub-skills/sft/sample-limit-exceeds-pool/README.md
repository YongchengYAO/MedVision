# Sample limits larger than the available pool

## User Persona
A user reproducing a published recipe who does not know the limit resolution rules.

## Scenario Coverage
- Skill area: `sft`
- Capability: sample-limit semantics and validation splitting
- Difficulty: intermediate
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/data-preparation.md`, `scripts/check_sample_limits.py`
- Trigger expectation: a plausible misconfiguration worry that the skill must resolve with the actual semantics.

## Expected Successful Behavior
Explains the resolution order: validation is carved out first, per task and grouped so that slices from one source volume do not straddle the split; per-task limits above the pool are no-ops; and the global cap is applied after concatenation, sampling with replacement when it exceeds the pool, so the number is a draw count rather than a count of distinct samples. Warns that a global cap below the sum of per-task limits truncates silently, and points at the bundled limit checker to show the resolved numbers before a run.

## Failure Signals
Calling it a misconfiguration without explaining bootstrap sampling; claiming the extra samples are distinct; omitting the validation carve-out; not mentioning the silent-truncation risk.
