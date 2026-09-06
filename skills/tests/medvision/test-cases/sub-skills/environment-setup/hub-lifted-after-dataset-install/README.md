# Dependency pin lifted by the dataset installer

## User Persona
A user setting up an evaluation environment for the first time, hitting the most common pin trap.

## Scenario Coverage
- Skill area: `environment-setup`
- Capability: version-pin diagnosis and repair
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/troubleshooting.md`,
  `references/requirements-catalog.md`, `scripts/check_env_pins.py`
- Trigger expectation: exact error text from a documented failure mode.

## Expected Successful Behavior
Identifies the cause as an incompatible hub/transformers pair (transformers 4.x requires
huggingface_hub below 1.0), and explains the precise mechanism: `medvision_ds` declares
`huggingface_hub>=0.35.3,<2.0`, the install only fills dependencies that are missing or outside that
range, and the env's 0.34.4 was below the floor — so pip resolved it up to the newest in-range
release. Repairs it by reinstalling the version the model's frozen requirements file specifies. Uses the bundled pin checker to show installed versus pinned versions
before and after, and warns that installation order matters because the frozen requirements must win.

## Failure Signals
Suggesting an unpinned upgrade of transformers or hub; blaming the model; not naming the frozen
requirements file as the source of truth; not offering a diagnostic that lists installed versus
pinned versions; asserting that the dataset install force-reinstalls or re-resolves already-satisfied
dependencies (it does not — a plain install then a `--no-deps --force-reinstall`).
