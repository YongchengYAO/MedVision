# Derive coronal test configs from an SFT list, offline

## User Persona
A user preparing an out-of-distribution evaluation who knows the task families but not MedVision's
naming rules.

## Scenario Coverage
- Skill area: `dataset-and-tasks`
- Capability: task-name to dataset-config derivation, planes, CoT suffix, offline work
- Difficulty: intermediate
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/task-lists.md`, `scripts/list_tasks.py`
- Trigger expectation: names MedVision task lists and planes.

## Expected Successful Behavior
Uses the bundled task-list helper to rewrite the plane and emit test-split config names, explains
that evaluation task names carry the CoT suffix while dataset configs never do, notes the
detection-only BoxCoordinate-to-BoxSize rewrite does not apply to T/L, points at the shipped
plane-OOD list as the authoritative roster, and downloads nothing.

## Failure Signals
Producing config names that keep the CoT suffix; asking the user to download data; hand-editing JSON;
confusing the SFT and evaluation naming conventions.
