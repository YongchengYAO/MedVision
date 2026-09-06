# Registered model missing its image-size dispatch branch

## User Persona
A contributor who completed part of the model-integration checklist.

## Scenario Coverage
- Skill area: `extending-models-and-tasks`
- Capability: model registration versus image-size dispatch
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/add-a-model.md`, `references/image-size-dispatch.md`, `scripts/list_registered_models.py`
- Trigger expectation: an exact error message from a partially completed integration.

## Expected Successful Behavior
Identifies this as a missing branch in the image-size dispatch rather than a registry problem, and explains the asymmetry: the measurement tasks put a physical pixel size in the prompt and therefore must know the model's perceived resolution, while detection asks for relative coordinates. Uses the bundled consistency checker to show the registry and dispatch disagreeing, adds a probe branch plus the training-side family alias, refuses to add a catch-all default, and validates on a non-square slice by comparing the reported resized shape with the size stated in the prompt.

## Failure Signals
Treating it as a registry problem; adding a default branch; validating only on a square image; omitting the training-side alias.
