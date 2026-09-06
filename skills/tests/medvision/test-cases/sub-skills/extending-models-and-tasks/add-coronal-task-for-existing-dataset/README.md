# Add a plane variant of an existing task

## User Persona
A maintainer extending an existing dataset to another slice plane.

## Scenario Coverage
- Skill area: `extending-models-and-tasks`
- Capability: task YAML authoring and registration
- Difficulty: intermediate
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/add-a-task.md`, `scripts/list_task_yamls.py`
- Trigger expectation: a scoped maintainer request that must not turn into a larger change.

## Expected Successful Behavior
Creates only the small per-task YAML, reusing the existing base file through the include directive rather than adding a new base. Sets the task name with the coronal plane and the chain-of-thought suffix, and the dataset config name with the coronal plane and the test split but no suffix. Registers the task name in the relevant task list, without which it never runs, and verifies with the bundled inventory tool that the new task shows as referenced.

## Failure Signals
Creating a new base YAML; putting the chain-of-thought suffix in the dataset config name; forgetting task-list registration; editing shared utilities unnecessarily.
