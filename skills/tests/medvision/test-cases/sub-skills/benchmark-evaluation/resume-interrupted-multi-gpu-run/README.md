# Resume an interrupted evaluation

## User Persona
An experienced user restarting a long multi-GPU evaluation, unsure how the caching interacts with an edit.

## Scenario Coverage
- Skill area: `benchmark-evaluation`
- Capability: resume semantics and cache invalidation
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/workflows.md`, `references/troubleshooting.md`, `scripts/check_results_tree.py`
- Trigger expectation: describes a specific interrupted run and a prompt edit, without naming any file.

## Expected Successful Behavior
Re-runs the identical command and explains the two independent skip levels: the completed-tasks tracker skips whole finished tasks, and the per-sample response cache skips finished samples within a task. States that the cache key includes a hash of the prompt, so the edited prompt invalidates exactly the affected entries automatically, while a token-budget or sampling change would not and would need the cache cleared. Offers the bundled results-tree checker to see which tasks are incomplete.

## Failure Signals
Suggesting deletion of the whole results tree; hand-editing result files; claiming the prompt edit is ignored; confusing the task tracker with the sample cache.
