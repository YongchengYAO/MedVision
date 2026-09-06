# Extend a finished judge campaign by one model

## User Persona
An experienced user who has run the judge pipeline before and is now extending it under a
time budget.

## Scenario Coverage
- Skill area: `llm-judge-parsing`
- Capability: incremental campaign growth, resumability, roster management
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/recipes.md`, `references/pipeline.md`,
  `scripts/make_roster_yaml.py`
- Trigger expectation: describes the judge campaign and a cost constraint without naming files.

## Expected Successful Behavior
Adds the model to the roster keyed by its results directory name, confirms that directory
already holds strict parsed records, runs only the queue-build, judge and analyze steps, and
explicitly does not run the destructive preparation step that would archive the finished sweep.
Explains that work already done is skipped because the queue identity does not depend on the
roster, that the existing models' outputs are rewritten but unchanged, and that a prompt or
budget edit would instead invalidate every queue.

## Failure Signals
Running the destructive prep step; re-judging the whole roster; using a display name as the
roster key; omitting the requirement that the new model already has parsed records; claiming
the previously judged numbers change.
