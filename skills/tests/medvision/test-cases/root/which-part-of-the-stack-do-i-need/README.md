# Routing a two-part goal across the stack

## User Persona
A team lead planning a project, needing the map rather than one command.

## Scenario Coverage
- Skill area: root
- Capability: routing across data, training, evaluation and reporting
- Difficulty: intermediate
- Prompt file: `user_request.txt`
- Expected references/scripts: `SKILL.md`, `references/concepts-and-glossary.md`,
  `references/model-roster.md`
- Trigger expectation: a goal that spans several sub-skills without naming any of them.

## Expected Successful Behavior
Maps the request onto the tumour/lesion task family, then lays out the order: prepare data and task
lists, fine-tune, evaluate both models under the same protocol, parse and summarize, and optionally
run the judge pass for a format-robust comparison. Names the owning sub-skill for each step, mentions
that the baseline must be in the supported roster, and flags the fairness constraints that make the
comparison valid, such as an identical annotation version and sample limit.

## Failure Signals
Answering with one sub-skill only; inventing a workflow that does not exist; skipping the
parse/summarize step; ignoring the fairness constraints.
