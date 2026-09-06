# End-to-end: evaluate, parse, summarize, judge

## User Persona
A team that owns a model and wants leaderboard-comparable numbers plus the judge-based robustness
check. Knows their model and the hardware, not this benchmark.

## Scenario Coverage
- Skill area: integration (root routing plus four sub-skills)
- Capability: the full four-step benchmark protocol with comparability constraints
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: root `SKILL.md` and `scripts/check_medvision_env.py`;
  `environment-setup`, `benchmark-evaluation`, `results-parsing-and-metrics`, `llm-judge-parsing`
- Trigger expectation: derived directly from the repository's own documented four-step pipeline,
  which is the closest thing it has to an end-to-end native example.

## Expected Successful Behavior
Sequences the four documented steps and names the owning sub-skill for each. Establishes
comparability up front: the published annotation version pinned with its acknowledgement variable,
the 1000-sample-per-subtask limit for an open-weight model, and the standard token budget. Checks the
environment before proposing GPU work, and warns that the model must already be supported by the
harness or else the extension workflow applies first. Gives a verification checkpoint per stage:
outputs present per task after evaluation, a parsed directory per model after parsing, the expected
summary files after summarizing, and the judge-suffixed twin reports afterwards. Explains that the
value of the judge pass is the difference between the two reports, and that the judge needs its own
environment with a newer serving stack than the evaluation code pins.

## Failure Signals
Collapsing the four steps into one command; omitting the annotation pin or the sample limit, which
breaks comparability; running the judge in the evaluation environment; skipping parsing and going
straight from evaluation to the judge; no verification checkpoints; not checking that the model is
supported.
