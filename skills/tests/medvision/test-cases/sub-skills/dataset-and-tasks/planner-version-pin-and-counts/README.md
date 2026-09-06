# Annotation version pin and sample counts

## User Persona
A researcher starting a T/L study who has met the loader's version guard for the first time.

## Scenario Coverage
- Skill area: `dataset-and-tasks`
- Capability: annotation versions, the planner pin, acknowledgement, and the version ceiling
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/concepts.md`, `references/downloading.md`,
  `references/troubleshooting.md`, `scripts/inspect_benchmark_plan.py`
- Trigger expectation: names the MedVision planner environment variable explicitly.

## Expected Successful Behavior
Explains that the pin is required, that it acts as a per-dataset ceiling (a dataset without a plan at
the requested version resolves to its newest plan at or below it), and that pinning below the newest
release additionally requires the acknowledgement variable. States that leaderboard numbers use the
v1.0.0 annotations, that only T/L annotations change between versions, and therefore that the sample
count legitimately differs. Offers the bundled offline plan inspector to show which plan version a
dataset would actually resolve to, without downloading anything.

## Failure Signals
Suggesting the pin is optional; omitting the acknowledgement variable; claiming Detection or A/D
counts change with the version; proposing a download to answer the question.
