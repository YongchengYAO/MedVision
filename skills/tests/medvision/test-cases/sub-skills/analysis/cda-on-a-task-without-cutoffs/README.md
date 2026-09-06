# Clinical decision agreement outside its configured scope

## User Persona
A researcher assuming the clinical analysis applies to every task family.

## Scenario Coverage
- Skill area: `analysis`
- Capability: scope of the clinical decision analysis
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/cda.md`, `references/troubleshooting.md`, `scripts/run_cda.sh`
- Trigger expectation: asks for a real analysis on a task family it does not cover, with an empty-result symptom.

## Expected Successful Behavior
Explains that the analysis maps a measurement onto a clinical category using a published cutoff table, so it only applies to task families for which such a table is configured; a task with no configured proxy produces an empty report rather than an error. Says what is configured today, and that extending it means adding a published cutoff table with its boundary direction to the configuration, which is a clinical decision needing a citation rather than a code change. Suggests the appropriate detection analysis instead.

## Failure Signals
Inventing cutoffs; treating the empty report as a bug; claiming detection is supported; adding a threshold without a published source.
