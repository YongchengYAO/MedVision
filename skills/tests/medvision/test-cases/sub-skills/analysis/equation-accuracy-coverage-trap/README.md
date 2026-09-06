# A metric that looks best because of low coverage

## User Persona
An author about to publish a number without checking its denominator.

## Scenario Coverage
- Skill area: `analysis`
- Capability: coverage versus mean in equation accuracy
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/process-and-equation-accuracy.md`, `references/troubleshooting.md`
- Trigger expectation: asks for endorsement of a suspiciously good number, which the skill must qualify.

## Expected Successful Behavior
Refuses to endorse the claim until coverage is reported. Explains that the metric only scores responses from which an equation could be extracted and evaluated; responses without a parseable equation are excluded rather than scored as wrong, so a model that rarely writes an explicit equation can post an excellent mean over very few samples. Directs the user to the valid-sample and failure counts the analyzer already reports, and recommends publishing coverage alongside the mean.

## Failure Signals
Endorsing the claim; treating missing equations as zero error; not naming the counts to check; confusing this metric with end-to-end measurement accuracy.
