# Re-score one dataset and merge into existing results

## User Persona
A maintainer repairing a single dataset's rows in a finished ablation.

## Scenario Coverage
- Skill area: `biomedparse-ablation`
- Capability: targeted re-runs, checkpoint selection, result merging
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/tracks.md`, `references/cli-reference.md`,
  `references/troubleshooting.md`
- Trigger expectation: a maintenance scenario specific to this ablation's tooling.

## Expected Successful Behavior
Uses the dataset filter supported by the data-preparation, inference and evaluation stages, switches
the task to tumour/lesion, and explicitly overrides the checkpoint because the default is the last
one while the published run used the best-validation checkpoint. Explains that the evaluator merges
refreshed rows back into the existing result files, and notes the tumour/lesion path additionally
needs the annotation acknowledgement variable.

## Failure Signals
Re-running all datasets; leaving the checkpoint at its default; not mentioning the merge behaviour;
omitting the task switch or the acknowledgement variable.
