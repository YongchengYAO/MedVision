# Re-summarize from the judge directory

## User Persona
An experienced user who has already run the judge pipeline and now wants the format-robust twin of
one report, scoped to two models.

## Scenario Coverage
- Skill area: `results-parsing-and-metrics`
- Capability: re-summarizing an alternative parsed directory
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/workflows.md`, `references/cli-reference.md`,
  `references/output-files.md`, `scripts/parse_and_summarize.sh`
- Trigger expectation: names MedVision-specific artifacts (judge pass, T/L results tree).

## Expected Successful Behavior
Runs only the summarize step (no re-parse), with `--parsed_dirname` pointing at the judge directory,
`--resps_key` switched to the judge record key, `--models` limiting the roster,
`--removed_samples_dir` supplying the T/L ambiguity filter, and `--skip_model_wo_parsed_files`.
Names the qualified output files that appear alongside the published ones, and explains that the
diff between the two reports is the share of apparent failure that was formatting.

## Failure Signals
Re-running `parse_outputs` over the judge directory; forgetting the response key so the summarizer
aborts; overwriting the published summary; applying the removed-samples filter to A/D or Detection;
inventing flag names.
