# Detection thresholds vs. mean IoU

## User Persona
A researcher comparing models on the detection leaderboard. Comfortable with detection metrics in
general, but has not read MedVision's failure-handling conventions.

## Scenario Coverage
- Skill area: `results-parsing-and-metrics`
- Capability: metric semantics - denominators and failure handling
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `sub-skills/results-parsing-and-metrics/references/metrics.md`,
  `scripts/metrics_demo.py`
- Trigger expectation: names MedVision summary fields (`IoU>0.5`) that only this benchmark produces.

## Expected Successful Behavior
Explains that the two numbers use different denominators and different failure handling: unparseable
responses count as IoU 0 in the mean (so the reported mean is the successful-sample mean times the
success rate), while `IoU>0.5` divides the count of samples at or above the threshold by the total
sample count. Notes the comparison operator is `>=`, that `Acc@IoU>=0.50` is the same quantity, and
that A/D and T/L behave differently (their MAE/MRE are NaN on failure and excluded). Points at the
bundled demo that proves the semantics against the installed package, and suggests checking
SuccessRate to quantify how much of the gap is formatting rather than accuracy.

## Failure Signals
Calling it a bug; inventing a formula; claiming failures are excluded from detection means; not
mentioning the total-count denominator; telling the user to open repository source files.
