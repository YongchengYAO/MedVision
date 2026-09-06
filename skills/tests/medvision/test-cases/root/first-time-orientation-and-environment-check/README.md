# Novice orientation and readiness check

## User Persona
A newcomer who knows medical imaging and VLMs but nothing about this benchmark.

## Scenario Coverage
- Skill area: root
- Capability: orientation, install, environment readiness, routing
- Difficulty: basic
- Prompt file: `user_request.txt`
- Expected references/scripts: `SKILL.md`, `references/concepts-and-glossary.md`,
  `scripts/check_medvision_env.py`
- Trigger expectation: names the benchmark and a natural first-use goal.

## Expected Successful Behavior
Describes the three quantitative task families and the units they are scored in, gives the install
commands for both packages plus the two required environment variables, and runs the bundled
environment checker to report whether this machine is ready, explicitly distinguishing what is
possible without a GPU from what needs one. Routes onward to the evaluation sub-skill and mentions
that the model must be supported by the harness.

## Failure Signals
Describing MedVision as a classification or report-generation benchmark; omitting the dataset package
or the required version pin; not offering a readiness check; dumping every sub-skill without routing.
