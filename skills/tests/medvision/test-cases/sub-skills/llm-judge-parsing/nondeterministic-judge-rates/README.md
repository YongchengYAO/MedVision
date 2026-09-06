# Judge non-determinism across machines

## User Persona
A researcher defending a methodological choice, needing an accurate account of what varies and
what cannot.

## Scenario Coverage
- Skill area: `llm-judge-parsing`
- Capability: reproducibility semantics and blast radius
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/design-notes.md`, `references/troubleshooting.md`
- Trigger expectation: a methodological dispute that requires the skill's reproducibility notes.

## Expected Successful Behavior
Explains that greedy decoding is not the same as reproducible: the variation comes from
numerics, because the inference kernels are not batch-invariant and prefix caching changes
batch composition, so the seed is irrelevant at temperature zero. States the operating rule
that judge-invalid rates are comparable only within one machine and checkpoint, and bounds the
blast radius: because the strict regex result takes unconditional precedence, a flip can only
cost or gain a recovered answer and can never rewrite a published value. Names the raw judge
output as the artifact of record and notes the downstream stages are deterministic. If strict
determinism is required, names the batch-invariance switch together with its costs.

## Failure Signals
Blaming sampling or the seed; claiming the pipeline is broken; asserting published numbers can
change; promising bit-exact reproducibility across machines; quoting flip rates measured on the
retired reader as if they applied to the current one.
