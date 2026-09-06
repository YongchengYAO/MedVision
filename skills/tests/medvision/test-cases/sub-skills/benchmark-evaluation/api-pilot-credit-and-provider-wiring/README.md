# API model via OpenRouter with a credit-hold failure

## User Persona
A user extending the pilot study to an API model, hitting provider-specific behaviour.

## Scenario Coverage
- Skill area: `benchmark-evaluation`
- Capability: API provider wiring, sample-limit conventions, cost failures
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/model-catalog.md`, `references/troubleshooting.md`, `scripts/make_eval_launcher.py`
- Trigger expectation: names an API model, a provider and a concrete error code.

## Expected Successful Behavior
Pushes back on 1000 samples: the published convention for API models is a 100-sample pilot, and cost scales with samples times the token budget, so comparing a 1000-sample API run against the open-weight tables is both expensive and not the published protocol. Wires the provider correctly, requiring the provider selection, the provider-prefixed model code and the matching key variable together. Diagnoses the 402 as the provider reserving the full requested token budget as credit rather than the account being empty. Sanitizes the key and notes that changing the sample limit means a distinct results tag.

## Failure Signals
Accepting 1000 samples without comment; setting the provider but not the prefixed model code or key; reading 402 as an empty account; omitting the key sanitising step.
