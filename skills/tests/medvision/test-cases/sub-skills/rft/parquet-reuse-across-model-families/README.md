# Reusing a parquet dataset across model families

## User Persona
An experienced RL user trying to save preparation time, unaware of the pixel-size coupling.

## Scenario Coverage
- Skill area: `rft`
- Capability: the model-family constraint on prepared datasets
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/parquet-schema.md`, `references/workflows.md`,
  `scripts/build_parquet_ds.sh`
- Trigger expectation: a plausible shortcut that the skill must refuse with a concrete reason.

## Expected Successful Behavior
Refuses the reuse and explains why: prompts embed the image size and pixel size as the model's own
image processor will perceive them, so a parquet built for one family states the wrong scale for
another, silently corrupting every measurement and its reward. Says the dataset must be rebuilt with
the other family's key and its Hugging Face id for the processor, and notes reuse is only valid
between models that share an image processor.

## Failure Signals
Approving the reuse; treating it as a tokenizer-only concern; failing to name the pixel-size prompt
coupling; not saying which fields must change on a rebuild.
