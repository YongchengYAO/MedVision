# Full-parameter fine-tuning out of memory

## User Persona
A practitioner who has tried the obvious memory knobs and needs the sharded-training specifics.

## Scenario Coverage
- Skill area: `sft`
- Capability: memory configuration for full-parameter training
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/training-configuration.md`, `references/troubleshooting.md`, `references/launcher-catalog.md`
- Trigger expectation: a specific failure point that distinguishes activation memory from gradient and optimizer memory.

## Expected Successful Behavior
Recognises that failing at the first optimizer step rather than in the forward pass points at gradient and optimizer state rather than activations, so more gradient checkpointing will not help. Works through the sharded-training options in order, names the launcher variant built for this exact case, discusses the precision choice and the memory-efficient optimizer, and explains the trade-off each one costs. Mentions the reduced image resolution as a last resort and warns which choices change resume behaviour.

## Failure Signals
Repeating gradient checkpointing or batch size; recommending settings unsupported by the entry point; ignoring the existence of a purpose-built launcher variant; not stating the trade-offs.
