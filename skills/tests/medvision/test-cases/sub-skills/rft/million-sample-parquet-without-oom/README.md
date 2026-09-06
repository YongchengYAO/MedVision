# Build a 1M-sample parquet dataset on a memory-limited host

## User Persona
An RL practitioner who has the MedVision data and now needs the training parquet, hitting a silent
out-of-memory kill.

## Scenario Coverage
- Skill area: `rft`
- Capability: parquet dataset construction at scale
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/workflows.md`, `references/cli-reference.md`,
  `references/troubleshooting.md`, `scripts/build_parquet_ds.sh`
- Trigger expectation: names the verl parquet build and the detection training scale.

## Expected Successful Behavior
Switches to the checkpointed/sharded builder and explains its shard-size control, reduces the
dataset-construction worker counts, and explains that a silent kill with no traceback is the
container memory ceiling rather than a Python error, so the host's reported RAM can be misleading.
Sets the target image shape and the model family that the parquet is built for, and warns that the
per-task limits and the global cap must agree or rows are silently truncated.

## Failure Signals
Recommending the non-checkpointed builder; no shard-size or worker guidance; treating the kill as a
crash to debug with a traceback; omitting the model-family/image-shape coupling.
