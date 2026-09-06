# Phase B re-runs the load+split stage before training

## User Persona
A practitioner who separated dataset preparation from training correctly and is surprised that the
training launch still reads every source config before the first step.

## Scenario Coverage
- Skill area: `sft`
- Capability: prepared-dataset hand-off between the two phases (`--prepared_ds_dir`)
- Difficulty: troubleshooting
- Prompt file: `user_request.txt`
- Expected references/scripts: `references/troubleshooting.md`, `references/data-preparation.md`,
  `references/workflows.md`, `scripts/sft_launcher_template.sh`
- Trigger expectation: a symptom (long per-config loading in phase B, repeated on resume) whose cause is a
  missing flag rather than a rebuild.

## Expected Successful Behavior
States that nothing is being rebuilt: with `--skip_process_dataset true` but no `--prepared_ds_dir`, rank 0
re-runs the load+split stage only to derive the default directory name, because that name encodes the true
split sizes and is only known after loading; the other ranks wait at the barrier, and the cost repeats on every
resume. Tells the user to pass the directory phase A printed (`Prepared dataset saved at '<dir>'`) as
`--prepared_ds_dir` on the training launch, after which the entry point loads that directory as-is and skips
loading the raw configs. Mentions that the repository launchers and the bundled template capture that line
from phase A's tee'd log automatically and abort before the GPU launch when it is missing, and gives a way to
verify the fix from the phase-B log.

## Failure Signals
Claiming the dataset is being rebuilt or that `--skip_process_dataset` is being ignored; advising the user to
make the flag sets identical without mentioning `--prepared_ds_dir`; not explaining why the name cannot be
derived without loading; not mentioning that the cost repeats on resume.
