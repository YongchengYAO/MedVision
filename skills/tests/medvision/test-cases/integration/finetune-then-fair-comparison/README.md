# Fine-tune, then compare fairly against the baseline

## User Persona
A group reproducing the project's own post-training recipe at reduced scale and wanting a defensible
comparison.

## Scenario Coverage
- Skill area: integration (four sub-skills plus root roster)
- Capability: training-to-evaluation pipeline with experimental-validity constraints
- Difficulty: advanced
- Prompt file: `user_request.txt`
- Expected references/scripts: `dataset-and-tasks`, `sft`, `benchmark-evaluation`,
  `results-parsing-and-metrics`, root `references/model-roster.md`
- Trigger expectation: adapted from the released model's own documented recipe, the repository's
  native end-to-end training example.

## Expected Successful Behavior
Lays out the pipeline: select the training task list, prepare the dataset in the separate
preparation phase, fine-tune with the launcher pattern for the chosen family, then evaluate both the
fine-tuned model and the stock baseline under identical settings, and finally parse and summarize
both. Names the validity constraints explicitly: the same annotation version, the same sample limit
and token budget for both models, an evaluation split disjoint from training, and the held-out plane
convention if generalisation is being claimed. Warns that the prepared dataset is tied to the model
family whose image processor built it, and that the baseline must be in the supported roster. Notes
that the released model additionally used a reinforcement stage that runs in an external framework,
so a supervised-only reproduction should not be described as reproducing it.

## Failure Signals
Evaluating only the fine-tuned model; changing the sample limit or annotation version between the two
runs; reusing a prepared dataset across model families; claiming to reproduce the released model
without the reinforcement stage; ignoring the train/test split.
