# Difficult-case plan — `medvision`

## Per sub-skill (two each)

| Sub-skill | Case 1 | Case 2 | Repository evidence extended |
| --- | --- | --- | --- |
| environment-setup | dependency pin lifted by the dataset installer | editable install silently shadowed by a copy | install source and its comments; neither has a repository test |
| dataset-and-tasks | annotation pin refused, counts differ from the leaderboard | derive coronal test configs from an SFT list, offline | the planner-version test exists but needs network; these do not |
| benchmark-evaluation | resume an interrupted multi-GPU run; does a prompt edit apply | API pilot via a reseller, with a credit-hold failure | no repository test covers the tracker/cache interaction |
| results-parsing-and-metrics | mean IoU above the threshold metric | re-summarize from the judge directory for two models | anchored to the detection metric-failure tests, which pass |
| llm-judge-parsing | extend a finished campaign by one model | differing judge-invalid rates across machines | anchored to the judge decision-table tests; the reasoning is untested upstream |
| sft | full-parameter training out of memory at the first optimizer step | sample limit larger than the pool | limit parsing has no repository test at the entry-point level |
| rft | build a million-sample parquet on a memory-limited host | reuse a parquet across model families | the verl prompt test passes; neither scenario is tested |
| analysis | clinical analysis on a task with no cutoff table | an equation-accuracy number that looks best because of low coverage | anchored to the equation-accuracy tests, which pass |
| extending-models-and-tasks | registered model missing its image-size branch | add a plane variant of an existing task | anchored to the registry/dispatch consistency check |
| biomedparse-ablation | scoped smoke test that leaves shipped results alone | re-score one dataset and merge with a chosen checkpoint | launchers are GPU-blocked, so both are reasoning cases |

Six of the twenty are refusal cases, where the correct behaviour is to decline or qualify a
plausible-sounding request. That is deliberate: the most damaging failure mode for this skill is
confidently endorsing an invalid comparison.

## Integrated cases (two)

Both are adapted from repository-native end-to-end material rather than synthesized, as required:

1. `integration/end-to-end-evaluate-parse-summarize-judge` — adapted from the repository's own
   documented four-step benchmark protocol. Crosses root routing, `environment-setup`,
   `benchmark-evaluation`, `results-parsing-and-metrics` and `llm-judge-parsing`. Tests whether the
   skill preserves comparability constraints (annotation version, sample limit, token budget) that
   are stated in different places across those sub-skills.
2. `integration/finetune-then-fair-comparison` — adapted from the released model's own documented
   training recipe. Crosses `dataset-and-tasks`, `sft`, `benchmark-evaluation`,
   `results-parsing-and-metrics` and the root roster. Tests whether the skill surfaces the
   experimental-validity constraints and correctly refuses to call a supervised-only run a
   reproduction of a model that also had a reinforcement stage.

No synthetic integrated case was needed; the repository supplied suitable cross-capability material
for both.
