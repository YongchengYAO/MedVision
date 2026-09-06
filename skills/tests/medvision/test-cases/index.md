# Usability test cases — `medvision`

26 case directories. Each holds `user_request.txt` (the copyable future-user prompt), `README.md`
(persona, coverage, expected behaviour, failure signals) and `assertions.json` (gradeable PASS/FAIL
assertions). These are review artifacts, not runtime skill content.

## Cases

| Case | Target | Capability | Difficulty | Tests |
| --- | --- | --- | --- | --- |
| `root/first-time-orientation-and-environment-check` | root | orientation, install, readiness | basic | route discovery |
| `root/which-part-of-the-stack-do-i-need` | root | routing and sequencing across sub-skills | intermediate | route discovery |
| `sub-skills/environment-setup/hub-lifted-after-dataset-install` | environment-setup | version-pin diagnosis (below-floor hub) | troubleshooting | refreshed behaviour (2026-09-05) |
| `sub-skills/environment-setup/editable-install-silently-shadowed` | environment-setup | install-mode diagnosis | troubleshooting | troubleshooting clarity |
| `sub-skills/dataset-and-tasks/planner-version-pin-and-counts` | dataset-and-tasks | annotation versions and the ceiling rule | troubleshooting | workflow depth |
| `sub-skills/dataset-and-tasks/sft-list-to-coronal-ood-configs` | dataset-and-tasks | name-to-config derivation, offline | intermediate | bundled-script executability |
| `sub-skills/benchmark-evaluation/resume-interrupted-multi-gpu-run` | benchmark-evaluation | resume and cache invalidation | advanced | workflow depth |
| `sub-skills/benchmark-evaluation/api-pilot-credit-and-provider-wiring` | benchmark-evaluation | API wiring and cost failures | troubleshooting | troubleshooting clarity |
| `sub-skills/results-parsing-and-metrics/threshold-vs-mean-iou-discrepancy` | results-parsing-and-metrics | metric denominators and failure handling | troubleshooting | troubleshooting clarity |
| `sub-skills/results-parsing-and-metrics/resummarize-judge-output-two-models` | results-parsing-and-metrics | alternative parsed directory | advanced | support-workflow discoverability |
| `sub-skills/llm-judge-parsing/add-one-model-to-finished-campaign` | llm-judge-parsing | incremental campaign growth | advanced | workflow depth |
| `sub-skills/llm-judge-parsing/nondeterministic-judge-rates` | llm-judge-parsing | reproducibility semantics | troubleshooting | troubleshooting clarity |
| `sub-skills/sft/fullft-oom-at-first-step` | sft | memory configuration | troubleshooting | troubleshooting clarity |
| `sub-skills/sft/phase-b-stalls-in-dataset-loading` | sft | prepared-dataset hand-off (`--prepared_ds_dir`) | troubleshooting | refreshed behaviour (2026-09-04) |
| `sub-skills/sft/sample-limit-exceeds-pool` | sft | limit resolution and validation split | intermediate | support-workflow discoverability; regression guard for the refresh |
| `sub-skills/rft/million-sample-parquet-without-oom` | rft | sharded dataset construction | troubleshooting | workflow depth |
| `sub-skills/rft/parquet-reuse-across-model-families` | rft | model-family coupling | advanced | correctness refusal |
| `sub-skills/analysis/cda-on-a-task-without-cutoffs` | analysis | analysis scope | troubleshooting | correctness refusal |
| `sub-skills/analysis/equation-accuracy-coverage-trap` | analysis | coverage versus mean | advanced | correctness refusal |
| `sub-skills/extending-models-and-tasks/registered-model-missing-resize-branch` | extending-models-and-tasks | registry versus dispatch | troubleshooting | troubleshooting clarity |
| `sub-skills/extending-models-and-tasks/add-coronal-task-for-existing-dataset` | extending-models-and-tasks | task authoring and registration | intermediate | workflow depth |
| `sub-skills/biomedparse-ablation/smoke-test-single-dataset-without-touching-results` | biomedparse-ablation | scoped smoke test and output isolation | intermediate | workflow depth |
| `sub-skills/biomedparse-ablation/rescore-one-dataset-with-finetuned-checkpoint` | biomedparse-ablation | filtered re-run and merging | advanced | workflow depth |
| `sub-skills/biomedparse-ablation/which-pins-the-dataset-install-can-still-move` | biomedparse-ablation | dependency-pin reasoning after the dataset install | troubleshooting | refreshed behaviour (2026-09-05); correctness refusal |
| `integration/end-to-end-evaluate-parse-summarize-judge` | 4 sub-skills + root | the full four-step protocol | advanced | cross-skill routing |
| `integration/finetune-then-fair-comparison` | 4 sub-skills + root | training to evaluation with validity constraints | advanced | cross-skill routing |

## Coverage against the coverage/depth matrix

Every one of the ten sub-skills has at least two cases, and the root has two. No sub-skill is untested.
`biomedparse-ablation` has three after the 2026-09-05 refresh.
Capabilities exercised indirectly rather than by a dedicated case: figure generation (catalogued as
reference-only by design), dataset-info regeneration (maintainer, reference-only), and the tool-use
SFT variant (covered inside the SFT cases). These are recorded in the long-tail gap register.

## Difficult-case coverage

- **Per sub-skill:** two difficult synthetic cases each, except `sft` and `biomedparse-ablation`, which have three, all grounded in repository evidence but going
  beyond the repository's own tests, which cover functions in isolation rather than user-facing
  reasoning. Six cases are deliberately *refusal* cases, where the correct answer is to decline or
  qualify a plausible request: reusing a prepared dataset across model families, publishing an
  equation-accuracy number without coverage, running the clinical analysis outside its configured
  scope, running a 1000-sample API pilot, claiming a supervised-only run reproduces the released
  model, and treating judge non-determinism as a broken pipeline.
- **Integrated:** two cases under `integration/`, both adapted from repository-native end-to-end
  material rather than synthesized: the documented four-step benchmark protocol, and the released
  model's own training recipe. Neither is invented.
- **Repository-native anchors:** assertions in the metric, judge, equation-accuracy and dispatch
  cases are anchored to the unit tests that were executed as ground truth, listed in
  `../reports/verification/native-verification-report.md`. The synthetic cases complement those
  tests; they do not replace them.

## Assertion coverage

All 26 cases have `assertions.json` with 5 to 7 gradeable assertions each. Capabilities whose
assertions are synthetic-only because their native execution is blocked on a GPU: local model
evaluation, fine-tuning, judge inference, and the BiomedParse tracks. No case carries fixtures; every
prompt is self-contained, and the two cases that would otherwise need data instead ask for offline
derivation.
