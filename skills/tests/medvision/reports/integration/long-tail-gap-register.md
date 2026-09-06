# Long-tail gap register — `medvision`

Capabilities and artifacts that the generated skill does not cover in depth, why, and what a
future extension would add.

| Capability or artifact | Why not covered | Risk | Suggested future extension |
| --- | --- | --- | --- |
| Figure and webpage export scripts (28 shell, 24 Python) | Paper- and website-specific: they read roster YAMLs that live beside them, write into repository figure trees, and several target an external site checkout. Catalogued by entry point and inputs in a root reference instead | low | A `visualization` sub-skill wrapping the three or four genuinely reusable per-sample overlay scripts with explicit paths |
| GRPO training itself | Runs in an external fork of the RL framework, where the reward functions, curriculum and recipes live. This repository contributes dataset preparation and documentation | medium | A companion skill for the fork; the RFT sub-skill already documents the recipe and its variables |
| Third-party model checkouts | Vendored upstream model repositories are excluded from extraction; the evaluation wrappers that use them are covered | low | None; the wrappers are the supported interface |
| Private experiment launchers | Excluded from extraction as private work in progress, and explicitly never to be launched | none | None |
| Dataset package internals | The dataset codebase ships with the Hugging Face dataset, not this repository. Its environment variables, versioning and on-disk layout are covered; its per-dataset preprocessing modules are not | medium | A separate skill for the dataset package, which would also own annotation generation |
| Sphinx documentation site | The published docs duplicate README content already distilled here | low | None |
| Experimental and legacy task lists | Named and explained, but their variants (visual-prompt, without-instruction, without-medical-image) are not documented workflow by workflow | low | A reference on ablation task variants if they become supported |
| Per-dataset annotation semantics | The skill explains annotation types and versions generically; it does not enumerate every dataset's task IDs and labels | medium | Generate a dataset catalogue reference from the shipped dataset-info files |
| API-model runtime behaviour | Cap tables and resize rules are documented, but no live API call was made | medium | Run the repository's own token-count probes once credentials are available |
| GPU runtime evidence for four workflows | No CUDA device on the preparation host | high | Re-run the blocked native cases on a GPU host before claiming backend verification |
| Two stale repository test suites | `unit-test/nMAE/test-{1,3,5}.py` and `unit-test/scaledPS/test-1.py` fail against the current source | low | Report upstream; the skill documents the current API, verified by inspection |
| One stray results directory | A non-roster judge directory makes the invariants gate fail | low | Report upstream; documented with the fix in the judge sub-skill |
