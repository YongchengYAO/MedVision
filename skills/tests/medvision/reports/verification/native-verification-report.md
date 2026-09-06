# Native repository verification — `medvision`

Machine-readable twin: `native-verification-report.json`.

The generated skill was integrated first; these are the original repository's own
tests, examples and CLI entry points run afterwards as ground truth. All runs used the
private inspection environment (CPU-only host, no GPU visible).

| Status | Count |
| --- | --- |
| PASS (native tests) | 27 |
| PASS (help-only / parser checks) | 22 |
| NATIVE_FAIL | 6 |
| SKILL_GAP | 0 |
| BLOCKED_REQUIRED_BACKEND | 5 |
| SKIP_UNSAFE | 4 |

## What passed

- **Metric semantics** — `unit-test/detection-metric-failure/test-{1,2,3}.py` confirm the
  conventions the skill states: detection failures score IoU/F1/Precision/Recall as 0
  while MAE is NaN, `parse_outputs` and the summarizer agree, threshold metrics divide by
  the total sample count, and `Acc@IoU>=0.50` equals `IoU>0.5`.
- **nMAE aggregation** — `unit-test/nMAE/test-2.py`.
- **Scaled pixel size** — `unit-test/scaledPS/test-{2,3,4,5}.py` (these need the data
  directory; they were re-run with it set and ran fully offline).
- **Equation accuracy** — `unit-test/equation-accuracy/test-{1..5}.py`, including the
  near-zero ground-truth filter.
- **LLM judge** — `unit-test/llm-parsing/test-{1,2,3,4,5,6,7,9,11}.py`: strict-parser replay
  parity, span verification, the decode path, budget invariants, record ordering, the
  decision table with strict-success precedence, both MOCK entry points, and the registry.
- **Tool-use sandbox** — `unit-test/tool-use/test-{1..4}.py`.
- **verl prompt construction** — `unit-test/detection-verl-nocot/test-1.py`.
- **CLI surface** — 22 help/parser checks across the benchmark, SFT, RFT, dataset and
  judge entry points. Every flag documented in the generated reference files was taken
  from this output rather than from memory.

## The six native failures (all pre-existing repository conditions, none a skill gap)

| Case | Classification | Why it is not a skill defect |
| --- | --- | --- |
| `unit-test/nMAE/test-{1,3,5}.py` | Stale test vs. current source | `_compute_physical_diagonal()` now takes a required keyword-only `explicit_scale`; the tests still call the old signature. The generated skill documents the current signature, verified by inspection. |
| `unit-test/scaledPS/test-1.py` | Stale test | The test extracts source with a regex that now also captures a trailing decorator, so the extracted snippet fails to parse. Tests 2-5 of the same suite pass. |
| `unit-test/nMAE/test-4.py` | Documented data condition | A distance sample returned `success=False` because the NIfTI images are not at the recorded paths, so the physical diagonal cannot be computed and the failure is silent. The results sub-skill's troubleshooting already lists this exact cause and its fix. |
| `unit-test/llm-parsing/test-8.py` | Repository data hygiene | Detection invariants fail because a stray non-roster directory (a `_CoT`-suffixed duplicate model folder) holds strict parsed records inside an `llm-parsed_<judge>/` folder. All 19 roster models are clean. The judge sub-skill's troubleshooting documents the symptom and the fix. |

Two of these are worth reporting back to the repository owner: the two stale test suites,
and the stray results directory that makes the judge invariants gate fail.

## Blocked on a required backend

No CUDA device is visible on this host, so the following could not receive runtime
evidence and remain explicit import blockers rather than skips:

1. Local open-weight model evaluation (`script/benchmark-*/eval__*.sh`).
2. LoRA and full-parameter fine-tuning (`script/sft/train__*.sh`) — argument parsers were
   verified on CPU, the training itself was not run.
3. LLM-judge inference (`run_llm_parsing.sh smoke|pilot|full`) and `test-sweep.sh`.
4. The BiomedParse ablation tracks, which additionally need `nvcc` for a source build.

Guidance for all of these was drafted from source, launchers and documentation, and is
supported by help/parser checks plus synthetic assertion-backed usability cases. Those
cannot substitute for GPU runtime evidence, and the final report keeps the block visible.

## Deliberately skipped

API image-resize probes (need live credentials), the judge environment installer
(multi-gigabyte downloads), the judge `prep` step (destructive), and the SFT loss-masking
tests (download model processors). Each is covered by a synthetic case or by documentation
instead.

---

## Update after the maintainer fixed three defects (same session)

Defects 1, 2 and 5 from the review notes were fixed, and a fourth case turned out to share defect 1's
root cause. The CPU suite now stands at **32 passed, 1 failed** (was 27 passed, 6 failed).

| Case | Was | Now | What changed |
| --- | --- | --- | --- |
| `unit-test/nMAE/test-1.py` | NATIVE_FAIL | **PASS** | ten call sites across three files now pass `explicit_scale=None`, the documented value that selects the hash-based path |
| `unit-test/nMAE/test-5.py` | NATIVE_FAIL | **PASS** | same |
| `unit-test/nMAE/test-3.py` | NATIVE_FAIL | **PASS** | same, plus its two success-path predictions are now wrapped in `<answer>` tags |
| `unit-test/nMAE/test-4.py` | NATIVE_FAIL | **PASS** | its three predictions wrapped in `<answer>` tags |
| `unit-test/scaledPS/test-1.py` | NATIVE_FAIL | **PASS** | the source-extraction lookahead now stops at a decorator as well as a `def` |
| `unit-test/llm-parsing/test-8.py` | NATIVE_FAIL | unchanged | the stray results directory was deliberately left alone |

**A correction to this report's earlier classification.** `unit-test/nMAE/test-4.py` was recorded as a
"documented data condition" where the images are not at their recorded paths. That was wrong. The
image files are present; the real cause was that the test fed the scorer a bare numeric string, and
the parser only reads numbers inside `<answer>` tags, so every sample scored as a parse failure. The
same cause was masking a second failure inside `test-3.py`. Fixing it also restored the test's intent:
its angle assertions previously passed vacuously, because everything was failing, and now genuinely
demonstrate that the normalised-error metric is suppressed for angles while the absolute-error metric
still succeeds.

This does not change any coverage conclusion: `SKILL_GAP` remains 0, and the four
`BLOCKED_REQUIRED_BACKEND` items are untouched.

## Update: task-YAML defects fixed

The eight task-YAML defects were fixed in the same session. The bundled inventory checker now
reports no problems across 1253 task YAMLs, and the vendored task registry indexes 1304 task names.

| Group | Files | Fix |
| --- | --- | --- |
| Five variant files copied from a plain sibling | one each in CrossMoDA and autoPET-III, three in BraTS24 | corrected the `task:` name and the `include:` base to match the filename, so each ablation variant is a distinct task pointing at its own base |
| Three MSD mask-size files | `MSD_MaskSize_Task10_Sagittal{,-VP,-VP-woMedImg}.yaml` | changed the overridden key from `dataset_path` to `dataset_name`, so they resolve to the dataset repository plus a config rather than to a repository named after a config |

Verified afterwards: every `include:` target exists, no task name is declared twice, each of the five
ablation variants is now declared by exactly one file, all four shipped task lists still resolve, the
plane-OOD task still resolves to its own file, and the three MSD tasks now produce the same
`load_dataset(repository, config)` shape as an always-correct sibling. The CPU suite is unchanged at
32 passed, 1 failed.
