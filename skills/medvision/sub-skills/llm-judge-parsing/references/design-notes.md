# Design notes: decision table, span verification, identity lock, invariants

Distilled from the pipeline's design specification (last verified there
2026-08-12, snapshot 2026-08-20) and re-checked against the module sources and
the unit tests, which pass on CPU. Read this before editing anything under the
checkout's `script/llm-parsing/` or before explaining a number in a judge report.

## 1. Task routing (what to read before an edit)

| you were asked to… | read | hard constraint |
|---|---|---|
| change the judge prompt / budget / window | §4, §6 | moves `prompt_fp` ⇒ Stage 0 rebuild (~1.1 GB) ⇒ Stage 1 aborts on old queues |
| change what counts as a parsed answer | §2, §3 | one decision table; unit test 7 asserts it row for row |
| loosen/tighten span acceptance | §3 | invariant I4 (no fabricated value) must survive every tier |
| add a task type | §6 | `arity` must equal the strict parser's `k`; add `STEP_SPECS` key + `EXPECTED_ROSTER_COUNTS` |
| fix judge JSON parsing | §5 | CPU re-parse via `reparse_judge_out.py`; never re-judge |
| re-run / repair a sweep | §4, `recipes.md` | `--redo_invalid` required, else a silent no-op; APPEND, never `cat new old` |
| explain a number in a report | §2, §8 | strict and judge columns share one summarizer code path |

## 2. The decision table (single source: `judge_decision.DECISION_TABLE_DOC`)

Two strict outcomes × five judge outcomes = ten rows; consumed by Stage 2,
Stage 4 and unit test 7.

```
 # | strict | judge row | judge_status | final_answer.status | span ok | pred     | mode
---+--------+-----------+--------------+---------------------+---------+----------+----------------------
 1 |  ok    | present   | ok           | present             |  yes    | strict   | conclusion_in_format
 2 |  ok    | present   | ok           | present             |  no     | strict   | conclusion_in_format
 3 |  ok    | present   | ok           | no_conclusion       |   -     | strict   | conclusion_in_format
 4 |  ok    | present   | invalid      |  -                  |   -     | strict   | conclusion_in_format
 5 |  ok    | ABSENT    |  -           |  -                  |   -     | strict   | conclusion_in_format
 6 | fail   | present   | ok           | present             |  yes    | verified | conclusion_off_format
 7 | fail   | present   | ok           | present             |  no     | ""       | undetermined
 8 | fail   | present   | ok           | no_conclusion       |   -     | ""       | no_conclusion
 9 | fail   | present   | invalid      |  -                  |   -     | ""       | undetermined
10 | fail   | ABSENT    |  -           |  -                  |   -     | ""       | undetermined
```

- `verify_span` runs only when `judge_status=="ok"` and
  `final_answer.status=="present"`.
- `is_success(mode) ⇔ mode ∈ {conclusion_in_format, conclusion_off_format} ⇔
  LLM_filtered_resps[0] != ""` — three spellings of one fact.
- `LLM_judge.judge_pred` stores the judge's *own* verified extraction, not the
  decided value; that is what agreement is measured against (against the decided
  value it would be 100% by construction on strict-success rows).
- Relationship to the regex, five couplings: **priority** (strict wins
  unconditionally, the judge is the fallback, never the reverse); **prompt
  isolation** (the judge is never shown `regex_pred`, target or image);
  **transcription** (every digit reaching `LLM_filtered_resps` is produced by the
  benchmark's own `NUM_RE`); **drift gate** (Stage 0 replays the strict parser
  and aborts on any mismatch — 0 of 522,868); **reference set** (agreement is
  measured on the regex's own successes, 38,322 TL / 31,360 AD / 403,878 Det).

## 3. Span verification (`judge_verify.verify_span(response, span, values, expected_arity)`)

Returns `{ok, reason, tier?, numbers, pred}`. Pre-checks: `empty_span`,
`no_values`, `arity_mismatch:{n}!={k}`, `non_numeric`. Matching is a
**contiguous run** of `NUM_RE` numbers searched **from the end** (the strict
parser's last-k rule); tolerance `1e-9` absorbs JSON float round-trip only.

| tier | containment test | numbers transcribed from |
|---|---|---|
| `exact` | whitespace-collapsed span ⊆ whitespace-collapsed response | the span |
| `normalized` | a span variant (casefold, whitespace deleted, control-char inversion, stripped wrapping quotes / judge-added closing tag, de-LaTeX) located in the response; plus an explicit check that every transcribed number appears in order in the response's own `NUM_RE` run (`normalized_values_not_in_response`) | the repaired variant |
| `value_anchor` | the values form a contiguous `NUM_RE` run in the response itself; all-zero values rejected (`all_zero_value_anchor`, the historical skeleton-echo signature) | the response |

Terminal failures: `span_not_found`, plus `values_not_in_span` (the span was located verbatim but its numbers do not form the claimed contiguous run — deliberately distinguished from a missing span) and `no_numbers_in_span`. The accepting tier is recorded in
`LLM_judge.verify_tier`. **I4: a value the model never wrote cannot be scored at
any tier.** Corpus tier distribution (2026-08-12): 431,254 exact / 5,731
normalized / 44,145 value_anchor; tier 3 supplies only 3.7% (TL) / 5.5% (AD) /
4.7% (Det) of *recoveries*. Job B steps (TL 4 steps; AD:distance and AD:angle 3
steps) run the same verifier at each step's `n_values`; persisted, never scored.

## 4. Identity: the prompt-fingerprint lock

`prompt_fingerprint(task_type, step_key)` = deterministic JSON over the rendered
messages for a sentinel response, the JSON schema, the response window
constants, the elision marker, `TASK_SPECS[task]["max_tokens"]` and
`JUDGE_REASONING_EFFORT`; `short_prompt_fp` = first 16 hex of its BLAKE2b hash.
Built by **rendering**, not by listing ingredients (a listing version once had
three silent holes). No reader identity enters it.

Lock semantics: queue rows and judge-out rows both carry the stamp;
`run_judge_vllm._expected_fps` and `apply_judge.assert_judge_out_prompt`
hard-abort on an unknown stamp; `--accept_prompt_fp` (driver `ACCEPT_FP`)
whitelists historical stamps so a file legitimately holding two prompts' rows
(a repair pass at a different budget) can be applied.

Stamps: current code (verified) TL `c515f64a54eafab8`, AD:distance
`02728aba5cea5964`, AD:angle `b2b30f63c35b946f`, Detection `fd1a0ea674ca6a44`
(@4096). Historical, whitelisted by default in `ACCEPT_FP`: TL `09ee44a311e85670`
(@1024) / `dd3c7fb2d50255db` (@3072); AD:dist `02a90adc517baab6` / `5cfdd478b855e9e4`;
AD:angle `a3fd2c4a87a9ce4e` / `56de1c8d5a7719b0`; Det `49c9fbdcf069aef9` (@256) /
`54e8d9ef545df4a6` (@512). Pre-2026-08-08 @4096 stamps (TL `8f7579e66a5ad982`,
AD:dist `d7acc730e39a248e`, AD:angle `7159a2e0c4b178dd`, Det `e730a9758384c648`)
no longer match the code (the skeleton-placeholder fix changed the prompt).

Budget: `DEFAULT_JUDGE_MAX_TOKENS=4096` for all tasks, and **keep it**.
Per-task right-sizing caused two silent truncation incidents (TL @1024: 8.70%
judge-invalid; Detection @256: 88.9% of residual failures at exactly 256
tokens). A too-small budget never fails loudly — it inflates judge-invalid.
Re-tokenising every stored completion showed that what hits a cap is
overwhelmingly degenerate repetition loops; Detection's real requirement is
p99 = 435 tokens. Raising above 4096 buys nothing and forces a fingerprint
rebuild.

## 5. Decode path (`judge_decode.py`)

1. Harmony channel split (`assistantfinal` / `<|channel|>final<|message|>`;
   last marker wins; text with neither passes through) — a no-op for the current
   reader, kept so archived files from the retired harmony-emitting reader stay
   re-parsable.
2. Candidate scan over every balanced `{…}` group (bounded: 64 starts, 400,000
   chars), preferring the first candidate containing `final_answer` (LaTeX
   braces like `{mm}` shadowed the object when the first group was taken).
3. Escape repairs in a deliberate order: `preescape_latex_escapes` always and
   **before** parsing (`\b`/`\f` are legal JSON escapes, so a verbatim `\boxed`
   would parse *successfully* into a corrupted span); `repair_json_escapes` only
   after a plain parse fails.
4. `validate_judge_obj` — shape only; rejects the v1 `absent` status; permanently
   accepts an all-zero `values` beside a non-`present` status (historical echo
   rows must stay re-parsable).

Any failure ⇒ `judge_status="invalid"` + reason ⇒ decision rows 4 or 9. A
decoder fix is a **CPU re-parse** (`reparse_judge_out.py`), never a GPU pass; it
refuses to write if any row moves ok→invalid or any ok row's values change.

## 6. Edit hazards

| edit | forced consequence |
|---|---|
| any text in `SYSTEM_PROMPT` / `USER_TEMPLATE` / `_output_skeleton` / `STEP_SPECS.what` | fingerprint moves ⇒ Stage 0 rebuild ⇒ every `cache_key` changes ⇒ old queues abort |
| `DEFAULT_JUDGE_MAX_TOKENS` or any `TASK_SPECS[*]["max_tokens"]` | same (budget is in the fingerprint) |
| `RESPONSE_WINDOW_*` / `RESPONSE_ELISION_MARKER` / `JUDGE_REASONING_EFFORT` | same |
| `TASK_SPECS[*]["arity"]` | must equal the strict parser's `k` (unit test 4) |
| a new analysis output written into `parsed/` | add its stem to `EXCLUDED_JSONL_STEMS` (`_proc_acc`, `_eq_acc`, `_judge`) **and** to every summarizer's filter |
| a new key in `llm-parsed` records | check unit test 8's key-order/identity assertions |
| relaxing `verify_span` | I4 must hold; add a fixture to unit test 2 |
| the duplicated strict primitives in `judge_io` (`extract_last_k_nums_within_answer_tag`, `NUM_RE`, `extract_response`, `iter_records`) | must stay byte-equivalent to `parse_utils`/`parse_outputs` (tests 1, 5; Stage 0 gate) |
| adding a `final_answer.status` value | v1-rejection logic and `ANSWER_MODES` both change (tests 3, 7) |
| registering a reader with an empty or duplicate `out_suffix` | re-widens every driver glob and lets two readers' rows interleave in one file (test 11 fails) |
| moving `JUDGE_REASONING_EFFORT` into the registry | invalidates every queue on disk |
| `JUDGE_MAX_MODEL_LEN` | **not** in the fingerprint; safe to change without a rebuild |

Never: re-implement a metric in the pipeline (import `cal_metrics`); keep
`filtered_resps` alongside `LLM_filtered_resps`; write judge output into
`parsed/`; treat `--limit` output as a full run; `cat new old` when merging a
repair; run Stage 1 repairs through the driver (they are launched by hand on
purpose).

## 7. Invariants and gates

| id | invariant | enforced at | test |
|---|---|---|---|
| I1 | strictly additive: `parsed/` never written | Stage 2 output path | — |
| I2 | strict-first: `LLM_filtered_resps == filtered_resps` on every strict-success row, every model | `decide_answer` | 7, 8 |
| I3 | a judge failure costs a recovery, never corrupts a published value | corollary of I2 | 8 |
| I4 | no fabricated value can be scored, at any tier | `verify_span` | 2 |
| I5 | one decision table, three consumers | `judge_decision` | 7 |
| I6 | no new metric family — judge and strict columns share the summarizer path | Stage 3 + `cal_metrics` | Stage-3 byte-identity gate |
| I7 | prompt identity is a hash of the real prompt, never a constant | `prompt_fingerprint` | 6 |
| I8 | no skeleton placeholder is a legal value (the `[0.0]*k` echo bug) | `_output_skeleton` renders `"<number>"` | 6 |
| I9 | exactly one `llm-parsed` record per source record; duplicates exist only in judge-out and resolve last-wins | `load_judge_index` | 8 |

| gate | stage | behaviour |
|---|---|---|
| replayed parser ≡ stored `filtered_resps` | 0 | abort + retract queue |
| roster counts 46,379 / 39,140 / 437,349 (19 models) | 0 | abort on mismatch; skipped under `--limit` and for a non-default `--task_dir` (`[gate n/a]`) |
| every roster model has `parsed/` (no glob fallback) | 0 | abort |
| `gate_valid_rate` ≥ `--min_valid_rate` (0.95) over the first 200 rows | 1 | abort **writing nothing**; repair queues must pass a low/0 rate |
| queue/output `prompt_fp` known | 1, 2 | abort |
| `--parsed_dirname parsed` regenerates published reports byte-identically | 3 | verified TL/AD/Detection |
| record invariants on real output, incl. mock provenance (`LLM_judge.judge_model != "mock"`) | test 8 | driver's `invariants` step |
| mock vs real provenance on resume | 1 | abort |
| merge cannot destroy repair-pass rows or a newer re-parse | 1 | refuse |
| shard count matches the shard filenames | 1 | a changed `NUM_SHARDS` is caught as stale |

Roster scoping (all stages): Stage 0 iterates `load_roster(...)`; Stage 1
inherits the queue; Stage 2 iterates the roster; Stages 3/3b receive
`--models <roster>` from the driver (the summarizers otherwise walk every
directory under the tree — 57 dirs under the TL tree against 19 in the roster);
Stage 4 iterates the roster. Unit test 8 scans **every** `llm-parsed<sfx>/`
directory under the tree, roster or not.

## 8. Measured snapshots

Current reader (`gemma-4-31b`), 19-model roster, 522,868 responses, read
2026-08-20 from the on-disk judge reports:

| | TL | AD | Detection |
|---|--:|--:|--:|
| responses | 46,379 | 39,140 | 437,349 |
| strict-parsed | 38,322 | 31,360 | 403,878 |
| SR strict regex | 82.6% | 80.1% | 92.3% |
| SR format-robust | 94.8% | 94.2% | 98.5% |
| ΔSR | +12.2 | +14.1 | +6.2 |
| recovered (`conclusion_off_format`) | 5,642 | 5,513 | 27,034 |
| `no_conclusion` | 2,135 | 2,242 | 3,430 |
| `undetermined` | 280 | 25 | 3,007 |
| judge-invalid (residual, last-wins) | 6 | 16 | 20 |
| judge–regex agreement on regex successes | 99.932% | 99.987% | 98.872% |
| span-verify rejections | 284 (0.61%) | 10 (0.03%) | 3,352 (0.77%) |

Open item: Detection agreement is the outlier — 2,977 of its 3,352 rejections
(88.8%) are `arity_mismatch:2!=4` (the judge quoting two numbers for a
four-number box); the verifier discards them, costing recoveries, not
correctness.

Retired reader (the paper's figures), 18-model roster, 496,296 responses: SR
strict 77.0% / 73.8% / 93.4% → format-robust 89.7% / 90.1% / 98.4% (ΔSR +12.6 /
+16.3 / +5.0); `undetermined` 693 / 173 / 1,085; agreement 99.78% / 99.96% /
99.94%. **Do not diff the two tables row by row** — they differ by reader,
roster and corpus at once.

## 9. Reproducibility rules

Judge output is **not** reproducible run to run — on the same GPU, not merely
across generations — and the cause is numerical non-determinism, not sampling
(`temperature=0.0`, `top_p=1.0`, so the seed is inert). Mechanisms: kernels are
not batch-invariant by default (reduction order depends on batch shape, which
depends on shard layout, chunk boundaries and arrival order); prefix caching
(`enable_prefix_caching=True` is hardcoded in `run_judge_vllm.run_gpu`, no
`--enforce_eager` flag exists); and, for the retired MoE reader only, top-k
expert routing as a discrete amplifier. The quoted flip rates (12.8% identical
verdicts on twice-judged rows; 63.0% vs 98.1% recovery at 256 vs 512 tokens)
were measured on the **retired** reader and on rows *selected for having failed*
— do not restate them for the current dense reader, whose flip rate has never
been measured (the current files hold zero twice-judged `(qid, prompt_fp)`
pairs).

What transfers to the current reader: greedy ≠ reproducible; non-batch-invariant
kernels; prefix caching; the blast radius (a flip can cost a recovery, never a
published value); and the four operating rules:

1. Never compare judge-invalid rates across machines or checkpoints.
2. Never credit a re-run's recovery to a code change without a same-raw A/B
   (`reparse_judge_out.py`) or a same-day control arm.
3. Never sample a probe in roster order (roster order is model-biased; a
   2,000-row roster-ordered probe suggested 82.9% where the population gave 63.0%).
4. Treat `judge-out_*.jsonl` as the artifact of record. Re-deriving reports from
   it (Stages 2–4) is byte-reproducible; re-judging is not.

The real fix exists in the pinned engine: `vllm==0.19.0` ships batch-invariant
kernels behind `VLLM_BATCH_INVARIANT=1`. Enabling it requires the FLASH_ATTN or
TRITON_ATTN attention backend (else a worker-init `RuntimeError`; `run_gpu`
pins none), forces deterministic NCCL settings (felt at `tensor_parallel=2`),
disables AOT compile/TF32 fast paths, and has an unmeasured, expectedly material
throughput cost. It does not invalidate queues (the fingerprint never includes
engine settings) but requires a re-judge to produce a new judge-out. Treat it as
what makes a determinism-critical arm possible, not as a production default.

## 10. Known limitations

- `undetermined` ≠ 0 (0.39% on the retired reader's corpus; 3,312 rows on the
  current one): dominated by degenerate repetition loops (~0.3% of any queue),
  `missing_values` shapes, and Detection arity mismatches (a box given as two
  points is a transcription question deliberately not guessed at).
- Job B step extractions are persisted but unscored; process/equation-accuracy
  integration is out of scope (`../../analysis/SKILL.md`).
- The driver drives repair only from Stage 2 (`analyze`); Stage 1 repair runs
  are launched by hand on purpose.
- Structured (grammar-constrained) decoding is attempted but not depended on;
  xgrammar's per-token bitmask over the free-form `span` serialises decode
  behind one CPU core. Valid JSON comes from the shown skeleton;
  `--min_valid_rate` is the real guard.
