# llm-parsing — agent spec

AUDIENCE: an agent modifying or reasoning about this directory. Optimised for
lookup and edit-safety, not for reading. Human how-to: `README.md`.
LAST VERIFIED: 2026-08-12 against the source in this directory.

---

## 0. TASK ROUTING

| you were asked to… | read | hard constraint |
|---|---|---|
| change the judge prompt / budget / window | §8 §9 §11 | moves `prompt_fp` ⇒ Stage 0 rebuild (~1.1 GB) ⇒ `run_judge` aborts on old queues |
| change what counts as a parsed answer | §5 §6 | one decision table only; `test-7` asserts it row-for-row |
| loosen/tighten span acceptance | §6 | the no-fabrication invariant I4 must survive every tier |
| add a task type | §2 §11 | `arity` must equal the strict parser's `k`; add `STEP_SPECS` key + `EXPECTED_ROSTER_COUNTS` |
| fix judge JSON parsing | §7 | CPU re-parse via `reparse_judge_out.py`; never re-judge (§9) |
| re-run / repair a sweep | §9 §10 | `--redo_invalid` required, else silent no-op; APPEND, never `cat new old` |
| explain a number in a report | §5 §13 | strict vs judge columns share one summarizer code path |

---

## 1. PURPOSE + STANCE

Published MAE/MRE/SR are computed only over regex-parsed samples; that subset is
selected by the regex (TL 23.0% / AD 26.2% / Detection 6.6% fail it, up to 90.8%
for one model). This pipeline re-parses **every** response with an LLM judge
(`google/gemma-4-31B-it`) so *format-following failure* is separable from
*measurement failure*.

Stance, load-bearing everywhere below:
- judge = **extraction device**, never evaluator. Never sees `target`, never sees
  the image, never asked whether a value is correct, never asked to compute.
- judge = **pointer**, not transcriber. It quotes; a regex transcribes (§6).
- judge = **additive only**. It cannot revise a published value (§5).

---

## 2. MODULE MAP

Symbols are the stable reference; line numbers drift.

| file | key symbols | responsibility |
|---|---|---|
| `judge_config.py` | `TASK_SPECS`, `STEP_SPECS`, `ANSWER_MODES`, `SUCCESS_MODES`, `FINAL_ANSWER_STATUSES`, `STEP_STATUSES`, `DEFAULT_JUDGE_MAX_TOKENS=4096`, `RESPONSE_WINDOW_*`, `EXPECTED_ROSTER_COUNTS`, `JUDGE_MODELS`, `JUDGE_DEFAULT_KEY`, `step_spec_key`, `resolve_judge_key`, `judge_entry`, `judge_suffix`, `llm_parsed_dirname`, `queue_filename` | all task/run constants **and the judge registry** (§16); imports nothing from `medvision_bm`. Also a CLI: `python judge_config.py --list` / `--shell [--judge KEY]`, which is how the shell reads the registry instead of mirroring it |
| `judge_io.py` | `NUM_RE`, `find_numbers`, `extract_last_k_nums_within_answer_tag`, `extract_response`, `iter_records`, `content_hash`, `load_roster`, `list_sample_files` | IO + the duplicated strict-parser primitives (§4 D1/D2) |
| `judge_prompts.py` | `SYSTEM_PROMPT`, `USER_TEMPLATE`, `_output_skeleton`, `_windowed_response`, `build_messages`, `build_schema`, `prompt_fingerprint`, `short_prompt_fp` | prompt assembly **and** the JSON schema (there is no `judge_schema.py`) |
| `judge_decode.py` | `split_final_channel`, `iter_json_object_candidates`, `extract_first_json_object`, `parse_judge_json`, `preescape_latex_escapes`, `repair_json_escapes`, `validate_judge_obj` | raw text → validated judge object |
| `judge_verify.py` | `verify_span`, `collapse_ws`, `_span_variants`, `_find_contiguous_run` | tiered span verification (§6) |
| `judge_decision.py` | `decide_answer`, `is_success`, `DECISION_TABLE_DOC` | the single decision table (§5) |
| `build_judge_queue.py` | `build_queue`, `_process_file`, `_fingerprints_for` | Stage 0 |
| `run_judge_vllm.py` | `run_mock`, `_mock_judge`, `gate_valid_rate`, `_expected_fps`, `load_done`, `_rows_from_outputs` | Stage 1 |
| `apply_judge.py` | `_process_file`, `_apply_steps`, `assert_judge_out_prompt`, `load_judge_index`, `_rename_resps_key`, `_compute_nmae` | Stage 2 |
| `judge_stats.py` | `decompose_failures`, `judge_validity`, `step_extraction_coverage`, `length_stratification` | Stage 4 metrics |
| `summarize_judge_task.py` | — | Stage 4 report |
| `reparse_judge_out.py` | `_marker_path` | CPU re-parse of stored raw (§9). On success it drops a dotfile marker `.<name>.reparsed` beside its `--out`; `test-sweep.sh` refuses to overwrite a merged judge-out whose marker is **at least as new as the file itself**, which is what stops a later sharded sweep silently reverting the re-parse. The marker is compared against the MERGED FILE, never the shards (a re-parse never touches a shard, so a shard anchor went silent exactly when a repair was in flight), and it is **self-clearing**: a legitimate merge rewrites the merged file, making it newer than the marker. To stand the guard down by hand, delete that dotfile. Writing `--out <merged>` in place is the guarded path; `--out <side>` + `mv` leaves the merged file unguarded |
| `setup_judge_env.sh`, `requirements*.txt` | `--judge`, `TORCH_INDEX_URL`, `PYTHON_BIN`, `requirements-cpu-stages.txt` | Builds the Stage-1 venv **per reader** and verifies it by ALLOCATING on the GPU, not by counting devices. One venv per reader is forced, not tidy: `requirements-gemma-4-31b.txt` pins `transformers==5.10.2` because Gemma-4's config declares `transformers_version 5.5.0.dev0` and no 4.x release can read it, while a vLLM declaring `transformers<5` cannot host that. A second reader with the opposite constraint could not share the venv. **Installation is two-phase for gemma-4-31b, and must be.** pip resolves ONE requirements file as ONE constraint set, so `vllm==0.19.0` beside `transformers==5.10.2` is not untidy — it is `ResolutionImpossible` and nothing installs (vllm 0.19 declares `transformers<5`, vLLM issue #39216; torch 2.10 pins `nvidia-nccl-cu12` with `==`). Overrides only take effect as a LATER pip invocation, where pip compares against the INSTALLED set, warns, and proceeds — exactly what `eval__gemma4.install_transformers_for_gemma4` does after `install_vllm`. Hence `post_requirements`: `requirements-gemma-4-31b-post.txt` holds every pin that contradicts phase 1, and the red "dependency resolver does not currently take into account..." notice on that pass is EXPECTED. The verification step (imports vllm, checks the transformers major, allocates on the GPU) is the real gate. test-11 reads both files and fails if a 5.x transformers pin ever moves back beside vllm. The requirements files, torch pin, expected transformers major and venv basename all come from `JUDGE_MODELS`, so this script hardcodes no version. **`requirements-cpu-stages.txt` is installed for EVERY reader**: the driver runs Stage 1 and Stages 2–4 under one `PYTHON`, and Stages 2–4 import `medvision_bm` (→ `datasets`, `nibabel`). Stage 1 needs neither, so an env without them builds cleanly, passes verification, sweeps for thirteen GPU-hours and dies on Stage 2's first line — which is why it is reader-independent (one file, not a copy per reader) and why `run_llm_parsing.sh` now probes the real `cal_metrics` import in its preflight, hard for `analyze`/`pilot` and as a warning for `full`/`smoke`. The pre-existing `resolve_medvision_ds` probe did NOT cover this: `medvision_ds` imports fine without `datasets`, so it passed while Stage 2 failed. |
| `judge_env.sh` | `JUDGE`, `JUDGE_KEY`, `JUDGE_SUFFIX`, `JUDGE_MODEL_HF`, `JUDGE_TP`, `judge_shard_devices` | **Sourced by BOTH `run_llm_parsing.sh` and `test-sweep.sh`, never executed.** The judge model string is stamped into judge-out and `load_done` compares provenance, so the two entry points resolving differently aborts a legitimate resume — this file makes the agreement structural rather than two blocks and a comment. It also owns `judge_shard_devices`, for the same reason: both entry points launch shards, and a disagreement about which GPUs a shard owns co-locates two engines and OOMs only after both have loaded. Reader selection is `eval`'d from `judge_config.py --shell`, never re-typed here — a bash copy of the registry is a second thing to keep aligned, and drift shows up as an aborted resume. An empty response from that query is checked explicitly, because a `PYTHON` that exits 0 printing nothing would otherwise surface as `unbound variable` from an unrelated line |
| `run_llm_parsing.sh` | `ACCEPT_FP`, `--fresh`, steps `prep/stage0/smoke/pilot/full/analyze` | single driver; re-roots to the repo, so any cwd works. `--fresh` additionally deletes every `llm-parsed*/` in `prep` — the one-command full re-judge. It deletes with `find Results/ …`: the **trailing slash is required**, because the repo root of a git worktree has `Results` as a SYMLINK and `find` (default `-P`) does not follow a symlink named as a starting point, so `find Results …` matches nothing and exits 0. The glob-based counter beside it *does* traverse the symlink, so the two disagreed silently |
| `test-sweep.sh` | GPU preflight, device-list resolution, shard/merge, `JUDGE_SUFFIX` in the out path | **Stage 1 runner — production, not a test.** `run_llm_parsing.sh` delegates the whole GPU sweep to it. Shard files are `<out>.n<NUM_SHARDS>.shard<S>`: the count is in the NAME so a changed `NUM_SHARDS` cannot be mistaken for a resumable stride (raising it used to merge overlapping strides and duplicate rows). The merge refuses to overwrite a merged file holding qids no shard has — that file is where `--redo_invalid` and `reparse_judge_out.py` write |
| `config-{TL,AD,detect}-CoT.yaml` | — | roster copies; `DEFAULT_ROSTER_YAML` resolves them relative to `judge_config.py`. 19 models since 2026-08-19: the 18 paper models plus the late-judged fullSFT checkpoint, whose view-campaign judge-out and queue rows were merged into the main trees the same day (same prompt stamps, so no re-judge; pre-merge artifacts in `Results/_archive_llm-parsing_2026-08-19/`). Its Detection eval lacks `BCV15_BoxCoordinate_Task01_Axial.jsonl` (27 files, not 28; 22,071 rows, not 23,071) — real, not a resolution bug |
| `config-{TL,detect}-CoT-{plane,task}OOD.yaml` | — | the OOD splits' 3-model rosters (baseline / SFT / SFT-RFT), copies of the radar-plot configs in `script/visualization/`. Model keys must equal the split trees' directory names — the TL task-OOD copy once carried the plane-OOD's `-fullSFT-` name for a directory called `Qwen2.5VL-7B-SFT-ds1.1.1`, which Stage 0's roster gate refuses. Each split is one `run_llm_parsing.sh` invocation with `TASK_DIR_<task>` + `ROSTER_YAML_<task>` set (README: "judge other Results trees"); those variables are honoured by `run_llm_parsing.sh`, `test-sweep.sh` **and** `unit-test/llm-parsing/test-8.py` — all three resolve the same tree or the invariants gate scans the wrong one. Run the splits sequentially (each sweep wants every visible GPU) with steps `stage0 full analyze`: no `prep` (on a resume it would archive the sweep being finished), no `smoke` (the in-run `--min_valid_rate` gate still guards every full run) |

There is exactly **one** pipeline driver. An older per-task variant
(`run_judge_analysis.sh`, Stages 0–4 for a single `TASK=`) was deleted
2026-08-12: it duplicated the stage wiring in `run_llm_parsing.sh`'s
`run_analysis`, and a second copy of that wiring is how the two drift into
disagreeing about which flags Stage 3 needs. Two thin wrappers were retired
2026-08-19, after their campaigns finished: `run_llm_parsing_ood.sh` (the four
OOD splits) and `run_llm_parsing_fullsft.sh` + `config-fullSFT-CoT.yaml` (the
late fullSFT model, judged through a symlink **view** directory and then merged
into the main trees). Neither contained stage wiring — only a tree→roster table
and a loop setting `TASK_DIR_<task>`/`ROSTER_YAML_<task>` around
`run_llm_parsing.sh` — and the invocations they wrapped are documented in
README.md. Copies live in `Results/_archive_llm-parsing_2026-08-19/retired-scripts/`.

Everything the pipeline needs is in this one directory. The only paths that reach
outside are repo-root-relative *data* paths (`Results/`, `Data/`,
`unit-test/llm-parsing/`), which is why both shell entry points re-root themselves
to the repository before doing anything.

---

## 3. DATA FLOW + ON-DISK SCHEMAS

```
parsed/*.jsonl ──Stage0──> judge-queue_{task}[_limitN].jsonl ──Stage1──> judge-out_{task}[_limitN].jsonl
                             judge-baseline_{task}[_limitN].json                  │
                                                                                  ▼
parsed/*.jsonl ─────────────────────────Stage2 (apply_judge)────────────> llm-parsed[-limitN]/*.jsonl
                                                                                  │
                                                     Stage3 existing summarizers ─┤─> summary_*__llm-parsed*.txt
                                                     Stage4 summarize_judge_task ─┴─> judge report + summary_metrics_judge_Task.json
```

Stage residency: 0 CPU · 1 GPU · 2 CPU · 3 CPU · 4 CPU.

### 3.1 queue row (Stage 0 → Stage 1)

```
{qid, task_type, model, file, dataset, doc_id, step_key,
 regex_pred, response_chars, was_windowed, response, cache_key, prompt_fp}
```
- `qid = content_hash(model, basename, doc_id)` — resume key, **no prompt component**.
- `cache_key = content_hash(response, full_fingerprint)` — dedup key, prompt-sensitive.
- `step_key ∈ {"TL","AD:distance","AD:angle",None}` via `step_spec_key`.
- `regex_pred` is **written and never read downstream** (see §4 R2). Diagnostic only.

### 3.2 judge-out row (Stage 1 → Stage 2)

```
{qid, cache_key, prompt_fp, doc_id, file, model, task_type,
 judge_model, judge_status ∈ {ok,invalid}, judge_reason,
 final_answer:{status,span,values}, steps?:[{index,status,span,values}],
 raw?, raw_len?}
```
- flat, **no `obj` wrapper** — `final_answer` is top level.
- `raw` persisted in full on invalid rows by default (`--keep_raw`); truncated raw
  is refused by `reparse_judge_out.py` (`len(raw) < raw_len`).
- duplicates per `doc_id` resolve **last-wins** (`load_judge_index`).

### 3.3 llm-parsed record (Stage 2 output)

Source `parsed/` record, mutated:
- `filtered_resps` **removed**; `LLM_filtered_resps=[pred]` occupies its slot
  (`_rename_resps_key` preserves key order).
- every derived metric recomputed from `pred` via `parse_utils.cal_metrics`:
  TL/AD → `avgMAE, avgMRE, SuccessRate, nMAE?`; drops legacy `MAE`/`MRE`.
  Detection → `avgMAE, avgIoU, F1, Precision, Recall, SuccessRate`; drops `avgMRE`.
- added: `LLM_judge_answer_mode`, `LLM_judge_SR{success}`,
  `LLM_judge{reason, strict_pred, judge_pred, judge_span, judge_model, verify_tier?}`,
  `LLM_judge_steps?` (TL/AD). `judge_model` is provenance copied from the judge-out
  row — `test-8` fails on `"mock"`; records written before 2026-08-12 carry `null`.
- `nMAE` tiers: canonical `_compute_physical_diagonal` → invert strict `MAE/nMAE`
  → omit and let the summarizer compute it. **Tier 2 must be handed the record as
  the STRICT pipeline left it** — `_process_file` snapshots `avgMAE`/`nMAE` into
  `strict_metrics` before overwriting them. Until 2026-08-12 it was passed the
  post-overwrite record, which makes tier 2 an algebraic no-op: `diagonal =
  mae_judge / nmae_stale`, hence `mae_judge / diagonal == nmae_stale`.

  **Measured impact: none.** Tier 2 is unreachable on exactly the rows it would
  corrupt. `_diagonal_from_strict` needs a *finite* strict MAE, and on a
  `conclusion_off_format` row the strict parser failed by definition, so
  `avgMAE.MAE` is NaN — verified on all 11,587 judge-recovered TL+AD rows, 0 with
  a finite strict MAE. Those rows are served by tier 1; if tier 1 were
  unavailable they would fall to tier 3 and have `nMAE` *dropped*, never
  falsified. On a strict-SUCCESS row the written prediction is the strict one, so
  `mae_judge == mae_strict` and the no-op returns the value that was already
  correct. The fix is therefore a latent-bug fix: it makes tier 2 do what its
  docstring says, and changes no number in this corpus.

  Note for future auditors: an `nMAE` in `llm-parsed` that equals the `parsed/`
  value is **not** evidence of this bug. `parse_outputs` writes `nMAE` only when
  absent, so a strict-fail record keeps the EVAL-TIME parser's `nMAE`; when the
  judge recovers the same number that parser found, tier 1 legitimately
  reproduces it exactly.

---

## 4. RELATIONSHIP TO THE STRICT REGEX PARSER

Five distinct couplings. Confusing them is the main way this design is
misread.

- **R1 · priority (not fallback).** `decide_answer` returns `strict_pred`
  unconditionally whenever it is non-empty, *before* any judge branch. The judge
  is the fallback for the regex; the regex is never a fallback for the judge.
  Both failing ⇒ `pred=""`, no third source.
- **R2 · prompt isolation.** The judge is never shown `regex_pred`, the target,
  or the image. `run_judge_vllm` reads only `response`, `task_type`, `step_key`,
  `qid`, `cache_key`, `prompt_fp` from the queue row.
- **R3 · transcription.** Every digit that reaches `LLM_filtered_resps` is
  produced by `judge_io.NUM_RE`, byte-identical to `parse_utils._NUM_RE` — from
  the judge's span (tiers exact/normalized) or from the response
  (value_anchor). The judge's own float is only ever a *matching key*.
- **R4 · drift gate.** Stage 0 replays `extract_last_k_nums_within_answer_tag`
  (D1, a deliberate duplicate of `parse_utils`) on every row and compares to the
  stored `filtered_resps`. `n_stored_mismatch > 0` ⇒ `[GATE FAIL]`, exit 1.
  Current: 0 / 496,296. `test-1` asserts duplicate ≡ original on the real corpus.
- **R5 · reference set.** `judge_stats.judge_validity` measures judge–regex
  agreement over the regex's own successes — 33,854 TL / 27,355 AD / 387,910
  Detection rows, a large and near-perfect reference. Measured agreement:
  99.78% / 99.96% / 99.94%. This is why 100% of the corpus is judged, not only
  the failures — the agreement measurement is free.

Duplicated-from-`medvision_bm` primitives that MUST stay byte-equivalent:
D1 `extract_last_k_nums_within_answer_tag` + `NUM_RE` (`parse_utils`),
D2 `extract_response` and `iter_records` order/limit rule (`parse_outputs`).
Stage 2 is the deliberate exception that *imports* `cal_metrics` — scoring is
never re-implemented.

Exception: `run_judge_vllm.py --mock` / `_mock_judge` substitutes a regex
extractor **for the judge** on CPU boxes. Not a production path, and isolated
from one by three independent mechanisms (all added 2026-08-12, after a review
found none of them existed):

- **separate output path** — `test-sweep.sh` writes `judge-out_<T>.MOCK.jsonl`.
  It previously shared the production filename.
- **provenance in the resume** — `load_done` treats a mock row as stale to a real
  run and vice versa. Resume keys on `(prompt_fp, qid)`, and a mock row carries
  the queue's own `prompt_fp`, so a single mock sweep used to mark every qid done:
  the next real run made **zero model calls**, exited 0, and shipped the regex
  stand-in's verdicts to Stages 2–4.
- **provenance in the record** — `apply_judge` copies `judge_model` into
  `LLM_judge`, and `test-8` fails on any record stamped `"mock"`. This is what
  §14 claimed all along; the check did not exist until now.

---

## 5. THE DECISION TABLE

Single source: `judge_decision.DECISION_TABLE_DOC`; consumed by Stage 2,
Stage 4 and `test-7`. 2 strict outcomes × 5 judge outcomes = 10 rows.

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

- `verify_span` is invoked **only** when `judge_status=="ok"` and
  `final_answer.status=="present"`; otherwise `verified is None`.
- `is_success(mode) ⇔ mode ∈ SUCCESS_MODES ⇔ LLM_filtered_resps[0] != ""` —
  three spellings of one fact, cross-asserted in `test-7`/`test-8`.
- `undetermined` = "judge unusable AND regex failed" ⇒ a recovery not attempted
  ⇒ reported ΔSR is a **lower bound**, and the `undetermined` count is the
  pipeline's own quality metric.
- `LLM_judge.judge_pred` stores the judge's *own* verified extraction, NOT the
  decided value. Under R1 the decided value equals `strict_pred` on every
  reference row, so agreement measured against it would be 100% by construction.
- `_final_answer_status` is defensive on purpose: v1 emitted bare strings/floats/
  lists for `final_answer`, and `.get` on those raises mid-sweep.

---

## 6. SPAN VERIFICATION (`verify_span`)

Contract: `verify_span(response, span, values, expected_arity) -> {ok, reason,
tier?, numbers, pred}`.

Pre-checks (all `ok=False`): `empty_span`, `no_values`,
`arity_mismatch:{n}!={k}`, `non_numeric`.

Matching is **contiguous-run**, not exact-list: a span carries ≥12 chars of
context by instruction, and context contains numbers
(`<step-1-answer>0.51,…</step-1-answer>` yields six via `NUM_RE`). The run is
searched **from the end**, matching the strict parser's last-k rule. Tolerance
`_REL_TOL=1e-9` absorbs JSON float round-trip only (0.51 vs 0.52 fails).

| tier | containment test | numbers transcribed from | I4 check |
|---|---|---|---|
| `exact` | `collapse_ws(span) ⊆ collapse_ws(response)` | the span | implicit: whitespace is collapsed, never deleted, so tokens cannot fuse |
| `normalized` | any `_span_variants(span)` under `_norm_key` (casefold + whitespace **deleted** + ctrl-char inversion) ⊆ same key of response | the repaired variant | **explicit** `_values_in_response` |
| `value_anchor` | values form a contiguous `NUM_RE` run in the **response itself** | the response | implicit: the test *is* the check |

The explicit guard on tier 2 was added 2026-08-12 after review found it could
violate I4. `_norm_key` deletes whitespace on both sides, so a judge quote that
dropped a separator between two numeric tokens let `NUM_RE` read a **merged**
number out of the variant: response `…2\n0.5…` quoted as `"the values are20.5and"`
transcribed `20.5`, a number the model never wrote. `_values_in_response` now
requires every transcribed number to appear, in order, in the response's own
`NUM_RE` run — reason `normalized_values_not_in_response`.

`_span_variants` strips only things the judge *added* — a wrapping quote pair, a
trailing `</tag>` it closed on a truncated response — or restores what JSON
decoding corrupted (`\x08oxed` → `\boxed`). None can introduce a number.

Guards:
- normalized tier **breaks** (does not fall through) when a variant locates but
  its values do not match: that is a real disagreement, not a location failure.
- `all_zero_value_anchor`: values that are all zero are rejected at tier 3. An
  unlocatable span carrying `[0.0]*k` is the historical prompt-skeleton echo
  signature (§8), and any incidental `0` in the response would otherwise anchor
  it. Tiers 1–2 still accept genuinely quoted zeros (`"Distance = 0."`).
- terminal failure reason `span_not_found`.

**I4 (invariant, all tiers): a value the model never wrote cannot be scored.**
Only the span's role as location *proof* relaxes across tiers, and the accepting
tier is recorded per record in `LLM_judge.verify_tier`.

Why tiered (measured 2026-08-07): of 3,877 strict-fail `span_not_found` rows only
19 (0.49%) were hallucination; the rest were quoting habits — judge-added closing
tag 35%, quotation marks 22%, whitespace deletion 16%, re-punctuation 12%.
Corpus tier distribution (re-counted from `llm-parsed/` 2026-08-12, i.e. after the
§8 fix): 431,254 `exact` / 5,731 `normalized` / **44,145** `value_anchor` — the
pre-fix count was 44,208, and the 63-row difference is exactly the echo leak §8
closed. Per task, `value_anchor` supplies only 3.7% (TL) / 5.5% (AD) / 4.7% (Det)
of *recoveries*, so tier 3 is a minor contributor that is cheap to tighten but not
free: a blanket ban on arity-1 anchoring would cost 333 real AD recoveries to
prevent 59 echoes.

Job B steps run the same verifier at `spec["n_values"]` arity
(`_apply_steps`), always emitting one entry per prescribed step in index order.
Step output is **persisted, never scored**.

---

## 7. DECODE PATH (`judge_decode.py`)

1. **Harmony split** (`split_final_channel`). vLLM 0.11's offline path decodes
   control tokens into bare words: the channel switch arrives as the literal
   `assistantfinal` (`_PLAIN_FINAL_MARKER`; 1,972 of 2,000 production rows) while
   the documented `<|channel|>final<|message|>` appears in 0. Both handled,
   **last** marker wins, text with neither passes through.
2. **Candidate scan** (`iter_json_object_candidates`, driven by
   `parse_judge_json`). Walks *every* balanced `{...}` group — one scan per
   opening brace, bounded by `_MAX_CANDIDATE_STARTS=64` /
   `_MAX_SCAN_CHARS=400_000` — preferring the first candidate
   containing `final_answer`, falling back to the first dict that parses.
   Committing to the first balanced group was the v2 pipeline's largest
   self-inflicted loss: the analysis channel is prose full of LaTeX braces, and
   `{mm}` / `{217.07, 110.01}` are real production extractions that shadowed the
   object.
3. **Escape repairs, ordered deliberately.**
   - `preescape_latex_escapes` — **unconditional, before** parsing. `\b`/`\f` are
     *legal* JSON escapes, so verbatim `\boxed{…}` parses **successfully** into a
     silently corrupted span. A failure-keyed repair never sees it. `\t\r\n` left
     alone (spans quoting multi-line responses need real newlines).
   - `repair_json_escapes` — **only after** a plain parse fails; doubles genuinely
     invalid escapes (`\[`, lone trailing `\`, short `\uXX`). Cannot change the
     reading of a blob that decodes on its own.
4. **`validate_judge_obj`** — shape only, never plausibility. Rejects v1 statuses
   (`absent`) outright so a stale v1 file cannot be reinterpreted. Retains a
   permanent exemption: an all-zero `values` array beside a non-`present` status
   is accepted (historical raw must stay reparsable — §8).

Failure at any step ⇒ `judge_status="invalid"` + reason; downstream that is
decision-table row 4 or 9.

---

## 8. THE SKELETON-ECHO CLASS OF BUG (closed 2026-08-08)

Root cause: `_output_skeleton` rendered `values` as `[0.0] * k` — the only
placeholder that is also a *legal value* — and the judge echoed it back. Live
Detection judge-out carries 3,440 clean `no_conclusion` verdicts beside
`[0.0,0.0,0.0,0.0]` (3,428 measured at repair time).

Containment, three layers, all now in place:
- **validator** — all-zero + non-`present` accepted (the judge is quoting us, not
  contradicting itself); all-zero + `present` still fails. Permanent.
- **span verification** — an echo beside `present` had to survive `verify_span`;
  tiers 1–2 require the zeros inside a located quote.
- **the leak, fixed** — tier 3 anchored an echoed `[0.0]` to any incidental `0`
  and scored `pred=0`. Measured 63 rows corpus-wide (59 arity-1 AD / 1 TL /
  3 Detection = 0.014% of judge-scored rows). Now `all_zero_value_anchor` (§6).
  Retroactive re-apply produced exactly 63 accepted→undetermined flips, 19
  cosmetic reason relabels, 328 unscored step updates, 0 unclassified; per-model
  SR moved ≤0.1pp.

Root cause removed: the skeleton now renders `"<number>"` slot markers, which are
not legal values. **Consequence: `prompt_fingerprint` moved.** The next Stage 0
build mints new fps for all three tasks and `run_judge` will refuse the on-disk
@4096 queue stamps — expected, rebuild. Stages 2–4 on existing judge-out are
unaffected via `ACCEPT_FP`.

Generalisation for future edits: **no placeholder in the prompt skeleton may be a
legal value of the field it occupies.** `test-6` asserts this.

---

## 9. IDENTITY: THE FINGERPRINT LOCK

`prompt_fingerprint(task_type, step_key)` = deterministic JSON over
`{rendered messages for a NUL-delimited sentinel response, build_schema output,
[RESPONSE_WINDOW_TRIGGER, HEAD, TAIL], RESPONSE_ELISION_MARKER,
TASK_SPECS[task]["max_tokens"], JUDGE_REASONING_EFFORT}`.
`short_prompt_fp` = first 16 hex of its BLAKE2b hash.

Built by **rendering**, not by listing ingredients — the listing version shipped
three silent holes (`NO_STEPS_NOTE` absent ⇒ 415,278 stale Detection answers
would have been reused; `MIN_SPAN_CONTEXT_CHARS` substituted after the payload
was taken; `RESPONSE_ELISION_MARKER` absent entirely). Anything that cannot
appear in a rendered sentinel is named explicitly. There is deliberately **no
hand-maintained version constant** — it would go stale and then disagree with the
hash.

Lock semantics:
- queue rows and judge-out rows both carry the stamp;
  `run_judge_vllm._expected_fps` and `apply_judge.assert_judge_out_prompt`
  hard-abort on an unknown stamp.
- `apply_judge --accept_prompt_fp` (driver: `ACCEPT_FP`) whitelists historical
  stamps so a file legitimately holding two prompts' rows can be applied.
  Whitelisting is explicit so mixing is a decision, not an accident.

| stamp | task | budget | | stamp | task | budget |
|---|---|---|---|---|---|---|
| `09ee44a311e85670` | TL | @1024 | | `dd3c7fb2d50255db` | TL | @3072 |
| `02a90adc517baab6` | AD:dist | @1024 | | `5cfdd478b855e9e4` | AD:dist | @3072 |
| `a3fd2c4a87a9ce4e` | AD:angle | @1024 | | `56de1c8d5a7719b0` | AD:angle | @3072 |
| `49c9fbdcf069aef9` | Det | @256 | | `54e8d9ef545df4a6` | Det | @512 |

On-disk queues (pre-2026-08-08 code) stamp @4096: TL `8f7579e66a5ad982`,
AD:dist `d7acc730e39a248e`, AD:angle `7159a2e0c4b178dd`, Det `e730a9758384c648`.
These no longer match current code (§8).

---

## 10. BUDGET · REPAIR · REPRODUCIBILITY

**Budget.** `DEFAULT_JUDGE_MAX_TOKENS = 4096`, shared by all tasks. Per-task
"right-sizing" caused two silent truncation incidents — TL @1024 (8.70%
judge-invalid) and Detection @256 (88.9% of residual failures at *exactly* 256
completion tokens; measured requirement p99.5 = 327). A too-small budget never
fails loudly, it inflates judge-invalid. Headroom is nearly free (decode stops at
EOS). Stage 1 counts `finish_reason == "length"` and WARNs. Fits
`JUDGE_MAX_MODEL_LEN=8192` (worst prompt ~2.8K tokens ⇒ ~5.3K usable completion,
so anything above ~5K also needs `--max_model_len` raised).

**4096 is sufficient — verified 2026-08-12** by re-tokenising every stored
completion with the judge's own tokenizer (`raw` is kept on invalid rows, i.e.
exactly the population a bigger budget would have to rescue):

| stamp | cap | n with raw | p50 | p90 | p99 | at cap | of which loops | rescuable |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Det `54e8…` | 512 | 15,479 | 224 | 308 | 435 | 110 (0.7%) | 13% | 96 |
| TL `dd3c…` | 3072 | 214 | 673 | 3066 | 3066 | 30 (14%) | 73% | 8 |
| AD `56de…` | 3072 | 76 | 2952 | 3070 | 3070 | 37 (49%) | 92% | 3 |
| AD `5cfd…` | 3072 | 33 | 3066 | 3066 | 3075 | 24 (73%) | 92% | 2 |

Reading: what reaches a cap is overwhelmingly **degenerate repetition** — `0000…`,
`999…`, `It's 12? It's 12?…`, `1040320 * 0.58 =` repeated — which no budget fixes,
because the loop has no natural end. Across the whole 496,296-response corpus only
**~109 non-loop rows** ever sat at a cap, and every one of them was capped at 512
or 3072, not 4096. Detection's genuine requirement is p99 = 435 tokens, so 4096 is
~9× the working distribution. Raising above 4096 would buy nothing but longer
loops, and costs a full fingerprint rebuild (§9). **Keep 4096.**

Corollary: the production judge-outs on disk were produced at 256/512/1024/3072,
not 4096, so ~109 rows in them are budget-truncated. Re-judging to recover them is
a bad trade — it is 0.02% of the corpus against the flip rate documented below.

Method caveat: re-tokenising a *decoded* string does not exactly reproduce the
generation token count, so "at cap" uses a proportional threshold (≥0.97·cap), not
equality. Persisting `finish_reason` per row would make this measurable directly;
it is currently counted at run time but not written to the output row.

**Repair passes.**
- `--redo_invalid` is REQUIRED: without it every invalid row counts as done and
  the re-run makes zero model calls — a silent no-op.
- `--accept_prompt_fp` whitelists the old stamp (budget is in the fingerprint).
- Merge direction is load-bearing: repaired rows must be **APPENDED**.
  `cat new old > merged` puts stale rows last, and last-wins silently reinstates
  exactly the verdicts the repair replaced — well-formed file, every qid present,
  no error anywhere.
- A **decoder** fix is a CPU re-parse (`reparse_judge_out.py`), never a GPU pass.
  It prints the transition table and an all-zero echo census, and refuses to
  write if any row moves ok→invalid or any ok row's values change.

**Reproducibility.** Judge output is NOT reproducible run to run — *on the same
GPU*, not merely across generations. Controlled experiment (all 15,479 invalid
Detection rows, same day, same H100s): re-judging at the **unchanged** 256 budget
recovered 63.0%; at 512 it recovered 98.1% ⇒ the budget's attributable effect is
+35.1 points. Independent on-disk confirmation: judge-out uses append semantics,
so TL carries **1,306 rows judged twice under the same `prompt_fp` and the same
3072 budget**, and only **12.8%** returned an identical verdict (1,092 changed
`judge_status`).

*Root cause (analysed 2026-08-12).* It is **not** sampling randomness — that
explanation is arithmetically impossible here:

- `JUDGE_TEMPERATURE = 0.0`, `JUDGE_TOP_P = 1.0` ⇒ greedy argmax. There is no
  sampling draw to randomise, and `JUDGE_SEED = 1024` is inert at temperature 0
  (it seeds a sampler that never samples). Any doc or comment attributing flips to
  "RNG" is wrong.

What remains is **numerical** non-determinism, amplified by architecture:

1. **MoE routing was a discrete amplifier** (dominant *for the reader retired on
   2026-08-17*, which was `num_local_experts=32`, `num_experts_per_tok=4`).
   Routing is a top-k over router logits, so a ~1e-6 perturbation near a tie flips
   which expert runs, and the FFN output changes by O(1), not by O(1e-6). **This
   term does not apply to the current reader**: `gemma-4-31B-it` is dense, so it
   should drift materially less. That is a prediction from architecture, NOT a
   measurement — the flip rates quoted in this section were all measured on the
   retired reader and must not be restated as properties of the current one.
2. **No batch-invariant kernels.** vLLM 0.11.0 as installed ships no
   batch-invariance module and no such env switch (checked). Reduction order in
   matmul/attention depends on batch shape, and V1 chunked prefill + CUDA-graph
   capture make the batch shape depend on what else was in flight — i.e. on shard
   layout, chunk boundaries and arrival order, none of which are pinned.
3. **`enable_prefix_caching=True`.** Whether a row's shared system-prefix KV is
   recomputed or reused depends on batch order and eviction, and the two paths are
   not bitwise equal — so identical input can take numerically different paths.

Why it *looks* catastrophic: the 63.0% and 12.8% figures are both measured on rows
**selected for having failed**. Selecting on a noisy binary outcome and re-running
regresses to the mean, so those numbers are an upper bound on instability, not an
estimate of it. The unbiased quantity — flip rate on a *random* sample — has never
been measured and needs a GPU. Note also that the observable is concentrated at a
truncation boundary: a row whose natural length is near the cap flips
valid/invalid on a few tokens of drift, which is why the effect was largest at the
under-sized 256 budget and why a generous budget (above) suppresses the symptom
without touching the cause.

Mitigations, in order of cost: keep a generous budget; pin shard count and
`--chunk_rows` when an A/B is intended; eager execution and prefix caching off for
a determinism-critical arm — slower, not a proof of bitwise equality, and **not
exposed by any flag**: `enable_prefix_caching=True` is hardcoded in
`run_judge_vllm.run_gpu` and there is no `--enforce_eager`, so this one means
editing that function rather than passing an option; accept
that bitwise reproducibility needs batch-invariant kernels this vLLM does not
have. Operationally unchanged: never compare judge-invalid rates across pods, and
never credit a re-run's recovery to a code change without a same-raw A/B
(`reparse_judge_out.py`) or a same-day control arm. (A 2,000-row roster-ordered
probe suggested 82.9% — roster order is model-biased; do not quote it.)

**Hardware.** `gemma-4-31B-it` ships plain bf16, ~62 GB of weights, so there is no
checkpoint preparation step and no per-pod dtype choice — vLLM fetches it on first
load. It does not fit one 80 GB card beside a usable KV cache, hence
`tensor_parallel = 2`; see **GPU topology** in §16. Data parallelism across shards
remains the efficient axis, with TP raised only to fit the weights.

---

## 11. EDIT HAZARDS

| edit | forced consequence |
|---|---|
| any text in `SYSTEM_PROMPT` / `USER_TEMPLATE` / `_output_skeleton` / `STEP_SPECS.what` | fp moves ⇒ Stage 0 rebuild (~1.1 GB) ⇒ every `cache_key` changes ⇒ old queues abort |
| `DEFAULT_JUDGE_MAX_TOKENS` or any `TASK_SPECS[*]["max_tokens"]` | same as above (budget is in the fp) |
| `RESPONSE_WINDOW_*` / `RESPONSE_ELISION_MARKER` / `JUDGE_REASONING_EFFORT` | same as above |
| `TASK_SPECS[*]["arity"]` | must equal the strict parser's `k`, else coverage becomes incomparable (`test-4`) |
| new analysis output written into `parsed/` | add its stem to `EXCLUDED_JSONL_STEMS` **and** to every summarizer's filter, or it is globbed as a sample file |
| new key in `llm-parsed` records | check `test-8` key-order/identity assertions |
| relaxing `verify_span` | I4 must still hold; add a fixture to `test-2` |
| touching `judge_io` D1/D2 duplicates | `test-1`/`test-5` parity, and Stage 0's R4 gate, must still pass |
| adding a `final_answer.status` value | v1-rejection logic in `validate_judge_obj` and `ANSWER_MODES` mapping both change; `test-3`/`test-7` |

Never: re-implement a metric in this directory (import `cal_metrics`); keep
`filtered_resps` alongside `LLM_filtered_resps`; write judge output into
`parsed/`; treat `--limit` output as a full run (the limit is in the *directory*
name for exactly this reason).

---

## 12. INVARIANTS + GATES

| id | invariant | enforced at | test |
|---|---|---|---|
| I1 | strictly additive: `parsed/` never written | Stage 2 output path | — |
| I2 | strict-first: `LLM_filtered_resps == filtered_resps` on every strict-success row, every model | `decide_answer` R1 | `test-7`, `test-8` |
| I3 | a judge failure costs a recovery, never corrupts a published value | corollary of I2 | `test-8` |
| I4 | no fabricated value can be scored, at any tier | `verify_span` | `test-2` |
| I5 | one decision table, three consumers | `judge_decision` | `test-7` |
| I6 | no new metric family — judge and strict columns share the summarizer code path | Stage 3 + `cal_metrics` import | Stage-3 byte-identity gate |
| I7 | prompt identity is a hash of the real prompt, never a constant | `prompt_fingerprint` | `test-6` |
| I8 | no skeleton placeholder is a legal value | `_output_skeleton` | `test-6` |
| I9 | exactly one `llm-parsed` record per source record; duplicates exist ONLY in judge-out and resolve last-wins | `load_judge_index` + one output row per `iter_records` row | `test-8` |

I9 in detail, because resumability makes it non-obvious. Stage 1 opens its output
in **append** mode, so duplicate `qid`s accumulate there by design: a plain resume
skips rows already written (`load_done`), while `--redo_invalid` deliberately
leaves invalid rows outstanding so a repair pass appends a second, better verdict.
Shard files are the authoritative resume state and the merged file is re-derived
(`cat` of an explicit shard list into a temp, then rename), so merging never
accumulates. Stage 2 then collapses everything: `load_judge_index` keys by
`doc_id` last-wins, and `_process_file` writes exactly one record per source
record. Verified over the whole corpus 2026-08-12: **0 duplicated `doc_id`s in
774 files, 0 row-count mismatches against `parsed/`, 0 mode/SR/prediction
inconsistencies, 0 orphan files, and llm-parsed coverage exactly equal to the
18-model roster on every task** (the 81 other model directories under `Results/`
are the superseded checkpoints roster gating exists to skip).

| gate | stage | behaviour |
|---|---|---|
| replayed parser ≡ stored `filtered_resps` | 0 | abort; 496,296/496,296 |
| roster counts 46,379 / 39,140 / 437,349 (19 models since 2026-08-19; the paper's 18-model totals were 43,938 / 37,080 / 415,278) | 0 | abort on mismatch (skipped under `--limit`, and for a non-default `--task_dir` — the counts describe the main trees only, so an OOD split prints `[gate n/a]` instead) |
| every roster model has `parsed/` (no glob fallback) | 0 | abort — a fallback-discovered model once burned 100% of its GPU time |
| `gate_valid_rate` ≥ `--min_valid_rate` (0.95) over first 200 buffered rows | 1 | abort **writing nothing**; repair queues must pass a low/0 rate |
| queue/output `prompt_fp` known | 1, 2 | abort |
| `--parsed_dirname parsed` regenerates published reports byte-identically | 3 | verified TL/AD/Detection |
| record invariants on real output | test-8 | strict-superset identity, key order, mode↔SR consistency, mock detection (`LLM_judge.judge_model != "mock"`) |
| mock vs real provenance on resume | 1 | `load_done` counts a cross-provenance row stale ⇒ abort |
| merge cannot destroy repair-pass rows | 1 | refuse if the merged file holds a qid no shard has |
| shard count matches the shard files | 1 | count is in the filename, so a changed `NUM_SHARDS` is caught as stale |

Chunked fsynced writes (`--chunk_rows`, default 2000): `llm.chat` returns only
when every prompt in the call finishes, so an unchunked group persists nothing
until the last row (a pod death once cost 2,000 judged rows). One model call per
distinct `cache_key`; repeats filled from cache and marked `+cached`.

---

## 13. MEASURED SNAPSHOT (18-model roster, 496,296 responses)

Read from the current `summary_judge_task__llm-parsed.txt` reports and the Stage 0
baselines on 2026-08-12 — i.e. **after** the §8 echo fix, which is why AD moved
0.1pp from the 2026-08-07 figures.

| | TL | AD | Detection |
|---|--:|--:|--:|
| responses | 43,938 | 37,080 | 415,278 |
| strict-parsed | 33,854 | 27,355 | 387,910 |
| SR strict regex | 77.0% | 73.8% | 93.4% |
| SR format-robust | 89.7% | 90.1% | 98.4% |
| ΔSR | +12.6 | +16.3 | +5.0 |
| recovered (`conclusion_off_format`) | 5,551 | 6,036 | 20,711 |
| `no_conclusion` | 3,840 | 3,516 | 5,572 |
| `undetermined` | 693 | 173 | 1,085 |
| judge-invalid (residual, last-wins) | 214 | 109 | 301 |
| judge–regex agreement on regex successes | 99.78% | 99.96% | 99.94% |

The AD deltas from the pre-fix snapshot (SR 90.2→90.1, `undetermined` 114→173) are
exactly the 59 AD echo rows §8 reclassified; TL's 692→693 is its 1 row.

Extreme case: Llama-3.2-11B TL 14.5% strict → 97.9% format-robust. Total
`undetermined` fell 10,653 → 1,891 across the 2026-08-07 fixes with zero
regressions on a record-level diff of all 496,296 rows (every mode transition
exits `undetermined`; no resolved answer changed), then rose to **1,951** when §8
reclassified the 63 echo rows — the one intended increase.

Attribution caveat for low-SR models: `conclusion_off_format` ⇒ format broke;
`no_conclusion` + responses piled at the token cap ⇒ budget broke the model;
`undetermined` ⇒ the judge failed. Discriminating truncation from clean stop
requires re-tokenising with the model's own tokenizer — character lengths spread
2.5–4 chars/token and hide the wall.

---

## 14. TESTS

`unit-test/llm-parsing/test-{1..11}.py` — standalone scripts, bare asserts, run
from the repo root with the judge interpreter and
`PYTHONPATH=src:<medvision_ds-src>`. All but `test-8` run without a GPU or a
Results tree; `test-8` reads the real records on disk.

| test | proves |
|---|---|
| 1 | replayed strict parser ≡ `parse_utils`, incl. real corpus |
| 2 | `verify_span`: genuine spans accepted, fabrication rejected, all three tiers, echo rejection |
| 3 | harmony split, candidate selection, preescape/repair order, validator shapes, v1 `absent` rejected |
| 4 | naming/arity invariants; pilot cannot collide with a full run; judge `k` ≡ strict `k` |
| 5 | record ordering and `--limit` ≡ `parse_outputs.py` |
| 6 | skeleton ≡ enforced schema; fingerprint completeness; no placeholder is a legal value |
| 7 | the 10 decision-table rows one-for-one, plus v1 degenerate shapes |
| 8 | invariants over the REAL `llm-parsed/` records on disk; flags mock output |
| 9 | the shell entry points actually RUN, in both MOCK modes, under `set -euo pipefail` |
| 11 | the judge registry (§16): the registered reader's filenames have not moved, **no reader has an empty `out_suffix`** (so every driver glob is anchored and the retired reader's archives are unreachable), no two readers share an output path, key resolution never guesses, `assert_single_judge_model` catches a mixed file, and the prompt fingerprint is judge-independent |

`test-9` exists because `bash -n` proves syntax, not exit-status semantics. A
`.MOCK` infix written as `OUT="…$([ "$MOCK" = 1 ] && echo .MOCK).jsonl"` assigns
the right string and parses cleanly, but a lone assignment inherits its last
command substitution's status — so with MOCK unset it returned 1 and `set -e`
killed Stage 1 silently on **every production sweep**, while `MOCK=1` passed. The
only mode testable without a GPU was the only mode that worked. `test-9` drives
the real script in a throwaway repo with a stub interpreter and asserts it reaches
the banner that names its output file; verified to fail on the reintroduced bug.

---

## 15b. ROSTER SCOPING (all five stages)

A results tree holds far more model directories than any one study reports on.
Measured 2026-08-15: **57** dirs under `MedVision-TL-v2-CoT` / 51 AD / 31 Detection
against an **18**-model roster — superseded `_bugfix-*` variants, training
checkpoints (`MedVision-V0-7B-curriculum-s*`), baselines (`random_detection`).

| stage | scoped by |
|---|---|
| 0 `build_judge_queue` | `for model in load_roster(...)` |
| 1 `run_judge_vllm` | reads the queue — inherits |
| 2 `apply_judge` | `for model in roster` |
| 3 / 3b `summarize_*_task` | **`--models`**, passed by the driver |
| 4 `summarize_judge_task` | `for model in roster` |

Stages 3/3b are the only ones with no roster concept of their own — they walk
`get_subfolders(task_dir)`. Until 2026-08-15 that meant:

- **stage3** landed on the roster *by accident*: it reads `llm-parsed<reader>/`,
  which only the (roster-scoped) Stage 2 creates, so `--skip_model_wo_parsed_files`
  removed the rest. Correct, but a property of the directory layout, not a rule.
- **stage3b** had no filter at all: it reads `parsed/`, which every directory has,
  so it processed all 57/51/30. Two consequences — the strict report the README
  tells you to diff carried 39 extra TL models the judge report has never heard of,
  and one malformed record in any non-study directory aborted the whole stage,
  after the GPU sweep.

Both now take `--models <roster>`. The filter lives inside
`parse_utils.get_subfolders(task_dir, models=None)`, **not** at the call sites: each
summarizer enumerates model dirs twice — once to process, once to aggregate — and
two filters that disagreed would produce a report covering a different set than was
processed, with nothing saying so. `models=None` is the default, so every other
caller of these summarizers is unaffected.

---

## 15. KNOWN LIMITATIONS

- `undetermined` ≠ 0 (1,951 = 0.39% of the corpus): dominated by degenerate
  repetition loops (~0.3% of any queue) plus `missing_values` shapes. Arity
  mismatches (a box given as two points) are a transcription question deliberately
  not guessed at — they are the single largest span-rejection reason on Detection
  (754 rows of `arity_mismatch:2!=4`).
- Job B step extractions persisted but unscored; process/equation-accuracy
  integration out of scope.
- `run_llm_parsing.sh` drives repair only from Stage 2 (`analyze`); Stage 1
  repair runs are launched by hand on purpose — they rewrite history.
- Judge-invalid rates comparable only within one pod / GPU generation (§10).
- Structured decoding is attempted but not depended on (vLLM 0.11 renamed
  `guided_decoding`→`structured_outputs`; xgrammar's per-token bitmask over the
  free-form `span` serialises decode behind one CPU core). Valid JSON comes from
  the shown skeleton; `--min_valid_rate` is the real guard.


---

## 16. THE JUDGE REGISTRY (`JUDGE_MODELS`)

The reader is resolved through `judge_config.JUDGE_MODELS`; `--judge KEY` /
`JUDGE=KEY` selects one and `run_llm_parsing.sh --judges` lists them. Registered:

| key | checkpoint | weights | chat kwargs | out_suffix | tensor_parallel | env |
|---|---|---|---|---|--:|---|
| `gemma-4-31b` *(default, only)* | `google/gemma-4-31B-it` | bf16 dense, ~62 GB | none | `_gemma-4-31b` | 2 | vllm 0.19.0, **then** transformers 5.10.2 in a 2nd pip pass |

**An earlier reader was retired on 2026-08-17.** It shipped quantized weights and
carried an empty `out_suffix`, and removing it took with it: the weight-conversion
step and its script, the `model` driver step, `JUDGE_MODEL_DIR` / `JUDGE_DEQUANT` /
`JUDGE_CACHE_BASENAME`, the per-pod dtype switch in `judge_env.sh`, the
marker-based provenance normalisation in `_model_key`, and test-10. Its judge-out
archives (322 MB under the unsuffixed names `judge-out_<task>.jsonl.v1`) were
deleted the same day, once it was confirmed that no `llm-parsed/` directory and no
published report still depended on them.

**Why a registry for one reader.** The judge is a measurement instrument, and the
claim it supports — "N% of apparent failure was formatting" — is only credible if
it does not turn on one model's idiosyncrasies. A second reader answering the SAME
queues under the SAME prompt with the SAME span verification and the SAME metrics
is the cheapest available cross-check, because the two reports would then differ
*only* by the reader. Keeping the table is what makes that an entry plus a
requirements file rather than a hunt through five modules.

Four things make a reader swap safe, and three of them fail silently if broken.

**1. The registered reader's names do not move.** `out_suffix` produces
`judge-out_<task>_gemma-4-31b.jsonl` and `llm-parsed_gemma-4-31b/`, which is what
the 109 output directories on disk are called. Changing it orphans the corpus in
one edit — Stage 2 would write a new tree beside the old one and Stage 3 would
report on whichever it was pointed at, with no error anywhere. Pinned by test-11.

**2. Two readers never share an output path — because they cannot be told
apart afterwards.** `apply_judge.load_judge_index` keys rows by
`(benchmarked_model, file) → doc_id` and takes last-wins; the *judge* is not in
that key. Two readers' rows in one judge-out file do not collide loudly, they
interleave, and each doc_id silently takes whichever reader wrote it last. The
resulting report is a blend of two instruments, is internally consistent, and
announces nothing. Filenames make that impossible for driver-produced files;
`apply_judge.assert_single_judge_model` catches the hand-run paths (`--out`
pointed at another reader's file, `cat` of two sweeps). It compares on
`_model_key`, which collapses filesystem spellings of one checkpoint (trailing
slash, relative path, symlink) so a legitimate resume is not aborted by a rename.

**3. No reader has an empty `out_suffix`, and that is now load-bearing.** While
the default's suffix was empty, `judge-out_TL*` and `llm-parsed*` matched every
reader at once, so `prep` had to archive with `judge-out_TL.jsonl*` **plus**
`judge-out_TL_limit*` and `--fresh` had to delete `llm-parsed` + `llm-parsed-limit*`
rather than `llm-parsed*`. Those special cases are gone: every glob is now anchored
on a reader name by construction, so `--fresh` under one reader cannot reach
another's records **or** any unsuffixed artifact left by an earlier one.
Registering `out_suffix=""` would silently re-arm that whole class of bug — and
worse, put two readers' rows in one judge-out file, where `load_judge_index`
blends them without complaint. test-11 fails on an empty suffix, and simulates
unsuffixed names against each reader's globs to prove the anchoring.

**4. The prompt fingerprint stays judge-independent.** `prompt_fingerprint`
hashes the rendered prompt, and no reader identity enters it. That is what lets a
new reader consume the 1.1 GB of queues already on disk instead of rebuilding
them, and it is why `reasoning_effort` remains in the fingerprint even though **no
current reader is sent it** — it describes the prompt this code authors, not who
is asked. `chat_kwargs` decides only who actually receives it. The two are checked
independently by `load_done` (prompt_fp, then `judge_model`), so identity is not
lost — it is carried by the other stamp. Moving `JUDGE_REASONING_EFFORT` into the
registry would invalidate every queue on disk. Pinned by test-11.

**Queues are deliberately NOT suffixed.** Every reader reads
`judge-queue_<task>.jsonl`. Suffixing it would double the disk and, worse, make
"same prompt" a claim rather than a fact.

**GPU topology: TP inside a shard, DP across shards.** Total GPUs =
`NUM_SHARDS × tensor_parallel`; `NUM_SHARDS` defaults to `devices / TP` (not
`devices`), and shard *S* owns devices `[S*TP, S*TP+TP)` via
`judge_env.judge_shard_devices` — one implementation, sourced by both launchers,
because a disagreement about ownership co-locates two engines and OOMs only after
both have loaded.

`tensor_parallel` is a **capacity floor, not a speed knob**. DP is the efficient
axis: shards share nothing, scale near-linearly and resume independently, whereas
TP inserts an all-reduce after every layer. A small or sparse model therefore
belongs at TP=1 with the spare cards running extra shards.

A dense 31B inverts it: ~62 GB of bf16 weights against 72 GB usable (80 GB card at
`gpu_memory_utilization=0.90`) leaves ~10 GB of KV cache, i.e. a handful of
concurrent sequences at `JUDGE_MAX_MODEL_LEN=12288` — capacity-bound, and an
outright OOM on smaller cards. TP=2 halves weights per GPU and hands ~30 GB back to
the KV cache. The eval path independently arrives at pure TP
(`eval__gemma4.py:308`, `tensor_parallel_size=num_processes`).

Registry values are floors derived from parameter counts, **not measurements** —
override with `TP=<n>` or `--tensor_parallel_size`. The preflight refuses
`NUM_SHARDS × TP > devices` and warns about the remainder, both before any weights
load. MOCK resolves no device list and falls back to the shard index (test-9 drives
the sharded mock path on a CPU box).

**Environments are per reader.** Gemma-4's config declares `transformers_version
5.5.0.dev0`, which no 4.x release can read, while a vLLM declaring
`transformers<5` cannot host it. Each reader therefore gets its own venv
(`.cache/judge-env<suffix>`, i.e. `.cache/judge-env_gemma-4-31b`), built by
`setup_judge_env.sh --judge KEY`, which reads the requirements file, torch pin and
expected transformers major from the registry rather than hardcoding them.

**Adding a reader** is one `JUDGE_MODELS` entry plus a requirements file. A
non-empty, unique `out_suffix` is mandatory; test-11 fails otherwise.
`resolve_judge_key` never guesses — an unrecognised `--model` aborts and asks for
`--judge`, because every consequence of a wrong key (wrong output directory, wrong
chat kwargs) is silent.
