# LLM-judge parsing pipeline (benchmark step 4)

The pipeline lives in the repository checkout under `script/llm-parsing/` (it is
**not** part of the installed `medvision_bm` package). Every command below is run
from a checkout, written `<repo>`; the driver re-roots itself to `<repo>` so the
working directory does not matter. All facts here were verified against the
driver, the stage modules (`--help` on CPU) and the on-disk artifacts of a finished
campaign.

## 1. Purpose and stance

The benchmark's strict regex parser (`parse_outputs`, owned by
`../../results-parsing-and-metrics/SKILL.md`) accepts an answer **only** inside
`<answer>…</answer>`. Everything else is scored as a miss, which mixes "the model
cannot measure" with "the model ignored the output format". The judge pipeline
re-reads **every** response with a language model whose only job is to find the
answer wherever it was written, then re-scores with the *same* metric code. You
end up with two reports per task: the published one and a format-robust one
carrying a `__llm-parsed_<reader>` suffix. Their difference is the formatting
share of apparent failure.

Load-bearing stance (from the design spec):

- the judge is an **extraction device**, never an evaluator: it never sees the
  image or the target, is never asked whether a value is right, never computes;
- it is a **pointer, not a transcriber**: it must quote the sentence the answer
  is in; the pipeline re-reads the digits out of that quote with the benchmark's
  own number regex, so a number the model never wrote cannot be scored;
- it is **additive only**: wherever the regex already succeeded, the regex value
  stands. The judge can add recoveries, never revise a published number.

Measured effect (18-model roster, retired reader — the paper's figures): T/L
success 77.0% → 89.7%, A/D 73.8% → 90.1%, Detection 93.4% → 98.4%.
Current reader (`gemma-4-31b`, 19-model roster, 522,868 responses): TL 82.6% → 94.8%,
AD 80.1% → 94.2%, Detection 92.3% → 98.5%; Llama-3.2-11B T/L 14.5% → 97.9% and
Qwen2.5-VL-32B T/L 16.4% → 99.9% (both current-reader extremes).
See `design-notes.md` §8 for the full snapshot and why the two tables must not be
diffed row by row.

**Never report numbers produced under `MOCK=1`** (section 9).

## 2. Stages and steps

Stage residency: 0 CPU · 1 GPU · 2 CPU · 3 CPU · 4 CPU.

```
parsed/*.jsonl ──Stage0──> judge-queue_<T>[_limitN].jsonl ──Stage1──> judge-out_<T>_<reader>[_limitN].jsonl
                           judge-baseline_<T>[_limitN].json                        │
parsed/*.jsonl ────────────────────Stage2 (apply_judge)──────────────> <model>/llm-parsed_<reader>[-limitN]/*.jsonl
                                     Stage3  existing summarizers  ──> summary_<task>_task[…]__llm-parsed_<reader>[-limitN].txt
                                     Stage3b existing summarizers  ──> strict summary on the SAME roster (for the diff)
                                     Stage4  summarize_judge_task  ──> summary_judge_task[_limitN]__llm-parsed_<reader>[-limitN].txt
                                     invariants: unit test 8 over the records just written
```

The driver `bash <repo>/script/llm-parsing/run_llm_parsing.sh` exposes these as
six **steps** (printed by `--list`; verified output):

| step | what it does | stages | cost |
|---|---|---|---|
| `prep` | move aside stale judge output (`.v1`, `.v2`…), delete stale queues; with `--fresh` also delete the reader's `llm-parsed*/` dirs. **The only destructive step and the only interactive one** (`Proceed? [y/N]`) | — | seconds |
| `stage0` | build the queues; replay the strict parser as a gate; roster gate; count gate | 0 | CPU, minutes |
| `smoke` | GATE G8: judge 200 evenly spaced rows of the first task in `TASKS` into a temp dir; assert ≥95% schema-valid and that spans were quoted | 1 | GPU, minutes |
| `pilot` | build `_limit<PILOT_LIMIT>` queues, judge them, run Stages 2–4 + invariants on the `-limitN` outputs | 0–4 | GPU, ~30 min |
| `full` | Stage 1 over the whole roster, every task in `TASKS` | 1 | GPU, ~13 h on two H100s |
| `analyze` | Stages 2, 3, 3b, 4 + invariants for the full sweep | 2–4 | CPU, ~1 h |

With no step named, **all six run in order** (so `prep` runs and prompts). The
run stops at the first failed gate.

### Per-step details

**prep.** Scoped to `TASKS` and to the selected reader (`judge-out_<T>_<reader>*.jsonl*`
and `<tree>/*/llm-parsed_<reader>*`). Judge-out files are *renamed* to `.v1`
(`.v2`, … never overwriting an archive); queues are deleted (pure derived data);
`.judge-shards_*` leftovers are removed. Without a terminal the prompt
auto-declines, so unattended runs need `--yes`/`-y` or `YES=1`. Existing
`llm-parsed*/` directories are only *reported* unless `--fresh`.

**stage0** (`build_judge_queue.py`, per task): resolves the roster YAML → model
directories; each must exist and hold `parsed/*.jsonl` (no glob fallback);
replays the strict parser on every row and compares with the stored
`filtered_resps` (`[GATE FAIL]` on any mismatch); on a full build over a
*default* tree checks the response total against `EXPECTED_ROSTER_COUNTS`
(TL 46,379 / AD 39,140 / Detection 437,349 for the 19-model roster) — for a
non-default `--task_dir` it prints `[gate n/a]` and skips; on `--limit` prints
`[gate n/a]`. Any gate failure **retracts** the written queue and baseline.
Writes `judge-queue_<T>[_limitN].jsonl` + `judge-baseline_<T>[_limitN].json`.

**smoke / full / pilot Stage 1** are delegated to `test-sweep.sh` (production
runner despite the name): GPU preflight (torch import, CUDA allocation, vllm
import, device list), shard layout (`NUM_SHARDS × TP ≤ visible GPUs`, idle-GPU
note), one `run_judge_vllm.py` process per shard writing
`<out>.n<NUM_SHARDS>.shard<S>`, then an explicit-list merge into
`judge-out_<T>_<reader>[_limitN].jsonl`. The merge **refuses** to overwrite a
merged file that holds qids no shard has (repair/re-parse rows) or whose
`.<name>.reparsed` marker is at least as new as the file.

**Stage 1 gates** (inside `run_judge_vllm.py`): the queue's `prompt_fp` must
equal the stamp the current code renders (else abort — rebuild with `stage0`);
the first 200 rows must validate at ≥ `--min_valid_rate` (0.95) or it aborts
**writing nothing**; `finish_reason == "length"` is counted and warned.
Resume: rows are skipped by `qid`; repeated response texts are served from an
in-file cache by `cache_key` and marked `+cached`; output is appended in fsynced
chunks of `--chunk_rows` (2000). A resume refuses rows from a different
`judge_model` (mock vs real, or a different reader).

**analyze** (per task): Stage 2 `apply_judge.py` (span verification, decision
table, `cal_metrics` re-scoring, writes `llm-parsed_<reader>[-limitN]/`) with
`--accept_prompt_fp` for every stamp in `ACCEPT_FP`; Stage 3
`python -m medvision_bm.benchmark.summarize_<TL|AD|detection>_task --task_dir <tree> --parsed_dirname llm-parsed_<reader> --resps_key LLM_filtered_resps --models <roster> -p <PROCS> --skip_model_wo_parsed_files` (TL adds
`--removed_samples_dir Data/Datasets`); Stage 3b the same summarizer on
`parsed/` with `--models <roster>` (the strict baseline on the same rows);
Stage 4 `summarize_judge_task.py`; then unit test 8 as the record-invariants
gate (`TASKS=<task>` and `MOCK` passed through).

## 3. Driver flags

Verified from the driver's option loop.

| flag | effect |
|---|---|
| `<step> [<step>…]` | run only these steps, in the order given |
| `--from <step>` | that step and every later one |
| `--list` | print the step table, run nothing |
| `--fresh` | `prep` additionally deletes the reader's `llm-parsed*/` directories (the one-command full re-judge); still asks |
| `--yes`, `-y` | answer the `prep` confirmation in advance (same as `YES=1`) |
| `--judge <key>` | pick the reader from the registry (default `gemma-4-31b`, the only one registered) |
| `--judges` | list registered readers (`judge_config.py --list`) |
| `--help`, `-h` | header comment + step table |

## 4. Environment variables (complete)

Every variable the driver, `judge_env.sh` (sourced) and `test-sweep.sh` read;
defaults are the source's.

| variable | default | meaning |
|---|---|---|
| `PYTHON` | `python3` (first on PATH; the banner says "PYTHON unset") | **Set it to the judge env**: `export PYTHON=<judge-env>/bin/python`. One interpreter runs all stages |
| `TASKS` | `TL AD Detection` | which tasks run; also scopes `prep` |
| `TASK_DIR_TL` / `TASK_DIR_AD` / `TASK_DIR_Detection` | `Results/MedVision-TL-v2-CoT` / `Results/MedVision-AD-v2-CoT` / `Results/MedVision-detect-v2` | re-point one task at another Results tree (repo-root-relative or absolute) |
| `ROSTER_YAML_TL` / `_AD` / `_Detection` | empty → registry default `config-TL-CoT.yaml` / `config-AD-CoT.yaml` / `config-detect-CoT.yaml` (resolved next to `judge_config.py`) | swap one task's roster |
| `JUDGE` | registry default `gemma-4-31b` | reader key (same as `--judge`) |
| `JUDGE_MODEL` | `JUDGE_MODEL_HF` | explicit checkpoint path/id (e.g. a local mirror); stamped into every judge-out row |
| `JUDGE_MODEL_HF` | `google/gemma-4-31B-it` (registry) | upstream hub id; downloaded on first load (~62 GB, needs `HF_HOME` room) |
| `TP` | `1` (driver overrides the registry's 2) | GPUs one engine spans; raise to *fit* weights, not for speed |
| `NUM_SHARDS` | `1`, or `GPU_NUM / TP` | independent processes (data parallel); total GPUs = `NUM_SHARDS × TP` |
| `GPU_NUM` | unset | one-knob layout; must divide by `TP`; explicit `NUM_SHARDS` wins |
| `CUDA_VISIBLE_DEVICES` | unset → all devices | honoured, never re-indexed; shard *S* gets devices `[S·TP, S·TP+TP)` of that list |
| `PROCS` | `32` | CPU workers for Stages 0/2/3 |
| `PILOT_LIMIT` | `100` | rows per file for `pilot` |
| `SMOKE_ROWS` | `200` | rows judged by `smoke` |
| `STRUCTURED` | empty in the driver (→ `run_judge_vllm` default); `auto` in `test-sweep.sh` | grammar-constrained decoding `auto`/`none`; `none` avoids the CPU-bound xgrammar path |
| `ACCEPT_FP` | the 8 historical stamps (`design-notes.md` §4) | extra prompt stamps Stage 2 accepts in a judge-out file |
| `MOCK` | `0` | `1` = regex stand-in instead of a model (section 9) |
| `FRESH` | `0` | set by `--fresh` |
| `YES` | `0` | `1` = skip the `prep` prompt |
| `SKIP_GPU_CHECK` | `0` | `1` = bypass the CUDA preflight (only when you are sure) |
| `MEDVISION_DS_SRC` | unset → upward search for a `MedVision/src` or `medvision_ds/src` sibling | where `medvision_ds` lives if not importable |
| `LIMIT` | set by the driver | `test-sweep.sh` only: judge the `_limit<N>` queues |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN` | — | whitespace-stripped by `judge_env.sh` (a trailing newline makes the auth header invalid) |
| `HF_HOME`, `HF_HUB_CACHE` | — | newline-stripped by `judge_env.sh` |
| `PYTHONPATH` | prefixed with `src` | Stage 3 imports `medvision_bm` from the checkout, not the installed copy |

Preflight order (verified): if `smoke`/`pilot`/`full` is selected, `vllm` must
import and a CUDA tensor must allocate under `PYTHON` (skipped under `MOCK=1`);
if `analyze`/`pilot` is selected, `medvision_ds` must resolve and the real
Stage-2/3 imports (`cal_metrics`, the selected tasks' summarizers,
`medvision_ds.utils.benchmark_planner`, `preprocess_utils`) must succeed — hard
abort; for `full`/`smoke` the same probe only warns.

## 5. Artifacts and naming

Per task tree `<tree>` (= `TASK_DIR_<task>`), reader suffix `<sfx>` =
`_gemma-4-31b`, optional limit `N`:

| artifact | path | producer |
|---|---|---|
| queue | `<tree>/judge-queue_<T>[_limitN].jsonl` (TL ≈ 228 MB, Detection ≈ 880 MB; ~1.1 GB total) | Stage 0; **not** reader-suffixed on purpose (every reader answers the same queue) |
| baseline | `<tree>/judge-baseline_<T>[_limitN].json` (per-model strict stats + counts) | Stage 0 |
| judge output | `<tree>/judge-out_<T><sfx>[_limitN].jsonl` (append-only; TL ≈ 53 MB, Detection ≈ 223 MB) | Stage 1 |
| shard files | `<tree>/judge-out_<T><sfx>[_limitN].jsonl.n<NUM_SHARDS>.shard<S>` | Stage 1 (authoritative resume state when sharded) |
| mock output | `<tree>/judge-out_<T><sfx>[_limitN].MOCK.jsonl` | Stage 1 under `MOCK=1` |
| archives | `<file>.v1`, `.v2`, … | `prep` |
| re-parse marker | `<tree>/.judge-out_<T><sfx>.jsonl.reparsed` | `reparse_judge_out.py` |
| re-parsed records | `<tree>/<model>/llm-parsed<sfx>[-limitN]/<same filename>.jsonl` | Stage 2 |
| judge metrics per model | `<tree>/<model>/llm-parsed<sfx>[-limitN]/summary_metrics_judge_Task[_limitN].json` | Stage 4 |
| format-robust summaries | TL: `<tree>/summary_TL_task_filtered[_limitN]__llm-parsed<sfx>[-limitN].txt` (the `_filtered` comes from `--removed_samples_dir`); AD: `summary_AD_task[_limitN]__llm-parsed<sfx>[-limitN].txt`; Detection: `summary_detection_task[…]__llm-parsed<sfx>[…].txt` and `summary_metrics_all_models_detect_Task[…]__llm-parsed<sfx>[…].json`; per model `llm-parsed<sfx>/summary_metrics_TL_Task[_filtered]__llm-parsed<sfx>.json` (**TL only** — AD and Detection per-model files keep their plain `summary_metrics_{AD,detect}_Task.json` names inside the judge folder) + `summary_values_…` | Stage 3 (existing summarizers) |
| strict summaries | `summary_TL_task_filtered.txt`, `summary_AD_task.txt`, `summary_detection_task.txt` (roster-scoped since the driver passes `--models`) | Stage 3b |
| judge report | `<tree>/summary_judge_task[_limitN]__llm-parsed<sfx>[-limitN].txt` | Stage 4 |

`parsed/` is never written (invariant I1). The limit lives in the *directory*
name (`llm-parsed<sfx>-limit100`) so a pilot can never be summarized as a full run.

### Row schemas (design spec §3, confirmed on disk)

Queue row (Stage 0 → 1):
```
{qid, task_type, model, file, dataset, doc_id, step_key, regex_pred,
 response_chars, was_windowed, response, cache_key, prompt_fp}
```
`qid = hash(model, file basename, doc_id)` — the resume key, **no prompt
component**; `cache_key = hash(response, full prompt fingerprint)` — the dedup
key; `step_key ∈ {"TL", "AD:distance", "AD:angle", null}`; `regex_pred` is
diagnostic only and never read downstream.

Judge-out row (Stage 1 → 2), flat (`final_answer` at top level):
```
{qid, cache_key, prompt_fp, doc_id, file, model, task_type,
 judge_model, judge_status ∈ {ok, invalid}, judge_reason,
 final_answer: {status ∈ {present, no_conclusion}, span, values},
 steps?: [{index, status ∈ {present, absent}, span, values}],
 raw?, raw_len?}
```
`raw` is kept in full on invalid rows by default (so a decoder fix is a CPU
re-parse); duplicates per `doc_id` resolve **last-wins** in Stage 2.

`llm-parsed` record (Stage 2): the `parsed/` record with `filtered_resps`
**removed** and `LLM_filtered_resps=[pred]` in its slot (key order preserved);
metrics recomputed from `pred` via `cal_metrics` (TL/AD → `avgMAE, avgMRE,
SuccessRate, nMAE?`; Detection → `avgMAE, avgIoU, F1, Precision, Recall,
SuccessRate`); added `LLM_judge_answer_mode`, `LLM_judge_SR{success}`,
`LLM_judge{reason, strict_pred, judge_pred, judge_span, judge_model,
verify_tier?}`, `LLM_judge_steps?` (TL/AD). Observed keys on disk:
`doc_id, doc, target, arguments, resps, LLM_filtered_resps, doc_hash,
prompt_hash, target_hash, avgMAE, avgMRE, SuccessRate, input, nMAE,
LLM_judge_answer_mode, LLM_judge_SR, LLM_judge, LLM_judge_steps`.

Answer modes (`LLM_judge_answer_mode`):

| mode | meaning | counts as parsed |
|---|---|---|
| `conclusion_in_format` | the regex already had it (decided by the strict parser) | yes |
| `conclusion_off_format` | found and span-verified by the judge | yes |
| `no_conclusion` | the response never states an answer (declined or stopped early) | no |
| `undetermined` | judge unusable **and** regex failed — the pipeline's own error rate; ΔSR is therefore a lower bound | no |

## 6. Identity and resumability

- Every queue row and judge-out row carries `prompt_fp` = first 16 hex of a
  BLAKE2b hash over the **rendered** prompt (system + user template + output
  skeleton + JSON schema), the response window constants, the elision marker,
  the task's `max_tokens`, and `JUDGE_REASONING_EFFORT`. There is no
  hand-maintained version constant. Reader identity is **not** in it (so a new
  reader reuses the queues) — it is carried by `judge_model` on each row.
- Current stamps (computed from the checkout's code and equal to the stamps on
  the finished campaign's files): TL `c515f64a54eafab8`, AD:distance
  `02728aba5cea5964`, AD:angle `b2b30f63c35b946f`, Detection `fd1a0ea674ca6a44`
  (all at `max_tokens=4096`).
- Editing the prompt, the skeleton, the budget, the window or the reasoning
  effort moves every stamp ⇒ Stage 1 refuses the old queues ⇒ `stage0` must
  rebuild (~1.1 GB) and the old judge-out rows answer a different question.
- Resume is by `qid` per output file; a plain rerun of `full` skips finished
  rows. Shard count is in the shard filename, so a changed `NUM_SHARDS` is caught
  as stale rather than merged into duplicates.
- A resume aborts if the `judge_model` stamp differs (mock vs real, or another
  reader): finish a campaign with the reader it started with.

## 7. GPU layout and cost

Default is **one GPU** (`TP=1`, `NUM_SHARDS=1`). The reader is ~62 GB bf16, so on
an 80 GB card at `gpu_memory_utilization=0.90` only ~10 GB is left for KV cache
and the sweep runs with few requests in flight (slow, and an outright OOM on
smaller cards). The registry's value is `tensor_parallel=2`; use
`GPU_NUM=4 TP=2` (two 2-GPU processes) or `GPU_NUM=4` (four 1-GPU processes).
Shards share nothing and scale near-linearly; TP buys capacity, not speed. The
driver refuses `NUM_SHARDS × TP > visible GPUs` before loading weights.

Rough cost on two H100s (README): one-off ~62 GB download, minutes of CPU for
`stage0`, ~30 min GPU for `pilot`, ~13 h GPU for `full` over 522,868 responses,
~1 h CPU for `analyze`. The four OOD splits (185,716 responses) took about 37%
of the main sweep's GPU time. `JUDGE_MAX_MODEL_LEN=12288`; worst measured
prompt 5,316 tokens (TL).

## 8. Reading the results

Diff the two reports per task (`summary_<task>_task…txt` vs
`…__llm-parsed_<reader>.txt`). The judge report
(`summary_judge_task__llm-parsed_<reader>.txt`) adds, per model: SR strict, SR
judge, dSR, `fmt-fail` (answers the regex missed) vs `non-answer`; the
four-mode table; and judge-vs-regex agreement on rows the regex could already
read (99.93% TL / 99.99% AD / 98.87% Detection with the current reader) — the
free reliability check that is why 100% of rows are judged. The pilot's reading
guide: dSR large and positive for known offenders (Qwen2.5-VL-32B,
Llama-3.2-11B) and ≈0 for MedVision-V0; agreement ≥ 99%; judge-invalid and
span-unverified both near 0.

Caveats when quoting: judge output is not bit-reproducible (run-to-run on the
same GPU, and across GPU generations) — release the judge-out file, not a
re-run; a low SR that survives re-parsing may be a token-budget wall, not a
measurement failure (check the eval run's generation settings).

## 9. MOCK mode (`MOCK=1`) — wiring checks only

What it does, exactly (verified in `run_judge_vllm._mock_judge`): replaces the
model with a deterministic regex extractor that scans a fixed pattern list from
the end of the response, returns `present` with the last `k` numbers of the
matched wrapper (span capped at 200 chars, `steps: []`) or `no_conclusion`;
stamps every row `judge_model: "mock"`; skips the GPU preflight; writes
`judge-out_<T><sfx>[_limitN].MOCK.jsonl`; `analyze` under `MOCK=1` reads that
file and unit test 8 then requires **every** record to be mock (and, in a real
run, requires none). Three isolation mechanisms keep mock rows out of real runs:
a separate output filename, provenance in the resume (`load_done` treats a mock
row as stale to a real run and vice versa), and provenance in the record
(`LLM_judge.judge_model`).

Hazard: Stage 2 under `MOCK=1` writes into the **same**
`llm-parsed<sfx>[-limitN]/` directory a real run uses — there is no MOCK infix
on the directory. Running a mock `analyze`/`pilot` on a tree that already holds
real records overwrites them file by file with mock verdicts, and the next real
invariants gate fails on `judge_model='mock'` until the real `analyze` is
re-run. Point `TASK_DIR_<task>` at a scratch copy for mock exercises.

**Never report MOCK numbers.** The banner says so on every mock run.

## 10. Tests

From `<repo>`, with the same interpreter as the pipeline:

```
for i in 1 2 3 4 5 6 7 9 11; do "$PYTHON" unit-test/llm-parsing/test-$i.py; done
"$PYTHON" unit-test/llm-parsing/test-8.py llm-parsed_gemma-4-31b   # after a full run; reads real records
```

1 strict-parser parity (incl. real corpus if present) · 2 span verification and
anti-fabrication · 3 decoder · 4 naming/arity/budget invariants · 5 record
ordering/limit · 6 prompt–schema agreement and fingerprint completeness · 7 the
ten decision-table rows · 8 invariants over the real `llm-parsed` records on
disk (skips cleanly when the directory is absent) · 9 the shell entry points run
in both MOCK modes with a stub interpreter · 11 the judge registry. Tests
1–7, 9, 11 pass on a CPU-only interpreter without vllm (verified). There is no
test 10 (retired with the earlier reader). `test-sweep.sh` is the GPU Stage-1
runner, not a test; it needs `PYTHON=<judge-env>/bin/python`.
