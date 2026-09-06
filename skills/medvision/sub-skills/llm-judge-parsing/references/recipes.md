# Recipes: running the judge pipeline on real campaigns

Every recipe is the **same driver** re-pointed with environment overrides. There
are no wrapper scripts: two that once existed (an OOD loop and a late-model loop)
were retired on 2026-08-19 once their campaigns finished. If a task needs a
different tree or roster, that is `TASK_DIR_<task>` and `ROSTER_YAML_<task>`, not
a new script.

Preconditions for every GPU recipe (`smoke`, `pilot`, `full`):

```bash
export PYTHON=<judge-env>/bin/python     # see judge-environment.md; NOT optional
python scripts/check_judge_env.py --python "$PYTHON" --repo-root <repo>
```

`prep`, `stage0`, `analyze` and the repair tools run on CPU. Task names are
exactly `TL`, `AD`, `Detection` (they are also the `TASK_DIR_*`/`ROSTER_YAML_*`
suffixes).

---

## 1. Judge your own results tree

You have `Results/<my-tree>/<model>/parsed/*.jsonl` from
`../../results-parsing-and-metrics/SKILL.md` step 2 and want the format-robust twin.

**Step 1 — a roster YAML.** One mapping, directory name → report label:

```yaml
model_display_name:
  "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL (7B)"
  "MedVision-V0-7B":        "MedVision-V0"
```

Build it from the tree rather than by hand — the bundled helper skips judge
artifacts, `summary_*`, `llm-parsed*`, `_archive*` and any directory without
`parsed/*.jsonl`, which are exactly the Stage-0 gate failures:

```bash
python scripts/make_roster_yaml.py --results-dir <repo>/Results/<my-tree> --dry-run
python scripts/make_roster_yaml.py --results-dir <repo>/Results/<my-tree> \
    --display-name-map '{"Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL (7B)"}' \
    --out <repo>/script/llm-parsing/config-my-roster.yaml
```

**Step 2 — one invocation per task:**

```bash
TASKS="TL" \
TASK_DIR_TL=Results/<my-tree> \
ROSTER_YAML_TL=script/llm-parsing/config-my-roster.yaml \
PYTHON=<judge-env>/bin/python \
bash <repo>/script/llm-parsing/run_llm_parsing.sh stage0 smoke full analyze
```

Notes:

- Keep `smoke` in the list the **first** time a new setup runs: minutes of GPU to
  prove the reader answers in the required shape before committing hours.
- No `prep`. `prep` is for retiring a *previous* campaign's artifacts; on a fresh
  or resuming tree it either does nothing or archives the sweep you are finishing.
- Stage 0's response-count gate applies only to the three default trees. For any
  other tree it prints `[gate n/a]` and relies on the parser-replay and roster
  gates, so a custom tree needs no count bookkeeping.
- Queues, judge output and reports land **inside the tree you pointed at**;
  nothing collides with another campaign.

---

## 2. Add one model after a finished campaign

The point: **no re-judge of the other models.** Stage 1 skips finished rows by
`qid`, and `qid` has no prompt component, so growing the roster judges only the
new model's rows. (This is only true while the prompt has not changed — see the
safety net below.)

1. Add the model to the roster YAML(s) it appears in. Its key is its directory
   name under each Results tree, and that directory must already hold `parsed/`
   records.
2. Update `EXPECTED_ROSTER_COUNTS` in the checkout's `judge_config.py`. The new
   per-task total is printed by a dry run:
   `"$PYTHON" <repo>/script/llm-parsing/build_judge_queue.py --task_type TL --dry_run`
3. Run **without `prep`**:
   ```bash
   PYTHON=<judge-env>/bin/python \
   bash <repo>/script/llm-parsing/run_llm_parsing.sh stage0 full analyze
   ```

What appears afterwards:

| artifact | what changes |
|---|---|
| `judge-queue_<T>.jsonl` | rebuilt, now including the new model's rows |
| `judge-out_<T>_<reader>.jsonl` | **appended** — old rows untouched, new model's rows added |
| `<new model>/llm-parsed_<reader>/` | created |
| every other model's `llm-parsed_<reader>/` | rewritten from the same judge-out rows → identical content |
| `summary_*__llm-parsed_<reader>.txt`, `summary_judge_task__llm-parsed_<reader>.txt` | regenerated with one more row |
| `summary_<task>_task*.txt` (strict) | regenerated **roster-scoped** by Stage 3b |

**The safety net.** Stage 1 aborts before judging anything if the on-disk
judge-out rows carry a prompt stamp the current code does not produce. If it
fires, the finished rows answer a *different* prompt than the new model would be
judged with; they cannot be mixed and the whole campaign needs a `--fresh`
re-judge under the current prompt.

**The cautious variant** (used for the 19th roster model, 2026-08-19; 26,572 rows
= 2,441 TL + 2,060 AD + 22,071 Detection): judge the new model in a side "view"
tree — a directory containing one symlink to the model's real result directory,
selected with `TASK_DIR_<task>` — then append its judge-out and queue rows to the
main trees' files. Equivalent to the steps above, and it leaves the main tree
untouched until the new model's sweep is verified. Append; never `cat new old`.

---

## 3. Judge the OOD trees

Four splits, each its own tree and roster, **one `run_llm_parsing.sh` invocation
per split, run sequentially** (each sweep wants every visible GPU). Together they
were 185,716 responses — about 37% of the main sweep's GPU time.

| split | `TASK_DIR_*` | `ROSTER_YAML_*` |
|---|---|---|
| TL plane-OOD | `Results/MedVision-TL-v2-CoT-planeOOD` | `script/llm-parsing/config-TL-CoT-planeOOD.yaml` |
| TL task-OOD | `Results/MedVision-TL-v2-CoT-taskOOD` | `script/llm-parsing/config-TL-CoT-taskOOD.yaml` |
| Detection plane-OOD | `Results/MedVision-detect-v2-CoT-planeOOD` | `script/llm-parsing/config-detect-CoT-planeOOD.yaml` |
| Detection task-OOD | `Results/MedVision-detect-v2-CoT-taskOOD` | `script/llm-parsing/config-detect-CoT-taskOOD.yaml` |

```bash
# split 1 of 4; repeat with the next row's tree/roster when it finishes
TASKS="TL" \
TASK_DIR_TL=Results/MedVision-TL-v2-CoT-planeOOD \
ROSTER_YAML_TL=script/llm-parsing/config-TL-CoT-planeOOD.yaml \
PYTHON=<judge-env>/bin/python \
bash <repo>/script/llm-parsing/run_llm_parsing.sh stage0 full analyze

# a Detection split uses TASKS="Detection" and TASK_DIR_Detection / ROSTER_YAML_Detection
```

Reading OOD output:

- `[gate n/a] non-default task_dir …` on the count gate is **expected**, not a
  warning to fix.
- Stage 3b rewrites each tree's *strict* summary roster-scoped (`--models`
  from the roster YAML). A summary that previously listed a non-roster model
  loses that row until a summarizer is re-run without `--models`
  (`../../results-parsing-and-metrics/SKILL.md`).
- `prep` and `smoke` are omitted for the same reasons as recipe 1.

---

## 4. Change the main campaign

Adding a model is recipe 2 and costs only that model's rows. Anything bigger —
swapping or removing roster models, moving the default trees — touches four
places in the checkout and forces a full re-judge:

1. `config-{TL,AD,detect}-CoT.yaml` — the roster.
2. `run_llm_parsing.sh`'s `task_dir()` — the shell default tree.
3. `judge_config.py`'s `DEFAULT_TASK_DIR` — the Python-stage default tree
   (**both**; they are separate).
4. `judge_config.py`'s `EXPECTED_ROSTER_COUNTS` — currently TL 46,379,
   AD 39,140, Detection 437,349.

The count gate exists precisely so a roster edit that changes the workload is
loud: Stage 0 fails until the expected totals match the tree. Then start over
with recipe 5.

---

## 5. Start over: a full re-judge (`--fresh`)

Use it when the prompt changed, the roster changed, or every answer must be
produced again.

```bash
bash <repo>/script/llm-parsing/run_llm_parsing.sh --fresh        # asks first
bash <repo>/script/llm-parsing/run_llm_parsing.sh --fresh --yes  # unattended
```

`--fresh` adds exactly one thing to a normal run: `prep` also deletes the
reader's `llm-parsed_<reader>*/` directories. That matters because Stage 2
rewrites those records **file by file** — anything it no longer produces (a
dropped model, a renamed result file) would otherwise survive and be read back
into the new reports as if it belonged there.

- Previous judge answers are **archived, not deleted**: renamed `.v1`, `.v2`, …
  never overwriting an existing archive.
- Queues are deleted outright (pure derived data, rebuilt by `stage0`).
- Deletion is announced with a count and waits for confirmation.
- Scoped to `TASKS` and to the selected reader: `--fresh` under one reader
  cannot touch another reader's records, and `TASKS=TL` cannot retire the AD or
  Detection sweeps.
- Changed your mind mid-way? Move an archived file back to its original name
  before the judging step. Finished answers are matched by identity, so the run
  continues where the previous one stopped.

---

## 6. Repair a partial run

Symptom: most rows are fine, a minority came back unusable — usually the reader
ran out of room and its JSON was cut off. Re-judge **only** the bad rows. This is
launched by hand on purpose; the driver does not drive Stage-1 repairs.

```bash
"$PYTHON" <repo>/script/llm-parsing/run_judge_vllm.py \
    --queue Results/<tree>/judge-queue_<TASK>.jsonl \
    --out   Results/<tree>/judge-out_<TASK>_<reader>.jsonl \
    --model "${JUDGE_MODEL:?}" \
    --redo_invalid \
    --accept_prompt_fp <stamp-of-the-existing-rows> \
    --max_tokens 4096
bash <repo>/script/llm-parsing/run_llm_parsing.sh analyze   # rebuild the reports
```

Two flags carry the whole idea:

- `--accept_prompt_fp` — "the existing rows are still valid". Every answer is
  stamped with the prompt and budget that produced it and the pipeline refuses to
  mix stamps; the stamp of the rows on disk is printed in the error you get if
  you omit the flag. Repeatable.
- `--redo_invalid` — "but re-do the ones that failed". **Without it the repair is
  a silent no-op**: rows are skipped by `qid`, and the bad rows already have one.

A repair **appends** to the same file, and Stage 2 resolves duplicates per
`doc_id` last-wins, so repaired rows win. Never `cat new old`.

If the failure was in *decoding* the reader's text rather than in the text
itself, do not spend GPU at all — re-parse on CPU:

```bash
"$PYTHON" <repo>/script/llm-parsing/reparse_judge_out.py \
    --in  Results/<tree>/judge-out_<TASK>_<reader>.jsonl \
    --out Results/<tree>/judge-out_<TASK>_<reader>.reparsed.jsonl
```

It prints a transition table and refuses to write if any row moves ok→invalid or
any already-ok row's values change.

---

## 7. Use a different second reader

The headline claim — "this much apparent failure was formatting" — is only worth
quoting if it is a property of the responses rather than of one reader's habits,
so the corpus can be re-read with another reader and compared.

```bash
bash <repo>/script/llm-parsing/run_llm_parsing.sh --judges              # what is registered
bash <repo>/script/llm-parsing/run_llm_parsing.sh --judge gemma-4-31b   # run with it
```

`gemma-4-31b` is the default **and the only registered reader**; every artifact
on disk came from it. Adding one is a `JUDGE_MODELS` entry in `judge_config.py`
with a unique, non-empty `out_suffix` plus its requirements files, then
`setup_judge_env.sh --judge <key>` (`judge-environment.md` §1–2).

What the mechanism guarantees:

- Each reader answers the **same work list** — same queues, same prompts, same
  span checking, same metrics — so two readers' reports differ only by who read.
- Each reader owns `judge-out_<task>_<reader>.jsonl` and `llm-parsed_<reader>/`;
  nothing from one can reach the other's reports, and `--fresh` under one leaves
  the other alone.
- Readers may **not** share a virtual environment (see `judge-environment.md`).
- A run refuses to resume if the reader changed: finish a campaign with the
  reader it started with.
- Run `smoke` on a reader's first outing before committing thirteen hours.

---

## 8. Read the results

Two reports per task land side by side in the tree. **Diff them:**

| diff this (strict) | against this (format-robust) |
|---|---|
| `summary_TL_task_filtered.txt` | `summary_TL_task_filtered__llm-parsed_<reader>.txt` |
| `summary_AD_task.txt` | `summary_AD_task__llm-parsed_<reader>.txt` |
| `summary_detection_task.txt` | `summary_detection_task__llm-parsed_<reader>.txt` |

Both sides come from the same summarizer code, so the difference is the parse,
not the metric. Stage 3b regenerates the strict side **on the same roster** so
the two tables cover the same rows.

`summary_judge_task__llm-parsed_<reader>.txt` is the third report and the one to
read for *why*. Per model it gives:

- `SR strict`, `SR judge`, `dSR`, then `fmt-fail` (answers the regex missed) vs
  `non-answer` (the response never states one) — the decomposition of the
  strict parser's failures;
- the four answer modes (`in-format`, `off-format`, `no-concl`, `undet`);
- judge-vs-regex **agreement on rows the regex could already read** — the free
  reliability check, and the reason 100% of rows are judged rather than only the
  failures (currently 99.93% TL / 99.99% AD / 98.87% Detection);
- judge validity, step-extraction coverage, length stratification.

Sanity pattern for a pilot: `dSR` large and positive for known format offenders
and ≈0 for a format-compliant model such as MedVision-V0; agreement ≥ 99%;
judge-invalid and span-unverified both near zero.

Two caveats when quoting numbers:

- Judge output is **not** bit-reproducible — run to run on one GPU, and across
  GPU generations. Release the `judge-out_*.jsonl` file, not a re-run; Stages 2–4
  are pure CPU functions of it and *are* reproducible.
- A low success rate that **survives** re-parsing is not automatically a
  measurement failure. If a model's responses pile up against its generation
  limit it ran out of room — check the evaluation run's token budget
  (`../../benchmark-evaluation/SKILL.md`), not the text.

Also: re-summarizing any `llm-parsed_<reader>/` directory by hand needs
`--parsed_dirname llm-parsed_<reader> --resps_key LLM_filtered_resps`; the
summarizer aborts without the second flag rather than reporting zeros
(`../../results-parsing-and-metrics/SKILL.md`).

---

## 9. Visualize re-parsed records

The repository's per-sample response viewers and the radar figures read the same
per-model folders and take `--parsed_dirname` (their shell drivers:
`PARSED_DIRNAME`). Those drivers **default to `llm-parsed_gemma-4-31b`**; set
`PARSED_DIRNAME=parsed` to plot the published strict-parse records instead.

Output from a non-default source is kept apart automatically — per-sample figures
land in a `…__llm-parsed_gemma-4-31b` folder and radar figures carry the same
suffix in the filename — so a judge-sourced figure cannot overwrite a published
one. Entry points are catalogued in `../../../references/visualization-catalog.md`.

---

## 10. MOCK mode (`MOCK=1`) — wiring checks only

```bash
MOCK=1 TASKS="TL" TASK_DIR_TL=Results/<scratch-copy> \
bash <repo>/script/llm-parsing/run_llm_parsing.sh stage0 pilot analyze
```

`MOCK=1` replaces the reader with a deterministic regex extractor: it scans a
fixed pattern list from the end of the response and returns either the last *k*
numbers of the matched wrapper or `no_conclusion`. It stamps every row
`judge_model: "mock"`, skips the GPU preflight, and writes to
`judge-out_<T>_<reader>[_limitN].MOCK.jsonl`. It exercises every gate on a CPU
box — which is its entire purpose.

**Never report numbers produced under `MOCK=1`.** The banner says so on every
mock run; the record invariants test additionally requires that a real run
contains no mock row, and that a mock run contains nothing else.

**Hazard.** Stage 2 under `MOCK=1` writes into the **same**
`llm-parsed_<reader>[-limitN]/` directory a real run uses — there is no MOCK
infix on the directory. A mock `analyze`/`pilot` on a tree that already holds
real records overwrites them file by file with mock verdicts, and the next real
invariants gate fails on `judge_model='mock'` until the real `analyze` is re-run.
Point `TASK_DIR_<task>` at a scratch copy for mock exercises.

---

## 11. Skip the confirmation prompt (`--yes` / `-y`)

The pipeline has exactly one interactive gate: `prep` announces what it will
archive and delete, then waits for `Proceed? [y/N]`.

```bash
bash <repo>/script/llm-parsing/run_llm_parsing.sh --fresh --yes
YES=1 bash <repo>/script/llm-parsing/run_llm_parsing.sh prep stage0 full analyze
```

- The prompt reads from the terminal (`/dev/tty`). In an unattended run — cron,
  `nohup`, CI — there is no terminal, the read fails, and `prep`
  **auto-declines** (the driver then stops with `prep declined -- nothing
  changed`). Redirecting stdin does not help; pass `--yes` or `YES=1`.
- The flag matters only when `prep` runs — by default, that is every invocation
  with no explicit step list.
