# LLM-as-Judge output parsing

Some models get a measurement right but write the answer in the wrong place. The
benchmark's regex parser only accepts answers inside `<answer>…</answer>`, so
everything else is scored as a miss — which mixes *"the model can't measure"*
together with *"the model didn't follow the output format"*.

This pipeline reads every response a second time with a language model whose only
job is to find the answer, wherever it was written. You end up with two versions
of the same report: the published one, and a format-robust one. The gap between
them is how much of a model's apparent failure was formatting.

Measured effect on the 18-model roster: T/L success 77.0% → 89.7%, A/D 73.8% →
90.1%, Detection 93.4% → 98.4%. Llama-3.2-11B's T/L success goes from 14.5% to
97.9% — almost all of its "failure" was formatting.

> Internals — module map, decision table, verification tiers, edit hazards:
> [DESIGN.md](DESIGN.md). Background on why the second reader is trustworthy:
> [What the second reader does](#what-the-second-reader-does-and-doesnt) below.

## Quick start

**1. Build the judge environment** (once per machine — the second reader needs a
newer vLLM than the evaluation code pins):

```bash
bash script/llm-parsing/setup_judge_env.sh          # into <repo>/.cache/judge-env_gemma-4-31b
export PYTHON=<printed target>/bin/python           # the script prints this line
```

See [Environment notes](#environment-notes) if the setup fails or prints
alarming red pip errors (for Gemma-4, several are expected).

**2. Run the pipeline.** Any working directory works; the script re-roots itself
to the repository:

```bash
bash script/llm-parsing/run_llm_parsing.sh              # everything, in order
bash script/llm-parsing/run_llm_parsing.sh --list       # show the steps, run nothing
bash script/llm-parsing/run_llm_parsing.sh analyze      # just the CPU stages + reports
bash script/llm-parsing/run_llm_parsing.sh --from full  # that step and everything after
bash script/llm-parsing/run_llm_parsing.sh --help       # flags and environment knobs
```

The steps, in order (also printed by `--list`):

| step | what it does | cost |
|---|---|---|
| `prep` | move aside stale judge output, delete stale queues | destructive, seconds — the only step that asks for confirmation |
| `stage0` | build the work queues; replay the strict parser as a gate | CPU, minutes |
| `smoke` | prove the judge returns the required schema | GPU, minutes |
| `pilot` | `PILOT_LIMIT` (100) rows per file, end to end + invariants | GPU, ~30 min |
| `full` | the judge over the whole roster, every task | GPU, ~13 h |
| `analyze` | verify, merge, reports, record invariants | CPU, ~1 h |

It stops at the first failed check, and it is resumable — finished work is keyed
by identity and skipped, so it can be killed and restarted freely. Rough cost on
two H100s: a one-off ~62 GB model download, minutes of CPU preparation, about
thirteen hours of GPU for the full sweep, an hour of CPU for the reports.

**3. Read the reports.** Two reports per task land side by side in each task's
Results tree: the published one, and the same report with a
`__llm-parsed_<reader>` suffix. Diff them. See
[Reading the results](#reading-the-results).

## One driver, any tree

`run_llm_parsing.sh` is the only driver. Out of the box it judges the main
campaign — the 19-model roster (`config-{TL,AD,detect}-CoT.yaml`: the paper's
18 models plus the late-added fullSFT checkpoint) over
`Results/MedVision-TL-v2-CoT`, `-AD-v2-CoT` and `-detect-v2`. Every other
campaign is the same driver re-pointed at another tree and roster via
environment overrides, one invocation per tree — no wrapper scripts:

```bash
TASKS="TL" \
TASK_DIR_TL=Results/MedVision-TL-v2-CoT-planeOOD \
ROSTER_YAML_TL=script/llm-parsing/config-TL-CoT-planeOOD.yaml \
PYTHON=<judge-env>/bin/python \
bash script/llm-parsing/run_llm_parsing.sh stage0 full analyze
```

For a re-pointed run choose the steps `stage0 full analyze` — no `prep` (on a
resume it would archive the very sweep being finished) and no `smoke` (the
reader already cleared the main sweep, and every full run is still guarded by
an in-run validity gate over its first 200 rows). The OOD-split and late-model
recipes below are instances of this pattern; two wrapper scripts that used to
loop over it (`run_llm_parsing_ood.sh`, `run_llm_parsing_fullsft.sh`) were
retired 2026-08-19 once their campaigns finished and (for fullSFT) the results
were merged into the main trees.

### Skipping the confirmation prompt: `--yes` / `-y`

The pipeline has exactly one interactive gate: `prep` announces what it will
archive/delete and waits for `Proceed? [y/N]`. `--yes` (short `-y`) answers it
in advance; setting `YES=1` in the environment is equivalent:

```bash
bash script/llm-parsing/run_llm_parsing.sh --fresh --yes
YES=1 bash script/llm-parsing/run_llm_parsing.sh prep stage0 full analyze
```

Details worth knowing:

- The prompt reads from the terminal (`/dev/tty`). In an unattended run — cron,
  `nohup`, a CI job — there is no terminal, the read fails, and `prep`
  **auto-declines**. A scripted run that includes `prep` must therefore pass
  `--yes` (or `YES=1`), not just redirect stdin.
- The flag only matters when `prep` runs: by default that is every
  `run_llm_parsing.sh` invocation with no explicit step list.

## Adapting it to your models and result paths

Everything the pipeline needs to know about *what* to judge lives in two places:

1. **A roster YAML** — which models to judge and what to call them in reports:

   ```yaml
   model_display_name:
     "<model directory name>": "<label used in reports>"
   ```

   The key is the model's directory name under the Results tree (the directory
   must contain the benchmark's `parsed/` records); the value is the display
   name. `judge_config.py`'s `DEFAULT_ROSTER_YAML` maps each task to its
   default file: TL → `config-TL-CoT.yaml`, AD → `config-AD-CoT.yaml`,
   Detection → `config-detect-CoT.yaml`.

2. **A Results tree per task** — where those model directories live. Defaults
   (in `run_llm_parsing.sh`'s `task_dir()` and, for the Python stages,
   `judge_config.py`'s `DEFAULT_TASK_DIR`):
   TL → `Results/MedVision-TL-v2-CoT`, AD → `Results/MedVision-AD-v2-CoT`,
   Detection → `Results/MedVision-detect-v2`.

Both are overridable **per task from the environment, with no editing**:
`TASK_DIR_<task>` re-points the tree, `ROSTER_YAML_<task>` swaps the roster
(`<task>` = `TL`, `AD` or `Detection`), and `TASKS` narrows which tasks run.
That is how the OOD splits were judged, and the recommended way to make
your own changes.

### Recipe: judge your own results tree

Write a roster YAML for the models in your tree, then point one task at it:

```bash
TASKS="TL" \
TASK_DIR_TL=Results/My-TL-tree \
ROSTER_YAML_TL=script/llm-parsing/config-my-roster.yaml \
PYTHON=<judge-env>/bin/python \
bash script/llm-parsing/run_llm_parsing.sh stage0 smoke full analyze
```

Notes:

- Stage 0's response-count gate only guards the default trees — for any other
  tree it prints `[gate n/a]` and relies on the remaining gates (parser
  agreement, roster resolution), so a custom tree needs no count bookkeeping.
- Queues, judge output and reports all land inside the tree you pointed at;
  nothing collides with the main campaign's files.
- Keep `smoke` in the step list the first time a new setup runs — it costs
  minutes and proves the reader answers in the required shape before you spend
  GPU hours.

### Recipe: add one model after a finished campaign

If the judge prompt has not changed since the campaign (the normal case), no
side campaign is needed: Stage 1 skips finished rows by identity, so growing
the roster re-judges **only the new model's rows**.

1. Add the model to `config-{TL,AD,detect}-CoT.yaml`. Its key is its directory
   name under each Results tree, and that directory must contain the
   benchmark's `parsed/` records.
2. Update `EXPECTED_ROSTER_COUNTS` in `judge_config.py` — Stage 0's dry run
   prints the new total per task:
   `python script/llm-parsing/build_judge_queue.py --task_type TL --dry_run`.
3. `bash script/llm-parsing/run_llm_parsing.sh stage0 full analyze` — no
   `prep` (it would archive the finished sweep).

The safety net: Stage 1 aborts before judging anything if the on-disk judge-out
rows carry a prompt stamp the current code does not produce. If it fires, the
finished rows answer a *different prompt* than the one the new model would be
judged with — the two cannot be mixed, and the whole campaign needs a re-judge
(`--fresh`) under the current prompt.

That is how the fullSFT checkpoint became the roster's 19th model (2026-08-19;
26,572 rows: 2,441 TL + 2,060 AD + 22,071 Detection — its Detection eval is
missing BCV15 Task01, which every other roster model has). It was judged in a
side "view" tree (a directory holding one symlink to the model's real result
directory, judged via the `TASK_DIR_<task>` override) and its judge-out and
queue rows were then appended to the main trees' files — equivalent to the
steps above, and useful when the main tree must stay untouched until the new
model's sweep is verified. The pre-merge artifacts are in
`Results/_archive_llm-parsing_2026-08-19/`.

### Recipe: judge other Results trees (the OOD splits)

When the same task type must be judged over multiple Results trees — each with
its own roster — run one re-pointed invocation per tree, sequentially (each
sweep wants every visible GPU). The four OOD splits (185,716 responses:
13,605 + 2,292 TL, 129,799 + 40,020 Detection — about 37% of the main sweep's
GPU time) are exactly this:

| split | tree | roster |
|---|---|---|
| TL plane-OOD | `Results/MedVision-TL-v2-CoT-planeOOD` | `config-TL-CoT-planeOOD.yaml` |
| TL task-OOD | `Results/MedVision-TL-v2-CoT-taskOOD` | `config-TL-CoT-taskOOD.yaml` |
| Detection plane-OOD | `Results/MedVision-detect-v2-CoT-planeOOD` | `config-detect-CoT-planeOOD.yaml` |
| Detection task-OOD | `Results/MedVision-detect-v2-CoT-taskOOD` | `config-detect-CoT-taskOOD.yaml` |

```bash
# one split; repeat with the next row's tree/roster when it finishes
TASKS="TL" \
TASK_DIR_TL=Results/MedVision-TL-v2-CoT-planeOOD \
ROSTER_YAML_TL=script/llm-parsing/config-TL-CoT-planeOOD.yaml \
PYTHON=<judge-env>/bin/python \
bash script/llm-parsing/run_llm_parsing.sh stage0 full analyze
```

(For a Detection split the overrides are `TASK_DIR_Detection` /
`ROSTER_YAML_Detection` and `TASKS="Detection"`.)

Two things to know when reading OOD-style output: the Stage 0 count gate prints
`[gate n/a]` for non-default trees (expected, see above), and Stage 3b rewrites
each tree's strict summary roster-scoped — a summary that previously included a
non-roster model loses that row until the summarizer is re-run without
`--models`.

### Changing the *main* campaign itself

Adding a model is the recipe above and costs only that model's rows. For
anything bigger — swapping or removing roster models, or moving the default
trees — edit `config-{TL,AD,detect}-CoT.yaml` (roster)
and the default paths in **both** `run_llm_parsing.sh` (`task_dir()`) and
`judge_config.py` (`DEFAULT_TASK_DIR`), and update `EXPECTED_ROSTER_COUNTS` in
`judge_config.py` — that count gate exists precisely so a roster edit that
changes the workload is loud, and Stage 0 fails until the expected totals match
(currently TL 46,379, AD 39,140, Detection 437,349). Expect a full re-judge:
start it with [`--fresh`](#starting-over-a-full-new-re-judge).

## Environment variables

All optional; every default works on the main campaign.

| variable | what it is for |
|---|---|
| `PYTHON` | interpreter that has PyTorch and vLLM — a GPU pod usually has several `python3` on `PATH`, and the wrong one reports "no CUDA device". Set it to the judge env |
| `TASKS` | which tasks run, default `TL AD Detection` |
| `TASK_DIR_<task>` | re-point one task at another Results tree (see [Adapting](#adapting-it-to-your-models-and-result-paths)) |
| `ROSTER_YAML_<task>` | swap one task's roster YAML the same way |
| `JUDGE` | which second reader to use, same as `--judge` |
| `JUDGE_MODEL` | a checkpoint to use instead of the registry's, e.g. a local mirror of the same weights |
| `TP` | GPUs one process spans. Default 1 (single GPU, overriding the reader's registry value); raise it only to *fit* the model |
| `NUM_SHARDS` | independent processes. Default 1; e.g. `TP=2 NUM_SHARDS=2` uses four GPUs |
| `GPU_NUM` | total GPUs — the one-knob spelling: `NUM_SHARDS` is derived as `GPU_NUM / TP` (must divide evenly). An explicit `NUM_SHARDS` wins |
| `PROCS` | CPU workers for the non-GPU stages (default 32) |
| `PILOT_LIMIT` | rows per file for the `pilot` step (default 100) |
| `YES=1` | answer the `prep` confirmation in advance, same as `--yes`/`-y` |
| `MOCK=1` | run everything on a CPU box with a regex stand-in instead of a model. Exercises every gate; **never report its numbers** |
| `MEDVISION_DS_SRC` | the dataset package, if it is not installed and not found next to this repository |

Flags on `run_llm_parsing.sh`: `--list`, `--from <step>`, `--fresh`,
`--yes`/`-y`, `--judge <key>`, `--judges`, `--help`.

## Environment notes

`setup_judge_env.sh` builds one environment per reader
(`.cache/judge-env<reader suffix>`, or a directory you name) and refuses to
finish unless the result actually works. Two failures it exists to prevent,
both of which otherwise appear much later and blame the wrong thing:

- **"The NVIDIA driver on your system is too old."** Almost always the opposite
  of what it says — the driver is fine and the PyTorch build is too new, usually
  because the pipeline picked up whichever `python3` came first on `PATH`. If
  your driver is older than CUDA 12.8, build against it instead:
  `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash script/llm-parsing/setup_judge_env.sh`
- **A silently wrong Transformers.** vLLM states only a lower bound, so an
  unpinned install can land on a line the reader cannot load. The pinned
  requirements files next to the setup script avoid it, and the setup refuses to
  finish on a mismatch.

For Gemma-4 the install runs in two passes, because the versions it needs
contradict what vLLM asks for and pip will not accept a contradiction in a
single step. **The second pass prints a block of red "incompatible" lines —
several of them — and that is expected**: forcing a newer Transformers makes
every installed package that asked for an older one complain. Those are version
*requests* written by package authors, not observed failures. The script then
imports every package that objected and only fails if something is genuinely
broken — the line to look for is:

```
  ...and all of them import cleanly, so those bounds are conservative.
```

The setup also installs the ordinary libraries the *reporting* stages need.
They are not needed to run the model, so an environment missing them looks
healthy right up until the reports — which is why the pipeline checks for them
before starting a sweep rather than after one.

## How the GPUs are used

By default the run stays on **one GPU** (`TP=1`, one shard) — note that a
~62 GB reader then keeps only ~10 GB of KV cache on an 80 GB card, so the sweep
runs with few requests in flight. Set the knobs to spread over more cards; the
sweep is then split two ways at once:

- **Across processes** (`NUM_SHARDS`) — each process takes every Nth item of
  the work list and writes its own output file, merged at the end. The
  processes share nothing, so this is close to free: two GPUs really do halve
  the wall clock, and an interrupted process resumes on its own.
- **Within a process** (`TP`) — a model too large for one card is spread over
  several. This is the expensive kind of splitting: the GPUs must talk after
  every layer, so it buys capacity, not speed.

Use the fewest GPUs per process the model fits in, and spend everything left on
more processes. `gemma-4-31b` is ~62 GB — it technically loads on one 80 GB card
but leaves little working memory, so two cards per process is its registry
setting, and four GPUs give two processes:

```bash
GPU_NUM=4 TP=2 bash script/llm-parsing/run_llm_parsing.sh   # two 2-GPU processes
GPU_NUM=4 bash script/llm-parsing/run_llm_parsing.sh        # four 1-GPU processes
```

The driver refuses to start if the layout needs more GPUs than exist, and says
so if some would sit idle — both before any model loads, since an out-of-memory
failure six hours into a sweep is expensive. A run also refuses to resume if the
reader changed, so finish a campaign with the reader it started with. The model
downloads on first use — make sure `HF_HOME` has room for ~62 GB.

## What the second reader does, and doesn't

It **finds and quotes** the answer. That is the whole job.

It never sees the image, never sees the correct answer, and is never asked
whether a number is right — so it cannot flatter or penalise a model. It is also
not trusted to copy numbers: it has to quote the sentence it found the answer
in, and the pipeline re-reads the digits out of that quote itself. A number the
model never wrote therefore cannot end up in the results.

It is not asked *why* an answer is missing — guessing produced confident
mistakes — so a missing answer is recorded as "no answer stated", nothing more.

Two things it is **not allowed** to change:

- **If the regex already found an answer, that answer stands.** The second
  reader can only add answers the regex missed, never revise a published
  number. A bad day for the judge costs a recovery, never a corrupted result.
- **The metrics are the existing ones.** MAE, MRE, IoU and success rate come
  from the same summarizer code that produced the published numbers, pointed at
  the re-parsed records. There is no second definition of any metric.

For T/L and A/D the second reader also pulls out the intermediate steps the
prompt asked for (landmark coordinates, axis endpoints, the computed value).
Those are **saved but not scored**.

Why this is worth a GPU campaign at all — across the 18-model roster, the
answers the regex rejects are not a small or random slice:

| task | responses | rejected by the regex | worst models |
|---|--:|--:|---|
| T/L | 43,938 | 23.0% | Llama-3.2-11B 85%, HuatuoGPT-34B 84%, Qwen2.5-VL-32B 84% |
| A/D | 37,080 | 26.2% | Qwen2.5-VL-32B 90.8% |
| Detection | 415,278 | 6.6% | GLM-4.6V 40.0% |

Most rejected responses contain a complete, correct-looking answer in a
different wrapper — `\boxed{407.02, 325.62}`, `**Answer:** 408.2, 326.4`,
`<final-answer>…</final-answer>`, or ordinary prose.

## What you get

Per task, in the task's Results tree: the judge's raw answers
(`judge-out_<task>_<reader>.jsonl`), and per model a sibling
`llm-parsed_<reader>/` folder next to `parsed/` with the re-parsed records. The
original `parsed/` files are never touched, and the new reports carry a
`__llm-parsed_<reader>` suffix so they cannot overwrite published ones.

Every record is labelled with one of four outcomes:

| outcome | meaning | counts as parsed |
|---|---|:--:|
| answer in expected format | the regex already had it | yes |
| answer in another format | found and verified by the second reader | yes |
| no answer stated | the response never gives one (declined, or stopped early) | no |
| undetermined | the second reader was unusable **and** the regex failed | no |

"Undetermined" is the pipeline's own error rate — recoveries never attempted, so
the reported improvement is a floor, not a ceiling. It currently sits near 0.4%
of all responses. Each record also keeps the sentence the answer was found in,
so any recovered number can be checked by eye.

## Using a different second reader

The headline number — "this much apparent failure was formatting" — is only
worth quoting if it is a property of the responses rather than of one reader's
habits. So the whole thing can be re-run with a different reader and compared:

```bash
bash script/llm-parsing/run_llm_parsing.sh --judges                # what's available
bash script/llm-parsing/run_llm_parsing.sh --judge gemma-4-31b     # run with it
```

| reader | model | notes |
|---|---|---|
| `gemma-4-31b` *(default, and the only one registered)* | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | every current result came from this one; ~62 GB of weights, so 2 GPUs per process |

Every reader writes to its own `judge-out_<task>_<reader>.jsonl` and
`llm-parsed_<reader>/`, and answers **the same work list** — same prompts, same
span checking, same metrics — so two readers' reports differ only by who did the
reading. Nothing from one can reach the other's reports, and `--fresh` under one
reader leaves the other's records alone. Each reader needs its own environment
(see [Environment notes](#environment-notes)); registering a new one happens in
`judge_config.py`'s `JUDGE_MODELS`. On a reader's first outing run the `smoke`
step before committing thirteen hours.

## Starting over: a full new re-judge

Use this when the prompt changed, the roster changed, or you want every answer
produced again from scratch:

```bash
bash script/llm-parsing/run_llm_parsing.sh --fresh          # asks before deleting
bash script/llm-parsing/run_llm_parsing.sh --fresh --yes    # doesn't
```

`--fresh` adds one thing to the normal run: `prep` also deletes the re-parsed
`llm-parsed_<reader>*/` records from the previous run. That matters because the
pipeline rewrites those records file by file, so anything it no longer produces —
a model dropped from the roster, a renamed result file — would otherwise survive
and be read back into the new reports as though it belonged there.

Previous judge answers are **archived, not deleted**: renamed with a `.v1`
suffix (`.v2`, `.v3`… if archives exist), so a run can always be compared
against the one before it. The deletion is announced with a count and waits for
confirmation — see [`--yes`](#skipping-the-confirmation-prompt---yes---y).

If you decide mid-way that you did not want to start over, move an archived
file back to its original name before the judging step; finished answers are
matched by identity, so the run continues from where the previous one stopped.

## Repairing a partial run

Sometimes most answers are fine and a minority came back unusable — most often
because the reader ran out of room and its reply was cut off. Redo only the bad
rows:

```bash
python script/llm-parsing/run_judge_vllm.py \
    --queue Results/<task-folder>/judge-queue_<TASK>.jsonl \
    --out   Results/<task-folder>/judge-out_<TASK>_<reader>.jsonl \
    --model "${JUDGE_MODEL:?}" \
    --redo_invalid \
    --accept_prompt_fp <stamp-of-the-existing-rows> \
    --max_tokens 4096
```

Then rebuild the reports with `run_llm_parsing.sh analyze`.

Two flags carry the whole idea. Every answer is stamped with the exact prompt
and token budget that produced it, and the pipeline refuses to mix stamps —
a changed prompt means the old answers are answers to a different question.
`--accept_prompt_fp` says "the existing rows are still valid", and
`--redo_invalid` says "but re-do the ones that failed". A repair appends to the
same file, and the reports read the last answer per record, so repaired rows
win. The stamp of the rows already on disk is printed in the error message you
get if you leave `--accept_prompt_fp` out.

## Reading the results

Two reports per task land side by side: the published one, and the same report
with a `__llm-parsed_<reader>` suffix. Diff them. The judge report additionally
breaks each model's failures into "wrong format" versus "no answer given", and
reports how often the second reader agreed with the regex on responses the
regex could already read — a free reliability check, which is why every
response is re-read rather than only the failures.

Two caveats when quoting numbers:

- The second reader's output is not bit-reproducible across GPU generations, so
  its own error rate is only comparable within one machine. The saved output
  file is the reproducible artefact — release that, not a re-run.
- A low success rate that survives re-parsing is not automatically a
  measurement failure. If a model's responses pile up against its token limit,
  it ran out of room; check the run's generation settings, not the text.

## Visualizing re-parsed records

The visualization scripts in `script/visualization/` (the three per-sample
response viewers and the radar) read the same per-model folders and take
`--parsed_dirname` (drivers: `PARSED_DIRNAME`) to choose the source. The
drivers default to `llm-parsed_gemma-4-31b`; set `PARSED_DIRNAME=parsed` to
plot the published strict-parse records instead. Output from a non-default
source is kept apart automatically — per-sample figures land in a
`…__llm-parsed_gemma-4-31b` folder and radar figures carry the same suffix in
the filename — so nothing can overwrite a published figure.

## Tests

```bash
for i in 1 2 3 4 5 6 7 9 11; do python unit-test/llm-parsing/test-$i.py; done
python unit-test/llm-parsing/test-8.py llm-parsed_gemma-4-31b   # after a full run
```

They cover, in order: agreement with the published parser, the
anti-fabrication checks, response decoding, naming rules, record ordering,
prompt/schema agreement, the outcome table, the invariants over real output on
disk, and — for the last two — that one reader's answers are never mistaken for
another's. Run them from the repository root with the same interpreter as the
pipeline.
