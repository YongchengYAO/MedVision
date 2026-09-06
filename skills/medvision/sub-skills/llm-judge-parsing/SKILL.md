---
name: llm-judge-parsing
description: "Runs and explains the MedVision LLM-as-judge second-pass parsing pipeline (benchmark step 4), which re-reads every model response with a second language model to find answers the strict <answer>...</answer> regex missed and produces a format-robust twin of each report. Covers the run_llm_parsing.sh driver steps (prep, stage0, smoke, pilot, full, analyze), the PYTHON / TASKS / TASK_DIR_* / ROSTER_YAML_* / JUDGE / MOCK environment knobs, the judge virtual environment and its vLLM pin, roster YAMLs, the judge-queue / judge-out / llm-parsed_<reader> artifacts, the prompt-fingerprint lock, span verification and the decision table, and reproducibility caveats. Use when a user asks to run or resume the LLM judge, add a model to a finished judge campaign, judge an OOD or custom Results tree, repair a partial judge sweep, interpret summary_judge_task or llm-parsed_<reader> reports, or debug a judge gate failure."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision LLM-as-judge output parsing (benchmark step 4)

The benchmark's strict parser accepts an answer **only** inside
`<answer>…</answer>`. Everything else is scored as a miss, which mixes *"the
model cannot measure"* with *"the model ignored the output format"*. This
pipeline re-reads **every** response with a second language model whose only job
is to find the answer wherever it was written, then re-scores with the *same*
metric code. The result is two reports per task: the published one and a
format-robust twin suffixed `__llm-parsed_<reader>`. Their difference is the
formatting share of apparent failure.

Measured on the 19-model roster (522,868 responses, reader `gemma-4-31b`):
T/L success 82.6% → 94.8%, A/D 80.1% → 94.2%, Detection 92.3% → 98.5%.

**Use this sub-skill when** you need to run, resume, extend or debug a judge
campaign; judge a custom or OOD Results tree; add one model to a finished
campaign; repair a partial sweep; or read a `summary_judge_task…txt` /
`llm-parsed_<reader>/` artifact. **Not** for the strict parse or the metric
definitions themselves — those are `../results-parsing-and-metrics/SKILL.md`.

> The pipeline ships in the **repository checkout** under `script/llm-parsing/`;
> it is not part of the installed `medvision_bm` package. A checkout is a
> prerequisite for the driver. In a checkout, `script/llm-parsing/README.md` is the
> operator guide and `script/llm-parsing/DESIGN.md` the full design record (measured
> snapshots, rejection taxonomy). The two bundled `scripts/` helpers below run
> standalone.

## Steps

The driver exposes six steps; with no step named, **all six run in order** — so
the destructive one runs and prompts. Named steps run in the order given;
`--from <step>` runs that step and every later one.

| step | what it does | resources | safety |
|---|---|---|---|
| `prep` | archive stale judge output (`.v1`, `.v2`…), delete stale queues; with `--fresh` also delete the reader's `llm-parsed*/` dirs | seconds | **DESTRUCTIVE**, the only interactive step (`Proceed? [y/N]`); auto-declines without a terminal |
| `stage0` | build the work queues; replay the strict parser as a gate; roster gate; count gate | CPU, minutes (~1.1 GB written) | safe; a failed gate retracts the queue |
| `smoke` | judge 200 evenly spaced rows into a temp dir; assert ≥95% schema-valid and spans quoted | **requires GPU**, minutes | safe; writes nothing permanent |
| `pilot` | `PILOT_LIMIT` (100) rows per file, end to end + invariants, into `-limitN` artifacts | **requires GPU**, ~30 min | safe; limit is in the directory name |
| `full` | the judge over the whole roster, every task in `TASKS` | **requires GPU**, ~13 h on two H100s | resumable by row identity |
| `analyze` | apply the judge, re-summarize, judge report, record invariants | CPU, ~1 h | rewrites `llm-parsed*/` and `__llm-parsed_*` reports only |

`parsed/` is never written. The strict regex always wins where it succeeded, so a
bad judge day costs a *recovery*, never a published number.

## Quick start

```bash
# 0. the judge needs its OWN interpreter (newer vLLM than the eval stacks pin)
export PYTHON=<judge-env>/bin/python
python scripts/check_judge_env.py --python "$PYTHON" --repo-root <repo>

# 1. what would run, and what it costs
bash <repo>/script/llm-parsing/run_llm_parsing.sh --list

# 2. the main campaign, everything in order (prep will ask before deleting)
bash <repo>/script/llm-parsing/run_llm_parsing.sh

# 3. a custom or OOD tree: same driver, three env overrides, no wrapper script
TASKS="TL" TASK_DIR_TL=Results/<my-tree> \
ROSTER_YAML_TL=script/llm-parsing/config-my-roster.yaml \
bash <repo>/script/llm-parsing/run_llm_parsing.sh stage0 smoke full analyze

# 4. CPU only: rebuild the reports from an existing judge-out file
bash <repo>/script/llm-parsing/run_llm_parsing.sh analyze
```

## Route

| you need | read / run |
|---|---|
| How the stages fit together, every driver flag, every environment variable with its default, the artifact names and row schemas | `references/pipeline.md` |
| A concrete campaign: your own tree, adding one model, the OOD splits, changing the main campaign, `--fresh`, repairs, another reader, reading and plotting the results, `MOCK=1`, `--yes` | `references/recipes.md` |
| Exact CLI surface of the driver, the five stage modules, the registry and the repair tool, plus the library modules | `references/cli-reference.md` |
| Building the judge virtual environment, the reader registry and its pins, the two-interpreter trap, token/cache hygiene | `references/judge-environment.md` |
| The decision table, span-verification tiers, the prompt-fingerprint lock, invariants and gates, measured snapshots, reproducibility, edit hazards — read before changing anything in the pipeline or explaining a number | `references/design-notes.md` |
| A gate failure, an abort message, a number that looks wrong, an unattended run that stopped | `references/troubleshooting.md` |
| Is this interpreter the judge env? (vllm vs the pin, transformers major, torch/CUDA, GPU alloc, CPU-stage imports, `medvision_ds`) | `python scripts/check_judge_env.py --help` |
| Build a roster YAML from a Results tree, skipping everything Stage 0 would reject | `python scripts/make_roster_yaml.py --help` |

## Load-bearing facts

- **`PYTHON` must point at the judge environment.** One interpreter runs every
  stage. Unset, the first `python3` on `PATH` wins — which is how a pod with four
  GPUs reports "no CUDA device". The judge needs `vllm==0.19.0` with a Transformers 5.x
  line, which **most** eval envs cannot host — eight of the ten pin vLLM 0.10.0-0.14.0. Two are close: `requirements_eval_gemma4.txt` pins the
  identical 0.19.0 stack, and `requirements_eval_glm4v.txt` pins 0.19.1 (near, but still not the judge
  environment). Every other eval env is the wrong interpreter for the judge.
- **Queues are fingerprint-locked.** Every queue and judge-out row carries
  `prompt_fp`, a hash of the *rendered* prompt plus the response-window constants
  and the task's `max_tokens`. Editing the prompt, the skeleton or the budget
  moves the stamp, Stage 1 refuses the old queues, and `stage0` must rebuild
  ~1.1 GB. Reader identity is deliberately **not** in the stamp, so a second
  reader reuses the same queues.
- **Resume is by `qid`**, which has no prompt component — so growing the roster
  re-judges only the new model's rows, and a killed sweep restarts for free.
- **`MOCK=1` numbers are never reportable.** It is a regex stand-in for wiring
  checks on a CPU box. It writes a `.MOCK.jsonl` judge-out, but Stage 2 writes
  into the *same* `llm-parsed_<reader>/` directory a real run uses — point
  `TASK_DIR_<task>` at a scratch copy.
- **`llm-parsed_<reader>/` records carry `LLM_filtered_resps`** and have
  `filtered_resps` removed. Summarize them with
  `--parsed_dirname llm-parsed_<reader> --resps_key LLM_filtered_resps`; the
  summarizer aborts without the second flag rather than reporting zeros.
- **The judge is an extraction device, not an evaluator.** It never sees the
  image or the target, is never asked whether a value is right, and must quote
  the sentence it found the answer in — the pipeline re-reads the digits out of
  that quote with the benchmark's own number regex, so a number the model never
  wrote cannot be scored.
- **Judge output is not bit-reproducible**, even run-to-run on one GPU. Release
  `judge-out_*.jsonl` — Stages 2–4 are byte-reproducible functions of it.

## Boundaries

- `../results-parsing-and-metrics/SKILL.md` owns the strict parse
  (`parse_outputs`) and the summarizers, including the metric definitions; this
  pipeline **consumes** its `parsed/` records and hands back
  `llm-parsed_<reader>/` for it to re-summarize. No metric is defined here.
- `../benchmark-evaluation/SKILL.md` owns the evaluation runs that produced the
  responses — go there when a low success rate survives re-parsing (a token-budget
  wall, not a measurement failure).
- `../environment-setup/SKILL.md` owns the evaluation/SFT installs and their
  pins; the judge environment is separate and documented in
  `references/judge-environment.md`. Never mix the two.
- `../analysis/SKILL.md` owns CDA, process accuracy and equation accuracy. The
  judge persists T/L and A/D intermediate steps but **does not score** them.
- Figure entry points that accept `--parsed_dirname` / `PARSED_DIRNAME`:
  `../../references/visualization-catalog.md`.
- Shared vocabulary (task names, dataset configs, `Results/` layout):
  `../../references/concepts-and-glossary.md`. Cross-cutting failures:
  `../../references/troubleshooting.md`.

## Safe operating rules

- **Name your steps.** `stage0 full analyze` for a resume or a re-pointed tree.
  Running with no step list includes `prep`, which archives and deletes.
- Run `smoke` before the first `full` of any new setup or new reader: minutes of
  GPU to avoid thirteen wasted hours.
- `TASKS`, `TASK_DIR_<task>` and `ROSTER_YAML_<task>` are three independent
  overrides — set all three when re-pointing, and run **one invocation per tree**,
  sequentially. There are no wrapper scripts, by design.
- Never `pip install` into an evaluation environment to satisfy a judge import,
  and never install judge requirements into one; the pins conflict.
- Treat `Results/` as data. Do not hand-edit judge JSONLs; use the repair tools,
  which append and refuse destructive merges.
- GPU steps need a real GPU: `SKIP_GPU_CHECK=1` bypasses a preflight that exists
  to fail cheaply, and is not a way to run on a CPU box.
- Do not update `EXPECTED_ROSTER_COUNTS` to silence the Stage-0 count gate; print
  the true total with the `--dry_run` build and change it deliberately.
