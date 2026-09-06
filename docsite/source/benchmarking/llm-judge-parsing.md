# LLM-judge output parsing

The published benchmark numbers come from a strict regex parser: an answer only
counts if it sits inside `<answer></answer>` tags (see
[Parsing and summarizing](parsing-and-summarizing.md)). That strictness has a
cost — a model that measures correctly but writes `\boxed{407.02, 325.62}` or
`**Answer:** 408.2, 326.4` is scored as a miss, which mixes *"the model can't
measure"* together with *"the model didn't follow the output format"*.

The LLM-judge pipeline (`script/llm-parsing/`) separates the two. It re-reads
every raw response with a second, offline language model whose only job is to
find the answer wherever it was written. You end up with two versions of every
report: the published one, and a format-robust one. The gap between them is how
much of a model's apparent failure was formatting.

Across the paper's 18-model roster — the now-retired reader — the regex rejected a
substantial, non-random slice:

| Task | Responses | Rejected by the regex | Worst model |
| --- | --: | --: | --- |
| Tumour/Lesion | 43,938 | 23.0% | Llama-3.2-11B, 85% |
| Angle/Distance | 37,080 | 26.2% | Qwen2.5-VL-32B, 90.8% |
| Detection | 415,278 | 6.6% | GLM-4.6V, 40.0% |

Measured effect of the re-parse on that roster: T/L success 77.0% → 89.7%, A/D 73.8% →
90.1%, Detection 93.4% → 98.4%.

The shipped pipeline now runs a **19-model roster** through the `gemma-4-31b` reader
(522,868 responses): T/L 82.6% → 94.8%, A/D 80.1% → 94.2%, Detection 92.3% → 98.5%, over
46,379 / 39,140 / 437,349 responses (strict-regex rejection 17.4% / 19.9% / 7.7%). See
`script/llm-parsing/DESIGN.md` §13 for the full snapshot and §13.1 for the retired reader.
Llama-3.2-11B's T/L success still goes from 14.5% to 97.9% — almost all of its "failure"
was formatting.

## What the judge does — and is not allowed to do

The judge is an [extraction device, never an evaluator]{.mv-accent}. It never
sees the image, never sees the ground truth, and is never asked whether a value
is correct — so it cannot flatter or penalise a model.

It is also not trusted to copy numbers. The judge must **quote** the sentence it
found the answer in; the pipeline then locates that quote in the original
response (span verification) and re-reads the digits out of the quote itself.
A number the model never wrote therefore cannot enter the results, even if the
judge claims otherwise.

Two invariants keep the published numbers safe:

- **If the regex already found an answer, that answer stands.** The judge can
  only add answers the regex missed; it can never revise a published number.
- **The metrics are the existing ones.** MAE, MRE, IoU and SuccessRate come from
  the same summarizer code that produced the published numbers, just pointed at
  the re-parsed records. There is no second definition of any metric.

For T/L and A/D the judge also extracts the intermediate steps the prompt asked
for (landmark coordinates, axis endpoints, the computed value). Those are saved
but **not scored**.

## What you get

Re-parsed records land in a sibling `llm-parsed_<reader>/` folder (e.g.
`llm-parsed_gemma-4-31b/`) next to each model's `parsed/`; the original `parsed/` files are
never touched. The directory is always reader-suffixed — a bare `llm-parsed/` is never
produced. Summary reports gain a matching `__llm-parsed_<reader>` suffix so they sit beside
the published ones instead of overwriting them.

Every record carries one of four outcomes:

| Outcome | Meaning | Counts as parsed |
| --- | --- | :-: |
| answer in expected format | the regex already had it | yes |
| answer in another format | found and span-verified by the judge | yes |
| no answer stated | the response never gives one (declined, or stopped early) | no |
| undetermined | the judge was unusable **and** the regex failed | no |

"Undetermined" is the pipeline's own error rate (≈0.6% of all responses for the current
`gemma-4-31b` reader; ≈0.4% for the paper's retired reader) — recoveries that were never attempted, so the reported improvement
is a floor, not a ceiling. Each recovered record also keeps the quoted sentence,
so any recovered number can be checked by eye.

## Running the pipeline

The pipeline lives in `script/llm-parsing/` and runs in four resumable stages:
build per-task work queues from the model roster, sweep them on GPU with offline
vLLM, verify spans and merge into `llm-parsed_<reader>/`, then report. The judge reader
comes from a registry; the current (and only) entry is
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) —
~62 GB of bf16 weights. Its registry entry recommends 2 GPUs per process, but the driver
forces `TP=1` unless you set `TP` (`GPU_NUM` only adds shards — it never widens one), so a default run loads the whole reader onto
one card — which leaves little KV cache, and OOMs below 80 GB.

### 1. Build the judge environment

The judge needs its own Python environment (a newer vLLM than the evaluation
code pins, and Transformers 5.x, which cannot resolve against vLLM's declared
bounds in a single pip pass — the install runs in two phases and prints
*expected* dependency-conflict errors, then imports every package that objected
to prove the bounds were conservative):

```bash
bash script/llm-parsing/setup_judge_env.sh    # builds <repo>/.cache/judge-env_gemma-4-31b
export PYTHON=<target>/bin/python             # the script prints this line
```

The setup ends with a GPU allocation check, so a bad CUDA/PyTorch pairing fails
here, in minutes — not thirteen hours into a sweep. If your driver is older than
CUDA 12.8, point the build at a matching wheel index:
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash script/llm-parsing/setup_judge_env.sh`.

### 2. Run everything

```bash
bash script/llm-parsing/run_llm_parsing.sh              # all stages, in order
bash script/llm-parsing/run_llm_parsing.sh --list       # show the steps, run nothing
bash script/llm-parsing/run_llm_parsing.sh analyze      # only the CPU stages + reports
bash script/llm-parsing/run_llm_parsing.sh --fresh      # start over (archives old judge output)
```

The driver stops at the first failed check and is resumable — finished work is
skipped, so it can be killed and restarted freely. All settings are optional
environment variables; the ones you are most likely to touch:

| Variable | Meaning |
| --- | --- |
| `PYTHON` | interpreter with the judge's PyTorch + vLLM (from step 1) |
| `JUDGE` | which registered reader to use (default `gemma-4-31b`) |
| `TASKS` | default `TL AD Detection` |
| `TP` | GPUs one judge process spans. Default **1** (single GPU), overriding the reader's registry value of 2; raise it only to fit the weights |
| `NUM_SHARDS` | independent judge processes. Default **1** |
| `GPU_NUM` | total GPUs to use; derives `NUM_SHARDS = GPU_NUM / TP` (must divide evenly) — the one knob for spreading over more cards |
| `TASK_DIR_<task>` / `ROSTER_YAML_<task>` | re-point one task at a different Results tree / roster YAML — how the OOD splits are judged |
| `MOCK=1` | exercise the whole pipeline on CPU with a stand-in — never report its numbers |

The defaults are single-GPU (`TP=1`, `NUM_SHARDS=1`). Set `GPU_NUM` to spread the
work: the list is split across `NUM_SHARDS` independent processes (cheap,
near-linear speedup) and each process spans `TP` GPUs (expensive tensor
parallelism, capacity not speed). Rough cost on two
H100s: a one-off model download, minutes of CPU preparation, ~13 hours of GPU
for the full sweep, and ~1 hour of CPU for the reports.

:::{tip}
Every judge answer is stamped with a fingerprint of the exact prompt and token
budget that produced it, and the pipeline refuses to mix stamps — a prompt edit
invalidates the queues instead of silently blending two prompts into one report.
To repair a minority of bad rows without re-judging everything, see "Repairing a
partial run" in the
[pipeline README](https://github.com/YongchengYAO/MedVision/blob/master/script/llm-parsing/README.md).
:::

## Summarizing the re-parsed records

The standard summarizers (see
[Parsing and summarizing](parsing-and-summarizing.md)) read judge output through
two flags, so the strict and format-robust columns share one code path:

```bash
python -m medvision_bm.benchmark.summarize_TL_task \
    --task_dir Results/MedVision-TL-v2-CoT -p 32 --skip_model_wo_parsed_files \
    --removed_samples_dir Data/Datasets \
    --parsed_dirname llm-parsed_gemma-4-31b --resps_key LLM_filtered_resps
```

| Flag | Meaning |
| --- | --- |
| `--parsed_dirname <dir>` | per-model subfolder to read: `parsed` (default) or an `llm-parsed*` folder |
| `--resps_key <key>` | record field holding the prediction: `filtered_resps` (regex) or `LLM_filtered_resps` (judge). The summarizer aborts if the key is absent, rather than silently dropping every record. |
| `--models <names…>` | restrict the run to the configured roster (a results tree also carries superseded `_bugfix-*` variants and baselines) |

The driver's `analyze` step runs these for you; the flags matter when you
aggregate by hand.

## Reading the results

Two reports per task land side by side — the published one and its
`__llm-parsed_<reader>` twin. Diff them. The judge report additionally splits each
model's failures into *wrong format* vs *no answer given*, and reports how often
the judge agreed with the regex on responses the regex could already read — a
free reliability check, which is why every response is re-read rather than only
the failures.

:::{warning}
Judge output is **not reproducible run to run**, even with greedy decoding on
identical hardware — the cause is numerical non-determinism in the inference
stack, not sampling. The pinned vLLM 0.19.0 does ship a batch-invariance switch
(`VLLM_BATCH_INVARIANT=1`) that the pipeline deliberately leaves off, at a throughput
cost. Treat the saved `judge-out_*.jsonl` files as the artefact of record: release those, don't re-run and expect the same rows back. The full
root-cause analysis is in
[`docs/LLM-Judge-Reproducibility.md`](https://github.com/YongchengYAO/MedVision/blob/master/docs/LLM-Judge-Reproducibility.md)
(measured on a since-retired reader; its headline rates do not transfer to the
current one, but the operating rules do).
:::

A low success rate that survives re-parsing is not automatically a measurement
failure either: if a model's responses pile up against its token limit it ran
out of room, which must be checked against the run's generation settings.

## Further reading

- [`script/llm-parsing/README.md`](https://github.com/YongchengYAO/MedVision/blob/master/script/llm-parsing/README.md)
  — full user guide: environments, GPU layout, alternative readers, repairs.
- [`script/llm-parsing/DESIGN.md`](https://github.com/YongchengYAO/MedVision/blob/master/script/llm-parsing/DESIGN.md)
  — module map, the outcome decision table, verification tiers, edit hazards.
- [Clinical Decision Agreement](clinical-decision-agreement.md) can score the
  judge re-parse instead of the strict parse, giving a format-robust view of
  clinical agreement as well.
