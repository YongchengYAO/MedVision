# Why LLM-judge output is not reproducible

> **Scope: this is a historical record.** Every measurement below was taken under
> offline vLLM 0.11.0 on the judge reader **retired on 2026-08-17** — a quantized
> Mixture-of-Experts model. The pipeline now runs `google/gemma-4-31B-it` (see
> `script/llm-parsing/DESIGN.md` §16). The numbers here are *not* restatable as
> properties of the current reader, and §5's MXFP4/bf16 discussion describes
> machinery that no longer exists. What still applies is marked below.
>
> **What transfers to the current reader:** the framing in §1 (greedy ≠
> reproducible), §3.2 (kernels are not batch-invariant), §3.3 (prefix caching),
> §6 (blast radius) and **all four operating rules in §8**. Those are properties
> of vLLM and of the pipeline, not of any one model.
>
> **What does not transfer:** §3.1, the *dominant* mechanism measured here, is
> Mixture-of-Experts routing. `gemma-4-31B-it` is **dense**, so it should drift
> materially less. That is a prediction from architecture — nobody has measured
> the current reader's flip rate. Do not quote 12.8% / 63.0% / 82.9% for it.

Root-cause report for the `script/llm-parsing/` pipeline (the judge reader retired
on 2026-08-17, offline vLLM 0.11.0).

**Verdict.** Judge output was not reproducible **run to run on the same GPU**, and
the cause is **numerical non-determinism amplified by a Mixture-of-Experts router**
— not sampling randomness. The pipeline decodes greedily, so there is no random
draw to blame. The published benchmark numbers are unaffected by design; what
moves is which rows the judge *recovers*.

---

## 1. What was believed, and why it was wrong

The pipeline's own notes attributed re-run differences to "pure RNG" and framed
the problem as one of *GPU generations*. Both framings are wrong:

| claim | status |
|---|---|
| "flips are sampling randomness / RNG" | **False.** `JUDGE_TEMPERATURE = 0.0`, `JUDGE_TOP_P = 1.0` ⇒ greedy argmax. There is no draw. `JUDGE_SEED = 1024` seeds a sampler that never samples, so it is inert and offers false comfort. |
| "output differs across GPU generations" | **True but misleading.** It also differs on the *same* GPU, in the same hour. And part of the cross-machine difference is not non-determinism at all — see §5. |

---

## 2. Evidence

### 2.1 Same prompt, same budget, same machine — 12.8% agreement

`judge-out_*.jsonl` uses append semantics, so a repair pass leaves the original
row beside the re-judged one. That makes the production files a natural A/B
experiment: rows sharing a `qid` **and** a `prompt_fp` were judged twice under an
identical prompt and identical decode budget.

| task | same-prompt duplicate pairs | identical verdict | `judge_status` changed |
|---|--:|--:|--:|
| TL (`dd3c7fb2…`, budget 3072) | 1,306 | **167 (12.8%)** | 1,092 |
| AD (`56de1c8d…`, budget 3072) | 833 | **109 (13.1%)** | 724 |

Two independent tasks, the same answer to one decimal place. Nothing about the
prompt, the budget, the queue or the code differed between the two passes.

### 2.2 The earlier controlled experiment

Re-judging all 15,479 invalid Detection rows on the same H100s, same day:
**63.0%** recovered at the *unchanged* 256-token budget (pure run-to-run
variation), versus **98.1%** at 512 — so the budget's attributable effect was
+35.1 points over the variation floor.

> Caveat: the control-arm output file is no longer on disk, so this figure is
> reported from the project record and is **not currently re-verifiable**. §2.1 is,
> and was measured independently for this report.

---

## 3. Why greedy decoding still produces different text

Greedy decoding is deterministic *given identical logits*. The logits are not
identical between runs, and this model turns tiny logit differences into large
output differences.

### 3.1 MoE routing is a discrete amplifier (dominant mechanism)

That reader was `num_local_experts = 32`, `num_experts_per_tok = 4`. Each token's
FFN path is chosen by a **top-k over router logits**. When two experts sit near a
tie, a perturbation of ~1e-6 changes *which expert runs*, and the block's output
then changes by O(1) — not by O(1e-6). One flipped route early in a 24-layer stack
changes every subsequent token.

A dense model of the same size would drift far less: small input perturbations
would stay small. The router converts a rounding difference into a branch.

### 3.2 The logits differ because kernels are not batch-invariant

vLLM 0.11.0 as installed ships **no batch-invariance module and no such env
switch** (verified by scanning the installed package). Reduction order in matmul
and attention depends on the batch shape, and under the V1 engine the batch shape
depends on what else is in flight — which depends on shard layout, chunk
boundaries and arrival order. None of those are pinned across runs.

### 3.3 Prefix caching adds a second path

`enable_prefix_caching=True`. The system prompt is shared by every row in a
`step_key` group, so whether a given row's prefix KV is **recomputed or reused**
depends on batch order and cache eviction. The two paths are not bitwise equal, so
the same row can take numerically different routes on different runs.

> Confidence: §3.1–3.3 are mechanisms **verified to be present** in this
> configuration. Their *relative* contribution is reasoned, not ablated — isolating
> them requires a GPU, which the analysis machine does not have. See §7.

---

## 4. The measured numbers overstate the instability

Both headline figures (63.0% and 12.8%) were measured on rows **selected for
having failed**. Selecting on a noisy binary outcome and re-running regresses to
the mean, so these are an **upper bound on instability, not an estimate of it**.

There is also a boundary effect. At an under-sized budget, a response whose
natural length sits near the cap flips valid/invalid on a few tokens of drift —
which is exactly why the effect looked catastrophic at Detection's 256-token
budget and why a generous budget suppresses the *symptom* without touching the
*cause*. (The budget is now 4096 for all tasks; only ~109 non-loop rows in the
entire 496,296-response corpus ever reached a cap.)

The unbiased quantity — the flip rate on a **random** sample — has never been
measured.

---

## 5. Cross-machine differences are partly not non-determinism

> **Obsolete as machinery, still valid as a caution.** The per-pod checkpoint
> switch described here was removed with the retired reader; the current reader
> ships one bf16 release and there is no second materialization to choose between.
> The *distinction* the section draws is what to keep.

"Not reproducible across GPU generations" conflated two different things:

- **Hopper** ran the checkpoint in native MXFP4 (~13 GB).
- **Ampere** had no sm_80 MXFP4 kernel, so the pipeline ran a **dequantized bf16
  checkpoint** (~40 GB).

Those are *different weights*. Output differences between them are an expected
consequence of running a different numeric representation of the model — not
run-to-run noise. Only same-hardware, same-checkpoint differences are the
phenomenon this report is about. That separation still matters whenever a reader
is served in more than one precision.

---

## 6. Blast radius: what this does and does not affect

**Cannot move a published number.** The decision table gives the strict regex
parser unconditional priority: wherever it succeeded, its answer is written
verbatim regardless of what the judge said. A judge flip can therefore cost a
*recovery* — it can never corrupt a published value. This is what makes the
judge-invalid rate a quality metric rather than a correctness risk.

**Affects, and how to treat it:**

| quantity | treatment |
|---|---|
| per-row judge verdict | not reproducible; never diff two runs row-by-row and call the delta a regression |
| judge-invalid rate | comparable **only** within one machine and one checkpoint |
| a re-run's "recovery" | never attributable to a code change without a same-raw A/B (`reparse_judge_out.py`) or a same-day control arm at the old setting |
| aggregate ΔSR over 496K rows | far more stable than the per-row figure implies, because the unstable population is the failing tail — but the aggregate's run-to-run spread has not been measured, so do not quote a stability figure for it |

**The reproducible artifact is the output file.** `judge-out_*.jsonl` is what
should be released and cited; Stages 2–4 are pure CPU functions of it and *are*
byte-reproducible.

---

## 7. What would make it reproducible, and what it costs

| option | effect | cost |
|---|---|---|
| generous decode budget (done: 4096) | removes the truncation boundary that turns drift into valid/invalid flips | none — decode stops at EOS |
| pin shard count and `--chunk_rows` for any intended A/B | removes one source of batch-shape variation | free, but not sufficient |
| eager execution, prefix caching off | removes CUDA-graph and cache-path variation | slower, and **not exposed**: `enable_prefix_caching=True` is hardcoded in `run_judge_vllm.run_gpu` and no `--enforce_eager` flag exists, so this means editing that function. Still not a proof of bitwise equality |
| batch-invariant kernels | the only real fix | not available in vLLM 0.11.0 |
| release the judge-out file | sidesteps the problem entirely | none — this is the current policy |

**The experiment still worth running** (needs a GPU): re-judge a *random* sample
of ~2,000 rows twice under identical settings and measure the flip rate. That
gives the unbiased instability figure §4 says we lack. A second arm with prefix
caching disabled would separate §3.3 from §3.1–3.2.

---

## 8. Operating rules

1. Never compare judge-invalid rates across machines or checkpoints.
2. Never credit a re-run's recovery to a code change without a same-raw A/B or a
   same-day control arm.
3. Never sample a probe in roster order — a 2,000-row roster-ordered probe once
   suggested 82.9% where the full population gave 63.0%. Roster order is
   model-biased.
4. Treat `judge-out_*.jsonl` as the artifact of record. Re-deriving reports from
   it is reproducible; re-judging is not.

---

## 9. Reproducing this analysis

All of §2.1 is CPU-only and reads the production files:

```python
# Group by (qid, prompt_fp). A group with >1 row is one sample judged more than
# once under an IDENTICAL prompt and budget -- the A/B pair.
#
# Group by qid alone and you get nothing: every duplicated qid also carries a row
# from the other budget (the repair pass), so "all rows share one fingerprint"
# excludes exactly the pairs of interest. The fingerprint belongs in the KEY.
import collections, json

def fa_key(r):
    fa = r.get("final_answer")
    if not isinstance(fa, dict):
        return ("NONOBJ", str(fa)[:40])
    return (fa.get("status"), tuple(fa.get("values") or []), (fa.get("span") or "")[:80])

by = collections.defaultdict(list)
for line in open("Results/MedVision-TL-v2-CoT/judge-out_TL.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    by[(r["qid"], r["prompt_fp"])].append((r["judge_status"], fa_key(r)))

groups = [v for v in by.values() if len(v) > 1]
same = sum(1 for v in groups if v[0] == v[-1])
print(len(groups), same, same / len(groups))   # -> 1306 167 0.1279...
```

Config facts in §3 come from `script/llm-parsing/judge_config.py`
(`JUDGE_TEMPERATURE`, `JUDGE_TOP_P`, `JUDGE_SEED`),
`script/llm-parsing/run_judge_vllm.py` (`tensor_parallel_size`,
`enable_prefix_caching`), and the judge checkpoint's `config.json`
(`num_local_experts`, `num_experts_per_tok`).

---

*Analysis date 2026-08-12. §2.1, §3 config facts and §5 verified directly; §2.2
reported from the project record and not currently re-verifiable; mechanism
ranking in §3 is reasoned, not ablated.*
