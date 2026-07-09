# Literature review: loss masking in LLM supervised fine-tuning (SFT)

**Question:** In supervised fine-tuning, should you mask the non-assistant / prompt tokens and
compute the loss only on the assistant completion?

**Bottom line:** Prompt masking (completion-only loss) is the mainstream default and a safe
baseline, but it is **not universally optimal**. The deciding variable is measurable — the
**generation ratio `Rg` = completion length / prompt length**. For long-completion data
(`Rg ≥ 1`, e.g. CoT / reasoning), masking vs. not-masking is statistically irrelevant to
downstream performance, so masking is a fine default. For short-completion data (`Rg < 1`),
full masking is empirically the *worst* choice and a small nonzero prompt weight wins.

> Methodology note: this review was produced by a multi-source, adversarially-verified research
> pass (5 search angles → 20 sources fetched → 90 candidate claims → 25 verified by 3-vote
> adversarial voting; 25/25 confirmed, 0 refuted). Every claim below is backed by a primary
> source (official docs/source code or a peer-reviewed paper).

---

## 1. Terminology — these all name the same operation

"Completion-only loss" / "prompt masking" / "assistant-only loss" / "instruction masking" /
`train_on_inputs=False` all mean: set the prompt / non-assistant token labels to the `-100`
ignore index so gradient flows only from response tokens.

The research literature generalizes the binary mask into a continuous **prompt-loss-weight
(PLW)** — `PLW = 0` is full masking, `PLW = 1` is full-sequence training. A further variant
(Weighted Instruction Tuning) uses independent weights `λp` (prompt) and `λr` (response).

HuggingFace TRL exposes two flags:
- `completion_only_loss` — for prompt-completion datasets.
- `assistant_only_loss` — for conversational (multi-turn) datasets, driven by the
  `{% generation %}` / `{% endgeneration %}` chat-template keywords.

*Sources: TRL docs; arXiv 2401.13586; 2405.14394; 2507.07817. Confidence: high.*

## 2. Framework default (HuggingFace TRL — verified; version-sensitive)

- **Masking is the default** for prompt-completion datasets. `completion_only_loss` defaults to
  `None`, which auto-resolves to completion-only loss for prompt-completion data and full-sequence
  loss for plain language-modeling data. Set `completion_only_loss=False` to train on the whole
  sequence.
- **Multi-turn:** `assistant_only_loss=True` masks **all** non-assistant turns (user *and* system),
  not just the final turn. It defaults to `False` and must be explicitly enabled.

> TRL docs, verbatim: *"By default, the trainer computes the loss on the completion tokens only,
> ignoring the prompt tokens. If you want to train on the full sequence, set
> `completion_only_loss=False`."*

*Sources: TRL docs v1.7.1 and v0.21.0; `sft_config.py`. Confidence: high.*

> **Coverage gap:** this review verified **TRL only**. It did not establish the concrete defaults /
> enabling flags for Axolotl (`train_on_inputs`), Llama-Factory, Unsloth, or torchtune. (Moot for
> MedVision, which uses custom collate functions.)

## 3. Does masking help? — governed by the generation ratio `Rg`

From **"Instruction Fine-Tuning: Does Prompt Loss Matter?"** (Huerta-Enochian & Ko, EMNLP 2024;
arXiv 2401.13586). `Rg = completion length / prompt length`.

- **Long-completion data (`Rg ≥ 1`** — CoT, reasoning traces, long answers): prompt-loss weight has
  **no statistically significant** relationship with performance. Masking (`PLW=0`) and full-sequence
  training (`PLW≈1`) are effectively equivalent; the paper states prompt loss *"can be safely ignored
  for many datasets."* (Null could not be rejected on AlpacaData `Rg=3.27`, AlpacaDataCleaned `Rg=7.83`.)
- **Short-completion data (`Rg < 1`** — short answers, long prompts): full masking (`PLW=0`) was the
  **worst** configuration. Performance follows a significant negative (concave-down) quadratic in PLW,
  so a **small nonzero prompt weight beats both** full masking and full-sequence training. Regression
  optimum **`PLW ≈ 0.242`**; small values (0.01–0.5) best for multiple-choice / short-generation,
  `≈1.0` best for long-generation. Mechanism: a small prompt weight acts as a regularizer keeping
  weights near the pretrained model, reducing overfitting.

*Confidence: high (3-0 verified).*

## 4. When training on inputs (NOT masking) genuinely helps

From **"Instruction Tuning With Loss Over Instructions"** (Shi et al., NeurIPS 2024; arXiv 2405.14394).
Instruction Modelling (IM) computes loss over both instruction and completion tokens. It beats the
completion-only default under **two checkable conditions**:
1. **High instruction-to-output length ratio** (long instructions, brief outputs).
2. **Small datasets** (few examples — the Superficial Alignment Hypothesis regime).

It adds little when outputs are long (ratio ~0.5, like Tulu-v2) or datasets are large (100k+).
Across 21 benchmarks IM improved MMLU, TruthfulQA, HumanEval, MT-Bench, AlpacaEval. IM is **not
universally superior** — it primarily reduces overfitting.

A third study, **Weighted Instruction Tuning (WIT)** ("On the Effect of Instruction Tuning Loss on
Generalization", Chatterjee et al., TACL 2025; arXiv 2507.07817), found the standard `λp=0, λr=1`
masking *"is never the optimal choice"* in its sweep — a nonzero optimal prompt weight in **56%** of
75 settings (avg ~6.55% relative gain), and better starting points for subsequent DPO. **Caveat:**
masking was still optimal in the other 44%, and much of WIT's gain may come from *lowering the
response weight below 1*, not from adding prompt weight.

*Confidence: high (3-0 verified).*

## 5. Multi-turn chat specifics & pitfalls

- Mask **all** non-assistant turns (`assistant_only_loss=True`), not just the final turn.
- The mechanism relies on the chat template containing `{% generation %}` / `{% endgeneration %}`
  Jinja keywords (which power transformers' `return_assistant_tokens_mask`). TRL auto-patches known
  families (e.g. Qwen3); **for other models a template lacking those keywords silently produces no
  assistant mask** ("template drift").
- Known silent-failure bugs: ignored under `use_liger_kernel=True` (TRL issue #3781) and when
  sequence length exceeds `max_length` (#3927).
- **EOS handling** under multi-turn masking (ensuring the end-of-turn token stays supervised so the
  model learns to stop) was named in the question but **not resolved** by the verified sources — see
  open questions.

*Confidence: high (3-0 verified).*

---

## Practical recommendation

| Situation | Recommendation |
|---|---|
| **Default / unsure** | Mask the prompt (completion-only loss). It's the framework default and never badly hurts. |
| **Long outputs / CoT / large datasets (`Rg ≥ 1`)** | Mask; don't bother tuning prompt loss — the choice is statistically irrelevant. |
| **Short outputs relative to prompts, and/or small datasets** | Don't fully mask. Use a small nonzero prompt weight (~0.1–0.25) or Instruction Modelling. |
| **Multi-turn chat** | Mask all non-assistant turns; verify the `{% generation %}` keywords are present in the template. |

---

## Implications for MedVision

MedVision's Gemma/Qwen SFT is **long-completion CoT** — the detection-CoT sample measures ~170
completion tokens with reasoning, and TL/AD are longer. That puts the data at **`Rg ≈ 1` or above**,
i.e. exactly the regime where the EMNLP 2024 result says masking-vs-not is *statistically irrelevant
to downstream accuracy*.

Consequences for the completion-only masking work on the Gemma collates
(`gemma4_utils.py` / `medgemma_utils.py`):
- The change is **unlikely to move detection/TL/AD accuracy** materially either way. Its justification
  is **loss-objective consistency with the Qwen collates and comparable `train/loss` curves**, not an
  expected accuracy gain.
- It is **not** a fix for the degenerate `<think> <answer>` repetition loop — that is a train/eval
  prompt-shape mismatch (a CoT-trained model fed the non-CoT prompt), independently established and
  reproduced in recorded data. Masking is untested against it (perfectly confounded in existing runs).
- The plan's deliberate choice to keep the **closing turn marker in the loss** is the correct handling
  of the EOS-supervision pitfall from §5 — the model still learns to stop.

---

## Sources

Primary:
- Huerta-Enochian & Ko, *Instruction Fine-Tuning: Does Prompt Loss Matter?* EMNLP 2024 — https://arxiv.org/abs/2401.13586
- Shi et al., *Instruction Tuning With Loss Over Instructions* (Instruction Modelling), NeurIPS 2024 — https://arxiv.org/abs/2405.14394
- Chatterjee et al., *On the Effect of Instruction Tuning Loss on Generalization* (Weighted Instruction Tuning), TACL 2025 — https://arxiv.org/pdf/2507.07817
- HuggingFace TRL SFTTrainer docs — https://huggingface.co/docs/trl/en/sft_trainer (pinned: https://huggingface.co/docs/trl/v0.21.0/en/sft_trainer)
- TRL source — https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md

Secondary / background:
- Sebastian Raschka, *When should you mask prompt tokens?* — https://sebastianraschka.com/faq/docs/when-mask-prompt-tokens.html
- *To Mask or Not to Mask* — https://towardsdatascience.com/to-mask-or-not-to-mask-the-effect-of-prompt-tokens-on-instruction-tuning-016f85fd67f4/
- HuggingFace, *Gotchas in tokenizer behavior* — https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior

---

## Caveats

- Framework-behavior claims rest on primary docs + source (strong, but **version-sensitive** — TRL's
  `completion_only_loss` literal default is `None` resolving conditionally, and defaults evolve
  release to release).
- All three empirical papers were run on **relatively small models (~1B–8B)** and classic datasets
  (Alpaca, LIMA, Tulu-v2). Transfer to large frontier models (30B–70B+) and to modern CoT / reasoning
  SFT is plausible but **not directly evidenced**.
- Effect sizes for nonzero prompt weight are modest (~6.55% relative) and setting-dependent; the "56%
  nonzero-optimal" figure is a bare majority.
- The two prompt-loss papers differ in emphasis (2401.13586: prompt loss "safely ignored" for
  long-completion data; WIT: nonzero prompt weight broadly beneficial) — reconciled by noting WIT's
  benefit concentrates in the short-output / small-data regime.

## Open questions

1. Exact prompt-masking defaults / flags for Axolotl (`train_on_inputs`), Llama-Factory, Unsloth,
   torchtune (only TRL was verified here).
2. Do these findings hold for large frontier models (30B–70B+) and CoT/reasoning SFT, or are they
   artifacts of the 1B–8B classic-dataset regime?
3. How to manage EOS / end-of-turn supervision under multi-turn assistant-only masking (name in the
   question, not resolved by verified sources — but see the MedVision note above).
4. A single heuristic reconciling "prompt loss safely ignored for long-completion" vs. "standard
   masking never optimal" — i.e., what best predicts when nonzero prompt weight pays off.
