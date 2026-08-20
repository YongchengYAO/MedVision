# Token-Budget Audit: Model Wrappers and Benchmark Scripts

*Audited 2026-08-08, on master at commit `09206a2` (HuatuoGPT-Vision decoding fix).*

Scope: every model wrapper in
`src/medvision_bm/medvision_lmms_eval/lmms_eval/models/` and every launch
script in `script/benchmark-{AD,TL,detect}/` (23 scripts per task, identical
model sets). Trigger: the HuatuoGPT-Vision wrapper silently generated under
the upstream 512-token default for all pre-2026-08-08 runs, and
budget-exhaustion analysis of the v2/CoT results (see
[Measured evidence](#measured-budget-exhaustion-evidence)).

## How the budget is resolved

Three channels, in order of precedence inside the wrappers:

1. **Task YAML `generation_kwargs.max_new_tokens`** — would win if set, but
   **no MedVision task YAML sets it** (verified: zero `max_new_tokens` matches
   under `lmms_eval/tasks/`). The YAML channel only carries
   `until` / `do_sample`.
2. **`model_args` `max_new_tokens`** — injected by the launcher
   (`--max_new_tokens`, default 4096 for local models, 16000 for API models).
   This is the channel that decides in practice.
3. **Wrapper default** — same values as the launcher defaults; only reached
   when a wrapper is invoked without the launcher.

A fourth, unwanted layer — third-party internal defaults — is only reachable
when a wrapper fails to set the budget at all. That was the HuatuoGPT-Vision
bug (upstream `HuatuoChatbot` hardcodes `max_new_tokens=512`), fixed in
`09206a2`.

## Wrapper audit: does every model have a budget?

**Yes — all 21 wrappers now have an explicit budget path.** Details and
exceptions:

| wrapper | default | task-YAML override honored | notes |
|---|---|---|---|
| `vllm_gemma3`, `vllm_gemma4`, `vllm_glm4v`, `vllm_internvl3`, `vllm_llama_3_2_vision`, `vllm_llava_onevision`, `vllm_minimax_m3`, `vllm_qwen25vl`, `vllm_qwen3vl` | 4096 | yes | uniform pattern: `gen_kwargs.setdefault` → vLLM `max_tokens` |
| `healthgpt`, `lingshu`, `medgemma`, `meddr` | 4096 | yes | HF generate/pipeline; per-request `gen_kwargs.get(...)` |
| `huatuogpt_vision` | 4096 | yes (since `09206a2`) | **before the fix: upstream 512, silently** — applies to all runs made before 2026-08-08 |
| `llava_med` | 4096 | **no** | generates with `self.max_new_tokens` only; model-args channel works, per-request channel ignored (also hardcodes `min_new_tokens=16`) |
| `biomedgpt` | 4096 (hardcoded fallback) | yes | no model-arg; budget goes through OFA `max_length`, `num_beams=5`; has no `eval__` launcher in `benchmark/` |
| `claude`, `gemini`, `kimi`, `openai` | 16000 (`max_tokens`) | yes | per-task `max_new_tokens` takes precedence when present |
| `vllm_qwen25vl_tooluse` | fixed 512 + 64 | n/a | two-phase tool-use pipeline: 512 tokens for `<think>` + `<tool_call>`, 64 for the final `<answer>`; budgets are deliberate per-phase caps, not configurable |

## Effective budgets in the benchmark scripts

Only three script families override the launcher defaults; everything else
inherits 4096 (local) / 16000 (API). Budgets are identical across
`benchmark-AD`, `benchmark-TL`, and `benchmark-detect`.

| script(s) | launcher | effective budget | source |
|---|---|---|---|
| GLM-4.6V, GLM-4.6V-Flash, Gemma-3-27B-it, Gemma-4-31B-it, InternVL3-38B, LLaVA-OneVision, Llama-3.2-Vision, Qwen-2.5-VL, Qwen-3-VL-32B-Thinking, MedVision-V0-7B | vLLM launchers / `medvision-model-rft` | 4096 | launcher default |
| HealthGPT-L14, Lingshu, MedDr, MedGemma, LLaVA_Med | HF launchers | 4096 | launcher default |
| HuatuoGPT-Vision-34B | `huatuogpt_vision` | 4096 (explicit `max_new_tokens=4096` in script) | script; **512 for all runs before 2026-08-08** |
| MiniMax-M3, MiniMax-M3-INT4 | `minimax_m3` | **16384** (explicit in script) | script |
| GPT5.5, GPT5.5-Pro | `openai` | **4096** (explicit `max_tokens=4096` in script) | script — *below* the 16000 API default, and o-series `max_completion_tokens` includes reasoning tokens |
| Claude-Fable5, Gemini-3.1-Pro, Kimi-K2.6 | API launchers | 16000 | launcher default |

## Measured budget-exhaustion evidence

From the 2026-08-08 analysis of the v2 / v2-CoT results (LLM-judge
`answer_mode` + exact re-tokenization of failing responses against each run's
budget):

| model / results dir | at-cap share | verdict |
|---|---|---|
| medgemma-27b-it, AD-v2-CoT | **29.5% of all samples** at exactly 4096, judge `no_conclusion` | budget exhaustion — post-judge SR 69.9% is a budget ceiling |
| MiniMax-M3-INT4, AD-v2-CoT | 100% of residual failures at 16384 (97% of recovered failures too) | budget exhaustion — reasoning routinely burns the full 16K |
| Llama-3.2-11B, TL/AD-v2-CoT | 1.9% / 0.6% of samples (degenerate repetition loops riding to 4096) | marginal — more budget would likely just loop longer |
| Qwen2.5-VL-32B, TL/AD-v2-CoT | 0% | not budget; off-format `\boxed{}` answers, recovered by judge |
| HuatuoGPT-34B, detect-v2 | ~0.3% of samples (upstream 512 cap) | not budget; missing `</answer>` tag, recovered by judge |

## Recommendations

- **CoT tasks on verbose/agentic models need more than 4096.** medgemma-27b-it
  demonstrates the failure mode at scale; Qwen-3-VL-32B-Thinking runs at 4096
  and is in the same risk class — check its truncation rate before trusting
  its SR.
- **GPT5.5 / GPT5.5-Pro at 4096 shared with reasoning tokens** deserves the
  same check: the visible answer competes with hidden reasoning inside one
  cap.
- **Re-runs of HuatuoGPT-Vision after `09206a2` use 4096** and are not
  comparable with pre-fix outputs (512); response caches keyed on prompt only
  must be cleared before re-eval.
- `llava_med` is the one wrapper where a future task-YAML budget would be
  silently ignored — worth aligning with the common per-request pattern if
  YAML budgets are ever introduced.
