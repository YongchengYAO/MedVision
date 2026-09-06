# The judge environment

The second reader needs a **newer vLLM than the evaluation stacks pin** (the
repository's per-model eval requirements pin vLLM 0.10.x/0.11.0 for most
wrappers; the judge registry pins `vllm==0.19.0`), and Gemma-4 needs a
Transformers 5.x line that a vLLM declaring `transformers<5` cannot host. So the
judge gets its **own** virtual environment, one per reader, and the pipeline is
told about it with a single variable:

```
export PYTHON=<judge-env>/bin/python
```

Everything — Stage 1 (GPU) and Stages 2–4 (CPU reports) — runs under that one
interpreter. Verify it with the bundled `scripts/check_judge_env.py` (section 5).

## 1. Registry (`JUDGE_MODELS` in the checkout's `judge_config.py`)

Verified with `judge_config.py --list` / `--shell` on CPU (the module imports
only `os`):

| field | value |
|---|---|
| key | `gemma-4-31b` (default and only registered reader) |
| `hf_id` | `google/gemma-4-31B-it` — plain bf16, ~62 GB, fetched by vLLM on first load |
| `chat_kwargs` | `{}` (Gemma templates take no `reasoning_effort`; it stays in the prompt fingerprint but is not sent) |
| `tensor_parallel` | `2` (capacity floor from the parameter count; the driver defaults `TP=1`) |
| `out_suffix` | `_gemma-4-31b` → `judge-out_<T>_gemma-4-31b.jsonl`, `llm-parsed_gemma-4-31b/`; **must be non-empty and unique** (pinned by unit test 11) |
| `requirements` | `requirements-gemma-4-31b.txt` — phase 1: `vllm==0.19.0`, `accelerate>=1.0` |
| `post_requirements` | `requirements-gemma-4-31b-post.txt` — phase 2 overrides: `transformers==5.10.2`, `nvidia-nccl-cu12==2.28.7` |
| `torch_pin` | `torch==2.10.0` (what vLLM 0.19.0 pulls) |
| `transformers_major` | `5` |
| venv basename | `judge-env_gemma-4-31b` → default target `<repo>/.cache/judge-env_gemma-4-31b` |

Other constants that matter operationally: `JUDGE_TEMPERATURE=0.0`,
`JUDGE_TOP_P=1.0`, `JUDGE_SEED=1024` (inert at temperature 0),
`JUDGE_MAX_MODEL_LEN=12288`, `DEFAULT_JUDGE_MAX_TOKENS=4096` (all tasks),
`CHUNK_ROWS_DEFAULT=2000`, `MIN_VALID_RATE_DEFAULT=0.95`,
`VALID_RATE_PROBE_ROWS=200`, response window trigger/head/tail
8000/2000/6000 characters.

History: an earlier reader (a quantized Mixture-of-Experts model, vLLM 0.11.0,
empty `out_suffix`) was retired on 2026-08-17; its unsuffixed
`judge-out_<task>.jsonl` archives were deleted, together with the weight
conversion step, `JUDGE_MODEL_DIR`/`JUDGE_DEQUANT`/`JUDGE_CACHE_BASENAME`, and
test 10. The paper's headline numbers came from that reader; every artifact on
disk now comes from `gemma-4-31b`.

## 2. What `setup_judge_env.sh` does (reference only — mutates an env, downloads GBs)

`bash <repo>/script/llm-parsing/setup_judge_env.sh [--judge KEY] [<target-dir>]`,
env knobs `JUDGE`, `TORCH_INDEX_URL`, `PYTHON_BIN` (base interpreter, must be
Python 3.10–3.12 — vLLM publishes no 3.13 wheels, and pip reports that as "no
matching distribution for vllm"). Steps, in order:

1. Refuses unknown `-*` options and refuses a driver step name (`prep`, `stage0`,
   `smoke`, `pilot`, `full`, `analyze`) as a target — a stray `pilot/` venv was
   once created that way. At most one positional target.
2. Reads the registry via `judge_config.py --shell` (requirements file, torch
   pin, transformers major, venv name); default target
   `<repo>/.cache/judge-env<out_suffix>`.
3. `python -m venv <target>` (reused if it exists); `pip install --upgrade pip`.
4. If `TORCH_INDEX_URL` is set, installs the torch pin from that index **first**
   (for drivers older than CUDA 12.8, e.g. `https://download.pytorch.org/whl/cu126`).
5. Phase 1: `pip install -r requirements-gemma-4-31b.txt` (vLLM pulls torch and
   the CUDA runtime; several GB).
6. `pip install -r requirements-cpu-stages.txt` — the measured import set of
   Stages 2–4 (`datasets`, `nibabel`, `pytz`, `loguru`, `sqlitedict`, `evaluate`,
   `sacrebleu`, `scikit-learn`, `scipy`, `numexpr`, `pandas`, `tenacity`,
   `portalocker`, `lxml`, `tabulate`, `zstandard`, `av`, `soundfile`,
   `matplotlib`, `SimpleITK`, `pynrrd`, plus listed transitives), deliberately
   **unpinned** and **without** torch/transformers/vllm. It is not the vendored
   lmms_eval's declared dependency set, which pins `torch<2.8` and would
   downgrade torch under vLLM.
7. Phase 2: `pip install -r requirements-gemma-4-31b-post.txt`. **A block of red
   "incompatible" lines is expected**: forcing `transformers==5.10.2` makes every
   package declaring `transformers<5` complain (vllm, xgrammar,
   compressed-tensors) and torch complains about `nvidia-nccl-cu12`. These are
   declared bounds, not observed failures. They cannot go in phase 1: pip
   resolves one requirements file as one constraint set, and
   `vllm==0.19.0` beside `transformers==5.10.2` is `ResolutionImpossible`.
8. Verification (the real gate): imports `torch`, `transformers`, `vllm`; asserts
   the transformers major equals the registry's; imports every installed package
   that declares `transformers<5` and fails only if one genuinely breaks
   (success line: `...and all of them import cleanly, so those bounds are
   conservative.`); then **allocates** a CUDA tensor (device_count alone passes
   on a torch built for a newer CUDA than the driver). On a CPU box it prints a
   NOTE and exits 0 — the CPU steps work there.
9. Prints `export PYTHON=<target>/bin/python`.

Why nccl is overridden: torch 2.10.0 calls `ncclCommShrink`, absent from the
`nvidia-nccl-cu12` release torch's metadata pins with `==`; leaving it gives an
undefined-symbol crash at vLLM import, not a resolver error.

## 3. The two-interpreter trap

Two independent probes exist because they fail independently: a conda base with
`medvision_bm` has no vllm; a bare judge venv without the CPU-stage list imports
vllm and dies on Stage 2's first line (`datasets`) — after a 13-hour sweep. The
driver probes both before running (`import vllm` + CUDA allocation for GPU
steps; `cal_metrics` + the selected summarizers + `medvision_ds` plan utilities
for `analyze`/`pilot`, warning-only for `full`/`smoke`). If `PYTHON` is unset it
says so in the banner and refuses to suggest a `pip install` into an interpreter
nobody chose.

`medvision_ds` is resolved separately (`MEDVISION_DS_SRC`, or an upward search
for a sibling `MedVision/src` / `medvision_ds/src`), because it imports fine
without `datasets` and so does not prove the CPU stages will run.

## 4. Secrets and caches

`judge_env.sh` (sourced by both shell entry points) strips all whitespace from
`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN` and newlines from
`HF_HOME`, `HF_HUB_CACHE`: a trailing newline turns the Authorization header
invalid and surfaces as "Can't load the configuration of `google/gemma-4-31B-it`",
which reads like a gated/missing model. Make sure `HF_HOME` has ~62 GB free
before the first `smoke`.

## 5. Checking an environment with the bundled script

```
python scripts/check_judge_env.py --python <judge-env>/bin/python --repo-root <repo>
python scripts/check_judge_env.py --python <judge-env>/bin/python --json
```

It probes the target interpreter in a subprocess and reports vllm vs the
registry pin (exit 1 missing, 2 mismatch), transformers major, torch/CUDA build,
GPU count and a CUDA allocation (`--skip-cuda-alloc` to only enumerate), the
CPU-stage imports (`cal_metrics`, `datasets`, `nibabel`, `yaml`) and
`medvision_ds`, and prints the `export PYTHON=…` line. With `--repo-root` or
`--llm-parsing-dir` it reads the live registry and requirements files; without
them it uses a snapshot of the registry recorded 2026-09-04 and says so. Its
`--help` works without vllm or torch.

## 6. Relation to the other environments

- The evaluation environments (`../../environment-setup/SKILL.md`,
  `../../benchmark-evaluation/SKILL.md`) **mostly** pin older vLLMs (0.10.0-0.14.0); never install
  the judge requirements into those — the torch/transformers pins conflict. Two do not conflict:
  the gemma4 eval env pins the identical 0.19.0 / transformers 5.10.2 / torch 2.10.0 stack, and
  glm4v pins vLLM 0.19.1.
- The CPU-only inspection or parsing environment can run `stage0`, `analyze`,
  `reparse_judge_out.py` and the unit tests (verified: tests 1–7, 9, 11 pass on
  a torch-CPU interpreter without vllm), but not `smoke`/`pilot`/`full`.
- Adding a reader = one `JUDGE_MODELS` entry with a unique non-empty
  `out_suffix` + its requirements file(s), then `setup_judge_env.sh --judge <key>`
  builds `.cache/judge-env<out_suffix>` without touching the first reader's env.
