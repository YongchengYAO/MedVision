# CLI reference: driver, stage modules, bundled scripts

All pipeline commands live in the repository checkout under
`<repo>/script/llm-parsing/` and are run from `<repo>` with the judge
interpreter (`"$PYTHON"`). `--help` of every Python module below was executed
on a CPU-only interpreter without vllm and works (argparse does not import vllm
at parse time); the library modules without a CLI are documented from their
sources.

## Driver: `bash <repo>/script/llm-parsing/run_llm_parsing.sh`

```
run_llm_parsing.sh [--judge KEY] [--fresh] [--yes|-y] [--from STEP] [--list] [--judges] [--help] [STEP ...]
STEPS: prep stage0 smoke pilot full analyze     (all six, in order, when none is named)
```

Environment knobs: `PYTHON`, `TASKS`, `TASK_DIR_<TL|AD|Detection>`,
`ROSTER_YAML_<TL|AD|Detection>`, `JUDGE`, `JUDGE_MODEL`, `JUDGE_MODEL_HF`,
`TP`, `NUM_SHARDS`, `GPU_NUM`, `CUDA_VISIBLE_DEVICES`, `PROCS`, `PILOT_LIMIT`,
`SMOKE_ROWS`, `STRUCTURED`, `ACCEPT_FP`, `MOCK`, `YES`, `SKIP_GPU_CHECK`,
`MEDVISION_DS_SRC` — defaults and meanings in `pipeline.md` §4.

## Stage-1 runner: `bash <repo>/script/llm-parsing/test-sweep.sh` (GPU; normally invoked by the driver)

Reads `TASKS`, `LIMIT` (judge the `_limit<N>` queues), `NUM_SHARDS`,
`CUDA_VISIBLE_DEVICES`, `TP`, `MOCK`, `STRUCTURED` (default `auto`), `PYTHON`,
`SKIP_GPU_CHECK`, `TASK_DIR_<task>`, `JUDGE`. Requires the queues to exist.
Writes `judge-out_<T><sfx>[_limitN][.MOCK].jsonl`, sharded as
`<out>.n<NUM_SHARDS>.shard<S>` when `NUM_SHARDS > 1`, refusing stale shard files
from a different shard count and refusing to overwrite a merged file holding
repair or re-parse rows.

## Registry: `"$PYTHON" <repo>/script/llm-parsing/judge_config.py`

```
judge_config.py [--judge JUDGE] [--shell] [--list]
  --list    list registered judges (`*` marks the default): key, hf id, env hint
  --shell   emit NAME=value lines for `eval`: JUDGE_KEY JUDGE_MODEL_HF JUDGE_SUFFIX JUDGE_TP
            JUDGE_ENV_HINT JUDGE_REQUIREMENTS JUDGE_POST_REQUIREMENTS JUDGE_TORCH_PIN
            JUDGE_TRANSFORMERS_MAJOR JUDGE_ENV_BASENAME
```
Imports only `os`; runs on any Python. Also the home of `TASK_SPECS`
(arity TL 2 / AD 1 / Detection 4), `STEP_SPECS`, `ANSWER_MODES`,
`SUCCESS_MODES`, `EXPECTED_ROSTER_COUNTS`, `DEFAULT_ROSTER_YAML`,
`DEFAULT_TASK_DIR`, `EXCLUDED_JSONL_STEMS`, and the naming helpers
`queue_filename`, `judge_out_filename`, `llm_parsed_dirname`, `limit_suffix`,
`judge_suffix`, `resolve_judge_key`, `judge_entry`, `step_spec_key`.

## Stage 0: `build_judge_queue.py`

```
build_judge_queue.py --task_type {TL,AD,Detection} [--task_dir TASK_DIR] [--config_yaml CONFIG_YAML]
                     [--limit LIMIT] [--processes|-p N] [--out_dir OUT_DIR] [--dry_run|--dry-run]
```
Defaults: `--task_dir` and `--config_yaml` per task from the registry;
`--out_dir` = `--task_dir`; `--limit` = first-N records per file by `doc_id`
ascending. Outputs `judge-queue_<T>[_limitN].jsonl` and
`judge-baseline_<T>[_limitN].json`. `--dry_run` counts and gates only (prints the
roster total you need for `EXPECTED_ROSTER_COUNTS`). Gates: roster resolution,
strict-parser replay, roster count (default trees, full builds only).

## Stage 1: `run_judge_vllm.py` (GPU; `--help` works on CPU)

```
run_judge_vllm.py --queue QUEUE --out OUT [--model MODEL] [--judge {gemma-4-31b}]
                  [--max_model_len N] [--tensor_parallel_size N] [--gpu_memory_utilization F] [--dtype DTYPE]
                  [--shard S] [--num_shards N] [--limit_rows N]
                  [--keep_raw] [--no_raw_on_invalid] [--raw_max_chars N]
                  [--structured {auto,none}] [--accept_prompt_fp FP]... [--redo_invalid]
                  [--max_tokens N] [--chunk_rows N] [--min_valid_rate F] [--mock]
```
- `--judge` decides chat-template kwargs and weight expectations; inferred from
  `--model` when it is a known hub id or a local dir whose basename starts with a
  key; **required** otherwise (never guessed). It does *not* choose the output
  path — pass `--out`.
- `--limit_rows` takes an evenly spaced sample spanning the whole queue (not the
  head, which is one model).
- `--keep_raw` persists raw text on every row; invalid rows keep raw by default
  (`--no_raw_on_invalid` saves ~45 MB and makes the next decoder fix a GPU pass).
- `--accept_prompt_fp` + `--redo_invalid` = repair pass (see `recipes.md`);
  without `--redo_invalid` a repair is a silent no-op.
- `--max_tokens` overrides the budget and therefore the fingerprint (a queue
  built at another budget is refused).
- `--chunk_rows` (2000) bounds the loss on interruption; `--min_valid_rate`
  (0.95, `0` disables) aborts before any long sweep if the first 200 rows fail
  validation.
- `--mock`: CPU stand-in, NOT a judge; every row stamped `judge_model: "mock"`.

## Stage 2: `apply_judge.py`

```
apply_judge.py --task_type {TL,AD,Detection} [--task_dir TASK_DIR] [--config_yaml CONFIG_YAML]
               --judge_out JUDGE_OUT [--limit LIMIT] [--accept_prompt_fp FP]... [--judge {gemma-4-31b}]
               [--processes|-p N]
```
Writes `<model_dir>/llm-parsed<sfx>[-limitN]/<same filename>.jsonl` with
`filtered_resps` removed and `LLM_filtered_resps` in its place, metrics
recomputed through `medvision_bm.utils.parse_utils.cal_metrics`, plus the
`LLM_judge*` keys. Aborts on an unknown `prompt_fp` unless whitelisted, and on a
judge-out file holding rows from two different readers
(`assert_single_judge_model`).

## Stage 3: the existing summarizers (owned by `../../results-parsing-and-metrics/SKILL.md`)

```
"$PYTHON" -m medvision_bm.benchmark.summarize_TL_task        --task_dir <tree> --parsed_dirname llm-parsed_gemma-4-31b \
      --resps_key LLM_filtered_resps --models <dir> [<dir>...] -p 32 --skip_model_wo_parsed_files --removed_samples_dir Data/Datasets
"$PYTHON" -m medvision_bm.benchmark.summarize_AD_task        ... (no --removed_samples_dir)
"$PYTHON" -m medvision_bm.benchmark.summarize_detection_task ... (no --removed_samples_dir)
```
`--resps_key LLM_filtered_resps` is mandatory on `llm-parsed*` directories
(`assert_resps_key` makes forgetting it a hard error, not an empty report).
Report names take a `__<parsed_dirname>` suffix (empty for `parsed`).

## Stage 4: `summarize_judge_task.py`

```
summarize_judge_task.py --task_type {TL,AD,Detection} [--task_dir TASK_DIR] [--config_yaml CONFIG_YAML]
                        [--limit LIMIT] [--judge {gemma-4-31b}] [--parsed_dirname PARSED_DIRNAME]
```
Outputs `<model_dir>/<parsed_dirname>/summary_metrics_judge_Task[_limitN].json`
and `<task_dir>/summary_judge_task[_limitN]__<parsed_dirname>.txt` (failure
decomposition, answer modes, judge validity, step-extraction coverage, length
stratification). Computes no model-quality metric.

## Repair on CPU: `reparse_judge_out.py`

```
reparse_judge_out.py --in IN_PATH --out OUT_PATH [--allow_regressions] [--allow_value_changes]
```
Re-runs decoder + validator over a judge-out file: rows with `raw` are fully
re-parsed, rows with only `final_answer` re-validated, rows with neither
untouched. Prints the transition table and an all-zero echo census; refuses to
write if any row moves ok→invalid (`--allow_regressions` to investigate) or any
ok row's values change (`--allow_value_changes`, only for reviewed decoder
changes). Writing `--out <merged>` in place drops the `.<name>.reparsed` marker
that guards the merged file against a later sharded sweep; `--out <side>` + `mv`
leaves it unguarded. Counts are per line (superseded duplicate lines included) —
trust Stage 1's "outstanding" for remaining work.

## Environment builder: `bash <repo>/script/llm-parsing/setup_judge_env.sh`

```
setup_judge_env.sh [--judge KEY] [--list] [--help] [<target-dir>]
env: JUDGE, TORCH_INDEX_URL, PYTHON_BIN
```
Reference only (mutates an env, downloads GBs); steps in `judge-environment.md` §2.

## Library modules (no CLI; documented from source)

| module | public symbols | role |
|---|---|---|
| `judge_io.py` | `NUM_RE`, `find_numbers(text)`, `extract_last_k_nums_within_answer_tag(text, k)`, `extract_response(data)`, `load_roster(roster_yaml)` → list of `model_display_name` keys in file order, `list_sample_files(model_dir, parsed_dirname, excluded_stems)`, `iter_records(jsonl_file, limit)`, `content_hash(...)`, `dataset_from_filename(path)`, `write_jsonl(path, rows)` | IO + the duplicated strict-parser primitives (must stay byte-equivalent to `parse_utils`/`parse_outputs`) |
| `judge_prompts.py` | `SYSTEM_PROMPT`, `USER_TEMPLATE`, `STEPS_BLOCK_TEMPLATE`, `NO_STEPS_NOTE`, `build_messages(task_type, step_key, response_text)`, `build_schema(task_type, step_key)`, `prompt_fingerprint(task_type, step_key)`, `short_prompt_fp(task_type, step_key)` | prompt assembly **and** the JSON schema |
| `judge_decode.py` | `split_final_channel`, `iter_json_object_candidates`, `extract_first_json_object`, `preescape_latex_escapes`, `repair_json_escapes`, `parse_judge_json(raw_text)`, `validate_judge_obj(obj, expects_steps)` | raw text → validated judge object |
| `judge_verify.py` | `collapse_ws(text)`, `verify_span(response, span, values, expected_arity)` | tiered span verification |
| `judge_decision.py` | `DECISION_TABLE_DOC`, `decide_answer(strict_pred, judge_row, verified)`, `is_success(mode)` | the single decision table |
| `judge_stats.py` | `decompose_failures(records)`, `judge_validity(records)`, `step_extraction_coverage(records, task_type)`, `length_stratification(records, n_bins)` | Stage 4 formulas; defines no model-quality metric |
| `judge_env.sh` | `JUDGE`, `JUDGE_KEY`, `JUDGE_SUFFIX`, `JUDGE_MODEL_HF`, `JUDGE_MODEL`, `JUDGE_TP`, `judge_shard_devices(shard, tp, devices...)` | sourced by both shell entry points; resolves the reader via `judge_config.py --shell`; token/cache hygiene |

## Bundled scripts (this sub-skill)

```
python scripts/check_judge_env.py [--python PY] [--judge KEY] [--llm-parsing-dir DIR | --repo-root REPO]
                                  [--timeout S] [--skip-cuda-alloc] [--json]
    exit 0 ok · 1 vllm missing/probe failed · 2 vllm ≠ registry pin · 3 bad arguments

python scripts/make_roster_yaml.py --results-dir Results/<task_tag> [--include-glob G]... [--exclude-glob G]...
                                   [--display-name-map JSON|file] [--no-require-parsed] [--out FILE | --dry-run] [--header TEXT]
    exit 0 ok · 1 no model directory qualified · 3 bad arguments
```
