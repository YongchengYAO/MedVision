---
name: extending-models-and-tasks
description: "Guides a maintainer adding a new VLM (vLLM, HuggingFace/accelerate or hosted API) or a new task/dataset to the MedVision benchmark: the AVAILABLE_MODELS registry and @register_model key, the get_resized_img_shape perceived-image-size dispatch and _process_img_* probes, the model_name/model_hf injection in evaluator.py, lmms_eval optional-dependency extras, eval__<model>.py entry points and the three benchmark launchers, explicit output-token budgets, API caps tables with client-side 28-grid pre-resize and key sanitising, base/per-task YAML anatomy with tags and create_doc_to_text_* factories, and registration in tasks_list JSONs. Use when adding, registering, scaffolding or debugging a MedVision model wrapper or task YAML, when a model is 'not recognised/supported', or when a new task silently never runs."
disable-model-invocation: true
license: CC-BY-4.0
metadata:
  disco-role: operating
---

# MedVision maintainer: adding models and tasks

Use this sub-skill when extending the MedVision benchmark itself: wiring a new vision-language
model into the vendored `lmms_eval` fork, or adding a new task / dataset / prompt variant.
Terminology: Detection (bounding box) / Tumor-Lesion size (T/L) / Angle-Distance (A/D);
`medvision_bm` (this package), `medvision_ds` (dataset package), `lmms_eval` (vendored fork).

## Route here for

- Adding a VLM: the registry entry, the wrapper class, the perceived-image-size dispatch branch,
  the `pyproject.toml` extra, `eval__<key>.py`, three launchers, a frozen requirements file, a
  resize test and the README roster row.
- API models: enumerated caps tables that live only in the model file, client-side pre-resize to
  the provider grid, why padding corrupts relative coordinates, key sanitising, `transformers`
  pins.
- Adding a task: base YAML + per-task YAML anatomy, the four `!function utils.*` hooks, the
  `create_doc_to_text_*` factory pattern, `tag` values, naming conventions, and registration in
  `tasks_list/*.json`.
- Diagnosing "`<model>` is not recognised/supported", a task that silently never runs, a wrong
  prompt from a mis-included base, or a systematic coordinate skew from server-side padding.

## Do not use for

- Running evaluations, launcher variables, resume/cache, results layout ->
  `../benchmark-evaluation/SKILL.md`.
- Installing `medvision_bm` / `medvision_ds`, pins, install order, wheel builds ->
  `../environment-setup/SKILL.md`.
- Dataset config naming, annotation versions, `MedVision_PLANNER_VERSION`, downloads, task-JSON
  semantics -> `../dataset-and-tasks/SKILL.md`.
- Metric definitions, `parse_outputs`, `summarize_*` -> `../results-parsing-and-metrics/SKILL.md`.
- Which models already exist and what each needs -> `../../references/model-roster.md`.
- Terms and file layout -> `../../references/concepts-and-glossary.md`; cross-cutting install and
  runtime failures -> `../../references/troubleshooting.md`.

## The one rule that ties a model together

A **model key** is one lowercase snake_case string used identically in four places: the
`@register_model("<key>")` decorator, the module file `lmms_eval/models/<key>.py`, the
`AVAILABLE_MODELS` dict key, and the `--model <key>` value the eval entry point passes.
`get_model()` resolves the registry value as `lmms_eval.models.<key>.<ClassName>`, so a mismatch
is an import failure, not a warning. A fifth spelling — the branch condition in
`get_resized_img_shape()` — must accept the same key (plus the SFT `model_family_name` alias, if
the family is also fine-tuned).

**No task YAML needs a per-model edit.** `evaluator.py` injects `model_name` (from `--model`) and
`model_hf` (from `--model_args model_hf=...`) into `lmms_eval_specific_kwargs` before any prompt
is built, so the task layer can compute the model's perceived image size itself.

## Fast paths

**Audit the current wiring (CPU, no network):**

```
python scripts/list_registered_models.py                 # registry x module x dispatch, with strategy letters
python scripts/list_registered_models.py --expect <key>  # checklist for one key; exit 1 if incomplete
```

**Scaffold every file a new model needs (writes to a scratch dir, never a checkout):**

```
python scripts/scaffold_new_model.py --key vllm_mymodel --class-name VLLM_MyModel \
    --kind vllm --hf-id Org/MyModel-7B --out-dir <scratch> --dry-run
```

`--kind vllm|hf|api` selects the wrapper, entry point and launcher templates. Then fill the TODO
markers, apply the two `patches/` snippets by hand, and re-run `--expect <key>`.

**Inventory task YAMLs and find what a task list actually uses:**

```
python scripts/list_task_yamls.py --dataset <Dataset>
python scripts/list_task_yamls.py --tasks-json <dir>/tasks_MedVision-TL-CoT.json --unused
```

Exit 1 means a real problem: a broken `include:`, a duplicate `task:` name, a task YAML with no
`dataset_name`, or a task-list entry with no YAML.

## Invariants worth memorising

1. **Perceived size, not raw size.** T/L and A/D prompts state the image size and pixel size the
   model must use for pixel -> mm arithmetic, and detection coordinates are relative to the
   perceived canvas. `get_resized_img_shape()` returns a pair — `(perceived_canvas_hw,
   content_hw)` — because letterboxing models must state the padded canvas but rescale the pixel
   size by the *pre-pad* content. Test with a **non-square** slice; square inputs hide per-axis
   errors.
2. **The `else: raise` in the dispatch is a feature.** An unknown model must fail loudly; a
   default branch would silently emit a wrong scale.
3. **One source of truth per API provider.** The caps table and the resize formula live in
   `lmms_eval/models/<provider>.py`; the dispatch branch imports them lazily. Never copy the
   formula into the task layer.
4. **Every wrapper sets an explicit output-token budget** (4096 local, 16000 API). A wrapper that
   sets none inherits a third-party default — the repository lost a full set of runs to a silent
   512-token cap once.
5. **`tasks_list/*.json` is the sole authority** for which task YAMLs run. The tree ships far more
   YAMLs than any list references. A new task with no list entry produces no error and no output.
6. **Launchers have two blocks.** The live "Method 1" block and the commented "Method 2" block are
   kept in sync in the repository — edit both.
7. **Two namespaces.** Evaluation task names use `BoxCoordinate` and carry `-CoT`; dataset configs
   and SFT task lists use `BoxSize` without `-CoT`. Same data, different strings.

## References and scripts

- Read `references/add-a-model.md` for the full checklist (file -> what to add -> why -> how to
  verify), what is required vs operational, wrapper and entry-point anatomy, the load-bearing
  env-setup call order, launcher anatomy and the token-budget rule.
- Read `references/api-model-rules.md` before adding any hosted model: caps tables, the
  floor-to-grid fixed point, why padding is not harmless, provider/auth/reasoning quirks, key
  sanitising, and the `transformers` pin that prevents the `is_offline_mode` ImportError.
- Read `references/image-size-dispatch.md` for every branch of `get_resized_img_shape()` with its
  keys, strategy class (A fixed / B probed / C API rule), exact fixed sizes, the canvas-vs-content
  contract and the dispatch-only SFT aliases.
- Read `references/add-a-task.md` for base and per-task YAML anatomy, the `create_doc_to_text_*`
  factory pattern, tags, naming conventions, a copyable minimal pair, and the (reference-only)
  `configs_to_tasks` / task-YAML regeneration tooling.
- Read `references/troubleshooting.md` when something is not recognised, not registered, not
  running, or quietly wrong.
- Run `scripts/list_registered_models.py` to cross-check the registry, the model modules and the
  dispatch branches, and to get a wiring checklist for a key you are adding (`--expect`).
- Run `scripts/scaffold_new_model.py` to generate the wrapper, patch snippets, entry point, three
  launchers, resize test and requirements stub for a new model (`--dry-run` first).
- Run `scripts/list_task_yamls.py` to inventory base and task YAMLs per dataset, resolve inherited
  tags, and see which tasks a given task list references (`--json` for machine output).

All three scripts are standard-library only, accept `--repo-root` / explicit paths instead of
assuming a checkout, and print `--help`.

## Safe operating rules

- These scripts read and generate; they never edit a checkout. `scaffold_new_model.py` refuses an
  output directory inside a MedVision checkout unless explicitly overridden — generate into a
  scratch directory, review, then let the user apply the changes.
- Static wiring checks run on CPU with no network. Importing a wrapper needs its backend
  (`vllm`, vendor SDK, `third_party` checkouts); running a model needs GPUs or credentials. Say so
  rather than approximating.
- Never `pip install` into a user's environment without pointing at the pins in
  `../environment-setup/SKILL.md`. The `huggingface_hub` pin is **per model stack**, not global —
  15 of the 20 eval entry points assert `0.35.3`, five (claude, gemini, openai, kimi, medgemma) assert
  `0.36.0`, and the gemma4 / glm4v / minimax-m3-int4 stacks are on the 1.x line. The per-model
  `transformers` pin is load-bearing across the whole benchmark.
- The repository ships its own maintainer guides for this topic — `docs/New-Models-Guide.md`
  (which embeds `AVAILABLE_MODELS` and the `get_resized_img_shape` dispatch **verbatim**, so it must be
  updated when either changes), `docs/New-Tasks-Guide.md`, `docs/Model-Image-Processing.md` and
  `docs/model-token-budget.md`. Read them alongside this unit.
- Verify a new model on **two samples of one T/L task** and read the logged prompt before
  spending GPU hours or API credit on a full run.
