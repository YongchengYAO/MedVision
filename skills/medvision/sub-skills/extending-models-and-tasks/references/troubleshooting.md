# Troubleshooting: extending models and tasks

Symptom -> likely cause -> fix and validation. Cross-cutting install/runtime failures live in
`../../../references/troubleshooting.md`; evaluation-run failures in
`../../benchmark-evaluation/SKILL.md`.

## Model wiring

### `ValueError: [Error] <key> is not recognised/supported.`

**Cause.** `get_resized_img_shape()` in `lmms_eval/tasks/medvision/medvision_utils.py` has no
branch for that `model_name`. Raised while building the *prompt*, so it fires on the first T/L or
A/D sample even though the model itself loaded fine.

**Fix.** Add the branch (see `image-size-dispatch.md`), listing every alias: the
`AVAILABLE_MODELS` key **and** the SFT/RFT `model_family_name` string if the family is also
fine-tuned. The `else: raise` fallthrough is intentional — never add a default branch; a wrong
perceived size silently corrupts every measurement instead of failing.

**Validate.** `python scripts/list_registered_models.py --expect <key>` (exit 0 when all three
code sites exist), then two samples of one T/L task.

**Also fires when.** The value passed to `--model` is right but the branch lists a *different*
spelling — e.g. `qwen25vl` (SFT family alias) is in the branch while the benchmark key
`vllm_qwen25vl` is not, or vice versa.

### `ValueError: Model <key> not found in available models.`

**Cause.** `get_model()` could not find the key in `AVAILABLE_MODELS`. Either the registry entry
is missing, or it is present but commented out (the registry keeps several commented entries for
models that are not part of the benchmark), or the eval entry point passes a `--model` string
that differs from the registry key.

**Fix.** Make all four spellings identical: `@register_model("<key>")`, the module file name
`lmms_eval/models/<key>.py`, the `AVAILABLE_MODELS` key, and the `lmmseval_module="<key>"`
argument in `eval__<key>.py`. `get_model()` resolves the registry *value* as
`lmms_eval.models.<key>.<ClassName>`, so the module name is derived from the key, not from the
class.

**Validate.** `python scripts/list_registered_models.py` — the report flags a registered key
whose module, decorator or class is missing, and lists commented-out entries and dispatch-only
aliases separately.

### `Failed to import <Class> from <key>` / `AttributeError` on the class

**Cause.** The class name in `AVAILABLE_MODELS` does not match the `class` statement in the
module, or the module raised at import time (a missing backend: `vllm`, the vendor SDK, a
`third_party` checkout on `sys.path`).

**Fix.** Compare the two names; then import the module directly in the model's evaluation
environment to see the real exception. The registry only re-raises after logging.

**When to stop.** Backend imports (`vllm`, `decord`, vendor SDKs) cannot be checked on a CPU-only
host without the model's requirements installed. Verify the *static* wiring with
`list_registered_models.py` and defer the import check to the target environment.

### Responses are truncated; many end exactly at the same length

**Cause.** No explicit output-token budget. The budget resolves as task YAML ->
`model_args max_new_tokens` -> wrapper default; if a wrapper sets none, a third-party default
takes over. This happened in the repository once: an upstream chat class hard-coded
`max_new_tokens=512`, and every run before the fix generated under that cap **silently**.

**Fix.** Give the wrapper an explicit default (4096 local, 16000 API) and make sure the launcher
passes `--max_new_tokens` / `--max_tokens`. Honour a per-request `gen_kwargs["max_new_tokens"]`
so a future task-YAML budget is not ignored.

**Validate.** Re-tokenise failing responses against the run's budget and compute the share of
samples that end exactly at the cap. A verbose CoT model can exhaust 4096 legitimately —
budget exhaustion shows up as a large *at-cap* share plus judge verdicts of "no conclusion".

**Warning.** Runs made with different budgets are not comparable, and the per-sample response
cache is keyed on the prompt only — clear it before re-evaluating with a new budget, or the old
truncated responses are replayed.

## API models

### `ValueError: [<provider>] Unsupported model code '<code>' (normalized '<norm>')`

**This is by design, not a bug.** The caps table is enumerated deliberately: models in the same
family differ in resolution caps and even in resize *rule family*, and a silent default would
state a wrong pixel size in every T/L and A/D prompt. It is raised twice — once in the model
class `__init__` (so a run fails before spending credit) and once from the dispatch branch.

**Fix.** Read the provider's official vision documentation for that exact model, then add one
`SUPPORTED_MODEL_CAPS` row in `lmms_eval/models/<provider>.py`. Do not add a fallback.

**Gateway ids.** `_normalize_model_code()` strips the gateway prefix (`anthropic/`, `google/`,
`openai/`, `moonshotai/`); some providers also normalise dots to dashes, some deliberately do not
(dots are part of the id). If a gateway id raises while the direct id works, the normaliser needs
the new prefix/suffix form, not a new table row.

### Coordinates or measurements are systematically off by a constant factor

**Cause (most likely).** The image sent is not a fixed point of the provider's vision pipeline,
so the server resized or **padded** it. Padding enlarges the canvas the model normalises relative
coordinates by, and MedVision's origin is at the lower-left — exactly where bottom padding lands.
Nothing errors; every coordinate is skewed by a constant.

**Fix.** Round each side **down** to the provider's grid (28 px for the 14x14-patch + 2x2-merge
families, 32 px for OpenAI patch models) inside `<provider>_resized_hw()`, and assert the
invariant in `_encode_image()`:

```python
assert new_h % GRID == 0 and new_w % GRID == 0
assert max(new_h, new_w) <= long_edge_cap and (new_h * new_w) / PX_PER_TOKEN <= max_tokens + 1
```

**Validate.** Offline: the resize tests (on-grid, within caps, never upscales, aspect preserved).
Live (credential-gated): a token-count probe — an on-grid image must incur **no** extra image
tokens relative to the predicted count. Also compare the `Original image size (HxW) ... Resized
image size (HxW)` line printed by the dispatch with the `The image size is ...` sentence in the
logged prompt.

**Second possible cause.** The stated size is right but `content_hw` was returned equal to a
padded canvas, inflating the short axis's pixel size on non-square slices. Square test images
hide this — always test a non-square one.

### `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`

**Cause.** `transformers` is unpinned in the API extra. The wrapper itself does not need
`transformers`, but the lmms_eval framework imports it, and a newer `transformers` imports
`is_offline_mode`, which the pinned `huggingface_hub==0.36.0` removed.

**Fix.** Pin the validated version in the extra
(`[project.optional-dependencies] <key> = ["<sdk>", "transformers==<validated>"]`) and repeat both
pins in `requirements/requirements_eval_<key>.txt`. All four shipped API extras pin the same
transformers version.

**Validate.** `pip show transformers huggingface_hub` in the eval env, then
`python -c "import lmms_eval"`.

### HTTP 401/400 with a key that looks correct

**Cause.** A pod/k8s-injected secret carries a trailing newline, which is an illegal HTTP header
value.

**Fix.** Both layers: `os.environ["<KEY>"].strip()` in the wrapper's `prepare_model()`, and
`export KEY="$(printf '%s' "${!KEY}" | tr -d '\n')"` in the launcher.

**Also.** 400s are deterministic; the retry decorator gives up on `status_code == 400` instead of
burning credit. Newer reasoning tiers reject `budget_tokens` and sampling parameters
(`temperature` / `top_p` / `top_k`) with a 400, and an explicit "thinking disabled" object also
400s — omit the parameter instead.

## Task wiring

### A new task never runs; no error at all

**Cause.** It is not in a task-list JSON. `medvision_bm.utils.utils.load_tasks` reads the
**top-level keys** of the JSON and iterates only those; a task YAML that no list references is
simply never selected. This is the single most common "silently does nothing" cause.

**Fix.** Add the exact `task:` string as a key in the right list
(`tasks_list/tasks_MedVision-{detect,TL,AD}-CoT.json`, `OOD/`, or an SFT list). The value is an
informational count.

**Validate.** `python scripts/list_task_yamls.py --tasks-json <list>` — the task must show `USED`,
and no "`<name>` has no task YAML" problem may be reported.

### `Tasks not found: <name>` / the task list entry resolves to nothing

**Cause.** The name in the JSON does not match the `task:` value in any YAML — usually a variant
suffix mismatch (`-CoT` present in one and not the other) or a plane/task-id typo.

**Fix.** Copy the name from the YAML, not from a file name. Note the two namespaces: evaluation
task names use `BoxCoordinate` while SFT lists use `BoxSize`; task names carry `-CoT` while SFT
names historically do not (a legacy naming inconsistency — both point at the same dataset
configs).

**Validate.** `scripts/list_task_yamls.py --tasks-json <list>` reports every unresolved entry.

### `include` errors, or the wrong prompt is used

**Cause.** `include:` paths are resolved relative to the **including file's directory**. A task
YAML includes a bare sibling base name; a base YAML includes `../medvision/...`. Copying a task
YAML into another dataset folder without changing the include silently pulls another dataset's
base — or fails if it does not exist.

**A worse, silent variant:** including the *plain* base from a variant task YAML (e.g. a
`-CoT-woInstruct` task including `..._base-CoT.yaml`). Every variant has its own base with a
different `doc_to_text`, so the run succeeds with the wrong prompt.

**Fix / validate.** `scripts/list_task_yamls.py` reports a missing include target as a problem
and prints each base's `doc_to_text`; check that the variant task points at the variant base.

### Two tasks collide / one silently overwrites the other

**Cause.** Two YAMLs declare the same `task:` name. Trailing whitespace is invisible in an editor
but stripped by YAML, so `task: Foo-VP ` in `Foo-VP-woMedImg.yaml` collides with `Foo-VP.yaml`.
The task index is a plain dict, so the last file walked wins — non-deterministically from the
reader's point of view.

**Fix.** One `task:` value per name, no trailing whitespace; the stem of the file name should
equal the task name.

**Validate.** `scripts/list_task_yamls.py` reports every duplicate with both file names. (The
shipped tree currently has **none** — `scripts/list_task_yamls.py --json` reports 0 problems over all
1253 task YAMLs — but many `task:` lines carry trailing whitespace, which is how a duplicate would arise.)

### `--tasks <tag>` selects nothing / the task is invisible to a tag

**Cause.** Usually the tag is not what you typed. The shipped `tag` is a single scalar,
`MedVision-<TaskType>,<Dataset>`, and `lmms_eval` registers that whole comma-joined string as **one** tag key
(`tasks/__init__.py:462-464` wraps a `str` in a one-element list; nothing splits on `,`). A bare family name
(`--tasks MedVision-TumorLesionSize`) or a bare dataset name (`--tasks BraTS24`) therefore matches nothing and
`__main__.py` raises `ValueError: Tasks not found`. Secondary cause: a typo, a missing dataset label, or a
`tag` rewritten on the task YAML creates a *new* single-member tag instead of joining the family.

**Fix.** Pass the exact composite string (`--tasks 'MedVision-BoxCoordinate,BraTS24'`) or a wildcard
(`--tasks 'MedVision-TumorLesionSize*'`) — `pattern_match` fnmatches against the registered keys. In practice
MedVision drives tasks one name at a time from `tasks_list/*.json` rather than by tag. If you did mean to join
a family, copy the tag line verbatim from a sibling base; a misspelled tag is a valid, empty-looking group
rather than an error.

**Validate.** `scripts/list_task_yamls.py --dataset <Dataset>` prints the tag of each base and the
inherited tag of each task.

### Dataset config not found on the Hub

**Cause.** `dataset_name` must be a published **dataset config**, which uses a different family
label than the task name: detection tasks named `..._BoxCoordinate_...` load `..._BoxSize_...`
configs, and every eval config ends in `_Test`. A `-CoT` suffix belongs to the task name only —
it must never appear in `dataset_name`.

**Second cause.** Overriding `dataset_path` in a task YAML with a bare config-looking string.
`dataset_path` is the **HF repo id** (`YongchengYAO/MedVision`); replacing it with
`MyDataset_MaskSize_Task10_Sagittal_Test` makes the loader look for a repository of that name.

**Fix / validate.** Set `dataset_name`, leave `dataset_path` inherited.
`scripts/list_task_yamls.py` flags a task YAML that sets neither, and one whose `dataset_path`
override contains no `/`. Which configs exist per annotation version is owned by
`../../dataset-and-tasks/SKILL.md`.

### An HF model raises `TypeError`/`KeyError` on construction for missing arguments

**Cause.** The wrapper needs parameters beyond `model_name` / `model_hf` (extra checkpoints,
adapter configuration, an instruction template) and there is no per-model block in
`lmms_eval/tasks/medvision/lmms_eval_specific_kwargs.yaml`. That file is included by every base
task YAML; the shipped example is the HealthGPT block (base/vision checkpoints + HLoRA settings +
`instruct_template`).

**Fix.** Add a block keyed by the model key. Its own header warns: do not rename the model-field
keys (they are read by `medvision_utils.py`) and do not delete the empty `dataset:` field.

**Prefer the alternative.** If the parameter is a plain scalar, pass it through
`--model_args key=value` from the eval entry point instead — no YAML edit, and it stays visible
in the run's logged arguments.

## Regeneration tooling

### `regen_all_tasks.py` / `configs_to_tasks` imports a stale helper

**Cause.** A non-editable `medvision_bm` in `site-packages` shadows the checkout, so an edited
helper is imported as its installed copy.

**Fix.** `export PYTHONPATH=<repo>/src` before running (the repository's own wrappers do this),
and confirm with `python -c "import medvision_bm; print(medvision_bm.__file__)"`.

### Counting a task list is slow or fails

**Cause.** `configs_to_tasks` streams every config from the Hub to compute counts; this needs
network access, a valid `HF_TOKEN`, and `MedVision_PLANNER_VERSION` exported.

**Fix.** Use `--no-count` for a naming-only run (counts are written as `0` and are informational
anyway). `regen_all_tasks.py` caches counts by `(config, resolved annotation version)` — run
versions oldest-first so later ones reuse the cache, and use `--no-count` to preview the work.

**When to stop.** Both tools need the dataset source tree, network and a token. On a CPU-only,
offline host, document the command instead of running it.

## When to stop and ask

- Anything requiring a **GPU**: importing `vllm`, loading open weights, running an evaluation.
  Static wiring checks and `--help` are the CPU-safe substitutes.
- Anything requiring **credentials**: API caps verification against a live endpoint, token-count
  probes, dataset streaming.
- Choosing an image-resize rule for a provider whose documentation does not state the geometry.
  The repository's convention is to state the assumption in the module docstring and guard it
  with an empirical probe — not to guess a formula.
- Editing files inside a user's checkout: `scripts/scaffold_new_model.py` deliberately refuses to
  write into one. Generate into a scratch directory, review, then let the user apply the changes.
