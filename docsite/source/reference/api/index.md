# Python API

This section is generated with Sphinx `autodoc` directly from the docstrings in the
`medvision_bm` source tree. It documents the **importable** helpers — the functions
and constants you would `import` when building on top of MedVision (computing metrics,
loading and formatting datasets, wiring up trainers).

```{note}
Most day-to-day use of MedVision happens through the command line, not the Python
API — see the {doc}`../cli` for the full list of `python -m medvision_bm.*` entry
points. The modules below are the reusable building blocks those commands are made of.
```

The heavy machine-learning dependencies (`torch`, `transformers`, `vllm`, …) and the
vendored `lmms-eval` fork are mocked at build time, so signatures render even though
those packages are not installed on the docs builder.

## Modules by area

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Metrics & parsing
:link: parse_utils
:link-type: doc

`utils.parse_utils` — IoU / F1 / MAE / MRE and answer extraction.
:::

:::{grid-item-card} Constants
:link: configs
:link-type: doc

`utils.configs` — seeds, thresholds, label maps, dataset mappings.
:::

:::{grid-item-card} I/O & task status
:link: utils
:link-type: doc

`utils.utils` — atomic JSON, task lists, run status tracking.
:::

:::{grid-item-card} Data download
:link: data_utils
:link-type: doc

`utils.data_utils` — resolve tasks to configs and fetch datasets.
:::

:::{grid-item-card} Environment setup
:link: install_utils
:link-type: doc

`utils.install_utils` — install the dataset, the vendored harness, CUDA/vLLM.
:::

:::{grid-item-card} SFT
:link: sft_utils
:link-type: doc

`sft.sft_utils` + `sft.sft_prompts` — dataset prep, trainers, samplers, prompts.
:::

:::{grid-item-card} RFT (verl)
:link: verl_utils
:link-type: doc

`rft.verl.verl_utils` + `rft.verl.rft_prompts` — parquet builders and system prompts.
:::

:::{grid-item-card} `benchmark` pipeline
:link: benchmark
:link-type: doc

`benchmark.eval_utils`, `parse_outputs`, `summarize_*` — the scoring backend.
:::
::::

Two importable modules are deliberately **not** documented here: `medvision_bm.utils.plan_utils`
(`find_plan_files`, `plan_version_of`, `resolve_plan_path`, `load_benchmark_plan` — the annotation-version
*ceiling* resolution every dataset caller depends on) and `medvision_bm.dataset.ds_utils` (the parquet
split/limit loaders). Read their docstrings in the source until pages exist.

```{toctree}
:hidden:
:maxdepth: 1

parse_utils
configs
utils
data_utils
install_utils
sft_utils
sft_prompts
verl_utils
rft_prompts
benchmark
```
