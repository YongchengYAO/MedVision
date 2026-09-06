# Release notes

This documentation is **versioned**. Each released version of `medvision_bm` has its
own build of this site, and each build opens with that release's notes. Use the
version switcher (the flyout at the bottom-right of the page on Read the Docs) to move
between versions:

- **latest** tracks the `master` branch — the newest, possibly-unreleased state.
- **v1.1.1**, and every future tagged release, is a frozen snapshot of the docs as they
  stood at that Git tag.

Versioned docs start at **v1.1.1**; releases before that predate this documentation site.

```{toctree}
:maxdepth: 1

v1.2.0
v1.1.1
```

## Version history

| Version | Highlights |
| --- | --- |
| [v1.2.0](v1.2.0.md) | LLM-as-Judge output parsing, the Clinical Decision Agreement (CDA) analysis suite, key model-evaluation bugfixes (decode budgets, mid-run reinstall, answer-tag parsing), COCO-grid Acc@IoU and A/D group rows, the Qwen2.5-VL tool-use eval driver, and dataset v1.2.0–v1.4.0 integration. |
| [v1.1.1](v1.1.1.md) | API-model evaluators (Claude, Gemini, GPT, Kimi), GLM-4.6V/Gemma-4/Qwen3-VL wrappers, F1/Precision/Recall + nMAE, dataset v1.1.1 integration, the resumable output cache, and this documentation site. |

Upstream copies of the codebase release notes (v1.1.0 onward; v1.0.0 predates them) live in
[`docs/codebase-release/`](https://github.com/YongchengYAO/MedVision/tree/master/docs/codebase-release)
in the repository.

## Adding a version (maintainers)

Read the Docs builds one version per Git ref. To publish the docs for a new release:

1. Copy the release note into `docsite/source/releases/`, e.g. `releases/v1.2.0.md`
   (promote its top heading to a single `#` H1 and its sections to `##`), and add it to
   the `toctree` above.
2. Tag the release commit and push the tag:

   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

3. On the Read the Docs dashboard, open **Versions**, then **activate** the new tag.
   Read the Docs builds it from `.readthedocs.yaml` and adds it to the version switcher.

The very first activation is a one-time step: after the initial `v1.1.1` tag is pushed,
activate **v1.1.1** on the dashboard so the switcher appears.

:::{tip}
To skip the per-release dashboard step, add a Read the Docs **Automation Rule**
(Admin → Automation Rules → *Activate version*, matching the SemVer / `v*` tag pattern).
Every tag you push afterwards is then activated and added to the version switcher
automatically — future versions show up on their own.
:::
