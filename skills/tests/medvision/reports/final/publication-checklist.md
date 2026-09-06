# Publication checklist — `medvision`

| Item | Status |
| --- | --- |
| Runtime tree contains only skill content (`SKILL.md`, `references/`, `scripts/`, `sub-skills/`) | yes |
| No check-only artifacts inside the runtime tree | yes; all cases and reports live under the artifact root |
| No caches, build output, logs, or symlinks | yes; one stray build log and two harness directories were removed |
| No local paths, environment prefixes, tokens, or machine details | yes; verified by grep over the whole tree |
| Frontmatter contract on every `SKILL.md` | yes; 11/11 |
| One consistent licence value across the tree | yes; `NOASSERTION` since the 2026-09-04 refresh (was `NO_LICENSE`), recorded with its resolver status |
| Runtime routing metadata present and minimal | yes; matches the external classification exactly |
| All relative links resolve inside the tree | yes; 202 links, 0 broken |
| Self-contained: no runtime instruction executes a repository path | yes; two checkout-only pipelines are explicitly reference-only |
| Every bundled script exercised at least with `--help` | yes |
| Provenance recorded for a later refresh | yes, including the dirty-checkout state |
| Usability cases and reports kept outside the published tree | yes |

## Known cosmetic issue

Most bundled scripts are mode 644 rather than 755. The permission gate on the authoring host
declined the change. This has no functional effect: every documented invocation runs them as
`python <script>` or `bash <script>`, which does not require the executable bit. A publisher may
normalise the modes.

## Do not publish

The artifact root (`skills/tests/medvision/`) and the routing decision folder
(`skills/disco/routing_decision/`) are review evidence, not skill content. The environment report
under `reports/env/` contains local paths and must never be copied into the runtime tree.
