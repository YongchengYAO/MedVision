# Troubleshooting coverage map — `medvision`

Every primary workflow and non-trivial support workflow must have actionable failure guidance:
symptom or error fragment, likely cause, concrete fix, and when to stop. This map records who
owns each failure surface.

| Failure surface | Evidence | Owner | Runtime location | Recovery guidance | Validation |
| --- | --- | --- | --- | --- | --- |
| Package not installed / wrong interpreter | package metadata, launcher env names | root + environment-setup | root `references/troubleshooting.md` §1; `environment-setup/references/troubleshooting.md` | install command per path, import verification | root env checker exits 1 |
| Editable install silently shadowed by a copy | packaging behaviour, README | environment-setup | `environment-setup/references/troubleshooting.md`; root §1 | print the resolved module file; reinstall editable or use the source path | env checker labels the install mode; 1 usability case |
| `huggingface_hub` / `transformers` incompatibility | source comments, frozen requirements | environment-setup | both troubleshooting files | re-pin from the model's requirements file | `check_env_pins.py` exits 1; 1 usability case |
| `datasets>=4` breaking remote code | dependency pin rationale | environment-setup | root §1 | pin the supported version | native import checks |
| torch / torchvision ABI mismatch | requirements pairs | environment-setup | root §1 | reinstall the pair together | documented |
| `setuptools` downgraded by a transitive pin | dataset package dependencies | environment-setup | root §1, sub-skill file | re-pin within the safe range | reproduced during environment preparation |
| Missing system GL library | headless host behaviour | environment-setup | root §1 | headless OpenCV variant | reproduced on this host |
| Shared-filesystem wheel build race | launcher comments | environment-setup | root §1 | build node-local under a lock | bundled helper |
| protobuf / wandb import failure at training start | SFT launchers | sft + environment-setup | root §1, `sft/references/troubleshooting.md` | pin protobuf after environment setup | documented |
| Conda solver failure | launcher preamble | environment-setup | root §1 | classic solver | documented |
| Missing annotation-version pin | loader guard | dataset-and-tasks | root §2, `dataset-and-tasks/references/troubleshooting.md` | set the variable before loading | 1 usability case; loader banners transcribed |
| Acknowledgement required for an older pin | loader guard | dataset-and-tasks | same | set the acknowledgement variable | same case |
| Sample counts differ from the leaderboard | version history, filtering flag | dataset-and-tasks | same | pin the leaderboard version; leave filtering off | same case |
| Gated dataset authorization failure | dataset card, download source | dataset-and-tasks | root §2, sub-skill file | per-source token guidance; whitespace sanitising | documented |
| Stale Arrow cache after a data change | loader download modes | dataset-and-tasks | root §2, sub-skill file | force-redownload plus one of the two data levers | documented |
| Config not found for a CoT-suffixed task name | derivation function | dataset-and-tasks | root §2, sub-skill file | use an SFT-style list, a CSV, or the bundled wrapper | reproduced and added to root troubleshooting |
| Plan version ceiling surprises | plan resolution source | dataset-and-tasks | root §2, sub-skill file | inspect resolved plans offline | ceiling fallback reproduced |
| Disk exhaustion during download | QC-figure sizing | dataset-and-tasks | root §2 | keep figures opt-in; budget the full copy | documented |
| GPU out of memory during evaluation | launcher settings, hardware doc | benchmark-evaluation | root §3, `benchmark-evaluation/references/troubleshooting.md` | expose more GPUs, lower batch, check footprints | documented |
| Tensor-parallel size vs visible GPUs | launcher behaviour | benchmark-evaluation | same | expose exactly the intended GPUs | documented |
| Stale or unwanted resume cache | cache design | benchmark-evaluation | same | hash invalidation explained; disable switch | 1 usability case (planned) |
| Task skipped because it is marked complete | tracker file | benchmark-evaluation | same | status flag or tracker edit | documented |
| Output truncated by the token budget | budget document | benchmark-evaluation | same | raise the budget; recognise the SuccessRate symptom | documented |
| API authorization, credit-hold and parameter errors | model wrappers | benchmark-evaluation | same | key sanitising, provider-specific causes | documented |
| Unsupported model in the resize dispatch | dispatch source | benchmark-evaluation + extending-models-and-tasks | both troubleshooting files | add the branch; checklist | registry consistency script |
| Missing parsed directory or wrong response key | summarizer guards | results-parsing-and-metrics | `results-parsing-and-metrics/references/troubleshooting.md`; root §4 | flag combinations; the summarizer's own abort | native tests |
| All-NaN nMAE | metric source | results-parsing-and-metrics | same | three distinct causes with fixes | native test reproduced cause (b) |
| Threshold metric lower than the mean | metric semantics | results-parsing-and-metrics | same + `metrics.md` | denominators explained | 1 usability case; demo script |
| Duplicated records across ranks | dedupe tool | results-parsing-and-metrics | same | dedupe into a new directory | help-only check |
| Judge interpreter not set | driver behaviour | llm-judge-parsing | `llm-judge-parsing/references/troubleshooting.md` | export the interpreter variable | bundled checker |
| Judge queue fingerprint lock | queue identity design | llm-judge-parsing | same | rebuild or restore the prompt | documented; 16 entries total |
| Destructive preparation step | driver | llm-judge-parsing | same | archive semantics and recovery | never executed |
| Stray non-roster judge directory | invariants gate | llm-judge-parsing | same | remove or exclude the directory | reproduced by native test |
| Judge non-reproducibility | reproducibility document | llm-judge-parsing | same + `design-notes.md` | operating rules; blast radius; determinism switch and its costs | 1 usability case |
| SFT out-of-memory and resume OOM | launcher variants | sft | `sft/references/troubleshooting.md` (pending) + root §3 | knob order, fp32-master variant | parsers verified |
| Dataset-construction memory kill | cgroup helpers | sft + rft | root §3, sub-skill files | reduce workers; use the sharded builder | 1 rft usability case |
| Sample-limit misconfiguration | limit parser | sft | `sft/references/data-preparation.md` | rejected values, bootstrap, validation carve-out | bundled limit checker |
| Parquet reused across model families | prompt coupling | rft | `rft/references/troubleshooting.md` | rebuild with the right family | 1 usability case |
| BiomedParse compiled-dependency build failure | setup script | biomedparse-ablation | `biomedparse-ablation/references/troubleshooting.md` | toolkit prerequisites; expected noise | bundled env checker |
| BiomedParse pin drift after the dataset installer | setup script comments | biomedparse-ablation | same | re-pin as the setup does | env checker reports drift |
| New task never runs | task-list authority | extending-models-and-tasks | `extending-models-and-tasks/references/troubleshooting.md` | register it in the task list | task YAML lister |

## Gaps

- `sft/references/troubleshooting.md` was still being written when this map was drafted; the
  integration gate re-checks it before hand-off, and the root file covers the SFT failure modes
  that overlap with environment and memory issues in the meantime.
- No failure surface is owned only by the root file: each root row points at the sub-skill that
  owns the detail.
