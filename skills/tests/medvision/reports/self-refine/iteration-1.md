# Self-refine iteration 1 — `medvision`

## Method

Three fresh agents, none with access to my research notes, were each given one usability prompt and
restricted to reading only the runtime skill directory. They could execute bundled scripts but could
not open the repository source, its documentation, or the artifact directory. Each returned an answer
plus an honest navigation log. I graded the answers against the corresponding `assertions.json`.

This is the strongest check available here: it measures whether the skill is usable by an agent that
has nothing else, which is exactly the deployment condition.

## Eval 1 — novice orientation (root routing)

Prompt: what is MedVision measuring, what do I need installed, and can this machine run an
evaluation?

**Grade: 8/8 PASS.**

| Assertion | Grade | Evidence |
| --- | --- | --- |
| Names the three task families and their units | PASS | produced the family table with mm and degrees |
| Install commands for both packages | PASS | both, including the dataset package installed into the data directory |
| Names the required environment variables | PASS | data directory, annotation version, and the acknowledgement variable |
| Runs or cites the bundled environment checker | PASS | ran it and reported a concrete verdict |
| Separates CPU-only from GPU-only capabilities | PASS | stated which pipeline steps work today and which need GPUs |
| Routes to the evaluation sub-skill | PASS | plus the extension sub-skill, correctly, since the user's model is not in the roster |
| Does not depend on files outside the skill directory | PASS | navigation log confirms |
| Does not leak authoring-environment details | PASS | paths it printed are the user's own machine, reported by the checker at runtime |

Notable behaviour beyond the assertions: it caught that three secrets on the host carried trailing
whitespace and gave the sanitising command, proposed a two-sample smoke test into a throwaway results
tag before spending a GPU allocation, and correctly warned that an unregistered model needs an
image-size dispatch branch before any measurement task will run.

## Eval 2 — metric troubleshooting

Prompt: mean IoU 0.62 versus `IoU>0.5` at 0.41, is the summarizer buggy?

**Grade: 7/7 PASS.**

It gave the correct explanation (different denominators, detection failures scored as zero and
included in the mean, the threshold comparison being inclusive), refused to call it a bug, and
contrasted the detection convention with the measurement tasks. It went further than required with an
independent feasibility bound showing the quoted pair implies a success rate of at least about 0.83,
and listed the concrete cross-file mistakes that would explain the pair otherwise.

**Caveat on this eval's strength.** The 0.62/0.41 example originated from the sub-skill's own drafting
agent, which proposed it as a difficult case and also wrote it into the troubleshooting table. My
usability case adopted those numbers. So this eval largely tests retrieval and explanation rather
than generalisation. The independent arithmetic the agent added is the part that genuinely
demonstrates depth. A future iteration should re-test this capability with numbers that do not appear
anywhere in the skill.

**One follow-up checked and dismissed.** The agent reported that the bundled metrics demo failed with
a missing-package error. I reproduced it: the script catches the import failure and prints the reason
plus a recovery instruction, which is the required behaviour. The agent had quoted only the reason
line. No change needed.

## Eval 3 — cross-skill integration

Prompt: complete tumour/lesion numbers for our own model on four H100s, leaderboard-comparable,
including the format-robust report, from a clean machine, with checks at each stage.

**Grade: 7/7 PASS.**

| Assertion | Grade |
| --- | --- |
| Sequences evaluation, parsing, summarizing and judge re-parsing in order | PASS |
| Names the owning sub-skill for each step | PASS |
| Pins the annotation version and states the sample limit | PASS, as an explicit run-identity table before any command |
| Checks the environment before proposing GPU work | PASS, ran the bundled checker and reported the verdict |
| States the judge needs a separate environment with a newer serving stack | PASS |
| Gives a verification checkpoint after each stage | PASS, every stage ends with one |
| Explains the diff between the two reports | PASS, and identified the judge decomposition report as the reviewer-facing artefact |

Behaviour beyond the assertions, all of it desirable:

- It opened with two clarifying questions it could not answer from the skill, the model's
  architecture family and whether the user holds the published results tree, and said explicitly that
  it was not guessing. Both genuinely change the procedure.
- It used the bundled launcher generator to produce the actual run script rather than hand-writing a
  command.
- It drew a distinction the assertions did not ask for and that matters: if responses are being cut
  off at the token limit, that is a truncation problem, not a formatting one, and the judge pass will
  not repair it. Raising the budget also requires clearing the affected cache shards, because the
  cache key covers the prompt but not the budget.
- It correctly scoped the published judge runtime figure to the whole 19-model roster rather than
  implying it applied to one model's tumour/lesion run.

## Eval 4 — unseen composition (added after evals 1-3)

Eval 2 reused an example the skill documents verbatim, so it tested retrieval more than
generalisation. To close that hole I ran a fourth eval on a scenario that appears nowhere in the
skill: a student evaluated a model on tumour/lesion "using the latest annotations at the time", left,
and the user now wants to publish an 8% improvement against that year-old number.

Answering it requires composing at least five separate facts that live in four different sub-skills,
none of which is stated as this scenario.

**Grade: PASS on every dimension I set for it.** The agent:

- Identified tumour/lesion as the only task family whose annotations change between releases, so the
  comparison is unsafe for this family specifically while it would be fine for the other two.
- Noticed, without being told, that the shipped tumour/lesion evaluation list draws entirely from the
  datasets whose train/test split moved in the latest annotation release, making the two runs a
  different benchmark rather than a noisy version of the same one.
- Raised a contamination risk the prompt never hinted at: if the new model was fine-tuned on this
  data, a moved subject-level split can put training subjects into the test set.
- Gave a forensic procedure to recover the unknown version from the raw records, having first checked
  and correctly reported that the record schema does not store it.
- Found the one documented harmoniser between two specific versions and correctly scoped it,
  including that it does not bridge the later releases.
- Separated a second, independent problem the user had not asked about: the relative-error metric
  averages over successful parses only, so it must be quoted with the success rate.
- Recommended a volume-level clustered bootstrap rather than treating slices as independent, citing
  the reason slices from one volume are correlated.
- Stated plainly that it could not determine the old run's version from the skill alone.

This is composition rather than retrieval, and it is the strongest evidence in this pass that the
skill is usable by an agent that has nothing else. It also validates the routing: the answer drew on
the dataset, results, evaluation and judge sub-skills plus the root glossary without being told which
to open.

## Summary of the graded evals

| Eval | Target | Assertions | Passed |
| --- | --- | --- | --- |
| 1 | root routing, novice | 8 | 8 |
| 2 | metric troubleshooting | 7 | 7 |
| 3 | cross-skill integration | 7 | 7 |
| 4 | unseen composition | qualitative | pass |

22 of 22 scored assertions passed, plus a qualitative pass on the unseen-composition eval. No
revision was required by any of the four. All four agents worked from the skill alone, and none
needed to open the repository. Eval 4 specifically addresses the retrieval-versus-generalisation
caveat raised against eval 2.

## Revisions made as a result of this pass

None required by evals 1 and 2. Both capabilities behaved as designed. Separate defects found during
integration, before this pass, were: 55 broken cross-links, an entry-point count error, an
overstated token-budget rule, and a stray build log in the tree. All were fixed.
