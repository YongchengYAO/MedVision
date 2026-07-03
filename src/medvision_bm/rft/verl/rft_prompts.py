"""System prompts that enforce the reasoning output format for RFT.

Both prompts instruct the assistant to think through the problem internally and
then give a final answer, wrapping the reasoning in ``<think>`` ... ``</think>``
tags and the final answer in ``<answer>`` ... ``</answer>`` tags. They are
supplied as the ``system`` message when building verl RFT records.

Two variants are provided:

- ``SYSTEM_PROMPT``: the full chain-of-thought (CoT) format. In addition to the
  ``<think>``/``<answer>`` structure, it asks the model to report each reasoning
  step inside ``<step-k-reasoning>`` ... ``</step-k-reasoning>`` tags followed by
  the intermediate result inside ``<step-k-answer>`` ... ``</step-k-answer>``
  tags. This structure exposes per-step landmark coordinates used for
  process-level rewards, and is paired with the CoT formatters.
- ``SYSTEM_PROMPT_LITE``: the minimal variant that keeps only the
  ``<think>``/``<answer>`` structure without the per-step tags, paired with the
  non-CoT ("lite") formatters.
"""

SYSTEM_PROMPT = (
    "A conversation between a User and an Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks through the reasoning process internally, then provides the User with the answer. "
    "The reasoning process and the final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. For example: <think> reasoning process here </think> <answer> answer here </answer>. "
    "Within the <think> </think> tags, report the reasoning process for each step inside <step-k-reasoning> </step-k-reasoning> tags, followed by the intermediate results in <step-k-answer> </step-k-answer> tags. "
    "For example: <think> <step-1-reasoning> reasoning for step 1 </step-1-reasoning> <step-1-answer> intermediate result from step 1 </step-1-answer> </think>."
)

SYSTEM_PROMPT_LITE = (
    "A conversation between a User and an Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks through the reasoning process internally, then provides the User with the answer. "
    "The reasoning process and the final answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. For example: <think> reasoning process here </think> <answer> answer here </answer>. "
)
