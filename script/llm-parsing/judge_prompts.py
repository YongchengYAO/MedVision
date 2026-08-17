"""LLM-as-Judge output parsing — prompt assembly and response schema.

Design notes
------------
Ground-truth blind. The judge sees only the response text. It never sees
``target``, never sees the image, and is never asked whether a value is correct.
A judge that scores correctness is a judge whose errors are indistinguishable
from the model's errors.

The judge is a pointer, not a transcriber. Every extracted value must come with a
verbatim ``span`` quoted from the response. ``apply_judge.py`` re-derives the
numbers from that span with the benchmark's own ``_NUM_RE`` and rejects the record
if they disagree. Hallucinated digits are therefore structurally impossible, not
merely unlikely.

Span before values. Under constrained decoding the model emits fields in schema
order, so quoting first and transcribing second means the transcription is
conditioned on the quote rather than the other way around.

Show the object, do not describe it (v2)
----------------------------------------
The v1 prompt named every field in prose -- ``"status"``, ``"span"``, ``"values"``,
``"absent"`` -- and never once showed the object. Run without a decoding grammar,
the v1 reader invented a plausible schema of its own::

    {"final_answer": "(40.542, 28.799)",
     "steps": [{"step": "1", "content": "The endpoints of the major axis: ..."}]}

That happened on 43,935 of 43,938 TL rows and 37,080 of 37,080 AD rows. The
*extraction* was right every time; only the container was wrong, and every one of
them was discarded. ``_output_skeleton`` now renders the literal object into the
system prompt, generated from the same ``TASK_SPECS``/``STEP_SPECS`` that
``build_schema`` reads, so the shape shown and the shape enforced cannot drift.
"""

import json

from judge_config import (
    FINAL_ANSWER_STATUSES,
    JUDGE_REASONING_EFFORT,
    MAX_ANSWER_SPAN_CHARS,
    MAX_STEP_SPAN_CHARS,
    MIN_SPAN_CONTEXT_CHARS,
    RESPONSE_ELISION_MARKER,
    RESPONSE_WINDOW_HEAD,
    RESPONSE_WINDOW_TAIL,
    RESPONSE_WINDOW_TRIGGER,
    STEP_SPECS,
    STEP_STATUSES,
    TASK_SPECS,
)

SYSTEM_PROMPT = """You extract numeric answers from a vision-language model's response to a medical measurement task.

You are given ONLY the response text. You never see the image and never see the ground truth. Do not judge correctness. Do not compute anything. Do not infer a value that is not written. Report only what the response literally states.

QUOTING
- Quote every span VERBATIM from the response, copied character for character. Never paraphrase, never reformat or round a number, never fix a typo.
- Include at least {min_ctx} characters of surrounding context in a span so it can be located unambiguously.
- The numbers you report must be exactly the numbers that appear in the span you quoted, in the order they are written there.

WHAT COUNTS AS A FINAL ANSWER
- A final answer counts as stated no matter how it is wrapped. All of these are stated answers: <answer>...</answer>, \\boxed{{...}}, **Answer:** ..., <final-answer>...</final-answer>, a LaTeX display, or a plain sentence such as "The major axis length is 20.0 mm and the minor axis length is 18.0 mm".
- The final answer is the quantity the TASK line asks for, in the unit it names. It is NOT an intermediate coordinate. If the response works through numbered steps, the final answer is the value the last step computes, never the (x, y) pairs an early step reports.
- If the response states several candidate answers and never settles on one, quote the LAST one stated.

CHOOSING final_answer.status -- exactly one of two values
- "present"        The response states a final answer. Use this even when the text is cut off AFTER the answer was stated, and even if the answer looks wrong. Once an answer is on the page, the status is "present".
- "no_conclusion"  No final answer is stated anywhere: the response refuses, only describes the image, says it cannot measure, is empty, stops part-way through its working, or simply ends without giving the requested quantity.

Do not try to work out WHY an answer is missing, and do not report it. Whether the response ran out of room or chose not to answer is decided elsewhere, from the generation settings; it cannot be read off the text and guessing at it is worse than leaving it alone. Your job is only whether an answer is there.

A line of the form
    ...[12345 characters elided]...
means WE removed the middle of a long response before showing it to you. The text after it is the model's real ending, so keep reading to the end before concluding that no answer is stated.

Set "span" to "" and "values" to [] whenever status is not "present".

OUTPUT
Return a single JSON object and nothing else: no prose, no markdown fence, no explanation, no trailing commentary. It must have exactly these keys, in this shape:

{output_skeleton}

Every key shown is required. Do not add keys. Do not rename keys. Do not replace an object with a string. "values" holds bare JSON numbers -- never strings, never tuples, never expressions."""


USER_TEMPLATE = """TASK: the model was asked to produce {description}.
EXPECTED_ANSWER_COUNT: {arity} number(s).
{steps_block}
RESPONSE:
<<<
{response}
>>>

Return only the JSON object."""


STEPS_BLOCK_TEMPLATE = """
The model was also instructed to show its work in {n_steps} numbered steps before answering. Extract what each step reports, or mark it absent if the response does not report it. A step counts as present however it is labelled -- "<step-2-answer>", "Step 2:", "**Step 2**", or an unlabelled paragraph that plainly does that step's work.
{step_lines}

IMPORTANT: these steps are INTERMEDIATE. Put them in "steps", never in "final_answer". In particular a <step-1-answer> block holds coordinates, not the answer -- taking it as the final answer is the single most common way to get this wrong.
"""


NO_STEPS_NOTE = """
This task prescribes a single reasoning step whose result IS the final answer, so a <step-1-answer> block here does hold the answer and counts as a stated final answer. Return no "steps" array.
"""


def _windowed_response(text):
    """Return the response text, windowed to head + tail if it is very long.

    Degenerate repetition loops reach ~68K characters. The middle of such a
    response carries no information the judge needs, and the final answer is
    always at the tail. The elision is marked explicitly so the judge keeps reading
    to the real ending instead of treating our cut as the model's -- otherwise a
    windowed response with a perfectly good answer at the tail would come back
    ``no_conclusion``.

    Args:
        text: The raw model response.

    Returns:
        tuple: ``(windowed_text, was_windowed)``.
    """
    if text is None:
        return "", False
    if len(text) <= RESPONSE_WINDOW_TRIGGER:
        return text, False
    n_elided = len(text) - RESPONSE_WINDOW_HEAD - RESPONSE_WINDOW_TAIL
    marker = RESPONSE_ELISION_MARKER.format(n=n_elided)
    return (
        text[:RESPONSE_WINDOW_HEAD] + marker + text[-RESPONSE_WINDOW_TAIL:],
        True,
    )


def _output_skeleton(task_type, step_key):
    """Render the literal JSON object the judge must emit.

    Generated from ``TASK_SPECS``/``STEP_SPECS`` -- the same source
    ``build_schema`` reads -- so the arity shown to the model and the arity
    enforced by the grammar cannot drift apart.

    The placeholder values are chosen so that NONE of them is also a legal
    answer. ``values`` renders as ``"<number>"`` slot markers (same style as
    the ``span`` placeholder); the status strings spell the whole enum inline
    so the model never has to recall it from the prose above.

    History: under every fingerprint stamped before 2026-08-08 (the ACCEPT_FP
    list in run_llm_parsing.sh / DESIGN.md section 7), ``values`` rendered as
    ``[0.0] * arity`` -- the one placeholder that WAS a legal answer -- and the
    judge echoed it: 3,440 Detection rows in the live judge-out carry a clean
    ``"no_conclusion"`` beside ``[0.0, 0.0, 0.0, 0.0]``.
    ``judge_decode.validate_judge_obj`` keeps its all-zero exemption so that
    historical raw stays reparsable, and ``judge_verify.verify_span`` refuses
    all-zero values at the value_anchor tier -- see both before touching this.

    Changing this placeholder moved ``prompt_fingerprint``: the first queue
    build after the change mints new fps for all three tasks and rebuilds
    ~1.1 GB of queues. Existing judge-out applies fine via ACCEPT_FP.

    Args:
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        step_key: Key into ``STEP_SPECS``, or ``None`` for a task without steps.

    Returns:
        str: A pretty-printed JSON object, ready to interpolate into the prompt.
    """
    obj = {
        "final_answer": {
            "status": " | ".join(FINAL_ANSWER_STATUSES),
            "span": "<verbatim quote from the response>",
            "values": ["<number>"] * TASK_SPECS[task_type]["arity"],
        }
    }
    if step_key is not None:
        obj["steps"] = [
            {
                "index": s["index"],
                "status": " | ".join(STEP_STATUSES),
                "span": "<verbatim quote from the response>",
                "values": ["<number>"] * s["n_values"],
            }
            for s in STEP_SPECS[step_key]
        ]
    return json.dumps(obj, indent=2)


def _steps_block(step_key):
    """Render the Job B instruction block for a task, or the step-less note.

    The two branches say opposite things about ``<step-1-answer>`` on purpose:
    for T/L and A/D it holds intermediate coordinates, for Detection it holds the
    bounding box itself. A single shared instruction cannot be right for both --
    conflating them made the judge return step-1 coordinates as the T/L answer.
    """
    if step_key is None:
        return NO_STEPS_NOTE
    specs = STEP_SPECS[step_key]
    lines = "\n".join(
        f"  Step {s['index']}: {s['what']} ({s['n_values']} number(s))" for s in specs
    )
    return STEPS_BLOCK_TEMPLATE.format(n_steps=len(specs), step_lines=lines)


def build_messages(task_type, step_key, response_text):
    """Assemble the chat messages for one judge call.

    The output skeleton goes in the SYSTEM message, not the user message, so it
    sits inside the prefix shared by every row in a ``step_key`` group. With
    ``enable_prefix_caching=True`` that costs one prefill per group rather than
    one per row.

    Args:
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        step_key: Key into ``STEP_SPECS``, or ``None`` for a task without steps.
        response_text: The raw model response to extract from.

    Returns:
        tuple: ``(messages, was_windowed)`` where ``messages`` is a list of
        ``{"role", "content"}`` dicts ready for ``LLM.chat``.
    """
    spec = TASK_SPECS[task_type]
    windowed, was_windowed = _windowed_response(response_text)
    user = USER_TEMPLATE.format(
        description=spec["description"],
        arity=spec["arity"],
        steps_block=_steps_block(step_key),
        response=windowed,
    )
    system = SYSTEM_PROMPT.format(
        min_ctx=MIN_SPAN_CONTEXT_CHARS,
        output_skeleton=_output_skeleton(task_type, step_key),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, was_windowed


def build_schema(task_type, step_key):
    """Build the JSON schema for a judge response.

    Exact per-step arity is deliberately NOT encoded as ``minItems``. Forcing a
    count during decoding would make the model pad or invent numbers to satisfy
    the grammar; ``apply_judge.py`` checks arity afterwards and marks a mismatch
    invalid, which fails loudly instead of fabricating.

    Args:
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        step_key: Key into ``STEP_SPECS``, or ``None`` for a task without steps.

    Returns:
        dict: A JSON Schema object.
    """
    spec = TASK_SPECS[task_type]
    final_answer = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "span", "values"],
        "properties": {
            "status": {"type": "string", "enum": list(FINAL_ANSWER_STATUSES)},
            "span": {"type": "string", "maxLength": MAX_ANSWER_SPAN_CHARS},
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "maxItems": spec["arity"],
            },
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["final_answer"],
        "properties": {"final_answer": final_answer},
    }
    if step_key is not None:
        n_steps = len(STEP_SPECS[step_key])
        max_step_values = max(s["n_values"] for s in STEP_SPECS[step_key])
        schema["required"].append("steps")
        schema["properties"]["steps"] = {
            "type": "array",
            "maxItems": n_steps,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "status", "span", "values"],
                "properties": {
                    "index": {"type": "integer", "minimum": 1, "maximum": n_steps},
                    "status": {"type": "string", "enum": list(STEP_STATUSES)},
                    "span": {"type": "string", "maxLength": MAX_STEP_SPAN_CHARS},
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "maxItems": max_step_values,
                    },
                },
            },
        }
    return schema


# Used only to render the messages for fingerprinting. NUL-delimited so it cannot
# collide with real response text, and short enough never to trigger windowing --
# which is why the window constants are listed separately in the payload below.
_FINGERPRINT_SENTINEL = "\x00SENTINEL-RESPONSE\x00"


def prompt_fingerprint(task_type, step_key):
    """Return a stable fingerprint of everything that reaches the judge.

    Built by RENDERING the messages for a sentinel response rather than by listing
    the ingredients that go into them. The list-the-ingredients version this
    replaces shipped three silent holes:

    - ``NO_STEPS_NOTE`` -- the entire steps block for every Detection row -- was
      never in the payload, so editing it would have reused 415,278 stale answers.
    - ``MIN_SPAN_CONTEXT_CHARS`` is substituted into the system prompt at
      ``build_messages`` time, but the payload copied the *unformatted* template.
    - ``RESPONSE_ELISION_MARKER`` was absent entirely.

    Rendering closes the class of bug rather than the three instances. Anything
    that cannot appear in a rendered sentinel message -- the window sizes, the
    elision marker, the decode settings that change the answer rather than its
    formatting -- is listed explicitly.

    Args:
        task_type: One of ``"TL"``, ``"AD"``, ``"Detection"``.
        step_key: Key into ``STEP_SPECS``, or ``None`` for a task without steps.

    Returns:
        str: Deterministic JSON, suitable for hashing into a cache key.
    """
    messages, _ = build_messages(task_type, step_key, _FINGERPRINT_SENTINEL)
    payload = {
        "messages": messages,
        "schema": build_schema(task_type, step_key),
        # Windowing never fires on the sentinel, so these cannot be captured by
        # rendering and must be named.
        "window": [RESPONSE_WINDOW_TRIGGER, RESPONSE_WINDOW_HEAD, RESPONSE_WINDOW_TAIL],
        "elision_marker": RESPONSE_ELISION_MARKER,
        # Decode settings that change the ANSWER, not merely its formatting.
        "max_tokens": TASK_SPECS[task_type]["max_tokens"],
        "reasoning_effort": JUDGE_REASONING_EFFORT,
    }
    return json.dumps(payload, sort_keys=True)


def short_prompt_fp(task_type, step_key):
    """Return the 16-hex stamp of ``prompt_fingerprint``.

    Written onto every queue row and carried through to every judge-output row, so
    Stage 1 can tell "already judged" from "already judged UNDER A DIFFERENT
    PROMPT". Shortened because the full fingerprint is a multi-kilobyte JSON blob
    and the Detection queue has 415,278 rows.
    """
    from judge_io import content_hash

    return content_hash(prompt_fingerprint(task_type, step_key))[:16]
