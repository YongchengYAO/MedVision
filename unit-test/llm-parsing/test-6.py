"""test-6: the prompt shows the schema it enforces, and the fingerprint sees it all.

Two failures this guards against, both of which have already happened.

1. THE PROMPT DESCRIBED THE OBJECT WITHOUT SHOWING IT.
   v1 named ``status``/``span``/``values`` in prose and never printed the object.
   Run without a decoding grammar, the v1 reader invented its own schema on 43,935
   of 43,938 TL rows and 37,080 of 37,080 AD rows. Every extraction was correct and
   every one was discarded. The skeleton is now GENERATED from the same specs
   ``build_schema`` reads, so these asserts fail the moment the two drift.

2. THE FINGERPRINT LISTED INGREDIENTS INSTEAD OF THE RENDERED PROMPT.
   ``NO_STEPS_NOTE`` (the whole steps block for every Detection row),
   ``MIN_SPAN_CONTEXT_CHARS`` and ``RESPONSE_ELISION_MARKER`` were all absent from
   the payload, so editing any of them changed what the judge saw without changing
   the cache key -- silently reusing the previous prompt's answers.

Run from the repo root:
    python unit-test/llm-parsing/test-6.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

import judge_config as JC
import judge_prompts as JP

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print("test-6: prompt/schema coherence and fingerprint completeness")

CASES = [
    ("TL", "TL"),
    ("AD", "AD:distance"),
    ("AD", "AD:angle"),
    ("AD", None),
    ("Detection", None),
]

# --- the skeleton is valid JSON and matches the schema exactly -------------
for task, step_key in CASES:
    label = f"{task}/{step_key}"
    raw = JP._output_skeleton(task, step_key)
    try:
        obj = json.loads(raw)
        ok_json = True
    except json.JSONDecodeError as e:
        obj, ok_json = {}, False
        check(f"{label}: skeleton is valid JSON", False, str(e))
    if not ok_json:
        continue
    check(f"{label}: skeleton is valid JSON", True)

    schema = JP.build_schema(task, step_key)
    check(f"{label}: skeleton keys == schema required",
          set(obj) == set(schema["required"]),
          f"{sorted(obj)} vs {sorted(schema['required'])}")

    fa_required = schema["properties"]["final_answer"]["required"]
    check(f"{label}: final_answer keys == schema required",
          set(obj["final_answer"]) == set(fa_required))
    check(f"{label}: final_answer arity matches TASK_SPECS",
          len(obj["final_answer"]["values"]) == JC.TASK_SPECS[task]["arity"])
    # maxItems is the grammar's ceiling; the skeleton must not exceed it.
    check(f"{label}: skeleton arity <= schema maxItems",
          len(obj["final_answer"]["values"])
          <= schema["properties"]["final_answer"]["properties"]["values"]["maxItems"])
    # The placeholder must NOT be a legal value. It was [0.0] * arity once, and
    # the judge echoed it verbatim beside non-present statuses (3,428 Detection
    # rows at repair time). A slot-marker string can never echo into a score.
    check(f"{label}: values placeholder is not a legal value",
          all(v == "<number>" for v in obj["final_answer"]["values"]),
          obj["final_answer"]["values"])

    if step_key is None:
        check(f"{label}: no steps key", "steps" not in obj)
    else:
        specs = JC.STEP_SPECS[step_key]
        check(f"{label}: step count matches STEP_SPECS", len(obj["steps"]) == len(specs))
        check(f"{label}: per-step arity matches STEP_SPECS",
              all(len(s["values"]) == spec["n_values"]
                  for s, spec in zip(obj["steps"], specs)))
        check(f"{label}: step values placeholders are not legal values",
              all(v == "<number>" for s in obj["steps"] for v in s["values"]))
        check(f"{label}: step indices are 1..n in order",
              [s["index"] for s in obj["steps"]] == [sp["index"] for sp in specs])

# --- the rendered system message actually carries the skeleton -------------
for task, step_key in CASES:
    label = f"{task}/{step_key}"
    messages, _ = JP.build_messages(task, step_key, "a response")
    system = messages[0]["content"]
    check(f"{label}: system message embeds the skeleton verbatim",
          JP._output_skeleton(task, step_key) in system)
    check(f"{label}: system message names both statuses",
          all(s in system for s in JC.FINAL_ANSWER_STATUSES))
    # Truncation is decided from the generation config, not the text. The prompt
    # must not reintroduce it as something the judge reports.
    check(f"{label}: prompt does not ask the judge to report truncation",
          '"truncated"' not in system)
    check(f"{label}: system message explains the elision marker",
          "characters elided" in system)
    check(f"{label}: no unsubstituted format placeholders",
          "{min_ctx}" not in system and "{output_skeleton}" not in system)

# --- the fingerprint moves when anything the judge sees moves --------------
BASE = {(t, k): JP.prompt_fingerprint(t, k) for t, k in CASES}
check("fingerprints differ across every (task, step_key)",
      len(set(BASE.values())) == len(CASES))


def moves(name, case, mutate, restore):
    """Assert that a mutation changes the fingerprint for ``case``."""
    mutate()
    try:
        changed = JP.prompt_fingerprint(*case) != BASE[case]
    finally:
        restore()
    check(f"fingerprint tracks {name}", changed)
    # And that the restore actually restored, or later checks are meaningless.
    check(f"fingerprint restored after {name}",
          JP.prompt_fingerprint(*case) == BASE[case])


_note = JP.NO_STEPS_NOTE
moves("NO_STEPS_NOTE (the entire Detection steps block)", ("Detection", None),
      lambda: setattr(JP, "NO_STEPS_NOTE", _note + "\nEDIT"),
      lambda: setattr(JP, "NO_STEPS_NOTE", _note))

_ctx = JP.MIN_SPAN_CONTEXT_CHARS
moves("MIN_SPAN_CONTEXT_CHARS (substituted into the prompt)", ("TL", "TL"),
      lambda: setattr(JP, "MIN_SPAN_CONTEXT_CHARS", _ctx + 7),
      lambda: setattr(JP, "MIN_SPAN_CONTEXT_CHARS", _ctx))

_marker = JP.RESPONSE_ELISION_MARKER
moves("RESPONSE_ELISION_MARKER", ("TL", "TL"),
      lambda: setattr(JP, "RESPONSE_ELISION_MARKER", "\n<<CUT {n}>>\n"),
      lambda: setattr(JP, "RESPONSE_ELISION_MARKER", _marker))

_sys = JP.SYSTEM_PROMPT
moves("SYSTEM_PROMPT", ("TL", "TL"),
      lambda: setattr(JP, "SYSTEM_PROMPT", _sys + "\nEDIT"),
      lambda: setattr(JP, "SYSTEM_PROMPT", _sys))

_steps = JP.STEPS_BLOCK_TEMPLATE
moves("STEPS_BLOCK_TEMPLATE", ("TL", "TL"),
      lambda: setattr(JP, "STEPS_BLOCK_TEMPLATE", _steps + "\nEDIT"),
      lambda: setattr(JP, "STEPS_BLOCK_TEMPLATE", _steps))

_user = JP.USER_TEMPLATE
moves("USER_TEMPLATE", ("TL", "TL"),
      lambda: setattr(JP, "USER_TEMPLATE", _user + "\nEDIT"),
      lambda: setattr(JP, "USER_TEMPLATE", _user))

_tok = JC.TASK_SPECS["TL"]["max_tokens"]
moves("max_tokens (a smaller budget truncates the JSON)", ("TL", "TL"),
      lambda: JC.TASK_SPECS["TL"].__setitem__("max_tokens", 64),
      lambda: JC.TASK_SPECS["TL"].__setitem__("max_tokens", _tok))

_eff = JP.JUDGE_REASONING_EFFORT
moves("JUDGE_REASONING_EFFORT", ("TL", "TL"),
      lambda: setattr(JP, "JUDGE_REASONING_EFFORT", "high"),
      lambda: setattr(JP, "JUDGE_REASONING_EFFORT", _eff))

_win = JP.RESPONSE_WINDOW_TAIL
moves("RESPONSE_WINDOW_TAIL", ("TL", "TL"),
      lambda: setattr(JP, "RESPONSE_WINDOW_TAIL", _win + 1),
      lambda: setattr(JP, "RESPONSE_WINDOW_TAIL", _win))

# The short stamp is what actually reaches the queue rows, so a fingerprint that
# moves but a stamp that does not would leave the resume guard blind.
check("short stamp is 16 hex chars",
      all(len(JP.short_prompt_fp(t, k)) == 16 for t, k in CASES))
check("short stamps are distinct across cases",
      len({JP.short_prompt_fp(t, k) for t, k in CASES}) == len(CASES))
_sys2 = JP.SYSTEM_PROMPT
before = JP.short_prompt_fp("TL", "TL")
JP.SYSTEM_PROMPT = _sys2 + "\nEDIT"
after = JP.short_prompt_fp("TL", "TL")
JP.SYSTEM_PROMPT = _sys2
check("short stamp moves with the prompt", before != after)
check("short stamp restored", JP.short_prompt_fp("TL", "TL") == before)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-6: all checks passed")
