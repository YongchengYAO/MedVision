"""test-3: harmony channel splitting and tolerant JSON extraction.

A reasoning model may emit an ``analysis`` channel before ``final``, and whether
vLLM strips it depends on the model, the version and reasoning-parser
configuration. The pipeline must handle
BOTH shapes and must never depend on constrained decoding having worked, because a
vLLM upgrade would then silently change results rather than just the invalid rate.

Brace-counting matters here: judge spans routinely contain braces
(``\\boxed{407.02}``), so a naive regex for ``{.*}`` would truncate the object.

Run from the repo root:
    python unit-test/llm-parsing/test-3.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from judge_decode import (
    extract_first_json_object,
    iter_json_object_candidates,
    parse_judge_json,
    preescape_latex_escapes,
    repair_json_escapes,
    split_final_channel,
    validate_judge_obj,
)

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print("test-3: harmony decoding + tolerant JSON")

# --- channel splitting ----------------------------------------------------
PLAIN = '{"final_answer": {"status": "no_conclusion", "span": "", "values": []}}'
check("passes through already-stripped text", split_final_channel(PLAIN) == PLAIN)

HARMONY = (
    "<|channel|>analysis<|message|>Let me look for the answer...<|end|>"
    "<|start|>assistant<|channel|>final<|message|>" + PLAIN + "<|return|>"
)
check("strips analysis channel and end token", split_final_channel(HARMONY) == PLAIN,
      repr(split_final_channel(HARMONY)))

TWO_FINAL = ("<|channel|>final<|message|>{\"a\":1}<|end|>"
             "<|channel|>final<|message|>" + PLAIN)
check("takes the LAST final channel", split_final_channel(TWO_FINAL) == PLAIN)
check("empty input is safe", split_final_channel("") == "" and split_final_channel(None) == "")

# --- balanced-object extraction ------------------------------------------
check("extracts a simple object", extract_first_json_object('noise {"a": 1} tail') == '{"a": 1}')

NESTED = '{"final_answer": {"status": "present", "values": [1, 2]}}'
check("handles nesting", extract_first_json_object("x " + NESTED + " y") == NESTED)

# A span containing braces -- the case a regex would break on.
BRACED = r'{"final_answer": {"status": "present", "span": "\\boxed{407.02, 325.62}", "values": [407.02, 325.62]}}'
got = extract_first_json_object(BRACED)
check("braces inside a string literal do not confuse the scanner", got == BRACED,
      repr(got))
check("...and it still parses", json.loads(got)["final_answer"]["values"] == [407.02, 325.62])

ESCAPED = r'{"span": "he said \"{\" once", "values": []}'
check("escaped quotes handled", extract_first_json_object(ESCAPED) == ESCAPED)

check("unbalanced object returns None", extract_first_json_object('{"a": 1') is None)
check("no object returns None", extract_first_json_object("no json here") is None)

# --- end-to-end parse -----------------------------------------------------
obj, reason = parse_judge_json(HARMONY)
check("parse_judge_json on harmony", obj is not None and reason == "ok", reason)

obj, reason = parse_judge_json("<|channel|>final<|message|>not json at all")
check("reports no_json_object", obj is None and reason == "no_json_object", reason)

obj, reason = parse_judge_json("<|channel|>final<|message|>{bad json}")
check("reports json_decode_error", obj is None and reason == "json_decode_error", reason)

obj, reason = parse_judge_json("")
check("reports empty_output", obj is None and reason == "empty_output", reason)

# Truncated JSON is exactly what max_tokens=256 would have caused on a TL row;
# it must fail loudly rather than silently yield a partial object.
obj, reason = parse_judge_json('{"final_answer": {"status": "present", "span": "abc')
check("truncated object is rejected", obj is None, reason)

# --- shape validation -----------------------------------------------------
ok, r = validate_judge_obj(json.loads(PLAIN), expects_steps=False)
check("valid no_conclusion object", ok, r)

ok, r = validate_judge_obj({"final_answer": {"status": "present", "span": "x 1", "values": [1]}}, False)
check("valid present object", ok, r)

# The prompt's own skeleton echoed back. This exact object -- byte for byte --
# appeared on 3,428 Detection rows: a clean "no_conclusion" verdict beside the
# [0.0]*arity placeholder the skeleton used to show. Rejecting it as a
# self-contradiction stranded 3,407 correct verdicts in "undetermined".
ok, r = validate_judge_obj(
    {"final_answer": {"status": "no_conclusion", "span": "", "values": [0.0, 0.0, 0.0, 0.0]}}, False)
check("accepts an echoed all-zero placeholder", ok, r)
ok, r = validate_judge_obj(
    {"final_answer": {"status": "no_conclusion", "span": "", "values": [0]}}, False)
check("accepts a single integer zero", ok, r)
# ...but a NON-zero value beside a non-present status is still the judge
# disagreeing with itself, and still fails.
ok, r = validate_judge_obj(
    {"final_answer": {"status": "no_conclusion", "span": "", "values": [0.0, 1.0]}}, False)
check("still rejects a non-zero value", (not ok) and r == "values_with_non_present_status", r)
# False == 0 in Python, so a naive all-zero test would swallow booleans.
ok, r = validate_judge_obj(
    {"final_answer": {"status": "no_conclusion", "span": "", "values": [False]}}, False)
check("still rejects booleans", (not ok) and r == "values_with_non_present_status", r)

for bad, why in [
    ({}, "missing_final_answer"),
    ({"final_answer": {"status": "maybe"}}, "bad_status"),
    # Both withdrawn statuses must be REJECTED: "absent" from v1, and "truncated"
    # from the short-lived three-way version. That rejection is what makes a
    # judge-out file written under either prompt unusable rather than silently
    # reinterpretable under the current two-way contract.
    ({"final_answer": {"status": "absent", "span": "", "values": []}}, "bad_status"),
    ({"final_answer": {"status": "truncated", "span": "", "values": []}}, "bad_status"),
    # A non-present status carrying values means the judge contradicted itself.
    ({"final_answer": {"status": "no_conclusion", "span": "", "values": [1]}},
     "values_with_non_present_status"),
    # isinstance(True, int) is True in Python, so a naive numeric check would
    # accept booleans and hand 1/0 to the span verifier as measurements.
    ({"final_answer": {"status": "present", "span": "x", "values": [True]}},
     "non_numeric_values"),
    ({"final_answer": {"status": "present", "span": "", "values": [1]}}, "missing_span"),
    ({"final_answer": {"status": "present", "span": "x", "values": []}}, "missing_values"),
    ({"final_answer": {"status": "present", "span": "x", "values": ["a"]}}, "non_numeric_values"),
]:
    ok, r = validate_judge_obj(bad, False)
    check(f"rejects {why}", (not ok) and r == why, r)

# --- candidate selection --------------------------------------------------
# Every fixture below is a SHAPE TAKEN FROM PRODUCTION. Under the old
# first-balanced-group rule each one decoded to LaTeX from the analysis prose and
# the row was written off as judge-invalid; the extractions quoted here (`{mm}`,
# `{217.07, 110.01}`) are the literal blobs those rows produced.
OBJ = '{"final_answer": {"status": "present", "span": "20.0 mm", "values": [20.0]}}'

for name, prefix, first in [
    ("LaTeX unit brace", r"The result is in \text{mm} units." + "\n", "{mm}"),
    ("LaTeX coordinate brace", r"\boxed{217.07, 110.01}" + "\n", "{217.07, 110.01}"),
    # A lone inch mark opens a string the old single-pass scanner never closed,
    # blinding it to every object in the rest of the response.
    ("desyncing quote", 'It is 12" wide {x}\n', "{x}"),
]:
    text = prefix + OBJ
    check(f"first balanced group is still LaTeX: {name}",
          extract_first_json_object(text) == first, repr(extract_first_json_object(text)))
    obj, reason = parse_judge_json(text)
    check(f"but the judge object is selected: {name}",
          reason == "ok" and obj.get("final_answer", {}).get("values") == [20.0], reason)

check("candidates are yielded left to right",
      list(iter_json_object_candidates('a {1} b {2} c')) == ["{1}", "{2}"])
check("a scan is not desynchronised by an earlier unmatched quote",
      '{"k": 1}' in list(iter_json_object_candidates('say "hi {n} then {"k": 1}')))

# The discriminator is final_answer; without one, any parsed dict still beats
# nothing, so a row that resolved before cannot stop resolving.
obj, reason = parse_judge_json('{"other": 1}')
check("falls back to a final_answer-less dict", reason == "ok" and obj == {"other": 1}, reason)
obj, reason = parse_judge_json('{"a": 1} ' + OBJ)
check("prefers the candidate carrying final_answer",
      reason == "ok" and "final_answer" in obj, repr(obj))

# --- escape repair --------------------------------------------------------
# The judge is told to quote VERBATIM and these responses are full of LaTeX, so a
# faithful span carries \[ and \] which are not JSON escapes.
LATEX_SPAN = r'{"final_answer": {"status": "present", "span": "so\n\[\n55.00\n\]", "values": [55.0]}}'
obj, reason = parse_judge_json(LATEX_SPAN)
check("repairs an invalid escape in a verbatim span",
      reason == "ok" and obj["final_answer"]["values"] == [55.0], reason)

check("valid escapes are left alone", repair_json_escapes(r'{"a":"\n\t\"\\\/"}')[1] is False)
check("a complete \\uXXXX is left alone", repair_json_escapes(r'{"a":"é"}')[1] is False)
check("a short \\u is repaired", repair_json_escapes(r'{"a":"\u12"}') == (r'{"a":"\\u12"}', True))
check("a trailing lone backslash is repaired", repair_json_escapes("\\") == ("\\\\", True))
check("repair is idempotent",
      repair_json_escapes(repair_json_escapes(r'{"a":"\[")}')[0])[1] is False)

# --- LaTeX macros that are LEGAL escapes ----------------------------------
# \b and \f decode SUCCESSFULLY, so a repair keyed on parse failure never sees
# them: "$\boxed{187.09}$" quietly became "$\x08oxed{187.09}$" on 100 shipped
# rows and span verification could no longer find the quote.
BOXED = r'{"final_answer": {"status": "present", "span": "$\boxed{187.09, 166.28}$", "values": [187.09, 166.28]}}'
obj, reason = parse_judge_json(BOXED)
check("a verbatim \\boxed span survives decoding",
      reason == "ok" and "\\boxed" in obj["final_answer"]["span"]
      and "\x08" not in obj["final_answer"]["span"], repr(obj))
FRAC = r'{"final_answer": {"status": "present", "span": "\frac{1}{2} of 5.0 mm", "values": [5.0]}}'
obj, reason = parse_judge_json(FRAC)
check("a verbatim \\frac span survives decoding",
      reason == "ok" and "\\frac" in obj["final_answer"]["span"], repr(obj))
check("preescape leaves an escaped backslash alone",
      preescape_latex_escapes(r'{"a":"\\boxed"}') == r'{"a":"\\boxed"}')
check("preescape leaves \\n alone (spans need real newlines)",
      preescape_latex_escapes(r'{"a":"x\ny"}') == r'{"a":"x\ny"}')
check("preescape is idempotent",
      preescape_latex_escapes(preescape_latex_escapes(BOXED)) == preescape_latex_escapes(BOXED))

# --- harmony markers as vLLM actually emits them --------------------------
# The offline LLM() path decodes the control tokens into bare words, so the
# channel switch is the WORD "assistantfinal", not <|channel|>final<|message|>.
# Without splitting on it the candidate scan reads the chain-of-thought too.
HARMONY_PLAIN = ('analysisThe response mentions {"final_answer": {"status": "present", '
                 '"span": "draft", "values": [9.9]}} as a draft.'
                 'assistantfinal{"final_answer": {"status": "present", "span": "20.0 mm", "values": [20.0]}}')
obj, reason = parse_judge_json(HARMONY_PLAIN)
check("plain-word assistantfinal splits the channels",
      reason == "ok" and obj["final_answer"]["values"] == [20.0], repr(obj))
check("text with neither marker still passes through",
      split_final_channel("no markers here") == "no markers here")

# --- scan bounds ----------------------------------------------------------
# A degenerate repetition loop reaches ~68K characters and can open a brace on
# every line; the bound is what keeps one such row off the critical path.
check("candidate count is bounded", len(list(iter_json_object_candidates("{x} " * 5000))) <= 64)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-3: all checks passed")
