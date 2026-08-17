"""test-2: span verification must make a hallucinated value impossible.

This is the mechanism the whole analysis rests on. If verification can be fooled,
the judge can inject numbers that were never in the response and no downstream
metric would notice.

Run from the repo root:
    python unit-test/llm-parsing/test-2.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from judge_verify import collapse_ws, verify_span

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print("test-2: span verification")

# --- accepts genuine extractions -----------------------------------------
RESP_BOXED = r"...computation...\[\boxed{407.02, 325.62}\]"
r = verify_span(RESP_BOXED, r"\boxed{407.02, 325.62}", [407.02, 325.62], 2)
check("accepts LaTeX \\boxed span", r["ok"] and r["pred"] == "407.02,325.62", r["reason"])

RESP_PROSE = "The major axis length is 20.0 mm and the minor axis length is 18.0 mm."
r = verify_span(RESP_PROSE, "major axis length is 20.0 mm and the minor axis length is 18.0 mm",
                [20.0, 18.0], 2)
check("accepts plain-prose span", r["ok"] and r["pred"] == "20.0,18.0", r["reason"])

# Tag numerals are noise inside the span: NUM_RE reads 1, 0.51, ..., 1 -- six
# numbers -- so exact-list matching would wrongly reject this.
RESP_TAG = "<step-1-answer>0.51, 0.42, 0.69, 0.59</step-1-answer>"
r = verify_span(RESP_TAG, "<step-1-answer>0.51, 0.42, 0.69, 0.59</step-1-answer>",
                [0.51, 0.42, 0.69, 0.59], 4)
check("contiguous-run match survives tag numerals",
      r["ok"] and r["pred"] == "0.51,0.42,0.69,0.59", r["reason"])

# Whitespace: models emit newlines inside answers.
RESP_NL = "<step-1-answer>\n0.45, 0.48,\n0.72, 0.78\n</step-1-answer>"
r = verify_span(RESP_NL, "<step-1-answer> 0.45, 0.48, 0.72, 0.78 </step-1-answer>",
                [0.45, 0.48, 0.72, 0.78], 4)
check("whitespace-insensitive containment", r["ok"], r["reason"])
check("collapse_ws normalizes runs", collapse_ws("a \n\t b  c") == "a b c")

# Transcription comes from the SPAN, not the judge's float: 56.02 must not
# become 56.0 just because the judge typed it that way.
r = verify_span("...= 56.02 mm", "= 56.02 mm", [56.02], 1)
check("transcribes span digits, not judge floats", r["ok"] and r["pred"] == "56.02", r["pred"])

# --- rejects fabrication --------------------------------------------------
r = verify_span(RESP_BOXED, r"\boxed{407.02, 325.62}", [999.9, 111.1], 2)
check("rejects values absent from the span", not r["ok"] and r["reason"] == "values_not_in_span")

r = verify_span(RESP_BOXED, "a span the model never wrote 12.3", [12.3], 1)
check("rejects a span not present in the response",
      not r["ok"] and r["reason"] == "span_not_found")

r = verify_span(RESP_BOXED, "", [407.02, 325.62], 2)
check("rejects empty span", not r["ok"] and r["reason"] == "empty_span")

r = verify_span(RESP_BOXED, r"\boxed{407.02, 325.62}", [407.02], 2)
check("rejects wrong arity", not r["ok"] and r["reason"].startswith("arity_mismatch"))

r = verify_span("no digits here", "no digits here", [], 0)
check("rejects a span with no numbers", not r["ok"])

# Order matters: a re-ordering is not a contiguous run.
r = verify_span(RESP_BOXED, r"\boxed{407.02, 325.62}", [325.62, 407.02], 2)
check("rejects re-ordered values", not r["ok"] and r["reason"] == "values_not_in_span")

# Tolerance is tight enough to catch a genuinely different number.
r = verify_span("value is 0.51", "value is 0.51", [0.52], 1)
check("rejects near-miss 0.51 vs 0.52", not r["ok"])
r = verify_span("value is 0.51", "value is 0.51", [0.51], 1)
check("accepts exact 0.51", r["ok"])

# Repeated numbers: the LAST occurrence wins, matching the strict last-k rule.
r = verify_span("first 5.0 then 5.0 and 7.0", "first 5.0 then 5.0 and 7.0", [5.0, 7.0], 2)
check("prefers the later contiguous run", r["ok"] and r["pred"] == "5.0,7.0", r["pred"])

# --- tiered verification ---------------------------------------------------
# Every shape below is a MEASURED cause from the corpus: of 3,877 span_not_found
# rows, only 19 were the judge hallucinating; the rest were quoting habits.
r = verify_span(RESP_BOXED, r"\boxed{407.02, 325.62}", [407.02, 325.62], 2)
check("tier 1 rows report tier=exact", r["ok"] and r.get("tier") == "exact", r)

# The judge wrapped its quote in quotation marks (22% of failures).
r = verify_span(RESP_PROSE, '"major axis length is 20.0 mm and the minor axis length is 18.0 mm"',
                [20.0, 18.0], 2)
check("normalized: strips a wrapping quote pair",
      r["ok"] and r.get("tier") == "normalized" and r["pred"] == "20.0,18.0", r)

# The judge closed a tag the model's truncated response left open (35%).
RESP_CUT = "working...\n<answer>\n[0.59, 0.27, 0.83, 0.46]"
r = verify_span(RESP_CUT, "<answer> [0.59, 0.27, 0.83, 0.46] </answer>",
                [0.59, 0.27, 0.83, 0.46], 4)
check("normalized: strips a judge-added closing tag",
      r["ok"] and r.get("tier") == "normalized" and r["pred"] == "0.59,0.27,0.83,0.46", r)

# JSON decoding corrupted the span: "\boxed" arrived as backspace + "oxed".
r = verify_span(RESP_BOXED, "\x08oxed{407.02, 325.62}", [407.02, 325.62], 2)
check("normalized: inverts control-char corruption",
      r["ok"] and r.get("tier") == "normalized", r)

# Whitespace DELETED rather than collapsed (16%).
RESP_2LINE = "lengths:\n20.0\nand\n18.0 mm"
r = verify_span(RESP_2LINE, "lengths:20.0and18.0 mm", [20.0, 18.0], 2)
check("normalized: whitespace-deleted quote still locates",
      r["ok"] and r.get("tier") == "normalized", r)

# The judge re-worded the quote entirely, but the values are the model's own,
# in the model's order -- transcribed from the RESPONSE, not the judge's text.
r = verify_span("I measure the axes at 20.07 and 18.03 mm.",
                "The final answer is 20.07, 18.03", [20.07, 18.03], 2)
check("value_anchor: reworded span, values transcribed from response",
      r["ok"] and r.get("tier") == "value_anchor" and r["pred"] == "20.07,18.03", r)

# The anchor searches from the END, consistent with the strict last-k rule.
r = verify_span("draft: 9.0, 9.5 ... final: 20.0 and 18.0", "reworded quote 20.0, 18.0",
                [20.0, 18.0], 2)
check("value_anchor: last contiguous run wins", r["ok"] and r["pred"] == "20.0,18.0", r)

# What NO tier may ever do: accept a value the model never wrote.
r = verify_span(RESP_PROSE, "totally invented span", [99.9, 88.8], 2)
check("no tier accepts fabricated values",
      not r["ok"] and r["reason"] == "span_not_found", r)

# All-zero values with an unlocatable span are the skeleton's own [0.0]
# placeholder echoed back (59 of the 63 leaked rows were arity-1 AD). The
# anchor tier must refuse them: the incidental "0" in this response would
# otherwise bind and score pred=0 for an answer the model never stated.
r = verify_span("the origin is at 0 and the angle measures 47.3 degrees",
                "totally invented span", [0.0], 1)
check("value_anchor refuses all-zero values (echo signature)",
      not r["ok"] and r["reason"] == "all_zero_value_anchor", r)

# ...but a genuinely QUOTED zero is a real answer: the span is the proof.
r = verify_span("computing... Distance = sqrt(0) so Distance = 0.",
                "Distance = 0.", [0.0], 1)
check("exact tier still accepts a model-stated zero",
      r["ok"] and r.get("tier") == "exact" and r["pred"] == "0", r)

# The same thing written in LaTeX. A model that answers "\[ \boxed{0.0} \]" has
# stated its zero just as plainly, but the judge quotes it as "[ 0.0 ]" -- the
# macros stripped. Before _strip_latex that quote was unlocatable, so the sample
# fell to the anchor tier and was refused as an echo, turning 263 stated answers
# (all Qwen2.5-VL-32B, whose ANB working collapses to arccos(1) = 0) into parse
# failures. Locating it restores the "span IS the proof" case above.
RESP_BOXED = ("### Step 3: compute\n\\[\n\\text{angle} = \\arccos(1) = 0\n\\]\n"
              "\n### Final Answer:\n\\[\n\\boxed{0.0}\n\\]")
r = verify_span(RESP_BOXED, "### Final Answer: [ 0.0 ]", [0.0], 1)
check("normalized tier accepts a LaTeX-boxed zero (judge strips the macros)",
      r["ok"] and r.get("tier") == "normalized" and r["pred"] == "0.0", r)

# The same span for a non-zero boxed answer must keep working -- the guard was
# never the only thing standing between these rows and acceptance.
r = verify_span("### Final Answer:\n\\[\n\\boxed{8.1}\n\\]",
                "### Final Answer: [ 8.1 ]", [8.1], 1)
check("normalized tier accepts a LaTeX-boxed non-zero",
      r["ok"] and r["pred"] == "8.1", r)

# De-LaTeXing is for LOCATION only. It must not let a value the model never
# wrote through, and an unlocatable all-zero span stays an echo even when the
# response happens to contain LaTeX.
r = verify_span(RESP_BOXED, "### Final Answer: [ 42.0 ]", [42.0], 1)
check("de-LaTeX does not admit a fabricated value", not r["ok"], r)
r = verify_span("the origin is at \\(0\\) and the angle measures 47.3 degrees",
                "totally invented span", [0.0], 1)
check("de-LaTeX leaves the echo guard intact",
      not r["ok"] and r["reason"] == "all_zero_value_anchor", r)
# ...or a re-ordering of real values (not a contiguous run in the response).
r = verify_span(RESP_PROSE, "reworded", [18.0, 20.0], 2)
check("no tier accepts re-ordered values", not r["ok"], r)

# The normalized tier deletes whitespace on BOTH sides to locate a span. If the
# judge's quote dropped a separator sitting between two numeric tokens, NUM_RE
# reads a MERGED number out of the span that never existed in the response.
# I4 must hold at tier 2 as well: every transcribed number has to appear, in
# order, in the response's own NUM_RE run.
RESP_SPLIT = "Landmark table:\nthe values are\n2\n0.5\nand that is all we know"
r = verify_span(RESP_SPLIT, "the values are20.5and that", [20.5], 1)
check("normalized tier refuses a number fused by whitespace deletion",
      not r["ok"] and r["reason"] == "normalized_values_not_in_response", r)

# ...but the whitespace-deleted quote of numbers that ARE real still passes.
r = verify_span(RESP_2LINE, "lengths:20.0and18.0 mm", [20.0, 18.0], 2)
check("normalized tier still accepts genuinely present values",
      r["ok"] and r.get("tier") == "normalized" and r["pred"] == "20.0,18.0", r)

# Tier 1 must FALL THROUGH, not terminate, when the span is located but its own
# numbers do not contain the judge's values. That is a disagreement about which
# numbers the quote holds -- exactly what tier 2 already falls through on -- and
# treating it as terminal discarded 185 real recoveries corpus-wide.
RESP_WIDE = "step 1 gives 3.0 and 4.0\nFinal: the answer is 20.0, 18.0 mm"
r = verify_span(RESP_WIDE, "step 1 gives 3.0 and 4.0", [20.0, 18.0], 2)
check("tier 1 falls through to the value anchor instead of terminating",
      r["ok"] and r.get("tier") == "value_anchor" and r["pred"] == "20.0,18.0", r)

# Falling through must not weaken I4: values absent from the response stay rejected,
# and the reason distinguishes "quote disagreed" from "quote not found".
r = verify_span(RESP_WIDE, "step 1 gives 3.0 and 4.0", [99.9, 88.8], 2)
check("fall-through still refuses values absent from the response",
      not r["ok"] and r["reason"] == "values_not_in_span", r)

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-2: all checks passed")
