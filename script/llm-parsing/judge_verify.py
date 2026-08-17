"""Span verification -- the mechanism that makes a hallucinated value impossible.

The judge is a POINTER, not a transcriber. It quotes a verbatim span; this module
re-derives the numbers from that span with the benchmark's own ``NUM_RE`` and
accepts the record only if the judge's reported values are actually there. The
value that reaches ``filtered_resps`` is the one the *regex* read out of the span,
never the float the judge typed -- so a wrong digit cannot survive even if the
judge and the span disagree in a way the tolerance would forgive.

Why contiguous-subsequence and not exact-list matching
------------------------------------------------------
A span carries context by design (the judge is told to include >=12 characters so
the quote can be located). Context contains numbers:

    "<step-1-answer>0.51, 0.42, 0.69, 0.59</step-1-answer>"

``NUM_RE`` reads SIX numbers there -- the ``1`` from ``step-1-answer`` at each end.
Requiring the judge's values to equal the full derived list would reject this
correct extraction. Requiring them to be a *contiguous run* inside the derived
list accepts it while still making fabrication impossible: a number that is not
in the span cannot match, and a re-ordering cannot match either.

Why verification is TIERED
--------------------------
Measured on the full corpus (2026-08-07): of 3,877 strict-fail rows rejected as
``span_not_found``, only 19 (0.49%) were the judge actually hallucinating. The
rest were the matcher being stricter than the judge's quoting: the judge closes a
tag the model's truncated response left open (35%), wraps the quote in quotation
marks (22%), normalises whitespace (16%), re-punctuates (12%), or the span text
was corrupted by JSON escape decoding (``\\boxed`` -> backspace + ``oxed``).
Every one of those rows was a real answer discarded over formatting.

So verification proceeds in three tiers, strictest first, and the tier that
accepted a row is recorded for audit:

  exact         The original contract: whitespace-collapsed span containment,
                values transcribed from the span.
  normalized    Same containment after normalisations that cannot invent a
                number: casefold, whitespace deletion, control-char inversion,
                stripping a wrapping quote pair or a judge-added closing tag.
                Values still transcribed from the (repaired) span.
  value_anchor  The span could not be located, but the judge's values appear as
                a CONTIGUOUS run of ``NUM_RE`` numbers in the RESPONSE itself
                (searched from the end, consistent with the strict parser's
                last-k rule). Values transcribed from the response text.

The guarantee that survives all three tiers: every digit that reaches
``filtered_resps`` exists verbatim in the model's response, in the order written
there. What ``value_anchor`` gives up is only the span's role as PROOF of
location -- it becomes a hint. A value the model never wrote remains impossible
at every tier.
"""

import math
import re

from judge_io import find_numbers

_WS_RE = re.compile(r"\s+")
_ALL_WS_RE = re.compile(r"\s+")

# Control characters that only ever arise from JSON decoding a verbatim LaTeX
# macro (``"\boxed"`` -> backspace + ``oxed``). Inverted before matching; a real
# response cannot contain them, so applying the inversion to both sides is safe.
_CTRL_INVERSIONS = (("\x08", "\\b"), ("\x0c", "\\f"))

# A closing tag the judge appended to tidy a truncated response -- the single
# largest span_not_found cause (35%): the model's output ends at
# "<answer>\n[0.59, ..." and the judge quotes it WITH the "</answer>" it never
# wrote.
_TRAILING_TAG_RE = re.compile(r"\s*</[A-Za-z][\w-]*>\s*$")

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`"))

# LaTeX maths wrappers, reduced for the containment key only (see _strip_latex).
# The delimiters map to their PLAIN equivalents rather than being deleted: the
# judge quotes ``\[ ... \]`` as ``[ ... ]``, so dropping the bracket on one side
# only would move the two keys further apart instead of together.
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_MATH_DELIMS = (("\\[", "["), ("\\]", "]"), ("\\(", "("), ("\\)", ")"))

# Relative tolerance when matching the judge's reported float against the number
# the regex read from the span. This absorbs JSON float round-trip only; it is far
# too tight to let a genuinely different number pass (0.51 vs 0.52 fails by 1e-2).
_REL_TOL = 1e-9


def collapse_ws(text):
    """Collapse all whitespace runs to a single space and strip.

    Models emit newlines inside answers (``<step-1-answer>\\n0.45, 0.48\\n</...>``),
    so containment must be whitespace-insensitive or valid spans fail to locate.
    """
    return _WS_RE.sub(" ", text or "").strip()


def _close(a, b):
    """Return True if two floats agree to within the round-trip tolerance.

    Non-finite operands are rejected outright. With either side infinite the
    tolerance test degenerates to ``inf <= inf`` -- True -- so an infinite value
    would match EVERY finite number and the "far too tight to let a genuinely
    different number pass" guarantee would silently invert. Both sides can go
    infinite in normal operation: ``json.loads`` decodes a bare ``Infinity`` or
    ``1e400`` from the judge, and ``NUM_RE`` can lift an overflowing literal out of
    the response (6 such rows exist in the production corpus). NaN already
    compared False; this makes inf behave the same way.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= _REL_TOL * max(1.0, abs(a), abs(b))


def _find_contiguous_run(derived_floats, values):
    """Return the start index of ``values`` inside ``derived_floats``, or -1.

    Searches from the END, so when a span contains the same number twice the
    later occurrence wins -- consistent with the strict parser's last-k rule.
    """
    n, k = len(derived_floats), len(values)
    if k == 0 or k > n:
        return -1
    for start in range(n - k, -1, -1):
        if all(_close(derived_floats[start + j], values[j]) for j in range(k)):
            return start
    return -1


def _invert_ctrl(text):
    """Undo JSON escape decoding of LaTeX macros (backspace -> ``\\b`` etc.)."""
    for ctrl, literal in _CTRL_INVERSIONS:
        text = text.replace(ctrl, literal)
    return text


def _strip_latex(text):
    """Reduce LaTeX maths wrappers to the plain text a judge quotes them as.

    ``\\boxed{X}`` becomes ``X`` and the display/inline math delimiters become
    their plain bracket equivalents, so a response written
    ``\\[ \\boxed{0.0} \\]`` and a span quoted ``[ 0.0 ]`` reduce to the same key.

    Without this the judge's quote of a boxed answer is unlocatable, the sample
    falls through to the value-anchor tier, and a genuinely stated zero is refused
    there as the skeleton's echo signature -- the one case tiers 1-2 are supposed
    to catch, because there the span IS the proof. Measured on the production
    corpus: 263 such rows, all Qwen2.5-VL-32B, which answers in ``\\boxed{}``
    rather than the ``<answer>`` tags and whose ANB working collapses to
    ``arccos(1) = 0``.

    Only ever applied inside ``_norm_key``, i.e. to the containment test. It can
    therefore change WHERE a span is located, never WHICH digits are transcribed:
    numbers come from the span/response text, and tier 2's I4 guard re-checks
    every transcribed number against the raw response.
    """
    text = _BOXED_RE.sub(r"\1", str(text))
    for macro, plain in _MATH_DELIMS:
        text = text.replace(macro, plain)
    return text


def _norm_key(text):
    """Reduce text to a form insensitive to the judge's quoting habits.

    Casefolded, whitespace DELETED (not collapsed -- the judge drops newlines
    entirely more often than it converts them to spaces), control chars inverted,
    LaTeX maths wrappers reduced to plain text (see ``_strip_latex``).
    Used only for the containment test; numbers are never derived from this.
    """
    return _ALL_WS_RE.sub("", _strip_latex(_invert_ctrl(str(text)))).casefold()


def _span_variants(span):
    """Yield progressively repaired versions of the judge's span, mildest first.

    Each variant strips something the judge ADDED (a quote pair, a closing tag
    it invented for a truncated response) or restores something JSON decoding
    corrupted. None of them can introduce a number that was not already there.
    """
    seen = set()
    base = _invert_ctrl(str(span)).strip()
    candidates = [base]
    stripped = base
    for open_q, close_q in _QUOTE_PAIRS:
        if len(stripped) >= 2 and stripped.startswith(open_q) and stripped.endswith(close_q):
            stripped = stripped[1:-1].strip()
            break
    candidates.append(stripped)
    candidates.append(_TRAILING_TAG_RE.sub("", base).strip())
    candidates.append(_TRAILING_TAG_RE.sub("", stripped).strip())
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            yield c


def _values_in_response(response, numbers):
    """True if ``numbers`` appear as a contiguous NUM_RE run in ``response``.

    The I4 invariant in one predicate: a number that is not literally in the
    model's own text, in the order written there, cannot be scored. Used to
    police the normalized tier, whose whitespace-deleting match would otherwise
    let two adjacent tokens fuse into a value the model never wrote.
    """
    derived = find_numbers(response)
    try:
        return _find_contiguous_run([float(x) for x in derived],
                                    [float(x) for x in numbers]) >= 0
    except (TypeError, ValueError):
        return False


def verify_span(response, span, values, expected_arity=None):
    """Verify a judge-reported span and transcribe its numbers, in three tiers.

    Args:
        response: The full raw model response the span must come from.
        span: The verbatim quote the judge returned.
        values: The numbers the judge reported for that span.
        expected_arity: Required value count, or ``None`` to accept any count.

    Returns:
        dict with keys:
            ok (bool)          -- whether the record may be used
            reason (str)       -- "ok", or why it was rejected
            tier (str)         -- which tier accepted: "exact" | "normalized" |
                                  "value_anchor" (absent on rejection)
            numbers (list[str])-- regex-transcribed numbers, digits as written in
                                  the span (exact/normalized) or the response
                                  (value_anchor)
            pred (str)         -- comma-joined ``numbers``, ready for filtered_resps
    """
    fail = lambda why: {"ok": False, "reason": why, "numbers": [], "pred": ""}

    if not span or not str(span).strip():
        return fail("empty_span")
    if values is None:
        return fail("no_values")
    if expected_arity is not None and len(values) != expected_arity:
        return fail(f"arity_mismatch:{len(values)}!={expected_arity}")

    try:
        want = [float(v) for v in values]
    except (TypeError, ValueError):
        return fail("non_numeric")

    def _accept(source_text, tier):
        """Contiguous-run check + transcription against ``source_text``."""
        derived = find_numbers(source_text)
        if not derived:
            return fail("no_numbers_in_span")
        try:
            derived_floats = [float(x) for x in derived]
        except (TypeError, ValueError):
            return fail("non_numeric")
        start = _find_contiguous_run(derived_floats, want)
        if start < 0:
            return fail("values_not_in_span")
        # Transcribe from the TEXT, not from the judge's floats, so the exact
        # digits the model wrote are what gets scored.
        matched = derived[start : start + len(want)]
        return {"ok": True, "reason": "ok", "tier": tier,
                "numbers": matched, "pred": ",".join(matched)}

    # Tier 1 -- the original contract, unchanged: whitespace-collapsed
    # containment, numbers from the span as given.
    located = False
    if collapse_ws(span) in collapse_ws(response):
        exact = _accept(span, "exact")
        if exact["ok"]:
            return exact
        # Located, but the span's own NUM_RE reading does not contain the judge's
        # values as a contiguous run. That is a DISAGREEMENT about which numbers
        # the quote holds, not a failure to find the quote -- and tier 2 already
        # treats the identical situation as "fall through" (see the break below).
        # Returning here made tier 1 terminal, so a judge that quoted a slightly
        # wider span than it transcribed never reached the tiers designed to
        # resolve exactly that: ~185-191 final-answer recoveries were discarded
        # corpus-wide. NOTE the wider blast radius: apply_judge._apply_steps calls
        # this same function per prescribed step, where the fall-through flips on
        # the order of 17,000 Job B step entries from absent to present. Steps are
        # persisted and never scored, so no published metric moves -- but any
        # later analysis built on LLM_judge_steps will see materially more.
        # Falling through cannot fabricate: tier 2 re-checks against the response
        # (I4 guard) and tier 3 requires the values to be a contiguous run in the
        # response itself.
        located = True

    # Tier 2 -- locate a repaired variant of the span under a quoting-insensitive
    # key. Numbers come from the VARIANT (original spacing intact), so nothing
    # the normalisation did can merge or invent a digit.
    resp_key = _norm_key(response)
    for variant in _span_variants(span):
        if _norm_key(variant) in resp_key:
            result = _accept(variant, "normalized")
            if result["ok"]:
                # I4 guard. _norm_key DELETES whitespace on both sides, so if the
                # judge's quote dropped a separator that sat between two numeric
                # tokens, NUM_RE reads a MERGED number out of the variant that
                # never existed in the response: response "...2\n0.5..." quoted as
                # "the values are20.5and" transcribes 20.5. Tier 1 collapses
                # whitespace and so cannot merge; tier 3 already re-checks against
                # the response. Require the same of tier 2 -- every transcribed
                # number must appear, in order, in the RESPONSE's own NUM_RE run.
                if _values_in_response(response, result["numbers"]):
                    return result
                return fail("normalized_values_not_in_response")
            # Located here too, so the terminal reason below must say so. Setting
            # this only at tier 1 left a variant that tier 2 DID locate reporting
            # span_not_found -- the very mislabel the tier-1 change removed.
            located = True
            break  # located but values not in it: a real disagreement, fall through

    # Tier 3 -- the span cannot be located, but the judge's values are a
    # contiguous run of the numbers the RESPONSE itself contains (searched from
    # the end, consistent with the strict parser's last-k rule). The digits that
    # get scored are transcribed from the response, so a value the model never
    # wrote remains impossible; only the span's role as location PROOF is lost.
    #
    # All-zero values never reach this tier: ``[0.0] * k`` is the historical
    # prompt skeleton's own placeholder (judge_prompts._output_skeleton), so an
    # unlocatable span carrying exactly that value is the echo signature, not
    # evidence. Without location proof, any incidental ``0`` in the response
    # would anchor it and score ``pred=0`` for an answer the model never stated
    # (measured: 63 rows corpus-wide, 59 of them arity-1 AD). Tiers 1-2 still
    # accept genuine quoted zeros ("Distance = 0.") because there the span IS
    # the proof.
    if all(v == 0 for v in want):
        return fail("all_zero_value_anchor")
    anchored = _accept(response, "value_anchor")
    if anchored["ok"]:
        return anchored
    # Report WHICH failure this was. "span_not_found" is wrong for a span that was
    # located verbatim and merely disagreed about its numbers, and that distinction
    # is what the Stage 4 span-rejection table exists to show.
    return fail("values_not_in_span" if located else "span_not_found")
