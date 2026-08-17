"""Decoding helpers for judge output: reasoning channels and tolerant JSON.

Kept separate from ``run_judge_vllm.py`` so every pure function here is unit
testable without importing vLLM or touching a GPU.

Why tolerant parsing rather than trusting constrained decoding
--------------------------------------------------------------
A reasoning model may emit its chain of thought before its answer, in a channel
format of its own. Whether ``output.text`` arrives already stripped to the final
channel, and whether a JSON grammar binds to that channel, depends on the model,
the vLLM version and whether a reasoning parser is active -- behaviour that has
shifted across vLLM releases and is under-tested on the offline ``LLM()`` path.

So the pipeline never depends on it: we request JSON in the prompt, attempt
structured output, and ALWAYS run these tolerant parsers. Anything that still
fails validation is marked ``invalid``, which simply leaves the strict parser's
own result standing -- the published value where the regex succeeded, and ``""``
where it did not. The judge never overwrites a strict success, so a vLLM upgrade
can move the judge-invalid RATE but can never make a metric wrong.

The channel splitting below is written for the *harmony* format (an ``analysis``
channel preceding a ``final`` one). It is a no-op on a response that has no such
markers, so it costs a model that does not speak harmony nothing -- and it is what
lets ``reparse_judge_out.py`` still read the judge-out archives on disk from the
reader retired on 2026-08-17, which did. Do not delete it to tidy up.

Why the parser SELECTS a candidate instead of taking the first
--------------------------------------------------------------
A reasoning channel is prose about a medical measurement, so it is full of
LaTeX, and LaTeX is full of braces. Taking the first brace-balanced group handed
``json.loads`` things like ``{mm}`` and ``{217.07, 110.01}`` -- both real
extractions from production -- and wrote the row off as judge-invalid while the
judge's actual object sat further down the same string. ``parse_judge_json``
therefore walks every candidate and picks the one carrying ``final_answer``,
falling back to the first that parses at all so a response that resolved before
still resolves.
"""

import json

from judge_config import FINAL_ANSWER_STATUSES

_FINAL_MARKER = "<|channel|>final<|message|>"
# What the marker ACTUALLY looks like in production: vLLM 0.11's offline LLM()
# path decodes the harmony control tokens into bare words, so the channel switch
# arrives as "...analysis...assistantfinal{...}" with no <|...|> anywhere.
# Measured on the 2,000-row Detection probe: 0 rows contain "<|channel|>",
# 1,972 contain "assistantfinal". Without this split the candidate scan reads
# the model's chain-of-thought too, and a draft object in the analysis could be
# selected over the real verdict.
_PLAIN_FINAL_MARKER = "assistantfinal"
_END_TOKENS = ("<|return|>", "<|end|>", "<|endoftext|>")

# Candidate-scan bounds. A degenerate repetition loop reaches ~68K characters and
# can open a brace on every line; without these, one such row would dominate the
# wall clock of a 400K-row pass.
_MAX_CANDIDATE_STARTS = 64
_MAX_SCAN_CHARS = 400_000

_JSON_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_HEXDIGITS = frozenset("0123456789abcdefABCDEF")

# JSON escapes that are also how common LaTeX macros begin. See
# preescape_latex_escapes for why exactly these two and not \t/\r/\n.
_LATEX_CONTROL_ESCAPES = frozenset("bf")


def split_final_channel(text):
    """Return the final-channel content of a harmony response.

    Handles both shapes: already-stripped text (no marker present) and a raw
    harmony string carrying the analysis channel first. Takes the content after
    the LAST final marker, since a response may legitimately contain more than one.

    Args:
        text: Raw generated text.

    Returns:
        str: The final-channel content, end tokens stripped.
    """
    if not text:
        return ""
    if _FINAL_MARKER in text:
        text = text.rsplit(_FINAL_MARKER, 1)[1]
    elif _PLAIN_FINAL_MARKER in text:
        # The production shape: control tokens decoded to bare words. Take the
        # LAST occurrence for the same reason as above. Text with neither marker
        # passes through untouched -- already-stripped output, or a response
        # truncated before its final channel (which then has no verdict to find).
        text = text.rsplit(_PLAIN_FINAL_MARKER, 1)[1]
    for tok in _END_TOKENS:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def _scan_balanced(text, start, budget):
    """Return the end index (exclusive) of the balanced ``{...}`` at ``start``.

    Brace-counting rather than a regex, because the payload contains quoted spans
    that may themselves contain braces (``\\boxed{407.02}`` is a real span). Skips
    braces inside string literals and honours backslash escapes.

    Args:
        text: The text being scanned.
        start: Index of a ``{`` in ``text``.
        budget: Maximum characters this scan may consume.

    Returns:
        tuple: ``(end, consumed)``. ``end`` is ``None`` if the object never
        closes or the budget ran out first.
    """
    depth = 0
    in_string = False
    escaped = False
    limit = min(len(text), start + budget)
    for i in range(start, limit):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1, i + 1 - start
    return None, limit - start


def iter_json_object_candidates(text):
    """Yield every balanced ``{...}`` substring of ``text``, left to right.

    One INDEPENDENT scan per opening brace, which is the point. A single
    left-to-right pass tracks string state globally, so one unmatched ``"`` in the
    model's prose -- an apostrophe-heavy sentence, a stray inch mark -- flips the
    scanner into "inside a string" for the entire rest of the text and hides every
    real object after it. Restarting per brace cannot be desynchronised by
    anything that came before.

    Bounded on both axes so a pathological response cannot cost seconds: at most
    ``_MAX_CANDIDATE_STARTS`` opening braces are tried, sharing a total character
    budget of ``_MAX_SCAN_CHARS``.

    Args:
        text: Text that should contain a JSON object.

    Yields:
        str: Candidate object source text, outermost-first at each start.
    """
    if not text:
        return
    budget = _MAX_SCAN_CHARS
    starts = 0
    pos = 0
    while starts < _MAX_CANDIDATE_STARTS and budget > 0:
        start = text.find("{", pos)
        if start == -1:
            return
        starts += 1
        end, consumed = _scan_balanced(text, start, budget)
        budget -= consumed
        if end is not None:
            yield text[start:end]
        pos = start + 1


def extract_first_json_object(text):
    """Extract the first balanced ``{...}`` object from text.

    Semantics changed with the candidate scan: this now returns the first
    BALANCED group anywhere in the text, where the original returned ``None``
    whenever the group opened by the first ``{`` never closed. No production
    caller depends on the old reading (only tests use this), and
    ``parse_judge_json`` no longer trusts it alone: on a harmony response the
    first balanced group is frequently LaTeX from the analysis prose (``{mm}``
    and ``{217.07, 110.01}`` are both real extractions from production
    failures), not the judge's object.

    Args:
        text: Text that should contain a JSON object.

    Returns:
        str | None: The object's source text, or ``None`` if none is balanced.
    """
    for blob in iter_json_object_candidates(text):
        return blob
    return None


def preescape_latex_escapes(blob):
    """Double the backslash in ``\\b`` / ``\\f`` so LaTeX macros survive decoding.

    The dangerous LaTeX macros are not the ones that BREAK the parse -- those are
    caught below -- but the ones that decode cleanly to the wrong thing:
    ``\\b`` and ``\\f`` are LEGAL JSON escapes, so a verbatim ``\\boxed{...}``
    or ``\\frac{...}`` parses without complaint and the span silently becomes
    ``\\x08oxed{...}``. Span verification then cannot find it in the response and
    a correct extraction is discarded (62 TL + 38 AD rows on disk shipped that
    way). A backspace or formfeed can never legitimately occur in a span quoted
    from a measurement response, so the inversion is safe to apply always.

    ``\\t``/``\\r``/``\\n`` are left alone: a span quoting a multi-line response
    NEEDS ``\\n`` to mean newline, and there is no way to tell an intended tab
    from ``\\times``. That residue is rare (one row on disk) and is a matter for
    span-matching, not decoding.

    Args:
        blob: Candidate JSON source text.

    Returns:
        str: The blob with every bare ``\\b``/``\\f`` escape doubled.
    """
    out = []
    i = 0
    n = len(blob)
    while i < n:
        ch = blob[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = blob[i + 1] if i + 1 < n else ""
        if nxt == "\\":
            # An escaped backslash: the judge already wrote correct JSON here.
            # Consuming both is what makes this pass idempotent.
            out.append("\\\\")
            i += 2
        elif nxt in _LATEX_CONTROL_ESCAPES:
            out.append("\\\\")
            out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def repair_json_escapes(blob):
    """Escape stray backslashes so a LaTeX-bearing span becomes legal JSON.

    The judge is told to quote spans VERBATIM, and medical measurement responses
    are full of LaTeX. A faithfully copied span therefore carries sequences like
    ``\\[`` and ``\\]`` which are not JSON escapes, so the whole object fails to
    decode over a quote that was doing exactly what we asked. (Macros that ARE
    legal escapes -- ``\\boxed``, ``\\frac`` -- decode without error and are
    handled earlier, by ``preescape_latex_escapes``.)

    Only ever applied AFTER a plain parse has already failed, so it cannot change
    the reading of a blob that decodes on its own.

    Args:
        blob: Candidate JSON source text.

    Returns:
        tuple: ``(repaired, changed)``.
    """
    out = []
    i = 0
    n = len(blob)
    changed = False
    while i < n:
        ch = blob[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = blob[i + 1] if i + 1 < n else ""
        if nxt in _JSON_SIMPLE_ESCAPES:
            # Includes "\\\\": consuming both characters is what stops an escaped
            # backslash from being re-escaped on every pass.
            out.append(ch)
            out.append(nxt)
            i += 2
        elif nxt == "u" and _is_hex4(blob, i + 2):
            out.append(blob[i : i + 6])
            i += 6
        else:
            # Invalid escape, or a lone trailing backslash. Double it and let the
            # following character be re-read as an ordinary literal.
            out.append("\\\\")
            changed = True
            i += 1
    return "".join(out), changed


def _is_hex4(s, j):
    """Return whether ``s[j:j+4]`` is four hex digits (a complete ``\\uXXXX``)."""
    return len(s) >= j + 4 and all(c in _HEXDIGITS for c in s[j : j + 4])


def _load_candidate(blob):
    """Parse one candidate, retrying once with escapes repaired.

    ``preescape_latex_escapes`` runs FIRST, unconditionally: the escapes it fixes
    parse successfully on their own, so a repair keyed on parse failure would
    never see them and the corruption would ship silently.

    Returns:
        object | None: The decoded value, or ``None`` if it never parsed.
    """
    blob = preescape_latex_escapes(blob)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    repaired, changed = repair_json_escapes(blob)
    if not changed:
        return None
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def parse_judge_json(raw_text):
    """Parse a judge response into a dict, tolerantly.

    Selects among candidates rather than committing to the first one. The
    discriminator is ``final_answer``: it is the one key the contract guarantees,
    so a candidate carrying it is the judge's object and a candidate without it is
    prose that happened to balance.

    Cannot regress a response that already parses. The old first-balanced-group is
    still candidate #1, and if it decodes to a dict with ``final_answer`` it is
    returned unchanged; the fallback returns the first parsed dict, so even a
    ``final_answer``-less object still wins over nothing.

    Args:
        raw_text: The model's generated text (harmony or plain).

    Returns:
        tuple: ``(obj, reason)``. ``obj`` is the parsed dict, or ``None`` on
        failure with ``reason`` naming the failure mode.
    """
    final = split_final_channel(raw_text)
    if not final:
        return None, "empty_output"

    fallback = None
    seen_candidate = False
    for blob in iter_json_object_candidates(final):
        seen_candidate = True
        obj = _load_candidate(blob)
        # Every candidate is a "{...}" substring, so a successful parse can only
        # be a dict -- no non-dict branch is possible here.
        if obj is None:
            continue
        if "final_answer" in obj:
            return obj, "ok"
        if fallback is None:
            fallback = obj

    if fallback is not None:
        # A candidate without "final_answer" can NEVER validate -- validate_judge_obj
        # returns missing_final_answer for it 100% of the time -- so returning it
        # raw was a guaranteed rejection dressed as reason "ok". Measured on the
        # live judge-out files: this branch fires 221 times and validation discards
        # all 221; in 203 of them the object IS the judge's own final_answer BODY
        # ({status, span, values}) sitting right after the "final_answer": key,
        # left stranded because the OUTER object was truncated around it.
        #
        # Re-wrap that body into the object the judge was writing. This invents no
        # content: the status, span and values are the judge's own, and the span
        # still has to survive verify_span exactly as it would have. A body that
        # does not look like a final_answer is returned unchanged, so nothing else
        # changes shape.
        if _looks_like_final_answer_body(fallback):
            return {"final_answer": fallback}, "ok"
        return fallback, "ok"
    if not seen_candidate:
        return None, "no_json_object"
    return None, "json_decode_error"


def _looks_like_final_answer_body(obj):
    """True if ``obj`` is a stranded ``final_answer`` value rather than some other object.

    The discriminator is the field set, not position: a final_answer body carries
    ``status`` plus at least one of ``span``/``values``, and carries no ``index``
    (which is what distinguishes it from a *step* entry, the one other object in
    the schema with the same three keys).
    """
    if not isinstance(obj, dict) or "index" in obj:
        return False
    return "status" in obj and ("span" in obj or "values" in obj)


def validate_judge_obj(obj, expects_steps):
    """Check a parsed judge object against the contract Stage 2 relies on.

    Deliberately shallow: it checks SHAPE, not plausibility. Value correctness is
    established by span verification in ``judge_verify``, which cannot be fooled by
    a well-formed object.

    The v1 status ``"absent"`` is rejected rather than remapped. That rejection is
    the mechanism that makes a stale v1 ``judge-out_*.jsonl`` unusable: every row in
    it fails here, so no v1 answer can be silently reinterpreted under the v2
    three-way contract.

    Args:
        obj: Parsed judge object.
        expects_steps: Whether this task should carry a ``steps`` array.

    Returns:
        tuple: ``(ok, reason)``.
    """
    fa = obj.get("final_answer")
    if not isinstance(fa, dict):
        return False, "missing_final_answer"
    status = fa.get("status")
    if status not in FINAL_ANSWER_STATUSES:
        return False, "bad_status"
    if status == "present":
        if not isinstance(fa.get("span"), str) or not fa["span"].strip():
            return False, "missing_span"
        values = fa.get("values")
        if not isinstance(values, list) or not values:
            return False, "missing_values"
        # Kept distinct from "missing_values": the two point at different prompt
        # problems, and the Stage 1 abort message reports the top reasons verbatim.
        if not _is_number_list(values):
            return False, "non_numeric_values"
    else:
        # A non-present status carries no answer by definition. NON-ZERO values
        # here would mean the judge disagreed with itself, and silently dropping
        # them would hide that -- so that stays a validation failure.
        #
        # An ALL-ZERO array is a different animal: it is this prompt's own
        # skeleton being echoed back. ``_output_skeleton`` used to render
        # ``values`` as ``[0.0] * arity``, the only placeholder in it that was
        # also a legal value, and Detection copied it verbatim beside a clean
        # ``"no_conclusion"``. Rejecting that discarded 3,407 correct verdicts and
        # mis-attributed them to judge unavailability. The judge is not
        # contradicting itself here -- it is quoting us.
        #
        # The skeleton stopped rendering ``[0.0]`` on 2026-08-08, but every
        # judge-out produced under the pre-change fingerprints (the ACCEPT_FP
        # list) contains these echoes, so this exemption must survive for
        # reparse_judge_out to keep working on historical raw.
        values = fa.get("values")
        if values and not (_is_number_list(values) and all(v == 0 for v in values)):
            return False, "values_with_non_present_status"
    if expects_steps:
        steps = obj.get("steps")
        if steps is not None and not isinstance(steps, list):
            return False, "bad_steps"
    return True, "ok"


def _is_number_list(values):
    """Return whether every element of ``values`` is a real JSON number.

    ``bool`` is excluded explicitly: ``isinstance(True, int)`` is ``True`` in
    Python, so a naive numeric check accepts ``[true, false]`` as a pair of
    measurements and lets it reach the span verifier as ``1`` and ``0``.
    """
    return all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )
