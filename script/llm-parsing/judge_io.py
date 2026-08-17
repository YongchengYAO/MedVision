"""LLM-as-Judge output parsing — shared I/O helpers.

Small utilities duplicated from ``medvision_bm`` so Stage 0/1/4 stay runnable
without installing the package (the ``clinical-decision-analysis/`` precedent).
Stage 2 is the deliberate exception: it imports the real ``cal_metrics`` because
scoring must not be re-implemented.

The two functions that MUST match the strict pipeline byte for byte are
``extract_response`` (parse_outputs.py:81) and ``iter_records`` (the doc_id sort
and limit rule at parse_outputs.py:218-221, 316-324). If either diverged, the
judge and strict columns would be computed over different rows.
"""

import glob
import hashlib
import json
import os
import re

# Byte-identical to medvision_bm.utils.parse_utils._NUM_RE (line 27). The judge's
# quoted spans are transcribed with THIS regex, so a number the strict parser
# would read one way cannot be read another way here.
NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def find_numbers(text):
    """Return every number in ``text`` as a list of strings, separators stripped.

    Args:
        text: Text to scan.

    Returns:
        list[str]: Matched numbers with thousands separators removed, in order.
    """
    if not text:
        return []
    return [m.replace(",", "") for m in NUM_RE.findall(text)]


def extract_last_k_nums_within_answer_tag(text, k):
    """Re-implementation of the strict parser, for baseline replay.

    Mirrors ``medvision_bm.utils.parse_utils.extract_last_k_nums_within_answer_tag``
    exactly. Duplicated rather than imported so Stage 0 runs without the package;
    ``test-1.py`` asserts the two agree on the real corpus.

    Args:
        text: Text expected to contain an ``<answer>`` block.
        k: Number of trailing numbers to return.

    Returns:
        str: Comma-separated last ``k`` numbers inside the first ``<answer>``
        block, or ``""`` if there is no such block or it holds fewer than ``k``.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not match:
        return ""
    numbers = find_numbers(match.group(1))
    if len(numbers) < k:
        return ""
    return ",".join(numbers[-k:])


def extract_response(data):
    """Extract the raw model response from a benchmark record.

    Byte-identical to ``parse_outputs._extract_response`` (line 81): some result
    files nest one extra list level inside ``resps[0][0]``.

    Args:
        data: One parsed/raw JSONL record.

    Returns:
        str: The response text.
    """
    try:
        if isinstance(data["resps"][0][0], list):
            return data["resps"][0][0][0]
        return data["resps"][0][0]
    except (IndexError, TypeError, KeyError):
        try:
            return data["resps"][0][0]
        except Exception:
            return ""


def load_roster(roster_yaml):
    """Load the evaluated-model roster from a visualization config YAML.

    Args:
        roster_yaml: Path to ``config-{TL,AD,detect}-CoT.yaml``.

    Returns:
        list[str]: Model directory names, in config order.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the config lacks a ``model_display_name`` mapping.
    """
    import yaml  # lazy: keeps the module importable without PyYAML

    with open(roster_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    if "model_display_name" not in cfg:
        raise KeyError(f"{roster_yaml} has no 'model_display_name' mapping")
    return list(cfg["model_display_name"].keys())


def list_sample_files(model_dir, parsed_dirname="parsed", excluded_stems=()):
    """List a model's per-sample JSONL files, excluding analysis outputs.

    Args:
        model_dir: Path to one model's result directory.
        parsed_dirname: Subdirectory to read (``"parsed"`` or an ``llm-parsed*``).
        excluded_stems: Substrings marking non-sample files (``_proc_acc`` etc.).

    Returns:
        list[str]: Sorted JSONL paths. Empty if the directory does not exist.
    """
    pattern = os.path.join(model_dir, parsed_dirname, "*.jsonl")
    files = sorted(glob.glob(pattern))
    return [f for f in files if not any(s in os.path.basename(f) for s in excluded_stems)]


def iter_records(jsonl_file, limit=None):
    """Yield records from a JSONL file in the benchmark's canonical order.

    Sorts by ``doc_id`` ascending and truncates to the first ``limit`` records --
    byte-for-byte the rule ``parse_outputs.py`` uses (sort at line 221, limit at
    lines 316-324). Any other ordering or sampling would make the judge's rows
    differ from the strict parser's, and the strict-vs-judge diff meaningless.

    Args:
        jsonl_file: Path to a JSONL file.
        limit: Maximum records to yield, or ``None`` for all.

    Yields:
        dict: One record.
    """
    records = []
    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda d: d.get("doc_id", 0))
    if limit is not None:
        records = records[:limit]
    for rec in records:
        yield rec


def content_hash(*parts):
    """Return a stable hash over the given strings.

    Used as the judge cache key so a rerun reuses prior answers, and so editing a
    prompt (which changes the fingerprint) correctly invalidates them.

    Args:
        *parts: Strings to hash in order.

    Returns:
        str: 32-character BLAKE2b hex digest.
    """
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def dataset_from_filename(path):
    """Extract the dataset name from a result filename.

    Mirrors the convention used by every summarizer, e.g.
    ``summarize_TL_task.py:362`` -- ``re.search(r"samples_([^_]+)_", ...)``.

    Args:
        path: A result JSONL path.

    Returns:
        str: Dataset name, or ``"unknown"`` if the pattern does not match.
    """
    m = re.search(r"samples_([^_]+)_", os.path.basename(path))
    return m.group(1) if m else "unknown"


def write_jsonl(path, rows):
    """Write rows to a JSONL file atomically via a temp file + rename.

    Args:
        path: Destination path.
        rows: Iterable of JSON-serializable dicts.

    Returns:
        int: Number of rows written.
    """
    tmp = path + ".tmp"
    n = 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    os.replace(tmp, path)
    return n
