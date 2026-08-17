"""test-1: the replayed strict parser must be byte-identical to the real one.

``judge_io`` duplicates ``_NUM_RE`` and ``extract_last_k_nums_within_answer_tag``
so Stage 0 runs without installing medvision_bm. Duplication is a drift risk:
if the copy diverged, the judge and strict columns would be computed over
different rows and every comparison in the analysis would be meaningless.

Run from the repo root:
    python unit-test/llm-parsing/test-1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from medvision_bm.utils.parse_utils import _NUM_RE as REAL_NUM_RE
from medvision_bm.utils.parse_utils import (
    extract_last_k_nums_within_answer_tag as real_extract,
)

from judge_io import NUM_RE as COPY_NUM_RE
from judge_io import extract_last_k_nums_within_answer_tag as copy_extract

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


print("test-1: strict-parser replay parity")

# 1. The regex source must be literally the same pattern.
check("_NUM_RE pattern identical", REAL_NUM_RE.pattern == COPY_NUM_RE.pattern)

# 2. Behavioural parity on the shapes that actually occur in the corpus.
CASES = [
    ("<answer>20.0, 18.0</answer>", 2),
    ("<answer>\n  0.45, 0.48, 0.72, 0.78\n</answer>", 4),
    ("no answer tag at all, 12.3", 1),
    ("<answer></answer>", 1),
    ("<answer>only-one 5</answer>", 2),
    # thousands separators are stripped by both
    ("<answer>1,234.5</answer>", 1),
    # multiple answer blocks: both take the FIRST block (non-greedy match)
    ("<answer>1,2</answer> then <answer>3,4</answer>", 2),
    # negative and exponent forms
    ("<answer>-0.59, -0.37</answer>", 2),
    ("<answer>1e-3, 2E+4</answer>", 2),
    # the real Qwen2.5-VL-32B failure shape: complete answer, no tag
    (r"\boxed{407.02, 325.62}", 2),
    # tag numerals leaking in (the MedDr false-success shape)
    ("<answer><step-1-answer>x</step-1-answer><step-2-answer>y</step-2-answer></answer>", 2),
    ("", 1),
]
mismatches = [
    (t, k, real_extract(t, k), copy_extract(t, k))
    for t, k in CASES
    if real_extract(t, k) != copy_extract(t, k)
]
check(f"behavioural parity on {len(CASES)} synthetic cases", not mismatches)
for m in mismatches:
    print(f"      text={m[0]!r} k={m[1]} real={m[2]!r} copy={m[3]!r}")

# 3. Parity on real responses, if the corpus is present.
import glob  # noqa: E402
import json  # noqa: E402

sample_files = sorted(
    glob.glob("Results/MedVision-TL-v2-CoT/*/parsed/*_samples_*.jsonl")
)[:4]
if sample_files:
    from judge_io import extract_response

    n = bad = 0
    for f in sample_files:
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i >= 300:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resp = extract_response(rec) or ""
                n += 1
                if real_extract(resp, 2) != copy_extract(resp, 2):
                    bad += 1
    check(f"parity on {n} real responses (mismatches={bad})", bad == 0)
else:
    print("  SKIP  real-corpus parity (Results/ not present)")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-1: all checks passed")
