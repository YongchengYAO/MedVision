"""test-5: record ordering and --limit must match the strict parser exactly.

``--limit N`` means "the first N records per JSONL file, sorted by doc_id
ascending" -- the rule ``parse_outputs.py`` uses (sort at line 221, limit break at
lines 316-324). If the judge selected a different subset, the strict-vs-judge
comparison would contrast different rows and mean nothing, while still looking
perfectly reasonable in a table.

Also checks the response-unwrapping path: some result files nest one extra list
level inside ``resps[0][0]``, and a copy that missed that case would read empty
strings for whole models.

Run from the repo root:
    python unit-test/llm-parsing/test-5.py
"""

import glob
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path("src").resolve()))
sys.path.insert(0, str(pathlib.Path("script/llm-parsing").resolve()))

from medvision_bm.benchmark.parse_outputs import _extract_response as real_extract_response

from judge_io import extract_response, iter_records

failures = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + extra) if extra and not cond else ''}")
    if not cond:
        failures.append(name)


print("test-5: ordering, limit, and response unwrapping")

# --- ordering and limit on a synthetic file -------------------------------
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "shuffled.jsonl")
    shuffled = [7, 3, 10, 1, 5, 9, 2, 8, 4, 6]
    with open(path, "w") as f:
        for d in shuffled:
            f.write(json.dumps({"doc_id": d, "resps": [[f"r{d}"]]}) + "\n")

    ids = [r["doc_id"] for r in iter_records(path)]
    check("sorts by doc_id ascending", ids == sorted(shuffled), str(ids))

    ids3 = [r["doc_id"] for r in iter_records(path, limit=3)]
    check("limit takes the FIRST n by doc_id, not file order", ids3 == [1, 2, 3], str(ids3))

    check("limit larger than file is safe",
          len(list(iter_records(path, limit=999))) == len(shuffled))
    check("limit=0 yields nothing", len(list(iter_records(path, limit=0))) == 0)

    # A blank line and a malformed line must not shift the selection.
    path2 = os.path.join(td, "dirty.jsonl")
    with open(path2, "w") as f:
        f.write(json.dumps({"doc_id": 2, "resps": [["b"]]}) + "\n")
        f.write("\n")
        f.write("{not json\n")
        f.write(json.dumps({"doc_id": 1, "resps": [["a"]]}) + "\n")
    ids = [r["doc_id"] for r in iter_records(path2)]
    check("skips blank and malformed lines", ids == [1, 2], str(ids))

# --- response unwrapping parity ------------------------------------------
CASES = [
    {"resps": [["plain text"]]},
    {"resps": [[["nested one level"]]]},
    {"resps": [[""]]},
]
mismatch = [c for c in CASES if extract_response(c) != real_extract_response(c)]
check("unwrapping matches parse_outputs._extract_response", not mismatch,
      str([(c, extract_response(c), real_extract_response(c)) for c in mismatch]))
check("nested list unwrapped", extract_response(CASES[1]) == "nested one level")
check("missing resps degrades to empty string", extract_response({}) == "")

# --- parity against the real corpus ---------------------------------------
real = sorted(glob.glob("Results/MedVision-TL-v2-CoT/*/parsed/*_samples_*.jsonl"))[:3]
if real:
    n = bad_order = bad_resp = 0
    for f in real:
        recs = list(iter_records(f, limit=100))
        ids = [r.get("doc_id") for r in recs]
        if ids != sorted(ids):
            bad_order += 1
        # the first 100 by doc_id must be exactly doc_ids <= the 100th smallest
        with open(f) as fh:
            all_ids = sorted(json.loads(l)["doc_id"] for l in fh if l.strip())
        if ids != all_ids[:100]:
            bad_order += 1
        for r in recs:
            n += 1
            if extract_response(r) != real_extract_response(r):
                bad_resp += 1
    check(f"ordering matches on {len(real)} real files", bad_order == 0)
    check(f"unwrapping matches on {n} real records (mismatches={bad_resp})", bad_resp == 0)
else:
    print("  SKIP  real-corpus parity (Results/ not present)")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("test-5: all checks passed")
