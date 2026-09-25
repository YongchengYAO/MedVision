print("=== SFT training: train-time cap on a PREPARED validation split ===")
print("Objective : Verify medvision_bm.sft.sft_utils.limit_prepared_validation_split")
print("            (1) per-task limits keep exactly `limit` rows of each capped task, in the")
print("                split's ORIGINAL order, and never upsample a task below its limit,")
print("            (2) the selection is seeded: two calls keep IDENTICAL rows,")
print("            (3) the total limit applies AFTER the per-task caps,")
print("            (4) non-positive / missing limits leave the split untouched,")
print("            (5) per-task limits on a split WITHOUT the task column warn and are")
print("                skipped, while the total limit still applies.")
print("Data      : synthetic in-memory rows; no MedVision data or GPU required.")

import contextlib
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

import datasets
from datasets import Dataset

datasets.disable_caching()

from medvision_bm.sft.sft_utils import limit_prepared_validation_split  # noqa: E402

TASK_COL = "__task_name"
COUNTS = {"AD": 30, "Detection": 500, "TL": 80}

# Interleave the tasks so per-task selection has to work on scattered rows.
labels = []
for i in range(max(COUNTS.values())):
    for task, n in COUNTS.items():
        if i < n:
            labels.append(task)
N_ROWS = len(labels)
VAL = Dataset.from_dict({"slice_idx": list(range(N_ROWS)), TASK_COL: labels})
assert len(VAL) == sum(COUNTS.values())


def counts(ds):
    out = {}
    for t in ds[TASK_COL]:
        out[t] = out.get(t, 0) + 1
    return out


# (1) per-task caps: AD 30->10, Detection 500->50, TL 80 stays 80 (limit 100 > count).
capped = limit_prepared_validation_split(
    VAL,
    per_task_limits={"AD": 10, "Detection": 50, "TL": 100},
    total_limit=-1,
    task_column=TASK_COL,
)
assert counts(capped) == {"AD": 10, "Detection": 50, "TL": 80}, counts(capped)
ids = capped["slice_idx"]
assert ids == sorted(ids), "rows must keep the split's original order"
assert set(ids) <= set(range(N_ROWS))
print("[PASS] (1) per-task caps: exact counts, original order, no upsampling")

# (2) determinism across calls (every rank must keep identical rows).
again = limit_prepared_validation_split(
    VAL,
    per_task_limits={"AD": 10, "Detection": 50, "TL": 100},
    total_limit=-1,
    task_column=TASK_COL,
)
assert again["slice_idx"] == ids, "seeded selection must be reproducible"
print("[PASS] (2) seeded: identical rows on a second call")

# (3) total limit after per-task caps: 140 -> 20.
total = limit_prepared_validation_split(
    VAL,
    per_task_limits={"AD": 10, "Detection": 50, "TL": 100},
    total_limit=20,
    task_column=TASK_COL,
)
assert len(total) == 20, len(total)
assert set(total["slice_idx"]) <= set(ids), "total cap must sub-select the per-task result"
print("[PASS] (3) total limit applies after the per-task caps")

# (4) no positive limit anywhere -> untouched.
same = limit_prepared_validation_split(
    VAL,
    per_task_limits={"AD": -1, "Detection": 0, "TL": None},
    total_limit=-1,
    task_column=TASK_COL,
)
assert same["slice_idx"] == VAL["slice_idx"]
print("[PASS] (4) non-positive / missing limits leave the split untouched")

# (5) missing task column: per-task caps warn + skip; total cap still applies.
no_col = VAL.remove_columns([TASK_COL])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    warned = limit_prepared_validation_split(
        no_col,
        per_task_limits={"AD": 10, "Detection": 50, "TL": 100},
        total_limit=-1,
        task_column=TASK_COL,
    )
assert len(warned) == len(no_col), "per-task caps must be skipped without the task column"
assert "[WARN]" in buf.getvalue() and TASK_COL in buf.getvalue(), buf.getvalue()
with contextlib.redirect_stdout(io.StringIO()):
    warned_total = limit_prepared_validation_split(
        no_col,
        per_task_limits={"AD": 10},
        total_limit=25,
        task_column=TASK_COL,
    )
assert len(warned_total) == 25, len(warned_total)
print("[PASS] (5) missing task column: warned, per-task skipped, total cap applied")

print("\nALL PASS")
