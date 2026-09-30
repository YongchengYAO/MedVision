"""
Report the case-level train/test swap of the T/L biometry plans between dataset
v1.0.0 and v1.4.0.

v1.4.0 keeps the per-split case counts but changes train/test membership for
collections whose earlier splits had been force-aligned to v1.0.0. A model
trained on the v1.4.0 training split has therefore seen some v1.0.0 test cases,
which inflates its score on the old test set.

Splits are case-level (benchmark_plan_biometry_v*.json.gz: tasks[].train_cases /
test_cases). Cases are pooled over each dataset's tasks, as in the release note:
  - test->train : v1.0.0 test cases now in the v1.4.0 train split
  - train->test : v1.0.0 train cases now in the v1.4.0 test split
  - changed     : both directions over all shared cases; this is the "case
                  assignment change" percentage of docs/dataset-release/release-v1.4.0.md

Usage:
    python count_case_split_swap.py \
        [--data_dir /path/to/MedVision/Data/Datasets] \
        [--old_version 1.0.0] [--new_version 1.4.0] [--datasets KiPA22 ...]
"""

import argparse
import glob
import gzip
import json
import os


def _load_tasks(data_dir, dataset, version):
    path = os.path.join(data_dir, dataset, f"benchmark_plan_biometry_v{version}.json.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as f:
        return {t["task_ID"]: t for t in json.load(f)["tasks"]}


def case_splits(tasks):
    """Map image_file -> "train"/"test", pooled over a dataset's tasks."""
    splits = {}
    for task in tasks.values():
        for split in ("train", "test"):
            for case in task[f"{split}_cases"]:
                img = case["image_file"]
                if splits.setdefault(img, split) != split:
                    raise ValueError(f"{img} is in both train and test across tasks")
    return splits


def count_case_swap(old_splits, new_splits):
    shared = old_splits.keys() & new_splits.keys()
    return {
        "shared": len(shared),
        "old_test": sum(old_splits[c] == "test" for c in shared),
        "old_train": sum(old_splits[c] == "train" for c in shared),
        "test_to_train": sum(
            old_splits[c] == "test" and new_splits[c] == "train" for c in shared
        ),
        "train_to_test": sum(
            old_splits[c] == "train" and new_splits[c] == "test" for c in shared
        ),
        "old_only": len(old_splits.keys() - new_splits.keys()),
        "new_only": len(new_splits.keys() - old_splits.keys()),
    }


def _frac(n, d):
    return f"{n:>5}/{d:<5} ({n / d if d else 0.0:.3f})"


def format_case_row(name, row):
    changed = row["test_to_train"] + row["train_to_test"]
    return (
        f"{name:<18} | {row['shared']:>6} | {_frac(row['test_to_train'], row['old_test'])} | "
        f"{_frac(row['train_to_test'], row['old_train'])} | "
        f"{_frac(changed, row['shared'])} | {row['old_only']:>8} | {row['new_only']:>8}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    parser.add_argument(
        "--data_dir", default=os.path.join(repo_dir, "Data", "Datasets")
    )
    parser.add_argument("--old_version", default="1.0.0")
    parser.add_argument("--new_version", default="1.4.0")
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    datasets = args.datasets or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(
            os.path.join(
                args.data_dir, "*", f"benchmark_plan_biometry_v{args.new_version}.json.gz"
            )
        )
    )

    case_rows = []
    skipped = []
    for ds in datasets:
        old_tasks = _load_tasks(args.data_dir, ds, args.old_version)
        new_tasks = _load_tasks(args.data_dir, ds, args.new_version)
        if old_tasks is None or new_tasks is None:
            skipped.append(ds)
            continue
        case_rows.append(
            (ds, count_case_swap(case_splits(old_tasks), case_splits(new_tasks)))
        )

    if case_rows:
        case_header = (
            f"{'Dataset':<18} | {'Shared':>6} | {'Test -> Train':<19} | "
            f"{'Train -> Test':<19} | {'Changed (both)':<19} | {'Old only':>8} | "
            f"{'New only':>8}"
        )
        print(
            f"Case-level train/test swap, v{args.old_version} -> v{args.new_version}\n"
            f"(cases pooled over each dataset's tasks; Test -> Train over v{args.old_version} "
            f"test cases, Train -> Test over v{args.old_version} train cases, Changed over "
            f"all shared cases; Old/New only = case in one version's plan only)\n"
        )
        print(case_header)
        print("-" * len(case_header))
        for ds, row in case_rows:
            print(format_case_row(ds, row))
        print("-" * len(case_header))
        total = {k: sum(r[k] for _, r in case_rows) for k in case_rows[0][1]}
        print(format_case_row("All", total))

    if skipped:
        print(f"\nSkipped (no v{args.old_version} or v{args.new_version} plan): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
