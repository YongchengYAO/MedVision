#!/usr/bin/env python
"""Resolve MedVision SFT sample-limit flags exactly as the trainers do.

Purpose
-------
Every ``python -m medvision_bm.sft.train__*`` entry point turns the ten
sample-limit flags into seven effective limits via
``medvision_bm.sft.sft_utils.parse_sample_limits``. This script accepts the
same flags, calls the REAL function, and prints the result together with the
downstream semantics (validation carve-out, per-task cap, global bootstrap).
It exits non-zero on the combinations the trainer rejects (any limit == 0).

Prerequisites
-------------
``medvision_bm`` importable (``pip install medvision-bm`` or an editable
install of the repository). Importing ``sft_utils`` pulls torch, accelerate,
datasets, nibabel, scipy and Pillow; no GPU, no network, no data needed.

Examples
--------
    python check_sample_limits.py \
        --tasks_list_json_path_AD tasks_AD.json \
        --tasks_list_json_path_detect tasks_detect.json \
        --tasks_list_json_path_TL tasks_TL.json \
        --train_sample_limit 121000 --val_sample_limit 200 \
        --train_sample_limit_task_AD 5500 --val_sample_limit_task_AD 45 \
        --train_sample_limit_task_Detection 110000 --val_sample_limit_task_Detection 105 \
        --train_sample_limit_task_TL 5500 --val_sample_limit_task_TL 50

    # Rejected (exit 2): a limit of 0 is ambiguous
    python check_sample_limits.py --tasks_list_json_path_TL t.json --train_sample_limit_task_TL 0

    # Simulate outcomes against known pool sizes (rows in the *_Train configs)
    python check_sample_limits.py --tasks_list_json_path_TL t.json \
        --train_sample_limit_task_TL 20000 --pool_TL 5500

The task-list paths are only used for presence (``None`` => task not used);
they are not opened, so placeholders are fine.
"""

from __future__ import annotations

import argparse
import sys

TASKS = (
    # (label, tasks_list kwarg, train-limit kwarg, val-limit kwarg, pool kwarg)
    ("AD", "tasks_list_json_path_AD", "train_sample_limit_task_AD", "val_sample_limit_task_AD", "pool_AD"),
    ("Detection", "tasks_list_json_path_detect", "train_sample_limit_task_Detection", "val_sample_limit_task_Detection", "pool_Detection"),
    ("TL", "tasks_list_json_path_TL", "train_sample_limit_task_TL", "val_sample_limit_task_TL", "pool_TL"),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Defaults mirror the trainer argparse: per-task train limits -1 (full), "
            "per-task val fallback 100, total val 100, total train -1. Unset flags "
            "are passed as None so the trainer fallbacks apply."
        ),
    )
    for _, path_key, _, _, _ in TASKS:
        p.add_argument(f"--{path_key}", type=str, default=None,
                       help="Task list JSON for this task (presence only; not opened).")
    p.add_argument("--train_sample_limit_per_task", type=int, default=None,
                   help="Per-task train cap used when the task-specific limit is unset/<=0 (trainer default -1).")
    p.add_argument("--val_sample_limit_per_task", type=int, default=None,
                   help="Per-task validation target used when the task-specific limit is unset/<=0 (trainer default 100).")
    for label, _, tr_key, va_key, _ in TASKS:
        p.add_argument(f"--{tr_key}", type=int, default=None, help=f"Train cap for the {label} task (-1 = full pool).")
        p.add_argument(f"--{va_key}", type=int, default=None, help=f"Validation target for the {label} task.")
    p.add_argument("--train_sample_limit", type=int, default=None,
                   help="GLOBAL train cap applied after concatenating tasks (trainer default -1). "
                        "Larger than the pool => sampling WITH replacement.")
    p.add_argument("--val_sample_limit", type=int, default=None,
                   help="GLOBAL validation cap applied after concatenation (trainer default 100).")
    for label, _, _, _, pool_key in TASKS:
        p.add_argument(f"--{pool_key}", type=int, default=None,
                       help=f"Optional: number of rows in the {label} *_Train pool, to simulate the outcome.")
    p.add_argument("--json", action="store_true", help="Print the resolved limits as JSON only.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from medvision_bm.sft.sft_utils import parse_sample_limits
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[error] cannot import medvision_bm.sft.sft_utils: {exc}\n"
              "        Install medvision_bm (pip install medvision-bm) with its SFT extras "
              "(torch, accelerate, datasets, nibabel, scipy, Pillow).", file=sys.stderr)
        return 3

    kwargs = {k: v for k, v in vars(args).items() if not k.startswith("pool_") and k != "json"}

    try:
        (train_AD, val_AD, train_det, val_det, train_TL, val_TL, train_total) = parse_sample_limits(**kwargs)
    except ValueError as exc:
        print(f"[rejected] {str(exc).strip()}", file=sys.stderr)
        return 2

    # The trainer also rejects an explicit --val_sample_limit 0 inside parse_sample_limits
    # (already covered above). Mirror the remaining assertion of load_split_limit_dataset:
    # a used task must have a validation target > 0.
    resolved = {
        "AD": (train_AD, val_AD),
        "Detection": (train_det, val_det),
        "TL": (train_TL, val_TL),
    }
    if args.json:
        import json
        print(json.dumps({
            "train_limit_AD": train_AD, "val_limit_AD": val_AD,
            "train_limit_detect": train_det, "val_limit_detect": val_det,
            "train_limit_TL": train_TL, "val_limit_TL": val_TL,
            "train_limit_total": train_total,
            "val_limit_total": args.val_sample_limit if args.val_sample_limit is not None else 100,
        }, indent=2))
        return 0

    def fmt(v):
        return "full" if v < 0 else str(v)

    print("Resolved by medvision_bm.sft.sft_utils.parse_sample_limits:")
    print(f"  {'task':<10} {'used':<5} {'train limit':<14} validation target")
    for label, path_key, _, _, pool_key in TASKS:
        used = getattr(args, path_key) is not None
        tr, va = resolved[label]
        tr_txt = "n/a (task not used)" if not used else ("full pool (-1)" if tr < 0 else str(tr))
        va_txt = "n/a" if not used else str(va)
        print(f"  {label:<10} {'yes' if used else 'no':<5} {tr_txt:<14} {va_txt}")
    print(f"  global train_sample_limit : {'none (-1)' if train_total < 0 else train_total}")
    val_total = args.val_sample_limit if args.val_sample_limit is not None else 100
    print(f"  global val_sample_limit   : {val_total}  (trainer default 100 when the flag is omitted; "
          "the repository launchers pass 200)")

    print("\nDefault prepared-dataset directory (for every UNSET limit the entry points substitute the TRUE "
          "split size measured after load+split, before formatting; caps are printed as given; then "
          "'__resized-wh-<W>x<H>' or '__original' is appended; phase A prints the final path):")
    print("  <data_dir>/SFT-CoT_datasets/<model_family_name>/"
          f"ds__AD{fmt(train_AD)}_D{fmt(train_det)}_TL{fmt(train_TL)}_all{fmt(train_total)}...")
    print("  (non-CoT entry point uses 'SFT_datasets'; the tool-use entry point appends '-tooluse' "
          "to the resize suffix.)")

    print("\nSemantics (verified in load_split_limit_dataset and the train__*.py entry points):")
    print("  * Per task: validation rows are carved out FIRST, grouped by source volume (image_file)"
          " and stratified by dataset_name, seed = medvision_bm.utils.configs.SEED; the greedy fill"
          " stops at or just above the target, so the split can slightly exceed it.")
    print("  * Per task: the train cap applies only when 0 < cap < remaining rows (shuffle + select)."
          " A cap larger than the pool is a no-op: the FULL remaining pool is used, never upsampled.")
    print("  * Global: train_sample_limit > concatenated size => np.random.choice(replace=True)"
          " (bootstrap duplicates, seed SEED); smaller => shuffle + select; -1 => shuffle only."
          " val_sample_limit behaves the same on the concatenated validation split.")

    any_pool = any(getattr(args, pk) is not None for *_, pk in TASKS)
    if any_pool:
        print("\nSimulation against the pools you supplied:")
        concat_train = 0
        for label, path_key, _, _, pool_key in TASKS:
            pool = getattr(args, pool_key)
            if pool is None or getattr(args, path_key) is None:
                continue
            tr, va = resolved[label]
            remaining = max(0, pool - va)
            if tr > 0 and tr < remaining:
                n_train, note = tr, "capped (shuffle + select)"
            else:
                n_train, note = remaining, "full remaining pool (cap >= pool or unset; no upsampling here)"
            concat_train += n_train
            print(f"  {label:<10} pool={pool:<8} val~{va:<5} train={n_train:<8} {note}")
        if train_total > 0:
            if train_total > concat_train:
                print(f"  global    : {train_total} > {concat_train} concatenated rows => bootstrap WITH replacement "
                      f"({train_total - concat_train} duplicate draws on average)")
            else:
                print(f"  global    : {train_total} <= {concat_train} => shuffle + select({train_total})")
        else:
            print(f"  global    : no cap => {concat_train} rows, shuffled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
