#!/usr/bin/env python3
"""
Deduplicate JSONL files by doc_id, keeping only the first occurrence.

Usage:
    python -m medvision_bm.benchmark.remove_duplicate_samples \
        --dir <Results/MedVision-detect/bak-w-duplicate-samples> \
        --out_dir <Results/MedVision-detect/deduped>

Behavior:
    - Iterates each subfolder under --dir
    - Copies all .json files as-is to the corresponding subfolder in --out_dir
    - Deduplicates each .jsonl file by "doc_id" (keeps first occurrence)
    - Writes deduplicated .jsonl files to --out_dir preserving subfolder structure
"""

import argparse
import json
import shutil
from pathlib import Path


def dedup_jsonl(input_path: Path, output_path: Path) -> tuple[int, int]:
    """
    Read a JSONL file and remove duplicate lines sharing the same doc_id.

    For each line, parse the JSON object and extract its "doc_id" field.
    Only the first line with a given doc_id is kept; subsequent duplicates
    are discarded. The deduplicated lines are written to output_path.

    Args:
        input_path:  Path to the source .jsonl file.
        output_path: Path to write the deduplicated .jsonl file.

    Returns:
        A tuple of (original_line_count, deduplicated_line_count).
    """
    seen_doc_ids = set()
    kept_lines = []
    original_count = 0

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            original_count += 1
            record = json.loads(line)
            doc_id = record.get("doc_id")
            # Keep only the first appearance of each doc_id
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                kept_lines.append(line)

    # Write deduplicated lines
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    return original_count, len(kept_lines)


def process_subfolder(
    subfolder: Path, work_dir: Path, out_dir: Path
) -> tuple[int, int, int]:
    """
    Process a single subfolder: copy .json files and deduplicate .jsonl files.

    Args:
        subfolder: Path to the source subfolder.
        work_dir:  Root working directory (parent of subfolder).
        out_dir:   Root output directory.

    Returns:
        A tuple of (jsonl_file_count, total_lines_removed, json_files_copied).
    """
    out_subfolder = out_dir / subfolder.name
    out_subfolder.mkdir(parents=True, exist_ok=True)

    # --- Copy .json files as-is ---
    json_files = sorted(subfolder.glob("*.json"))
    for json_file in json_files:
        shutil.copy2(json_file, out_subfolder / json_file.name)

    if json_files:
        print(f"  Copied {len(json_files)} .json file(s)")

    # --- Deduplicate .jsonl files ---
    file_count = 0
    lines_removed = 0

    for jsonl_file in sorted(subfolder.glob("*.jsonl")):
        rel_path = jsonl_file.relative_to(work_dir)
        output_path = out_dir / rel_path

        original, deduped = dedup_jsonl(jsonl_file, output_path)
        removed = original - deduped
        file_count += 1
        lines_removed += removed

        status = f"-{removed} dup" if removed > 0 else "clean"
        print(f"  {jsonl_file.name}  ({original} -> {deduped}, {status})")

    return file_count, lines_removed, len(json_files)


def main():
    """
    Entry point: parse CLI arguments and process all subfolders.
    """
    parser = argparse.ArgumentParser(
        description="Deduplicate JSONL files by doc_id (keep first occurrence)."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Working folder with subfolders containing JSONL files.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory (preserves subfolder structure).",
    )
    args = parser.parse_args()

    work_dir = Path(args.dir)
    out_dir = Path(args.out_dir)

    if not work_dir.is_dir():
        print(f"Error: '{work_dir}' is not a directory.")
        return

    total_jsonl_files = 0
    total_json_copied = 0
    total_removed = 0

    subfolders = sorted(d for d in work_dir.iterdir() if d.is_dir())
    print(f"Source:  {work_dir}")
    print(f"Output:  {out_dir}")
    print(f"Subfolders: {len(subfolders)}\n")

    for subfolder in subfolders:
        print(f"[{subfolder.name}]")
        n_files, n_removed, n_json = process_subfolder(subfolder, work_dir, out_dir)
        total_jsonl_files += n_files
        total_json_copied += n_json
        total_removed += n_removed

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Subfolders processed : {len(subfolders)}")
    print(f"  JSONL files deduped  : {total_jsonl_files}")
    print(f"  JSON  files copied   : {total_json_copied}")
    print(f"  Duplicate lines removed : {total_removed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
