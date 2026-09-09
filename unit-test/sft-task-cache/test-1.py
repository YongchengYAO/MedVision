print("=== SFT dataset preparation: resumable per-task / per-chunk cache ===")
print("Objective : Verify medvision_bm.sft.task_cache")
print("            (1) chunked formatting returns the SAME rows in the ORIGINAL order")
print("                as a plain row-by-row reference (sort-by-volume is undone),")
print("            (2) an interrupted task resumes at the first MISSING chunk and")
print("                re-formats only the remaining rows,")
print("            (3) a finished task is published with a manifest, its chunk")
print("                scratch is removed, and a later run hits the cache,")
print("            (4) the cache key tracks the formatter's referenced globals,")
print("            (5) a directory without a manifest is never reused.")
print("Data      : synthetic in-memory rows; no MedVision data or GPU required.")

import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path("src").resolve()))

import datasets
from datasets import Dataset, DatasetDict

# Keep HF's own map cache out of the way so chunk reuse is the only reuse.
datasets.disable_caching()

from medvision_bm.sft.task_cache import (  # noqa: E402
    MANIFEST_NAME,
    _format_split_chunked,
    build_task_cache,
    load_cached_task,
    task_cache_inputs,
    task_cache_key,
)

N_ROWS = 60
CHUNK = 7
TASK_COL = "__task_name"

# Row order is deliberately NOT sorted by image_file: the formatter sorts by
# volume for cache locality, so the restore-order path must be exercised.
ROWS = {
    "image_file": [f"/vol/{(N_ROWS - 1 - i) % 7}.nii.gz" for i in range(N_ROWS)],
    "slice_dim": [i % 3 for i in range(N_ROWS)],
    "slice_idx": list(range(N_ROWS)),
    "dataset_name": ["synthetic"] * N_ROWS,  # dropped by keys_to_keep
}

CALLS = []          # rows the formatter actually processed
FAIL_AFTER = [None]  # crash trigger for the resume test
PREFIX = "m"        # global the formatter reads; test (4) mutates it


def format_row(example, model_name=None, model_hf=None, process_img=False,
               save_processed_img_to_disk=False, new_shape_hw=None):
    CALLS.append(example["slice_idx"])
    if FAIL_AFTER[0] is not None and len(CALLS) > FAIL_AFTER[0]:
        raise RuntimeError("simulated crash during formatting")
    example["messages"] = f"{PREFIX}:{example['slice_idx']}"
    example["labels"] = "L"
    return example


def make_split():
    return Dataset.from_dict(dict(ROWS))


if __name__ == "__main__":
    tmp_root = tempfile.mkdtemp(prefix="medvision-task-cache-")
    failures = []


    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  {name:<52} {status}{(' ' + detail) if detail else ''}")
        if not condition:
            failures.append(name)


    # --- (1) chunked formatting preserves rows and original order -----------------
    print("\n[1] chunked formatting == reference, original order restored")
    CALLS.clear()
    FAIL_AFTER[0] = None
    out = _format_split_chunked(
        make_split(),
        split_name="train",
        mapping_func=format_row,
        fn_kwargs={},
        keys_to_keep=["messages", "labels", "image_file", "slice_dim", "slice_idx"],
        task_label="TL",
        temperature_sampler_task_column=TASK_COL,
        num_workers=1,
        writer_batch_size=8,
        chunk_root=str(pathlib.Path(tmp_root) / "case1"),
        chunk_size=CHUNK,
    )
    expected = [f"m:{i}" for i in ROWS["slice_idx"]]
    check("row count", len(out) == N_ROWS, f"({len(out)})")
    check("messages match reference IN ORIGINAL ORDER", out["messages"] == expected)
    check("image_file order restored", out["image_file"] == ROWS["image_file"])
    check("every row formatted exactly once", sorted(CALLS) == sorted(ROWS["slice_idx"]))
    check("dropped column removed", "dataset_name" not in out.column_names)
    check("task column added", out[TASK_COL] == ["TL"] * N_ROWS)

    # --- (2) crash mid-task, then resume ------------------------------------------
    print("\n[2] interrupted task resumes at the first missing chunk")
    chunk_root = pathlib.Path(tmp_root) / "case2"
    CALLS.clear()
    FAIL_AFTER[0] = 3 * CHUNK + 3  # dies partway through chunk index 3
    crashed = False
    try:
        _format_split_chunked(
            make_split(), split_name="train", mapping_func=format_row, fn_kwargs={},
            keys_to_keep=["messages", "labels", "image_file", "slice_dim", "slice_idx"],
            task_label="TL", temperature_sampler_task_column=TASK_COL,
            num_workers=1, writer_batch_size=8,
            chunk_root=str(chunk_root), chunk_size=CHUNK,
        )
    except Exception:
        crashed = True
    saved = sorted(p.name for p in chunk_root.glob("chunk-*"))
    check("run 1 crashed as intended", crashed)
    check("only completed chunks were published", saved == [f"chunk-{i:06d}" for i in range(3)],
          f"({len(saved)} chunks)")

    CALLS.clear()
    FAIL_AFTER[0] = None
    resumed = _format_split_chunked(
        make_split(), split_name="train", mapping_func=format_row, fn_kwargs={},
        keys_to_keep=["messages", "labels", "image_file", "slice_dim", "slice_idx"],
        task_label="TL", temperature_sampler_task_column=TASK_COL,
        num_workers=1, writer_batch_size=8,
        chunk_root=str(chunk_root), chunk_size=CHUNK,
    )
    check("run 2 re-formatted only the missing rows",
          len(CALLS) == N_ROWS - 3 * CHUNK, f"({len(CALLS)} of {N_ROWS} rows)")
    check("resumed result identical to uninterrupted run", resumed["messages"] == expected)

    # --- (3) task-level publish + reuse -------------------------------------------
    print("\n[3] finished task is published, scratch removed, later run hits cache")
    data_dir = pathlib.Path(tmp_root) / "data"
    tasks_json = pathlib.Path(tmp_root) / "tasks.json"
    tasks_json.write_text(json.dumps({"SomeDataset_TumorLesionSize_Task01_Axial_Train": {}}))
    cache_dir = str(data_dir / "SFT-CoT_datasets" / "qwen25vl" / "_task_cache" / "TL__deadbeef")
    inputs = task_cache_inputs(
        task_label="TL", tasks_list_json_path=str(tasks_json), train_limit=-1,
        val_limit=10, tag_ds="TumorLesionSize", mapping_func=format_row,
        model_family_name="qwen25vl", base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
        process_img=False, save_processed_img_to_disk=False, new_shape_hw=[512, 512],
        temperature_sampler_task_column=TASK_COL,
    )
    raw = DatasetDict({"train": make_split(), "validation": make_split().select(range(10))})
    built = build_task_cache(
        raw, cache_dir=cache_dir, inputs=inputs, mapping_func=format_row,
        model_family_name="qwen25vl", base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
        num_workers_format_dataset=1, process_img=False,
        save_processed_img_to_disk=False, new_shape_hw=[512, 512],
        task_label="TL", temperature_sampler_task_column=TASK_COL, chunk_size=CHUNK,
    )
    check("both splits built", sorted(built.keys()) == ["train", "validation"])
    check("train rows correct", len(built["train"]) == N_ROWS)
    check("manifest published", pathlib.Path(cache_dir, MANIFEST_NAME).is_file())
    check("chunk scratch removed", not pathlib.Path(cache_dir + ".chunks").exists())
    manifest = json.loads(pathlib.Path(cache_dir, MANIFEST_NAME).read_text())
    check("manifest records rows", manifest["rows"]["train"] == N_ROWS)
    check("manifest records key", manifest["key"] == task_cache_key(inputs))

    CALLS.clear()
    hit = load_cached_task(cache_dir)
    check("cache hit returns the splits", hit is not None and len(hit["train"]) == N_ROWS)
    check("cache hit re-formatted nothing", len(CALLS) == 0)
    check("cached rows identical", hit["train"]["messages"] == expected)

    # --- (4) key tracks the formatter's referenced globals -------------------------
    print("\n[4] cache key invalidation")
    key_base = task_cache_key(inputs)
    key_same = task_cache_key(
        task_cache_inputs(
            task_label="TL", tasks_list_json_path=str(tasks_json), train_limit=-1,
            val_limit=10, tag_ds="TumorLesionSize", mapping_func=format_row,
            model_family_name="qwen25vl", base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
            process_img=False, save_processed_img_to_disk=False, new_shape_hw=[512, 512],
            temperature_sampler_task_column=TASK_COL,
        )
    )
    check("identical inputs -> identical key", key_base == key_same)

    key_limit = task_cache_key({**inputs, "train_limit": 110000})
    check("changed sample limit -> new key", key_limit != key_base)

    tasks_json.write_text(json.dumps({"OtherDataset_TumorLesionSize_Task01_Axial_Train": {}}))
    key_json = task_cache_key(
        task_cache_inputs(
            task_label="TL", tasks_list_json_path=str(tasks_json), train_limit=-1,
            val_limit=10, tag_ds="TumorLesionSize", mapping_func=format_row,
            model_family_name="qwen25vl", base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
            process_img=False, save_processed_img_to_disk=False, new_shape_hw=[512, 512],
            temperature_sampler_task_column=TASK_COL,
        )
    )
    check("edited task-list JSON -> new key", key_json != key_base)

    globals()["PREFIX"] = "CHANGED"  # a global the formatter reads
    key_global = task_cache_key(
        task_cache_inputs(
            task_label="TL", tasks_list_json_path=str(tasks_json), train_limit=-1,
            val_limit=10, tag_ds="TumorLesionSize", mapping_func=format_row,
            model_family_name="qwen25vl", base_model_hf="Qwen/Qwen2.5-VL-7B-Instruct",
            process_img=False, save_processed_img_to_disk=False, new_shape_hw=[512, 512],
            temperature_sampler_task_column=TASK_COL,
        )
    )
    check("edited formatter global (prompt) -> new key", key_global != key_json)
    globals()["PREFIX"] = "m"

    # --- (5) a manifest-less directory is never reused -----------------------------
    print("\n[5] incomplete cache directory is not reused")
    pathlib.Path(cache_dir, MANIFEST_NAME).unlink()
    check("no manifest -> cache miss", load_cached_task(cache_dir) is None)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("OK")
