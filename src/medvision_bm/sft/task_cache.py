"""Resumable, chunked on-disk cache for the SFT dataset-preparation stage.

Dataset preparation formats every MedVision row into the chat ``messages`` layout
and (optionally) writes one PNG per row. On the full v1.4.0 training set that is
tens of millions of rows and takes days, yet the prepared ``DatasetDict`` used to
be written only once, after *all* tasks had been formatted: a crash in the last
task discarded every completed task.

This module makes that stage restartable at two granularities:

* **per task** -- a task's formatted splits are saved under a content-addressed
  directory as soon as that task finishes, so a later run reuses it and skips
  both the raw Arrow generation and the formatting for that task;
* **per chunk** -- each split is formatted in fixed-size chunks that are saved
  as they complete, so an interrupted task resumes at the first missing chunk
  instead of from row 0.

Cache identity is a hash of everything that can change the formatted rows: the
task-list JSON *content*, the per-task sample limits, the model identifiers, the
image-processing options, the dataset planner version, the seed, and a
:class:`datasets.fingerprint.Hasher` fingerprint of the formatting code. That
fingerprint covers the formatter, the helpers it reaches, and the prompt
modules, so editing a prompt template or a ``_doc_to_*`` helper yields a new key
while unrelated edits elsewhere do not.
"""

import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import time

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_from_disk

from medvision_bm.utils.configs import SEED

# Written inside a task cache directory as the very last step before the
# directory is published under its final name, so its presence means "this cache
# is complete". A directory without it is never reused.
MANIFEST_NAME = "medvision_task_cache.json"

# Rows per chunk. Small chunks bound the work lost to a crash; large chunks
# amortise the per-chunk worker-pool startup. 20k rows is ~1 minute of warm
# formatting (PNGs already on disk) and a few hours of cold formatting.
DEFAULT_CHUNK_SIZE = 20000


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Prompt templates live here and are imported *inside* the formatting helpers,
# so they never appear as module globals and must be fingerprinted directly.
_PROMPT_MODULES = (
    "medvision_bm.sft.sft_prompts",
    "medvision_bm.sft.sft_prompts_tooluse",
)

_IMMUTABLE_TYPES = (str, bytes, int, float, bool, type(None))


def _is_immutable(value, depth=0):
    """True for values whose repr is a stable stand-in for their content.

    Mutable containers are excluded on purpose: ``sft_utils`` keeps a
    module-level volume cache that holds a decoded 3D image once formatting has
    run, and folding that into the fingerprint would make the key depend on
    which volume happened to be cached.
    """
    if isinstance(value, _IMMUTABLE_TYPES):
        return True
    if depth < 3 and isinstance(value, (tuple, frozenset)):
        return all(_is_immutable(item, depth + 1) for item in value)
    return False


def _formatter_fingerprint(mapping_func, max_depth=4):
    """Digest of the formatting code that decides a row's content.

    Covers the formatter's own source, the source of every ``medvision_bm``
    function it reaches through module globals (the ``_doc_to_*`` and
    ``img_proccessor_*`` helpers), the immutable module-level constants those
    read, and the full source of the prompt modules -- prompt templates are
    imported *inside* the helpers, so they are not visible as globals and are
    picked up from the module instead.

    Not covered: modules imported inside a helper other than the prompt modules.
    Editing one of those does not invalidate the cache, so delete the cache
    directory (its path is printed on every build) after such a change.
    """
    parts = []
    seen = set()

    def visit(func, depth):
        ident = (getattr(func, "__module__", "?"), func.__qualname__)
        if ident in seen or depth > max_depth:
            return
        seen.add(ident)
        try:
            parts.append(f"# {ident}\n" + inspect.getsource(func))
        except (OSError, TypeError):
            parts.append(f"# {ident} <source unavailable>")
        func_globals = getattr(func, "__globals__", {})
        for name in sorted(getattr(func.__code__, "co_names", ())):
            if name not in func_globals:
                continue
            value = func_globals[name]
            if inspect.isfunction(value) and getattr(
                value, "__module__", ""
            ).startswith("medvision_bm"):
                visit(value, depth + 1)
            elif _is_immutable(value):
                parts.append(f"{name} = {value!r}")
            else:
                parts.append(f"{name} = <{type(value).__name__}>")

    visit(mapping_func, 0)

    # Imported explicitly rather than scanned out of sys.modules: the
    # fingerprint must not depend on which modules happen to be loaded yet.
    for name in _PROMPT_MODULES:
        try:
            parts.append(
                f"# {name}\n" + inspect.getsource(importlib.import_module(name))
            )
        except Exception:
            parts.append(f"# {name} <source unavailable>")

    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def task_cache_inputs(
    *,
    task_label,
    tasks_list_json_path,
    train_limit,
    val_limit,
    tag_ds,
    mapping_func,
    model_family_name,
    base_model_hf,
    process_img,
    save_processed_img_to_disk,
    new_shape_hw,
    temperature_sampler_task_column,
):
    """Collect every input that can change a task's formatted rows.

    Returned as a plain dict so it can be both hashed into the cache key and
    stored verbatim in the manifest for debugging a cache miss.
    """
    return {
        "task_label": task_label,
        "tasks_json_sha256": _sha256_file(tasks_list_json_path),
        "train_limit": train_limit,
        "val_limit": val_limit,
        "tag_ds": tag_ds,
        "formatter": _formatter_fingerprint(mapping_func),
        "model_family_name": model_family_name,
        "base_model_hf": base_model_hf,
        "process_img": bool(process_img),
        "save_processed_img_to_disk": bool(save_processed_img_to_disk),
        "new_shape_hw": list(new_shape_hw) if new_shape_hw is not None else None,
        "temperature_sampler_task_column": temperature_sampler_task_column,
        "planner_version": os.environ.get("MedVision_PLANNER_VERSION"),
        "seed": SEED,
    }


def task_cache_key(inputs):
    """Short, stable digest of :func:`task_cache_inputs`."""
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def task_cache_dir(data_dir, datasets_dirname, model_family_name, task_label, key):
    """Directory holding one task's formatted splits.

    Deliberately independent of the prepared-dataset directory *name*: that name
    is derived from *all* tasks' row counts, which are unknown until each task
    has been loaded, so a cache keyed on it could never be consulted first. It
    does sit under the same parent folder, so ``datasets_dirname`` has no default
    -- callers pass the same literal they build the prepared directory from
    (``SFT-CoT_datasets`` for the CoT entry points, ``SFT_datasets`` for the
    non-CoT one), which keeps a task's cache beside the dataset it feeds.
    """
    return os.path.join(
        data_dir,
        datasets_dirname,
        model_family_name,
        "_task_cache",
        f"{task_label}__{key}",
    )


def _load_if_complete(path):
    """Load a saved dataset, or return ``None`` when it is absent or unusable."""
    if not os.path.isdir(path):
        return None
    try:
        return load_from_disk(path)
    except Exception:
        return None


def load_cached_task(cache_dir):
    """Return a task's cached formatted splits, or ``None`` on a cache miss."""
    if not os.path.isfile(os.path.join(cache_dir, MANIFEST_NAME)):
        return None
    dataset = _load_if_complete(cache_dir)
    if dataset is None:
        return None
    sizes = ", ".join(f"{split}={len(ds)}" for split, ds in dataset.items())
    print(f"[Info] Reusing cached formatted task from {cache_dir} ({sizes})")
    return dataset


def _save_workers(dataset, cap):
    """Clamp save parallelism: ``save_to_disk`` raises when num_proc > row count.

    For a ``DatasetDict`` the cap is the smallest split, since ``num_proc``
    applies per split.
    """
    if isinstance(dataset, DatasetDict):
        rows = min((len(ds) for ds in dataset.values()), default=1)
    else:
        rows = len(dataset)
    return max(1, min(cap, rows))


def _atomic_save(dataset, dest, cap, extra_files=None):
    """Save to a temporary sibling and rename, so ``dest`` is never half-written."""
    tmp = f"{dest}.tmp-{os.getpid()}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # num_proc=1 still takes datasets' parallel path (a worker pool per call);
    # None is the single-process path, which is what a one-worker save wants.
    workers = _save_workers(dataset, cap)
    dataset.save_to_disk(tmp, num_proc=workers if workers > 1 else None)
    for name, payload in (extra_files or {}).items():
        with open(os.path.join(tmp, name), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.replace(tmp, dest)


def _format_split_chunked(
    dataset,
    *,
    split_name,
    mapping_func,
    fn_kwargs,
    keys_to_keep,
    task_label,
    temperature_sampler_task_column,
    num_workers,
    writer_batch_size,
    chunk_root,
    chunk_size,
):
    """Format one split in resumable chunks and return the reassembled split.

    Rows are sorted by source volume before chunking so each worker's slice hits
    the per-process volume cache in the image loader, then the original order is
    restored at the end: the seeded shuffles downstream permute positions, so the
    row order this returns must match what an uncached run produced.
    """
    n_rows = len(dataset)
    restore_order = "image_file" in dataset.column_names and n_rows > 1
    if restore_order:
        paths = dataset["image_file"]
        order = sorted(range(n_rows), key=paths.__getitem__)
        dataset = dataset.select(order)

    def _finish(part):
        part = part.remove_columns(
            [col for col in part.column_names if col not in keys_to_keep]
        )
        if temperature_sampler_task_column:
            part = part.add_column(
                temperature_sampler_task_column, [task_label] * len(part)
            )
        return part

    if n_rows == 0:
        return _finish(
            dataset.map(mapping_func, fn_kwargs=fn_kwargs, desc="Formatting dataset")
        )

    n_chunks = math.ceil(n_rows / chunk_size)
    cached = sum(
        1
        for i in range(n_chunks)
        if os.path.isdir(os.path.join(chunk_root, f"chunk-{i:06d}"))
    )
    print(
        f"[Info] Formatting {split_name}: {n_rows} rows in {n_chunks} chunk(s) of "
        f"{chunk_size} ({cached} already cached) -> {chunk_root}"
    )

    parts = []
    for i in range(n_chunks):
        chunk_dir = os.path.join(chunk_root, f"chunk-{i:06d}")
        part = _load_if_complete(chunk_dir)
        if part is None:
            lo = i * chunk_size
            hi = min(lo + chunk_size, n_rows)
            part = dataset.select(range(lo, hi))
            # Cap workers at the chunk's row count (datasets raises when
            # num_proc exceeds it) and drop to the single-process path at one
            # worker, so a small chunk does not pay for a worker pool.
            chunk_workers = max(1, min(num_workers, hi - lo))
            part = part.map(
                mapping_func,
                fn_kwargs=fn_kwargs,
                num_proc=chunk_workers if chunk_workers > 1 else None,
                writer_batch_size=writer_batch_size,
                desc=f"Formatting {split_name} chunk {i + 1}/{n_chunks}",
            )
            _atomic_save(_finish(part), chunk_dir, num_workers)
            # Re-open the saved copy so the chunk is memory-mapped from its
            # final location instead of the map's temporary cache files.
            part = _load_if_complete(chunk_dir)
        parts.append(part)

    formatted = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if restore_order:
        inverse_order = np.empty(n_rows, dtype=np.int64)
        inverse_order[np.asarray(order, dtype=np.int64)] = np.arange(n_rows)
        formatted = formatted.select(inverse_order)
    return formatted


def build_task_cache(
    dataset,
    *,
    cache_dir,
    inputs,
    mapping_func,
    model_family_name,
    base_model_hf,
    num_workers_format_dataset,
    process_img,
    save_processed_img_to_disk,
    new_shape_hw,
    task_label,
    temperature_sampler_task_column,
    chunk_size=DEFAULT_CHUNK_SIZE,
    writer_batch_size=50,
):
    """Format one task's splits into the chat layout and publish them as a cache.

    Equivalent to the previous ``format_clean_dataset`` + task-column tagging,
    but every chunk is written as it completes and the finished task is saved
    under ``cache_dir`` before the caller combines it with the other tasks.

    Args:
        dataset (DatasetDict): Train/validation splits from stage 1.
        cache_dir (str): Destination from :func:`task_cache_dir`.
        inputs (dict): From :func:`task_cache_inputs`; stored in the manifest.
        chunk_size (int): Rows per resumable chunk.
        writer_batch_size (int): Rows buffered before each Arrow flush.

    Returns:
        DatasetDict: The formatted, tagged splits, memory-mapped from the cache.
    """
    from medvision_bm.sft.sft_utils import get_cgroup_limited_cpus

    num_workers = min(num_workers_format_dataset, get_cgroup_limited_cpus())
    fn_kwargs = {
        "model_name": model_family_name,
        "model_hf": base_model_hf,
        "process_img": process_img,
        "save_processed_img_to_disk": save_processed_img_to_disk,
        "new_shape_hw": new_shape_hw,
    }
    keys_to_keep = ["messages", "labels", "image_file", "slice_dim", "slice_idx"]
    if process_img:
        keys_to_keep.append("processed_images")
    if save_processed_img_to_disk:
        keys_to_keep.append("image_file_png")
    if temperature_sampler_task_column:
        keys_to_keep.append(temperature_sampler_task_column)

    chunks_root = f"{cache_dir}.chunks"
    print(
        f"\n[Info] Building task cache for '{task_label}' with {num_workers} workers "
        f"(chunk_size={chunk_size}, writer_batch_size={writer_batch_size})\n"
        f"       cache: {cache_dir}\n"
        f"       delete that directory to force a rebuild"
    )

    formatted = DatasetDict(
        {
            split: _format_split_chunked(
                split_ds,
                split_name=split,
                mapping_func=mapping_func,
                fn_kwargs=fn_kwargs,
                keys_to_keep=keys_to_keep,
                task_label=task_label,
                temperature_sampler_task_column=temperature_sampler_task_column,
                num_workers=num_workers,
                writer_batch_size=writer_batch_size,
                chunk_root=os.path.join(chunks_root, split, f"cs{chunk_size}"),
                chunk_size=chunk_size,
            )
            for split, split_ds in dataset.items()
        }
    )

    manifest = {
        "task_label": task_label,
        "key": task_cache_key(inputs),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunk_size": chunk_size,
        "rows": {split: len(ds) for split, ds in formatted.items()},
        "inputs": inputs,
    }
    _atomic_save(
        formatted,
        cache_dir,
        num_workers,
        extra_files={MANIFEST_NAME: manifest},
    )
    shutil.rmtree(chunks_root, ignore_errors=True)
    print(f"[Info] Task '{task_label}' cached at {cache_dir}")
    return load_from_disk(cache_dir)
