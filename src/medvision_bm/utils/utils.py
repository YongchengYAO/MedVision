import json
import os
import tempfile

import torch


def atomic_write_json(json_path, data, indent=4):
    """Write ``data`` as JSON to ``json_path`` atomically.

    Writes to a temp file in the same directory, fsyncs it, then ``os.replace``-es
    it over the target. If the write fails (e.g. ENOSPC on a full disk), the
    original file is left untouched instead of being truncated to 0 bytes, which
    is what opening the target directly with mode ``"w"`` would do.

    Args:
        json_path (str): Destination path for the JSON file.
        data: Any JSON-serializable object to write.
        indent (int): Indentation passed to ``json.dump``. Defaults to ``4``.

    Raises:
        BaseException: Re-raises any error raised while writing or replacing the
            file, after removing the partial temp file.
    """
    dir_name = os.path.dirname(json_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, json_path)
    except BaseException:
        # Don't leave a partial temp file behind on failure.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def str2bool(v):
    """Coerce a string (or bool) to a boolean, for use as an ``argparse`` type.

    Args:
        v: A boolean, or a string such as ``"yes"``, ``"true"``, ``"1"``,
            ``"no"``, ``"false"``, or ``"0"`` (case-insensitive).

    Returns:
        bool: ``True`` for truthy tokens and ``False`` for falsy tokens. A value
        that is already a ``bool`` is returned unchanged.

    Raises:
        argparse.ArgumentTypeError: If ``v`` is not a recognized boolean token.
    """
    import argparse

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_cuda_num_processes():
    """Determine the number of GPU processes to launch from the CUDA environment.

    Reads ``CUDA_VISIBLE_DEVICES``. When it is unset, all GPUs reported by
    ``torch.cuda.device_count()`` are used; otherwise the number of device ids
    listed in the variable is used (at least 1).

    Returns:
        int: The number of processes to run, one per visible GPU.
    """
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is None:
        num_processes = torch.cuda.device_count()
        print(
            f"No CUDA_VISIBLE_DEVICES found. Using all available GPUs: {num_processes}"
        )
        return num_processes
    else:
        num_processes = max(1, len([d for d in cuda_visible.split(",") if d.strip()]))
        print(
            f"Using CUDA_VISIBLE_DEVICES={cuda_visible}; num_processes={num_processes}"
        )
        return num_processes


def update_task_status(json_path, model_name, task_name):
    """Mark a (model, task) pair as completed in a JSON tracking file.

    Loads the existing status file (creating the parent directory and treating a
    missing file as empty), sets ``data[model_name][task_name]`` to ``True``, and
    writes the file back atomically.

    Args:
        json_path (str): Path to the JSON tracking file.
        model_name (str): Model whose status is updated.
        task_name (str): Task to mark as completed.

    Returns:
        bool: Always ``False``; the return value is unused by callers.
    """
    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Update the completion status
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    if model_name not in data:
        data[model_name] = {}
    data[model_name][task_name] = True
    atomic_write_json(json_path, data)

    return False


def load_tasks(json_file_path):
    """Load task names from a JSON file mapping task names to their definitions.

    Args:
        json_file_path (str): Path to a JSON file whose top-level keys are task
            names.

    Returns:
        list: The task names (the top-level keys), in file order.
    """
    with open(json_file_path, "r") as f:
        tasks_dict = json.load(f)
    tasks = list(tasks_dict.keys())
    print(f"\nFound {len(tasks)} tasks to process: {tasks}\n")
    return tasks


def load_tasks_status(tasks_status_file, model_name):
    """Return the completion-status mapping for a single model.

    Args:
        tasks_status_file (str): Path to the JSON status file. A missing file is
            treated as empty.
        model_name (str): Model whose status entry is returned.

    Returns:
        dict: Mapping of task name to completion flag for ``model_name``, or an
        empty dict if the file or the model entry is absent.

    Raises:
        ValueError: If the status file exists but cannot be read or parsed.
    """
    if os.path.exists(tasks_status_file):
        try:
            with open(tasks_status_file, "r") as f:
                completed_all = json.load(f)
        except Exception as e:
            raise ValueError(
                f"Error loading tasks status file: {tasks_status_file}\nError: {e}"
            )
    else:
        completed_all = {}
    return completed_all.get(model_name, {})
