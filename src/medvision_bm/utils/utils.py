import json
import os
import tempfile

import torch


def atomic_write_json(json_path, data, indent=4):
    """
    Write ``data`` as JSON to ``json_path`` atomically.

    Writes to a temp file in the same directory, fsyncs it, then ``os.replace``-es
    it over the target. If the write fails (e.g. ENOSPC on a full disk), the
    original file is left untouched instead of being truncated to 0 bytes, which
    is what opening the target directly with mode "w" would do.
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
    """
    Update a JSON tracking file.

    Args:
        json_path (str): Path to the JSON file
        model_name (str): Model name to update
        task_name (str): Task that has been completed

    Returns:
        bool: True if update succeeded, False otherwise
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
    with open(json_file_path, "r") as f:
        tasks_dict = json.load(f)
    tasks = list(tasks_dict.keys())
    print(f"\nFound {len(tasks)} tasks to process: {tasks}\n")
    return tasks


def load_tasks_status(tasks_status_file, model_name):
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
