import os
import json


def set_cuda_num_processes(minimum_gpu: int = 1):
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is None:
        print(f"No CUDA_VISIBLE_DEVICES found. Using minimum GPU: {minimum_gpu}")
        num_processes = minimum_gpu
        return num_processes
    else:
        num_processes = max(1, len([d for d in cuda_visible.split(",") if d.strip()]))
        print(
            f"Using CUDA_VISIBLE_DEVICES={cuda_visible}; num_processes={num_processes}"
        )
        return num_processes


def setup_env_var(data_dir: str):
    # Set environment variables BEFORE importing datasets
    os.environ["MedVision_DATA_DIR"] = data_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        data_dir, ".cache", "huggingface", "datasets"
    )
    os.environ["HF_HOME"] = os.path.join(data_dir, ".cache", "huggingface")


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
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    return False


def load_tasks(json_file_path: str):
    with open(json_file_path, "r") as f:
        tasks_dict = json.load(f)
    tasks = list(tasks_dict.keys())
    print(f"\nFound {len(tasks)} tasks to process: {tasks}\n")
    return tasks


def load_tasks_status(tasks_status_file: str, model_name: str):
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
