from datasets import load_dataset


def tasks_to_configs(tasks, split):
    """Convert task names into MedVision dataset config names for a split.

    Appends ``"_Train"`` or ``"_Test"`` to each task name and rewrites the legacy
    ``"BoxCoordinate"`` token to ``"BoxSize"`` to match the dataset's config
    naming. Task names keep ``"BoxCoordinate"`` because ``"BoxSize"`` is reserved
    there for mask-size estimation tasks.

    Args:
        tasks (list): Task names to convert.
        split (str): Data split, ``"train"`` or ``"test"`` (case-insensitive).

    Returns:
        list: Dataset config names, one per input task.

    Raises:
        AssertionError: If ``split`` is not ``"train"`` or ``"test"``.
    """
    assert split.lower() in ["train", "test"], "Split must be 'train' or 'test'"
    if split.lower() == "train":
        split = "Train"
    else:
        split = "Test"

    # Generate dataset configurations based on tasks and split: appending "_Train" or "_Test" to each task
    configs = []
    for task in tasks:
        config = f"{task}_{split}"
        configs.append(config)

    # [Fix legacy naming issue] Replace "BoxCoordinate" with "BoxSize" in config names
    #   - the dataset uses "BoxSize" instead of "BoxCoordinate"
    #   - the tasks are named with "BoxCoordinate"
    # [Note] Why we did not change the task name?
    #   - Using "BoxSize" in task names is reserved for mask size estimation tasks.
    configs = [config.replace("BoxCoordinate", "BoxSize") for config in configs]
    return configs


def download_datasets_from_configs(configs, split="test"):
    """Download MedVision dataset splits for the given config names.

    Args:
        configs (list): Dataset config names to download.
        split (str): Split requested from ``datasets.load_dataset``. Defaults to
            ``"test"``.

    Note:
        The MedVision data-loading script downloads the raw data for both the
        train and test splits regardless of the requested ``split`` or config, so
        any split/config downloads the entire dataset. See the loading script at
        https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/MedVision.py
        and the config list at
        https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/info.
    """
    for config in configs:
        print(f"Downloading dataset for config: {config}, split: {split}")
        load_dataset(
            "YongchengYAO/MedVision",
            name=config,
            trust_remote_code=True,
            split=split,
            streaming=False,
        )
        print("Finished downloading.")


def download_datasets_from_tasks(tasks, split="test"):
    """Download MedVision datasets for the given task names and split.

    Convenience wrapper that converts ``tasks`` to config names via
    ``tasks_to_configs`` and then calls ``download_datasets_from_configs``.

    Args:
        tasks (list): Task names to download data for.
        split (str): Data split, ``"train"`` or ``"test"``. Defaults to
            ``"test"``.
    """
    configs = tasks_to_configs(tasks, split)
    download_datasets_from_configs(configs, split)
