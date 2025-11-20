from datasets import load_dataset


def tasks_to_configs(tasks, split):
    assert split.lower() in [
        "train", "test"], "Split must be 'train' or 'test'"
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
    """
    # NOTE: Raw data from both the train and test splits are downloaded here, regardless of the specified split and data configuration.
    #       Therefore, using any split and data config will result in downloading the entire dataset.
    #       This is due to the design of the MedVision data loading script: https://huggingface.co/datasets/YongchengYAO/MedVision/blob/main/MedVision.py
    #
    # List of configs: https://huggingface.co/datasets/YongchengYAO/MedVision/tree/main/info
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
    configs = tasks_to_configs(tasks, split)
    download_datasets_from_configs(configs, split)
