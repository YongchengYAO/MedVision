import argparse
import csv
import json
from medvision_bm.utils.data_utils import download_datasets_from_configs, download_datasets_from_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Download MedVision datasets from configs or tasks"
    )
    parser.add_argument(
        "--configs_csv",
        type=str,
        help="Path to CSV file containing list of dataset configurations to download"
    )
    parser.add_argument(
        "--tasks_json",
        type=str,
        help="Path to JSON file containing list of task names to download"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Split to download (default: test)"
    )

    args = parser.parse_args()

    if not args.configs_csv and not args.tasks_json:
        parser.error(
            "At least one of --configs_csv or --tasks_json must be provided")

    if args.configs_csv and args.tasks_json:
        parser.error(
            "Please provide either --configs_csv or --tasks_json, not both")

    if args.configs_csv:
        with open(args.configs_csv, 'r') as f:
            reader = csv.reader(f)
            # Extract first column, skip empty rows
            configs = [row[0] for row in reader if row]
        download_datasets_from_configs(configs, split=args.split)
    elif args.tasks_json:
        with open(args.tasks_json, 'r') as f:
            tasks_dict = json.load(f)
            # Extract task names from dictionary keys
            tasks = list(tasks_dict.keys())
        download_datasets_from_tasks(tasks, split=args.split)


if __name__ == "__main__":
    main()
