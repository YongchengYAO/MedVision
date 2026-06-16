import argparse
import json
import multiprocessing
import os

from huggingface_hub import HfApi
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Clean PEFT/LoRA wrappers from safetensors checkpoint files from verl."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the model directory containing safetensors files",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the cleaned model to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Hugging Face repository ID to push the cleaned model",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    assert os.path.exists(model_dir), f"Model path {model_dir} does not exist."
    assert os.path.isdir(model_dir), f"Model path {model_dir} is not a directory."

    if args.push_to_hub:
        assert (
            args.repo_id is not None
        ), "repo_id must be provided when push_to_hub is set."

    return args


def push_to_hf_hub(model_dir, repo_id):
    api = HfApi()

    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Successfully uploaded {model_dir} to {repo_id}")


def process_single_file(file_path):
    # Fast check to see if we even need to load the file
    has_peft_wrapper = False
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("base_model.model."):
                    has_peft_wrapper = True
                    break
    except Exception:
        # If safe_open fails, fall back to full load
        try:
            state_dict = load_file(file_path)
        except Exception as e:
            return f"Failed: {str(e)}"

        has_peft_wrapper = any(
            k.startswith("base_model.model.") for k in state_dict.keys()
        )
        if not has_peft_wrapper:
            return "Skipped"

        # Continue with processing using already loaded state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("base_model.model."):
                new_k = k.replace("base_model.model.", "", 1)
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        save_file(new_state_dict, file_path)
        return "Modified"

    if not has_peft_wrapper:
        return "Skipped"

    # Load the original weights
    state_dict = load_file(file_path)
    new_state_dict = {}

    for k, v in state_dict.items():
        # Strips PEFT/LoRA wrappers only if key starts with "base_model.model."
        if k.startswith("base_model.model."):
            new_k = k.replace("base_model.model.", "", 1)
            new_state_dict[new_k] = v
        else:
            # Keep the key as is if it doesn't have the wrapper
            new_state_dict[k] = v

    # Save the "cleaned" version back to the same path
    save_file(new_state_dict, file_path)
    return "Modified"


def patch_layer_names(model_dir):
    file_paths = [
        os.path.join(model_dir, filename)
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    ]

    # Use number of CPUs available
    num_processes = min(multiprocessing.cpu_count(), 32)
    print(f"Using {num_processes} processes to patch layer names...")

    modified_count = 0
    skipped_count = 0
    failed_count = 0

    with multiprocessing.Pool(processes=num_processes) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_single_file, file_paths), total=len(file_paths)
        )
        for result in pbar:
            if result == "Modified":
                modified_count += 1
            elif result == "Skipped":
                skipped_count += 1
            elif result.startswith("Failed"):
                failed_count += 1
                tqdm.write(f"Error: {result}")

            pbar.set_postfix(
                modified=modified_count, skipped=skipped_count, failed=failed_count
            )

    print(
        f"\nProcessing complete. Modified: {modified_count}, Skipped: {skipped_count}, Failed: {failed_count}"
    )


def patch_index_file(model_dir):
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        print(f"Index file {index_file} not found. Skipping.")
        return

    print(f"Patching index file: {index_file}")
    try:
        with open(index_file, "r") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        new_weight_map = {}
        modified = False

        for k, v in weight_map.items():
            if k.startswith("base_model.model."):
                new_k = k.replace("base_model.model.", "", 1)
                new_weight_map[new_k] = v
                modified = True
            else:
                new_weight_map[k] = v

        if modified:
            index_data["weight_map"] = new_weight_map
            with open(index_file, "w") as f:
                json.dump(index_data, f, indent=2)
            print("Index file patched successfully.")
        else:
            print("No changes needed for index file.")

    except Exception as e:
        print(f"Failed to patch index file: {e}")


def main():
    args = parse_arguments()

    patch_layer_names(args.model_dir)
    patch_index_file(args.model_dir)
    if args.push_to_hub:
        push_to_hf_hub(args.model_dir, args.repo_id)


if __name__ == "__main__":
    main()
