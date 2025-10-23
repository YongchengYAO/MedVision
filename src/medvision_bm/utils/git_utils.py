import json
import os
import random
import subprocess
import time


def run_with_retry(cmd, max_retries=10, **kwargs):
    for i in range(max_retries):
        try:
            subprocess.run(cmd, check=True, shell=True, **kwargs)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed (attempt {i+1}/{max_retries}): {e}")
            if i == max_retries - 1:
                print(f"Max retries reached for command: {cmd}")
                return False
            import time

            time.sleep(2)  # Wait before retrying


def update_task_status_and_gitpush(json_path, model_name, task_name, max_retries=10):
    """
    Update a JSON tracking file in a Git repository with retry mechanism for conflicts.

    Args:
        json_path (str): Path to the JSON file
        model_name (str): Model name to update
        task_name (str): Task that has been completed
        max_retries (int): Maximum number of retry attempts

    Returns:
        bool: True if update succeeded, False otherwise
    """
    for attempt in range(1, max_retries + 1):
        try:
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

            # Check if there are changes to commit for the JSON file
            status = subprocess.run(
                ["git", "status", "--porcelain", json_path],
                capture_output=True,
                text=True,
                check=True,
            )
            if status.stdout.strip():
                subprocess.run(["git", "add", json_path], check=True)
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"Update {model_name} completion status for {task_name}",
                    ],
                    check=True,
                )
                subprocess.run(["git", "pull", "--rebase"], check=True)
                subprocess.run(["git", "push"], check=True)
                print(
                    f"Successfully updated {json_path} after rebase on attempt {attempt}"
                )
                return True
            else:
                print(f"No changes to commit for {json_path}")
                return True

        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                sleep_time = random.randint(1, 5)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Maximum retries reached. Failed to update JSON file.")
                return False

    return False


def push_results_to_huggingface(result_dir, task, max_retries=10):
    """
    Push results to HuggingFace repository with retry mechanism.

    Args:
        result_dir (str): Path to the results directory (git repository)
        task (str): Task name for the commit message
        max_retries (int): Maximum number of retry attempts

    Returns:
        bool: True if push succeeded, False otherwise
    """
    # Save current directory to restore later
    original_dir = os.getcwd()

    try:
        print(f"Task {task} completed. Saving results to HF...")
        os.chdir(result_dir)

        for attempt in range(1, max_retries + 1):
            try:
                status = subprocess.run(
                    "git status --porcelain", capture_output=True, text=True, shell=True
                )
                if status.stdout.strip():
                    subprocess.run("git add .", check=True, shell=True)
                    subprocess.run(
                        f"git commit -m 'update results: {task}'",
                        check=True,
                        shell=True,
                    )
                    subprocess.run("git pull --rebase", check=True, shell=True)
                    subprocess.run("git push", check=True, shell=True)
                    print(
                        f"Successfully pushed results for {task} on attempt {attempt}"
                    )
                    return True
                else:
                    print("No changes to commit")
                    return True

            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    sleep_time = random.randint(1, 5)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(
                        f"Maximum retries reached. Failed to push results for {task}."
                    )
                    return False

        return False

    except Exception as e:
        print(f"Error pushing results to HF: {e}")
        return False
    finally:
        # Restore original directory regardless of success/failure
        os.chdir(original_dir)


def clone_github_repo_with_fallback(repo_url, target_dir, commit_hash, parent_dir=None):
    """
    Clone a GitHub repository using git, with fallback to zip download if git fails.

    Args:
        repo_url: GitHub repository URL (e.g., "https://github.com/user/repo")
        target_dir: Target directory path where the repo should be cloned
        commit_hash: Git commit hash to checkout
        parent_dir: Parent directory for git operations (defaults to target_dir's parent)

    Raises:
        RuntimeError: If both git clone and zip download fail
    """
    if parent_dir is None:
        parent_dir = os.path.dirname(target_dir)

    repo_name = os.path.basename(target_dir)

    # Try git clone first
    try:
        print(f"Attempting to clone {repo_url} using git...")
        subprocess.run(
            f"git clone {repo_url} {repo_name}",
            cwd=parent_dir,
            check=True,
            shell=True,
        )
        # Checkout specific commit
        subprocess.run(
            f"git checkout {commit_hash}",
            cwd=target_dir,
            check=True,
            shell=True,
        )
        print(f"Successfully cloned {repo_url} to {target_dir}")
        return
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed: {e}")
        print("Attempting fallback to zip download...")

    # Fallback: download as zip file
    zip_url = f"{repo_url}/archive/{commit_hash}.zip"
    zip_filename = f"{repo_name}.zip"
    zip_path = os.path.join(parent_dir, zip_filename)

    try:
        # Download zip file
        print(f"Downloading {zip_url}...")
        subprocess.run(
            f"curl -L {zip_url} -o {zip_path}",
            check=True,
            shell=True,
        )

        # Extract zip file
        print(f"Extracting {zip_path}...")
        subprocess.run(
            f"unzip -q {zip_path} -d {parent_dir}",
            check=True,
            shell=True,
        )

        # Move extracted directory to the correct location
        extracted_dir = os.path.join(parent_dir, f"{repo_name}-{commit_hash}")
        print(f"Moving {extracted_dir} to {target_dir}...")

        if not os.path.exists(extracted_dir) or not os.path.isdir(extracted_dir):
            raise RuntimeError(
                f"Failed to extract {zip_path} properly - extracted dir {extracted_dir} not found"
            )

        # Remove existing directory if it exists
        if os.path.exists(target_dir):
            subprocess.run(f"rm -rf {target_dir}", check=True, shell=True)

        # Move extracted directory to target location
        subprocess.run(
            f"mv {extracted_dir} {target_dir}",
            check=True,
            shell=True,
        )

        # Clean up zip file
        os.remove(zip_path)
        print(f"Successfully downloaded and extracted {repo_name} to {target_dir}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to download and extract {repo_name} repository: {str(e)}"
        )
