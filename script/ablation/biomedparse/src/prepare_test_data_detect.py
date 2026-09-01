import argparse
import importlib
import json
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from _paths import add_medvision_to_path

add_medvision_to_path()

try:
    from medvision_bm.sft.sft_utils import _load_resize_nifti_2d, _load_single_dataset
    from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Please ensure you are running this script with the correct python environment and repository structure."
    )
    sys.exit(1)


def normalize_ct(img, window_width, window_level):
    v_min = window_level - (window_width / 2)
    v_max = window_level + (window_width / 2)
    img_normalized = np.clip(img, v_min, v_max)
    img_normalized = ((img_normalized - v_min) / (v_max - v_min)) * 255.0
    return img_normalized.astype(np.uint8)


def normalize_general(img):
    v_min = np.percentile(img, 0.5)
    v_max = np.percentile(img, 99.5)
    if v_max - v_min == 0:
        return img.astype(np.uint8)
    img_normalized = np.clip(img, v_min, v_max)
    img_normalized = ((img_normalized - v_min) / (v_max - v_min)) * 255.0
    return img_normalized.astype(np.uint8)


def _doc_to_visual_adaptive_norm(doc, image_modality, label_name, new_shape_hw=None):
    from medvision_bm.utils.configs import CT_HU_windows_WL, label_map_regroup

    img_path = doc["image_file"]
    slice_dim = doc["slice_dim"]
    slice_idx = doc["slice_idx"]
    _, img_2d = _load_resize_nifti_2d(img_path, slice_dim, slice_idx, new_shape_hw)

    label_group = label_map_regroup.get(label_name, "Others")
    hu_window_WL = CT_HU_windows_WL.get(label_group) if image_modality.lower() == "ct" else None
    if hu_window_WL is not None:
        img_2d_normalized = normalize_ct(img_2d, hu_window_WL[0], hu_window_WL[1])
    else:
        img_2d_normalized = normalize_general(img_2d)

    img_3d = np.expand_dims(img_2d_normalized, axis=0)
    return [img_3d]


def __get_image_info(doc, processor_module="preprocess_detection"):
    dataset_name = doc["dataset_name"]
    dataset_module = DATASETS_NAME2PACKAGE.get(dataset_name)
    if dataset_module is None:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS_NAME2PACKAGE.")

    preprocess_detection = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.{processor_module}"
    )

    taskID = doc["taskID"]
    bm_plan = preprocess_detection.benchmark_plan
    task_info = bm_plan["tasks"][int(taskID) - 1]

    label = str(doc["label"])
    labels_map = task_info["labels_map"]
    if label not in labels_map:
        raise ValueError(f"Label {label} not found in labels_map.")
    else:
        label_name = labels_map.get(label)

    if label_name == "arota":
        label_name = "aorta"

    image_modality = task_info["image_modality"]
    return image_modality, label_name


def process_one_sample(args):
    doc, output_dir = args
    try:
        label = str(doc["label"])
        image_modality, label_name = __get_image_info(doc)

        visuals = _doc_to_visual_adaptive_norm(doc, image_modality, label_name)
        pil_img = visuals[0]
        img_array = np.array(pil_img)

        dataset_name = doc["dataset_name"]

        text_prompts_dict = {label: label_name}
        text_prompts_dict["instance_label"] = 0

        image_file = os.path.basename(doc["image_file"])
        slice_dim = doc["slice_dim"]
        slice_idx = doc["slice_idx"]
        pixel_size = doc["pixel_size"]

        filename = f"{dataset_name}__{image_file.replace('.nii.gz', '')}__dim{slice_dim}__idx{slice_idx}__lbl{label}.npz"
        save_path = os.path.join(output_dir, filename)

        np.savez(
            save_path,
            imgs=img_array,
            text_prompts=text_prompts_dict,
            pixel_size=pixel_size,
            slice_dim=slice_dim,
            slice_idx=slice_idx,
        )
        return True

    except Exception as e:
        raise ValueError(f"Error processing row: {e}")
        return False


def prepare_data(tasks_json, output_dir, num_workers=None, limit_per_subtask=-1, filter_dataset=None):
    with open(tasks_json) as f:
        tasks_dict = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Loading {len(tasks_dict)} task configs from HuggingFace (YongchengYAO/MedVision) ...")

    limit = limit_per_subtask if limit_per_subtask > 0 else None
    # filter_dataset accepts a comma-separated list; a task is kept when its dataset matches any entry
    _wanted = {d.strip() for d in filter_dataset.split(",") if d.strip()} if filter_dataset else None

    all_dfs = []
    for task_key in tasks_dict:
        if filter_dataset is not None and task_key.split("_BoxSize_")[0] not in _wanted:
            continue
        config = task_key + "_Test"
        # _load_single_dataset uses ds.select(range(limit)) — first N rows in HF native order,
        # matching lmms-eval's islice(iterator, 0, limit, 1) behavior exactly.
        ds = _load_single_dataset(
            "YongchengYAO/MedVision",
            dataset_name=task_key.split("_BoxSize_")[0],
            config=config,
            split="test",
            limit=limit,
        )
        all_dfs.append(ds.to_pandas())
        print(f"  {config}: {len(ds)} samples")

    if not all_dfs:
        sys.exit(f"No task in {tasks_json} matches dataset '{filter_dataset}'")
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total samples after loading all configs: {len(df)}")

    if filter_dataset is not None:
        df = df[df["dataset_name"].isin(_wanted)].reset_index(drop=True)
        print(f"Filtered to dataset '{filter_dataset}': {len(df)} samples")

    docs = df.to_dict("records")
    tasks = [(doc, output_dir) for doc in docs]

    if num_workers is not None and num_workers > 1:
        print(f"Starting multiprocessing with {num_workers} workers...")
        with multiprocessing.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_one_sample, tasks),
                    total=len(tasks),
                    desc="Processing samples",
                )
            )
    else:
        print("Processing sequentially...")
        results = []
        for task in tqdm(tasks, desc="Processing samples"):
            results.append(process_one_sample(task))

    success_count = sum(results)
    error_count = len(results) - success_count

    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare NPZ test data for BiomedParse Detection, loading from HuggingFace "
                    "in native order to match lmms-eval's first-N selection."
    )
    parser.add_argument(
        "--tasks_json",
        type=str,
        required=True,
        help="Path to tasks list JSON (e.g. tasks_MedVision-detect__train_SFT.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for the prepared .npz files",
    )
    parser.add_argument(
        "--processes", "-p",
        type=int,
        default=None,
        help="Number of workers for multiprocessing. If not set, run sequentially.",
    )
    parser.add_argument(
        "--limit_per_subtask",
        type=int,
        default=-1,
        help="Max samples per subtask config (first N in HF order). -1 means no limit.",
    )
    parser.add_argument(
        "--filter_dataset",
        type=str,
        default=None,
        help="Comma-separated dataset_name(s) to keep (e.g. KiPA22 or KiPA22,BraTS24). "
             "Tasks of other datasets are skipped before loading.",
    )
    args = parser.parse_args()

    prepare_data(
        args.tasks_json,
        args.output_dir,
        num_workers=args.processes,
        limit_per_subtask=args.limit_per_subtask,
        filter_dataset=args.filter_dataset,
    )
