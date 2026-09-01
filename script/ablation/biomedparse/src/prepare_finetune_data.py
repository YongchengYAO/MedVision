"""
Prepares MedVision detection data for BiomedParse fine-tuning.

Loads from HuggingFace directly (same pipeline as the SFT build script) so the
110k training samples are identical to those used for VLM SFT training. Uses
group_train_test_split + shuffle(SEED).select(range(train_limit)) to match the
SFT pipeline exactly.

Output structure:
    <output_dir>/
        train/             <- 3-channel 512x512 PNG images
        train_mask/        <- binary mask PNGs (uint8, values 0/1, NOT 0/255)
        train.json
        val/               <- val carve-out (default 105 samples)
        val_mask/
        val.json

Image normalization:
    Uses `normalize_img` from `medvision_bm.sft.sft_utils`.  That function
    handles:
      - CT with organ-specific HU windows (via label_map_regroup + CT_HU_windows_WL)
      - Contrast CT datasets like KiPA22 (forced to general min/max via
        TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION in configs.py)
      - All other modalities (MRI, PET, ultrasound): general min/max

Usage:
    python prepare_finetune_data.py \\
        --tasks_json <repo>/tasks_list/tasks_MedVision-detect__train_SFT.json \\
        --output_dir <ablation>/data/finetune/detect \\
        --train_limit 110000 \\
        --val_limit   105 \\
        --processes   8
"""

import argparse
import importlib
import json
import multiprocessing
import os

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from datasets import concatenate_datasets
from tqdm import tqdm

from _paths import add_medvision_to_path

add_medvision_to_path()

from medvision_bm.sft.sft_utils import normalize_img, _load_single_dataset, group_train_test_split
from medvision_bm.utils.configs import DATASETS_NAME2PACKAGE, SEED

IMG_SIZE = (512, 512)


def load_nifti_2d(nii_path, slice_dim, slice_idx):
    img_nib = nib.load(nii_path)
    image_3d = img_nib.get_fdata().astype("float32")
    if slice_dim == 0:
        return image_3d[slice_idx, :, :]
    elif slice_dim == 1:
        return image_3d[:, slice_idx, :]
    else:
        return image_3d[:, :, slice_idx]


def get_label_name(row):
    """Return the human-readable label name for a parquet row."""
    dataset_module = DATASETS_NAME2PACKAGE.get(row["dataset_name"])
    if dataset_module is None:
        raise ValueError(f"Dataset {row['dataset_name']} not in DATASETS_NAME2PACKAGE")
    preprocess = importlib.import_module(
        f"medvision_ds.datasets.{dataset_module}.preprocess_detection"
    )
    task_info = preprocess.benchmark_plan["tasks"][int(row["taskID"]) - 1]
    label_str = str(row["label"])
    label_name = task_info["labels_map"].get(label_str)
    if label_name is None:
        raise ValueError(f"Label {label_str} not in labels_map for {row['dataset_name']}")
    if label_name == "arota":
        label_name = "aorta"
    return label_name


def process_one_sample(args):
    row, output_dir, split = args
    try:
        label_name = get_label_name(row)

        # --- Image ---
        # normalize_img handles: CT with HU windows, contrast CT (KiPA22 etc.),
        # and general min-max for MRI/PET/ultrasound.
        # See medvision_bm/sft/sft_utils.py:normalize_img and
        #     medvision_bm/utils/configs.py:TASK_LIST_FORCE_STANDARD_IMAGE_NORMALIZATION
        img_2d = load_nifti_2d(row["image_file"], int(row["slice_dim"]), int(row["slice_idx"]))
        img_norm = normalize_img(row, img_2d)  # returns uint8 [H, W]

        img_resized = cv2.resize(img_norm, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        # Replicate grayscale to 3-channel (all channels identical; BGR/RGB order
        # is irrelevant since the values are the same across channels)
        img_rgb = np.stack([img_resized, img_resized, img_resized], axis=2)

        # --- Mask ---
        # Extract binary mask: foreground = pixels matching this label, saved as 0/1 uint8.
        # CRITICAL: must NOT save as 0/255 — BiomedParseDataset zeroes out {0,255} masks.
        mask_2d = load_nifti_2d(row["mask_file"], int(row["slice_dim"]), int(row["slice_idx"]))
        binary_mask = (mask_2d == int(row["label"])).astype(np.uint8)
        mask_resized = cv2.resize(binary_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        # --- Filenames ---
        img_basename = os.path.basename(row["image_file"]).replace(".nii.gz", "")
        filename = (
            f"{row['dataset_name']}__{img_basename}"
            f"__dim{row['slice_dim']}__idx{row['slice_idx']}__lbl{row['label']}.png"
        )

        cv2.imwrite(os.path.join(output_dir, split, filename), img_rgb)
        cv2.imwrite(os.path.join(output_dir, f"{split}_mask", filename), mask_resized)

        return {
            "file_name": filename,
            "mask_file": filename,
            "class_prompts": {"1": label_name},
            "instance_label": True,
        }

    except Exception as e:
        return ("error", str(e))


def prepare_split(df, output_dir, split, num_workers):
    df = df.reset_index(drop=True)
    print(f"[{split}] Processing {len(df)} samples")

    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"{split}_mask"), exist_ok=True)

    tasks = [(row, output_dir, split) for row in df.to_dict("records")]

    if num_workers and num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_one_sample, tasks), total=len(tasks)))
    else:
        results = [process_one_sample(t) for t in tqdm(tasks)]

    errors = [r for r in results if isinstance(r, tuple) and r[0] == "error"]
    if errors:
        print(f"[{split}] First 3 errors:")
        for _, msg in errors[:3]:
            print(f"  {msg}")
    annotations = [r for r in results if isinstance(r, dict)]
    skipped = len(df) - len(annotations)
    print(f"[{split}] Saved {len(annotations)} samples (skipped {skipped})")

    json_path = os.path.join(output_dir, f"{split}.json")
    with open(json_path, "w") as f:
        json.dump({"annotations": annotations}, f)
    print(f"[{split}] Wrote {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare BiomedParse fine-tuning data from HuggingFace, "
                    "using the same split as the MedVision SFT pipeline."
    )
    parser.add_argument(
        "--tasks_json",
        type=str,
        required=True,
        help="Path to tasks list JSON (e.g. tasks_MedVision-detect__train_SFT.json)",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for the PNG + JSON fine-tuning data")
    parser.add_argument(
        "--train_limit",
        type=int,
        default=110000,
        help="Training samples to select after val carve-out (matches SFT default)",
    )
    parser.add_argument(
        "--val_limit",
        type=int,
        default=105,
        help="Val samples carved out via group_train_test_split (matches SFT default)",
    )
    parser.add_argument("--processes", "-p", type=int, default=None)
    parser.add_argument(
        "--filter_dataset",
        type=str,
        default=None,
        help="Comma-separated dataset_name(s) whose _Train configs are loaded (e.g. KiPA22). "
             "NOTE: a filtered pool does NOT reproduce the identical-110k-sample guarantee.",
    )
    args = parser.parse_args()

    with open(args.tasks_json) as f:
        tasks_dict = json.load(f)

    # Load all _Train configs from HuggingFace (no per-task limit — full pool)
    print(f"Loading {len(tasks_dict)} train configs from HuggingFace ...")
    # filter_dataset accepts a comma-separated list; a task is kept when its dataset matches any entry
    _wanted = {d.strip() for d in args.filter_dataset.split(",") if d.strip()} if args.filter_dataset else None
    all_ds = []
    for task_key in tasks_dict:
        if _wanted is not None and task_key.split("_BoxSize_")[0] not in _wanted:
            continue
        config = task_key + "_Train"
        ds = _load_single_dataset(
            "YongchengYAO/MedVision",
            dataset_name=task_key.split("_BoxSize_")[0],
            config=config,
            split="train",
            limit=None,
        )
        all_ds.append(ds)
        print(f"  {config}: {len(ds)} samples")

    if not all_ds:
        raise SystemExit(f"No task in {args.tasks_json} matches dataset '{args.filter_dataset}'")
    combined = concatenate_datasets(all_ds)
    print(f"Total pool: {len(combined)} samples")

    # Carve out val with group-aware split — same as SFT pipeline
    split_ds = group_train_test_split(
        combined,
        group_column="image_file",
        test_size=args.val_limit,
        seed=SEED,
        stratify_column="dataset_name",
    )

    # Select train samples — same shuffle + select as SFT pipeline
    train_ds = split_ds["train"].shuffle(seed=SEED).select(
        range(min(args.train_limit, len(split_ds["train"])))
    )
    val_ds = split_ds["validation"]

    train_df = train_ds.to_pandas()
    val_df = val_ds.to_pandas()
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    prepare_split(train_df, args.output_dir, "train", args.processes)
    prepare_split(val_df, args.output_dir, "val", args.processes)
