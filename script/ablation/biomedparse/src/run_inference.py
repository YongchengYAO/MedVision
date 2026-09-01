import argparse
import os

from _paths import BIOMEDPARSE_DIR, add_biomedparse_to_path


def parse_args():
    parser = argparse.ArgumentParser(description="BiomedParse v2 inference on prepared MedVision .npz samples (Detection or T/L)")
    parser.add_argument(
        "--npz_dir",
        type=str,
        required=True,
        help="Directory containing prepared .npz files",
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        required=True,
        help="Directory to save segmentation results (.nii.gz)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value (e.g. '0', '0,1')",
    )
    parser.add_argument(
        "--slice_batch_size",
        type=int,
        default=4,
        help="Slice batch size passed to the model",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples whose _pred_mask.nii.gz already exists in seg_dir",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a local .ckpt file (e.g. finetuned model). "
            "If omitted, downloads biomedparse_v2.ckpt from HuggingFace."
        ),
    )
    parser.add_argument(
        "--filter_dataset",
        type=str,
        default=None,
        help="If set, only run inference on NPZ files from this dataset (e.g. KiPA22).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Must be set before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Upstream modules (`utils`, `inference`, `src.*`) live in the pinned checkout.
    add_biomedparse_to_path()

    import hydra
    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from inference import merge_multiclass_masks, postprocess
    from tqdm import tqdm
    from utils import process_input, process_output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=os.path.join(BIOMEDPARSE_DIR, "configs", "model"), job_name="medvision_inference"
    )
    cfg = compose(config_name="biomedparse_3D")

    model = hydra.utils.instantiate(cfg, _convert_="object")

    if args.checkpoint:
        print(f"Loading local checkpoint: {args.checkpoint}")
        ckpt_path = args.checkpoint
    else:
        from huggingface_hub import hf_hub_download
        print("Downloading pretrained checkpoint from HuggingFace...")
        ckpt_path = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v2.ckpt")

    model.load_pretrained(ckpt_path)
    model = model.to(device).train(False)

    os.makedirs(args.seg_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    if args.filter_dataset:
        prefix = f"{args.filter_dataset}__"
        files = [f for f in files if f.startswith(prefix)]
        print(f"Filtered to dataset '{args.filter_dataset}': {len(files)} NPZ files")

    for file_name in tqdm(files, desc="Running BiomedParse inference"):
        pred_mask_path = os.path.join(
            args.seg_dir, file_name.replace(".npz", "_pred_mask.nii.gz")
        )
        if args.skip_existing and os.path.exists(pred_mask_path):
            continue

        file_path = os.path.join(args.npz_dir, file_name)
        try:
            npz_data = np.load(file_path, allow_pickle=True)
            imgs = npz_data["imgs"]
            text_prompts = npz_data["text_prompts"].item()
        except Exception as load_err:
            print(f"WARNING: skipping {file_name} — failed to load: {load_err}")
            continue

        print(f"Loaded image shape: {imgs.shape}")
        print(f"Text prompts: {text_prompts}")

        ids = [int(k) for k in text_prompts.keys() if k != "instance_label"]
        ids.sort()
        text = "[SEP]".join([text_prompts[str(i)] for i in ids])
        print(f"text: {text}")

        imgs_input, pad_width, padded_size, valid_axis = process_input(imgs, 512)
        imgs_tensor = imgs_input.to(device).int()

        input_tensor = {
            "image": imgs_tensor.unsqueeze(0),
            "text": [text],
        }

        print("Running inference...")
        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=args.slice_batch_size)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(
            mask_preds, size=(512, 512), mode="bicubic", align_corners=False, antialias=True
        )
        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        unique_labels = np.unique(
            mask_preds.cpu().numpy() if isinstance(mask_preds, torch.Tensor) else mask_preds
        )
        print(f"Unique labels in predicted mask: {unique_labels}")

        if isinstance(mask_preds, torch.Tensor):
            mask_preds = mask_preds.cpu().numpy()

        print(f"Processed mask shape: {mask_preds.shape}")

        affine = np.eye(4)
        nib.save(
            nib.Nifti1Image(imgs, affine),
            os.path.join(args.seg_dir, file_name.replace(".npz", ".nii.gz")),
        )
        nib.save(
            nib.Nifti1Image(mask_preds.astype(np.float32), affine),
            pred_mask_path,
        )
        print(f"Saved mask: {file_name.replace('.npz', '_pred_mask.nii.gz')}")


if __name__ == "__main__":
    main()
