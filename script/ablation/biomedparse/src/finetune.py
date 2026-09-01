"""
Standalone PyTorch Lightning fine-tuning script for BiomedParse on MedVision data.

Does not depend on AzureML/Olympus. Uses Hydra only to instantiate the model
architecture from the existing YAML configs.

Prerequisites:
  1. Run prepare_finetune_data.py to prepare PNG + JSON data.
     That script creates both train/ and val/ splits in <data_dir>.
  2. Have biomedparse_v2.ckpt available (the launchers download it into models/).

Usage:
    python finetune.py \\
        --data_dir  <ablation>/data/finetune/detect \\
        --checkpoint <ablation>/models/biomedparse_v2.ckpt \\
        --output_dir <ablation>/models/finetuned-detect \\
        --batch_size 4 \\
        --lr 1e-5 \\
        --epochs 10 \\
        --gpus 1

Multi-GPU:
    torchrun --nproc_per_node=4 finetune.py \\
        --data_dir <ablation>/data/finetune/detect --checkpoint <ablation>/models/biomedparse_v2.ckpt \\
        --output_dir <ablation>/models/finetuned-detect --gpus 4
"""

import argparse
import os

import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from _paths import BIOMEDPARSE_DIR, add_biomedparse_to_path, add_medvision_to_path

add_biomedparse_to_path()
add_medvision_to_path()

from medvision_bm.utils.configs import SEED
from src.datasets.biomedparse_dataset import BiomedParseDataset
from src.losses.biomedparse_loss import BiomedParseLossCLS
from src.losses.medsam_loss import MedSamLoss


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def load_checkpoint(model, checkpoint_path):
    """Load biomedparse_v2.ckpt into the model (strips 'model.' prefix, strict=False)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"Loaded checkpoint from {checkpoint_path}")


class BiomedParseFineTuner(L.LightningModule):
    def __init__(self, model, lr, loss_cfg):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = BiomedParseLossCLS(
            loss=MedSamLoss(reduction="none"),
            cls_coeff=loss_cfg["cls_coeff"],
            pos_weight=loss_cfg["pos_weight"],
            edge_coeff=loss_cfg["edge_coeff"],
        )

    def forward(self, batch):
        inputs = {"image": batch["image"].float(), "text": batch["text"]}
        return self.model(inputs, mode="train")

    def training_step(self, batch, _):
        outputs = self(batch)
        loss = self.loss_fn(outputs["predictions"], batch["labels"])
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        outputs = self(batch)
        loss = self.loss_fn(outputs["predictions"], batch["labels"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


def build_model(config_dir):
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.abspath(config_dir), job_name="finetune")
    # edge_queries=4 matches finetune_biomedparse.yaml; required because
    # BiomedParseLossCLS always references `edge_loss` in the total_loss sum.
    cfg = compose(config_name="biomedparse", overrides=["+edge_queries=4"])
    model = hydra.utils.instantiate(cfg, _convert_="object")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory produced by prepare_finetune_data.py "
             "(contains train/, train_mask/, train.json, val/, val_mask/, val.json)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to biomedparse_v2.ckpt")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and Lightning logs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    # Loss coefficients matching finetune_biomedparse.yaml defaults
    parser.add_argument("--cls_coeff", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    # edge_coeff must be >0 when edge_queries>0; BiomedParseLossCLS references
    # `edge_loss` unconditionally in the total_loss sum (bug in original code).
    parser.add_argument("--edge_coeff", type=float, default=1.0)
    parser.add_argument("--save_top_k", type=int, default=-1, help="-1 saves all checkpoints; N saves only the N best by val/loss")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to a .ckpt file to resume training from")
    args = parser.parse_args()

    # Deterministic seeding for reproducibility (project-wide SEED). seed_everything
    # seeds python/numpy/torch and (with workers=True) DataLoader worker RNGs.
    L.seed_everything(SEED, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Datasets ---
    # Use the prepared val/ split as the validation set rather than a random
    # split of training data, so the two sets never overlap.
    train_ds = BiomedParseDataset(root_dir=args.data_dir, split="train")
    val_ds = BiomedParseDataset(root_dir=args.data_dir, split="val")
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --- Model ---
    config_dir = os.path.join(BIOMEDPARSE_DIR, "configs", "model")
    model = build_model(config_dir)
    load_checkpoint(model, args.checkpoint)

    loss_cfg = {
        "cls_coeff": args.cls_coeff,
        "pos_weight": args.pos_weight,
        "edge_coeff": args.edge_coeff,
    }
    module = BiomedParseFineTuner(model=model, lr=args.lr, loss_cfg=loss_cfg)

    # --- Trainer ---
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename="biomedparse_medvision_{epoch:02d}_{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=args.save_top_k,
            save_last=True,
            mode="min",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    strategy = "ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        precision="bf16-mixed",
        strategy=strategy,
        gradient_clip_val=5.0,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
    print(f"Training complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
