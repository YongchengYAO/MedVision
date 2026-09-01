#!/bin/bash
# Track B · step 2 — fine-tune BiomedParse v2 on the MedVision detection set.
#
# PyTorch Lightning, bf16-mixed, DDP via torchrun when N_GPUS > 1. Checkpoints
# land in models/finetuned-detect/ as biomedparse_medvision_epoch=XX_val_loss=Y.ckpt
# plus last.ckpt (safe resume target).
#
# Usage:
#   bash scripts/finetune/2_finetune.sh
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/finetune/2_finetune.sh
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"

DATA_DIR="${ABLATION_DIR}/data/finetune/detect"
OUTPUT_DIR="${ABLATION_DIR}/models/finetuned-detect"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS=2

BATCH_SIZE=4        # per GPU
LR=1e-5
EPOCHS=10
NUM_WORKERS=4       # DataLoader CPU workers
CLS_COEFF=1.0       # loss coefficients (match upstream finetune_biomedparse.yaml)
POS_WEIGHT=3.0
EDGE_COEFF=1.0
SAVE_TOP_K=-1       # -1 keeps every epoch; N keeps only the N best by val_loss
RESUME_FROM_CHECKPOINT=""   # e.g. "${OUTPUT_DIR}/last.ckpt"

ensure_pretrained_ckpt

TRAIN_ARGS=(
    --data_dir    "${DATA_DIR}"
    --checkpoint  "${PRETRAINED_CKPT}"
    --output_dir  "${OUTPUT_DIR}"
    --batch_size  ${BATCH_SIZE}
    --lr          ${LR}
    --epochs      ${EPOCHS}
    --gpus        ${N_GPUS}
    --num_workers ${NUM_WORKERS}
    --cls_coeff   ${CLS_COEFF}
    --pos_weight  ${POS_WEIGHT}
    --edge_coeff  ${EDGE_COEFF}
    --save_top_k  ${SAVE_TOP_K}
)
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

export CUDA_VISIBLE_DEVICES
if [ "${N_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node=${N_GPUS} "${ABLATION_DIR}/src/finetune.py" "${TRAIN_ARGS[@]}"
else
    python "${ABLATION_DIR}/src/finetune.py" "${TRAIN_ARGS[@]}"
fi
