#!/bin/bash

set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openood

REPO=   # change your own path
cd $REPO

BACKBONE=${1:-"dinov2_vitb14"}
DINOSAUR_CKPT=${2:-"dinov2_checkpoint_epoch_199.pt"}

echo "Using backbone: $BACKBONE"
echo "Using dinosaur checkpoint: $DINOSAUR_CKPT"

# ---- Stage A: Train -------------------------------------------------------
echo "====== Stage A: Training OCONet ======"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
           configs/preprocessors/base_preprocessor.yml \
           configs/networks/oco_net.yml \
           configs/pipelines/train/train_oco.yml \
  --dataset.image_size 224 \
  --num_gpus 4 \
  --ood_dataset.image_size 224 \
  --network.backbone "$BACKBONE" \
  --network.dinosaur_checkpoint "$DINOSAUR_CKPT" \
  --merge_option merge \
  --seed 0

RESULT_DIR="results/imagenet_oco_net_oco_e20_lr0.0004_default/s0"

BEST_CKPT=$(ls -t ${RESULT_DIR}/best_epoch*.ckpt | head -n 1)

if [ -z "$BEST_CKPT" ]; then
    echo "Error: Could not find best_epoch*.ckpt in $RESULT_DIR"
    exit 1
fi

echo "Found best checkpoint: $BEST_CKPT"

# ---- Stage B: Build F_train -----------------------------------------------
echo ""
echo "====== Stage B: Building F_train ======"
python scripts/ood/oco/build_f_train.py \
  --config configs/datasets/imagenet/imagenet.yml \
           configs/preprocessors/base_preprocessor.yml \
           configs/networks/oco_net.yml \
           configs/pipelines/train/train_oco.yml \
  --network.pretrained True \
  --network.checkpoint $BEST_CKPT \
  --network.backbone "$BACKBONE" \
  --network.dinosaur_checkpoint "$DINOSAUR_CKPT" \
  --merge_option merge \
  --output ./results/oco_f_train.pkl

# ---- Stage C: OOD Evaluation ----------------------------------------------
echo ""
echo "====== Stage C1: OOD Evaluation ======"
python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
           configs/datasets/imagenet/imagenet_ood.yml \
           configs/preprocessors/base_preprocessor.yml \
           configs/networks/oco_net.yml \
           configs/pipelines/test/test_ood.yml \
           configs/postprocessors/oco.yml \
  --dataset.image_size 224 \
  --ood_dataset.image_size 224 \
  --network.pretrained True \
  --network.checkpoint $BEST_CKPT \
  --network.backbone "$BACKBONE" \
  --network.dinosaur_checkpoint "$DINOSAUR_CKPT" \
  --postprocessor.postprocessor_args.f_train_cache ./results/oco_f_train.pkl \
  --merge_option merge


echo ""
echo "====== Stage C2: FSOOD Evaluation ======"
python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
           configs/datasets/imagenet/imagenet_fsood.yml \
           configs/preprocessors/base_preprocessor.yml \
           configs/networks/oco_net.yml \
           configs/pipelines/test/test_fsood.yml \
           configs/postprocessors/oco.yml \
  --dataset.image_size 224 \
  --ood_dataset.image_size 224 \
  --network.pretrained True \
  --network.checkpoint $BEST_CKPT \
  --network.backbone "$BACKBONE" \
  --network.dinosaur_checkpoint "$DINOSAUR_CKPT" \
  --postprocessor.postprocessor_args.f_train_cache ./results/oco_f_train.pkl \
  --merge_option merge

echo ""
echo "====== Done! ======"
