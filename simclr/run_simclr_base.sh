#!/bin/bash
#SBATCH --job-name=run_simclr_base
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2080rtx:1

source /home/users/zg78/_conda/bin/activate
conda activate base_torch

nvidia-smi

python simclr_base.py \
    --temp 0.5 \
    --batch_size 256 \
    --lr 0.5 \
    --epochs 200
