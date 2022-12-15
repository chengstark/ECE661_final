#!/bin/bash
#SBATCH --job-name=run_simclr_base
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2080rtx:1

source /home/users/zg78/_conda/bin/activate
conda activate base_torch

nvidia-smi

echo $1

python supervised_baseline.py \
    --label_percentage $1

