#!/bin/bash
#SBATCH --job-name=jupyter_launch
#SBATCH --time=3-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2080rtx:1
#SBATCH --output=./jupyter_log.log
#SBATCH --partition=compsci-gpu

source /home/users/zg78/_conda/bin/activate
conda activate base_torch
jupyter-notebook --ip=0.0.0.0 --port=8886 --no-browser