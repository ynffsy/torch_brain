#!/bin/bash
#SBATCH -J IBL-NDT2
#SBATCH -o /home/hice1/aandre8/scratch/alex/logs/SlurmPretrain.out
#SBATCH -e /home/hice1/aandre8/scratch/alex/logs/SlurmPretrain.err
#SBATCH -GH100:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source /home/hice1/aandre8/scratch/alex/venv/bin/activate
srun python train.py --config-name pretrain_large