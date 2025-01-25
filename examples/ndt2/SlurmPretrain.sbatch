#!/bin/bash
#SBATCH -J IBL-NDT2
#SBATCH -A gts-edyer3-fy20phase1
#SBATCH -qinferno
#SBATCH -o /storage/home/hcoda1/5/smahato7/scratch/alex/logs/SlurmPretrain.out
#SBATCH -e /storage/home/hcoda1/5/smahato7/scratch/alex/logs/SlurmPretrain.err
#SBATCH -GH200:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

source /home/hice1/aandre8/scratch/alex/venv/bin/activate
srun python train.py --config-name pretrain_large