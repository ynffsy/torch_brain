#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=main
#
#set -e
dataset=allen_brain_observatory_calcium

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

conda activate poyo

# wandb credentials
set -a
source .env
set +a

# Uncompress the data to SLURM_TMPDIR single node
snakemake --rerun-triggers=mtime -c1 allen_brain_observatory_calcium_unfreeze
nvidia-smi

srun python train.py \
        data_root=$SLURM_TMPDIR/uncompressed/ \
        train_datasets=$dataset \
        val_datasets=$dataset \
        name=allen_bo_test \
        epochs=1000 