#!/bin/bash
#SBATCH -J IBL-NDT2
#SBATCH -o ${LOG_DIR}/SlurmFinetune.out
#SBATCH -e ${LOG_DIR}/SlurmFinetune.err
#SBATCH -GH100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source ${VENV_PATH}
srun python train.py --config-name finetune