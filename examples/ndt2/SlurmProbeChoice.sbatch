#!/bin/bash
#SBATCH -J IBL-NDT2
#SBATCH -o ${LOG_DIR}/SlurmProbe.out
#SBATCH -e ${LOG_DIR}/SlurmProbe.err
#SBATCH -GA100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00

source ${VENV_PATH}
srun python train.py --config-name probe_choice
