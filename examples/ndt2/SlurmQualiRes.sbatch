#!/bin/bash
#SBATCH -J IBL-NDT2
#SBATCH -o ${LOG_DIR}/SlurmEval.out
#SBATCH -e ${LOG_DIR}/SlurmEval.err
#SBATCH -GA100:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=00:10:00


source ${VENV_PATH}

srun python eval_ndt2_single_task.py --eid 'db4df448-e449-4a6f-a0e7-288711e7a75a' --behavior wheel --ckpt_postfix '_long_finetune_5e-5'
srun python eval_ndt2_single_task.py --eid 'db4df448-e449-4a6f-a0e7-288711e7a75a' --behavior whisker --ckpt_postfix '_long_finetune_4e-4'
