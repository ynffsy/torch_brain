#!/bin/bash
#SBATCH --job-name=multi_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --partition=long

dataset=openscope_calcium


module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2

conda activate poyo

# wandb credentials
set -a
source .env
set +a

# Uncompress the data to SLURM_TMPDIR
#snakemake --forceall --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c1 openscope_calcium_unfreeze
snakemake  --rerun-triggers=mtime --config tmp_dir=$SLURM_TMPDIR -c1 openscope_calcium_unfreeze

# Important info for parallel GPU processing
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))
export NCCL_BLOCKING_WAIT=1

echo $MASTER_ADDR:$MASTER_PORT

nvidia-smi

# Run experiments
pwd
which python
srun python train.py \
        data_root=$SLURM_TMPDIR/uncompressed/ \
        train_datasets=$dataset \
        val_datasets=$dataset \
        name=multi_sess_poyo_1 \
        epochs=1500 \
        batch_size=64 \
        nodes=1 \
        gpus=2