# Running on the mila cluster

Mila is a SLURM-based cluster. `run.sh` gives an example one-node pipeline, `run.sh` 
`run_parallel.sh` gives an example single-node, multi-GPU pipeline, `run_multi_node.sh`, 
a multi-node multi-GPU pipeline. In practice, multi-node, multi-GPU runs are only 
feasible during deadtimes during conferences, but still, it works!

## Datasets

We keep a canonical version of the raw data in the common location at 
`/network/projects/neuro-galaxy/raw`. This prevents having to download the data 
multiple times, which could easily take a day. Processed, compressed data is stored in
your personal scratch folder, which prevents undefined behaviour when multiple people 
are modifying the same `process_data.py` pipeline.

Because mila is SLURM-based, data is first copied to the local node (SLURM_TMPDIR), 
then processed in jobs. Because the file system is distributed and doesn't like to deal 
with small files, we use tarballs compressed with lz4, which is a ridiculously fast 
compression algorithm.

## Environment

Set up a conda environment with the right packages, as defined in `requirements.txt`.

## Partitions

Mila has a number of partitions, `unkillable`, `short-unkillable`, `main` and `long`. 
Use 1-GPU `unkillable` jobs for debugging. Run a 4-GPU job on `short-unkillable` to get 
very quick results (3 hours max, but equivalent to 12 hours of a 1 GPU job). Use the 
main and long partitions for longer jobs.

[Reference](https://docs.mila.quebec/Userguide.html#partitioning)

## wandb credentials

Store them in `.env` in the root of the project. This file is ignored by git. It should
look like:

```
WANDB_PROJECT=poyo
WANDB_ENTITY=neuro-galaxy
WANDB_API_KEY=<secret-key>
```

Get the API key from the wandb website.