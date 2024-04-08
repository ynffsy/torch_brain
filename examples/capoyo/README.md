# [WIP] CaPOYO: POYO applied to Calcium Imaging Data

### Datasets
To download and prepare the openscope calcium dataset, run the following inside the 
`project-kirby` directory:
```bash
snakemake --cores 8 gillon_richards_responses_2023
```
```bash
snakemake --cores 8 allen_brain_observatory_calcium
```

### Training CaPOYO
```bash
python train.py --config-name train_openscope_calcium.yaml data_root=/kirby/processed
```
```bash
python train.py --config-name train_allen_bo.yaml data_root=/kirby/processed
```