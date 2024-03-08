# [WIP] CaPOYO: POYO applied to Calcium Imaging Data

### Datasets
To download and prepare the openscope calcium dataset, run the following inside the 
`project-kirby` directory:
```bash
snakemake --cores 8 gillon_richards_responses_2023
```

### Training CaPOYO
```bash
python train.py --config-name train_openscope_calcium.yaml data_root=/kirby/processed
```