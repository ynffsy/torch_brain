# project-kirby
Poyo!

# Installation

**#todo** add more specific instructions, make sure that requirements.txt is up to date.
- Python 3.9 (also requires python3.9-dev)
- PyTorch 2.0.0
- CUDA 11.3 - 11.7 
- xformers is optional, but recommended for training with memory efficient attention

Add this package to your python path
```
export PYTHONPATH=$PYTHONPATH:/path/to/project-kirby
```


##### Preparing the data

Download data from [Dropbox](https://www.dropbox.com/scl/fo/j9wwle1ta0r4hpxqu885n/h?dl=0&rlkey=o6mf1l1y9c5i3npeetwqi1krl).

Put data in `data/` folder. The folder should contain the following files:
```
data/
├── Perich, Miller
│   ├── raw
│   │   ├── AdaptationData
│   │   ├── ReachingData
│   ├── prepare_data.py
|   ├── all.txt
|   ├── chewie.txt
│   └── processed
├── nlb_maze
```

Process the data in each folder by running:
```
python3 prepare_data.py
```
This should create a `processed` folder in each folder.

# Training
To train POYO-1:
```bash
python3 matt_model/train_poyo_1.py --checkpoint_epochs 1
```
Everything is logged to tensorboard. Checkpoints are saved every `checkpoint_epochs`, and the latest checkpoint is saved with a `-latest.pt` suffix.

# Evaluation
Testing is done on the CPU, which is super slow, but was convenient so that training could be done using all GPUs. **#todo** update scripts to use GPU for testing.
```
python3 evaluate_poyo_1.py --ckpt_path runs/May11_14-06-51_bmedyer-gpu3/spike-1-matt-latest.pt --tensorboard runs/May11_14-06-51_bmedyer-gpu3
```

# Finetuning
To finetune a model: **#todo** description needs to be updated
```
python3 finetune.py --ckpt_path weights/perceiver-chewie-latest.pt --eval_epochs 100 --base_lr 1e-3 --num_samples 32 --batch_size 32
```


# todo
- [ ] Memory efficient attention is only used for self-attention. Need to implement for cross-attention.
- [ ] Dataloading needs to be streamlined to accompodate for multiple tasks.
- [ ] Add ability to bin the data + add baselines 
- [ ] Merge branch that implements hyperparameter tuning (with raytune)
