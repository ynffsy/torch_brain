# project-kirby
Poyo!

# Installation
- Python 3.8 (need to test other versions)
- PyTorch 2.0
- CUDA 11.3 - 11.7 

### Add this package to your python path
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
Multi-GPU training:
```
CUDA_VISIBLE_DEVICES=0,1,... python3 train_multi_session.py --batch_size 256
```

Single GPU training:
```
CUDA_VISIBLE_DEVICES=0 python3 train_multi_session.py
```

# Testing
To run testing:
```
python3 test.py --ckpt_path runs/Apr20_22-44-40_bmedyer-gpu3/perceiver-chewie-latest.pt
```

# Finetuning
To finetune a model:
```
python3 finetune.py --ckpt_path weights/perceiver-chewie-latest.pt --eval_epochs 100
```
