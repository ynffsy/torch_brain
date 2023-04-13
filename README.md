# project-kirby
Poyo!

# Installation
- Python 3.8 (need to test other versions)
- PyTorch 2.0
- CUDA 11.3 - 11.7 


#### Downloading data from dandi
```
pip install dandi
```

##### Preparing the data

Download data from [Dropbox](https://www.dropbox.com/scl/fo/j9wwle1ta0r4hpxqu885n/h?dl=0&rlkey=o6mf1l1y9c5i3npeetwqi1krl).

Put data in `data/` folder. The folder should contain the following files:
```
data/
├── NHPData
│   ├── raw
│   │   ├── AdaptationData
│   │   ├── ReachingData
│   ├── prepare_data.py
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
CUDA_VISIBLE_DEVICES=0,1,... python3 train_perceiver_rotary_multi_session.py
```

Single GPU training:
```
CUDA_VISIBLE_DEVICES=0 python3 train_perceiver_rotary_multi_session.py
```
