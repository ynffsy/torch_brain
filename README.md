# project-kirby
Poyo!

Multi-GPU training:
```
CUDA_VISIBLE_DEVICES=0,1,... python train_perceiver.py --ddp_port=<some-open-port>
```
