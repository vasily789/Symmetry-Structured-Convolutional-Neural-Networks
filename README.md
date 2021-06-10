## Symmetry Structured Convolutional Neural Networks

The code is tested on a single NVIDIA® QUADRO® P5000 with PyTorch 1.1.0 and Python 3.6.9.


## Requirements
```bash
pip install -r requirements.txt
```

## Implementation
SCosRec convolutional kernels use the half Normal Glorot initialization.

## Training SCosRec models from scratch
To train SCosRec model on `ml1m` dataset with learning rate decay (best results): 
```
python train.py --dataset=ml1m --lr_decay True --model_type=scnn
```
Note: You should be able to obtain MAPs of ~0.1970 with the above settings.

To train SCosRec model on `gowalla` dataset with learning rate decay (best results):
```
python train.py --dataset=gowalla --d=100 --fc_dim=50 --l2=1e-6 --lr_decay True --model_type=scnn
```
Note: You should be able to obtain MAPs of ~0.1006 with the above settings.

## Acknowledgments
This project is built on top of [CosRec](https://github.com/zzxslp/CosRec).
