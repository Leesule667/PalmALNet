'''
Build the PalmALNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import Net
# from timm.models.registry import register_model

Net_config = {
        'img_size': 224,
        'patch_size': 8,
        'embed_dim': [64, 256, 384, 512],
        'depth': [8,10,12,14],
        'num_heads': [8,8,8,8],
        'window_size': [7, 7, 7, 7],
        'kernels': [7,7, 5,5, 3,3,3, 3],
        'ga': 16
    }
def PalmALNet(num_classes=1000,  model_cfg = Net_config):
    model = Net(num_classes=num_classes,  **model_cfg)
    return model


