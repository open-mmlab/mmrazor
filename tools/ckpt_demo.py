# Copyright (c) OpenMMLab. All rights reserved.
import torch

ckpt_path = '/mnt/lustre/humu/experiments/adaround/quantizied.pth'
# ckpt_path =
# '/mnt/petrelfs/humu/share/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
# ckpt_path = '/tmp/humu/resnet18_uniform8/checkpoint.pth.tar'
# ckpt_path = '/tmp/humu/resnet18_uniform8/quantized_checkpoint.pth.tar'

state_dict = torch.load(ckpt_path, map_location='cpu')

for k, v in state_dict['state_dict'].items():
    print(k)
