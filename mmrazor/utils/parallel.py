# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import warnings

import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def _serialize_to_tensor(data, group=None):
    device = torch.cuda.current_device()
    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        buffer_size = len(buffer) / (1024**3)
        rank, _ = get_dist_info()
        warnings.warn('Rank {rank} trying to all-gather'
                      f'{buffer_size:.2f} GB of data on device {device}')
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def broadcast_object(obj, group=None):
    """make sure obj is picklable."""
    _, world_size = get_dist_info()
    if world_size == 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj).cuda()
    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()
    dist.broadcast(numel, 0)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    dist.broadcast(serialized_tensor, 0)
    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj
