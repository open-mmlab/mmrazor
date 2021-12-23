# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
import tempfile

import mmcv.fileio
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def broadcast_object_list(object_list, src=0):
    """Broadcasts picklable objects in ``object_list`` to the whole group.

    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the src rank will be
            broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
    """
    my_rank, _ = get_dist_info()

    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
    object_list_return = list()
    if my_rank == src:
        mmcv.mkdir_or_exist('.dist_broadcast')
        tmpdir = tempfile.mkdtemp(dir='.dist_broadcast')
        mmcv.dump(object_list, osp.join(tmpdir, 'object_list.pkl'))
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
        dir_tensor[:len(tmpdir)] = tmpdir

    dist.broadcast(dir_tensor, src)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()

    if my_rank != src:
        object_list_return = mmcv.load(osp.join(tmpdir, 'object_list.pkl'))

    dist.barrier()
    if my_rank == src:
        shutil.rmtree(tmpdir)
        object_list_return = object_list

    return object_list_return
