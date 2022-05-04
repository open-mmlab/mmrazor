# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import warnings
from typing import Any, List, Optional, Tuple

import torch
from mmcv.utils import TORCH_VERSION, digit_version
from torch import Tensor
from torch import distributed as dist

from .utils import get_backend, get_default_group, get_rank, get_world_size


def _object_to_tensor(obj: Any) -> Tuple[Tensor, Tensor]:
    """Serialize picklable python object to tensor."""
    byte_storage = torch.ByteStorage.from_buffer(pickle.dumps(obj))
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor
    # and specifying dtype. Otherwise, it will cause 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor: Tensor, tensor_size: int) -> Any:
    """Deserialize tensor to picklable python object."""
    buf = tensor.cpu().numpy().tobytes()[:tensor_size]
    return pickle.loads(buf)


def _broadcast_object_list(object_list: List[Any],
                           src: int = 0,
                           group: Optional[dist.ProcessGroup] = None) -> None:
    """Broadcast picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is ``None`` by default.
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In
    # the case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == dist.Backend.NCCL
    current_device = torch.device('cpu')
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in
        # docstring. We cannot simply use my_rank since rank == device is
        # not necessarily true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(),
            dtype=torch.uint8,
        )

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)
    dist.broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device('cpu'):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


def broadcast_object_list(data: List[Any],
                          src: int = 0,
                          group: Optional[dist.ProcessGroup] = None) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.
    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.
    Note:
        Calling ``broadcast_object_list`` in non-distributed environment does
        nothing.
    Args:
        data (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank
            will be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.
    Note:
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.
    Examples:
        >>> import torch
        >>> import mmrazor.core.utils as dist
        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]
        >>> dist.broadcast_object_list(data)
        >>> data
        ['foo', 12, {1: 2}]
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     data = ["foo", 12, {1: 2}]  # any picklable object
        >>> else:
        >>>     data = [None, None, None]
        >>> dist.broadcast_object_list(data)
        >>> data
        ["foo", 12, {1: 2}]  # Rank 0
        ["foo", 12, {1: 2}]  # Rank 1
    """
    warnings.warn(
        '`broadcast_object_list` is now without return value, '
        'and it\'s input parameters are: `data`,`src` and '
        '`group`, but its function is similar to the old\'s', UserWarning)
    assert isinstance(data, list)

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
            dist.broadcast_object_list(data, src, group)
        else:
            _broadcast_object_list(data, src, group)
