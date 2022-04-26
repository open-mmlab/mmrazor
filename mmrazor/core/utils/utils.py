# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from torch import distributed as dist


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return dist.is_available() and dist.is_initialized()


def get_default_group() -> Optional[dist.ProcessGroup]:
    """Return default process group."""

    return dist.distributed_c10d._get_default_group()


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.
    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return dist.get_rank(group)
    else:
        return 0


def get_backend(group: Optional[dist.ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.
    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.
    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return dist.get_backend(group)
    else:
        return None


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return dist.get_world_size(group)
    else:
        return 1
