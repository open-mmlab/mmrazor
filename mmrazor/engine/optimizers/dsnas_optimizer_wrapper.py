# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch.optim import Optimizer

from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class DsnasOptimWrapper(OptimWrapper):
    """Optimizer wrapper provides a common interface for updating parameters.

    Optimizer wrapper provides a unified interface for single precision
    training and automatic mixed precision training with different hardware.
    OptimWrapper encapsulates optimizer to provide simplified interfaces
    for commonly used training techniques such as gradient accumulative and
    grad clips. ``OptimWrapper`` implements the basic logic of gradient
    accumulation and gradient clipping based on ``torch.optim.Optimizer``.
    The subclasses only need to override some methods to implement the mixed
    precision training. See more information in :class:`AmpOptimWrapper`.

    Args:
        optimizer (Optimizer): Optimizer used to update model parameters.
        accumulative_counts (int): The number of iterations to accumulate
            gradients. The parameters will be updated per
            ``accumulative_counts``.
        clip_grad (dict, optional): If ``clip_grad`` is not None, it will be
            the arguments of ``torch.nn.utils.clip_grad``.

    Note:
        If ``accumulative_counts`` is larger than 1, perform
        :meth:`update_params` under the context of  ``optim_context``
        could avoid unnecessary gradient synchronization.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.

    Note:
        The subclass should ensure that once :meth:`update_params` is called,
        ``_inner_count += 1`` is automatically performed.

    Examples:
        >>> # Config sample of OptimWrapper.
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=1,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> # Use OptimWrapper to update model.
        >>> import torch.nn as nn
        >>> import torch
        >>> from torch.optim import SGD
        >>> from torch.utils.data import DataLoader
        >>> from mmengine.optim import OptimWrapper
        >>>
        >>> model = nn.Linear(1, 1)
        >>> dataset = torch.randn(10, 1, 1)
        >>> dataloader = DataLoader(dataset)
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>>
        >>> for data in dataloader:
        >>>     loss = model(data)
        >>>     optim_wrapper.update_params(loss)
        >>> # Enable gradient accumulation
        >>> optim_wrapper_cfg = dict(
        >>>     type='OptimWrapper',
        >>>     _accumulative_counts=3,
        >>>     clip_grad=dict(max_norm=0.2))
        >>> ddp_model = DistributedDataParallel(model)
        >>> optimizer = SGD(ddp_model.parameters(), lr=0.1)
        >>> optim_wrapper = OptimWrapper(optimizer)
        >>> optim_wrapper.initialize_count_status(0, len(dataloader))
        >>> # If model is a subclass instance of DistributedDataParallel,
        >>> # `optim_context` context manager can avoid unnecessary gradient
        >>> #  synchronize.
        >>> for iter, data in enumerate(dataloader):
        >>>     with optim_wrapper.optim_context(ddp_model):
        >>>         loss = model(data)
        >>>     optim_wrapper.update_params(loss)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        super().__init__(
            optimizer,
            accumulative_counts=accumulative_counts,
            clip_grad=clip_grad,
        )

    def update_params(self,
                      loss: torch.Tensor,
                      retain_graph: bool = False) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
        """
        loss = self.scale_loss(loss)
        self.backward(loss, retain_graph=retain_graph)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step()
            self.zero_grad()

    def backward(self, loss: torch.Tensor, retain_graph: bool = False) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Note:
            If subclasses inherit from ``OptimWrapper`` override
            ``backward``, ``_inner_count +=1`` must be implemented.

        Args:
            loss (torch.Tensor): The loss of current iteration.
        """
        loss.backward(retain_graph=retain_graph)
        self._inner_count += 1
