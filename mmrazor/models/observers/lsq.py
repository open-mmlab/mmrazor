# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.distributed as dist

from mmrazor.registry import MODELS

try:
    from torch.ao.quantization.observer import (MinMaxObserver,
                                                PerChannelMinMaxObserver)
except ImportError:
    from mmrazor.utils import get_placeholder
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')


def sync_tensor(tensor):
    """Synchronize the target tensor during distributed training."""
    if torch.distributed.is_initialized() and tensor.is_cuda:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


class LSQObserverMixIn:
    """A mixin class for LSQObserver which can provide the initialized
    floating-point scale factor."""

    def __init__(self):
        self.tensor_norm = None

    @torch.jit.export
    def _calculate_scale(self):
        """Calculate the initialized floating-point scale factor.

        Each layer of weights and each layer of activations has a distinct step
        size, represented as a fp32 value, initialized to 2<|v|> / sqrt(Q_p),
        computed on either the initial weights values or the first batch of
        activations, respectively.
        """
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        sync_tensor(scale)
        return scale


@MODELS.register_module()
class LSQObserver(MinMaxObserver, LSQObserverMixIn):
    """LSQ observer.

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """

    def __init__(self, *args, **kwargs):
        MinMaxObserver.__init__(self, *args, **kwargs)
        LSQObserverMixIn.__init__(self)

    def forward(self, x_orig):
        """Records the running minimum, maximum and tensor_norm of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        self.tensor_norm = x.abs().mean()
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        _, zero_point = MinMaxObserver.calculate_qparams(self)
        scale = LSQObserverMixIn._calculate_scale(self)
        return scale, zero_point


@MODELS.register_module()
class LSQPerChannelObserver(PerChannelMinMaxObserver, LSQObserverMixIn):
    """LSQ per-channel observer.

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """

    def __init__(self, *args, **kwargs):
        PerChannelMinMaxObserver.__init__(self, *args, **kwargs)
        LSQObserverMixIn.__init__(self)

    def forward(self, x_orig):
        """Records the per-channel running minimum, maximum and tensor_norm of
        ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)

        self.tensor_norm = y.abs().mean(1)

        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        """Calculates the quantization parameters."""
        _, zero_point = PerChannelMinMaxObserver.calculate_qparams(self)
        scale = LSQObserverMixIn._calculate_scale(self)
        return scale, zero_point
