# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch

from mmrazor.registry import MODELS
from ..utils import _is_symmetric_quant, pot_quantization, sync_tensor
from .base import BaseObserver


@MODELS.register_module()
class LSQObserver(BaseObserver):
    """LSQ observer."""

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=-1,
                 is_pot_scale=False,
                 factory_kwargs=None,
                 eps=torch.finfo(torch.float32).eps) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                         ch_axis, is_pot_scale, factory_kwargs, eps)

        self.tensor_norm = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.tensor_norm = x.abs().mean()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.tensor_norm = y.abs().mean(1)
            self.min_val, self.max_val = torch._aminmax(y, 1)

        return x

    def calculate_qparams(self):
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        zero_point = torch.zeros_like(self.tensor_norm)
        sync_tensor(scale)
        sync_tensor(zero_point)
        if self.pot_scale:
            scale = pot_quantization(scale)
        if not _is_symmetric_quant(self.qscheme):
            zero_point = self.quant_min - torch.round(self.min_val / scale)
        return scale, zero_point
