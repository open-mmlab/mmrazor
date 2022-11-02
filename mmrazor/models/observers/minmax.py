# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS
from .base import BaseObserver


@MODELS.register_module()
class MinMaxObserver(BaseObserver):

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
        super(MinMaxObserver, self).__init__(dtype, qscheme, reduce_range,
                                             quant_min, quant_max, ch_axis,
                                             is_pot_scale, factory_kwargs, eps)
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
        min_val = torch.min(self.min_val, min_val_cur)
        max_val = torch.max(self.max_val, max_val_cur)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        return x
