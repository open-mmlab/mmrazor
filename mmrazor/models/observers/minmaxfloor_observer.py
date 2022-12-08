# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.ao.quantization.observer import UniformQuantizationObserverBase

from mmrazor.registry import MODELS
from ..utils import _is_float_qparams, _is_symmetric_quant, sync_tensor


@MODELS.register_module()
class MinMaxFloorObserver(UniformQuantizationObserverBase):
    """Calculate minmax of whole calibration dataset with floor but round.

    Args:
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified,
            it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified,
            it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to
            `torch.finfo(torch.float32).eps`.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 factory_kwargs=None,
                 eps=torch.finfo(torch.float32).eps) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                         factory_kwargs, eps)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        if (self.qscheme == torch.per_tensor_symmetric and self.reduce_range
                and self.dtype == torch.quint8):
            raise NotImplementedError('Cannot reduce range for symmetric \
                                       quantization for quint8')

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        """Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        if not _is_symmetric_quant(self.qscheme) and not _is_float_qparams(
                self.qscheme):
            scale = (self.max_val - self.min_val) / float(self.quant_max -
                                                          self.quant_min)
            scale = torch.max(scale, self.eps)
            zero_point = self.quant_min - torch.floor(self.min_val / scale).to(
                torch.int)
            zero_point = torch.clamp(zero_point, self.quant_min,
                                     self.quant_max)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point

    @torch.jit.export
    def extra_repr(self) -> str:
        return 'min_val={}, max_val={}'.format(self.min_val, self.max_val)

    @torch.jit.export
    def reset_min_max_vals(self) -> None:
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float('inf')))
        self.max_val.copy_(torch.tensor(float('-inf')))
