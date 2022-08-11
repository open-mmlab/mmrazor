import torch
from torch.ao.quantization.observer import _ObserverBase
from mmrazor.models.utils import sync_tensor, pot_quantization
from mmrazor.registry import MODELS
from typing import Tuple, Dict, List, Any
# from mmengine.model import BaseModule

class BaseObserver(_ObserverBase):

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        ch_axis=-1,
        is_pot_scale=False,
        factory_kwargs=None) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max, 
            factory_kwargs)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        self.is_pot_scale = is_pot_scale

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        if self.is_pot_scale:
            scale = pot_quantization(scale)
        return scale, zero_point


    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={} ch_axis={} is_pot_scale={}".format(
            self.min_val, self.max_val, self.ch_axis, self.is_pot_scale)

@MODELS.register_module()
class MinMaxObserver(BaseObserver):
    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        ch_axis=-1,
        is_pot_scale=False,
        factory_kwargs=None,
        memoryless=False) -> None:
        super(MinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                             ch_axis, is_pot_scale, factory_kwargs)
        self.memoryless = memoryless
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        elif self.memoryless:
            self.reset_min_max_vals()
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

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float("inf")))
        self.max_val.copy_(torch.tensor(float("-inf")))