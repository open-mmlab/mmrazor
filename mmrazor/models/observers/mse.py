# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import MODELS
from .base import BaseObserver

_version_under_1100 = int(torch.__version__.split('.')[1]) < 10


@MODELS.register_module()
class MSEObserver(BaseObserver):

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=-1,
                 is_pot_scale=False,
                 p=2.0,
                 factory_kwargs=None,
                 eps=torch.finfo(torch.float32).eps) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                         ch_axis, is_pot_scale, factory_kwargs, eps)
        self.p = p

    def lp_loss(self, pred, tgt, dim=None):
        """loss function measured in L_p Norm."""
        return (pred - tgt).abs().pow(
            self.p).mean(dim) if dim else (pred -
                                           tgt).abs().pow(self.p).mean()

    def mse(self,
            x: torch.Tensor,
            x_min: torch.Tensor,
            x_max: torch.Tensor,
            iter=80):
        best_score = 1e+10
        best_min, best_max = torch.tensor(
            [1.0], dtype=torch.float), torch.tensor([1.0], dtype=torch.float)
        best_min.copy_(x_min)
        best_max.copy_(x_max)
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_tensor_affine(
                x, scale.item(), int(zero_point.item()), self.quant_min,
                self.quant_max)
            score = self.lp_loss(x_q, x)
            if score < best_score:
                best_score = score
                best_min, best_max = new_min, new_max
        return best_min, best_max

    def mse_perchannel(self,
                       x: torch.Tensor,
                       x_min: torch.Tensor,
                       x_max: torch.Tensor,
                       iter=80,
                       ch_axis=0):
        assert x_min.shape == x_max.shape
        assert ch_axis >= 0, f'{ch_axis}'
        best_score = 1e+10 * torch.ones_like(x_min)
        best_min, best_max = x_min.clone(), x_max.clone()
        reduce_dim = tuple([i for i in range(len(x.shape)) if i != ch_axis])
        for i in range(iter):
            new_min = x_min * (1.0 - (i * 0.01))
            new_max = x_max * (1.0 - (i * 0.01))
            scale, zero_point = self._calculate_qparams(new_min, new_max)
            x_q = torch.fake_quantize_per_channel_affine(
                x, scale,
                zero_point.long() if _version_under_1100 else zero_point,
                ch_axis, self.quant_min, self.quant_max)
            score = self.lp_loss(x_q, x, reduce_dim)
            update_idx = (score < best_score)
            best_score[update_idx] = score[update_idx]
            best_min[update_idx] = new_min[update_idx]
            best_max[update_idx] = new_max[update_idx]
        return best_min, best_max

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val_cur, max_val_cur = self.mse(
                x, min_val_cur, max_val_cur, iter=95)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            min_val_cur, max_val_cur = self.mse_perchannel(
                x, min_val_cur, max_val_cur, iter=80, ch_axis=self.ch_axis)

        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)
        return x


@MODELS.register_module()
class EMAMSEObserver(MSEObserver):

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=-1,
                 is_pot_scale=False,
                 p=2.0,
                 ema_ratio=0.9,
                 factory_kwargs=None,
                 eps=torch.finfo(torch.float32).eps) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                         ch_axis, is_pot_scale, p, factory_kwargs, eps)
        self.ema_ratio = ema_ratio

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.clone().detach().to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val_cur, max_val_cur = self.mse(
                x, min_val_cur, max_val_cur, iter=95)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            x_channel = x.permute(new_axis_list)
            y = torch.flatten(x_channel, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            min_val_cur, max_val_cur = self.mse_perchannel(
                x, min_val_cur, max_val_cur, iter=80, ch_axis=self.ch_axis)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (
                1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (
                1.0 - self.ema_ratio)
        return x
