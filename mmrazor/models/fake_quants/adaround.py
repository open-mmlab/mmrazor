# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parameter import Parameter

from mmrazor.registry import MODELS
from .base import FakeQuantize

_version_under_1100 = int(torch.__version__.split('.')[1]) < 10


@MODELS.register_module()
class AdaRoundFakeQuantize(FakeQuantize):

    def __init__(self, observer, **observer_kwargs):
        super().__init__(observer, **observer_kwargs)
        self.adaround = False

    def init(self, weight_tensor: torch.Tensor):
        self.adaround = True
        self.observer_enabled[0] = 0
        self.fake_quant_enabled[0] = 1

        # self.soft_targets = False  # delete this
        self.gamma = -0.1
        self.zeta = 1.1
        self.init_alpha(x=weight_tensor.data.clone())

    def init_alpha(self, x: torch.Tensor):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.data.reshape(new_shape)
        else:
            scale = self.scale.data
        x_floor = torch.floor(x / scale)
        rest = (x / scale) - x_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) /
                           (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
        self.alpha = Parameter(alpha)

    def rectified_sigmoid(self):
        """Function to generate rounding mask.

        Args:
            x (torch.Tensor):
            zeta (torch.Tensor):
            gamma (torch.Tensor):
        Returns:
            torch.Tensor:
        """
        return ((self.zeta - self.gamma) * torch.sigmoid(self.alpha) +
                self.gamma).clamp(0, 1)

    def adaround_forward(self, x, hard_value=False):
        if self.ch_axis != -1:
            new_shape = [1] * len(x.shape)
            new_shape[self.ch_axis] = x.shape[self.ch_axis]
            scale = self.scale.reshape(new_shape)
            zero_point = self.zero_point.reshape(new_shape)
        x = torch.floor(x / scale)
        if hard_value:
            x += (self.alpha >= 0).float()
        else:
            x += self.rectified_sigmoid(self.alpha, self.zeta, self.gamma)
        x += zero_point
        x = torch.clamp(x, self.quant_min, self.quant_max)
        x = (x - zero_point) * scale
        return x

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(
                self.zero_point.device)
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            if not self.adaround:
                if self.is_per_channel:
                    X = torch.fake_quantize_per_channel_affine(
                        X, self.scale,
                        self.zero_point.long()
                        if _version_under_1100 else self.zero_point,
                        self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                        X, self.scale.item(), int(self.zero_point.item()),
                        self.quant_min, self.quant_max)
            else:
                if not hasattr(self, 'alpha'):
                    raise NotImplementedError
                X = self.adaround_forward(X)
        return X
