# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parameter import Parameter

from mmrazor.registry import MODELS
from .base import FakeQuantize


@MODELS.register_module()
class QDropFakeQuantize(FakeQuantize):

    def __init__(self, observer, **observer_kwargs):
        super().__init__(observer, **observer_kwargs)
        self.scale = Parameter(torch.tensor([1.0], dtype=torch.float))
        self.prob = 1.0

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
            x_orig = X
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max)
            if self.prob < 1.0:
                x_prob = torch.where(torch.rand_like(X) < self.prob, X, x_orig)
                return x_prob
        return X
