# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import NORM_LAYERS
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables.base_mutable import BaseMutable
from ..base import ChannelDynamicOP


class _DynamicBatchNorm(_BatchNorm, ChannelDynamicOP):
    """Applies Batch Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        num_features_cfg (Dict): Config related to `num_features`.
    """
    accepted_mutables = {'mutable_num_features'}
    batch_norm_type: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_num_features: Optional[BaseMutable] = None

    def mutate_num_features(self, mutable_num_features: BaseMutable) -> None:
        if not self.affine and not self.track_running_stats:
            raise RuntimeError(
                'num_features can not be mutated if both `affine` and '
                '`tracking_running_stats` are False')

        self.check_mutable_channels(mutable_num_features)
        self.mutable_num_features = mutable_num_features

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable `num_features`."""
        return self.mutable_num_features

    def _get_out_mask(self) -> Optional[torch.Tensor]:
        if self.affine:
            refer_tensor = self.weight
        elif self.track_running_stats:
            refer_tensor = self.running_mean
        else:
            return None

        if self.mutable_num_features is not None:
            out_mask = self.mutable_num_features.current_mask.to(
                refer_tensor.device)
        else:
            out_mask = torch.ones_like(refer_tensor).bool()

        return out_mask

    def _get_dynamic_params(
        self
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
               Optional[Tensor]]:
        out_mask = self._get_out_mask()

        if self.affine:
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            running_mean = self.running_mean[out_mask] \
                if not self.training or self.track_running_stats else None
            running_var = self.running_var[out_mask] \
                if not self.training or self.track_running_stats else None
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return running_mean, running_var, weight, bias

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = \
                    self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)

        running_mean, running_var, weight, bias = self._get_dynamic_params()

        out = F.batch_norm(input, running_mean, running_var, weight, bias,
                           bn_training, exponential_average_factor, self.eps)

        if self.training and self.track_running_stats:
            out_mask = self._get_out_mask()
            self.running_mean.masked_scatter_(out_mask, running_mean)
            self.running_var.masked_scatter_(out_mask, running_var)

        return out

    def to_static_op(self) -> nn.Module:
        self.check_if_mutables_fixed()

        running_mean, running_var, weight, bias = self._get_dynamic_params()
        if self.mutable_num_features is not None:
            num_features = self.mutable_num_features.current_mask.sum().item()
        else:
            num_features = self.num_features

        static_bn = getattr(nn, self.batch_norm_type)(
            num_features=num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats)

        if running_mean is not None:
            static_bn.running_mean.copy_(running_mean)
        if running_var is not None:
            static_bn.running_var.copy_(running_var)
        if weight is not None:
            static_bn.weight = nn.Parameter(weight)
        if bias is not None:
            static_bn.bias = nn.Parameter(bias)

        return static_bn


@NORM_LAYERS.register_module()
class DynamicBatchNorm1d(_DynamicBatchNorm):
    batch_norm_type: str = 'BatchNorm1d'

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


@NORM_LAYERS.register_module()
class DynamicBatchNorm2d(_DynamicBatchNorm):
    batch_norm_type: str = 'BatchNorm2d'

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


@NORM_LAYERS.register_module()
class DynamicBatchNorm3d(_DynamicBatchNorm):
    batch_norm_type: str = 'BatchNorm3d'

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))
