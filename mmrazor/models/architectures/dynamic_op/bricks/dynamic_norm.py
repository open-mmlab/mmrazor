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
    """Dynamic BatchNormxd OP.

    Note:
        Arguments for ``__init__`` of ``DynamicBatchNormxd`` is totally same as
        :obj:`torch.nn.BatchNormxd`.

    Attributes:
        mutable_num_features (BaseMutable, optional): Mutable for controlling
            ``num_features``.
        batch_norm_type (str): Type of BatchNorm.
    """
    accepted_mutables = {'mutable_num_features'}
    batch_norm_type: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_num_features: Optional[BaseMutable] = None

    def mutate_num_features(self, mutable_num_features: BaseMutable) -> None:
        """Mutate ``num_features`` with given mutable.

        Args:
            mutable_num_features (BaseMutable): Mutable for controlling
                ``num_features``.

        Raises:
            RuntimeError: Error if both ``affine`` and
                ``tracking_running_stats`` are False.
            ValueError: Error if size of mask if not same as ``num_features``.
        """
        if not self.affine and not self.track_running_stats:
            raise RuntimeError(
                'num_features can not be mutated if both `affine` and '
                '`tracking_running_stats` are False')

        self.check_mutable_channels(mutable_num_features)
        mask_size = mutable_num_features.current_mask.size(0)
        if mask_size != self.num_features:
            raise ValueError(
                f'Expect mask size of mutable to be {self.num_features} as '
                f'`num_features`, but got: {mask_size}.')

        self.mutable_num_features = mutable_num_features

    @property
    def mutable_in(self) -> Optional[BaseMutable]:
        """Mutable for controlling ``num_features``."""
        return self.mutable_num_features

    @property
    def mutable_out(self) -> Optional[BaseMutable]:
        """Mutable for controlling ``num_features``."""
        return self.mutable_num_features

    def _get_num_features_mask(self) -> Optional[torch.Tensor]:
        """Get mask of ``num_features``"""
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
        """Get dynamic parameters that will be used in forward process.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor],
                Optional[Tensor]]: Sliced running_mean, running_var, weight and
                bias.
        """
        out_mask = self._get_num_features_mask()

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
        """Forward of dynamic BatchNormxd OP."""
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

        # copy changed running statistics
        if self.training and self.track_running_stats:
            out_mask = self._get_num_features_mask()
            self.running_mean.masked_scatter_(out_mask, running_mean)
            self.running_var.masked_scatter_(out_mask, running_var)

        return out

    def to_static_op(self) -> nn.Module:
        """Convert dynamic BatchNormxd to :obj:`torch.nn.BatchNormxd`.

        Returns:
            torch.nn.BatchNormxd: :obj:`torch.nn.BatchNormxd` with sliced
                parameters.
        """
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
    """Dynamic BatchNorm1d OP."""
    batch_norm_type: str = 'BatchNorm1d'

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


@NORM_LAYERS.register_module()
class DynamicBatchNorm2d(_DynamicBatchNorm):
    """Dynamic BatchNorm2d OP."""
    batch_norm_type: str = 'BatchNorm2d'

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


@NORM_LAYERS.register_module()
class DynamicBatchNorm3d(_DynamicBatchNorm):
    """Dynamic BatchNorm3d OP."""
    batch_norm_type: str = 'BatchNorm3d'

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))
