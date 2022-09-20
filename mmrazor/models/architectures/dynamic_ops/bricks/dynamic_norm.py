# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from .dynamic_mixins import DynamicBatchNormMixin


class _DynamicBatchNorm(_BatchNorm, DynamicBatchNormMixin):
    """Dynamic BatchNormxd OP.

    Note:
        Arguments for ``__init__`` of ``DynamicBatchNormxd`` is totally same as
        :obj:`torch.nn.BatchNormxd`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `num_features`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    accepted_mutable_attrs = {'num_features'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: _BatchNorm):
        """Convert a _BatchNorm module to a DynamicBatchNorm.

        Args:
            module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        """
        dynamic_bn = cls(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats)

        return dynamic_bn

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

        running_mean, running_var, weight, bias = self.get_dynamic_params()

        out = F.batch_norm(input, running_mean, running_var, weight, bias,
                           bn_training, exponential_average_factor, self.eps)

        # copy changed running statistics
        if self.training and self.track_running_stats:
            out_mask = self._get_num_features_mask()
            self.running_mean.masked_scatter_(out_mask, running_mean)
            self.running_var.masked_scatter_(out_mask, running_var)

        return out


@MODELS.register_module()
class DynamicBatchNorm1d(_DynamicBatchNorm):
    """Dynamic BatchNorm1d OP."""

    @property
    def static_op_factory(self):
        return nn.BatchNorm1d

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(
                input.dim()))


@MODELS.register_module()
class DynamicBatchNorm2d(_DynamicBatchNorm):
    """Dynamic BatchNorm2d OP."""

    @property
    def static_op_factory(self):
        return nn.BatchNorm2d

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))


@MODELS.register_module()
class DynamicBatchNorm3d(_DynamicBatchNorm):
    """Dynamic BatchNorm3d OP."""

    @property
    def static_op_factory(self):
        return nn.BatchNorm3d

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))


class SwitchableBatchNorm2d(DynamicBatchNorm2d):
    """Employs independent batch normalization for different switches in a
    slimmable network.

    To train slimmable networks, ``SwitchableBatchNorm2d`` privatizes all
    batch normalization layers for each switch in a slimmable network.
    Compared with the naive training approach, it solves the problem of feature
    aggregation inconsistency between different switches by independently
    normalizing the feature mean and variance during testing.

    Args:
        module_name (str): Name of this `SwitchableBatchNorm2d`.
        num_features_cfg (Dict): Config related to `num_features`.
        eps (float): A value added to the denominator for numerical stability.
            Same as that in :obj:`torch.nn._BatchNorm`. Default: 1e-5
        momentum (float): The value used for the running_mean and running_var
            computation. Can be set to None for cumulative moving average
            (i.e. simple average). Same as that in :obj:`torch.nn._BatchNorm`.
            Default: 0.1
        affine (bool): A boolean value that when set to True, this module has
            learnable affine parameters. Same as that in
            :obj:`torch.nn._BatchNorm`. Default: True
        track_running_stats (bool): A boolean value that when set to True, this
            module tracks the running mean and variance, and when set to False,
            this module does not track such statistics, and initializes
            statistics buffers running_mean and running_var as None. When these
            buffers are None, this module always uses batch statistics. in both
            training and eval modes. Same as that in
            :obj:`torch.nn._BatchNorm`. Default: True
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_bn = nn.ModuleDict()

    def init_candidates(self, candidates: List):
        assert len(self.candidate_bn) == 0
        self._check_candidates(candidates)
        for num in candidates:
            self.candidate_bn[str(num)] = nn.BatchNorm2d(num)

    def forward(self, input: Tensor) -> Tensor:
        choice_num = self.activated_channel_num()
        if choice_num == self.num_features:
            return super().forward(input)
        else:
            assert str(choice_num) in self.candidate_bn
            return self.candidate_bn[str(choice_num)](input)

    def to_static_op(self: _BatchNorm) -> nn.Module:
        choice_num = self.activated_channel_num()
        if choice_num == self.num_features:
            return super().to_static_op()
        else:
            assert str(choice_num) in self.candidate_bn
            return self.candidate_bn[str(choice_num)]

    # private methods

    def activated_channel_num(self):
        mask = self._get_num_features_mask()
        choice_num = (mask == 1).sum().item()
        return choice_num

    def _check_candidates(self, candidates: List):
        for value in candidates:
            assert isinstance(value, int)
            assert 0 < value <= self.num_features

    @property
    def static_op_factory(self):
        return nn.BatchNorm2d
