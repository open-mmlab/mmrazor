# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from mmrazor.models.mutables.mutable_channel import MutableChannel
from mmrazor.registry import MODELS
from ...mutables import MutableManageMixIn
from .base import MUTABLE_CFGS_TYPE, ChannelDynamicOP


class DynamicConv2d(nn.Conv2d, ChannelDynamicOP):
    """Applies a 2D convolution over an input signal composed of several input
    planes according to the `mutable_in_channels` and `mutable_out_channels`
    dynamically.

    Args:
        in_channels_cfg (Dict): Config related to `in_channels`.
        out_channels_cfg (Dict): Config related to `out_channels`.
    """

    accepted_mutable_keys = {'in_channels', 'out_channels', 'kernel_size'}

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **conv_kwargs) -> None:
        super().__init__(**conv_kwargs)

        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_channels_mutable(mutable_cfgs)
        self._register_kernel_size_mutable(mutable_cfgs)

        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'

    def _register_channels_mutable(self,
                                   mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'in_channels' or 'out_channels' in mutable_cfgs:
            assert 'in_channels' in mutable_cfgs and \
                'out_channels' in mutable_cfgs, \
                'both `in_channels` and `out_channels` ' \
                'should be contained in `mutable_cfgs`'
            in_channels_cfg = copy.deepcopy(mutable_cfgs['in_channels'])
            in_channels_cfg.update(num_channels=self.in_channels)
            self.in_channels_mutable = MODELS.build(in_channels_cfg)

            out_channels_cfg = copy.deepcopy(mutable_cfgs['out_channels'])
            out_channels_cfg.update(dict(num_channels=self.out_channels))
            self.out_channels_mutable = MODELS.build(out_channels_cfg)

            assert isinstance(self.in_channels_mutable, MutableChannel)
            assert isinstance(self.out_channels_mutable, MutableChannel)
        else:
            self.register_parameter('in_channels_mutable', None)
            self.register_parameter('out_channels_mutable', None)

    def _register_kernel_size_mutable(self,
                                      mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'kernel_size' in mutable_cfgs:
            kernel_size_cfg = copy.deepcopy(mutable_cfgs['kernel_size'])
            self.kernel_size_mutable = MODELS.build(kernel_size_cfg)
            # FIXME
            # use correct type after MutableValue is implemented
            assert isinstance(self.kernel_size_mutable, type)
            self.kernel_size_mutable.current_choice = self.kernel_size
        else:
            self.register_parameter('kernel_size_mutable', None)

    @property
    def mutable_in(self) -> MutableChannel:
        """Mutable `in_channels`."""
        return self.in_channels_mutable

    @property
    def mutable_out(self) -> MutableChannel:
        """Mutable `out_channels`."""
        return self.out_channels_mutable

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_channels` and
        `mutable_out_channels`, and forward."""
        groups = self.groups
        if self.groups == self.in_channels == self.out_channels:
            groups = input.size(1)
        weight, bias = self._get_dynamic_params()

        return F.conv2d(input, weight, bias, self.stride, self.padding,
                        self.dilation, groups)

    # FIXME
    # add logic of mutable value
    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        in_mask = self.in_channels_mutable.current_mask.to(self.weight.device)
        out_mask = self.out_channels_mutable.current_mask.to(
            self.weight.device)

        if self.groups == 1:
            weight = self.weight[out_mask][:, in_mask]
        elif self.groups == self.in_channels == self.out_channels:
            # depth-wise conv
            weight = self.weight[out_mask]
        else:
            raise NotImplementedError(
                'Current `ChannelMutator` only support pruning the depth-wise '
                '`nn.Conv2d` or `nn.Conv2d` module whose group number equals '
                f'to one, but got {self.groups}.')

        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def to_static_op(self) -> nn.Conv2d:
        assert self.in_channels_mutable.is_fixed and \
            self.out_channels_mutable.is_fixed

        weight, bias, = self._get_dynamic_params()
        groups = self.groups
        if groups == self.in_channels == self.out_channels:
            groups = self.in_channels_mutable.current_mask.sum().item()
        out_channels = weight.size(0)
        in_channels = weight.size(1) * groups

        static_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            dilation=self.dilation,
            groups=groups,
            bias=True if bias is not None else False)

        static_conv2d.weight = nn.Parameter(weight)
        if bias is not None:
            static_conv2d.bias = nn.Parameter(bias)

        return static_conv2d


class DynamicLinear(nn.Linear, ChannelDynamicOP):
    """Applies a linear transformation to the incoming data according to the
    `mutable_in_features` and `mutable_out_features` dynamically.

    Args:
        in_features_cfg (Dict): Config related to `in_features`.
        out_features_cfg (Dict): Config related to `out_features`.
    """

    accepted_mutable_keys = {'in_features', 'out_features'}

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **linear_kwargs) -> None:
        super().__init__(**linear_kwargs)

        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_features_mutable(mutable_cfgs)

    def _register_features_mutable(self,
                                   mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'in_features' or 'out_features' in mutable_cfgs:
            assert 'in_features' in mutable_cfgs and \
                'out_features' in mutable_cfgs, \
                'both `in_features` and `out_features` ' \
                'should be contained in `mutable_cfgs`'
            in_features_cfg = copy.deepcopy(mutable_cfgs['in_features'])
            in_features_cfg.update(num_channels=self.in_features)
            self.in_features_mutable = MODELS.build(in_features_cfg)

            out_features_cfg = copy.deepcopy(mutable_cfgs['out_features'])
            out_features_cfg.update(dict(num_channels=self.out_features))
            self.out_features_mutable = MODELS.build(out_features_cfg)

            assert isinstance(self.in_features_mutable, MutableChannel)
            assert isinstance(self.out_features_mutable, MutableChannel)
        else:
            self.register_parameter('in_features_mutable', None)
            self.register_parameter('out_features_mutable', None)

    @property
    def mutable_in(self) -> MutableChannel:
        """Mutable `in_features`."""
        return self.in_features_mutable

    @property
    def mutable_out(self) -> MutableChannel:
        """Mutable `out_features`."""
        return self.out_features_mutable

    def _get_dynamic_params(self) -> Tuple[Tensor, Optional[Tensor]]:
        in_mask = self.in_features_mutable.current_mask.to(self.weight.device)
        out_mask = self.out_features_mutable.current_mask.to(
            self.weight.device)

        weight = self.weight[out_mask][:, in_mask]
        bias = self.bias[out_mask] if self.bias is not None else None

        return weight, bias

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_in_features` and
        `mutable_out_features`, and forward."""
        weight, bias = self._get_dynamic_params()

        return F.linear(input, weight, bias)

    def to_static_op(self) -> nn.Module:
        assert self.in_features_mutable.is_fixed and \
            self.out_features_mutable.is_fixed

        weight, bias = self._get_dynamic_params()
        out_features = weight.size(0)
        in_features = weight.size(1)

        static_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True if bias is not None else False)

        static_linear.weight = nn.Parameter(weight)
        if bias is not None:
            static_linear.bias = nn.Parameter(bias)

        return static_linear


class DynamicBatchNorm(_BatchNorm, ChannelDynamicOP):
    """Applies Batch Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        num_features_cfg (Dict): Config related to `num_features`.
    """
    accepted_mutable_keys = {'num_features'}

    def __init__(self, *, mutable_cfgs: MUTABLE_CFGS_TYPE,
                 **bn_kwargs) -> None:
        super().__init__(**bn_kwargs)

        mutable_cfgs = self.parse_mutable_cfgs(mutable_cfgs)
        self._register_num_features_mutable(mutable_cfgs)

    def _register_num_features_mutable(
            self, mutable_cfgs: MUTABLE_CFGS_TYPE) -> None:
        if 'num_features' in mutable_cfgs:
            num_features_mutable = copy.deepcopy(mutable_cfgs['num_features'])
            num_features_mutable.update(dict(num_channels=self.num_features))
            self.num_features_mutable = MODELS.build(num_features_mutable)

            assert isinstance(self.num_features_mutable, MutableChannel)
        else:
            self.register_parameter('num_features_mutable', None)

    # FIXME
    # might be None
    @property
    def mutable_in(self) -> MutableChannel:
        """Mutable `num_features`."""
        return self.num_features_mutable

    @property
    def mutable_out(self) -> MutableChannel:
        """Mutable `num_features`."""
        return self.num_features_mutable

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
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

        if self.affine:
            out_mask = self.num_features_mutable.current_mask.to(
                self.weight.device)
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            out_mask = self.num_features_mutable.current_mask.to(
                self.running_mean.device)
            running_mean = self.running_mean[out_mask] \
                if not self.training or self.track_running_stats else None
            running_var = self.running_var[out_mask] \
                if not self.training or self.track_running_stats else None
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return F.batch_norm(input, running_mean, running_var, weight, bias,
                            bn_training, exponential_average_factor, self.eps)

    # TODO
    def to_static_op(self) -> nn.Module:
        raise NotImplementedError


class DynamicInstanceNorm(_InstanceNorm, MutableManageMixIn):
    """Applies Instance Normalization over an input according to the
    `mutable_num_features` dynamically.

    Args:
        num_features_cfg (Dict): Config related to `num_features`.
    """

    def __init__(self, num_features_cfg, *args, **kwargs):
        super(DynamicInstanceNorm, self).__init__(*args, **kwargs)

        num_features_cfg_ = copy.deepcopy(num_features_cfg)
        num_features_cfg_.update(dict(num_channels=self.num_features))
        self.mutable_num_features = MODELS.build(num_features_cfg_)

    @property
    def mutable_in(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    @property
    def mutable_out(self):
        """Mutable `num_features`."""
        return self.mutable_num_features

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_features`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_features.current_mask.to(
                self.weight.device)
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        if self.track_running_stats:
            out_mask = self.mutable_num_features.current_mask.to(
                self.running_mean.device)
            running_mean = self.running_mean[out_mask]
            running_var = self.running_var[out_mask]
        else:
            running_mean, running_var = self.running_mean, self.running_var

        return F.instance_norm(input, running_mean, running_var, weight, bias,
                               self.training or not self.track_running_stats,
                               self.momentum, self.eps)


class DynamicGroupNorm(GroupNorm, MutableManageMixIn):
    """Applies Group Normalization over a mini-batch of inputs according to the
    `mutable_num_channels` dynamically.

    Args:
        num_channels_cfg (Dict): Config related to `num_channels`.
    """

    def __init__(self, num_channels_cfg, *args, **kwargs):
        super(DynamicGroupNorm, self).__init__(*args, **kwargs)

        num_channels_cfg_ = copy.deepcopy(num_channels_cfg)
        num_channels_cfg_.update(dict(num_channels=self.num_channels))
        self.mutable_num_channels = MODELS.build(num_channels_cfg_)

    @property
    def mutable_in(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    @property
    def mutable_out(self):
        """Mutable `num_channels`."""
        return self.mutable_num_channels

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_channels`, and
        forward."""
        if self.affine:
            out_mask = self.mutable_num_channels.current_mask.to(
                self.weight.device)
            weight = self.weight[out_mask]
            bias = self.bias[out_mask]
        else:
            weight, bias = self.weight, self.bias

        return F.group_norm(input, self.num_groups, weight, bias, self.eps)
