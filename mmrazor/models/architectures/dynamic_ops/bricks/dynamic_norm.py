# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.utils import _BatchNormXd
from torch import Tensor
from torch.nn import LayerNorm
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ..mixins import DynamicBatchNormMixin, DynamicLayerNormMixin

PartialType = Callable[[Any, Optional[nn.Parameter]], Tuple]


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
        """Corresponding Pytorch OP."""
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
        """Corresponding Pytorch OP."""
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
        """Corresponding Pytorch OP."""
        return nn.BatchNorm3d

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(
                input.dim()))


class SwitchableBatchNorm2d(DynamicBatchNorm2d):
    """A switchable DynamicBatchNorm2d. It mmploys independent batch
    normalization for different switches in a slimmable network.

    To train slimmable networks, ``SwitchableBatchNorm2d`` privatizes all batch
    normalization layers for each switch in a slimmable network. Compared with
    the naive training approach, it solves the problem of feature aggregation
    inconsistency between different switches by independently normalizing the
    feature mean and variance during testing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_bn = nn.ModuleDict()

    def init_candidates(self, candidates: List):
        """Initialize candicates."""
        assert len(self.candidate_bn) == 0
        self._check_candidates(candidates)
        for num in candidates:
            self.candidate_bn[str(num)] = nn.BatchNorm2d(
                num, self.eps, self.momentum, self.affine,
                self.track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        """Forward."""
        choice_num = self.activated_channel_num()
        if choice_num == self.num_features:
            return super().forward(input)
        else:
            assert str(choice_num) in self.candidate_bn
            return self.candidate_bn[str(choice_num)](input)

    def to_static_op(self: _BatchNorm) -> nn.Module:
        """Convert to a normal BatchNorm."""
        choice_num = self.activated_channel_num()
        if choice_num == self.num_features:
            return super().to_static_op()
        else:
            assert str(choice_num) in self.candidate_bn
            return self.candidate_bn[str(choice_num)]

    # private methods

    def activated_channel_num(self):
        """The number of activated channels."""
        mask = self._get_num_features_mask()
        choice_num = (mask == 1).sum().item()
        return choice_num

    def _check_candidates(self, candidates: List):
        """Check if candidates aviliable."""
        for value in candidates:
            assert isinstance(value, int)
            assert 0 < value <= self.num_features

    @property
    def static_op_factory(self):
        """Return initializer of static op."""
        return nn.BatchNorm2d


@MODELS.register_module()
class DynamicLayerNorm(LayerNorm, DynamicLayerNormMixin):
    """Applies Layer Normalization over a mini-batch of inputs according to the
    `mutable_num_channels` dynamically.

    Note:
        Arguments for ``__init__`` of ``DynamicLayerNorm`` is totally same as
        :obj:`torch.nn.LayerNorm`.
    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `num_features`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    accepted_mutable_attrs = {'num_features'}

    def __init__(self, *args, **kwargs):
        super(DynamicLayerNorm, self).__init__(*args, **kwargs)

        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return LayerNorm

    @classmethod
    def convert_from(cls, module: LayerNorm):
        """Convert a _BatchNorm module to a DynamicBatchNorm.

        Args:
            module (:obj:`torch.nn._BatchNorm`): The original BatchNorm module.
        """
        dynamic_ln = cls(
            normalized_shape=module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine)

        return dynamic_ln

    def forward(self, input: Tensor) -> Tensor:
        """Slice the parameters according to `mutable_num_channels`, and
        forward."""
        self._check_input_dim(input)

        weight, bias = self.get_dynamic_params()
        self.normalized_shape = (
            self.mutable_num_features.activated_channels, )

        return F.layer_norm(input, self.normalized_shape, weight, bias,
                            self.eps)

    def _check_input_dim(self, input: Tensor) -> None:
        """Check if input dimension is valid."""
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(
                input.dim()))


class DynamicSyncBatchNorm(nn.SyncBatchNorm, DynamicBatchNormMixin):
    """DynamicOp for sync bn."""

    def __init__(self,
                 num_features: int,
                 eps: float = 0.00001,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 process_group: Optional[Any] = None) -> None:
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, process_group)
        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module):
        return cls(module.num_features, module.eps, module.momentum,
                   module.affine, module.track_running_stats,
                   module.process_group)

    @property
    def static_op_factory(self):
        return nn.SyncBatchNorm

    def forward(self, input: Tensor) -> Tensor:
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError(
                'SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)
        if hasattr(self, '_check_non_zero_input_channels'):
            self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = (1.0 /
                                              self.num_batches_tracked.item())
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization
        rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when
        buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is
                                                           None)
        r"""
        Buffers are only updated if they are to be tracked and we are in
        training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when
        they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean
            if not self.training or self.track_running_stats else None)
        running_var = (
            self.running_var
            if not self.training or self.track_running_stats else None)

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (bn_training and self.training)
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        running_mean, running_var, weight, bias = self.get_dynamic_params()

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            out = F.batch_norm(
                input,
                running_mean,
                running_var,
                weight,
                bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            out = sync_batch_norm.apply(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,
                world_size,
            )

        # copy changed running statistics
        if self.training and self.track_running_stats:
            out_mask = self._get_num_features_mask()
            self.running_mean.masked_scatter_(out_mask, running_mean)
            self.running_var.masked_scatter_(out_mask, running_var)

        return out


class DynamicBatchNormXd(_DynamicBatchNorm):
    """Dynamic op for _DynamicBatchNorm."""

    @property
    def static_op_factory(self):
        return _BatchNormXd

    def _check_input_dim(self, input: torch.Tensor):
        return


@MODELS.register_module()
class DMCPBatchNorm2d(DynamicBatchNorm2d):
    accepted_mutable_attrs = {'num_features'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mutable_attrs: Dict[str, Optional[BaseMutable]] = nn.ModuleDict()

    def forward(self,
                input: Tensor,
                arch_param=None,
                arch_attr=None) -> Tensor:
        """Forward of dynamic DMCPBatchNorm2d."""
        out = self.forward_batchnorm(input)
        if arch_param is not None:
            out = self.forward_arch_param(out, arch_param, arch_attr)
        return out

    def forward_batchnorm(self, input: Tensor) -> Tensor:
        """Forward of BatchNorm2d."""
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

    def forward_arch_param(self, input: Tensor, arch_param,
                           arch_attr) -> Tensor:
        """Forward of arch parameters."""
        size_x = input.size()
        (group_size, num_groups, min_ch) = arch_attr

        if num_groups == 0 or size_x[1] == min_ch:
            return input

        arch = torch.clamp(arch_param, min=0)
        prob_distribute = torch.exp(-arch)

        prob = torch.cumprod(prob_distribute, dim=0).view(num_groups, 1)
        tp_x = input.transpose(0, 1).contiguous()
        tp_group_x = tp_x[min_ch:]

        size_tp_group = tp_group_x.size()
        num_groups = size_tp_group[0] // group_size
        tp_group_x = tp_group_x.view(num_groups, -1) * prob[:num_groups]
        tp_group_x = tp_group_x.view(size_tp_group)

        out = torch.cat([tp_x[:min_ch], tp_group_x]).transpose(0,
                                                               1).contiguous()
        return out

    def set_forward_args(self, arch_param: nn.Parameter,
                         arch_attr: Optional[Tuple]) -> None:
        """Interface for modifying the arch_param using partial."""
        forward_with_default_args: PartialType = \
            partial(self.forward, arch_param=arch_param, arch_attr=arch_attr)
        setattr(self, 'forward', forward_with_default_args)
