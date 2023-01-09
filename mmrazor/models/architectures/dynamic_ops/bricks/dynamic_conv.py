# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.registry import MODELS
from ..mixins.dynamic_conv_mixins import (BigNasConvMixin, DynamicConvMixin,
                                          FuseConvMixin, OFAConvMixin)

GroupWiseConvWarned = False


@MODELS.register_module()
class DynamicConv2d(nn.Conv2d, DynamicConvMixin):
    """Dynamic Conv2d OP.

    Note:
        Arguments for ``__init__`` of ``DynamicConv2d`` is totally same as
        :obj:`torch.nn.Conv2d`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    mutable_attrs: nn.ModuleDict
    accepted_mutable_attrs = {'in_channels', 'out_channels'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'DynamicConv2d':
        """Convert an instance of nn.Conv2d to a new instance of
        DynamicConv2d."""

        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of dynamic conv2d OP."""
        return self.forward_mixin(x)


@MODELS.register_module()
class BigNasConv2d(nn.Conv2d, BigNasConvMixin):
    """Conv2d used in BigNas.

    Note:
        Arguments for ``__init__`` of ``DynamicConv2d`` is totally same as
        :obj:`torch.nn.Conv2d`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    mutable_attrs: nn.ModuleDict
    accepted_mutable_attrs = {'in_channels', 'out_channels', 'kernel_size'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'BigNasConv2d':
        """Convert an instance of `nn.Conv2d` to a new instance of
        `BigNasConv2d`."""
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of bignas' conv2d."""
        return self.forward_mixin(x)


@MODELS.register_module()
class OFAConv2d(nn.Conv2d, OFAConvMixin):
    """Conv2d used in `Once-for-All`.

    Refers to `Once-for-All: Train One Network and Specialize it for Efficient
    Deployment <http://arxiv.org/abs/1908.09791>`_.
    """
    """Dynamic Conv2d OP.

    Note:
        Arguments for ``__init__`` of ``OFAConv2d`` is totally same as
        :obj:`torch.nn.Conv2d`.

    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    mutable_attrs: nn.ModuleDict
    accepted_mutable_attrs = {'in_channels', 'out_channels'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        assert self.padding_mode == 'zeros'
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'OFAConv2d':
        """Convert an instance of `nn.Conv2d` to a new instance of
        `OFAConv2d`."""

        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of OFA's conv2d."""
        return self.forward_mixin(x)


@MODELS.register_module()
class FuseConv2d(nn.Conv2d, FuseConvMixin):
    """FuseConv2d used in `DCFF`.

    Refers to `Training Compact CNNs for Image Classification
    using Dynamic-coded Filter Fusion <https://arxiv.org/abs/2107.06916>`_.
    Attributes:
        mutable_attrs (ModuleDict[str, BaseMutable]): Mutable attributes,
            such as `in_channels`. The key of the dict must in
            ``accepted_mutable_attrs``.
    """
    mutable_attrs: nn.ModuleDict
    accepted_mutable_attrs = {'in_channels', 'out_channels'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mutable_attrs: Dict[str, BaseMutable] = nn.ModuleDict()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'FuseConv2d':
        """Convert an instance of `nn.Conv2d` to a new instance of
        `FuseConv2d`."""
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            padding_mode=module.padding_mode)

    @property
    def conv_func(self) -> Callable:
        """The function that will be used in ``forward_mixin``."""
        return F.conv2d

    @property
    def static_op_factory(self):
        """Corresponding Pytorch OP."""
        return nn.Conv2d

    def forward(self, x: Tensor) -> Tensor:
        """Forward of fused conv2d."""
        return self.forward_mixin(x)


class DynamicConv2dAdaptivePadding(DynamicConv2d):
    """Dynamic version of mmcv.cnn.bricks.Conv2dAdaptivePadding."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (
            max((output_h - 1) * self.stride[0] +
                (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (
            max((output_w - 1) * self.stride[1] +
                (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
        return super().forward(x)

    @property
    def static_op_factory(self):
        from mmcv.cnn.bricks import Conv2dAdaptivePadding
        return Conv2dAdaptivePadding

    def to_static_op(self) -> nn.Conv2d:
        self.check_if_mutables_fixed()

        weight, bias, padding = self.get_dynamic_params()
        groups = self.groups
        if groups == self.in_channels == self.out_channels and \
                self.mutable_in_channels is not None:
            mutable_in_channels = self.mutable_attrs['in_channels']
            groups = mutable_in_channels.current_mask.sum().item()
        out_channels = weight.size(0)
        in_channels = weight.size(1) * groups

        kernel_size = tuple(weight.shape[2:])

        static_conv = self.static_op_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=groups,
            bias=True if bias is not None else False)

        static_conv.weight = nn.Parameter(weight)
        if bias is not None:
            static_conv.bias = nn.Parameter(bias)

        return static_conv
