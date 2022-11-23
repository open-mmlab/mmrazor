# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict

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
        # a group-wise conv will not be converted to dynamic conv
        if module.groups > 1 and not (module.groups == module.out_channels ==
                                      module.in_channels):
            global GroupWiseConvWarned
            if GroupWiseConvWarned is False:
                from mmengine import MMLogger
                logger = MMLogger.get_current_instance()
                logger.warning(
                    ('Group-wise convolutional layers are not supported to be'
                     'pruned now, so they are not converted to new'
                     'DynamicConvs.'))
                GroupWiseConvWarned = True

            return module
        else:
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
