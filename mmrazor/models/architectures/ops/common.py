# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmrazor.registry import MODELS
from .base import BaseOP


@MODELS.register_module()
class Identity(BaseOP):
    """Base class for searchable operations.

    Args:
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: None.
    """

    def __init__(self,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=None,
                 **kwargs):
        super(Identity, self).__init__(**kwargs)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x


@MODELS.register_module()
class InputResizer(BaseOP):
    valid_interpolation_type = {
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area',
        'nearest-exact'
    }

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 interpolation_type: str = 'bicubic',
                 align_corners: bool = False,
                 scale_factor: Optional[Union[float, List[float]]] = None,
                 **kwargs) -> None:
        super(InputResizer, self).__init__(**kwargs)

        if size is not None:
            if len(size) != 2:
                raise ValueError('Length of size must be 2, '
                                 f'but got: {len(size)}')
        self._size = size
        if interpolation_type not in self.valid_interpolation_type:
            raise ValueError(
                'Expect `interpolation_type` be '
                f'one of {self.valid_interpolation_type}, but got: '
                f'{interpolation_type}')
        self._interpolation_type = interpolation_type
        self._scale_factor = scale_factor
        self._align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return x

        return F.interpolate(
            input=x,
            size=self._size,
            mode=self._interpolation_type,
            scale_factor=self._scale_factor,
            align_corners=self._align_corners)
