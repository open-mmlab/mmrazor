# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import NonLocal2d

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


class NonLocal2dMaxpoolNstride(NonLocal2d):
    """Nonlocal block for 2-dimension inputs, with a configurable
    maxpool_stride.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Defaults to 2.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to `nn.Conv2d`.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to `BN`. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: dot_product.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        maxpool_stride (int): The stride of the maxpooling module.
            Defaults to 2.
        zeros_init (bool): Whether to use zero to initialize weights of
            `conv_out`. Defaults to True.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 norm_cfg: Dict = dict(type='BN'),
                 mode: str = 'embedded_gaussian',
                 sub_sample: bool = False,
                 maxpool_stride: int = 2,
                 zeros_init: bool = True,
                 **kwargs) -> None:
        """Inits the NonLocal2dMaxpoolNstride module."""
        super().__init__(
            in_channels=in_channels,
            sub_sample=sub_sample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            reduction=reduction,
            mode=mode,
            zeros_init=zeros_init,
            **kwargs)
        self.norm_cfg = norm_cfg

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(
                kernel_size=(maxpool_stride, maxpool_stride))
            self.g: nn.Sequential = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi: nn.Sequential = nn.Sequential(
                    self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


@MODELS.register_module()
class FBKDStudentConnector(BaseConnector):
    """Improve Object Detection with Feature-based Knowledge Distillation:
    Towards Accurate and Efficient Detectors, ICLR2021.
    https://openreview.net/pdf?id=uKhGRvM8QNH.

    Student connector for FBKD.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Defaults to 2.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to `nn.Conv2d`.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to `BN`. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: dot_product.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        maxpool_stride (int): The stride of the maxpooling module.
            Defaults to 2.
        zeros_init (bool): Whether to use zero to initialize weights of
            `conv_out`. Defaults to True.
        spatial_T (float): Temperature used in spatial-wise pooling.
            Defaults to 0.5.
        channel_T (float): Temperature used in channel-wise pooling.
            Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 norm_cfg: Dict = dict(type='BN'),
                 mode: str = 'dot_product',
                 sub_sample: bool = False,
                 maxpool_stride: int = 2,
                 zeros_init: bool = True,
                 spatial_T: float = 0.5,
                 channel_T: float = 0.5,
                 init_cfg: Optional[Dict] = None,
                 **kwargs) -> None:
        """Inits the FBKDStuConnector."""
        super().__init__(init_cfg)
        self.channel_wise_adaptation = nn.Linear(in_channels, in_channels)

        self.spatial_wise_adaptation = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1)

        self.adaptation_layers = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.student_non_local = NonLocal2dMaxpoolNstride(
            in_channels=in_channels,
            reduction=reduction,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            mode=mode,
            sub_sample=sub_sample,
            maxpool_stride=maxpool_stride,
            zeros_init=zeros_init,
            **kwargs)

        self.non_local_adaptation = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.in_channels = in_channels
        self.spatial_T = spatial_T
        self.channel_T = channel_T

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input student features.

        Returns:
            s_spatial_mask (torch.Tensor): Student spatial-wise mask.
            s_channel_mask (torch.Tensor): Student channel-wise mask.
            s_feat_adapt (torch.Tensor): Adaptative student feature.
            s_channel_pool_adapt (torch.Tensor): Student feature which through
                channel-wise pooling and adaptation_layers.
            s_spatial_pool_adapt (torch.Tensor): Student feature which through
                spatial-wise pooling and adaptation_layers.
            s_relation_adapt (torch.Tensor): Adaptative student relations.
        """
        # Calculate spatial-wise mask.
        s_spatial_mask = torch.mean(torch.abs(x), [1], keepdim=True)
        size = s_spatial_mask.size()
        s_spatial_mask = s_spatial_mask.view(x.size(0), -1)

        # Soften or sharpen the spatial-wise mask by temperature.
        s_spatial_mask = torch.softmax(
            s_spatial_mask / self.spatial_T, dim=1) * size[-1] * size[-2]
        s_spatial_mask = s_spatial_mask.view(size)

        # Calculate channel-wise mask.
        s_channel_mask = torch.mean(torch.abs(x), [2, 3], keepdim=True)
        channel_mask_size = s_channel_mask.size()
        s_channel_mask = s_channel_mask.view(x.size(0), -1)

        # Soften or sharpen the channel-wise mask by temperature.
        s_channel_mask = torch.softmax(
            s_channel_mask / self.channel_T, dim=1) * self.in_channels
        s_channel_mask = s_channel_mask.view(channel_mask_size)

        # Adaptative and pool student feature through channel-wise.
        s_feat_adapt = self.adaptation_layers(x)
        s_channel_pool_adapt = self.channel_wise_adaptation(
            torch.mean(x, [2, 3]))

        # Adaptative and pool student feature through spatial-wise.
        s_spatial_pool = torch.mean(x, [1]).view(
            x.size(0), 1, x.size(2), x.size(3))
        s_spatial_pool_adapt = self.spatial_wise_adaptation(s_spatial_pool)

        # Calculate non_local_adaptation.
        s_relation = self.student_non_local(x)
        s_relation_adapt = self.non_local_adaptation(s_relation)

        return (s_spatial_mask, s_channel_mask, s_channel_pool_adapt,
                s_spatial_pool_adapt, s_relation_adapt, s_feat_adapt)


@MODELS.register_module()
class FBKDTeacherConnector(BaseConnector):
    """Improve Object Detection with Feature-based Knowledge Distillation:
    Towards Accurate and Efficient Detectors, ICLR2021.
    https://openreview.net/pdf?id=uKhGRvM8QNH.

    Teacher connector for FBKD.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Defaults to 2.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to `nn.Conv2d`.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to `BN`. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: dot_product.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        maxpool_stride (int): The stride of the maxpooling module.
            Defaults to 2.
        zeros_init (bool): Whether to use zero to initialize weights of
            `conv_out`. Defaults to True.
        spatial_T (float): Temperature used in spatial-wise pooling.
            Defaults to 0.5.
        channel_T (float): Temperature used in channel-wise pooling.
            Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 conv_cfg: Dict = dict(type='Conv2d'),
                 norm_cfg: Dict = dict(type='BN'),
                 mode: str = 'dot_product',
                 sub_sample: bool = False,
                 maxpool_stride: int = 2,
                 zeros_init: bool = True,
                 spatial_T: float = 0.5,
                 channel_T: float = 0.5,
                 init_cfg: Optional[Dict] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg)
        self.teacher_non_local = NonLocal2dMaxpoolNstride(
            in_channels=in_channels,
            reduction=reduction,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            mode=mode,
            sub_sample=sub_sample,
            maxpool_stride=maxpool_stride,
            zeros_init=zeros_init,
            **kwargs)

        self.in_channels = in_channels
        self.spatial_T = spatial_T
        self.channel_T = channel_T

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input teacher features.

        Returns:
            t_spatial_mask (torch.Tensor): Teacher spatial-wise mask.
            t_channel_mask (torch.Tensor): Teacher channel-wise mask.
            t_spatial_pool (torch.Tensor): Teacher features which through
                spatial-wise pooling.
            t_relation (torch.Tensor): Teacher relation matrix.
        """
        # Calculate spatial-wise mask.
        t_spatial_mask = torch.mean(torch.abs(x), [1], keepdim=True)
        size = t_spatial_mask.size()
        t_spatial_mask = t_spatial_mask.view(x.size(0), -1)

        # Soften or sharpen the spatial-wise mask by temperature.
        t_spatial_mask = torch.softmax(
            t_spatial_mask / self.spatial_T, dim=1) * size[-1] * size[-2]
        t_spatial_mask = t_spatial_mask.view(size)

        # Calculate channel-wise mask.
        t_channel_mask = torch.mean(torch.abs(x), [2, 3], keepdim=True)
        channel_mask_size = t_channel_mask.size()
        t_channel_mask = t_channel_mask.view(x.size(0), -1)

        # Soften or sharpen the channel-wise mask by temperature.
        t_channel_mask = torch.softmax(
            t_channel_mask / self.channel_T, dim=1) * self.in_channels
        t_channel_mask = t_channel_mask.view(channel_mask_size)

        # Adaptative and pool student feature through spatial-wise.
        t_spatial_pool = torch.mean(x, [1]).view(
            x.size(0), 1, x.size(2), x.size(3))

        # Calculate non_local relation.
        t_relation = self.teacher_non_local(x)

        return (t_spatial_mask, t_channel_mask, t_spatial_pool, t_relation, x)
