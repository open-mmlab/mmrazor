# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


class NonLocalBlockND(nn.Module):
    """Nonlocal block for n-dimension inputs.

    Formulations:
    out = w * y + x;
    y = softmax(x * θ * φ) * g(x);
    x is input feature;
    * means dot-product.

    Args:
        in_channel (int): The number of input channel.
        inter_channel (int, optional): The number of inter channel.
            Defaults to None.
        dimension (int): Dimension of input feature. Defaults to 2.
        use_bn_layer (bool): Whether to use BN layer. Defaults to True.
        with_downsample (bool): Whether to downsample. Defaults to True.
        downsample_stride (int): Downsample stride. Defaults to 2.
    """

    def __init__(self,
                 in_channel: int,
                 inter_channel: Optional[int] = None,
                 dimension: int = 2,
                 use_bn_layer: bool = True,
                 with_downsample: bool = True,
                 downsample_stride: int = 2) -> None:
        """Inits the NonLocalBlockND module."""
        super().__init__()

        assert dimension in [1, 2, 3], \
            f'"dimension" must be 1, 2 or 3, but got {dimension}.'
        self.inter_channel = inter_channel

        if self.inter_channel is None:
            self.inter_channel = in_channel // 2
            if self.inter_channel == 0:
                self.inter_channel = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(
                kernel_size=(1, downsample_stride, downsample_stride))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(
                kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(downsample_stride))
            bn = nn.BatchNorm1d

        # Function g() in non-local formulations.
        self.func_g = conv_nd(
            in_channels=in_channel,
            out_channels=self.inter_channel,
            kernel_size=1,
            stride=1,
            padding=0)

        if use_bn_layer:
            # Matrix w in non-local formulations.
            self.weight_w = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channel,
                    out_channels=in_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0), bn(in_channel))
            nn.init.constant_(self.weight_w[1].weight, 0)
            nn.init.constant_(self.weight_w[1].bias, 0)
        else:
            self.weight_w = conv_nd(
                in_channels=self.inter_channel,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding=0)
            nn.init.constant_(self.weight_w.weight, 0)
            nn.init.constant_(self.weight_w.bias, 0)

        # Matrix θ in non-local formulations.
        self.theta = conv_nd(
            in_channels=in_channel,
            out_channels=self.inter_channel,
            kernel_size=1,
            stride=1,
            padding=0)

        # Matrix φ in non-local formulations.
        self.phi = conv_nd(
            in_channels=in_channel,
            out_channels=self.inter_channel,
            kernel_size=1,
            stride=1,
            padding=0)

        if with_downsample:
            self.func_g = nn.Sequential(self.func_g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of NonLocalBlockND.

        Args:
            x (torch.Tensor): Input feature.

        Returns:
            out (torch.Tensor): Non-local relation matrix.
        """
        batch_size = x.size(0)

        # Calculate θ and φ in the non-local formulations.
        theta_x = self.theta(x).view(batch_size, self.inter_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channel, -1)
        func_f = torch.matmul(theta_x, phi_x)
        shape = func_f.size(-1)
        f_div_C = func_f / shape

        # Calculate y in the non-local formulations.
        g_x = self.func_g(x).view(batch_size, self.inter_channel, -1)
        g_x = g_x.permute(0, 2, 1)
        nonlocal_y = torch.matmul(f_div_C, g_x)
        nonlocal_y = nonlocal_y.permute(0, 2, 1).contiguous()
        nonlocal_y = nonlocal_y.view(batch_size, self.inter_channel,
                                     *x.size()[2:])

        # Calculate out in the non-local formulations.
        W_mul_y = self.weight_w(nonlocal_y)
        out = W_mul_y + x

        return out


@MODELS.register_module()
class FBKDStudentConnector(BaseConnector):
    """Improve Object Detection with Feature-based Knowledge Distillation:
    Towards Accurate and Efficient Detectors, ICLR2021.
    https://openreview.net/pdf?id=uKhGRvM8QNH.

    Student connector of FBKD.

    Args:
        in_channel (int): Number of input channels.
        inter_channel (int, optional): Number of inter channels.
        with_downsample (bool): Whether to downsample.
            Defaults to True.
        downsample_stride (int): Downsample stride. Defaults to 2.
        spatial_T (float): Temperature used in spatial-wise pooling.
            Defaults to 0.5.
        channel_T (float): Temperature used in channel-wise pooling.
            Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 in_channel: int,
                 inter_channel: int = None,
                 with_downsample: bool = True,
                 downsample_stride: int = 2,
                 spatial_T: float = 0.5,
                 channel_T: float = 0.5,
                 init_cfg: Optional[Dict] = None) -> None:
        """Inits the FBKDStuConnector."""
        super().__init__(init_cfg)
        self.channel_wise_adaptation = nn.Linear(in_channel, in_channel)

        self.spatial_wise_adaptation = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1)

        self.adaptation_layers = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0)

        self.student_non_local = NonLocalBlockND(
            in_channel=in_channel,
            inter_channel=inter_channel,
            with_downsample=with_downsample,
            downsample_stride=downsample_stride)

        self.non_local_adaptation = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0)

        self.in_channel = in_channel
        self.spatial_T = spatial_T
        self.channel_T = channel_T

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input student features.

        Returns:
            s_spatial_mask (torch.Tensor): Student spatial-wise mask.
            s_channel_mask (torch.Tensor): Student channel-wise mask.
            s_adapt_feat (torch.Tensor): Adaptative student feature.
            s_channel_pool_adapt (torch.Tensor): Student feature which through
                channel-wise pooling and adaptation_layers.
            s_spatial_pool_adapt (torch.Tensor): Student feature which through
                spatial-wise pooling and adaptation_layers.
            s_adapt_relation (torch.Tensor): Adaptative student relations.
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
            s_channel_mask / self.channel_T, dim=1) * self.in_channel
        s_channel_mask = s_channel_mask.view(channel_mask_size)

        # Adaptative and pool student feature through channel-wise.
        s_adapt_feat = self.adaptation_layers(x)
        s_channel_pool_adapt = self.channel_wise_adaptation(
            torch.mean(x, [2, 3]))

        # Adaptative and pool student feature through spatial-wise.
        s_spatial_pool = torch.mean(x, [1]).view(
            x.size(0), 1, x.size(2), x.size(3))
        s_spatial_pool_adapt = self.spatial_wise_adaptation(s_spatial_pool)

        # Calculate non_local_adaptation.
        s_relation = self.student_non_local(x)
        s_adapt_relation = self.non_local_adaptation(s_relation)

        return (s_spatial_mask, s_channel_mask, s_channel_pool_adapt,
                s_spatial_pool_adapt, s_adapt_relation, s_adapt_feat)


@MODELS.register_module()
class FBKDTeacherConnector(BaseConnector):
    """Improve Object Detection with Feature-based Knowledge Distillation:
    Towards Accurate and Efficient Detectors, ICLR2021.
    https://openreview.net/pdf?id=uKhGRvM8QNH.

    Teacher connector of FBKD.

    Args:
        in_channel (int): Number of input channels.
        inter_channel (int): Number of inter channels.
        with_downsample (bool, optional): Whether to downsample.
            Defaults to True.
        downsample_stride (int): Downsample stride. Defaults to 2.
        spatial_T (float): Temperature used in spatial-wise pooling.
            Defaults to 0.5.
        channel_T (float): Temperature used in channel-wise pooling.
            Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 in_channel,
                 inter_channel=None,
                 with_downsample=True,
                 downsample_stride=2,
                 spatial_T: float = 0.5,
                 channel_T: float = 0.5,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)
        self.teacher_non_local = NonLocalBlockND(
            in_channel=in_channel,
            inter_channel=inter_channel,
            with_downsample=with_downsample,
            downsample_stride=downsample_stride)

        self.in_channel = in_channel
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
            t_channel_mask / self.channel_T, dim=1) * self.in_channel
        t_channel_mask = t_channel_mask.view(channel_mask_size)

        # Adaptative and pool student feature through spatial-wise.
        t_spatial_pool = torch.mean(x, [1]).view(
            x.size(0), 1, x.size(2), x.size(3))

        # Calculate non_local relation.
        t_relation = self.teacher_non_local(x)

        return (t_spatial_mask, t_channel_mask, t_spatial_pool, t_relation, x)
