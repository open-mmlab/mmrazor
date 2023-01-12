# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class ConvModuleConnector(BaseConnector):
    """Convolution connector that bundles conv/norm/activation layers.

    Args:
        in_channel (int): The input channel of the connector.
        out_channel (int): The output channel of the connector.
        kernel_size (int | tuple[int, int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int, int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int, int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int, int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[str, bool] = 'auto',
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Dict = dict(type='ReLU'),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = 'zeros',
        order: tuple = ('conv', 'norm', 'act'),
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.conv_module = ConvModule(in_channel, out_channel, kernel_size,
                                      stride, padding, dilation, groups, bias,
                                      conv_cfg, norm_cfg, act_cfg, inplace,
                                      with_spectral_norm, padding_mode, order)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            feature (torch.Tensor): Input feature.
        """
        for layer in self.conv_module.order:
            if layer == 'conv':
                if self.conv_module.with_explicit_padding:
                    feature = self.conv_module.padding_layer(feature)
                feature = self.conv_module.conv(feature)
            elif layer == 'norm' and self.conv_module.with_norm:
                feature = self.conv_module.norm(feature)
            elif layer == 'act' and self.conv_module.with_activation:
                feature = self.conv_module.activate(feature)
        return feature
