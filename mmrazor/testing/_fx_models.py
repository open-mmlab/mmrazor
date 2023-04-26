# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule

from mmrazor.registry import MODELS


@MODELS.register_module()
class ConvBNReLU(nn.Module):

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
        super().__init__()
        self.conv_module = ConvModule(in_channel, out_channel, kernel_size,
                                      stride, padding, dilation, groups, bias,
                                      conv_cfg, norm_cfg, act_cfg, inplace,
                                      with_spectral_norm, padding_mode, order)
        self.toy_attr1 = 1
        self.toy_attr2 = 2

    def forward(self, x):
        x = self.conv_module.conv(x)
        x = self.conv_module.norm(x)
        x = self.conv_module.activate(x)
        return x
