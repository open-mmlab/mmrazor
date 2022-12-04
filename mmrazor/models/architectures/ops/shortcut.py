# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import build_conv_layer


class ShortcutLayer(nn.Module):
    """Shortcut Module used in AttentiveNAS.

    Args:
        in_channels (int): The input channels of the Shortcut layer.
        out_channels (int): The output channels of the Shortcut layer.
        reduction (int): Equals to the stride of the parallel conv module.
    """

    def __init__(self, in_channels, out_channels, reduction=1, conv_cfg=None):
        super(ShortcutLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction

        self.conv = build_conv_layer(
            conv_cfg,
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            bias=False)

    def forward(self, x):
        # identity mapping
        if self.in_channels == self.out_channels and self.reduction == 1:
            return x
        # average pooling, if size doesn't match
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)
        # 1*1 conv, if #channels doesn't match
        if self.in_channels != self.out_channels:
            x = self.conv(x)

        return x
