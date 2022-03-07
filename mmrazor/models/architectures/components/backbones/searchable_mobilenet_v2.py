# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import make_divisible
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm

from ...utils import Placeholder


@BACKBONES.register_module()
class SearchableMobileNetV2(BaseBackbone):
    """Searchable MobileNetV2 backbone.

    Args:
        first_channels (int): Channel width of first ConvModule. Default: 32.
        last_channels (int): Channel width of last ConvModule. Default: 1200.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        arch_setting_type (str): Specify architecture setting.
            Default: 'original'.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.
    """

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride.
    arch_settings_dict = {
        'original': [[16, 1, 1], [24, 2, 2], [32, 3, 2], [64, 4, 2],
                     [96, 3, 1], [160, 3, 2], [320, 1, 1]],
        'proxyless_gpu': [[24, 1, 1], [32, 4, 2], [56, 4, 2], [112, 4, 2],
                          [128, 4, 1], [256, 4, 2], [432, 1, 1]],
    }

    def __init__(self,
                 first_channels=32,
                 last_channels=1280,
                 widen_factor=1.,
                 out_indices=(7, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 arch_setting_type='original',
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(SearchableMobileNetV2, self).__init__(init_cfg)

        arch_settings = self.arch_settings_dict.get(arch_setting_type)
        if arch_settings is None:
            raise ValueError(f'Expect `arch_setting_type`: '
                             f'{list(self.arch_settings_dict.keys())}, '
                             f'but got: {arch_setting_type}')
        self.arch_settings = arch_settings
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = make_divisible(first_channels * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            channel, num_blocks, stride = layer_cfg
            out_channels = make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                stage_idx=i)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

        if widen_factor > 1.0:
            self.out_channel = int(last_channels * widen_factor)
        else:
            self.out_channel = last_channels

        layer = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.add_module('conv2', layer)
        self.layers.append('conv2')

    def make_layer(self, out_channels, num_blocks, stride, stage_idx):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): number of blocks.
            stride (int): stride of the first block. Default: 1
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio. Default: 6.
        """
        layers = []
        for i in range(num_blocks):
            if i >= 1:
                stride = 1
            # HACK
            # do not search first block
            if stage_idx == 0:
                group = 'first_blocks'
            else:
                group = 'searchable_blocks'
            layers.append(
                Placeholder(
                    group=group,
                    space_id=f'stage_{stage_idx}_block_{i}',
                    choice_args=dict(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=stride,
                    )))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        """Freeze params not to update in the specified stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(SearchableMobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
