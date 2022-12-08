# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.model import Sequential, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.architectures.dynamic_ops.bricks import DynamicSequential
from mmrazor.models.architectures.ops.mobilenet_series import MBBlock
from mmrazor.models.architectures.utils.mutable_register import (
    mutate_conv_module, mutate_mobilenet_layer)
from mmrazor.models.mutables import (MutableChannelContainer,
                                     OneShotMutableChannel,
                                     OneShotMutableChannelUnit,
                                     OneShotMutableValue)
from mmrazor.models.utils.parse_values import parse_values
from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
    from mmcls.models.utils import make_divisible
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')
    make_divisible = get_placeholder('mmcls')

logger = MMLogger.get_current_instance()


@MODELS.register_module()
class AttentiveMobileNetV3(BaseBackbone):
    """Searchable MobileNetV3 backbone.

    Args:
        arch_setting (list[list]): Architecture settings.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        stride_list (list[list]): stride setting in each stage.
            Default: None
        with_se_list (list[list]): Whether to use se-layer in each stage.
            Default: None
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Zero norm param in linear conv of MBBlock
            or not when there is a shortcut. Default: True.
        with_attentive_shortcut (bool): Use shortcut in AttentiveNAS or not.
            Defaults to True.
        init_cfg (dict | list[dict], optional): initialization configuration
            dict to define initializer. OpenMMLab has implemented
            6 initializers, including ``Constant``, ``Xavier``, ``Normal``,
            ``Uniform``, ``Kaiming``, and ``Pretrained``.
    """

    def __init__(self,
                 arch_setting: Dict[str, List],
                 widen_factor: float = 1.,
                 out_indices: Sequence[int] = (7, ),
                 frozen_stages: int = -1,
                 stride_list: List = None,
                 with_se_list: List = None,
                 conv_cfg: Dict = dict(type='BigNasConv2d'),
                 norm_cfg: Dict = dict(type='DynamicBatchNorm2d'),
                 act_cfg: Dict = dict(type='Swish'),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 zero_init_residual: bool = True,
                 with_attentive_shortcut: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):

        super().__init__(init_cfg)

        self.arch_setting = arch_setting
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')
        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.with_cp = with_cp
        self.with_attentive_shortcut = with_attentive_shortcut

        self.stride_list = stride_list if stride_list is not None \
            else [1, 2, 2, 2, 1, 2, 1]
        self.with_se_list = with_se_list if with_se_list is not None \
            else [False, False, True, False, True, True, True]

        # adapt mutable settings
        self.kernel_size_list = parse_values(self.arch_setting['kernel_size'])
        self.num_blocks_list = parse_values(self.arch_setting['num_blocks'])
        self.expand_ratio_list = \
            parse_values(self.arch_setting['expand_ratio'])
        self.num_channels_list = \
            parse_values(self.arch_setting['num_out_channels'])

        self.num_channels_list = [[
            make_divisible(c * widen_factor, 8) for c in channels
        ] for channels in self.num_channels_list]

        self.first_out_channels_list = self.num_channels_list.pop(0)
        self.last_out_channels_list = self.num_channels_list.pop(-1)
        self.last_expand_ratio_list = self.expand_ratio_list.pop(-1)
        assert len(self.kernel_size_list) == len(self.num_blocks_list) == \
            len(self.expand_ratio_list) == len(self.num_channels_list)

        self.layers = self._make_layer()

        self.register_mutables()

    def _make_layer(self):
        """Build multiple mobilenet layers."""
        layers = []
        self.in_channels = max(self.first_out_channels_list)

        self.first_conv = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='Swish'))

        # self.first_mutable = OneShotMutableChannel(
        #     num_channels=self.in_channels,
        #     candidate_choices=self.first_out_channels_list)

        for i, (num_blocks, kernel_sizes, expand_ratios, num_channels) in \
            enumerate(zip(self.num_blocks_list, self.kernel_size_list,
                          self.expand_ratio_list, self.num_channels_list)):
            inverted_res_layer = self._make_single_layer(
                out_channels=num_channels,
                num_blocks=num_blocks,
                kernel_sizes=kernel_sizes,
                expand_ratios=expand_ratios,
                stride=self.stride_list[i],
                use_se=self.with_se_list[i])
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            layers.append(inverted_res_layer)

        last_expand_channels = \
            self.in_channels * max(self.last_expand_ratio_list)
        self.out_channels = max(self.last_out_channels_list)
        last_layers = Sequential(
            OrderedDict([('final_expand_layer',
                          ConvModule(
                              in_channels=self.in_channels,
                              out_channels=last_expand_channels,
                              kernel_size=1,
                              padding=0,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=dict(type='Swish'))),
                         ('pool', nn.AdaptiveAvgPool2d((1, 1))),
                         ('feature_mix_layer',
                          ConvModule(
                              in_channels=last_expand_channels,
                              out_channels=self.out_channels,
                              kernel_size=1,
                              padding=0,
                              bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=None,
                              act_cfg=dict(type='Swish')))]))
        self.add_module('last_conv', last_layers)
        layers.append(last_layers)
        return layers

    def _make_single_layer(self, out_channels: List, num_blocks: List,
                           kernel_sizes: List, expand_ratios: List,
                           stride: int, use_se: bool):
        """Stack InvertedResidual blocks (MBBlocks) to build a layer for
        MobileNetV3.

        Args:
            out_channels (List): out_channels of block.
            num_blocks (List): num of blocks.
            kernel_sizes (List): num of kernel sizes.
            expand_ratios (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
            stride (int): stride of the first block.
            use_se (bool): Use SE layer in MBBlock or not.
        """
        _layers = []
        for i in range(max(num_blocks)):
            if i >= 1:
                stride = 1
            if use_se:
                se_cfg = dict(
                    act_cfg=(dict(type='ReLU'), dict(type='HSigmoid')),
                    ratio=4,
                    conv_cfg=self.conv_cfg)
            else:
                se_cfg = None  # type: ignore

            mb_layer = MBBlock(
                in_channels=self.in_channels,
                out_channels=max(out_channels),
                kernel_size=max(kernel_sizes),
                stride=stride,
                expand_ratio=max(expand_ratios),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp,
                se_cfg=se_cfg,
                with_attentive_shortcut=self.with_attentive_shortcut)

            _layers.append(mb_layer)
            self.in_channels = max(out_channels)

        dynamic_seq = DynamicSequential(*_layers)
        return dynamic_seq

    def register_mutables(self):
        """Mutate the BigNAS-style MobileNetV3."""
        OneShotMutableChannelUnit._register_channel_container(
            self, MutableChannelContainer)

        self.last_mutable = OneShotMutableChannel(
            num_channels=max(self.first_out_channels_list),
            candidate_choices=self.first_out_channels_list)

        # self.first_module = self.last_mutable.derive_same_mutable

        # mutate_conv_module(
        #     self.first_conv, mutable_out_channels=self.first_module)
        
        MutableChannelContainer.register_mutable_channel_to_module(
            self.first_conv,
            self.last_mutable.derive_same_mutable,
            True,
            start=0,
            end=24)

        # mutate the built mobilenet layers
        for i, layer in enumerate(self.layers[:-1]):
            num_blocks = self.num_blocks_list[i]
            kernel_sizes = self.kernel_size_list[i]
            expand_ratios = self.expand_ratio_list[i]
            out_channels = self.num_channels_list[i]

            mutable_kernel_size = OneShotMutableValue(
                value_list=kernel_sizes, default_value=max(kernel_sizes))
            mutable_expand_value = OneShotMutableValue(
                value_list=expand_ratios, default_value=max(expand_ratios))
            layer.out_channels = OneShotMutableChannel(
                num_channels=max(out_channels), candidate_choices=out_channels)

            # mutable_channel_name = f'layer{i+1}_mutable_out_channels'
            # setattr(
            #     self, mutable_channel_name,
            #     OneShotMutableChannel(
            #         num_channels=max(out_channels),
            #         candidate_choices=out_channels))

            se_ratios = [i / 4 for i in expand_ratios]
            mutable_se_channels = OneShotMutableValue(
                value_list=se_ratios, default_value=max(se_ratios))

            for k in range(max(self.num_blocks_list[i])):
                mutate_mobilenet_layer(layer[k], 
                                       self.last_mutable,
                                       layer.out_channels,
                                       mutable_se_channels,
                                       mutable_expand_value,
                                       mutable_kernel_size)
                self.last_mutable = layer.out_channels

            mutable_depth = OneShotMutableValue(
                value_list=num_blocks, default_value=max(num_blocks))
            layer.register_mutable_attr('depth', mutable_depth)

        self.last_mutable_out_channels = OneShotMutableChannel(
            num_channels=self.out_channels,
            candidate_choices=self.last_out_channels_list)
        last_mutable_expand_value = OneShotMutableValue(
            value_list=self.last_expand_ratio_list,
            default_value=max(self.last_expand_ratio_list))

        derived_expand_channels = self.last_mutable * last_mutable_expand_value
        mutate_conv_module(
            self.layers[-1].final_expand_layer,
            mutable_in_channels=self.last_mutable,
            mutable_out_channels=derived_expand_channels)
        mutate_conv_module(
            self.layers[-1].feature_mix_layer,
            mutable_in_channels=derived_expand_channels,
            mutable_out_channels=self.last_mutable_out_channels)

        self.last_mutable = self.last_mutable_out_channels

    def forward(self, x):
        # import pdb;pdb.set_trace()
        # print(x[:, 0, 0:5, 0]) # tensor([[-0.5085, -0.6097, -0.8077,  0.4523,  1.3825]], device='cuda:0')
        x = self.first_conv(x)
        # tensor([[-0.0688,  0.3314,  7.5532,  0.9856,  4.2129]], device='cuda:0')
        outs = []
        for i, layer in enumerate(self.layers): # 8
            x = layer(x)
            if i in [0, 2, 3, 4, 5, 6]:
                print(x.shape)
                # import pdb;pdb.set_trace() 
            if i in self.out_indices:
                outs.append(x)

        # outs[0][:, 5:10, 0, 0] tensor([[590458.9375, 432793.2812, 269742.4688,     -0.0000,     -0.0000]],
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def init_weights(self) -> None:
        super().init_weights()

        if self.zero_init_residual:
            for name, module in self.named_modules():
                if isinstance(module, MBBlock):
                    if module.with_res_shortcut or \
                            module.with_attentive_shortcut:
                        norm_layer = module.linear_conv.norm
                        constant_init(norm_layer, val=0)
                        logger.debug(
                            f'init {type(norm_layer)} of linear_conv in '
                            f'`{name}` to zero')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.first_conv.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
