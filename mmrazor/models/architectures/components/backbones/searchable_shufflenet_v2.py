# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from mmcv.cnn import ConvModule, constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm

from ...utils import Placeholder


@BACKBONES.register_module()
class SearchableShuffleNetV2(BaseBackbone):
    """Based on ShuffleNetV2 backbone.
    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        with_last_layer (bool): Whether is last layer.
            Default: True, which means not need to add `Placeholder``.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        init_cfg (dict): Init config dict for ``BaseBackbone``.
    """

    def __init__(self,
                 stem_multiplier=1,
                 widen_factor=1.0,
                 out_indices=(4, ),
                 frozen_stages=-1,
                 with_last_layer=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 **kwargs):
        super(SearchableShuffleNetV2, self).__init__(**kwargs)
        self.stage_blocks = [4, 4, 8, 4]
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        channels = [
            round(64 * self.widen_factor),
            round(160 * self.widen_factor),
            round(320 * self.widen_factor),
            round(640 * self.widen_factor)
        ]
        last_channels = 1024

        self.in_channels = 16 * stem_multiplier
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks, i)
            self.layers.append(layer)

        if with_last_layer:
            self.layers.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=last_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def _make_layer(self, out_channels, num_blocks, stage_idx):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
            stage_idx (int): used in ``space_id``.
        Returns:
            torch.nn.Sequential: The layer made.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(
                Placeholder(
                    group='all_blocks',
                    space_id=f'stage_{stage_idx}_block_{i}',
                    choice_args=dict(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=stride,
                    )))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        """Freeze params not to update in the specified stages."""
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        """Init weights of ``SearchableShuffleNetV2``."""
        super(SearchableShuffleNetV2, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    normal_init(m, mean=0, std=0.01)
                else:
                    normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, val=1, bias=0.0001)
                if isinstance(m, _BatchNorm):
                    if m.running_mean is not None:
                        nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        x = self.conv1(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(SearchableShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
