# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
import torch.nn as nn
from mmcls.models import *  # noqa:F403,F401

from mmrazor.models import *  # noqa:F403,F401
from mmrazor.registry import MODELS

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)
MODELS.register_module(name='torchMaxPool2d', module=nn.MaxPool2d, force=True)
MODELS.register_module(name='torchAvgPool2d', module=nn.AvgPool2d, force=True)


class TestDartsBackbone(TestCase):

    def setUp(self) -> None:
        self.mutable_cfg = dict(
            type='DiffMutableOP',
            candidates=dict(
                torch_conv2d_3x3=dict(
                    type='torchConv2d',
                    kernel_size=3,
                    padding=1,
                ),
                torch_conv2d_5x5=dict(
                    type='torchConv2d',
                    kernel_size=5,
                    padding=2,
                ),
                torch_conv2d_7x7=dict(
                    type='torchConv2d',
                    kernel_size=7,
                    padding=3,
                ),
            ))

        self.route_cfg = dict(
            type='DiffChoiceRoute',
            with_arch_param=True,
        )

        self.backbone_cfg = dict(
            type='mmrazor.DartsBackbone',
            in_channels=3,
            base_channels=16,
            num_layers=8,
            num_nodes=4,
            stem_multiplier=3,
            out_indices=(7, ),
            mutable_cfg=self.mutable_cfg,
            route_cfg=self.route_cfg)

        self.mutator_cfg = dict(
            type='NasMutator',
            custom_groups=None,
        )

    def test_darts_backbone(self):
        model = MODELS.build(self.backbone_cfg)
        custom_group = self.generate_key(model)

        assert model is not None
        self.mutable_cfg.update(custom_group=custom_group)
        mutator = MODELS.build(self.mutator_cfg)
        assert mutator is not None

        mutator.prepare_from_supernet(model)
        # mutator.modify_supernet_forward(mutator.arch_params)

        inputs = torch.randn(4, 3, 224, 224)
        outputs = model(inputs)
        assert outputs is not None

    def test_darts_backbone_with_auxliary(self):
        self.backbone_cfg.update(
            auxliary=True, aux_channels=256, aux_out_channels=512)
        model = MODELS.build(self.backbone_cfg)
        custom_group = self.generate_key(model)

        assert model is not None
        self.mutable_cfg.update(custom_groups=custom_group)
        mutator = MODELS.build(self.mutator_cfg)
        assert mutator is not None
        mutator.prepare_from_supernet(model)
        # mutator.modify_supernet_forward(mutator.arch_params)

        inputs = torch.randn(4, 3, 224, 224)
        outputs = model(inputs)
        assert outputs is not None

    def generate_key(self, model):
        """auto generate custom group for darts."""
        tmp_dict = dict()

        for key, _ in model.named_modules():
            node_type = key.split('._candidates')[0].split('.')[-1].split(
                '_')[0]
            if node_type not in ['normal', 'reduce']:
                # not supported type
                continue

            node_name = key.split('._candidates')[0].split('.')[-1]
            if node_name not in tmp_dict.keys():
                tmp_dict[node_name] = [key.split('._candidates')[0]]
            else:
                current_key = key.split('._candidates')[0]
                if current_key not in tmp_dict[node_name]:
                    tmp_dict[node_name].append(current_key)

        return list(tmp_dict.values())


if __name__ == '__main__':
    unittest.main()
