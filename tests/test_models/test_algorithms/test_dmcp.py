# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch

from mmrazor.models import DMCP
from mmrazor.registry import MODELS

MUTATOR_CFG = dict(
    channel_mutator=dict(
        type='mmrazor.DMCPChannelMutator',
        channel_unit_cfg={
            'type': 'DMCPChannelUnit'
        },
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='BackwardTracer'),))

DISTILLER_CFG = dict(
    _scope_='mmrazor',
    type='ConfigurableDistiller',
    teacher_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    student_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    distill_losses=dict(
        loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
    loss_forward_mappings=dict(
        loss_kl=dict(
            preds_S=dict(recorder='fc', from_student=True),
            preds_T=dict(recorder='fc', from_student=False))))

ALGORITHM_CFG = dict(
    type='mmrazor.DMCP',
    architecture=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    mutators=MUTATOR_CFG,
    distiller=DISTILLER_CFG)


class TestDMCP(TestCase):
    def test_init(self):
        ALGORITHM_CFG_SUPERNET = copy.deepcopy(ALGORITHM_CFG)
        # initiate dmcp with built `algorithm`.
        dmcp_algo = MODELS.build(ALGORITHM_CFG_SUPERNET)
        self.assertIsInstance(dmcp_algo, DMCP)
        # dmcp mutators include channel_mutator and value_mutator
        assert 'channel_mutator' in dmcp_algo.mutators
        assert 'value_mutator' in dmcp_algo.mutators

        # dmcp_algo support training
        self.assertTrue(dmcp_algo.is_supernet)

        # initiate dmcp without any `mutator`.
        ALGORITHM_CFG_SUPERNET.pop('type')
        ALGORITHM_CFG_SUPERNET['mutators'] = None
        none_type = type(ALGORITHM_CFG_SUPERNET['mutators'])
        with self.assertRaisesRegex(
                TypeError, f'mutator should be a `dict` but got {none_type}'):
            _ = DMCP(**ALGORITHM_CFG_SUPERNET)

        # initiate dmcp with error type `mutator`.
        backwardtracer_cfg = dict(
            type='OneShotChannelMutator',
            channel_unit_cfg=dict(
                type='OneShotMutableChannelUnit',
                default_args=dict(
                    candidate_choices=list(i / 12 for i in range(2, 13)),
                    choice_mode='ratio')),
            parse_cfg=dict(
                type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')))
        ALGORITHM_CFG_SUPERNET['mutators'] = dict(
            channel_mutator=backwardtracer_cfg,
            value_mutator=dict(type='mmrazor.DynamicValueMutator'))
        with self.assertRaisesRegex(AssertionError,
                                    'DMCP only support predefined.'):
            _ = DMCP(**ALGORITHM_CFG_SUPERNET)

    def test_loss(self):
        # supernet
        inputs = torch.randn(1, 3, 224, 224)
        dmcp = MODELS.build(ALGORITHM_CFG)
        loss = dmcp(inputs)
        assert loss.size(1) == 1000
