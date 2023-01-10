# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch

from mmrazor.models import DMCP, DMCPChannelMutator
from mmrazor.registry import MODELS

MUTATOR_CFG = dict(
    type='mmrazor.DMCPChannelMutator',
    channel_unit_cfg={'type': 'DMCPChannelUnit'},
    parse_cfg=dict(
        type='ChannelAnalyzer',
        demo_input=(1, 3, 224, 224),
        tracer_type='BackwardTracer'),
)

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
    mutator_cfg=MUTATOR_CFG,
    distiller=DISTILLER_CFG)


class TestDMCP(TestCase):

    def test_init(self):
        ALGORITHM_CFG_SUPERNET = copy.deepcopy(ALGORITHM_CFG)
        # initiate dmcp with built `algorithm`.
        dmcp_algo = MODELS.build(ALGORITHM_CFG_SUPERNET)
        self.assertIsInstance(dmcp_algo, DMCP)
        # dmcp mutators include channel_mutator and value_mutator
        assert isinstance(dmcp_algo.mutator, DMCPChannelMutator)

        # dmcp_algo support training
        self.assertTrue(dmcp_algo.is_supernet)

        # initiate dmcp without any `mutator`.
        ALGORITHM_CFG_SUPERNET.pop('type')
        ALGORITHM_CFG_SUPERNET['mutator_cfg'] = None

        with self.assertRaisesRegex(
                AttributeError, "'NoneType' object has no attribute 'get'"):
            _ = DMCP(**ALGORITHM_CFG_SUPERNET)

    def test_loss(self):
        # supernet
        inputs = torch.randn(1, 3, 224, 224)
        dmcp = MODELS.build(ALGORITHM_CFG)
        dmcp.is_supernet = False
        loss = dmcp(inputs, mode='tensor')
        assert loss.size(1) == 1000
