# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, Union
from unittest import TestCase

import pytest
import torch
import torch.distributed as dist
from mmcls.structures import ClsDataSample
from mmengine import MessageHub
from mmengine.optim.optimizer import OptimWrapper, OptimWrapperDict
from torch.optim import SGD

from mmrazor.models.algorithms import DMCP, DMCPDDP
from mmrazor.models.mutators import DMCPChannelMutator
from mmrazor.registry import MODELS

MUTATOR_TYPE = Union[torch.nn.Module, Dict]
DISTILLER_TYPE = Union[torch.nn.Module, Dict]

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
    distiller=DISTILLER_CFG,
    strategy=['max', 'min', 'scheduled_random', 'arch_random'],
    arch_start_train=10,
    distillation_times=10,
    arch_train_freq=10)


class TestDMCP(TestCase):

    def _prepare_fake_data(self) -> Dict:
        imgs = torch.randn(16, 3, 224, 224).to(self.device)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 1000,
                                                       (16, ))).to(self.device)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def test_init(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ALGORITHM_CFG_SUPERNET = copy.deepcopy(ALGORITHM_CFG)
        # initiate dmcp with built `algorithm`.
        dmcp_algo = MODELS.build(ALGORITHM_CFG_SUPERNET)
        self.assertIsInstance(dmcp_algo, DMCP)
        # dmcp mutators include channel_mutator and value_mutator
        assert isinstance(dmcp_algo.mutator, DMCPChannelMutator)

        ALGORITHM_CFG_SUPERNET.pop('type')
        fake_distiller = 'distiller'
        # initiate dmcp without `distiller`.
        with self.assertRaisesRegex(
                TypeError, 'distiller should be a `dict` or '
                '`ConfigurableDistiller` instance, but got '
                f'{type(fake_distiller)}'):
            ALGORITHM_CFG_SUPERNET['distiller'] = fake_distiller
            _ = DMCP(**ALGORITHM_CFG_SUPERNET)

        # initiate dmcp without any `mutator`.
        ALGORITHM_CFG_SUPERNET['mutator_cfg'] = None
        with self.assertRaisesRegex(
                AttributeError, "'NoneType' object has no attribute 'get'"):
            _ = DMCP(**ALGORITHM_CFG_SUPERNET)

    def test_loss(self):
        # subernet
        inputs = torch.randn(1, 3, 224, 224)
        dmcp = MODELS.build(ALGORITHM_CFG)
        loss = dmcp(inputs, mode='tensor')
        assert loss.size(1) == 1000

    def test_dmcp_train_step(self):
        # supernet
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = self._prepare_fake_data()
        dmcp = MODELS.build(ALGORITHM_CFG)
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(dmcp.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(dmcp.parameters(), lr=0.01)))

        message_hub = MessageHub.get_current_instance()

        message_hub.update_info('iter', 20)
        dmcp.cur_sample_prob = -1

        losses = dmcp.train_step(inputs, optim_wrapper_dict)

        assert len(losses) == 9
        assert losses['max_subnet1.loss'] > 0
        assert losses['min_subnet1.loss'] > 0
        assert losses['min_subnet1.loss_kl'] + 1e-5 > 0
        assert losses['direct_subnet1.loss'] > 0
        assert losses['direct_subnet1.loss_kl'] + 1e-5 > 0
        assert losses['direct_subnet2.loss'] > 0
        assert losses['direct_subnet2.loss_kl'] + 1e-5 > 0
        assert losses['arch.loss'] > 0
        assert losses['flops.loss'] > 0

        message_hub.update_info('iter', 0)
        dmcp.arch_train = False
        losses = dmcp.train_step(inputs, optim_wrapper_dict)

        assert len(losses) == 4
        assert losses['max_subnet1.loss'] > 0
        assert losses['min_subnet1.loss'] > 0
        assert losses['random_subnet1.loss'] > 0
        assert losses['random_subnet2.loss'] > 0

    def test_dmcp_compute_flops_loss(self):
        dmcp = MODELS.build(ALGORITHM_CFG)
        for type in ['l2', 'inverted_log_l1', 'log_l1', 'l1']:
            dmcp.flops_loss_type = type
            fake_flops = torch.tensor(100)
            dmcp._compute_flops_loss(expected_flops=fake_flops)


class TestDMCPDDP(TestDMCP):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # initialize the process group
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend, rank=0, world_size=1)

    def prepare_model(self, device_ids=None) -> DMCPDDP:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dmcp_algo = MODELS.build(ALGORITHM_CFG).to(self.device)
        self.assertIsInstance(dmcp_algo, DMCP)

        return DMCPDDP(
            module=dmcp_algo,
            find_unused_parameters=True,
            device_ids=device_ids)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='cuda device is not avaliable')
    def test_init(self) -> None:
        ddp_model = self.prepare_model()
        self.assertIsInstance(ddp_model, DMCPDDP)

    def test_dmcpddp_train_step(self) -> None:
        ddp_model = self.prepare_model()
        data = self._prepare_fake_data()
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(ddp_model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(ddp_model.parameters(), lr=0.01)))

        message_hub = MessageHub.get_current_instance()

        message_hub.update_info('iter', 20)
        ddp_model.module.cur_sample_prob = -1
        loss = ddp_model.train_step(data, optim_wrapper_dict)

        message_hub.update_info('iter', 0)
        ddp_model.module.arch_train = False
        loss = ddp_model.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)
