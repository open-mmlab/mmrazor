# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from typing import Dict, List, Tuple, Union
from unittest import TestCase
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from mmcls.structures import ClsDataSample
from mmengine.optim import build_optim_wrapper

from mmrazor import digit_version
from mmrazor.models.algorithms import AutoSlim, AutoSlimDDP

MUTATOR_TYPE = Union[torch.nn.Module, Dict]
DISTILLER_TYPE = Union[torch.nn.Module, Dict]

ARCHITECTURE_CFG = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

MUTATOR_CFG = dict(
    type='OneShotChannelMutator',
    channel_unit_cfg=dict(
        type='OneShotMutableChannelUnit',
        default_args=dict(
            candidate_choices=list(i / 12 for i in range(2, 13)),
            choice_mode='ratio')),
    parse_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')))

DISTILLER_CFG = dict(
    type='ConfigurableDistiller',
    teacher_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    student_recorders=dict(fc=dict(type='ModuleOutputs', source='head.fc')),
    distill_losses=dict(
        loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
    loss_forward_mappings=dict(
        loss_kl=dict(
            preds_S=dict(recorder='fc', from_student=True),
            preds_T=dict(recorder='fc', from_student=False))))

OPTIM_WRAPPER_CFG = dict(
    optimizer=dict(
        type='mmcls.SGD',
        lr=0.5,
        momentum=0.9,
        weight_decay=4e-05,
        _scope_='mmrazor'),
    paramwise_cfg=dict(
        bias_decay_mult=0.0, norm_decay_mult=0.0, dwconv_decay_mult=0.0),
    clip_grad=None,
    accumulative_counts=4)


class FakeMutator:
    ...


class ToyDataPreprocessor(torch.nn.Module):

    def forward(
            self,
            data: Dict,
            training: bool = True) -> Tuple[torch.Tensor, List[ClsDataSample]]:
        return data


@unittest.skipIf(
    digit_version(torch.__version__) == digit_version('1.8.1'),
    'PyTorch version 1.8.1 is not supported by the Backward Tracer.')
class TestAutoSlim(TestCase):
    device: str = 'cpu'

    def test_init(self) -> None:
        mutator_wrong_type = FakeMutator()
        with pytest.raises(Exception):
            _ = self.prepare_model(mutator_wrong_type)

        algo = self.prepare_model()
        self.assertSequenceEqual(
            algo.mutator.mutable_units[0].candidate_choices,
            list(i / 12 for i in range(2, 13)),
        )

    def test_autoslim_train_step(self) -> None:
        algo = self.prepare_model()
        data = self._prepare_fake_data()
        optim_wrapper = build_optim_wrapper(algo, OPTIM_WRAPPER_CFG)
        fake_message_hub = Mock()
        fake_message_hub.runtime_info = {'iter': 0, 'max_iters': 100}
        optim_wrapper.message_hub = fake_message_hub
        assert not algo._optim_wrapper_count_status_reinitialized
        losses = algo.train_step(data, optim_wrapper)

        assert len(losses) == 7
        assert losses['max_subnet.loss'] > 0
        assert losses['min_subnet.loss'] > 0
        assert losses['min_subnet.loss_kl'] + 1e-5 > 0
        assert losses['random_subnet_0.loss'] > 0
        assert losses['random_subnet_0.loss_kl'] + 1e-5 > 0
        assert losses['random_subnet_1.loss'] > 0
        assert losses['random_subnet_1.loss_kl'] + 1e-5 > 0

        assert algo._optim_wrapper_count_status_reinitialized
        assert optim_wrapper._inner_count == 4
        assert optim_wrapper._max_counts == 400

        losses = algo.train_step(data, optim_wrapper)
        assert algo._optim_wrapper_count_status_reinitialized

    def _prepare_fake_data(self) -> Dict:
        imgs = torch.randn(16, 3, 224, 224).to(self.device)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 1000,
                                                       (16, ))).to(self.device)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def prepare_model(self,
                      mutator_cfg: MUTATOR_TYPE = MUTATOR_CFG,
                      distiller_cfg: DISTILLER_TYPE = DISTILLER_CFG,
                      architecture_cfg: Dict = ARCHITECTURE_CFG,
                      num_samples: int = 2) -> AutoSlim:
        model = AutoSlim(
            mutator=mutator_cfg,
            distiller=distiller_cfg,
            architecture=architecture_cfg,
            data_preprocessor=ToyDataPreprocessor(),
            num_samples=num_samples)
        model.to(self.device)

        return model


class TestAutoSlimDDP(TestAutoSlim):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        if torch.cuda.is_available():
            backend = 'nccl'
            cls.device = 'cuda'
        else:
            backend = 'gloo'
        dist.init_process_group(backend, rank=0, world_size=1)

    def prepare_model(self,
                      mutator_cfg: MUTATOR_TYPE = MUTATOR_CFG,
                      distiller_cfg: DISTILLER_TYPE = DISTILLER_CFG,
                      architecture_cfg: Dict = ARCHITECTURE_CFG,
                      num_samples: int = 2) -> AutoSlim:
        model = super().prepare_model(
            mutator_cfg=mutator_cfg,
            distiller_cfg=distiller_cfg,
            architecture_cfg=architecture_cfg,
            num_samples=num_samples)

        return AutoSlimDDP(module=model, find_unused_parameters=True)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='cuda device is not avaliable')
    def test_init(self) -> None:
        model = super().prepare_model()
        ddp_model = AutoSlimDDP(module=model, device_ids=[0])

        self.assertIsInstance(ddp_model, AutoSlimDDP)
