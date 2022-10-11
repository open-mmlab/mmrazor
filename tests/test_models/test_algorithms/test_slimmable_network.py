# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Tuple
from unittest import TestCase
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from mmcls.structures import ClsDataSample
from mmengine.optim import build_optim_wrapper

from mmrazor.models.algorithms import SlimmableNetwork, SlimmableNetworkDDP

MODEL_CFG = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
CHANNEL_CFG_PATH = 'tests/data/MBV2_slimmable_config.json'

MUTATOR_CFG = dict(
    type='SlimmableChannelMutator',
    channel_unit_cfg=dict(type='SlimmableChannelUnit', units=CHANNEL_CFG_PATH),
    parse_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')))

CHANNEL_CFG_PATHS = [
    'tests/data/MBV2_220M.yaml',
    'tests/data/MBV2_320M.yaml',
    'tests/data/MBV2_530M.yaml',
]

OPTIMIZER_CFG = dict(
    type='SGD', lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.0001)
OPTIM_WRAPPER_CFG = dict(optimizer=OPTIMIZER_CFG, accumulative_counts=3)


class FakeMutator:
    ...


class ToyDataPreprocessor(torch.nn.Module):

    def forward(
            self,
            data: Dict,
            training: bool = True) -> Tuple[torch.Tensor, List[ClsDataSample]]:
        return data


class TestSlimmable(TestCase):
    device: str = 'cpu'

    def test_init(self) -> None:

        mutator_wrong_type = FakeMutator()
        with pytest.raises(AttributeError):
            _ = self.prepare_model(mutator_wrong_type, MODEL_CFG)

        # assert has prunable units
        algo = SlimmableNetwork(MUTATOR_CFG, MODEL_CFG)
        self.assertGreater(len(algo.mutator.mutable_units), 0)

        # assert can generate config template
        mutator_cfg = copy.deepcopy(MUTATOR_CFG)
        mutator_cfg['channel_unit_cfg']['units'] = {}
        algo = SlimmableNetwork(mutator_cfg, MODEL_CFG)
        try:
            algo.mutator.config_template()
        except Exception:
            self.fail()

    def test_is_deployed(self) -> None:
        slimmable_should_not_deployed = \
            SlimmableNetwork(MUTATOR_CFG, MODEL_CFG)
        assert not slimmable_should_not_deployed.is_deployed

        slimmable_should_deployed = \
            SlimmableNetwork(MUTATOR_CFG, MODEL_CFG, deploy_index=0)
        assert slimmable_should_deployed.is_deployed

    def test_slimmable_train_step(self) -> None:
        algo = self.prepare_slimmable_model()
        data = self._prepare_fake_data()
        optim_wrapper_cfg = copy.deepcopy(OPTIM_WRAPPER_CFG)
        optim_wrapper_cfg['accumulative_counts'] = 1
        optim_wrapper = build_optim_wrapper(algo, optim_wrapper_cfg)
        fake_message_hub = Mock()
        fake_message_hub.runtime_info = {'iter': 0, 'max_iters': 100}
        optim_wrapper.message_hub = fake_message_hub
        assert not algo._optim_wrapper_count_status_reinitialized
        losses = algo.train_step(data, optim_wrapper)

        assert len(losses) == 3
        assert losses['subnet_0.loss'] > 0
        assert losses['subnet_1.loss'] > 0
        assert losses['subnet_2.loss'] > 0

        self.assertTrue(algo._optim_wrapper_count_status_reinitialized)
        self.assertEqual(optim_wrapper._inner_count, 3)
        self.assertEqual(optim_wrapper._max_counts, 300)

        losses = algo.train_step(data, optim_wrapper)
        assert algo._optim_wrapper_count_status_reinitialized

    def test_fixed_train_step(self) -> None:
        algo = self.prepare_fixed_model()
        data = self._prepare_fake_data()
        optim_wrapper = build_optim_wrapper(algo, OPTIM_WRAPPER_CFG)
        losses = algo.train_step(data, optim_wrapper)

        assert len(losses) == 1
        assert losses['loss'] > 0

    def _prepare_fake_data(self) -> Dict:
        imgs = torch.randn(16, 3, 224, 224).to(self.device)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 1000,
                                                       (16, ))).to(self.device)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def prepare_slimmable_model(self) -> SlimmableNetwork:
        return self.prepare_model(MUTATOR_CFG, MODEL_CFG)

    def prepare_fixed_model(self) -> SlimmableNetwork:

        return self.prepare_model(MUTATOR_CFG, MODEL_CFG, deploy=0)

    def prepare_model(self,
                      mutator_cfg: Dict,
                      model_cfg: Dict,
                      deploy=-1) -> SlimmableNetwork:
        model = SlimmableNetwork(mutator_cfg, model_cfg, deploy,
                                 ToyDataPreprocessor())
        model.to(self.device)

        return model


class TestSlimmableDDP(TestSlimmable):

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
                      mutator_cfg: Dict,
                      model_cfg: Dict,
                      deploy=-1) -> SlimmableNetwork:
        model = super().prepare_model(mutator_cfg, model_cfg, deploy)
        return SlimmableNetworkDDP(module=model, find_unused_parameters=True)

    def test_is_deployed(self) -> None:
        ...

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='cuda device is not avaliable')
    def test_init(self) -> None:
        model = super().prepare_slimmable_model()
        ddp_model = SlimmableNetworkDDP(module=model, device_ids=[0])

        self.assertIsInstance(ddp_model, SlimmableNetworkDDP)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()
