# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple
from unittest import TestCase
from unittest.mock import MagicMock

import pytest
import torch
from mmcls.structures import ClsDataSample
from mmengine.optim import build_optim_wrapper

from mmrazor.models.algorithms import DCFF

MODEL_CFG = dict(
    cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False)

CHANNEL_CFG_PATH = 'configs/pruning/mmcls/dcff/resnet_cls.json'

MUTATOR_CFG = dict(
    type='DCFFChannelMutator',
    channel_unit_cfg=dict(type='DCFFChannelUnit', units=CHANNEL_CFG_PATH),
    parse_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')))

OPTIMIZER_CFG = dict(
    type='SGD', lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.0001)
OPTIM_WRAPPER_CFG = dict(optimizer=OPTIMIZER_CFG, accumulative_counts=1)


class FakeMutator:
    ...


class ToyDataPreprocessor(torch.nn.Module):

    def forward(
            self,
            data: Dict,
            training: bool = True) -> Tuple[torch.Tensor, List[ClsDataSample]]:
        return data['inputs'], data['data_samples']


class TestDCFF(TestCase):
    device: str = 'cpu'

    def test_init(self) -> None:

        mutator_wrong_type = FakeMutator()
        with pytest.raises(AttributeError):
            _ = self.prepare_model(mutator_wrong_type, MODEL_CFG)

    def test_dcff_train_step(self) -> None:
        algo = self.prepare_dcff_model()
        data = self._prepare_fake_data()
        optim_wrapper_cfg = copy.deepcopy(OPTIM_WRAPPER_CFG)
        optim_wrapper_cfg['accumulative_counts'] = 1
        optim_wrapper = build_optim_wrapper(algo, optim_wrapper_cfg)
        fake_message_hub = MagicMock()
        fake_message_hub.runtime_info = {
            'iter': 0,
            'max_iters': 100,
        }
        optim_wrapper.message_hub = fake_message_hub
        assert not algo._optim_wrapper_count_status_reinitialized
        losses = algo.train_step(data, optim_wrapper)

        assert len(losses) == 1
        assert losses['loss'] > 0

        self.assertTrue(algo._optim_wrapper_count_status_reinitialized)
        self.assertEqual(optim_wrapper._inner_count, 1)

        losses = algo.train_step(data, optim_wrapper)
        assert algo._optim_wrapper_count_status_reinitialized

    def test_fixed_train_step(self) -> None:
        algo = self.prepare_fixed_model()
        data = self._prepare_fake_data()
        optim_wrapper_cfg = copy.deepcopy(OPTIM_WRAPPER_CFG)
        optim_wrapper = build_optim_wrapper(algo, optim_wrapper_cfg)
        fake_message_hub = MagicMock()
        fake_message_hub.runtime_info = {
            'iter': 0,
            'max_iters': 100,
        }
        optim_wrapper.message_hub = fake_message_hub
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

    def prepare_dcff_model(self) -> DCFF:
        return self.prepare_model(MUTATOR_CFG, MODEL_CFG)

    def prepare_fixed_model(self) -> DCFF:
        return self.prepare_model(MUTATOR_CFG, MODEL_CFG, deploy=0)

    def prepare_model(self,
                      mutator_cfg: Dict,
                      model_cfg: Dict,
                      deploy=-1) -> DCFF:
        model = DCFF(mutator_cfg, model_cfg, deploy, ToyDataPreprocessor())
        model.to(self.device)

        return model
