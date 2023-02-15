# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict
from unittest import TestCase

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcls.structures import ClsDataSample
from mmengine.model import BaseModel
from mmengine.optim import build_optim_wrapper
from mmengine.optim.optimizer import OptimWrapper, OptimWrapperDict
from torch import Tensor
from torch.optim import SGD

from mmrazor.models import Darts, DiffMutableOP, NasMutator
from mmrazor.models.algorithms.nas.darts import DartsDDP
from mmrazor.registry import MODELS
from mmrazor.structures import load_fix_subnet

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)
MODELS.register_module(name='torchMaxPool2d', module=nn.MaxPool2d, force=True)
MODELS.register_module(name='torchAvgPool2d', module=nn.AvgPool2d, force=True)


@MODELS.register_module()
class ToyDiffModule2(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)

        self.candidates = dict(
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
        )
        module_kwargs = dict(
            in_channels=3,
            out_channels=8,
            stride=1,
        )
        self.mutable = DiffMutableOP(
            candidates=self.candidates,
            module_kwargs=module_kwargs,
            alias='normal')

        self.bn = nn.BatchNorm2d(8)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            out = self.bn(self.mutable(batch_inputs))
            return dict(loss=out)
        elif mode == 'predict':
            out = self.bn(self.mutable(batch_inputs)) + 1
            return out
        elif mode == 'tensor':
            out = self.bn(self.mutable(batch_inputs)) + 2
            return out


class TestDarts(TestCase):

    def setUp(self) -> None:
        self.device: str = 'cpu'

        OPTIMIZER_CFG = dict(
            type='SGD',
            lr=0.5,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001)

        self.OPTIM_WRAPPER_CFG = dict(optimizer=OPTIMIZER_CFG)

    def test_init(self) -> None:
        # initiate darts when `norm_training` is True.
        model = ToyDiffModule2()
        mutator = NasMutator()
        algo = Darts(architecture=model, mutator=mutator, norm_training=True)
        algo.eval()
        self.assertTrue(model.bn.training)

        # initiate darts with built mutator
        model = ToyDiffModule2()
        mutator = NasMutator()
        algo = Darts(model, mutator)
        self.assertIs(algo.mutator, mutator)

        # initiate darts with unbuilt mutator
        mutator = dict(type='NasMutator')
        algo = Darts(model, mutator)
        self.assertIsInstance(algo.mutator, NasMutator)

        # test load fix_subnet
        fix_subnet = {
            'normal': {
                'chosen': ['torch_conv2d_3x3', 'torch_conv2d_7x7']
            }
        }
        load_fix_subnet(model, fix_subnet)
        algo = Darts(model, mutator)
        self.assertEqual(algo.architecture.mutable.num_choices, 2)

        # initiate darts with error type `mutator`
        with self.assertRaisesRegex(TypeError, 'mutator should be'):
            Darts(model, model)

    def test_forward_loss(self) -> None:
        inputs = torch.randn(1, 3, 8, 8)
        model = ToyDiffModule2()

        # supernet
        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        mutator.prepare_arch_params()

        # subnet
        fix_subnet = fix_subnet = {
            'normal': {
                'chosen': ['torch_conv2d_3x3', 'torch_conv2d_7x7']
            }
        }
        load_fix_subnet(model, fix_subnet)
        loss = model(inputs, mode='loss')
        self.assertIsInstance(loss, dict)

    def _prepare_fake_data(self) -> Dict:
        imgs = torch.randn(16, 3, 224, 224).to(self.device)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 1000,
                                                       (16, ))).to(self.device)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def test_search_subnet(self) -> None:
        model = ToyDiffModule2()

        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        mutator.prepare_arch_params()

        algo = Darts(model, mutator)
        subnet = algo.mutator.sample_choices()
        self.assertIsInstance(subnet, dict)

    def test_darts_train_step(self) -> None:
        model = ToyDiffModule2()

        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        mutator.prepare_arch_params()

        # data is tensor
        algo = Darts(model, mutator)
        data = self._prepare_fake_data()
        optim_wrapper = build_optim_wrapper(algo, self.OPTIM_WRAPPER_CFG)
        loss = algo.train_step(data, optim_wrapper)

        self.assertTrue(isinstance(loss['loss'], Tensor))

        # data is tuple or list
        algo = Darts(model, mutator)
        data = [self._prepare_fake_data() for _ in range(2)]
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(model.parameters(), lr=0.01)))
        loss = algo.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)

    def test_darts_with_unroll(self) -> None:
        model = ToyDiffModule2()

        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        mutator.prepare_arch_params()

        # data is tuple or list
        algo = Darts(model, mutator, unroll=True)
        data = [self._prepare_fake_data() for _ in range(2)]
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(model.parameters(), lr=0.01)))
        loss = algo.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)


class TestDartsDDP(TestDarts):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # initialize the process group
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend, rank=0, world_size=1)

    def prepare_model(self, unroll=False, device_ids=None) -> Darts:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = ToyDiffModule2()

        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        mutator.prepare_arch_params()

        algo = Darts(model, mutator, unroll=unroll).to(self.device)

        return DartsDDP(
            module=algo, find_unused_parameters=True, device_ids=device_ids)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='cuda device is not avaliable')
    def test_init(self) -> None:
        ddp_model = self.prepare_model()
        self.assertIsInstance(ddp_model, DartsDDP)

    def test_dartsddp_train_step(self) -> None:
        # data is tensor
        ddp_model = self.prepare_model()
        data = self._prepare_fake_data()
        optim_wrapper = build_optim_wrapper(ddp_model, self.OPTIM_WRAPPER_CFG)
        loss = ddp_model.train_step(data, optim_wrapper)

        self.assertIsNotNone(loss)

        # data is tuple or list
        ddp_model = self.prepare_model()
        data = [self._prepare_fake_data() for _ in range(2)]
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(ddp_model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(ddp_model.parameters(), lr=0.01)))
        loss = ddp_model.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)

    def test_dartsddp_with_unroll(self) -> None:
        # data is tuple or list
        ddp_model = self.prepare_model(unroll=True)
        data = [self._prepare_fake_data() for _ in range(2)]
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(ddp_model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(ddp_model.parameters(), lr=0.01)))
        loss = ddp_model.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)
