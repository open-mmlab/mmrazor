# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest import TestCase
from unittest.mock import patch

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

from mmrazor.models import DSNAS, NasMutator, OneHotMutableOP
from mmrazor.models.algorithms.nas.dsnas import DSNASDDP
from mmrazor.registry import MODELS
from mmrazor.structures import load_fix_subnet

MODELS.register_module(name='torchConv2d', module=nn.Conv2d, force=True)
MODELS.register_module(name='torchMaxPool2d', module=nn.MaxPool2d, force=True)
MODELS.register_module(name='torchAvgPool2d', module=nn.AvgPool2d, force=True)


@MODELS.register_module()
class ToyDiffModule(BaseModel):

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
        module_kwargs = dict(in_channels=3, out_channels=8, stride=1)

        self.mutable = OneHotMutableOP(
            candidates=self.candidates, module_kwargs=module_kwargs)
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


class TestDsnas(TestCase):

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
        # initiate dsnas when `norm_training` is True.
        model = ToyDiffModule()
        mutator = NasMutator()
        algo = DSNAS(architecture=model, mutator=mutator, norm_training=True)
        algo.eval()
        self.assertTrue(model.bn.training)

        # initiate Dsnas with built mutator
        model = ToyDiffModule()
        mutator = NasMutator()
        algo = DSNAS(model, mutator)
        self.assertIs(algo.mutator, mutator)

        # initiate Dsnas with unbuilt mutator
        mutator = dict(type='NasMutator')
        algo = DSNAS(model, mutator)
        self.assertIsInstance(algo.mutator, NasMutator)

        # test load fix_subnet
        fix_subnet = {'mutable': {'chosen': 'torch_conv2d_5x5'}}
        load_fix_subnet(model, fix_subnet)
        algo = DSNAS(model, mutator)
        self.assertEqual(algo.architecture.mutable.num_choices, 1)

        # initiate Dsnas with error type `mutator`
        with self.assertRaisesRegex(TypeError, 'mutator should be'):
            DSNAS(model, model)

    def test_forward_loss(self) -> None:
        inputs = torch.randn(1, 3, 8, 8)
        model = ToyDiffModule()

        # supernet
        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        algo = DSNAS(model, mutator)
        loss = algo(inputs, mode='loss')
        self.assertIsInstance(loss, dict)

        # subnet
        fix_subnet = {'mutable': {'chosen': 'torch_conv2d_5x5'}}
        load_fix_subnet(model, fix_subnet)
        loss = model(inputs, mode='loss')
        self.assertIsInstance(loss, dict)

    def _prepare_fake_data(self):
        imgs = torch.randn(16, 3, 224, 224).to(self.device)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 1000,
                                                       (16, ))).to(self.device)
        ]
        return {'inputs': imgs, 'data_samples': data_samples}

    def test_search_subnet(self) -> None:
        model = ToyDiffModule()

        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        algo = DSNAS(model, mutator)
        subnet = algo.mutator.sample_choices()
        self.assertIsInstance(subnet, dict)

    @patch('mmengine.logging.message_hub.MessageHub.get_info')
    def test_dsnas_train_step(self, mock_get_info) -> None:
        model = ToyDiffModule()
        mutator = NasMutator()
        mutator.prepare_from_supernet(model)
        mock_get_info.return_value = 2

        algo = DSNAS(model, mutator)
        data = self._prepare_fake_data()
        optim_wrapper = build_optim_wrapper(algo, self.OPTIM_WRAPPER_CFG)
        loss = algo.train_step(data, optim_wrapper)

        self.assertTrue(isinstance(loss['loss'], Tensor))

        algo = DSNAS(model, mutator)
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(model.parameters(), lr=0.01)))
        loss = algo.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)


class TestDsnasDDP(TestDsnas):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # initialize the process group
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend, rank=0, world_size=1)

    def prepare_model(self, device_ids=None) -> DSNAS:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = ToyDiffModule()
        mutator = NasMutator()
        mutator.prepare_from_supernet(model)

        algo = DSNAS(model, mutator).to(self.device)

        return DSNASDDP(
            module=algo, find_unused_parameters=True, device_ids=device_ids)

    @classmethod
    def tearDownClass(cls) -> None:
        dist.destroy_process_group()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='cuda device is not avaliable')
    def test_init(self) -> None:
        ddp_model = self.prepare_model()
        self.assertIsInstance(ddp_model, DSNASDDP)

    @patch('mmengine.logging.message_hub.MessageHub.get_info')
    def test_dsnasddp_train_step(self, mock_get_info) -> None:
        ddp_model = self.prepare_model()
        mock_get_info.return_value = 2

        data = self._prepare_fake_data()
        optim_wrapper = build_optim_wrapper(ddp_model, self.OPTIM_WRAPPER_CFG)
        loss = ddp_model.train_step(data, optim_wrapper)

        self.assertIsNotNone(loss)

        ddp_model = self.prepare_model()
        optim_wrapper_dict = OptimWrapperDict(
            architecture=OptimWrapper(SGD(ddp_model.parameters(), lr=0.1)),
            mutator=OptimWrapper(SGD(ddp_model.parameters(), lr=0.01)))
        loss = ddp_model.train_step(data, optim_wrapper_dict)

        self.assertIsNotNone(loss)
