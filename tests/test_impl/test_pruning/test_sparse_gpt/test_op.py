# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmrazor import digit_version
from mmrazor.implementations.pruning import sparse_gpt


class TestSparseGptOps(unittest.TestCase):

    @torch.no_grad()
    def test_op(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('torch<1.12.0')

        def get_loss(linear, linear1, data):
            y = linear(data)
            y1 = linear1(data)
            return (y - y1).square().sum()

        def infer(model, dataset):
            for x in dataset:
                model(x)

        for device in ['cpu']:
            device = torch.device(device)

            # prepare

            linear = nn.Linear(12, 20, bias=False).to(device)
            sparse_linear = sparse_gpt.SparseGptLinear(
                12, 20, bias=False).to(device)
            sparse_linear.load_state_dict(linear.state_dict(), strict=False)

            random_data = torch.rand([10, 5, 12]).to(
                device)  # [loader_batch,batch,feature]
            data_0 = random_data[0]

            self.assertTrue(get_loss(linear, sparse_linear, data_0) == 0)

            # prune

            sparse_linear.init_hessian()
            sparse_linear.register_hessian_hook()
            infer(sparse_linear, random_data)
            sparse_linear.remove_hessian_hook()

            sparse_linear.prune(0.5)

            # compare

            print('norm:', linear(data_0).norm(2))
            print('distance:', get_loss(linear, sparse_linear, data_0))

    @torch.no_grad()
    def test_model(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('torch<1.12.0')
        import torchvision
        model = torchvision.models.resnet18()

        mutator = sparse_gpt.SparseGptCompressor()
        mutator.prepare(model)

        x = torch.rand(10, 3, 224, 224)
        mutator.init_hessian()
        mutator.register_hessian_hooks()
        model(x)
        mutator.remove_hessian_hooks()
        mutator.prune_24()

        model = mutator.to_static_model(model)
        assert type(model.conv1) is nn.Conv2d
