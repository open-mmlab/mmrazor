# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmrazor import digit_version
from mmrazor.implementations.quantization import gptq


class TestGPTQOps(unittest.TestCase):

    @torch.no_grad()
    def test_op(self):
        if digit_version(torch.__version__) < digit_version(
                '1.12.0') or not torch.cuda.is_available():
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
            gptq_linear = gptq.GPTQLinear(
                in_features=12, out_features=20, bias=False).to(device)
            gptq_linear.load_state_dict(linear.state_dict(), strict=False)

            random_data = torch.rand([10, 5, 12]).to(
                device)  # [loader_batch,batch,feature]
            data_0 = random_data[0]

            self.assertTrue(get_loss(linear, gptq_linear, data_0) == 0)

            # quant

            gptq_linear.init_hessian()
            gptq_linear.register_hessian_hook()
            infer(gptq_linear, random_data)
            gptq_linear.remove_hessian_hook()

            qconfig = dict(bits=4, perchannel=True, sym=False)
            quantizer = gptq.Quantizer()
            quantizer.configure(**qconfig)
            gptq_linear.quant(quantizer=quantizer)

            # compare

            print('norm:', linear(data_0).norm(2))
            print('distance:', get_loss(linear, gptq_linear, data_0))

    @torch.no_grad()
    def test_model(self):
        if digit_version(torch.__version__) < digit_version(
                '1.12.0') or not torch.cuda.is_available():
            self.skipTest('torch<1.12.0')
        import torchvision
        model = torchvision.models.resnet18()

        compressor = gptq.GPTQCompressor()
        compressor.prepare(model, use_triton_ops=False)

        x = torch.rand(10, 3, 224, 224)
        compressor.init_hessian()
        compressor.register_hessian_hooks()
        model(x)
        compressor.remove_hessian_hooks()
        compressor.quant_with_default_qconfig()

        model = compressor.to_static_model(model)
        assert type(model.conv1) is nn.Conv2d
